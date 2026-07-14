from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from data.sequence_dataset import WorldModelSequenceDataset
    from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.training.train_world_model`
    from src.data.sequence_dataset import WorldModelSequenceDataset
    from src.models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig


CHANNEL_NAMES: tuple[str, ...] = (
    "static_obstacles",
    "scripted_agents",
    "observer_location",
    "out_of_bounds",
)


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: str
    epochs: int
    batch_size: int
    sequence_length: int
    learning_rate: float
    device: str
    seed: int
    checkpoint_dir: str
    smoke_test: bool
    smoke_train_sequences: int = 64
    smoke_validation_sequences: int = 32

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequence_datasets(
    data_dir: Path | str,
    sequence_length: int = 8,
    smoke_test: bool = False,
    smoke_train_sequences: int = 64,
    smoke_validation_sequences: int = 32,
) -> dict[str, WorldModelSequenceDataset]:
    train_dataset = WorldModelSequenceDataset(data_dir=data_dir, split_name="train", sequence_length=sequence_length)
    validation_dataset = WorldModelSequenceDataset(
        data_dir=data_dir,
        split_name="validation",
        sequence_length=sequence_length,
    )
    test_dataset = WorldModelSequenceDataset(data_dir=data_dir, split_name="test", sequence_length=sequence_length)

    if smoke_test:
        return {
            "train": _dataset_subset(train_dataset, smoke_train_sequences),
            "validation": _dataset_subset(validation_dataset, smoke_validation_sequences),
            "test": _dataset_subset(test_dataset, smoke_validation_sequences),
        }
    return {"train": train_dataset, "validation": validation_dataset, "test": test_dataset}


def _dataset_subset(dataset: WorldModelSequenceDataset, limit: int) -> Subset[dict[str, Tensor]]:
    subset_length = min(limit, len(dataset))
    return Subset(dataset, list(range(subset_length)))


def build_dataloader(
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_world_model_loss_and_metrics(logits: Tensor, targets: Tensor) -> tuple[Tensor, dict[str, float]]:
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    predictions = (torch.sigmoid(logits) >= 0.5).to(targets.dtype)
    channel_accuracy = (predictions == targets).float().mean(dim=(0, 1, 3, 4))
    overall_accuracy = float((predictions == targets).float().mean().item())

    metrics = {
        "loss": float(loss.item()),
        "overall_binary_accuracy": overall_accuracy,
    }
    for index, channel_name in enumerate(CHANNEL_NAMES):
        metrics[f"{channel_name}_binary_accuracy"] = float(channel_accuracy[index].item())
    return loss, metrics


def train_one_epoch(
    model: EncoderGruDecoderWorldModel,
    dataloader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    metric_sums = _empty_metric_sums()
    transition_weight = 0

    for batch in dataloader:
        observations = batch["observations"].to(device)
        actions = batch["actions"].to(device)
        next_observations = batch["next_observations"].to(device)

        optimizer.zero_grad()
        logits, _ = model(observations, actions)
        loss, metrics = compute_world_model_loss_and_metrics(logits, next_observations)
        loss.backward()
        optimizer.step()

        batch_weight = int(observations.shape[0])
        transition_weight += batch_weight
        _accumulate_metrics(metric_sums, metrics, batch_weight)

    return _finalize_metrics(metric_sums, transition_weight)


def evaluate(
    model: EncoderGruDecoderWorldModel,
    dataloader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metric_sums = _empty_metric_sums()
    transition_weight = 0

    with torch.no_grad():
        for batch in dataloader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            next_observations = batch["next_observations"].to(device)

            logits, _ = model(observations, actions)
            _, metrics = compute_world_model_loss_and_metrics(logits, next_observations)

            batch_weight = int(observations.shape[0])
            transition_weight += batch_weight
            _accumulate_metrics(metric_sums, metrics, batch_weight)

    return _finalize_metrics(metric_sums, transition_weight)


def save_checkpoint(
    checkpoint_dir: Path | str,
    model: EncoderGruDecoderWorldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    training_config: TrainingConfig,
) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint_dir) / "world_model_best.pt"
    metadata_path = Path(checkpoint_dir) / "world_model_best.metadata.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "validation_metrics": validation_metrics,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    metadata_payload = {
        "epoch": epoch,
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "validation_metrics": validation_metrics,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return checkpoint_path, metadata_path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: EncoderGruDecoderWorldModel,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def run_training(
    training_config: TrainingConfig,
    model_config: WorldModelConfig | None = None,
) -> dict[str, Any]:
    if training_config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if training_config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if training_config.sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    set_deterministic_seed(training_config.seed)
    device = torch.device(training_config.device)
    datasets = build_sequence_datasets(
        data_dir=training_config.data_dir,
        sequence_length=training_config.sequence_length,
        smoke_test=training_config.smoke_test,
        smoke_train_sequences=training_config.smoke_train_sequences,
        smoke_validation_sequences=training_config.smoke_validation_sequences,
    )
    train_loader = build_dataloader(datasets["train"], training_config.batch_size, True, training_config.seed)
    validation_loader = build_dataloader(datasets["validation"], training_config.batch_size, False, training_config.seed)

    model = EncoderGruDecoderWorldModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    best_validation_loss = float("inf")
    best_checkpoint_path: Path | None = None
    best_metadata_path: Path | None = None
    epoch_history: list[dict[str, Any]] = []

    for epoch_index in range(training_config.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(model, validation_loader, device)
        epoch_number = epoch_index + 1

        print(
            f"Epoch {epoch_number}/{training_config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['overall_binary_accuracy']:.4f}"
        )

        epoch_record = {
            "epoch": epoch_number,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        epoch_history.append(epoch_record)

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_checkpoint_path, best_metadata_path = save_checkpoint(
                checkpoint_dir=training_config.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                validation_metrics=validation_metrics,
                training_config=training_config,
            )

    return {
        "model": model,
        "optimizer": optimizer,
        "history": epoch_history,
        "best_checkpoint_path": best_checkpoint_path,
        "best_metadata_path": best_metadata_path,
        "datasets": datasets,
    }


def _empty_metric_sums() -> dict[str, float]:
    return {"loss": 0.0, "overall_binary_accuracy": 0.0, **{f"{name}_binary_accuracy": 0.0 for name in CHANNEL_NAMES}}


def _accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float], weight: int) -> None:
    for name, value in metrics.items():
        metric_sums[name] += value * weight


def _finalize_metrics(metric_sums: dict[str, float], total_weight: int) -> dict[str, float]:
    if total_weight <= 0:
        raise ValueError("No batches were processed.")
    return {name: value / total_weight for name, value in metric_sums.items()}
