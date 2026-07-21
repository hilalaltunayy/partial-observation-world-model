from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    from data.sequence_dataset import WorldModelSequenceDataset
    from evaluation.metrics import build_channel_metrics
    from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
    from training.imbalance import (
        build_balanced_sequence_sample_weights,
        build_pos_weight_tensor,
        compute_channel_pos_weights,
        compute_training_split_channel_statistics,
        summarize_balanced_sampling,
    )
    from world_model_constants import CHANNEL_NAMES
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.training.train_world_model`
    from src.data.sequence_dataset import WorldModelSequenceDataset
    from src.evaluation.metrics import build_channel_metrics
    from src.models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
    from src.training.imbalance import (
        build_balanced_sequence_sample_weights,
        build_pos_weight_tensor,
        compute_channel_pos_weights,
        compute_training_split_channel_statistics,
        summarize_balanced_sampling,
    )
    from src.world_model_constants import CHANNEL_NAMES


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
    loss_mode: str = "standard_bce"
    pos_weight_cap: float = 25.0
    use_balanced_sampling: bool = False
    agent_sequence_sampling_weight: float = 4.0
    validation_objective: str = "sparse_channel_mean_f1"
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
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if sampler is not None and shuffle:
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, generator=generator)


def compute_world_model_loss_and_metrics(
    logits: Tensor,
    targets: Tensor,
    *,
    loss_mode: str = "standard_bce",
    pos_weight: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    loss = compute_world_model_loss(logits, targets, loss_mode=loss_mode, pos_weight=pos_weight)
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


def compute_world_model_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    loss_mode: str = "standard_bce",
    pos_weight: Tensor | None = None,
) -> Tensor:
    if loss_mode == "standard_bce":
        return nn.BCEWithLogitsLoss()(logits, targets)
    if loss_mode == "weighted_bce":
        if pos_weight is None:
            raise ValueError("pos_weight must be provided when loss_mode='weighted_bce'.")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def train_one_epoch(
    model: EncoderGruDecoderWorldModel,
    dataloader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_mode: str = "standard_bce",
    pos_weight: Tensor | None = None,
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
        loss, metrics = compute_world_model_loss_and_metrics(
            logits,
            next_observations,
            loss_mode=loss_mode,
            pos_weight=pos_weight,
        )
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
    *,
    loss_mode: str = "standard_bce",
    pos_weight: Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    metric_sums = _empty_metric_sums()
    transition_weight = 0
    true_positive = [0 for _ in CHANNEL_NAMES]
    false_positive = [0 for _ in CHANNEL_NAMES]
    false_negative = [0 for _ in CHANNEL_NAMES]

    with torch.no_grad():
        for batch in dataloader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            next_observations = batch["next_observations"].to(device)

            logits, _ = model(observations, actions)
            _, metrics = compute_world_model_loss_and_metrics(
                logits,
                next_observations,
                loss_mode=loss_mode,
                pos_weight=pos_weight,
            )
            predictions = (torch.sigmoid(logits) >= 0.5).to(next_observations.dtype)
            _update_channel_counts(predictions, next_observations, true_positive, false_positive, false_negative)

            batch_weight = int(observations.shape[0])
            transition_weight += batch_weight
            _accumulate_metrics(metric_sums, metrics, batch_weight)

    finalized_metrics = _finalize_metrics(metric_sums, transition_weight)
    channel_metrics = build_channel_metrics(true_positive, false_positive, false_negative)
    for channel_name, summary in channel_metrics.items():
        finalized_metrics[f"{channel_name}_precision"] = summary.precision
        finalized_metrics[f"{channel_name}_recall"] = summary.recall
        finalized_metrics[f"{channel_name}_f1"] = summary.f1_score
        finalized_metrics[f"{channel_name}_iou"] = summary.iou
    finalized_metrics["sparse_channel_mean_f1"] = float(
        (channel_metrics["static_obstacles"].f1_score + channel_metrics["scripted_agents"].f1_score) / 2.0
    )
    return finalized_metrics


def save_checkpoint(
    checkpoint_dir: Path | str,
    model: EncoderGruDecoderWorldModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict[str, float],
    training_config: TrainingConfig,
    imbalance_settings: dict[str, Any],
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
        "imbalance_settings": imbalance_settings,
    }
    torch.save(checkpoint_payload, checkpoint_path)

    metadata_payload = {
        "epoch": epoch,
        "model_config": model.config.to_dict(),
        "training_config": training_config.to_dict(),
        "validation_metrics": validation_metrics,
        "imbalance_settings": imbalance_settings,
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
    if training_config.loss_mode not in {"standard_bce", "weighted_bce"}:
        raise ValueError("loss_mode must be either 'standard_bce' or 'weighted_bce'.")
    if training_config.pos_weight_cap <= 0:
        raise ValueError("pos_weight_cap must be positive.")
    if training_config.agent_sequence_sampling_weight < 1.0:
        raise ValueError("agent_sequence_sampling_weight must be at least 1.0.")
    if training_config.validation_objective != "sparse_channel_mean_f1":
        raise ValueError("validation_objective must be 'sparse_channel_mean_f1' for this milestone.")

    set_deterministic_seed(training_config.seed)
    device = torch.device(training_config.device)
    datasets = build_sequence_datasets(
        data_dir=training_config.data_dir,
        sequence_length=training_config.sequence_length,
        smoke_test=training_config.smoke_test,
        smoke_train_sequences=training_config.smoke_train_sequences,
        smoke_validation_sequences=training_config.smoke_validation_sequences,
    )
    channel_statistics = compute_training_split_channel_statistics(training_config.data_dir)
    computed_pos_weights = compute_channel_pos_weights(channel_statistics, max_pos_weight_cap=training_config.pos_weight_cap)
    pos_weight_tensor = (
        build_pos_weight_tensor(computed_pos_weights, device)
        if training_config.loss_mode == "weighted_bce"
        else None
    )
    balanced_sampling_summary = summarize_balanced_sampling(datasets["train"])
    train_sampler = None
    if training_config.use_balanced_sampling:
        sample_weights = build_balanced_sequence_sample_weights(
            datasets["train"],
            training_config.agent_sequence_sampling_weight,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(training_config.seed),
        )

    train_loader = build_dataloader(
        datasets["train"],
        training_config.batch_size,
        True,
        training_config.seed,
        sampler=train_sampler,
    )
    validation_loader = build_dataloader(datasets["validation"], training_config.batch_size, False, training_config.seed)

    model = EncoderGruDecoderWorldModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    best_validation_loss = float("inf")
    best_validation_objective = float("-inf")
    best_checkpoint_path: Path | None = None
    best_metadata_path: Path | None = None
    epoch_history: list[dict[str, Any]] = []
    imbalance_settings = {
        "loss_mode": training_config.loss_mode,
        "pos_weight_cap": training_config.pos_weight_cap,
        "computed_pos_weights": computed_pos_weights,
        "channel_positive_prevalence": {
            channel_name: stats.positive_prevalence
            for channel_name, stats in channel_statistics.channel_stats.items()
        },
        "use_balanced_sampling": training_config.use_balanced_sampling,
        "agent_sequence_sampling_weight": training_config.agent_sequence_sampling_weight,
        "scripted_agent_positive_sequence_count": balanced_sampling_summary.scripted_agent_positive_sequence_count,
        "training_sequence_count": balanced_sampling_summary.total_sequence_count,
        "validation_objective": training_config.validation_objective,
    }

    print("Training split channel prevalence:")
    for channel_name, stats in channel_statistics.channel_stats.items():
        print(
            f"  {channel_name}: positive_count={stats.positive_count} "
            f"negative_count={stats.negative_count} prevalence={stats.positive_prevalence:.6f}"
        )
    print("Computed positive weights:")
    for channel_name, weight in computed_pos_weights.items():
        print(f"  {channel_name}: {weight:.6f}")
    print(
        "Training sequence sampling: "
        f"{balanced_sampling_summary.scripted_agent_positive_sequence_count}/"
        f"{balanced_sampling_summary.total_sequence_count} sequences contain scripted agents."
    )

    for epoch_index in range(training_config.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_mode=training_config.loss_mode,
            pos_weight=pos_weight_tensor,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            loss_mode=training_config.loss_mode,
            pos_weight=pos_weight_tensor,
        )
        epoch_number = epoch_index + 1

        print(
            f"Epoch {epoch_number}/{training_config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['overall_binary_accuracy']:.4f} "
            f"obstacle_f1={validation_metrics['static_obstacles_f1']:.4f} "
            f"obstacle_iou={validation_metrics['static_obstacles_iou']:.4f} "
            f"agent_f1={validation_metrics['scripted_agents_f1']:.4f} "
            f"agent_iou={validation_metrics['scripted_agents_iou']:.4f}"
        )

        epoch_record = {
            "epoch": epoch_number,
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        epoch_history.append(epoch_record)

        validation_objective_value = float(validation_metrics[training_config.validation_objective])

        if should_select_best_checkpoint(
            validation_metrics=validation_metrics,
            best_validation_objective=best_validation_objective,
            best_validation_loss=best_validation_loss,
            validation_objective_name=training_config.validation_objective,
        ):
            best_validation_objective = validation_objective_value
            best_validation_loss = validation_metrics["loss"]
            best_checkpoint_path, best_metadata_path = save_checkpoint(
                checkpoint_dir=training_config.checkpoint_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                validation_metrics=validation_metrics,
                training_config=training_config,
                imbalance_settings=imbalance_settings,
            )

    return {
        "model": model,
        "optimizer": optimizer,
        "history": epoch_history,
        "best_checkpoint_path": best_checkpoint_path,
        "best_metadata_path": best_metadata_path,
        "datasets": datasets,
        "imbalance_settings": imbalance_settings,
        "channel_statistics": channel_statistics,
        "computed_pos_weights": computed_pos_weights,
        "balanced_sampling_summary": balanced_sampling_summary,
    }


def should_select_best_checkpoint(
    validation_metrics: dict[str, float],
    best_validation_objective: float,
    best_validation_loss: float,
    validation_objective_name: str,
) -> bool:
    validation_objective_value = float(validation_metrics[validation_objective_name])
    is_better_objective = validation_objective_value > best_validation_objective
    is_tie_with_better_loss = (
        np.isclose(validation_objective_value, best_validation_objective)
        and validation_metrics["loss"] < best_validation_loss
    )
    return bool(is_better_objective or is_tie_with_better_loss)


def _empty_metric_sums() -> dict[str, float]:
    return {"loss": 0.0, "overall_binary_accuracy": 0.0, **{f"{name}_binary_accuracy": 0.0 for name in CHANNEL_NAMES}}


def _accumulate_metrics(metric_sums: dict[str, float], metrics: dict[str, float], weight: int) -> None:
    for name, value in metrics.items():
        metric_sums[name] += value * weight


def _finalize_metrics(metric_sums: dict[str, float], total_weight: int) -> dict[str, float]:
    if total_weight <= 0:
        raise ValueError("No batches were processed.")
    return {name: value / total_weight for name, value in metric_sums.items()}


def _update_channel_counts(
    predictions: Tensor,
    targets: Tensor,
    true_positive: list[int],
    false_positive: list[int],
    false_negative: list[int],
) -> None:
    prediction_mask = predictions.to(torch.bool)
    target_mask = targets.to(torch.bool)
    for index, _ in enumerate(CHANNEL_NAMES):
        channel_predictions = prediction_mask[:, :, index]
        channel_targets = target_mask[:, :, index]
        true_positive[index] += int((channel_predictions & channel_targets).sum().item())
        false_positive[index] += int((channel_predictions & ~channel_targets).sum().item())
        false_negative[index] += int((~channel_predictions & channel_targets).sum().item())
