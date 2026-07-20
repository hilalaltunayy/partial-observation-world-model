from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from evaluation.checkpoint_loader import load_world_model_checkpoint, load_world_model_metadata
from evaluation.metrics import compute_precision_recall_f1_iou
from evaluation.world_model_evaluation import evaluate_persistence_baseline, predict_next_observation
from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
from training.world_model_training import TrainingConfig, save_checkpoint


def _create_checkpoint_files(tmp_path: Path) -> tuple[Path, Path]:
    model = EncoderGruDecoderWorldModel(WorldModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_config = TrainingConfig(
        data_dir="data/generated",
        epochs=1,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        device="cpu",
        seed=7,
        checkpoint_dir=str(tmp_path),
        smoke_test=False,
    )
    checkpoint_path, metadata_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation_metrics={"loss": 0.25, "overall_binary_accuracy": 0.75},
        training_config=training_config,
    )
    return checkpoint_path, metadata_path


def test_checkpoint_loader_reconstructs_model_from_metadata(tmp_path: Path) -> None:
    checkpoint_path, metadata_path = _create_checkpoint_files(tmp_path)

    loaded = load_world_model_checkpoint(checkpoint_path, metadata_path, device="cpu")

    assert loaded.model.config == WorldModelConfig()
    assert loaded.device.type == "cpu"
    assert loaded.checkpoint_payload["epoch"] == 1


def test_metadata_validation_rejects_missing_model_config_key(tmp_path: Path) -> None:
    _, metadata_path = _create_checkpoint_files(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["model_config"]["gru_hidden_dim"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="missing model_config keys"):
        load_world_model_metadata(metadata_path)


def test_checkpoint_loader_rejects_mismatched_checkpoint_and_metadata_config(tmp_path: Path) -> None:
    checkpoint_path, metadata_path = _create_checkpoint_files(tmp_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_config"]["gru_hidden_dim"] = 32
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match metadata"):
        load_world_model_checkpoint(checkpoint_path, metadata_path, device="cpu")


def test_checkpoint_loader_rejects_invalid_tensor_shapes(tmp_path: Path) -> None:
    checkpoint_path, metadata_path = _create_checkpoint_files(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["model_state_dict"]["decoder.2.weight"] = torch.zeros((10, 10), dtype=torch.float32)
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="Incompatible checkpoint tensor shapes"):
        load_world_model_checkpoint(checkpoint_path, metadata_path, device="cpu")


def test_predict_next_observation_returns_expected_shapes(tmp_path: Path) -> None:
    checkpoint_path, metadata_path = _create_checkpoint_files(tmp_path)
    loaded = load_world_model_checkpoint(checkpoint_path, metadata_path, device="cpu")
    observation = torch.zeros((1, 1, 4, 7, 7), dtype=torch.float32)
    action = torch.tensor([[0]], dtype=torch.int64)

    logits, predictions, hidden_state = predict_next_observation(
        loaded.model,
        observation=observation,
        action=action,
        device=torch.device("cpu"),
    )

    assert logits.shape == (1, 1, 4, 7, 7)
    assert predictions.shape == (1, 1, 4, 7, 7)
    assert hidden_state.shape == (1, 1, loaded.model.config.gru_hidden_dim)


def test_persistence_baseline_uses_current_observation_as_prediction() -> None:
    batch = {
        "observations": torch.tensor([[[[[1.0]], [[0.0]], [[1.0]], [[0.0]]]]], dtype=torch.float32),
        "actions": torch.tensor([[0]], dtype=torch.int64),
        "next_observations": torch.tensor([[[[[1.0]], [[1.0]], [[1.0]], [[0.0]]]]], dtype=torch.float32),
    }
    dataloader = DataLoader([batch], batch_size=None)

    summary = evaluate_persistence_baseline(dataloader, device=torch.device("cpu"))

    assert summary.overall_binary_accuracy == 0.75
    assert summary.per_channel["static_obstacles"].precision == 1.0
    assert summary.per_channel["scripted_agents"].recall == 0.0


def test_precision_recall_f1_and_iou_calculation() -> None:
    metrics = compute_precision_recall_f1_iou(true_positive=3, false_positive=1, false_negative=2)

    assert metrics.precision == pytest.approx(0.75)
    assert metrics.recall == pytest.approx(0.6)
    assert metrics.f1_score == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert metrics.iou == pytest.approx(0.5)
