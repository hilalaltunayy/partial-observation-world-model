from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from training.world_model_training import TrainingConfig, save_checkpoint
from visualization.prediction import WorldModelPredictorRuntime
from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig


def _build_runtime(tmp_path: Path) -> WorldModelPredictorRuntime:
    model = EncoderGruDecoderWorldModel(WorldModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_config = TrainingConfig(
        data_dir="data/generated",
        epochs=1,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        device="cpu",
        seed=13,
        checkpoint_dir=str(tmp_path),
        smoke_test=False,
    )
    checkpoint_path, metadata_path = save_checkpoint(
        checkpoint_dir=tmp_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        validation_metrics={"loss": 0.5, "overall_binary_accuracy": 0.5},
        training_config=training_config,
    )
    return WorldModelPredictorRuntime.from_paths(checkpoint_path, metadata_path, device="cpu")


def test_prediction_runtime_prepares_one_step_state_without_window(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    current_observation = np.zeros((4, 7, 7), dtype=np.int8)
    current_observation[2, 3, 3] = 1
    next_observation = current_observation.copy()
    next_observation[1, 3, 4] = 1

    prediction_state = runtime.prepare_transition_state(current_observation, action=3, real_next_observation=next_observation)

    assert prediction_state.current_observation.shape == (4, 7, 7)
    assert prediction_state.predicted_next_observation is not None
    assert prediction_state.predicted_next_observation.shape == (4, 7, 7)
    assert prediction_state.real_next_observation is not None
    assert prediction_state.real_next_observation.shape == (4, 7, 7)
    assert prediction_state.absolute_prediction_error is not None
    assert prediction_state.absolute_prediction_error.shape == (7, 7)
    assert prediction_state.prediction_loss is not None
    assert prediction_state.status_message is None


def test_prediction_runtime_handles_missing_checkpoint_without_crashing(tmp_path: Path) -> None:
    runtime = WorldModelPredictorRuntime.from_paths(
        checkpoint_path=tmp_path / "missing.pt",
        metadata_path=tmp_path / "missing.metadata.json",
        device="cpu",
    )
    idle_state = runtime.prepare_idle_state(np.zeros((4, 7, 7), dtype=np.int8))

    assert not runtime.is_available
    assert idle_state.status_message is not None
