from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler, WeightedRandomSampler

from data.sequence_dataset import WorldModelSequenceDataset
from training.imbalance import (
    build_balanced_sequence_sample_weights,
    build_pos_weight_tensor,
    compute_channel_pos_weights,
    compute_training_split_channel_statistics,
    summarize_balanced_sampling,
)
from training.world_model_training import (
    TrainingConfig,
    build_dataloader,
    run_training,
    should_select_best_checkpoint,
)


DATA_DIR = Path("data/generated")


def _write_train_npz(path: Path, next_observations: np.ndarray) -> None:
    transition_count = next_observations.shape[0]
    observations = np.zeros_like(next_observations, dtype=np.int8)
    actions = np.zeros((transition_count,), dtype=np.int8)
    episode_ids = np.array([f"train_episode_{index:04d}" for index in range(transition_count)], dtype="<U32")
    time_steps = np.zeros((transition_count,), dtype=np.int16)
    episode_seeds = np.arange(transition_count, dtype=np.int32)
    observer_positions = np.zeros((transition_count, 2), dtype=np.int16)
    scripted_agent_positions = np.zeros((transition_count, 2, 2), dtype=np.int16)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        next_observations=next_observations.astype(np.int8),
        episode_ids=episode_ids,
        time_steps=time_steps,
        episode_seeds=episode_seeds,
        observer_positions=observer_positions,
        scripted_agent_positions=scripted_agent_positions,
    )


def test_correct_positive_and_negative_counts(tmp_path: Path) -> None:
    data_dir = tmp_path / "generated"
    data_dir.mkdir(parents=True)
    next_observations = np.zeros((2, 4, 2, 2), dtype=np.int8)
    next_observations[0, 0, 0, 0] = 1
    next_observations[0, 1, 0, 1] = 1
    next_observations[1, 1, 1, 1] = 1
    next_observations[:, 2, 0, 0] = 1
    next_observations[1, 3, :, :] = 1
    _write_train_npz(data_dir / "train.npz", next_observations)

    stats = compute_training_split_channel_statistics(data_dir)

    assert stats.total_cells_per_channel == 8
    assert stats.channel_stats["static_obstacles"].positive_count == 1
    assert stats.channel_stats["static_obstacles"].negative_count == 7
    assert stats.channel_stats["scripted_agents"].positive_count == 2
    assert stats.channel_stats["observer_location"].positive_count == 2
    assert stats.channel_stats["out_of_bounds"].positive_count == 4


def test_pos_weight_calculation_and_capping(tmp_path: Path) -> None:
    data_dir = tmp_path / "generated"
    data_dir.mkdir(parents=True)
    next_observations = np.zeros((1, 4, 2, 2), dtype=np.int8)
    next_observations[0, 0, 0, 1] = 1
    next_observations[0, 1, 1, 0] = 1
    next_observations[0, 2, 0, 0] = 1
    next_observations[0, 3, 1, 1] = 1
    _write_train_npz(data_dir / "train.npz", next_observations)

    stats = compute_training_split_channel_statistics(data_dir)
    uncapped = compute_channel_pos_weights(stats, max_pos_weight_cap=None)
    capped = compute_channel_pos_weights(stats, max_pos_weight_cap=2.5)
    pos_weight_tensor = build_pos_weight_tensor(capped, torch.device("cpu"))

    assert uncapped["observer_location"] == 3.0
    assert capped["static_obstacles"] == 2.5
    assert pos_weight_tensor.shape == (1, 1, 4, 1, 1)


def test_balanced_sampling_is_deterministic() -> None:
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8, max_sequences=64)
    weights = build_balanced_sequence_sample_weights(dataset, agent_sequence_sampling_weight=5.0)
    sampler_a = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=torch.Generator().manual_seed(42))
    sampler_b = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=torch.Generator().manual_seed(42))

    indices_a = list(iter(sampler_a))
    indices_b = list(iter(sampler_b))

    assert indices_a == indices_b


def test_validation_and_test_dataloaders_are_not_resampled() -> None:
    validation_dataset = WorldModelSequenceDataset(DATA_DIR, "validation", sequence_length=8, max_sequences=16)
    test_dataset = WorldModelSequenceDataset(DATA_DIR, "test", sequence_length=8, max_sequences=16)

    validation_loader = build_dataloader(validation_dataset, batch_size=4, shuffle=False, seed=1)
    test_loader = build_dataloader(test_dataset, batch_size=4, shuffle=False, seed=1)

    assert isinstance(validation_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)


def test_balanced_sampling_summary_reports_scripted_agent_sequences() -> None:
    dataset = WorldModelSequenceDataset(DATA_DIR, "train", sequence_length=8, max_sequences=64)
    summary = summarize_balanced_sampling(dataset)

    assert summary.total_sequence_count == len(dataset)
    assert summary.scripted_agent_positive_sequence_count == dataset.metadata.scripted_agent_positive_sequence_count


def test_checkpoint_metadata_contains_imbalance_settings(tmp_path: Path) -> None:
    training_config = TrainingConfig(
        data_dir=str(DATA_DIR),
        epochs=1,
        batch_size=8,
        sequence_length=8,
        learning_rate=1e-3,
        device="cpu",
        seed=17,
        checkpoint_dir=str(tmp_path),
        smoke_test=True,
        loss_mode="weighted_bce",
        pos_weight_cap=10.0,
        use_balanced_sampling=True,
        agent_sequence_sampling_weight=3.0,
    )

    result = run_training(training_config)
    metadata_path = result["best_metadata_path"]
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    assert metadata["imbalance_settings"]["loss_mode"] == "weighted_bce"
    assert metadata["imbalance_settings"]["pos_weight_cap"] == 10.0
    assert "scripted_agents" in metadata["imbalance_settings"]["computed_pos_weights"]


def test_best_checkpoint_selection_uses_sparse_channel_mean_f1_then_loss() -> None:
    baseline_metrics = {"sparse_channel_mean_f1": 0.40, "loss": 0.25}
    better_f1_metrics = {"sparse_channel_mean_f1": 0.41, "loss": 0.50}
    tie_break_metrics = {"sparse_channel_mean_f1": 0.40, "loss": 0.20}

    assert should_select_best_checkpoint(better_f1_metrics, 0.40, 0.25, "sparse_channel_mean_f1")
    assert should_select_best_checkpoint(tie_break_metrics, 0.40, 0.25, "sparse_channel_mean_f1")
    assert not should_select_best_checkpoint(baseline_metrics, 0.40, 0.25, "sparse_channel_mean_f1")
