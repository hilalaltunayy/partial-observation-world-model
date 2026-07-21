from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

try:
    from data.sequence_dataset import WorldModelSequenceDataset
    from world_model_constants import CHANNEL_NAMES, SCRIPTED_AGENTS_CHANNEL_INDEX
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.training.train_world_model`
    from src.data.sequence_dataset import WorldModelSequenceDataset
    from src.world_model_constants import CHANNEL_NAMES, SCRIPTED_AGENTS_CHANNEL_INDEX


@dataclass(frozen=True)
class ChannelClassBalance:
    positive_count: int
    negative_count: int
    positive_prevalence: float


@dataclass(frozen=True)
class TrainingSplitChannelStatistics:
    split_name: str
    channel_stats: dict[str, ChannelClassBalance]
    total_cells_per_channel: int


@dataclass(frozen=True)
class BalancedSamplingSummary:
    total_sequence_count: int
    scripted_agent_positive_sequence_count: int


def compute_training_split_channel_statistics(data_dir: Path | str) -> TrainingSplitChannelStatistics:
    train_path = Path(data_dir) / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split file for channel statistics: {train_path}")

    with np.load(train_path, allow_pickle=False) as data:
        if "next_observations" not in data.files:
            raise ValueError("Training split is missing 'next_observations', which are required for class statistics.")
        next_observations = data["next_observations"]

    if next_observations.ndim != 4 or next_observations.shape[1] != len(CHANNEL_NAMES):
        raise ValueError(
            f"Expected next_observations with shape [N, {len(CHANNEL_NAMES)}, H, W], got {next_observations.shape}."
        )

    total_cells_per_channel = int(next_observations.shape[0] * next_observations.shape[2] * next_observations.shape[3])
    channel_stats: dict[str, ChannelClassBalance] = {}
    for index, channel_name in enumerate(CHANNEL_NAMES):
        positive_count = int(next_observations[:, index].sum())
        negative_count = total_cells_per_channel - positive_count
        positive_prevalence = 0.0 if total_cells_per_channel == 0 else float(positive_count / total_cells_per_channel)
        channel_stats[channel_name] = ChannelClassBalance(
            positive_count=positive_count,
            negative_count=negative_count,
            positive_prevalence=positive_prevalence,
        )

    return TrainingSplitChannelStatistics(
        split_name="train",
        channel_stats=channel_stats,
        total_cells_per_channel=total_cells_per_channel,
    )


def compute_channel_pos_weights(
    channel_statistics: TrainingSplitChannelStatistics,
    max_pos_weight_cap: float | None = None,
) -> dict[str, float]:
    if max_pos_weight_cap is not None and max_pos_weight_cap <= 0:
        raise ValueError("max_pos_weight_cap must be positive when provided.")

    weights: dict[str, float] = {}
    for channel_name, stats in channel_statistics.channel_stats.items():
        if stats.positive_count <= 0:
            if max_pos_weight_cap is None:
                raise ValueError(f"Cannot compute pos_weight for channel '{channel_name}' with zero positive cells.")
            raw_weight = float("inf")
            clipped_weight = float(max_pos_weight_cap)
        else:
            raw_weight = stats.negative_count / stats.positive_count
            clipped_weight = min(raw_weight, max_pos_weight_cap) if max_pos_weight_cap is not None else raw_weight
        weights[channel_name] = float(clipped_weight)
    return weights


def build_pos_weight_tensor(channel_pos_weights: dict[str, float], device: torch.device) -> torch.Tensor:
    missing_channels = [channel_name for channel_name in CHANNEL_NAMES if channel_name not in channel_pos_weights]
    if missing_channels:
        raise ValueError(f"Missing channel pos_weight entries: {missing_channels}")
    values = [float(channel_pos_weights[channel_name]) for channel_name in CHANNEL_NAMES]
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, 1, len(CHANNEL_NAMES), 1, 1)


def summarize_balanced_sampling(dataset: WorldModelSequenceDataset | Subset[dict[str, torch.Tensor]]) -> BalancedSamplingSummary:
    sequence_infos, next_observations = _resolve_sequence_storage(dataset)
    positive_sequence_count = 0
    for sequence_info in sequence_infos:
        sequence_slice = slice(sequence_info.start_index, sequence_info.end_index)
        has_scripted_agents = bool(np.any(next_observations[sequence_slice, SCRIPTED_AGENTS_CHANNEL_INDEX] == 1))
        if has_scripted_agents:
            positive_sequence_count += 1
    return BalancedSamplingSummary(
        total_sequence_count=len(sequence_infos),
        scripted_agent_positive_sequence_count=positive_sequence_count,
    )


def build_balanced_sequence_sample_weights(
    dataset: WorldModelSequenceDataset | Subset[dict[str, torch.Tensor]],
    agent_sequence_sampling_weight: float,
) -> torch.Tensor:
    if agent_sequence_sampling_weight < 1.0:
        raise ValueError("agent_sequence_sampling_weight must be at least 1.0.")

    sequence_infos, next_observations = _resolve_sequence_storage(dataset)
    weights = torch.ones(len(sequence_infos), dtype=torch.float32)
    if agent_sequence_sampling_weight == 1.0:
        return weights

    for index, sequence_info in enumerate(sequence_infos):
        sequence_slice = slice(sequence_info.start_index, sequence_info.end_index)
        has_scripted_agents = bool(np.any(next_observations[sequence_slice, SCRIPTED_AGENTS_CHANNEL_INDEX] == 1))
        if has_scripted_agents:
            weights[index] = float(agent_sequence_sampling_weight)
    return weights


def _resolve_sequence_storage(
    dataset: WorldModelSequenceDataset | Subset[dict[str, torch.Tensor]],
) -> tuple[list[object], np.ndarray]:
    if isinstance(dataset, WorldModelSequenceDataset):
        return dataset.sequence_infos, dataset.next_observations
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if not isinstance(base_dataset, WorldModelSequenceDataset):
            raise ValueError("Balanced sampling requires a WorldModelSequenceDataset or a Subset of one.")
        sequence_infos = [base_dataset.sequence_infos[int(index)] for index in dataset.indices]
        return sequence_infos, base_dataset.next_observations
    raise ValueError("Balanced sampling requires a WorldModelSequenceDataset or a Subset of one.")
