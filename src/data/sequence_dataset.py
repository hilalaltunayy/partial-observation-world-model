from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from data.rollouts import ACTION_LABELS, OBSERVATION_CHANNELS
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.training.train_world_model`
    from src.data.rollouts import ACTION_LABELS, OBSERVATION_CHANNELS


REQUIRED_ARRAY_NAMES: tuple[str, ...] = (
    "observations",
    "actions",
    "next_observations",
    "episode_ids",
    "time_steps",
    "episode_seeds",
    "observer_positions",
    "scripted_agent_positions",
)


@dataclass(frozen=True)
class SequenceIndex:
    episode_id: str
    start_index: int
    end_index: int
    start_time_step: int


@dataclass(frozen=True)
class SequenceDatasetMetadata:
    split_name: str
    sequence_length: int
    transition_count: int
    sequence_count: int
    episode_count: int
    observation_shape: tuple[int, int, int]
    action_count: int
    scripted_agent_positive_sequence_count: int


class WorldModelSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset of contiguous fixed-length transition sequences.

    Each item provides:
    - observations: [T, 4, 7, 7]
    - actions: [T]
    - next_observations: [T, 4, 7, 7]
    """

    def __init__(
        self,
        data_dir: Path | str,
        split_name: str,
        sequence_length: int = 8,
        max_sequences: int | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be a positive integer.")

        self.data_dir = Path(data_dir)
        self.split_name = split_name
        self.sequence_length = sequence_length
        self._arrays = self._load_npz(self.data_dir / f"{split_name}.npz")
        self._validate_raw_arrays()
        self.sequence_infos = self._build_sequence_indices()
        scripted_agent_positive_sequence_count = self._count_sequences_with_scripted_agents()
        if max_sequences is not None:
            if max_sequences <= 0:
                raise ValueError("max_sequences must be positive when provided.")
            self.sequence_infos = self.sequence_infos[:max_sequences]
            scripted_agent_positive_sequence_count = self._count_sequences_with_scripted_agents()
        self.metadata = SequenceDatasetMetadata(
            split_name=split_name,
            sequence_length=sequence_length,
            transition_count=int(self._arrays["actions"].shape[0]),
            sequence_count=len(self.sequence_infos),
            episode_count=int(np.unique(self._arrays["episode_ids"]).shape[0]),
            observation_shape=tuple(int(value) for value in self._arrays["observations"].shape[1:]),
            action_count=len(ACTION_LABELS),
            scripted_agent_positive_sequence_count=scripted_agent_positive_sequence_count,
        )

    @property
    def observations(self) -> np.ndarray:
        return self._arrays["observations"]

    @property
    def actions(self) -> np.ndarray:
        return self._arrays["actions"]

    @property
    def next_observations(self) -> np.ndarray:
        return self._arrays["next_observations"]

    @property
    def episode_ids(self) -> np.ndarray:
        return self._arrays["episode_ids"]

    @property
    def time_steps(self) -> np.ndarray:
        return self._arrays["time_steps"]

    def __len__(self) -> int:
        return len(self.sequence_infos)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence_info = self.sequence_infos[index]
        sequence_slice = slice(sequence_info.start_index, sequence_info.end_index)

        observation_sequence = self.observations[sequence_slice]
        action_sequence = self.actions[sequence_slice]
        next_observation_sequence = self.next_observations[sequence_slice]

        self._validate_sequence_slice(
            episode_ids=self.episode_ids[sequence_slice],
            time_steps=self.time_steps[sequence_slice],
            actions=action_sequence,
            observations=observation_sequence,
            next_observations=next_observation_sequence,
        )

        return {
            "observations": torch.from_numpy(observation_sequence.astype(np.float32, copy=False)),
            "actions": torch.from_numpy(action_sequence.astype(np.int64, copy=False)),
            "next_observations": torch.from_numpy(next_observation_sequence.astype(np.float32, copy=False)),
        }

    def _load_npz(self, path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset split file: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        missing = [name for name in REQUIRED_ARRAY_NAMES if name not in arrays]
        if missing:
            raise ValueError(f"Missing required arrays in {path.name}: {missing}")
        return arrays

    def _validate_raw_arrays(self) -> None:
        observations = self._arrays["observations"]
        actions = self._arrays["actions"]
        next_observations = self._arrays["next_observations"]
        episode_ids = self._arrays["episode_ids"]
        time_steps = self._arrays["time_steps"]

        transition_count = actions.shape[0]
        if observations.shape != next_observations.shape:
            raise ValueError("observations and next_observations must have identical shapes.")
        if observations.shape[0] != transition_count:
            raise ValueError("observations and actions must have the same number of transitions.")
        if observations.shape[1:] != (len(OBSERVATION_CHANNELS), 7, 7):
            raise ValueError(f"Unexpected observation shape: {observations.shape[1:]}")
        if episode_ids.shape[0] != transition_count or time_steps.shape[0] != transition_count:
            raise ValueError("episode_ids and time_steps must align with transitions.")
        if np.any(actions < 0) or np.any(actions >= len(ACTION_LABELS)):
            raise ValueError("actions contain out-of-range class ids.")

        last_episode_id = None
        last_time_step = None
        seen_closed_episodes: set[str] = set()
        for episode_id_raw, time_step_raw in zip(episode_ids, time_steps):
            episode_id = str(episode_id_raw)
            time_step = int(time_step_raw)
            if episode_id != last_episode_id:
                if episode_id in seen_closed_episodes:
                    raise ValueError(f"Episode {episode_id} is not stored contiguously.")
                if time_step != 0:
                    raise ValueError(f"Episode {episode_id} must begin at time step 0.")
                if last_episode_id is not None:
                    seen_closed_episodes.add(str(last_episode_id))
                last_episode_id = episode_id
                last_time_step = 0
                continue
            expected_time_step = int(last_time_step) + 1
            if time_step != expected_time_step:
                raise ValueError(
                    f"Non-contiguous time step ordering for episode {episode_id}: {time_step} != {expected_time_step}"
                )
            last_time_step = time_step

    def _build_sequence_indices(self) -> list[SequenceIndex]:
        episode_ids = self.episode_ids
        time_steps = self.time_steps
        sequence_infos: list[SequenceIndex] = []
        episode_start = 0

        for index in range(1, episode_ids.shape[0] + 1):
            is_episode_end = index == episode_ids.shape[0] or episode_ids[index] != episode_ids[episode_start]
            if not is_episode_end:
                continue

            episode_length = index - episode_start
            if episode_length >= self.sequence_length:
                for start_index in range(episode_start, index - self.sequence_length + 1):
                    sequence_infos.append(
                        SequenceIndex(
                            episode_id=str(episode_ids[episode_start]),
                            start_index=start_index,
                            end_index=start_index + self.sequence_length,
                            start_time_step=int(time_steps[start_index]),
                        )
                    )
            episode_start = index

        if not sequence_infos:
            raise ValueError("No valid sequences could be constructed for the requested sequence length.")
        return sequence_infos

    def _validate_sequence_slice(
        self,
        episode_ids: np.ndarray,
        time_steps: np.ndarray,
        actions: np.ndarray,
        observations: np.ndarray,
        next_observations: np.ndarray,
    ) -> None:
        if len(np.unique(episode_ids)) != 1:
            raise ValueError("A sequence crossed an episode boundary.")
        expected_time_steps = np.arange(time_steps[0], time_steps[0] + self.sequence_length, dtype=np.int16)
        if not np.array_equal(time_steps, expected_time_steps):
            raise ValueError("Time steps inside a sequence must be contiguous.")
        if observations.shape != (self.sequence_length, len(OBSERVATION_CHANNELS), 7, 7):
            raise ValueError("Observation sequence shape mismatch.")
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations shape mismatch within sequence.")
        if actions.shape != (self.sequence_length,):
            raise ValueError("Action sequence shape mismatch.")
        if np.any(actions < 0) or np.any(actions >= len(ACTION_LABELS)):
            raise ValueError("Action sequence contains invalid values.")

    def _count_sequences_with_scripted_agents(self) -> int:
        count = 0
        for sequence_info in self.sequence_infos:
            sequence_slice = slice(sequence_info.start_index, sequence_info.end_index)
            if np.any(self.next_observations[sequence_slice, 1] == 1):
                count += 1
        return count
