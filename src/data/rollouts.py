from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

try:
    from environment import Action, GridWorldConfig, GridWorldEnvironment
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.data.generate`
    from src.environment import Action, GridWorldConfig, GridWorldEnvironment


OBSERVATION_CHANNELS: tuple[str, ...] = (
    "static_obstacles",
    "scripted_agents",
    "observer_location",
    "out_of_bounds",
)
ACTION_LABELS: tuple[str, ...] = ("up", "down", "left", "right")
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetGenerationConfig:
    episodes: int
    steps_per_episode: int
    seed: int
    output_dir: Path = Path("data/generated")
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    action_policy: str = "seeded_random"
    environment_config: GridWorldConfig = GridWorldConfig()

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be a positive integer.")
        if self.steps_per_episode <= 0:
            raise ValueError("steps_per_episode must be a positive integer.")
        if self.train_ratio <= 0 or self.validation_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("split ratios must be positive.")
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError("split ratios must sum to 1.0.")
        if self.output_dir.name != "generated" and self.output_dir == Path():
            raise ValueError("output_dir must be a valid path.")


@dataclass
class SplitDataset:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    episode_ids: np.ndarray
    time_steps: np.ndarray
    episode_seeds: np.ndarray
    observer_positions: np.ndarray
    scripted_agent_positions: np.ndarray
    split_name: str
    seeds: list[int]
    episode_count: int
    steps_per_episode: int

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "next_observations": self.next_observations,
            "episode_ids": self.episode_ids,
            "time_steps": self.time_steps,
            "episode_seeds": self.episode_seeds,
            "observer_positions": self.observer_positions,
            "scripted_agent_positions": self.scripted_agent_positions,
        }

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class DatasetBundle:
    config: DatasetGenerationConfig
    splits: dict[str, SplitDataset]
    manifest: dict[str, Any]
    dataset_card: str


def generate_dataset_bundle(config: DatasetGenerationConfig) -> DatasetBundle:
    split_episode_counts = _compute_split_counts(config.episodes, config.train_ratio, config.validation_ratio)
    split_seeds = _allocate_split_seeds(config.seed, split_episode_counts)

    splits: dict[str, SplitDataset] = {}
    for split_name in SPLIT_NAMES:
        splits[split_name] = _generate_split_dataset(
            split_name=split_name,
            episode_seeds=split_seeds[split_name],
            steps_per_episode=config.steps_per_episode,
            environment_config=config.environment_config,
            action_policy=config.action_policy,
        )

    manifest = build_manifest(config, splits)
    dataset_card = build_dataset_card(config, splits, manifest)
    bundle = DatasetBundle(config=config, splits=splits, manifest=manifest, dataset_card=dataset_card)
    validate_dataset_bundle(bundle)
    return bundle


def save_dataset_bundle(bundle: DatasetBundle) -> list[Path]:
    output_dir = bundle.config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for split_name, split in bundle.splits.items():
        path = output_dir / f"{split_name}.npz"
        np.savez_compressed(path, **split.arrays())
        saved_paths.append(path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(bundle.manifest, indent=2), encoding="utf-8")
    saved_paths.append(manifest_path)

    dataset_card_path = output_dir / "DATASET_CARD.md"
    dataset_card_path.write_text(bundle.dataset_card, encoding="utf-8")
    saved_paths.append(dataset_card_path)

    _verify_saved_bundle(bundle)
    return saved_paths


def load_split_dataset(path: Path, split_name: str, seeds: list[int], episode_count: int, steps_per_episode: int) -> SplitDataset:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    return SplitDataset(
        observations=arrays["observations"],
        actions=arrays["actions"],
        next_observations=arrays["next_observations"],
        episode_ids=arrays["episode_ids"],
        time_steps=arrays["time_steps"],
        episode_seeds=arrays["episode_seeds"],
        observer_positions=arrays["observer_positions"],
        scripted_agent_positions=arrays["scripted_agent_positions"],
        split_name=split_name,
        seeds=list(seeds),
        episode_count=episode_count,
        steps_per_episode=steps_per_episode,
    )


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    split_seed_sets = {name: set(split.seeds) for name, split in bundle.splits.items()}
    if split_seed_sets["train"] & split_seed_sets["validation"]:
        raise ValueError("train and validation split seeds overlap.")
    if split_seed_sets["train"] & split_seed_sets["test"]:
        raise ValueError("train and test split seeds overlap.")
    if split_seed_sets["validation"] & split_seed_sets["test"]:
        raise ValueError("validation and test split seeds overlap.")

    for split_name, split in bundle.splits.items():
        arrays = split.arrays()
        transition_count = split.episode_count * split.steps_per_episode
        expected_obs_shape = (
            transition_count,
            len(OBSERVATION_CHANNELS),
            bundle.config.environment_config.local_view_size,
            bundle.config.environment_config.local_view_size,
        )
        if split.observations.shape != expected_obs_shape:
            raise ValueError(f"{split_name} observations shape mismatch: {split.observations.shape} != {expected_obs_shape}")
        if split.next_observations.shape != expected_obs_shape:
            raise ValueError(f"{split_name} next_observations shape mismatch.")

        for name, array in arrays.items():
            if array.shape[0] != transition_count:
                raise ValueError(f"{split_name} {name} has inconsistent transition length.")

        if split.actions.dtype != np.int8:
            raise ValueError(f"{split_name} actions must use int8 dtype.")
        if np.any(split.actions < 0) or np.any(split.actions >= len(ACTION_LABELS)):
            raise ValueError(f"{split_name} contains invalid action values.")
        if split.observations.dtype != np.int8 or split.next_observations.dtype != np.int8:
            raise ValueError(f"{split_name} observations must use int8 dtype.")
        if split.observer_positions.dtype != np.int16:
            raise ValueError(f"{split_name} observer_positions must use int16 dtype.")
        if split.scripted_agent_positions.dtype != np.int16:
            raise ValueError(f"{split_name} scripted_agent_positions must use int16 dtype.")
        if split.time_steps.dtype != np.int16:
            raise ValueError(f"{split_name} time_steps must use int16 dtype.")
        if split.episode_seeds.dtype != np.int32:
            raise ValueError(f"{split_name} episode_seeds must use int32 dtype.")
        if split.episode_ids.dtype.kind not in {"U", "S"}:
            raise ValueError(f"{split_name} episode_ids must be stored as strings.")
        if not np.array_equal(split.time_steps[: split.steps_per_episode], np.arange(split.steps_per_episode, dtype=np.int16)):
            raise ValueError(f"{split_name} time steps are not aligned with episode progression.")
        _validate_observation_alignment(split, split_name)


def build_manifest(config: DatasetGenerationConfig, splits: dict[str, SplitDataset]) -> dict[str, Any]:
    manifest_splits: dict[str, Any] = {}
    for split_name, split in splits.items():
        manifest_splits[split_name] = {
            "episode_count": split.episode_count,
            "transition_count": split.transition_count,
            "steps_per_episode": split.steps_per_episode,
            "seeds": split.seeds,
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in split.arrays().items()
            },
        }

    return {
        "dataset_name": "partial-observation-gridworld-phase1",
        "dataset_schema": {
            "transition_fields": [
                "observations",
                "actions",
                "next_observations",
                "episode_ids",
                "time_steps",
                "episode_seeds",
                "observer_positions",
                "scripted_agent_positions",
            ]
        },
        "array_names": list(next(iter(splits.values())).arrays().keys()),
        "generation_parameters": {
            "episodes": config.episodes,
            "steps_per_episode": config.steps_per_episode,
            "seed": config.seed,
            "action_policy": config.action_policy,
            "environment_config": _serialize_environment_config(config.environment_config),
        },
        "observation_channel_meanings": list(OBSERVATION_CHANNELS),
        "action_encoding": {
            "type": "integer_class_id",
            "mapping": {str(index): label for index, label in enumerate(ACTION_LABELS)},
        },
        "split_sizes": {
            split_name: split.transition_count
            for split_name, split in splits.items()
        },
        "episode_counts": {
            split_name: split.episode_count
            for split_name, split in splits.items()
        },
        "steps_per_episode": config.steps_per_episode,
        "seeds_by_split": {
            split_name: split.seeds
            for split_name, split in splits.items()
        },
        "splits": manifest_splits,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_dataset_card(config: DatasetGenerationConfig, splits: dict[str, SplitDataset], manifest: dict[str, Any]) -> str:
    lines = [
        "# Dataset Card",
        "",
        "## Purpose",
        "",
        "This dataset provides deterministic grid-world rollout transitions for Phase 1 world-model training.",
        "Each record captures a local observation, the observer action, and the next local observation.",
        "",
        "## Rollout Generation",
        "",
        "- Episodes are generated from the existing deterministic `20x20` `GridWorldEnvironment`.",
        "- Obstacles and scripted-agent behavior are determined by the per-episode seed.",
        "- Observer actions are sampled from a seed-controlled random policy using the same 4-action space as the environment.",
        f"- The dataset was generated with `{config.episodes}` total episodes and `{config.steps_per_episode}` steps per episode.",
        "",
        "## Environment and Observation Format",
        "",
        "- Environment: deterministic `20x20` grid world with static obstacles, one observer, and two scripted moving agents.",
        f"- Observation shape: `[4, {config.environment_config.local_view_size}, {config.environment_config.local_view_size}]`.",
        "- Observations are centered on the observer and preserve the existing local-view encoding.",
        "",
        "## Observation Channels",
        "",
        "- `0`: static obstacles",
        "- `1`: scripted agents",
        "- `2`: observer location",
        "- `3`: out-of-bounds / unknown cells",
        "",
        "## Action Encoding",
        "",
        "- Integer class IDs:",
        "- `0`: up",
        "- `1`: down",
        "- `2`: left",
        "- `3`: right",
        "",
        "## Split Strategy",
        "",
        "- Train, validation, and test use disjoint deterministic episode seeds.",
        "- Splits are created from the total episode count using a `70/15/15` episode allocation with remainder assigned deterministically.",
        "",
        "## Dataset Sizes and Shapes",
        "",
    ]

    for split_name, split in splits.items():
        lines.extend(
            [
                f"### {split_name.title()}",
                "",
                f"- Episodes: `{split.episode_count}`",
                f"- Transitions: `{split.transition_count}`",
                f"- Observations: `{list(split.observations.shape)}` `{split.observations.dtype}`",
                f"- Actions: `{list(split.actions.shape)}` `{split.actions.dtype}`",
                f"- Next observations: `{list(split.next_observations.shape)}` `{split.next_observations.dtype}`",
                f"- Observer positions: `{list(split.observer_positions.shape)}` `{split.observer_positions.dtype}`",
                f"- Scripted-agent positions: `{list(split.scripted_agent_positions.shape)}` `{split.scripted_agent_positions.dtype}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Known Limitations",
            "",
            "- Observer actions are not policy-optimized; they come from a simple seed-controlled random sampler.",
            "- The dataset covers only local observations and does not include rewards, goals, or full global-state rollouts.",
            "- The environment action space currently excludes a `stay` action.",
            "",
            "## Regeneration",
            "",
            f"Run `python -m src.data.generate --episodes {config.episodes} --steps {config.steps_per_episode} --seed {config.seed}`.",
            "",
            "## Future World-Model Use",
            "",
            "The world model will later use these transitions to learn next-local-observation prediction from observation-action pairs.",
            "",
            "## Manifest Reference",
            "",
            "Machine-readable dataset metadata is stored in `manifest.json` alongside the split files.",
        ]
    )
    return "\n".join(lines) + "\n"


def _generate_split_dataset(
    split_name: str,
    episode_seeds: list[int],
    steps_per_episode: int,
    environment_config: GridWorldConfig,
    action_policy: str,
) -> SplitDataset:
    if action_policy != "seeded_random":
        raise ValueError(f"Unsupported action policy: {action_policy}")

    observations: list[np.ndarray] = []
    actions: list[int] = []
    next_observations: list[np.ndarray] = []
    episode_ids: list[str] = []
    time_steps: list[int] = []
    episode_seeds_array: list[int] = []
    observer_positions: list[tuple[int, int]] = []
    scripted_agent_positions: list[list[tuple[int, int]]] = []

    env = GridWorldEnvironment(environment_config)
    for episode_index, episode_seed in enumerate(episode_seeds):
        current_observation, _ = env.reset(seed=episode_seed)
        episode_id = f"{split_name}_episode_{episode_index:04d}"
        action_rng = random.Random((episode_seed * 9973) + 17)

        for time_step in range(steps_per_episode):
            action_value = action_rng.randrange(len(ACTION_LABELS))
            next_observation, _, _, _ = env.step(action_value)
            state = env.get_full_state()

            observations.append(current_observation.astype(np.int8, copy=True))
            actions.append(action_value)
            next_observations.append(next_observation.astype(np.int8, copy=True))
            episode_ids.append(episode_id)
            time_steps.append(time_step)
            episode_seeds_array.append(episode_seed)
            observer_positions.append(state.observer_position)
            scripted_agent_positions.append([agent.position for agent in state.scripted_agents])

            current_observation = next_observation

    return SplitDataset(
        observations=np.stack(observations).astype(np.int8, copy=False),
        actions=np.asarray(actions, dtype=np.int8),
        next_observations=np.stack(next_observations).astype(np.int8, copy=False),
        episode_ids=np.asarray(episode_ids, dtype="<U32"),
        time_steps=np.asarray(time_steps, dtype=np.int16),
        episode_seeds=np.asarray(episode_seeds_array, dtype=np.int32),
        observer_positions=np.asarray(observer_positions, dtype=np.int16),
        scripted_agent_positions=np.asarray(scripted_agent_positions, dtype=np.int16),
        split_name=split_name,
        seeds=list(episode_seeds),
        episode_count=len(episode_seeds),
        steps_per_episode=steps_per_episode,
    )


def _compute_split_counts(total_episodes: int, train_ratio: float, validation_ratio: float) -> dict[str, int]:
    train_count = int(total_episodes * train_ratio)
    validation_count = int(total_episodes * validation_ratio)
    test_count = total_episodes - train_count - validation_count
    if train_count <= 0 or validation_count <= 0 or test_count <= 0:
        raise ValueError("Episode count must be large enough to allocate all three splits.")
    return {"train": train_count, "validation": validation_count, "test": test_count}


def _allocate_split_seeds(base_seed: int, split_episode_counts: dict[str, int]) -> dict[str, list[int]]:
    split_seeds: dict[str, list[int]] = {}
    next_seed = base_seed
    for split_name in SPLIT_NAMES:
        count = split_episode_counts[split_name]
        split_seeds[split_name] = list(range(next_seed, next_seed + count))
        next_seed += count
    return split_seeds


def _serialize_environment_config(config: GridWorldConfig) -> dict[str, Any]:
    serialized = asdict(config)
    serialized["scripted_agent_specs"] = [
        {
            "start_position": list(spec.start_position),
            "action_cycle": [action.name.lower() for action in spec.action_cycle],
        }
        for spec in config.scripted_agent_specs
    ]
    serialized["observer_start_position"] = list(config.observer_start_position)
    return serialized


def _verify_saved_bundle(bundle: DatasetBundle) -> None:
    for split_name, split in bundle.splits.items():
        path = bundle.config.output_dir / f"{split_name}.npz"
        reloaded = load_split_dataset(
            path=path,
            split_name=split_name,
            seeds=split.seeds,
            episode_count=split.episode_count,
            steps_per_episode=split.steps_per_episode,
        )
        for name, original in split.arrays().items():
            reloaded_array = reloaded.arrays()[name]
            if not np.array_equal(original, reloaded_array):
                raise ValueError(f"Reload validation failed for {split_name}::{name}.")

    manifest_path = bundle.config.output_dir / "manifest.json"
    dataset_card_path = bundle.config.output_dir / "DATASET_CARD.md"
    if not manifest_path.exists() or not dataset_card_path.exists():
        raise ValueError("Manifest or dataset card was not saved successfully.")


def _validate_observation_alignment(split: SplitDataset, split_name: str) -> None:
    if split.observations.shape != split.next_observations.shape:
        raise ValueError(f"{split_name} observation tensors must match next-observation shape.")
    if split.scripted_agent_positions.shape[1:] != (2, 2):
        raise ValueError(f"{split_name} scripted_agent_positions must have shape [N, 2, 2].")
    if split.observer_positions.shape[1:] != (2,):
        raise ValueError(f"{split_name} observer_positions must have shape [N, 2].")

