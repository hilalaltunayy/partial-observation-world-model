import json
from pathlib import Path

import numpy as np

from data.rollouts import (
    ACTION_LABELS,
    OBSERVATION_CHANNELS,
    DatasetGenerationConfig,
    build_manifest,
    generate_dataset_bundle,
    load_split_dataset,
    save_dataset_bundle,
)


def make_config(tmp_path: Path, episodes: int = 10, steps: int = 4, seed: int = 42) -> DatasetGenerationConfig:
    return DatasetGenerationConfig(
        episodes=episodes,
        steps_per_episode=steps,
        seed=seed,
        output_dir=tmp_path / "generated",
    )


def test_generation_is_deterministic(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=10, steps=3, seed=10)
    bundle_a = generate_dataset_bundle(config)
    bundle_b = generate_dataset_bundle(config)

    for split_name in bundle_a.splits:
        split_a = bundle_a.splits[split_name]
        split_b = bundle_b.splits[split_name]
        for name, array_a in split_a.arrays().items():
            assert np.array_equal(array_a, split_b.arrays()[name]), f"Mismatch in {split_name}::{name}"


def test_observation_and_action_alignment(tmp_path: Path) -> None:
    bundle = generate_dataset_bundle(make_config(tmp_path, episodes=10, steps=5, seed=99))
    split = bundle.splits["train"]

    assert split.observations.shape[1:] == (4, 7, 7)
    assert split.next_observations.shape[1:] == (4, 7, 7)
    assert np.all(split.actions >= 0)
    assert np.all(split.actions < len(ACTION_LABELS))


def test_next_observation_alignment_and_time_steps(tmp_path: Path) -> None:
    bundle = generate_dataset_bundle(make_config(tmp_path, episodes=10, steps=4, seed=50))
    split = bundle.splits["validation"]

    assert np.array_equal(split.time_steps[:4], np.array([0, 1, 2, 3], dtype=np.int16))
    assert np.array_equal(split.observations[1], split.next_observations[0])


def test_expected_array_names_shapes_and_dtypes(tmp_path: Path) -> None:
    bundle = generate_dataset_bundle(make_config(tmp_path, episodes=10, steps=4, seed=8))
    split = bundle.splits["test"]

    assert list(split.arrays().keys()) == [
        "observations",
        "actions",
        "next_observations",
        "episode_ids",
        "time_steps",
        "episode_seeds",
        "observer_positions",
        "scripted_agent_positions",
    ]
    assert split.observations.dtype == np.int8
    assert split.actions.dtype == np.int8
    assert split.next_observations.dtype == np.int8
    assert split.time_steps.dtype == np.int16
    assert split.episode_seeds.dtype == np.int32
    assert split.observer_positions.dtype == np.int16
    assert split.scripted_agent_positions.dtype == np.int16
    assert split.observations.shape[1:] == (len(OBSERVATION_CHANNELS), 7, 7)
    assert split.scripted_agent_positions.shape[1:] == (2, 2)


def test_split_seed_separation(tmp_path: Path) -> None:
    bundle = generate_dataset_bundle(make_config(tmp_path, episodes=20, steps=3, seed=12))
    train_seeds = set(bundle.splits["train"].seeds)
    validation_seeds = set(bundle.splits["validation"].seeds)
    test_seeds = set(bundle.splits["test"].seeds)

    assert not (train_seeds & validation_seeds)
    assert not (train_seeds & test_seeds)
    assert not (validation_seeds & test_seeds)


def test_npz_save_and_load_round_trip(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=10, steps=3, seed=21)
    bundle = generate_dataset_bundle(config)
    save_dataset_bundle(bundle)

    path = config.output_dir / "train.npz"
    reloaded = load_split_dataset(path, "train", bundle.splits["train"].seeds, bundle.splits["train"].episode_count, 3)

    assert np.array_equal(reloaded.observations, bundle.splits["train"].observations)
    assert np.array_equal(reloaded.actions, bundle.splits["train"].actions)


def test_manifest_creation_and_correctness(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=10, steps=3, seed=31)
    bundle = generate_dataset_bundle(config)
    save_dataset_bundle(bundle)

    manifest = json.loads((config.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_schema"]["transition_fields"][0] == "observations"
    assert manifest["observation_channel_meanings"] == list(OBSERVATION_CHANNELS)
    assert manifest["action_encoding"]["mapping"]["0"] == "up"
    assert manifest["splits"]["train"]["arrays"]["observations"]["shape"][1:] == [4, 7, 7]


def test_dataset_card_creation(tmp_path: Path) -> None:
    config = make_config(tmp_path, episodes=10, steps=3, seed=41)
    bundle = generate_dataset_bundle(config)
    save_dataset_bundle(bundle)

    dataset_card = (config.output_dir / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "## Purpose" in dataset_card
    assert "## Regeneration" in dataset_card
    assert "python -m src.data.generate" in dataset_card


def test_invalid_generation_arguments_raise_errors(tmp_path: Path) -> None:
    try:
        DatasetGenerationConfig(episodes=0, steps_per_episode=3, seed=1, output_dir=tmp_path / "generated")
    except ValueError as exc:
        assert "episodes" in str(exc)
    else:
        raise AssertionError("Expected invalid episode count to raise ValueError.")

    try:
        DatasetGenerationConfig(episodes=5, steps_per_episode=0, seed=1, output_dir=tmp_path / "generated")
    except ValueError as exc:
        assert "steps_per_episode" in str(exc)
    else:
        raise AssertionError("Expected invalid step count to raise ValueError.")
