from __future__ import annotations

import argparse

from .rollouts import DatasetGenerationConfig, generate_dataset_bundle, save_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic rollout datasets for the grid-world environment.")
    parser.add_argument("--episodes", type=int, required=True, help="Total number of episodes across all splits.")
    parser.add_argument("--steps", type=int, required=True, help="Number of transitions per episode.")
    parser.add_argument("--seed", type=int, required=True, help="Base seed used to allocate deterministic episode seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetGenerationConfig(
        episodes=args.episodes,
        steps_per_episode=args.steps,
        seed=args.seed,
    )
    bundle = generate_dataset_bundle(config)
    saved_paths = save_dataset_bundle(bundle)
    _print_summary(bundle, saved_paths)


def _print_summary(bundle: object, saved_paths: list[object]) -> None:
    print("Generated dataset bundle:")
    for split_name, split in bundle.splits.items():
        print(
            f"  {split_name}: episodes={split.episode_count}, "
            f"transitions={split.transition_count}, "
            f"obs_shape={list(split.observations.shape)}"
        )
    print(f"  output_dir: {bundle.config.output_dir}")
    print(f"  files: {', '.join(str(path.name) for path in saved_paths)}")


if __name__ == "__main__":
    main()
