from __future__ import annotations

import argparse

try:
    from training.world_model_training import TrainingConfig, run_training
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.training.train_world_model`
    from src.training.world_model_training import TrainingConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 1 encoder-GRU-decoder world model.")
    parser.add_argument("--data-dir", type=str, default="data/generated", help="Directory containing train/validation/test NPZ files.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--sequence-length", type=int, default=8, help="Contiguous sequence length.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device. Use cpu for local smoke testing.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic training seed.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for best validation checkpoints.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny one-epoch smoke training pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = 1 if args.smoke_test else args.epochs
    training_config = TrainingConfig(
        data_dir=args.data_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        smoke_test=args.smoke_test,
    )
    run_training(training_config)


if __name__ == "__main__":
    main()
