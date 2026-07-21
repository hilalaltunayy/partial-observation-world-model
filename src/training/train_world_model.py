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
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="standard_bce",
        choices=("standard_bce", "weighted_bce"),
        help="Loss function mode.",
    )
    parser.add_argument("--pos-weight-cap", type=float, default=25.0, help="Maximum positive-class weight cap.")
    parser.add_argument("--use-balanced-sampling", action="store_true", help="Use deterministic balanced sequence sampling for training.")
    parser.add_argument(
        "--agent-sequence-sampling-weight",
        type=float,
        default=4.0,
        help="Relative weight for sequences containing scripted-agent positives.",
    )
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
        loss_mode=args.loss_mode,
        pos_weight_cap=args.pos_weight_cap,
        use_balanced_sampling=args.use_balanced_sampling,
        agent_sequence_sampling_weight=args.agent_sequence_sampling_weight,
    )
    run_training(training_config)


if __name__ == "__main__":
    main()
