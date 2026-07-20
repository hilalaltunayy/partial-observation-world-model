from __future__ import annotations

import argparse

import torch

try:
    from evaluation.checkpoint_loader import load_world_model_checkpoint
    from evaluation.world_model_evaluation import (
        evaluate_persistence_baseline,
        evaluate_world_model,
        format_evaluation_summary,
    )
    from training.world_model_training import build_dataloader, build_sequence_datasets
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.evaluation.evaluate_world_model`
    from src.evaluation.checkpoint_loader import load_world_model_checkpoint
    from src.evaluation.world_model_evaluation import (
        evaluate_persistence_baseline,
        evaluate_world_model,
        format_evaluation_summary,
    )
    from src.training.world_model_training import build_dataloader, build_sequence_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained world model checkpoint against a dataset split.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint file.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the checkpoint metadata JSON file.")
    parser.add_argument("--data-dir", type=str, default="data/generated", help="Directory containing generated dataset splits.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument("--batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic dataloader seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_checkpoint = load_world_model_checkpoint(
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        device=args.device,
    )

    training_config = loaded_checkpoint.metadata.get("training_config", {})
    sequence_length = int(training_config.get("sequence_length", 8))
    datasets = build_sequence_datasets(
        data_dir=args.data_dir,
        sequence_length=sequence_length,
        smoke_test=False,
    )
    dataloader = build_dataloader(
        dataset=datasets[args.split],
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    device = torch.device(args.device)

    model_summary = evaluate_world_model(loaded_checkpoint.model, dataloader, device)
    baseline_summary = evaluate_persistence_baseline(dataloader, device)

    print(
        f"Evaluating split='{args.split}' with sequence_length={sequence_length} "
        f"from checkpoint='{loaded_checkpoint.checkpoint_path.name}' on device='{device.type}'."
    )
    print()
    print(format_evaluation_summary("Trained world model", model_summary))
    print()
    print(format_evaluation_summary("Persistence baseline", baseline_summary))
    print()
    print(_build_comparison_line(model_summary, baseline_summary))


def _build_comparison_line(model_summary: object, baseline_summary: object) -> str:
    model_f1 = sum(channel.f1_score for channel in model_summary.per_channel.values()) / len(model_summary.per_channel)
    baseline_f1 = sum(channel.f1_score for channel in baseline_summary.per_channel.values()) / len(
        baseline_summary.per_channel
    )
    if model_summary.loss < baseline_summary.loss and model_f1 > baseline_f1:
        return "Comparison: the trained world model beats the persistence baseline on both BCE loss and mean per-channel F1."
    if model_summary.loss < baseline_summary.loss:
        return "Comparison: the trained world model improves BCE loss, but not mean per-channel F1."
    if model_f1 > baseline_f1:
        return "Comparison: the trained world model improves mean per-channel F1, but not BCE loss."
    return "Comparison: the persistence baseline remains competitive or better on the tested summary metrics."


if __name__ == "__main__":
    main()
