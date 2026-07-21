from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from evaluation.metrics import CHANNEL_DISPLAY_NAMES, EvaluationSummary, build_channel_metrics
    from models.world_model import EncoderGruDecoderWorldModel
    from world_model_constants import CHANNEL_NAMES
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.evaluation.evaluate_world_model`
    from src.evaluation.metrics import CHANNEL_DISPLAY_NAMES, EvaluationSummary, build_channel_metrics
    from src.models.world_model import EncoderGruDecoderWorldModel
    from src.world_model_constants import CHANNEL_NAMES


def evaluate_world_model(
    model: EncoderGruDecoderWorldModel,
    dataloader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> EvaluationSummary:
    model.eval()
    loss_sum = 0.0
    total_elements = 0
    correct_predictions = 0
    true_positive = [0 for _ in CHANNEL_NAMES]
    false_positive = [0 for _ in CHANNEL_NAMES]
    false_negative = [0 for _ in CHANNEL_NAMES]

    with torch.no_grad():
        for batch in dataloader:
            observations = batch["observations"].to(device)
            actions = batch["actions"].to(device)
            targets = batch["next_observations"].to(device)

            logits, _ = model(observations, actions)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).to(targets.dtype)

            loss_sum += float(F.binary_cross_entropy(probabilities, targets, reduction="sum").item())
            total_elements += targets.numel()
            correct_predictions += int((predictions == targets).sum().item())
            _update_channel_counts(predictions, targets, true_positive, false_positive, false_negative)

    return EvaluationSummary(
        loss=_safe_divide(loss_sum, total_elements),
        overall_binary_accuracy=_safe_divide(correct_predictions, total_elements),
        per_channel=build_channel_metrics(true_positive, false_positive, false_negative),
    )


def evaluate_persistence_baseline(
    dataloader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> EvaluationSummary:
    loss_sum = 0.0
    total_elements = 0
    correct_predictions = 0
    true_positive = [0 for _ in CHANNEL_NAMES]
    false_positive = [0 for _ in CHANNEL_NAMES]
    false_negative = [0 for _ in CHANNEL_NAMES]

    with torch.no_grad():
        for batch in dataloader:
            observations = batch["observations"].to(device)
            targets = batch["next_observations"].to(device)

            probabilities = observations.clamp(0.0, 1.0)
            probabilities = probabilities.clamp(min=1e-6, max=1.0 - 1e-6)
            predictions = (observations >= 0.5).to(targets.dtype)

            loss_sum += float(F.binary_cross_entropy(probabilities, targets, reduction="sum").item())
            total_elements += targets.numel()
            correct_predictions += int((predictions == targets).sum().item())
            _update_channel_counts(predictions, targets, true_positive, false_positive, false_negative)

    return EvaluationSummary(
        loss=_safe_divide(loss_sum, total_elements),
        overall_binary_accuracy=_safe_divide(correct_predictions, total_elements),
        per_channel=build_channel_metrics(true_positive, false_positive, false_negative),
    )


def predict_next_observation(
    model: EncoderGruDecoderWorldModel,
    observation: Tensor,
    action: Tensor,
    device: torch.device,
    hidden_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    observation = observation.to(device)
    action = action.to(device)

    if observation.shape != (1, 1, model.config.observation_channels, model.config.observation_height, model.config.observation_width):
        raise ValueError(
            "observation must have shape [1, 1, C, H, W] matching the world-model configuration."
        )
    if action.shape != (1, 1):
        raise ValueError("action must have shape [1, 1].")

    with torch.no_grad():
        logits, hidden_state_next = model(observation, action, hidden_state=hidden_state)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).to(observation.dtype)
    return logits, predictions, hidden_state_next


def format_evaluation_summary(title: str, summary: EvaluationSummary) -> str:
    lines = [
        f"{title}:",
        f"  BCE loss: {summary.loss:.6f}",
        f"  overall binary accuracy: {summary.overall_binary_accuracy:.6f}",
    ]
    for channel_name in CHANNEL_NAMES:
        metrics = summary.per_channel[channel_name]
        display_name = CHANNEL_DISPLAY_NAMES[channel_name]
        lines.append(
            "  "
            f"{display_name}: precision={metrics.precision:.6f} "
            f"recall={metrics.recall:.6f} "
            f"f1={metrics.f1_score:.6f} "
            f"iou={metrics.iou:.6f}"
        )
    return "\n".join(lines)


def _update_channel_counts(
    predictions: Tensor,
    targets: Tensor,
    true_positive: list[int],
    false_positive: list[int],
    false_negative: list[int],
) -> None:
    prediction_mask = predictions.to(torch.bool)
    target_mask = targets.to(torch.bool)
    for index, _ in enumerate(CHANNEL_NAMES):
        channel_predictions = prediction_mask[:, :, index]
        channel_targets = target_mask[:, :, index]
        true_positive[index] += int((channel_predictions & channel_targets).sum().item())
        false_positive[index] += int((channel_predictions & ~channel_targets).sum().item())
        false_negative[index] += int((~channel_predictions & channel_targets).sum().item())


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
