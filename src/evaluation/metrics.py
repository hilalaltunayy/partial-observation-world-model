from __future__ import annotations

from dataclasses import dataclass

try:
    from world_model_constants import CHANNEL_DISPLAY_NAMES, CHANNEL_NAMES
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.evaluation.evaluate_world_model`
    from src.world_model_constants import CHANNEL_DISPLAY_NAMES, CHANNEL_NAMES


@dataclass(frozen=True)
class ChannelMetrics:
    precision: float
    recall: float
    f1_score: float
    iou: float


@dataclass(frozen=True)
class EvaluationSummary:
    loss: float
    overall_binary_accuracy: float
    per_channel: dict[str, ChannelMetrics]


def compute_precision_recall_f1_iou(true_positive: int, false_positive: int, false_negative: int) -> ChannelMetrics:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    iou = _safe_divide(true_positive, true_positive + false_positive + false_negative)
    return ChannelMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou=iou,
    )


def build_channel_metrics(
    true_positive: list[int],
    false_positive: list[int],
    false_negative: list[int],
) -> dict[str, ChannelMetrics]:
    metrics: dict[str, ChannelMetrics] = {}
    for index, channel_name in enumerate(CHANNEL_NAMES):
        metrics[channel_name] = compute_precision_recall_f1_iou(
            true_positive=true_positive[index],
            false_positive=false_positive[index],
            false_negative=false_negative[index],
        )
    return metrics


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
