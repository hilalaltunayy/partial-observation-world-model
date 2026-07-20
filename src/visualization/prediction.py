from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from evaluation import load_world_model_checkpoint, predict_next_observation
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.visualization.app`
    from src.evaluation import load_world_model_checkpoint, predict_next_observation


@dataclass(frozen=True)
class PredictionState:
    current_observation: np.ndarray
    predicted_next_observation: np.ndarray | None
    real_next_observation: np.ndarray | None
    absolute_prediction_error: np.ndarray | None
    prediction_loss: float | None
    checkpoint_name: str
    device_label: str
    status_message: str | None


class WorldModelPredictorRuntime:
    def __init__(
        self,
        model: object | None,
        checkpoint_name: str,
        device: torch.device,
        status_message: str | None = None,
    ) -> None:
        self.model = model
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.status_message = status_message
        self.hidden_state: torch.Tensor | None = None

    @classmethod
    def from_paths(
        cls,
        checkpoint_path: Path | str,
        metadata_path: Path | str,
        device: str | torch.device = "cpu",
    ) -> "WorldModelPredictorRuntime":
        checkpoint_file = Path(checkpoint_path)
        metadata_file = Path(metadata_path)
        try:
            loaded = load_world_model_checkpoint(checkpoint_file, metadata_file, device=device)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            return cls(
                model=None,
                checkpoint_name=checkpoint_file.name,
                device=torch.device(device),
                status_message=str(exc),
            )
        return cls(
            model=loaded.model,
            checkpoint_name=loaded.checkpoint_path.name,
            device=loaded.device,
            status_message=None,
        )

    @property
    def is_available(self) -> bool:
        return self.model is not None and self.status_message is None

    def reset(self) -> None:
        self.hidden_state = None

    def prepare_idle_state(self, current_observation: np.ndarray) -> PredictionState:
        return PredictionState(
            current_observation=current_observation.astype(np.int8, copy=True),
            predicted_next_observation=None,
            real_next_observation=None,
            absolute_prediction_error=None,
            prediction_loss=None,
            checkpoint_name=self.checkpoint_name,
            device_label=self.device.type,
            status_message=self.status_message,
        )

    def prepare_transition_state(
        self,
        current_observation: np.ndarray,
        action: int,
        real_next_observation: np.ndarray,
    ) -> PredictionState:
        current_copy = current_observation.astype(np.int8, copy=True)
        next_copy = real_next_observation.astype(np.int8, copy=True)

        if not self.is_available:
            return PredictionState(
                current_observation=current_copy,
                predicted_next_observation=None,
                real_next_observation=next_copy,
                absolute_prediction_error=None,
                prediction_loss=None,
                checkpoint_name=self.checkpoint_name,
                device_label=self.device.type,
                status_message=self.status_message,
            )

        observation_tensor = torch.from_numpy(current_copy.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
        action_tensor = torch.tensor([[int(action)]], dtype=torch.int64)
        target_tensor = torch.from_numpy(next_copy.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0).to(self.device)

        logits, predictions, hidden_state_next = predict_next_observation(
            model=self.model,
            observation=observation_tensor,
            action=action_tensor,
            device=self.device,
            hidden_state=self.hidden_state,
        )
        self.hidden_state = hidden_state_next.detach()

        prediction_loss = float(F.binary_cross_entropy_with_logits(logits, target_tensor).item())
        predicted_array = predictions.squeeze(0).squeeze(0).to(torch.int8).cpu().numpy()
        error_array = np.max(
            np.abs(predicted_array.astype(np.int16) - next_copy.astype(np.int16)),
            axis=0,
        ).astype(np.int8)

        return PredictionState(
            current_observation=current_copy,
            predicted_next_observation=predicted_array,
            real_next_observation=next_copy,
            absolute_prediction_error=error_array,
            prediction_loss=prediction_loss,
            checkpoint_name=self.checkpoint_name,
            device_label=self.device.type,
            status_message=None,
        )
