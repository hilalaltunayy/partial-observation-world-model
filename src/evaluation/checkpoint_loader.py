from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

try:
    from models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig
except ModuleNotFoundError:  # pragma: no cover - supports `python -m src.evaluation.evaluate_world_model`
    from src.models.world_model import EncoderGruDecoderWorldModel, WorldModelConfig


REQUIRED_MODEL_CONFIG_KEYS: tuple[str, ...] = (
    "observation_channels",
    "observation_height",
    "observation_width",
    "action_dim",
    "encoder_hidden_dim",
    "observation_embedding_dim",
    "gru_hidden_dim",
    "decoder_hidden_dim",
)


@dataclass(frozen=True)
class LoadedWorldModelCheckpoint:
    model: EncoderGruDecoderWorldModel
    metadata: dict[str, Any]
    checkpoint_payload: dict[str, Any]
    checkpoint_path: Path
    metadata_path: Path
    device: torch.device


def load_world_model_metadata(metadata_path: Path | str) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata file: {path}")

    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Checkpoint metadata is not valid JSON: {path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object.")

    model_config = metadata.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint metadata must include a 'model_config' object.")

    missing_keys = [key for key in REQUIRED_MODEL_CONFIG_KEYS if key not in model_config]
    if missing_keys:
        raise ValueError(f"Checkpoint metadata is missing model_config keys: {missing_keys}")

    for key in REQUIRED_MODEL_CONFIG_KEYS:
        value = model_config[key]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Checkpoint metadata field '{key}' must be a positive integer.")

    training_config = metadata.get("training_config")
    if training_config is not None:
        if not isinstance(training_config, dict):
            raise ValueError("Checkpoint metadata field 'training_config' must be an object when provided.")
        if "sequence_length" in training_config:
            sequence_length = training_config["sequence_length"]
            if not isinstance(sequence_length, int) or sequence_length <= 0:
                raise ValueError("Checkpoint metadata training_config.sequence_length must be a positive integer.")

    return metadata


def load_world_model_checkpoint(
    checkpoint_path: Path | str,
    metadata_path: Path | str,
    device: str | torch.device = "cpu",
) -> LoadedWorldModelCheckpoint:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_file}")

    metadata_file = Path(metadata_path)
    metadata = load_world_model_metadata(metadata_file)
    model_config = WorldModelConfig(**metadata["model_config"])

    map_location = torch.device(device)
    payload = torch.load(checkpoint_file, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if "model_state_dict" not in payload:
        raise ValueError("Checkpoint payload is missing 'model_state_dict'.")
    if not isinstance(payload["model_state_dict"], dict):
        raise ValueError("Checkpoint field 'model_state_dict' must be a state-dict object.")

    checkpoint_model_config = payload.get("model_config")
    if checkpoint_model_config is not None and checkpoint_model_config != metadata["model_config"]:
        raise ValueError("Checkpoint payload model_config does not match metadata model_config.")

    model = EncoderGruDecoderWorldModel(model_config).to(map_location)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Incompatible checkpoint tensor shapes or parameter names: {exc}") from exc
    model.eval()

    return LoadedWorldModelCheckpoint(
        model=model,
        metadata=metadata,
        checkpoint_payload=payload,
        checkpoint_path=checkpoint_file,
        metadata_path=metadata_file,
        device=map_location,
    )
