from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldModelConfig:
    observation_channels: int = 4
    observation_height: int = 7
    observation_width: int = 7
    action_dim: int = 4
    encoder_hidden_dim: int = 128
    observation_embedding_dim: int = 64
    gru_hidden_dim: int = 96
    decoder_hidden_dim: int = 128

    @property
    def observation_size(self) -> int:
        return self.observation_channels * self.observation_height * self.observation_width

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EncoderGruDecoderWorldModel(nn.Module):
    """Small MLP encoder + GRU + MLP decoder for next-observation prediction.

    Input:
    - observations: [B, T, 4, 7, 7]
    - actions: [B, T] integer class ids

    Output:
    - next_observation_logits: [B, T, 4, 7, 7]
    """

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()

        self.encoder = nn.Sequential(
            nn.Linear(self.config.observation_size, self.config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.encoder_hidden_dim, self.config.observation_embedding_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=self.config.observation_embedding_dim + self.config.action_dim,
            hidden_size=self.config.gru_hidden_dim,
            batch_first=True,
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.config.gru_hidden_dim, self.config.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.decoder_hidden_dim, self.config.observation_size),
        )

    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        hidden_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_inputs(observations, actions)
        batch_size, sequence_length = observations.shape[:2]

        flattened_observations = observations.reshape(batch_size * sequence_length, self.config.observation_size)
        observation_embeddings = self.encoder(flattened_observations).reshape(
            batch_size,
            sequence_length,
            self.config.observation_embedding_dim,
        )

        action_one_hot = F.one_hot(actions.long(), num_classes=self.config.action_dim).to(observations.dtype)
        gru_inputs = torch.cat([observation_embeddings, action_one_hot], dim=-1)
        gru_outputs, hidden_state_next = self.gru(gru_inputs, hidden_state)

        decoded = self.decoder(gru_outputs.reshape(batch_size * sequence_length, self.config.gru_hidden_dim))
        logits = decoded.reshape(
            batch_size,
            sequence_length,
            self.config.observation_channels,
            self.config.observation_height,
            self.config.observation_width,
        )
        return logits, hidden_state_next

    def _validate_inputs(self, observations: Tensor, actions: Tensor) -> None:
        expected_obs_shape = (
            self.config.observation_channels,
            self.config.observation_height,
            self.config.observation_width,
        )
        if observations.ndim != 5:
            raise ValueError(f"observations must have shape [B, T, C, H, W], got {tuple(observations.shape)}")
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape [B, T], got {tuple(actions.shape)}")
        if tuple(observations.shape[2:]) != expected_obs_shape:
            raise ValueError(f"Unexpected observation shape: {tuple(observations.shape[2:])}")
        if observations.shape[:2] != actions.shape:
            raise ValueError("observations and actions must share batch and sequence dimensions.")
