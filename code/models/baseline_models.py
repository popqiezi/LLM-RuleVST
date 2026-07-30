from __future__ import annotations

import math

import torch
from torch import nn


class DualHeadOutputMixin:
    prediction_horizon: int

    def make_heads(
        self,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dimension, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.prediction_horizon * 2),
        )
        self.cri_head = nn.Sequential(
            nn.Linear(hidden_dimension, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.prediction_horizon),
            nn.Sigmoid(),
        )

    def decode_heads(self, representation: torch.Tensor):
        trajectory = self.trajectory_head(representation).view(
            -1, self.prediction_horizon, 2
        )
        cri = self.cri_head(representation).view(
            -1, self.prediction_horizon, 1
        )
        return trajectory, cri


class LSTMForecast(nn.Module, DualHeadOutputMixin):
    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 512,
        number_of_layers: int = 2,
        dropout: float = 0.1,
        prediction_horizon: int = 20,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(
            input_dimension,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.make_heads(hidden_dimension, dropout)

    def forward(self, x: torch.Tensor):
        _, (hidden, _) = self.encoder(x)
        return self.decode_heads(hidden[-1])


class Seq2SeqLSTMForecast(nn.Module):
    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 512,
        number_of_layers: int = 2,
        dropout: float = 0.1,
        prediction_horizon: int = 20,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(
            input_dimension,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.decoder = nn.LSTM(
            hidden_dimension,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.start_projection = nn.Linear(
            input_dimension, hidden_dimension
        )
        self.trajectory_projection = nn.Linear(
            hidden_dimension, 2
        )
        self.cri_projection = nn.Sequential(
            nn.Linear(hidden_dimension, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        _, state = self.encoder(x)
        decoder_input = self.start_projection(
            x[:, -1]
        ).unsqueeze(1).repeat(
            1, self.prediction_horizon, 1
        )
        decoded, _ = self.decoder(decoder_input, state)
        return (
            self.trajectory_projection(decoded),
            self.cri_projection(decoded),
        )


class AttentionLSTMForecast(nn.Module):
    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 512,
        number_of_layers: int = 2,
        dropout: float = 0.1,
        prediction_horizon: int = 20,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.encoder = nn.LSTM(
            input_dimension,
            hidden_dimension,
            num_layers=number_of_layers,
            batch_first=True,
            dropout=dropout if number_of_layers > 1 else 0.0,
        )
        self.queries = nn.Parameter(
            torch.randn(prediction_horizon, hidden_dimension)
            / math.sqrt(hidden_dimension)
        )
        self.trajectory_projection = nn.Linear(
            hidden_dimension, 2
        )
        self.cri_projection = nn.Sequential(
            nn.Linear(hidden_dimension, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        encoded, _ = self.encoder(x)
        queries = self.queries.unsqueeze(0).expand(
            x.shape[0], -1, -1
        )
        scores = torch.matmul(
            queries, encoded.transpose(1, 2)
        ) / math.sqrt(encoded.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, encoded)
        return (
            self.trajectory_projection(context),
            self.cri_projection(context),
        )


class SinusoidalEncoding(nn.Module):
    def __init__(
        self,
        dimension: int,
        maximum_length: int = 40,
    ) -> None:
        super().__init__()
        position = torch.arange(
            maximum_length, dtype=torch.float32
        ).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, dimension, 2).float()
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(
            maximum_length, dimension
        )
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer(
            "encoding", encoding.unsqueeze(0)
        )

    def forward(self, x: torch.Tensor):
        return x + self.encoding[:, :x.shape[1]]


class TransformerForecast(nn.Module, DualHeadOutputMixin):
    def __init__(
        self,
        input_dimension: int = 4,
        latent_dimension: int = 512,
        number_of_layers: int = 2,
        number_of_attention_heads: int = 8,
        feedforward_dimension: int = 2048,
        dropout: float = 0.1,
        prediction_horizon: int = 20,
    ) -> None:
        super().__init__()
        self.prediction_horizon = prediction_horizon
        self.input_projection = nn.Linear(
            input_dimension, latent_dimension
        )
        self.position = SinusoidalEncoding(
            latent_dimension, 40
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dimension,
            nhead=number_of_attention_heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, number_of_layers
        )
        self.make_heads(latent_dimension, dropout)

    def forward(self, x: torch.Tensor):
        encoded = self.encoder(
            self.position(self.input_projection(x))
        )
        return self.decode_heads(encoded[:, -1])


def build_baseline(name: str, **kwargs) -> nn.Module:
    normalized = name.lower()
    if normalized == "lstm":
        return LSTMForecast(**kwargs)
    if normalized == "seq2seq_lstm":
        return Seq2SeqLSTMForecast(**kwargs)
    if normalized == "attention_lstm":
        return AttentionLSTMForecast(**kwargs)
    if normalized == "transformer":
        return TransformerForecast(**kwargs)
    raise ValueError(f"Unsupported baseline: {name}")
