"""Small one-layer market encoders; each independent window resets hidden state."""

import math

import torch
from torch import nn


class MarketEncoder(nn.Module):
    def __init__(self, kind="rnn", hidden_size=16, dropout=0.1):
        super().__init__()
        if kind not in ("rnn", "lstm", "transformer"):
            raise ValueError("Unknown temporal encoder")
        self.kind = kind
        if kind == "transformer":
            if hidden_size < 2 or hidden_size % 2:
                raise ValueError("Temporal Transformer hidden_size must be even and >= 2 (two heads)")
            self.projection = nn.Linear(1, hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=2, dim_feedforward=2*hidden_size,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden_size)
        else:
            cell = nn.RNN if kind == "rnn" else nn.LSTM
            self.encoder = cell(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        # A zero-initialized increment head starts from persistence, not an arbitrary price.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, windows):
        if self.kind == "transformer":
            tokens = self.projection(windows)
            # Relative month positions within the historical window, never query labels.
            position = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype)[:, None]
            frequency = torch.exp(torch.arange(0, tokens.shape[2], 2, device=tokens.device,
                                                dtype=tokens.dtype)*(-math.log(10000.)/tokens.shape[2]))
            encoding = torch.zeros_like(tokens[0])
            encoding[:, 0::2] = torch.sin(position*frequency)
            encoding[:, 1::2] = torch.cos(position*frequency)
            mask = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device,
                              dtype=torch.bool).triu(diagonal=1)
            outputs = self.norm(self.encoder(tokens+encoding, mask=mask))
        else:
            outputs, _ = self.encoder(windows)
        return self.head(outputs[:, -1]).squeeze(-1)
