"""Embedding MLP and a small feature-token Transformer (not a time encoder)."""

import torch
from torch import nn


class EmbeddingMLP(nn.Module):
    def __init__(self, numeric_width, cardinalities, hidden_size=128, dropout=0.1):
        super().__init__()
        widths = [min(16, max(4, int(c ** 0.25) * 2)) for c in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(c, w, padding_idx=0) for c, w in zip(cardinalities, widths)])
        width = numeric_width + sum(widths)
        self.net = nn.Sequential(nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden_size // 2, 1))

    def forward(self, numeric, categorical):
        parts = [numeric] + [e(categorical[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat(parts, dim=1)).squeeze(-1)


class FeatureTransformer(nn.Module):
    def __init__(self, numeric_width, cardinalities, d_model=32, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        # The second half contains missing flags for the same numerical fields.
        # Combine value + missing flag into one token rather than doubling tokens.
        self.numeric_fields = numeric_width // 2
        self.numeric_weight = nn.Parameter(torch.randn(self.numeric_fields, d_model) * 0.02)
        self.numeric_missing = nn.Parameter(torch.zeros(self.numeric_fields, d_model))
        self.numeric_bias = nn.Parameter(torch.zeros(self.numeric_fields, d_model))
        self.embeddings = nn.ModuleList([nn.Embedding(c, d_model, padding_idx=0) for c in cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 2,
                                          dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, numeric, categorical):
        values = numeric[:, :self.numeric_fields].unsqueeze(-1)
        missing = numeric[:, self.numeric_fields:].unsqueeze(-1)
        tokens = [values * self.numeric_weight + missing * self.numeric_missing + self.numeric_bias]
        if self.embeddings:
            tokens.append(torch.stack([e(categorical[:, i]) for i, e in enumerate(self.embeddings)], dim=1))
        tokens.insert(0, self.cls.expand(len(numeric), -1, -1))
        return self.head(self.encoder(torch.cat(tokens, dim=1))[:, 0]).squeeze(-1)
