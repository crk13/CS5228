"""Small one-layer market encoders; each independent window resets hidden state."""

from torch import nn


class MarketEncoder(nn.Module):
    def __init__(self, kind="rnn", hidden_size=16, dropout=0.1):
        super().__init__()
        if kind not in ("rnn", "lstm"):
            raise ValueError("Stage 1-3 supports rnn/lstm; temporal Transformer is a later stage")
        cell = nn.RNN if kind == "rnn" else nn.LSTM
        self.encoder = cell(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        # A zero-initialized increment head starts from persistence, not an arbitrary price.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, windows):
        outputs, _ = self.encoder(windows)
        return self.head(outputs[:, -1]).squeeze(-1)
