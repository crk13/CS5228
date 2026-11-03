"""Monthly target aggregation, causal windows and fixed-origin recursive forecasts."""

import numpy as np
import pandas as pd
import torch

from .datasets import monthly_windows
from .temporal import MarketEncoder
from .training import seed_everything

DATE = "RENT_APPROVAL_DATE"


def monthly_market(X, y):
    values = np.asarray(y, dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Monthly market requires one finite positive rent per row")
    months = pd.PeriodIndex(X[DATE], freq="M")
    # Positional construction avoids aligning y against a nontrivial DataFrame index.
    frame = pd.DataFrame({"month": months, "rent": values})
    grouped = frame.groupby("month").rent.agg(["median", "size"])
    calendar = pd.period_range(months.min(), months.max(), freq="M")
    grouped = grouped.reindex(calendar)
    grouped["observed"] = grouped["median"].notna()
    grouped["median"] = grouped["median"].ffill()  # never bfill/interpolate from future
    grouped["size"] = grouped["size"].fillna(0).astype(int)
    return grouped


class MarketForecaster:
    def __init__(self, kind="carry", lookback=6, hidden_size=16, epochs=160,
                 learning_rate=0.005, damping=0.8, inner_months=6, patience=25,
                 random_state=5228, threads=1, dropout=0.1, weight_decay=0.01,
                 forecast_strength=1.0):
        self.kind = kind
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.damping = damping
        self.inner_months = inner_months
        self.patience = patience
        self.random_state = random_state
        self.threads = threads
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.forecast_strength = forecast_strength

    def _rollout(self, history, horizons, network=None, center=None, scale=None):
        values = list(np.asarray(history, dtype=float))
        predictions = []
        if network is not None:
            network.eval()
        for horizon in range(1, horizons + 1):
            if self.kind == "carry":
                change = 0.0
            elif self.kind == "damped":
                # Reuse the fitted history slope, not the query batch or future truth.
                k = min(12, len(history))
                change = float(np.polyfit(np.arange(k), np.asarray(history)[-k:], 1)[0])
            else:
                window = (np.asarray(values[-self.lookback:]) - center) / scale
                with torch.inference_mode():
                    change = float(network(torch.tensor(window[None, :, None], dtype=torch.float32))) * scale
            value = max(1.0, values[-1] + change * self.damping ** horizon * self.forecast_strength)
            if not np.isfinite(value):
                raise ValueError("Nonfinite recursive market forecast")
            values.append(value)
            predictions.append(value)
        return np.asarray(predictions)

    def _fit_network(self, history, epochs, validation=None):
        seed_everything(self.random_state, self.threads)
        center = float(np.mean(history))
        scale = max(float(np.std(history)), 1.0)
        windows, targets, positions = monthly_windows(history, self.lookback, center, scale)
        network = MarketEncoder(self.kind, self.hidden_size, self.dropout)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        best, best_epoch, stale, log = float("inf"), 1, 0, []
        for epoch in range(1, epochs + 1):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (network(windows) - targets).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite temporal loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            row = {"epoch": epoch, "train_mse_scaled_increment": float(loss.detach())}
            if validation is not None:
                prediction = self._rollout(history, len(validation), network, center, scale)
                score = float(np.sqrt(np.mean((prediction - validation) ** 2)))
                row["inner_market_rmse_sgd"] = score
                if score < best - 1e-5:
                    best, best_epoch, stale = score, epoch, 0
                else:
                    stale += 1
            log.append(row)
            if validation is not None and stale >= self.patience:
                break
        return network, center, scale, log, best_epoch, positions

    def fit(self, table):
        if self.kind not in ("carry", "damped", "rnn", "lstm"):
            raise ValueError("Unknown market encoder")
        if not 0 < self.damping <= 1 or self.lookback < 1 or self.inner_months < 1 or self.epochs < 1 or not 0 < self.forecast_strength <= 1:
            raise ValueError("Invalid market forecast parameters")
        self.history_ = table["median"].copy()
        self.origin_ = self.history_.index[-1]
        self.selection_history_ = []
        self.audit_ = {"origin": str(self.origin_), "kind": self.kind, "months": len(table),
                       "lookback": self.lookback, "damping": self.damping,
                       "unobserved_months": int((~table.observed).sum()),
                       "future_actuals_used": False, "recursive": True}
        self.audit_.update(dropout=self.dropout, weight_decay=self.weight_decay,
                           forecast_strength=self.forecast_strength)
        if self.kind in ("carry", "damped"):
            self.selected_epochs_ = 0
            return self
        history = self.history_.to_numpy(float)
        if len(history) <= self.lookback:
            raise ValueError("Insufficient monthly history for RNN/LSTM")
        prefix = len(history) - self.inner_months
        if prefix > self.lookback:
            _, _, _, self.selection_history_, selected, _ = self._fit_network(
                history[:prefix], self.epochs, history[prefix:])
            self.audit_["inner_train_end"] = str(self.history_.index[prefix-1])
            self.audit_["inner_valid_start"] = str(self.history_.index[prefix])
        else:
            selected = self.epochs
        self.selected_epochs_ = selected
        self.network_, self.center_, self.scale_, self.training_history_, _, positions = self._fit_network(history, selected)
        self.audit_.update(selected_epochs=selected, windows=len(positions),
                           first_target_month=str(self.history_.index[positions[0]]),
                           last_target_month=str(self.history_.index[positions[-1]]),
                           last_input_month=str(self.history_.index[positions[-1]-1]))
        return self

    def predict_months(self, months):
        periods = pd.PeriodIndex(months, freq="M")
        if not len(periods):
            return np.empty(0)
        if (periods < self.history_.index.min()).any():
            raise ValueError("Cannot predict months preceding fitted history")
        horizons = np.asarray(periods.asi8 - self.origin_.ordinal)
        max_horizon = max(0, int(horizons.max()))
        future = self._rollout(self.history_.to_numpy(), max_horizon,
                               getattr(self, "network_", None), getattr(self, "center_", None),
                               getattr(self, "scale_", None))
        result = np.empty(len(months), dtype=float)
        observed = horizons <= 0
        result[observed] = self.history_.reindex(periods[observed]).to_numpy()
        result[~observed] = future[horizons[~observed]-1]
        return result
