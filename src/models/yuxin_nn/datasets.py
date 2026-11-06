"""Tensor conversion and strictly historical monthly windows."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def row_dataset(numeric, categorical, target):
    return TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical),
                         torch.as_tensor(target, dtype=torch.float32))


def monthly_windows(values, lookback, center, scale):
    values = np.asarray(values, dtype=float)
    if lookback < 1 or len(values) <= lookback:
        raise ValueError("Need more months than lookback")
    # Window ending at t-1 predicts the increment from t-1 to t.
    positions = np.arange(lookback, len(values))
    windows = np.stack([(values[t-lookback:t] - center) / scale for t in positions])
    target = (values[positions] - values[positions-1]) / scale
    return (torch.tensor(windows[:, :, None], dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32), positions)
