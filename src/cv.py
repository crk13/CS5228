"""Shared expanding-window splits. Indices are positional, for use with iloc."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

DATE_COL = "RENT_APPROVAL_DATE"


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str


HOLDOUT = SplitDefinition("holdout", "2021-01", "2023-11", "2023-12", "2025-03")
SPLITS = {
    "holdout": (HOLDOUT,),
    "rolling": (
        HOLDOUT,
        SplitDefinition("rolling_2024_04", "2021-01", "2024-03", "2024-04", "2025-03"),
    ),
}


@dataclass
class Fold:
    definition: SplitDefinition
    train_idx: np.ndarray
    valid_idx: np.ndarray


def get_splits(df: pd.DataFrame, split: str = "holdout") -> list[Fold]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {list(SPLITS)}")
    months = pd.to_datetime(df[DATE_COL], format="%Y-%m", errors="raise").dt.to_period("M")
    if months.isna().any():
        raise ValueError("Split dates must not be missing")
    folds = []
    for definition in SPLITS[split]:
        start, end, valid_start, valid_end = (
            pd.Period(value, freq="M") for value in (
                definition.train_start, definition.train_end,
                definition.valid_start, definition.valid_end,
            )
        )
        if not start <= end < valid_start <= valid_end:
            raise ValueError(f"Invalid time boundaries: {definition}")
        train_idx = np.flatnonzero(months.between(start, end).to_numpy())
        valid_idx = np.flatnonzero(months.between(valid_start, valid_end).to_numpy())
        if not len(train_idx) or not len(valid_idx):
            raise ValueError(f"Empty training or validation fold: {definition.name}")
        folds.append(Fold(definition, train_idx, valid_idx))
    return folds
