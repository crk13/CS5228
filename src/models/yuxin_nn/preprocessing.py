"""Training-only numerical statistics and categorical vocabularies."""

import numpy as np
import pandas as pd

NUMERIC = (
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "RENT_MONTH", "MONTH_INDEX",
    "FLAT_AGE", "REMAINING_LEASE", "FLAT_TYPE_ORDINAL", "LATITUDE",
    "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
)
STATIC_NUMERIC = tuple(c for c in NUMERIC if c not in ("MONTH_INDEX", "RENT_MONTH"))
CATEGORICAL = (
    "TOWN", "FLAT_TYPE", "FLAT_MODEL", "STREET", "SUBZONE", "PLANNING_AREA",
    "REGION", "BLOCK_KEY",
)


def categorical_frame(X, columns):
    frame = X.copy()
    if "BLOCK_KEY" in columns:
        # Both components are needed: a block number alone is not an address.
        frame["BLOCK_KEY"] = frame["BLOCK"].astype("string") + "|" + frame["STREET"].astype("string")
    return frame[list(columns)].astype("string")


class TabularPreprocessor:
    """UNK=0, MISSING=1, frequent training categories start at 2."""

    def __init__(self, numeric=NUMERIC, categorical=CATEGORICAL, min_frequency=5):
        self.numeric = numeric
        self.categorical = categorical
        self.min_frequency = min_frequency

    def fit(self, X):
        if self.min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        values = X[list(self.numeric)].to_numpy(dtype=float, na_value=np.nan)
        values[~np.isfinite(values)] = np.nan
        # Explicit fallback also supports an all-missing training column.
        self.medians_ = np.array([
            np.nanmedian(v) if np.isfinite(v).any() else 0.0 for v in values.T
        ])
        filled = np.where(np.isnan(values), self.medians_, values)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = filled.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        cats = categorical_frame(X, self.categorical)
        self.vocabularies_ = {}
        for column in self.categorical:
            counts = cats[column].value_counts()
            frequent = sorted(counts[counts >= self.min_frequency].index.tolist())
            self.vocabularies_[column] = {v: i + 2 for i, v in enumerate(frequent)}
        self.cardinalities_ = [len(self.vocabularies_[c]) + 2 for c in self.categorical]
        self.numeric_width_ = len(self.numeric) * 2  # fixed missing indicator per field
        return self

    def transform(self, X):
        values = X[list(self.numeric)].to_numpy(dtype=float, na_value=np.nan)
        missing = ~np.isfinite(values)
        filled = np.where(missing, self.medians_, values)
        numeric = np.concatenate([(filled - self.mean_) / self.scale_, missing], axis=1).astype(np.float32)
        cats = categorical_frame(X, self.categorical)
        encoded = []
        for column in self.categorical:
            values = cats[column]
            ids = values.map(self.vocabularies_[column]).fillna(0).to_numpy(dtype=np.int64)
            ids[values.isna().to_numpy()] = 1
            encoded.append(ids)
        categorical = np.stack(encoded, axis=1) if encoded else np.empty((len(X), 0), dtype=np.int64)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite numerical tensor after preprocessing")
        return numeric, categorical
