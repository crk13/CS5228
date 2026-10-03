"""Combine original flat columns and explicitly enabled feature blocks."""

import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.data_cleaning import NUM_COLS, CAT_COLS, TARGET, ID_COL
from .block import BLOCK_COLUMNS
from .registry import make_feature

BASE_COLUMNS = [c for c in NUM_COLS + CAT_COLS if c not in BLOCK_COLUMNS]


class FeaturePipeline:
    def __init__(self, names: list[str]):
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        self.blocks = [(name, make_feature(name)) for name in names]

    def _combine(self, df, outputs):
        frames = [df[BASE_COLUMNS].copy()]
        columns = set(BASE_COLUMNS)
        for name, output in outputs:
            if not isinstance(output, pd.DataFrame) or not output.index.equals(df.index):
                raise ValueError(f"Feature {name!r} must preserve the input index and row order")
            if not output.columns.is_unique or columns.intersection(output.columns):
                raise ValueError(f"Feature {name!r} returned duplicate columns")
            if {TARGET, ID_COL}.intersection(output.columns):
                raise ValueError(f"Feature {name!r} returned a target or ID column")
            columns.update(output.columns)
            frames.append(output)
        return pd.concat(frames, axis=1)

    def fit(self, train_df: pd.DataFrame):
        for _, block in self.blocks:
            block.fit(train_df.copy())
        self.columns_ = list(self._transform(train_df).columns)
        return self

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        outputs = [(name, block.fit_transform(train_df.copy())) for name, block in self.blocks]
        result = self._combine(train_df, outputs)
        self.columns_ = list(result.columns)
        return result

    def _transform(self, df):
        inputs = df.drop(columns=[TARGET, ID_COL], errors="ignore")
        outputs = [(name, block.transform(inputs.copy())) for name, block in self.blocks]
        return self._combine(inputs, outputs)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "columns_")
        result = self._transform(df)
        if list(result.columns) != self.columns_:
            raise ValueError("Feature columns changed between fit and transform")
        return result
