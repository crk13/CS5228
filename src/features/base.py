"""Independent feature blocks fitted on the current training fold only."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator

from src.data_cleaning import TARGET, ID_COL


class FeatureBlock(BaseEstimator, ABC):
    def fit(self, train_df: pd.DataFrame):
        """May read TARGET here; do not load training data independently."""
        return self

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only added columns, with exactly the input index and row order."""

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Override for OOF training encodings; retain full-fold state for transform."""
        self.fit(train_df)
        return self.transform(train_df.drop(columns=[TARGET, ID_COL], errors="ignore"))
