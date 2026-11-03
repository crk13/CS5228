"""Model-owned preprocessing and optional invertible target transformations."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from src.data_cleaning import TARGET, ID_COL


class TargetTransform:
    def __init__(self, name: str = "none"):
        if name not in ("none", "log", "per_sqm"):
            raise ValueError("Target transform must be none, log, or per_sqm")
        self.name = name

    def _area(self, X):
        area = X["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        if not np.isfinite(area).all() or (area <= 0).any():
            raise ValueError("per_sqm requires finite, positive FLOOR_AREA_SQM")
        return area

    def transform(self, y, X):
        values = np.asarray(y, dtype=float)
        if self.name == "log":
            if (values <= 0).any():
                raise ValueError("log requires positive rent")
            return np.log(values)
        if self.name == "per_sqm":
            return values / self._area(X)
        return values

    def inverse_transform(self, prediction, X):
        values = np.asarray(prediction, dtype=float)
        if self.name == "log":
            return np.exp(values)
        if self.name == "per_sqm":
            return values * self._area(X)
        return values


class ModelWrapper(RegressorMixin, BaseEstimator):
    """Estimator may be an sklearn Pipeline; columns are selected inside the model."""

    def __init__(self, estimator, columns: list[str], target_transform: str = "none"):
        self.estimator = estimator
        self.columns = columns
        self.target_transform = target_transform

    def _select(self, X):
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Declare a nonempty list of unique model columns")
        if {TARGET, ID_COL}.intersection(self.columns):
            raise ValueError("Model columns must not include the target or ID")
        return X[self.columns]

    def fit(self, X: pd.DataFrame, y):
        values = np.asarray(y, dtype=float)
        if values.shape != (len(X),) or not np.isfinite(values).all():
            raise ValueError("y must contain one finite target per row")
        self.target_transform_ = TargetTransform(self.target_transform)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._select(X), self.target_transform_.transform(values, X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.target_transform_.inverse_transform(self.estimator_.predict(self._select(X)), X)
