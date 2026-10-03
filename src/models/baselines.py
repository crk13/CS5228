"""Small reference models, with all preprocessing owned by their wrappers."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from src.config import SEED
from src.cv import DATE_COL
from src.features import BASIC_COLUMNS
from .base import ModelWrapper


class RecentMedianRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, months: int = 6):
        self.months = months

    def fit(self, X, y):
        if self.months < 1:
            raise ValueError("months must be positive")
        months = pd.to_datetime(X[DATE_COL], format="%Y-%m").dt.to_period("M")
        recent = months >= months.max() - (self.months - 1)
        frame = X[["TOWN", "FLAT_TYPE"]].copy()
        frame["rent"] = np.asarray(y, dtype=float)
        frame = frame.loc[recent]
        self.groups_ = frame.groupby(["TOWN", "FLAT_TYPE"], dropna=False)["rent"].median()
        self.flat_types_ = frame.groupby("FLAT_TYPE", dropna=False)["rent"].median()
        self.global_ = float(frame["rent"].median())
        return self

    def predict(self, X):
        check_is_fitted(self, "groups_")
        keys = pd.MultiIndex.from_frame(X[["TOWN", "FLAT_TYPE"]])
        values = pd.Series(self.groups_.reindex(keys).to_numpy(), index=X.index)
        return values.fillna(X["FLAT_TYPE"].map(self.flat_types_)).fillna(self.global_).to_numpy()


def group_median_model() -> ModelWrapper:
    return ModelWrapper(RecentMedianRegressor(), ["TOWN", "FLAT_TYPE", DATE_COL])


def _categorical_values(X):
    return X.astype(object).where(X.notna(), np.nan).to_numpy()


def hist_gradient_boosting_model(target_transform: str = "none") -> ModelWrapper:
    numeric = [
        "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "RENT_MONTH", *BASIC_COLUMNS,
        "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
    ]
    categorical = ["TOWN", "FLAT_MODEL", "SUBZONE", "PLANNING_AREA", "REGION"]
    encoder = Pipeline([
        ("values", FunctionTransformer(_categorical_values, feature_names_out="one-to-one")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan)),
    ])
    estimator = Pipeline([
        ("columns", ColumnTransformer([
            ("numeric", "passthrough", numeric), ("categorical", encoder, categorical),
        ], sparse_threshold=0)),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30,
            l2_regularization=1.0, early_stopping=False, random_state=SEED,
            categorical_features=[False] * len(numeric) + [True] * len(categorical),
        )),
    ])
    return ModelWrapper(estimator, numeric + categorical, target_transform)
