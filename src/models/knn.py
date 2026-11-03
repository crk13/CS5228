"""KNN regressors on location, flat attributes and rental month.

knn_model is the basic delivered version; rent_knn_model adds the configurable improvements.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .base import ModelWrapper

KNN_FEATURES = ["basic", "block"]
# MONTH_INDEX is a neighbour dimension, so later months match the most recent training rows.
KNN_COLUMNS = [
    "LATITUDE", "LONGITUDE", "FLOOR_AREA_SQM", "FLAT_TYPE_ORDINAL", "REMAINING_LEASE", "MONTH_INDEX",
]
GEO = ("LATITUDE", "LONGITUDE")
KM_PER_DEGREE = 111.32
SINGAPORE_LATITUDE = 1.35
RIDGE_CATEGORICAL = ("FLAT_TYPE", "FLAT_MODEL", "TOWN")
RIDGE_NUMERIC = ("FLOOR_AREA_SQM", "REMAINING_LEASE")
ALL_GROUPS = "__all__"


def knn_model(n_neighbors: int = 20, weights: str = "distance", target_transform: str = "none") -> ModelWrapper:
    estimator = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1),
    )
    return ModelWrapper(estimator, KNN_COLUMNS, target_transform)


def _average(distance, values, weights):
    """Same rule as KNeighborsRegressor: exact matches, when present, take all the weight."""
    if weights == "uniform":
        return values.mean(axis=1)
    with np.errstate(divide="ignore"):
        w = 1.0 / distance
    exact = np.isinf(w)
    rows = exact.any(axis=1)
    w[rows] = exact[rows]
    return (w * values).sum(axis=1) / w.sum(axis=1)


class RentKNNRegressor(RegressorMixin, BaseEstimator):
    """KNN on standardized features, each multiplied by its weight.

    - geo_km: latitude/longitude in km with one shared scale, so distance is isotropic.
    - clip_month: rows after the last training month use that month's MONTH_INDEX.
    - level="market": learn y / monthly mean y; predict times the mean of the last level_months.
    - base="ridge": a Ridge on one-hot flat type, model and town plus area and lease gives the
      main prediction, and KNN smooths its residual.
    - by_flat_type: a separate neighbour index per FLAT_TYPE_ORDINAL.
    - tune: choose k and feature weights by coordinate search on the last inner_months of the
      training rows, scored by RMSE on the target the estimator receives.
    """

    def __init__(self, features=tuple(KNN_COLUMNS), n_neighbors=80, weights="distance",
                 feature_weights=None, geo_km=False, clip_month=False, level="none", level_months=3,
                 base="none", alpha=1.0, by_flat_type=False, tune=False, inner_months=6,
                 k_grid=(20, 40, 80, 160), weight_grid=(0.0, 0.5, 1.0, 2.0)):
        self.features = features
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.feature_weights = feature_weights
        self.geo_km = geo_km
        self.clip_month = clip_month
        self.level = level
        self.level_months = level_months
        self.base = base
        self.alpha = alpha
        self.by_flat_type = by_flat_type
        self.tune = tune
        self.inner_months = inner_months
        self.k_grid = k_grid
        self.weight_grid = weight_grid

    def _initial_weights(self):
        given = dict(self.feature_weights or {})
        unknown = set(given) - set(self.features)
        if unknown:
            raise ValueError(f"feature_weights for unused features: {sorted(unknown)}")
        return {f: float(given.get(f, 1.0)) for f in self.features}

    def _design(self, X):
        Z = X[list(self.features)].astype(float)
        if self.clip_month and "MONTH_INDEX" in Z:
            Z["MONTH_INDEX"] = Z["MONTH_INDEX"].clip(upper=self.month_max_)
        Z = Z.fillna(self.medians_)
        if self.geo_km:
            Z["LATITUDE"] = Z["LATITUDE"] * KM_PER_DEGREE
            Z["LONGITUDE"] = Z["LONGITUDE"] * KM_PER_DEGREE * np.cos(np.radians(SINGAPORE_LATITUDE))
        return Z

    def _scaled(self, X):
        Z = (self._design(X) - self.center_) / self.scale_
        return (Z * pd.Series(self.feature_weights_)).to_numpy()

    def _ridge_frame(self, X):
        frame = X[list(RIDGE_CATEGORICAL)].astype(str)
        return frame.join(X[list(RIDGE_NUMERIC)].astype(float))

    def _groups(self, X):
        if self.by_flat_type:
            return X["FLAT_TYPE_ORDINAL"].to_numpy()
        return np.full(len(X), ALL_GROUPS, dtype=object)

    def _tune(self, X, y):
        month = X["MONTH_INDEX"].to_numpy(dtype=float)
        inner = month > month.max() - self.inner_months
        X_fit, X_val, y_fit, y_val = X[~inner], X[inner], y[~inner], y[inner]

        def score(k, w):
            params = {**self.get_params(), "tune": False, "n_neighbors": k, "feature_weights": w}
            model = type(self)(**params)
            return float(np.sqrt(np.mean((y_val - model.fit(X_fit, y_fit).predict(X_val)) ** 2)))

        k, w = self.n_neighbors, self._initial_weights()
        best = score(k, w)
        for candidate in self.k_grid:
            if candidate != k and (s := score(candidate, w)) < best:
                best, k = s, candidate
        for feature in self.features:
            for candidate in self.weight_grid:
                trial = {**w, feature: float(candidate)}
                if candidate != w[feature] and (s := score(k, trial)) < best:
                    best, w = s, trial
        return k, w, best

    def fit(self, X, y):
        if self.level not in ("none", "market") or self.base not in ("none", "ridge"):
            raise ValueError("level must be none/market and base must be none/ridge")
        y = np.asarray(y, dtype=float)
        if self.tune:
            self.n_neighbors_, self.feature_weights_, self.inner_rmse_ = self._tune(X, y)
        else:
            self.n_neighbors_, self.feature_weights_ = self.n_neighbors, self._initial_weights()
        month = X["MONTH_INDEX"].to_numpy(dtype=float)
        self.month_max_ = float(month.max())
        target = y
        self.reference_ = 1.0
        if self.level == "market":
            levels = pd.Series(y).groupby(month).mean()
            target = y / levels.reindex(month).to_numpy()
            self.reference_ = float(levels.iloc[-self.level_months:].mean())
        if self.base == "ridge":
            self.ridge_ = make_pipeline(ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(RIDGE_CATEGORICAL)),
                ("numeric", StandardScaler(), list(RIDGE_NUMERIC)),
            ]), Ridge(alpha=self.alpha))
            frame = self._ridge_frame(X)
            target = target - self.ridge_.fit(frame, target).predict(frame)
        self.medians_ = X[list(self.features)].astype(float).median()
        Z = self._design(X)
        self.center_ = Z.mean()
        self.scale_ = Z.std(ddof=0).replace(0, 1.0)
        if self.geo_km:
            self.scale_[list(GEO)] = float(np.sqrt((Z[list(GEO)].var(ddof=0)).mean()))
        Zs = self._scaled(X)
        groups = self._groups(X)
        self.indexes_ = {}
        for group in dict.fromkeys([*np.unique(groups), ALL_GROUPS]):  # ALL_GROUPS backs unseen groups
            mask = np.ones(len(X), bool) if group == ALL_GROUPS else groups == group
            k = min(self.n_neighbors_, int(mask.sum()))
            self.indexes_[group] = (NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Zs[mask]), target[mask])
        return self

    def predict(self, X):
        check_is_fitted(self, "indexes_")
        Zs = self._scaled(X)
        groups = self._groups(X)
        residual = np.empty(len(X))
        for group in np.unique(groups):
            mask = groups == group
            index, values = self.indexes_.get(group, self.indexes_[ALL_GROUPS])
            distance, neighbours = index.kneighbors(Zs[mask])
            residual[mask] = _average(distance, values[neighbours], self.weights)
        base = self.ridge_.predict(self._ridge_frame(X)) if self.base == "ridge" else 0.0
        return (base + residual) * self.reference_


def rent_knn_model(target_transform: str = "none", **params) -> ModelWrapper:
    estimator = RentKNNRegressor(**params)
    columns = [*estimator.features, "MONTH_INDEX"]
    if estimator.by_flat_type:
        columns.append("FLAT_TYPE_ORDINAL")
    if estimator.base == "ridge":
        columns += [*RIDGE_CATEGORICAL, *RIDGE_NUMERIC]
    return ModelWrapper(estimator, list(dict.fromkeys(columns)), target_transform)
