"""Ridge on rent / monthly market level, scaled back by the last training month's level."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

DATE = "RENT_APPROVAL_DATE"
CATS = ["SUBZONE", "BLOCK_KEY", "FLAT_TYPE", "FLAT_MODEL"]
NUMS = ["FLOOR_AREA_SQM", "FLAT_AGE", "MRT_KM"]
REQUIRED = ["SUBZONE", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM",
            "RENT_YEAR", "LEASE_COMMENCE_DATE", "MRT_KM", DATE]


def _prepare(X):
    """Derive FLAT_AGE and BLOCK_KEY from raw columns; categoricals become plain strings."""
    d = pd.DataFrame(index=X.index)
    d["SUBZONE"] = X["SUBZONE"].astype(str)
    d["BLOCK_KEY"] = X["BLOCK"].astype(str) + " " + X["STREET"].astype(str)
    d["FLAT_TYPE"] = X["FLAT_TYPE"].astype(str)
    d["FLAT_MODEL"] = X["FLAT_MODEL"].astype(str)
    d["FLOOR_AREA_SQM"] = X["FLOOR_AREA_SQM"].astype(float)
    d["FLAT_AGE"] = (X["RENT_YEAR"] - X["LEASE_COMMENCE_DATE"]).astype(float)
    d["MRT_KM"] = X["MRT_KM"].astype(float)
    return d


def _design(train, test):
    enc = OneHotEncoder(handle_unknown="ignore")
    a = sparse.hstack([enc.fit_transform(train[CATS]), sparse.csr_matrix(train[NUMS].to_numpy())]).tocsr()
    b = sparse.hstack([enc.transform(test[CATS]), sparse.csr_matrix(test[NUMS].to_numpy())]).tocsr()
    return a, b


class MarketRatioRidge(RegressorMixin, BaseEstimator):
    """Fit Ridge on rent / monthly median rent; predict ratio * last training month's level.

    Alpha is chosen inside the training rows: the last 6 months are used, with the final
    2 of them as an inner validation block. Needs RENT_APPROVAL_DATE in the model columns.
    """

    def __init__(self, alphas=(0.3, 1, 3, 10, 30, 100), recent_months=6):
        self.alphas = alphas
        self.recent_months = recent_months

    def _pick_alpha(self, d, y, months):
        keep = (months >= np.sort(months.unique())[-self.recent_months]).to_numpy()
        d, y, months = d[keep].reset_index(drop=True), y[keep], months[keep].reset_index(drop=True)
        last2 = np.sort(months.unique())[-2]
        inner = (months < last2).to_numpy()
        a, b = _design(d[inner], d[~inner])
        scores = {}
        for alpha in self.alphas:
            pred = Ridge(alpha=alpha).fit(a, y[inner]).predict(b)
            scores[alpha] = float(np.sqrt(np.mean((y[~inner] - pred) ** 2)))
        return min(scores, key=scores.get), scores

    def fit(self, X, y):
        missing = [c for c in REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"Missing model columns: {missing}")
        y = np.asarray(y, dtype=float)
        months = X[DATE].astype(str).reset_index(drop=True)
        d = _prepare(X).reset_index(drop=True)
        self.levels_ = pd.Series(y).groupby(months).median()
        self.level_ = float(self.levels_.iloc[-1])
        ratio = y / months.map(self.levels_).to_numpy(dtype=float)
        self.alpha_, self.alpha_scores_ = self._pick_alpha(d, y, months)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore")
        a = sparse.hstack([self.encoder_.fit_transform(d[CATS]), sparse.csr_matrix(d[NUMS].to_numpy())]).tocsr()
        self.ridge_ = Ridge(alpha=self.alpha_).fit(a, ratio)
        return self

    def predict(self, X):
        check_is_fitted(self, "ridge_")
        d = _prepare(X)
        b = sparse.hstack([self.encoder_.transform(d[CATS]), sparse.csr_matrix(d[NUMS].to_numpy())]).tocsr()
        return self.ridge_.predict(b) * self.level_
