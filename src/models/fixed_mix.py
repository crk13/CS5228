"""Fold-local target adjustment using a fixed mix of flat-type medians."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

from src.cv import DATE_COL

INDEX_GROUPS = ('2-room', '3-room', '4-room', '5-room', 'executive')


class FixedMixTargetRegressor(RegressorMixin, BaseEstimator):
    """Learn constant-price rents; hold the market index flat after fit origin.

    Put this INSIDE RecentWindowRegressor so index weights, medians and the
    underlying model all use exactly the same training rows. No labels or
    auxiliary files are read by predict(). One-room flats stay in model fitting
    but do not determine the market index.
    """

    def __init__(self, estimator, groups=INDEX_GROUPS):
        self.estimator = estimator
        self.groups = groups

    def fit(self, X, y):
        values = np.asarray(y, dtype=float)
        if values.shape != (len(X),) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Expected one positive finite rent for each training row')
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError('Index groups must be nonempty and unique')
        months = pd.to_datetime(X[DATE_COL], format='%Y-%m', errors='raise').dt.to_period('M')
        if months.isna().any():
            raise ValueError('Training dates must not be missing')
        table = pd.DataFrame({'month': months.astype(str).to_numpy(),
                              'type': X['FLAT_TYPE'].to_numpy(), 'rent': values})
        selected = table.loc[table.type.isin(self.groups)]
        counts = selected.groupby(['month', 'type']).size().unstack().reindex(columns=list(self.groups))
        calendar = pd.period_range(months.min(), months.max(), freq='M').astype(str)
        counts = counts.reindex(calendar)
        if counts.isna().any().any() or not counts.gt(0).all().all():
            raise ValueError('Every selected flat type needs observations in every training month')
        self.weights_ = counts.sum() / counts.to_numpy().sum()
        self.monthly_medians_ = selected.groupby(['month', 'type']).rent.median().unstack().reindex(
            index=calendar, columns=list(self.groups))
        self.market_levels_ = self.monthly_medians_.mul(self.weights_, axis=1).sum(axis=1)
        self.origin_ = str(months.max())
        self.reference_level_ = float(self.market_levels_.loc[self.origin_])
        # A dimensionless index preserves an SGD-sized target for unchanged
        # CatBoost parameters: y / (S_t / S_origin), rather than y / S_t.
        self.relative_index_ = self.market_levels_ / self.reference_level_
        self.fit_rows_ = len(table)
        self.index_rows_ = len(selected)
        self.excluded_from_index_rows_ = len(table) - len(selected)
        self.estimator_ = clone(self.estimator).fit(X, values / self.index_for(X))
        return self

    def index_for(self, X):
        check_is_fitted(self, 'relative_index_')
        months = pd.to_datetime(X[DATE_COL], format='%Y-%m', errors='raise').dt.to_period('M')
        if months.isna().any():
            raise ValueError('Prediction dates must not be missing')
        factors = months.astype(str).map(self.relative_index_).to_numpy(dtype=float, copy=True)
        future = (months > pd.Period(self.origin_, freq='M')).to_numpy()
        factors[future] = 1.0  # Last known market level; never future realized rents.
        if not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError('Cannot adjust dates before the fitted index history')
        return factors

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        prediction = np.asarray(self.estimator_.predict(X), dtype=float)
        return prediction * self.index_for(X)
