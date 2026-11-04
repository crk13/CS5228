"""Protect target inversion, forecast information boundaries and recent-window fit."""
import unittest
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.models.fixed_mix import FixedMixTargetRegressor
from src.models.trees import RecentWindowRegressor


class CaptureMean(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.targets_ = np.asarray(y).copy()
        self.indices_ = X.index.to_numpy()
        return self

    def predict(self, X):
        return np.full(len(X), self.targets_.mean())


class FixedMixTests(unittest.TestCase):
    def example(self):
        return pd.DataFrame({
            'RENT_APPROVAL_DATE': ['2024-01', '2024-01', '2024-02', '2024-02', '2024-01'],
            'FLAT_TYPE': ['2-room', '4-room', '2-room', '4-room', '1-room'],
        }, index=[91, 4, 75, 2, 31]), np.array([100., 200., 200., 400., 900.])

    def test_fixed_weights_target_inversion_and_rare_type_retained(self):
        X, y = self.example()
        base = CaptureMean()
        model = clone(FixedMixTargetRegressor(base, ('2-room', '4-room'))).fit(X, y)
        np.testing.assert_allclose(model.weights_, [.5, .5])
        np.testing.assert_allclose(model.market_levels_, [150, 300])
        np.testing.assert_allclose(model.estimator_.targets_, [200, 400, 200, 400, 1800])
        np.testing.assert_array_equal(model.estimator_.indices_, X.index)
        self.assertEqual(model.index_rows_, 4)
        self.assertEqual(model.fit_rows_, 5)
        np.testing.assert_allclose(model.predict(X), np.array([.5, .5, 1., 1., .5]) * 600)
        self.assertFalse(hasattr(base, 'targets_'))

    def test_future_index_is_fixed_without_future_labels(self):
        X, y = self.example()
        model = FixedMixTargetRegressor(CaptureMean(), ('2-room', '4-room')).fit(X, y)
        query = pd.DataFrame({'RENT_APPROVAL_DATE': ['2024-03', '2026-07'],
                              'FLAT_TYPE': ['never-seen', '1-room']})
        np.testing.assert_array_equal(model.index_for(query), [1., 1.])
        np.testing.assert_array_equal(model.predict(query), [600., 600.])
        self.assertEqual(model.origin_, '2024-02')
        query['RENT_APPROVAL_DATE'] = '2023-12'
        with self.assertRaises(ValueError):
            model.predict(query)

    def test_window_precedes_index_and_missing_groups_fail_explicitly(self):
        X, y = self.example()
        old = pd.DataFrame({'RENT_APPROVAL_DATE': ['2021-01'], 'FLAT_TYPE': ['2-room']}, index=[100])
        joined = pd.concat([old, X])
        model = RecentWindowRegressor(FixedMixTargetRegressor(CaptureMean(), ('2-room', '4-room')), 2)
        model.fit(joined, np.r_[1e12, y])
        self.assertEqual(model.fit_rows_, 5)
        self.assertNotIn('2021-01', model.estimator_.relative_index_.index)
        np.testing.assert_allclose(model.estimator_.weights_, [.5, .5])
        np.testing.assert_array_equal(model.estimator_.estimator_.indices_, X.index)
        with self.assertRaises(ValueError):
            FixedMixTargetRegressor(CaptureMean(), ('2-room', '5-room')).fit(X, y)


if __name__ == '__main__':
    unittest.main()
