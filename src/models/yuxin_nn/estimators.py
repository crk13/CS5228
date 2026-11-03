"""sklearn-cloneable neural estimators; predict always returns original-scale SGD."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .market import DATE, MarketForecaster, monthly_market
from .preprocessing import CATEGORICAL, NUMERIC, STATIC_NUMERIC, TabularPreprocessor
from .tabular import EmbeddingMLP, FeatureTransformer
from .training import predict_network, seed_everything, train_rows


def validate_target(X, y):
    values = np.asarray(y, dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Expected one finite positive rent per input row")
    if not len(X):
        raise ValueError("Empty training data")
    pd.PeriodIndex(X[DATE], freq="M")
    return values


class TabularNNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, kind="mlp", epochs=30, batch_size=1024, learning_rate=0.001,
                 weight_decay=0.01, hidden_size=128, d_model=32, nhead=4, layers=2,
                 dropout=0.1, min_frequency=5, inner_months=3, patience=5,
                 market_normalize=False, random_state=5228, threads=1, verbose=False):
        self.kind = kind
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.dropout = dropout
        self.min_frequency = min_frequency
        self.inner_months = inner_months
        self.patience = patience
        self.market_normalize = market_normalize
        self.random_state = random_state
        self.threads = threads
        self.verbose = verbose

    def _preprocessor(self, X):
        numeric = STATIC_NUMERIC if self.market_normalize else NUMERIC
        return TabularPreprocessor(numeric, CATEGORICAL, self.min_frequency).fit(X)

    def _network(self, preprocessor):
        if self.kind == "mlp":
            return EmbeddingMLP(preprocessor.numeric_width_, preprocessor.cardinalities_,
                                self.hidden_size, self.dropout)
        if self.kind == "transformer":
            return FeatureTransformer(preprocessor.numeric_width_, preprocessor.cardinalities_,
                                      self.d_model, self.nhead, self.layers, self.dropout)
        raise ValueError("kind must be mlp or transformer")

    def _target(self, X, y):
        if self.market_normalize:
            market = monthly_market(X, y)["median"]
            values = y / market.reindex(pd.PeriodIndex(X[DATE], freq="M")).to_numpy(float)
            carry = float(market.iloc[-1])
        else:
            values, carry = y, 1.0
        center, scale = float(np.mean(values)), max(float(np.std(values)), 1e-6)
        return (values - center) / scale, center, scale, carry

    def fit(self, X, y):
        y = validate_target(X, y)
        if self.epochs < 1 or self.batch_size < 1 or self.inner_months < 0 or self.patience < 1:
            raise ValueError("Invalid neural training settings")
        seed_everything(self.random_state, self.threads)
        months = pd.PeriodIndex(X[DATE], freq="M")
        self.origin_ = months.max()
        unique = months.unique().sort_values()
        self.selection_history_ = []
        self.inner_audit_ = {"enabled": False}
        selected = self.epochs
        if self.inner_months and len(unique) > self.inner_months + 1:
            valid_start = unique[-self.inner_months]
            mask = np.asarray(months < valid_start)
            fit_X, valid_X = X.iloc[np.flatnonzero(mask)], X.iloc[np.flatnonzero(~mask)]
            processor = self._preprocessor(fit_X)
            target, center, scale, carry = self._target(fit_X, y[mask])
            network = self._network(processor)
            self.selection_history_, selected = train_rows(
                network, processor.transform(fit_X), target, epochs=self.epochs,
                batch_size=self.batch_size, learning_rate=self.learning_rate,
                weight_decay=self.weight_decay, seed=self.random_state,
                validation=(processor.transform(valid_X), y[~mask], center, scale, carry),
                patience=self.patience, verbose=self.verbose)
            self.inner_audit_ = {"enabled": True, "train_end": str(months[mask].max()),
                                 "valid_start": str(valid_start), "valid_end": str(months.max()),
                                 "train_rows": int(mask.sum()), "valid_rows": int((~mask).sum()),
                                 "selection_metric": "RMSE_SGD", "preprocessing_fit_on_inner_train_only": True}
        # Fresh preprocessing, weights and optimizer for the outer training fold.
        seed_everything(self.random_state, self.threads)
        self.selected_epochs_ = selected
        self.preprocessor_ = self._preprocessor(X)
        target, self.target_center_, self.target_scale_, self.carry_ = self._target(X, y)
        self.network_ = self._network(self.preprocessor_)
        self.training_history_, _ = train_rows(
            self.network_, self.preprocessor_.transform(X), target, epochs=selected,
            batch_size=self.batch_size, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, seed=self.random_state, verbose=self.verbose)
        self.audit_ = {"kind": self.kind, "origin": str(self.origin_), "train_rows": len(X),
                       "selected_epochs": selected, "target": "market_ratio" if self.market_normalize else "rent",
                       "target_center": self.target_center_, "target_scale": self.target_scale_,
                       "numerical": list(self.preprocessor_.numeric), "categorical": list(CATEGORICAL),
                       "cardinalities": self.preprocessor_.cardinalities_, "device": "cpu",
                       "inner": self.inner_audit_, "future_actuals_used": False}
        return self

    def predict_ratio(self, X):
        check_is_fitted(self, "network_")
        if not self.market_normalize:
            raise ValueError("predict_ratio requires market_normalize=True")
        values = predict_network(self.network_, *self.preprocessor_.transform(X), self.batch_size)
        return np.maximum(values.astype(float) * self.target_scale_ + self.target_center_, 1e-6)

    def predict(self, X):
        check_is_fitted(self, "network_")
        if self.market_normalize:
            return self.predict_ratio(X) * self.carry_
        values = predict_network(self.network_, *self.preprocessor_.transform(X), self.batch_size)
        return np.maximum(values.astype(float) * self.target_scale_ + self.target_center_, 1.0)


class MarketHybridRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, encoder="carry", property_epochs=30, property_hidden_size=128,
                 batch_size=1024, learning_rate=0.001, weight_decay=0.01,
                 lookback=6, hidden_size=16, temporal_epochs=160, temporal_lr=0.005,
                 damping=0.8, min_frequency=5, inner_months=3, temporal_inner_months=6,
                 patience=5, random_state=5228, threads=1, verbose=False):
        self.encoder = encoder
        self.property_epochs = property_epochs
        self.property_hidden_size = property_hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.temporal_epochs = temporal_epochs
        self.temporal_lr = temporal_lr
        self.damping = damping
        self.min_frequency = min_frequency
        self.inner_months = inner_months
        self.temporal_inner_months = temporal_inner_months
        self.patience = patience
        self.random_state = random_state
        self.threads = threads
        self.verbose = verbose

    def fit(self, X, y):
        y = validate_target(X, y)
        self.property_ = TabularNNRegressor(
            epochs=self.property_epochs, hidden_size=self.property_hidden_size,
            batch_size=self.batch_size, learning_rate=self.learning_rate,
            weight_decay=self.weight_decay, min_frequency=self.min_frequency,
            inner_months=self.inner_months, patience=self.patience,
            market_normalize=True, random_state=self.random_state,
            threads=self.threads, verbose=self.verbose).fit(X, y)
        table = monthly_market(X, y)
        self.market_ = MarketForecaster(
            self.encoder, self.lookback, self.hidden_size, self.temporal_epochs,
            self.temporal_lr, self.damping, self.temporal_inner_months,
            random_state=self.random_state, threads=self.threads).fit(table)
        self.origin_ = self.market_.origin_
        self.audit_ = {"kind": "market_" + self.encoder, "origin": str(self.origin_),
                       "train_rows": len(X), "property": self.property_.audit_,
                       "market": self.market_.audit_, "device": "cpu", "future_actuals_used": False}
        return self

    def predict(self, X):
        check_is_fitted(self, "property_")
        ratio = self.property_.predict_ratio(X)
        level = self.market_.predict_months(X[DATE])
        return np.maximum(ratio * level, 1.0)
