"""Equal-weight, same-family neural ensembles; never blend teammate predictions."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .estimators import TabularNNRegressor, validate_target
from .market import DATE, MarketForecaster, monthly_market


class AveragedPropertyRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, members, random_state=5228):
        self.members = members
        self.random_state = random_state

    def fit(self, X, y):
        y = validate_target(X, y)
        if not self.members:
            raise ValueError("At least one neural member is required")
        if len({p["kind"] for p in self.members}) != 1:
            raise ValueError("Only same-family ensembles are supported")
        if not all(p.get("market_normalize") for p in self.members):
            raise ValueError("Ensemble members must predict market-normalized ratios")
        self.models_ = [TabularNNRegressor(**dict(p, random_state=self.random_state)).fit(X, y)
                        for p in self.members]
        self.audit_ = {"kind": "averaged_property", "aggregation": "equal_weight_mean",
                       "n_members": len(self.models_), "train_rows": len(X),
                       "members": [m.audit_ for m in self.models_], "future_actuals_used": False}
        self.selection_history_ = [m.selection_history_ for m in self.models_]
        self.training_history_ = [m.training_history_ for m in self.models_]
        return self

    def predict_ratio(self, X):
        check_is_fitted(self, "models_")
        return np.mean([m.predict_ratio(X) for m in self.models_], axis=0)


class NeuralFamilyEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, members, encoder="carry", temporal_parameters=None, random_state=5228):
        self.members = members
        self.encoder = encoder
        self.temporal_parameters = temporal_parameters
        self.random_state = random_state

    def fit(self, X, y):
        y = validate_target(X, y)
        self.property_ = AveragedPropertyRegressor(self.members, self.random_state).fit(X, y)
        params = dict(self.temporal_parameters or {}, kind=self.encoder, random_state=self.random_state)
        self.market_ = MarketForecaster(**params).fit(monthly_market(X, y))
        self.audit_ = {"kind": "neural_family_ensemble", "encoder": self.encoder,
                       "origin": str(self.market_.origin_), "train_rows": len(X),
                       "property": self.property_.audit_, "market": self.market_.audit_,
                       "future_actuals_used": False}
        return self

    def predict(self, X):
        check_is_fitted(self, "property_")
        return np.maximum(self.property_.predict_ratio(X)*self.market_.predict_months(X[DATE]), 1.)
