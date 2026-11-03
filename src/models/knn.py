"""KNN regressor on standardized location, flat attributes and rental month."""

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .base import ModelWrapper

KNN_FEATURES = ["basic", "block"]
# MONTH_INDEX is a neighbour dimension, so later months match the most recent training rows.
KNN_COLUMNS = [
    "LATITUDE", "LONGITUDE", "FLOOR_AREA_SQM", "FLAT_TYPE_ORDINAL", "REMAINING_LEASE", "MONTH_INDEX",
]


def knn_model(n_neighbors: int = 20, weights: str = "distance", target_transform: str = "none") -> ModelWrapper:
    estimator = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1),
    )
    return ModelWrapper(estimator, KNN_COLUMNS, target_transform)
