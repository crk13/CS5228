"""Straight-line distance to the nearest open MRT station."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.data_cleaning import load

from .base import FeatureBlock
from .registry import register_feature

EARTH_RADIUS_KM = 6371


@register_feature("mrt_distance")
class MrtDistance(FeatureBlock):
    """MRT_KM: haversine distance (km) to the nearest open station; needs no rent."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        mrt = load("auxiliary/mrt_stations.csv")
        mrt = mrt[mrt["STATUS"] == "open"].drop_duplicates(["LATITUDE", "LONGITUDE"])
        tree = BallTree(np.radians(mrt[["LATITUDE", "LONGITUDE"]].to_numpy()), metric="haversine")
        dist, _ = tree.query(np.radians(df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)), k=1)
        return pd.DataFrame({"MRT_KM": dist[:, 0] * EARTH_RADIUS_KM}, index=df.index)
