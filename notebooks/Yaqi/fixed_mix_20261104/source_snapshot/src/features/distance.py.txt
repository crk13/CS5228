"""Distances from the supplied facility snapshot; historical dates are unknown."""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.utils.validation import check_is_fitted

from src.data_cleaning import load
from .base import FeatureBlock
from .registry import register_feature

DISTANCE_COLUMNS = ["DIST_MRT_OPEN_KM", "DIST_MALL_KM", "DIST_PRIMARY_SCHOOL_KM"]
EARTH_RADIUS_KM = 6371.0088
# MRT file also contains LRT and the international RTS link, which are excluded.
NON_MRT_LINES = {"bp", "pe", "pw", "se", "sw", "ptc", "stc", "rts"}


@register_feature("distance")
class DistanceFeatures(FeatureBlock):
    def fit(self, train_df):
        mrt = load("auxiliary/mrt_stations.csv")
        schools = load("auxiliary/schools.csv")
        groups = {
            DISTANCE_COLUMNS[0]: mrt.loc[mrt.STATUS.eq("open") & ~mrt.LINE.isin(NON_MRT_LINES)],
            DISTANCE_COLUMNS[1]: load("auxiliary/shopping_malls.csv"),
            DISTANCE_COLUMNS[2]: schools.loc[schools.MAINLEVEL_CODE.eq("primary")],
        }
        self.trees_ = {}
        self.audit_ = {"unit": "km", "distance": "great-circle, not walking distance",
                       "historical_opening_dates_available": False,
                       "facility_status": "provided file snapshot, not historical reconstruction",
                       "non_mrt_lines_excluded": sorted(NON_MRT_LINES), "facilities": {}}
        for column, frame in groups.items():
            coordinates = frame[["LATITUDE", "LONGITUDE"]].dropna().drop_duplicates().to_numpy(float)
            if not len(coordinates) or not np.isfinite(coordinates).all():
                raise ValueError(f"No valid facility coordinates: {column}")
            self.trees_[column] = BallTree(np.radians(coordinates), metric="haversine")
            self.audit_["facilities"][column] = {"records": len(frame), "unique_coordinates": len(coordinates)}
        return self

    def transform(self, df):
        check_is_fitted(self, "trees_")
        coordinates = df[["LATITUDE", "LONGITUDE"]].to_numpy(float)
        valid = np.isfinite(coordinates).all(axis=1)
        result = pd.DataFrame(np.nan, index=df.index, columns=DISTANCE_COLUMNS)
        if valid.any():
            unique, inverse = np.unique(coordinates[valid], axis=0, return_inverse=True)
            query = np.radians(unique)
            for column, tree in self.trees_.items():
                distances = tree.query(query, k=1, return_distance=True)[0].ravel() * EARTH_RADIUS_KM
                result.loc[valid, column] = distances[inverse]
        return result
