"""Expose the building columns already joined by data_cleaning."""

import pandas as pd

from .base import FeatureBlock

BLOCK_COLUMNS = [
    "POSTAL_CODE", "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
    "SUBZONE", "PLANNING_AREA", "REGION",
]


class BlockFeatures(FeatureBlock):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[BLOCK_COLUMNS].copy()
