"""Calendar and flat attributes; leave missing values for each model to handle."""

import pandas as pd

from .base import FeatureBlock

BASIC_COLUMNS = ["MONTH_INDEX", "FLAT_AGE", "REMAINING_LEASE", "FLAT_TYPE_ORDINAL"]
FLAT_TYPE_ORDER = {f"{n}-room": n for n in range(1, 6)} | {"executive": 6}


class BasicFeatures(FeatureBlock):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "MONTH_INDEX": (df["RENT_YEAR"] - 2021) * 12 + df["RENT_MONTH"] - 1,
            "FLAT_AGE": df["RENT_YEAR"] - df["YEAR_COMPLETED"],
            "REMAINING_LEASE": 99 - (df["RENT_YEAR"] - df["LEASE_COMMENCE_DATE"]),
            "FLAT_TYPE_ORDINAL": df["FLAT_TYPE"].map(FLAT_TYPE_ORDER).astype(float),
        }, index=df.index)
