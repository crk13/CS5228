"""P1 auxiliary features: amenity distances and counts, plus lagged monthly macro indicators.

MRT STATUS is a snapshot without opening dates and also marks CRL, JRL and other future lines as
open. AUX_MRT_* therefore use only lines running before 2025 (CORE_LINES); AUX_RAIL_KM uses every
station marked open, LRT included. None of these need rent.

Macro values for month m come from month m - macro_lag; earlier months take the first value.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.cv import DATE_COL
from src.data_cleaning import load

from .base import FeatureBlock
from .registry import register_feature

EARTH_RADIUS_KM = 6371.0
CORE_LINES = ("ns", "ew", "cg", "ne", "cc", "dt", "te")
CBD_STATION = "raffles place mrt station"
# Developers and REITs with full price history; 9CI.SI starts in 2021-09.
PROPERTY_SYMBOLS = (
    "C09.SI", "U14.SI", "U06.SI", "H15.SI", "A17U.SI", "BUOU.SI", "C38U.SI", "HMN.SI",
    "K71U.SI", "M44U.SI", "ME8U.SI", "N2IU.SI", "T82U.SI",
)
AUX_GROUPS = {
    "mrt": ["AUX_MRT_KM", "AUX_MRT_N1KM", "AUX_MRT_LINES_1KM"],
    "rail": ["AUX_RAIL_KM"],
    "cbd": ["AUX_CBD_KM"],
    "mall": ["AUX_MALL_KM", "AUX_MALL_N1KM"],
    "school": ["AUX_PRI_KM", "AUX_PRI_N1KM"],
    "hdb": ["AUX_HDB_N500M"],
    "macro": ["AUX_COE", "AUX_SGX_INDEX", "AUX_PROPERTY_INDEX"],
}
AUX_COLUMNS = [c for columns in AUX_GROUPS.values() for c in columns]


def _tree(df):
    return BallTree(np.radians(df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)), metric="haversine")


def _price_index(prices):
    """Equal-weight index of monthly mean adjusted closes, each stock rebased to its first month."""
    monthly = prices.groupby(["YEAR_MONTH", "SYMBOL"])["ADJUSTED_CLOSE"].mean().unstack()
    monthly = monthly.loc[:, monthly.notna().all()]
    return (monthly / monthly.iloc[0]).mean(axis=1)


def _monthly_macro():
    coe = load("auxiliary/coe_prices.csv")
    coe = coe[coe["CATEGORY"].isin(["a", "b"])].groupby("YEAR_MONTH")["QUOTA_PREMIUM"].mean()
    stocks = load("auxiliary/stock_prices.csv")
    sgx = stocks[stocks["MARKET"] == "sgx"]
    return pd.DataFrame({
        "AUX_COE": coe,
        "AUX_SGX_INDEX": _price_index(sgx),
        "AUX_PROPERTY_INDEX": _price_index(sgx[sgx["SYMBOL"].isin(PROPERTY_SYMBOLS)]),
    })


class P1AuxFeatures(FeatureBlock):
    def __init__(self, macro_lag: int = 1):
        self.macro_lag = macro_lag

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Spatial features depend only on the building, so compute them once per coordinate.
        unique, inverse = np.unique(df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float), axis=0,
                                    return_inverse=True)
        points = np.radians(unique)
        km = lambda tree: tree.query(points, k=1)[0][:, 0] * EARTH_RADIUS_KM
        count = lambda tree, radius_km: tree.query_radius(points, r=radius_km / EARTH_RADIUS_KM, count_only=True)

        stations = load("auxiliary/mrt_stations.csv")
        stations = stations[stations["STATUS"] == "open"]
        core = stations[stations["LINE"].isin(CORE_LINES)]
        core_stations = core.drop_duplicates("NAME")
        core_tree = _tree(core_stations)
        line_bits = {line: 1 << i for i, line in enumerate(CORE_LINES)}
        masks = core.groupby("NAME")["LINE"].agg(lambda s: sum({line_bits[l] for l in s}))
        masks = masks.reindex(core_stations["NAME"]).to_numpy()
        nearby = core_tree.query_radius(points, r=1.0 / EARTH_RADIUS_KM)
        schools = load("auxiliary/schools.csv")
        primary_tree = _tree(schools[schools["MAINLEVEL_CODE"].str.contains("primary|p1", regex=True)])
        malls_tree = _tree(load("auxiliary/shopping_malls.csv"))

        spatial = pd.DataFrame({
            "AUX_MRT_KM": km(core_tree),
            "AUX_MRT_N1KM": count(core_tree, 1.0),
            "AUX_MRT_LINES_1KM": [bin(int(np.bitwise_or.reduce(masks[i], initial=0))).count("1") for i in nearby],
            "AUX_RAIL_KM": km(_tree(stations.drop_duplicates("NAME"))),
            "AUX_CBD_KM": km(_tree(stations.loc[stations["NAME"] == CBD_STATION].head(1))),
            "AUX_MALL_KM": km(malls_tree),
            "AUX_MALL_N1KM": count(malls_tree, 1.0),
            "AUX_PRI_KM": km(primary_tree),
            "AUX_PRI_N1KM": count(primary_tree, 1.0),
            "AUX_HDB_N500M": count(_tree(load("auxiliary/hdb_blocks.csv")), 0.5),
        })
        out = spatial.iloc[inverse.reshape(-1)].set_axis(df.index)

        macro = _monthly_macro()
        months = pd.period_range(macro.index.min(), macro.index.max(), freq="M").strftime("%Y-%m")
        macro = macro.reindex(months).ffill().shift(self.macro_lag).bfill()
        for column in AUX_GROUPS["macro"]:
            out[column] = df[DATE_COL].astype(str).map(macro[column]).to_numpy(dtype=float)
        return out[AUX_COLUMNS]


register_feature("p1_aux", P1AuxFeatures)
register_feature("p1_aux_lag0", lambda: P1AuxFeatures(macro_lag=0))
