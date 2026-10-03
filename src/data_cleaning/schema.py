"""Column schemas of the cleaned data and helpers to load it with correct dtypes.

CSV loses type information (e.g. POSTAL_CODE "090034" would be read as 90034),
so always load the cleaned files through `load()` / `load_train()` / `load_test()`.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data"
CLEAN_DIR = PROJECT_ROOT / "data_cleaned"

TARGET = "MONTHLY_RENT"
ID_COL = "Id"  # test only; equals the row index used in the Kaggle submission

# train.csv / test.csv
NUM_COLS = [
    "FLOOR_AREA_SQM",
    "LEASE_COMMENCE_DATE",
    "RENT_YEAR",
    "RENT_MONTH",
    "LATITUDE",
    "LONGITUDE",
    "MAX_FLOOR",
    "YEAR_COMPLETED",
]
CAT_COLS = [
    "RENT_APPROVAL_DATE",
    "TOWN",
    "BLOCK",
    "STREET",
    "FLAT_TYPE",
    "FLAT_MODEL",
    "POSTAL_CODE",
    "SUBZONE",
    "PLANNING_AREA",
    "REGION",
]

# Per-file schemas: relative path under data_cleaned/ -> numeric / categorical columns.
SCHEMAS = {
    "train.csv": {"num": NUM_COLS + [TARGET], "cat": CAT_COLS},
    "test.csv": {"num": [ID_COL] + NUM_COLS, "cat": CAT_COLS},
    "auxiliary/hdb_blocks.csv": {
        "num": ["LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"],
        "cat": ["TOWN", "BLOCK", "STREET", "POSTAL_CODE", "SUBZONE", "PLANNING_AREA", "REGION"],
    },
    "auxiliary/mrt_stations.csv": {
        "num": ["LATITUDE", "LONGITUDE"],
        "cat": ["CODE", "LINE", "NAME", "STATUS", "SUBZONE", "PLANNING_AREA", "REGION"],
    },
    "auxiliary/schools.csv": {
        "num": ["LATITUDE", "LONGITUDE"],
        "cat": [
            "NAME", "URL", "STREET", "POSTAL_CODE", "MRT_STATIONS", "BUS_LINES",
            "PLANNING_AREA", "ZONE", "TYPE_CODE", "NATURE_CODE", "SESSION_CODE", "MAINLEVEL_CODE",
        ],
    },
    "auxiliary/shopping_malls.csv": {
        "num": ["LATITUDE", "LONGITUDE"],
        "cat": ["NAME"],
    },
    "auxiliary/coe_prices.csv": {
        "num": [
            "YEAR", "MONTH", "ROUND", "QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED",
        ],
        "cat": ["YEAR_MONTH", "CATEGORY"],
    },
    "auxiliary/stock_prices.csv": {
        "num": ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"],
        "cat": ["NAME", "SYMBOL", "MARKET", "DATE", "YEAR_MONTH"],
    },
}


def load(name: str) -> pd.DataFrame:
    """Load a cleaned file, e.g. load("train.csv") or load("auxiliary/schools.csv").

    Categorical columns are read as strings so codes like BLOCK / POSTAL_CODE keep their exact form.
    """
    schema = SCHEMAS[name]
    return pd.read_csv(CLEAN_DIR / name, dtype={c: "string" for c in schema["cat"]})


def load_train() -> pd.DataFrame:
    return load("train.csv")


def load_test() -> pd.DataFrame:
    return load("test.csv")
