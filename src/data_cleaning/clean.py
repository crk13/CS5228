"""Basic, model-agnostic cleaning of the raw data.

Only format fixes, table joins and consistency checks happen here. Anything model-specific
(outlier removal, encodings, scaling, feature engineering) is left to each model owner.

Run from the project root:
    python -m src.data_cleaning.clean
"""
import pandas as pd

from .schema import CLEAN_DIR, ID_COL, RAW_DIR, TARGET

AUX_RAW = RAW_DIR / "auxiliary"
AUX_CLEAN = CLEAN_DIR / "auxiliary"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

HDB_COLS = ["POSTAL_CODE", "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
            "SUBZONE", "PLANNING_AREA", "REGION"]
MAIN_ORDER = ["RENT_APPROVAL_DATE", "RENT_YEAR", "RENT_MONTH", "TOWN", "BLOCK", "STREET",
              "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"] + HDB_COLS


def norm_str(s: pd.Series) -> pd.Series:
    """Lowercase, strip and collapse inner whitespace."""
    return s.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def read_raw(path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kw)


def clean_hdb_blocks() -> pd.DataFrame:
    df = read_raw(AUX_RAW / "sg-hdb-block.csv")
    for c in ["TOWN", "BLOCK", "STREET", "SUBZONE", "PLANNING_AREA", "REGION"]:
        df[c] = norm_str(df[c])
    # "NIL" marks a missing postal code
    df["POSTAL_CODE"] = df["POSTAL_CODE"].str.strip().replace("NIL", pd.NA).astype("string")
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    # -1 marks missing MAX_FLOOR / YEAR_COMPLETED
    for c in ["MAX_FLOOR", "YEAR_COMPLETED"]:
        v = df[c].astype(int)
        df[c] = v.where(v > 0).astype("Int64")
    assert not df.duplicated(["BLOCK", "STREET"]).any(), "hdb blocks: (BLOCK, STREET) is not unique"
    return df


def clean_main(df: pd.DataFrame, hdb: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    n = len(df)
    for c in ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]:
        df[c] = norm_str(df[c])
    # "4 room" and "4-room" are the same flat type
    df["FLAT_TYPE"] = df["FLAT_TYPE"].str.replace(" ", "-", regex=False)

    df["RENT_APPROVAL_DATE"] = df["RENT_APPROVAL_DATE"].str.strip()
    date = pd.to_datetime(df["RENT_APPROVAL_DATE"], format="%Y-%m")
    df["RENT_YEAR"] = date.dt.year
    df["RENT_MONTH"] = date.dt.month

    df["FLOOR_AREA_SQM"] = df["FLOOR_AREA_SQM"].astype(float)
    df["LEASE_COMMENCE_DATE"] = df["LEASE_COMMENCE_DATE"].astype(int)

    # Constant columns carry no information (all "yes" / 0 in both train and test)
    assert set(df["FURNISHED"].str.strip().str.lower()) == {"yes"}, "FURNISHED is no longer constant"
    assert set(df["FEE"].astype(float)) == {0.0}, "FEE is no longer constant"
    df = df.drop(columns=["FURNISHED", "FEE"])

    df = df.merge(hdb.rename(columns={"TOWN": "HDB_TOWN"}), on=["BLOCK", "STREET"],
                  how="left", validate="many_to_one")
    assert len(df) == n
    assert df["LATITUDE"].notna().all(), "some flats could not be joined to sg-hdb-block"
    assert (df["TOWN"] == df["HDB_TOWN"]).all(), "TOWN disagrees with sg-hdb-block"
    df = df.drop(columns="HDB_TOWN")

    if is_train:
        df[TARGET] = df[TARGET].astype(int)
        return df[MAIN_ORDER + [TARGET]]
    df.insert(0, ID_COL, range(n))
    return df[[ID_COL] + MAIN_ORDER]


def clean_mrt() -> pd.DataFrame:
    df = read_raw(AUX_RAW / "sg-mrt-stations.csv")
    for c in ["CODE", "NAME", "STATUS", "SUBZONE", "PLANNING_AREA", "REGION"]:
        df[c] = norm_str(df[c])
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    # One row per (station, line); interchange stations appear once per line, e.g. ns16 / cr11
    df["LINE"] = df["CODE"].str.extract(r"^([a-z]+)", expand=False)
    assert df["LINE"].notna().all()
    return df[["CODE", "LINE", "NAME", "STATUS", "LATITUDE", "LONGITUDE",
               "SUBZONE", "PLANNING_AREA", "REGION"]]


def clean_schools() -> pd.DataFrame:
    df = read_raw(AUX_RAW / "sg-schools.csv")
    # Fix column-name typo; REGION here is the MOE zone (incl. "south"), not the URA region used elsewhere
    df = df.rename(columns={"NATRUE_CODE": "NATURE_CODE", "REGION": "ZONE"})
    for c in ["NAME", "STREET", "MRT_STATIONS", "BUS_LINES", "PLANNING_AREA", "ZONE",
              "TYPE_CODE", "NATURE_CODE", "SESSION_CODE", "MAINLEVEL_CODE"]:
        df[c] = norm_str(df[c])
    df["URL"] = df["URL"].str.strip()
    # Leading zeros were lost for some postal codes (e.g. 88256 -> 088256)
    df["POSTAL_CODE"] = df["POSTAL_CODE"].str.strip().str.zfill(6)
    # Align planning-area spelling with sg-hdb-block / sg-mrt-stations
    df["PLANNING_AREA"] = df["PLANNING_AREA"].replace({"seng kang": "sengkang"})
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    return df


def clean_malls() -> pd.DataFrame:
    df = read_raw(AUX_RAW / "sg-shopping-malls.csv")
    df["NAME"] = norm_str(df["NAME"])
    df["LATITUDE"] = df["LATITUDE"].astype(float)
    df["LONGITUDE"] = df["LONGITUDE"].astype(float)
    return df


def clean_coe() -> pd.DataFrame:
    df = read_raw(AUX_RAW / "sg-coe-prices.csv")
    df["CATEGORY"] = norm_str(df["CATEGORY"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = norm_str(df["MONTH"]).map(MONTHS)
    assert df["MONTH"].notna().all()
    df["MONTH"] = df["MONTH"].astype(int)
    df["ROUND"] = df["ROUND"].astype(int)
    # "$40,609" / "1,280" -> integers
    for c in ["QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]:
        df[c] = df[c].str.replace(r"[$,\s]", "", regex=True).astype(int)
    df["YEAR_MONTH"] = df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2)
    df = df.sort_values(["YEAR", "MONTH", "ROUND", "CATEGORY"]).reset_index(drop=True)
    return df[["YEAR_MONTH", "YEAR", "MONTH", "ROUND", "CATEGORY", "QUOTA_PREMIUM",
               "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]]


def stock_market(symbol: str) -> str:
    for suffix, market in [(".SI", "sgx"), (".HK", "hkex"), (".AS", "euronext")]:
        if symbol.endswith(suffix):
            return market
    return "us"


def clean_stocks() -> pd.DataFrame:
    df = pd.read_csv(AUX_RAW / "sg-stock-prices.csv", dtype={"NAME": str, "SYMBOL": str, "DATE": str})
    price_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]
    # Rows where every price field is empty (non-trading days) carry nothing
    df = df.dropna(subset=price_cols, how="all")
    assert df[price_cols].notna().all().all(), "stock rows with partially missing prices"
    df["NAME"] = df["NAME"].str.strip()
    df["SYMBOL"] = df["SYMBOL"].str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d")
    df["YEAR_MONTH"] = df["DATE"].str[:7]
    # Prices are quoted in the listing market's currency (SGD / HKD / USD / EUR)
    df["MARKET"] = df["SYMBOL"].map(stock_market)
    df = df.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)
    return df[["NAME", "SYMBOL", "MARKET", "DATE", "YEAR_MONTH"] + price_cols]


def main():
    AUX_CLEAN.mkdir(parents=True, exist_ok=True)

    hdb = clean_hdb_blocks()
    train = clean_main(read_raw(RAW_DIR / "train.csv"), hdb, is_train=True)
    test = clean_main(read_raw(RAW_DIR / "test.csv"), hdb, is_train=False)

    outputs = {
        CLEAN_DIR / "train.csv": train,
        CLEAN_DIR / "test.csv": test,
        AUX_CLEAN / "hdb_blocks.csv": hdb,
        AUX_CLEAN / "mrt_stations.csv": clean_mrt(),
        AUX_CLEAN / "schools.csv": clean_schools(),
        AUX_CLEAN / "shopping_malls.csv": clean_malls(),
        AUX_CLEAN / "coe_prices.csv": clean_coe(),
        AUX_CLEAN / "stock_prices.csv": clean_stocks(),
    }
    for path, df in outputs.items():
        df.to_csv(path, index=False)
        print(f"{path.relative_to(CLEAN_DIR.parent)}: {df.shape}")


if __name__ == "__main__":
    main()
