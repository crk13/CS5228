"""Sanity checks on the cleaned output (reads data_cleaned/ back through `load`).

Run from the project root after cleaning:
    python -m src.data_cleaning.checks
"""
import pandas as pd

from .schema import CAT_COLS, ID_COL, NUM_COLS, RAW_DIR, SCHEMAS, TARGET, load

# Loose bounding box of Singapore
LAT_RANGE = (1.15, 1.48)
LON_RANGE = (103.6, 104.1)

# Columns allowed to contain missing values: file -> columns
ALLOWED_NA = {
    "train.csv": {"POSTAL_CODE", "MAX_FLOOR", "YEAR_COMPLETED"},
    "test.csv": {"POSTAL_CODE", "MAX_FLOOR", "YEAR_COMPLETED"},
    "auxiliary/hdb_blocks.csv": {"POSTAL_CODE", "MAX_FLOOR", "YEAR_COMPLETED"},
}

FLAT_TYPES = {"1-room", "2-room", "3-room", "4-room", "5-room", "executive"}
# Free-text / URL / mixed-case-by-nature columns exempt from the lowercase check
CASE_EXEMPT = {"URL", "NAME", "SYMBOL", "MARKET"}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def check_common(name, df):
    schema = SCHEMAS[name]
    check(df.columns.is_unique and set(df.columns) == set(schema["num"]) | set(schema["cat"]),
          f"{name}: columns match schema")
    non_numeric = [c for c in schema["num"] if not pd.api.types.is_numeric_dtype(df[c])]
    check(not non_numeric, f"{name}: all NUM columns numeric {non_numeric or ''}")
    na_cols = set(df.columns[df.isna().any()])
    check(na_cols <= ALLOWED_NA.get(name, set()),
          f"{name}: missing values only in {sorted(ALLOWED_NA.get(name, set())) or 'none'}")
    for c in schema["cat"]:
        if c in CASE_EXEMPT:
            continue
        s = df[c].dropna()
        bad = s[(s != s.str.strip()) | s.str.contains("[A-Z]") | s.str.contains("  ")]
        check(bad.empty, f"{name}: {c} is lowercase / trimmed")
    if "LATITUDE" in df:
        check(df["LATITUDE"].between(*LAT_RANGE).all() and df["LONGITUDE"].between(*LON_RANGE).all(),
              f"{name}: coordinates inside Singapore")
    if "POSTAL_CODE" in df:
        p = df["POSTAL_CODE"].dropna()
        check(p.str.fullmatch(r"\d{6}").all(), f"{name}: POSTAL_CODE is 6 digits")


def check_main():
    raw_tr = pd.read_csv(RAW_DIR / "train.csv")
    raw_te = pd.read_csv(RAW_DIR / "test.csv")
    tr, te = load("train.csv"), load("test.csv")

    check(len(tr) == len(raw_tr) == 150_000 and len(te) == len(raw_te) == 50_000, "row counts preserved")
    check((te[ID_COL] == range(len(te))).all(), "test Id == row index 0..n-1")
    check(set(tr.columns) - {TARGET} == set(te.columns) - {ID_COL}, "train/test share feature columns")
    check(set(NUM_COLS) | set(CAT_COLS) == set(te.columns) - {ID_COL}, "NUM_COLS + CAT_COLS cover all features")

    for raw, df, n in [(raw_tr, tr, "train"), (raw_te, te, "test")]:
        for c in ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]:
            check((raw[c].values == df[c].values).all(), f"{n}: {c} unchanged vs raw")
        check((raw["STREET"].str.lower().values == df["STREET"].values).all(), f"{n}: STREET only lowercased")
        check(set(df["FLAT_TYPE"]) <= FLAT_TYPES, f"{n}: FLAT_TYPE normalised")
        ym = df["RENT_YEAR"].astype(str) + "-" + df["RENT_MONTH"].astype(str).str.zfill(2)
        check((ym == df["RENT_APPROVAL_DATE"]).all(), f"{n}: RENT_YEAR/RENT_MONTH match RENT_APPROVAL_DATE")
    check((raw_tr[TARGET].values == tr[TARGET].values).all(), "train: MONTHLY_RENT unchanged vs raw")

    for c in ["TOWN", "FLAT_TYPE", "FLAT_MODEL", "SUBZONE", "PLANNING_AREA", "REGION"]:
        unseen = set(te[c]) - set(tr[c])
        check(not unseen, f"test {c} values all seen in train {sorted(unseen) or ''}")
    check(tr["RENT_APPROVAL_DATE"].max() <= te["RENT_APPROVAL_DATE"].min(), "test is not earlier than train")


def check_aux():
    coe = load("auxiliary/coe_prices.csv")
    check(coe["MONTH"].between(1, 12).all(), "coe: MONTH in 1..12")
    check(not coe.duplicated(["YEAR_MONTH", "ROUND", "CATEGORY"]).any(), "coe: (YEAR_MONTH, ROUND, CATEGORY) unique")
    stocks = load("auxiliary/stock_prices.csv")
    check(not stocks.duplicated(["SYMBOL", "DATE"]).any(), "stocks: (SYMBOL, DATE) unique")
    mrt = load("auxiliary/mrt_stations.csv")
    check(not mrt["CODE"].duplicated().any(), "mrt: CODE unique")
    hdb = load("auxiliary/hdb_blocks.csv")
    check(not hdb.duplicated(["BLOCK", "STREET"]).any(), "hdb: (BLOCK, STREET) unique")


def main():
    for name in SCHEMAS:
        print(name)
        check_common(name, load(name))
    print("train/test")
    check_main()
    print("auxiliary")
    check_aux()
    print("All checks passed.")


if __name__ == "__main__":
    main()
