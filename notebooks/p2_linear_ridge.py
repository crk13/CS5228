"""P2 (statistical models): linear / Ridge experiments on the holdout split.

Fit months: up to 2023-11. Validation: 2023-12 .. 2025-03 (same as src/cv.py HOLDOUT).
Run from the repo root:  python notebooks/p2_linear_ridge.py
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import BallTree
from sklearn.preprocessing import OneHotEncoder

DATE = "RENT_APPROVAL_DATE"
TARGET = "MONTHLY_RENT"
FIT_END, VALID_START = "2023-11", "2023-12"


def load():
    df = pd.read_csv("data_cleaned/train.csv", dtype={"BLOCK": str, "POSTAL_CODE": str})
    mrt = pd.read_csv("data_cleaned/auxiliary/mrt_stations.csv")
    mrt = mrt[mrt.STATUS == "open"].drop_duplicates(["LATITUDE", "LONGITUDE"])
    tree = BallTree(np.radians(mrt[["LATITUDE", "LONGITUDE"]].values), metric="haversine")
    dist, _ = tree.query(np.radians(df[["LATITUDE", "LONGITUDE"]].values), k=1)
    df["MRT_KM"] = dist[:, 0] * 6371  # straight-line distance to nearest open station
    df["FLAT_AGE"] = df.RENT_YEAR - df.LEASE_COMMENCE_DATE
    df["BLOCK_KEY"] = df.BLOCK + " " + df.STREET
    return df


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def design(train, test, cats, nums):
    enc = OneHotEncoder(handle_unknown="ignore")
    a = sparse.hstack([enc.fit_transform(train[cats]), sparse.csr_matrix(train[nums].values)]).tocsr()
    b = sparse.hstack([enc.transform(test[cats]), sparse.csr_matrix(test[nums].values)]).tocsr()
    return a, b


def run(train, test, cats, nums, model, target=None, scale=1.0):
    a, b = design(train, test, cats, nums)
    y = train[TARGET] if target is None else target
    return rmse(test[TARGET], model.fit(a, y).predict(b) * scale)


def pick_alpha(train, cats, nums, alphas=(0.3, 1, 3, 10, 30, 100)):
    """Choose alpha inside the training months only (last 2 months as inner check)."""
    months = sorted(train[DATE].unique())
    inner_fit, inner_val = train[train[DATE] < months[-2]], train[train[DATE] >= months[-2]]
    scores = {a: run(inner_fit, inner_val, cats, nums, Ridge(alpha=a)) for a in alphas}
    return min(scores, key=scores.get), scores


def main():
    df = load()
    fit, valid = df[df[DATE] <= FIT_END], df[df[DATE] >= VALID_START]
    recent = fit[fit[DATE] >= sorted(fit[DATE].unique())[-6]]  # last 6 fit months
    rows = []

    # 1) baseline: median rent by town x flat type over the last 6 fit months
    med = recent.groupby(["TOWN", "FLAT_TYPE"])[TARGET].median()
    pred = valid.join(med.rename("p"), on=["TOWN", "FLAT_TYPE"])["p"].fillna(recent[TARGET].median())
    rows.append(("baseline: median by town x flat type (last 6 months)", rmse(valid[TARGET], pred)))

    # 2) plain linear regression, adding features step by step (last 6 months)
    lin = LinearRegression()
    rows.append(("linear: floor area only, ALL fit months", run(fit, valid, [], ["FLOOR_AREA_SQM"], lin)))
    rows.append(("linear: floor area only, last 6 months", run(recent, valid, [], ["FLOOR_AREA_SQM"], lin)))
    rows.append(("linear: town + flat type + area", run(recent, valid, ["TOWN", "FLAT_TYPE"], ["FLOOR_AREA_SQM"], lin)))
    base_nums = ["FLOOR_AREA_SQM", "FLAT_AGE"]
    rows.append(("linear: + flat age + flat model", run(recent, valid, ["TOWN", "FLAT_TYPE", "FLAT_MODEL"], base_nums, lin)))
    rows.append(("linear: town -> subzone", run(recent, valid, ["SUBZONE", "FLAT_TYPE", "FLAT_MODEL"], base_nums, lin)))
    rows.append(("linear: subzone + block (overfits)", run(recent, valid, ["SUBZONE", "BLOCK_KEY", "FLAT_TYPE", "FLAT_MODEL"], base_nums, lin)))

    # 3) Ridge with alpha chosen inside the training months
    cats = ["SUBZONE", "BLOCK_KEY", "FLAT_TYPE", "FLAT_MODEL"]
    nums = base_nums + ["MRT_KM"]
    alpha, scores = pick_alpha(recent, cats, nums)
    print("alpha check inside training months:", {k: round(v) for k, v in scores.items()}, "-> alpha", alpha)
    rows.append((f"ridge (alpha={alpha}): subzone + block + MRT distance, last 6 months", run(recent, valid, cats, nums, Ridge(alpha=alpha))))
    rows.append((f"ridge (alpha={alpha}): same, ALL fit months, raw rent", run(fit, valid, cats, nums, Ridge(alpha=alpha))))

    # 4) time trend: divide by the monthly market level, learn the ratio, scale back
    level = fit.groupby(DATE)[TARGET].median()  # monthly median rent = market level
    ratio = fit[TARGET] / fit[DATE].map(level)
    carry = level.iloc[-1]  # assume the level stays at the last fit month (rents flat since 2023)
    rows.append((f"ridge (alpha={alpha}): ALL fit months, rent / market level, carry last level {carry:.0f}",
                 run(fit, valid, cats, nums, Ridge(alpha=alpha), target=ratio, scale=carry)))

    print(f"\nvalidation rows: {len(valid)}  ({VALID_START} .. {valid[DATE].max()})\n")
    for name, score in rows:
        print(f"{score:7.0f}  {name}")

    # 5) both shared splits, and three ways to extend the market level into the validation months
    print("\nmarket-ratio Ridge on both splits (alpha chosen inside training months):")
    for split, fit_end, valid_start in [("A", "2023-11", "2023-12"), ("B", "2024-03", "2024-04")]:
        fit_s, valid_s = df[df[DATE] <= fit_end], df[df[DATE] >= valid_start]
        recent_s = fit_s[fit_s[DATE] >= sorted(fit_s[DATE].unique())[-6]]
        alpha_s, _ = pick_alpha(recent_s, cats, nums)
        level_s = fit_s.groupby(DATE)[TARGET].median()
        a, b = design(fit_s, valid_s, cats, nums)
        ratio_pred = Ridge(alpha=alpha_s).fit(a, fit_s[TARGET] / fit_s[DATE].map(level_s)).predict(b)
        for how in ("carry", "trend_last_12m", "trend_all"):
            scale = valid_s[DATE].map(extend_level(level_s, how, valid_s[DATE].unique())).values
            print(f"  split {split}  {how:15s} {rmse(valid_s[TARGET], ratio_pred * scale):6.0f}")


def extend_level(level, how, future_months):
    """Market level for unseen months: carry the last value, or extend a straight line."""
    if how == "carry":
        return {m: level.iloc[-1] for m in future_months}
    month_no = lambda m: int(m[:4]) * 12 + int(m[5:7])
    k = len(level) if how == "trend_all" else 12
    x = np.array([month_no(m) for m in level.index[-k:]])
    slope, intercept = np.polyfit(x, level.values[-k:], 1)
    return {m: intercept + slope * month_no(m) for m in future_months}


if __name__ == "__main__":
    main()
