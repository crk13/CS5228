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
        if split == "A":
            error_analysis(valid_s.assign(pred=ratio_pred * level_s.iloc[-1]))

    plot_market_level(df)


def error_analysis(v):
    """Where does the model go wrong? err > 0 means the guess was too high."""
    v = v.assign(err=v["pred"] - v[TARGET])
    groupings = {
        "actual rent": pd.cut(v[TARGET], [0, 2000, 2500, 3000, 3500, 4000, 1e5]),
        "predicted rent": pd.cut(v["pred"], [0, 2500, 3000, 3500, 4000, 1e5]),
        "region": v["REGION"],
        "flat type": v["FLAT_TYPE"],
        "flat age": pd.cut(v["FLAT_AGE"], [-1, 10, 20, 30, 40, 100]),
    }
    for name, key in groupings.items():
        g = v.groupby(key, observed=True)["err"]
        table = pd.DataFrame({"rows": g.size(), "rmse": g.apply(lambda e: np.sqrt((e ** 2).mean())), "mean_err": g.mean()})
        print(f"\n  error by {name} (split A):\n" + table.round(0).astype(int).to_string())
    # Note: grouping by ACTUAL rent always shows cheap rows over-predicted and expensive rows
    # under-predicted (selection on the outcome); grouping by PREDICTED rent checks real bias.


def plot_market_level(df, out_dir="notebooks/p2_figs"):
    """Two figures: straight-line vs curve extensions of the market level into split A's validation months."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    os.makedirs(out_dir, exist_ok=True)
    month_no = lambda m: (int(m[:4]) - 2021) * 12 + int(m[5:7]) - 1
    actual = df.groupby(DATE)[TARGET].median()
    fit_level = actual[actual.index <= FIT_END]
    x_fit = np.array([month_no(m) for m in fit_level.index])
    y_fit = fit_level.values.astype(float)
    future = np.arange(month_no(VALID_START), month_no("2026-07") + 1)
    s_curve = lambda x, top, k, mid, base: base + top / (1 + np.exp(-k * (x - mid)))
    s_params, _ = curve_fit(s_curve, x_fit, y_fit, p0=[1200, 0.2, 20, 2000], maxfev=20000)
    lines = {
        "carry last month": np.full(len(future), y_fit[-1]),
        "straight line, last 12 fit months": np.polyval(np.polyfit(x_fit[-12:], y_fit[-12:], 1), future),
        "straight line, all fit months": np.polyval(np.polyfit(x_fit, y_fit, 1), future),
        "parabola, all fit months": np.polyval(np.polyfit(x_fit, y_fit, 2), future),
        "S-curve, all fit months": s_curve(future, *s_params),
    }
    groups = {"market_level_straight_lines.png": list(lines)[:3],
              "market_level_curves.png": ["carry last month", "parabola, all fit months", "S-curve, all fit months"]}
    ticks = [month_no(f"{y}-01") for y in range(2021, 2027)]
    for filename, names in groups.items():
        plt.figure(figsize=(10, 4.8))
        plt.plot([month_no(m) for m in actual.index], actual.values, "k-", lw=2, label="actual monthly median rent")
        plt.axvspan(month_no(VALID_START), month_no(actual.index[-1]), color="orange", alpha=0.12, label="validation (split A)")
        plt.axvspan(month_no(actual.index[-1]), month_no("2026-07"), color="grey", alpha=0.10, label="test months (rent hidden)")
        for name in names:
            plt.plot(future, lines[name], "--", lw=2, label=name)
        plt.xticks(ticks, [f"{y}-01" for y in range(2021, 2027)])
        plt.ylim(1800, 4600)
        plt.ylabel("SGD per month")
        plt.title("Extending the market level beyond the last fit month (split A)")
        plt.legend(loc="upper left", fontsize=8.5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename), dpi=130)
        plt.close()
    print(f"\nfigures saved to {out_dir}/")


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
