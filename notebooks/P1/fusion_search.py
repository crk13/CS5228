"""P1 fusion search: linear blends, nonlinear stacking on top, and three test-level strategies.

Run from the repo root:  python notebooks/P1/fusion_search.py <stage>
Each stage appends one row per config to notebooks/P1/fusion/<stage>.csv and skips configs already
there, so an interrupted stage resumes. `finalize` writes test submissions for the best configs.

Design
- Pool: base runs from fusion_inputs (default POOL7: one run per model type).
- Layer 1 (linear): weights fitted on month-demeaned predictions, so the shared monthly level (and
  the 2024-07 jump inside validation) does not drive them. Its level is the weighted base level.
- Layer 2 (optional, nonlinear): a meta-model learns layer-1 residuals from base-prediction spreads
  and flat context. Residuals are demeaned per month when fitting and the meta output is demeaned
  per month when predicting, so layer 2 only reorders flats within a month.
- Evaluation, never fitting and scoring on the same rows:
    A: fit on holdout-fold rows 2023-12..2024-06, score holdout rows 2024-07..2025-03
    B: fit on holdout-fold rows 2023-12..2024-03, score rolling_2024_04 rows 2024-04..2025-03
  Each reports raw RMSE and RMSE after removing each month's mean error (cross-sectional skill).
  Validation straddles the 2024-07 jump, so it cannot rank level strategies.
- Final fit uses all holdout-fold rows (16-month horizon, like the 17-month test).
- Test level: L1 keeps the layer-1 level; L2 rescales each test month to the mean of the flat-level
  models (trees + KNN, i.e. no market change after 2025-03); L3 is L2 plus the post-jump trend of
  the hedonic rent index (2024-07..2025-03) compounded by months since 2025-03.
"""

import argparse
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fusion_inputs import SOURCES, list_runs, load_inputs
from src.config import SEED
from src.data_cleaning import load_test, load_train
from src.experiment import write_submission
from src.features import FeaturePipeline

OUT = HERE / "fusion"
SHORT = {"knn_basic": "knn1", "knn_v2 ridge_residual + add_age_floor": "knn2", "knn_v3 aux": "knn3",
         "ridge_market_ratio_carry": "ridge", "yaqi_final_xgboost_base": "xgb",
         "yaqi_final_lightgbm_location_coe": "lgbm", "yaqi_final_catboost_location_coe": "cat",
         "yuxin_market_lstm": "lstm", "yuxin_market_rnn": "rnn", "yuxin_tabular_transformer_carry": "tfm"}
POOLS = {"pool7": ["ridge", "xgb", "lgbm", "cat", "rnn", "tfm", "knn3"],
         "all10": list(SHORT.values()),
         "no_knn": ["ridge", "xgb", "lgbm", "cat", "rnn", "tfm"],
         "top4": ["rnn", "tfm", "cat", "ridge"]}
FLAT_FAMILY = ["xgb", "lgbm", "cat", "knn3"]
CONTEXT_NUM = ["FLAT_TYPE_ORDINAL", "FLOOR_AREA_SQM", "REMAINING_LEASE", "FLAT_AGE", "MAX_FLOOR",
               "LATITUDE", "LONGITUDE", "HORIZON"]
CONTEXT_CAT = ["TOWN", "REGION", "FLAT_MODEL"]
LOCATION = ["LATITUDE", "LONGITUDE", "TOWN"]
CTX_SETS = {"none": [], "full": CONTEXT_NUM + CONTEXT_CAT,
            "no_location": [c for c in CONTEXT_NUM + CONTEXT_CAT if c not in LOCATION]}
TRAIN_END = {"holdout": "2023-11", "rolling_2024_04": "2024-03", "test": "2025-03"}


def month_index(months):
    p = pd.PeriodIndex(months, freq="M")
    return (p.year - 2021) * 12 + p.month - 1


# ---------------------------------------------------------------- data
def _build_data():
    runs = list_runs()
    valid, base, test = load_inputs(runs)
    names = [SHORT[n] for n in runs["name"]]
    valid.columns, test.columns = names, names
    train, test_df = load_train(), load_test()
    pipe = FeaturePipeline(["basic", "block"])
    ctx_tr, ctx_te = pipe.fit_transform(train), pipe.transform(test_df)
    ctx_va = ctx_tr.iloc[valid.index.get_level_values("row_index")].reset_index(drop=True)
    for frame, months, ends in ((ctx_va, base["RENT_APPROVAL_DATE"].astype(str).to_numpy(),
                                 [TRAIN_END[f] for f in valid.index.get_level_values("fold")]),
                                (ctx_te, test_df["RENT_APPROVAL_DATE"].astype(str).to_numpy(), ["2025-03"] * len(test_df))):
        frame["HORIZON"] = np.asarray(month_index(months)) - np.asarray(month_index(ends))
    for c in CONTEXT_CAT:  # shared integer codes for train and test
        cats = pd.Index(sorted(set(ctx_va[c].astype(str)) | set(ctx_te[c].astype(str))))
        ctx_va[c], ctx_te[c] = cats.get_indexer(ctx_va[c].astype(str)), cats.get_indexer(ctx_te[c].astype(str))
    keep = CONTEXT_NUM + CONTEXT_CAT
    return {"valid": valid.reset_index(), "y": base["MONTHLY_RENT"].to_numpy(float),
            "month": base["RENT_APPROVAL_DATE"].astype(str).to_numpy(), "test": test,
            "test_month": test_df["RENT_APPROVAL_DATE"].astype(str).to_numpy(),
            "ctx_valid": ctx_va[keep], "ctx_test": ctx_te[keep].reset_index(drop=True),
            "growth": _post_jump_growth(train)}


def _post_jump_growth(train):
    """Monthly log growth of a hedonic rent index over 2024-07..2025-03 (month effects of log rent)."""
    cats = pd.DataFrame({"cell": train.TOWN.astype(str) + "|" + train.FLAT_TYPE.astype(str),
                         "model": train.FLAT_MODEL.astype(str),
                         "block": train.BLOCK.astype(str) + "|" + train.STREET.astype(str),
                         "month": train.RENT_APPROVAL_DATE.astype(str)})
    enc = OneHotEncoder(handle_unknown="ignore").fit(cats)
    num = np.column_stack([np.log(train.FLOOR_AREA_SQM), train.RENT_YEAR - train.LEASE_COMMENCE_DATE])
    X = sparse.hstack([enc.transform(cats), sparse.csr_matrix((num - num.mean(0)) / num.std(0))]).tocsr()
    coef = Ridge(alpha=1.0).fit(X, np.log(train.MONTHLY_RENT)).coef_
    start = sum(len(c) for c in enc.categories_[:3])
    index = pd.Series(coef[start:start + len(enc.categories_[3])], index=enc.categories_[3])
    post = index.loc["2024-07":"2025-03"]
    return float(np.polyfit(np.arange(len(post)), post.to_numpy(), 1)[0])


def data():
    key = "_".join(c[:7] for _, c, _ in SOURCES.values())
    cache = Path(tempfile.gettempdir()) / f"p1_fusion_data_{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    d = _build_data()
    cache.write_bytes(pickle.dumps(d))
    return d


# ---------------------------------------------------------------- helpers
def demean(values, groups):
    values = np.asarray(values, dtype=float)
    return values - pd.DataFrame(values).groupby(np.asarray(groups)).transform("mean").to_numpy().reshape(values.shape)


def schemes(d):
    fold, month = d["valid"]["fold"].to_numpy(), d["month"]
    h, r = fold == "holdout", fold == "rolling_2024_04"
    return {"A": (h & (month < "2024-07"), h & (month >= "2024-07")),
            "B": (h & (month < "2024-04"), r)}


def scores(y, pred, month):
    err = pred - y
    return float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean(demean(err, month) ** 2)))


# ---------------------------------------------------------------- layer 1
def fit_linear(P, y, month, cfg, segments=None):
    """Return a predict(P, segments) closure; weights are fitted on month-demeaned values."""
    kind = cfg.get("linear", "nnls")
    k = P.shape[1]
    if kind in ("equal", "single"):
        w = np.full(k, 1.0 / k) if kind == "equal" else np.eye(k)[POOLS[cfg["pool"]].index(cfg["name"])]
        return lambda Q, s=None: Q @ w, {"weights": w}

    def solve(Pf, yf, mf):
        X, t = (demean(Pf, mf), demean(yf, mf)) if cfg.get("demean", True) else (Pf, yf)
        if kind == "nnls":
            w, _ = nnls(X, t)
        else:  # ridge on demeaned values, weights may be negative
            w = Ridge(alpha=cfg.get("alpha", 1.0), fit_intercept=False).fit(X, t).coef_
        return w / w.sum()

    w = solve(P, y, month)
    if cfg.get("segment") is None:
        return lambda Q, s=None: Q @ w, {"weights": w}
    seg_w, shrink = {}, cfg.get("shrink", 2000)
    for g in np.unique(segments):
        m = segments == g
        if m.sum() >= 200:
            lam = m.sum() / (m.sum() + shrink)  # shrink segment weights toward the global ones
            seg_w[g] = lam * solve(P[m], y[m], month[m]) + (1 - lam) * w

    def predict(Q, s):
        out = Q @ w
        for g, wg in seg_w.items():
            out[s == g] = Q[s == g] @ wg
        return out
    return predict, {"weights": w}


# ---------------------------------------------------------------- layer 2
def meta_features(P, lin, ctx, cfg):
    parts = [P - lin[:, None], lin[:, None]]
    if CTX_SETS[cfg.get("ctx", "full")]:
        parts.append(ctx[CTX_SETS[cfg.get("ctx", "full")]].to_numpy(dtype=float))
    return np.column_stack(parts)


def make_meta(cfg, n_pred):
    kind = cfg["meta"]
    if kind == "hgb":
        cat_mask = [False] * (n_pred + 1) + [c in CONTEXT_CAT for c in CTX_SETS[cfg.get("ctx", "full")]]
        return HistGradientBoostingRegressor(
            max_iter=cfg.get("max_iter", 200), learning_rate=cfg.get("lr", 0.05),
            max_leaf_nodes=cfg.get("leaves", 31), min_samples_leaf=cfg.get("min_leaf", 100),
            l2_regularization=cfg.get("l2", 1.0), categorical_features=cat_mask, random_state=SEED)
    if kind == "et":
        return ExtraTreesRegressor(n_estimators=cfg.get("trees", 300), min_samples_leaf=cfg.get("min_leaf", 100),
                                   max_features=cfg.get("max_features", 0.5), n_jobs=-1, random_state=SEED)
    if kind == "mlp":
        return MLPRegressor(hidden_layer_sizes=tuple(cfg.get("hidden", (64, 32))), alpha=cfg.get("alpha", 1e-3),
                            learning_rate_init=cfg.get("lr", 1e-3), max_iter=cfg.get("epochs", 200),
                            early_stopping=True, n_iter_no_change=10, random_state=SEED)
    raise ValueError(kind)


class Meta:
    """One layer-2 model: learns month-demeaned layer-1 residuals."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _X(self, P, lin, ctx):
        X = meta_features(P, lin, ctx, self.cfg)
        return X if self.cfg["meta"] == "hgb" else np.nan_to_num(X, nan=0.0)

    def fit(self, P, lin, ctx, target):
        X = self._X(P, lin, ctx)
        self.scaler, self.scale = None, 1.0
        if self.cfg["meta"] == "mlp":
            self.scaler, self.scale = StandardScaler().fit(X), float(target.std())
            X = self.scaler.transform(X)
        self.model = make_meta(self.cfg, P.shape[1]).fit(X, target / self.scale)
        return self

    def predict(self, P, lin, ctx):
        X = self._X(P, lin, ctx)
        return self.model.predict(self.scaler.transform(X) if self.scaler else X) * self.scale


class Fusion:
    """Layer 1 + optional layer 2. cfg["meta"] is one kind, or a list of member configs to average."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _members(self):
        meta = self.cfg.get("meta")
        if not meta:
            return []
        return [{**self.cfg, **m} for m in meta] if isinstance(meta, list) else [self.cfg]

    def _seg(self, ctx):
        return ctx[self.cfg["segment"]].to_numpy() if self.cfg.get("segment") else None

    def fit(self, P, y, month, ctx):
        self.linear, self.info = fit_linear(P, y, month, self.cfg, self._seg(ctx))
        lin = self.linear(P, self._seg(ctx))
        target = demean(y - lin, month)
        self.metas = [Meta(m).fit(P, lin, ctx, target) for m in self._members()]
        return self

    def predict(self, P, month, ctx):
        lin = self.linear(P, self._seg(ctx))
        if not self.metas:
            return lin
        corr = np.mean([m.predict(P, lin, ctx) for m in self.metas], axis=0)
        return lin + demean(corr, month) * self.cfg.get("meta_weight", 1.0)


# ---------------------------------------------------------------- evaluation
def evaluate(d, cfg):
    cols = POOLS[cfg.get("pool", "pool7")]
    P, y, month, ctx = d["valid"][cols].to_numpy(float), d["y"], d["month"], d["ctx_valid"]
    row = {}
    for name, (fit_m, eval_m) in schemes(d).items():
        model = Fusion(cfg).fit(P[fit_m], y[fit_m], month[fit_m], ctx[fit_m].reset_index(drop=True))
        pred = model.predict(P[eval_m], month[eval_m], ctx[eval_m].reset_index(drop=True))
        row[f"{name}_raw"], row[f"{name}_dm"] = scores(y[eval_m], pred, month[eval_m])
    row["mean_dm"] = (row["A_dm"] + row["B_dm"]) / 2
    row["mean_raw"] = (row["A_raw"] + row["B_raw"]) / 2
    return row


class Stage:
    def __init__(self, name):
        self.path = OUT / f"{name}.csv"
        self.done = set(pd.read_csv(self.path)["config"]) if self.path.exists() else set()
        self.d = data()

    def run(self, label, cfg):
        key = json.dumps(cfg, sort_keys=True)
        if key in self.done:
            return
        start = time.time()
        row = {"label": label, **evaluate(self.d, cfg), "seconds": round(time.time() - start, 1), "config": key}
        OUT.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        self.done.add(key)
        print(f"{label:60s} A {row['A_raw']:.2f}/{row['A_dm']:.2f}  B {row['B_raw']:.2f}/{row['B_dm']:.2f}  "
              f"mean_dm {row['mean_dm']:.2f}", flush=True)


STAGES = {}


def stage(fn):
    STAGES[fn.__name__] = fn
    return fn


@stage
def linear(s):
    """Reference points and layer-1 variants."""
    for name in ("rnn", "lstm", "ridge", "cat", "tfm"):
        s.run(f"single {name}", {"pool": "all10", "linear": "single", "name": name})
    for pool in POOLS:
        s.run(f"equal {pool}", {"pool": pool, "linear": "equal"})
        s.run(f"nnls raw {pool}", {"pool": pool, "linear": "nnls", "demean": False})
        s.run(f"nnls demeaned {pool}", {"pool": pool, "linear": "nnls"})
    for alpha in (1e3, 1e5, 1e7, 1e9):
        s.run(f"ridge demeaned alpha={alpha:g}", {"pool": "pool7", "linear": "ridge", "alpha": alpha})
    for seg in ("FLAT_TYPE_ORDINAL", "REGION", "TOWN"):
        for shrink in (500, 2000, 8000):
            s.run(f"nnls segment={seg} shrink={shrink}", {"pool": "pool7", "linear": "nnls", "segment": seg, "shrink": shrink})


@stage
def hgb(s):
    """Layer 2 = HistGradientBoosting on layer-1 residuals."""
    for ctx in ("none", "no_location", "full"):
        for leaves in (7, 15, 31):
            for lr, iters in ((0.03, 300), (0.1, 100)):
                for min_leaf in (100, 400, 1600):
                    for l2 in (0.0, 5.0):
                        cfg = {"pool": "pool7", "linear": "nnls", "meta": "hgb", "ctx": ctx,
                               "leaves": leaves, "lr": lr, "max_iter": iters, "min_leaf": min_leaf, "l2": l2}
                        s.run(f"hgb ctx={ctx} leaves={leaves} lr={lr} it={iters} minleaf={min_leaf} l2={l2}", cfg)


@stage
def et(s):
    """Layer 2 = ExtraTrees on layer-1 residuals."""
    for ctx in ("none", "no_location", "full"):
        for min_leaf in (25, 50, 100, 200, 400, 800):
            for max_features in (0.3, 0.6, 1.0):
                s.run(f"et ctx={ctx} minleaf={min_leaf} maxfeat={max_features}",
                      {"pool": "pool7", "linear": "nnls", "meta": "et", "ctx": ctx,
                       "min_leaf": min_leaf, "max_features": max_features})


@stage
def mlp(s):
    """Layer 2 = MLP on layer-1 residuals."""
    for ctx in ("none", "no_location", "full"):
        for hidden in ((32,), (64, 32), (128, 64)):
            for alpha in (1e-4, 1e-2, 1.0):
                for lr in (1e-3, 3e-3):
                    s.run(f"mlp ctx={ctx} hidden={hidden} alpha={alpha} lr={lr}",
                          {"pool": "pool7", "linear": "nnls", "meta": "mlp", "ctx": ctx,
                           "hidden": list(hidden), "alpha": alpha, "lr": lr})


BEST = {"et": {"meta": "et", "ctx": "no_location", "min_leaf": 50, "max_features": 0.3},
        "hgb": {"meta": "hgb", "ctx": "no_location", "leaves": 7, "lr": 0.03, "max_iter": 300, "min_leaf": 400, "l2": 0.0},
        "mlp": {"meta": "mlp", "ctx": "no_location", "hidden": [64, 32], "alpha": 1.0, "lr": 0.003}}


@stage
def refine(s):
    """Around the best layer-2 configs: averaged meta-models, shrinkage, pools, edges of the grids."""
    base = {"pool": "pool7", "linear": "nnls"}
    combos = {"et+hgb": ["et", "hgb"], "et+mlp": ["et", "mlp"], "hgb+mlp": ["hgb", "mlp"], "et+hgb+mlp": ["et", "hgb", "mlp"]}
    for label, names in combos.items():
        s.run(f"avg {label}", {**base, "meta": [BEST[n] for n in names]})
    for w in (0.5, 0.75, 1.25):
        s.run(f"et meta_weight={w}", {**base, **BEST["et"], "meta_weight": w})
        s.run(f"avg et+hgb+mlp meta_weight={w}", {**base, "meta": [BEST[n] for n in ("et", "hgb", "mlp")], "meta_weight": w})
    for pool in ("all10", "no_knn", "top4"):
        s.run(f"et pool={pool}", {**base, **BEST["et"], "pool": pool})
        s.run(f"avg et+hgb+mlp pool={pool}", {**base, "pool": pool, "meta": [BEST[n] for n in ("et", "hgb", "mlp")]})
    s.run("et linear=equal", {**base, **BEST["et"], "linear": "equal"})
    s.run("avg et+hgb+mlp linear=equal", {**base, "linear": "equal", "meta": [BEST[n] for n in ("et", "hgb", "mlp")]})
    for mf in (0.15, 0.2):
        for ml in (25, 50, 100):
            s.run(f"et minleaf={ml} maxfeat={mf}", {**base, **BEST["et"], "min_leaf": ml, "max_features": mf})
    s.run("et trees=800", {**base, **BEST["et"], "trees": 800})
    for alpha in (3.0, 10.0):
        s.run(f"mlp alpha={alpha}", {**base, **BEST["mlp"], "alpha": alpha})


FINAL = {"nnls": {"pool": "pool7", "linear": "nnls"},
         "et": {"pool": "pool7", "linear": "nnls", **BEST["et"]},
         "stack3": {"pool": "pool7", "linear": "nnls", "meta": [BEST[n] for n in ("et", "hgb", "mlp")]}}


def levels(pred, d):
    """L1 keeps the fused level; L2 sets each test month to the flat-family mean; L3 adds post-jump growth."""
    month = d["test_month"]
    flat = d["test"][FLAT_FAMILY].to_numpy(float).mean(axis=1)
    target = pd.Series(flat).groupby(month).transform("mean").to_numpy()
    l2 = pred * target / pd.Series(pred).groupby(month).transform("mean").to_numpy()
    h = np.asarray(month_index(month)) - month_index(["2025-03"])[0]
    return {"L1": pred, "L2": l2, "L3": l2 * np.exp(d["growth"] * h)}


def finalize():
    """Fit each FINAL config on all holdout-fold rows and write L1/L2/L3 submissions to results/P1/fusion."""
    d, test_df = data(), load_test()
    out = HERE.parents[1] / "results" / "P1" / "fusion"
    scored = pd.concat([pd.read_csv(p) for p in OUT.glob("*.csv")]).drop_duplicates("config").set_index("config")
    fit_m = (d["valid"]["fold"] == "holdout").to_numpy()
    rows = []
    for name, cfg in FINAL.items():
        cols = POOLS[cfg["pool"]]
        P = d["valid"][cols].to_numpy(float)
        model = Fusion(cfg).fit(P[fit_m], d["y"][fit_m], d["month"][fit_m], d["ctx_valid"][fit_m].reset_index(drop=True))
        pred = model.predict(d["test"][cols].to_numpy(float), d["test_month"], d["ctx_test"])
        key = json.dumps(cfg, sort_keys=True)
        val = scored.loc[key] if key in scored.index else evaluate(d, cfg)
        for level, values in levels(pred, d).items():
            path = write_submission(test_df, values, out / f"fusion_{name}_{level}.csv")
            rows.append({"submission": path.name, "method": name, "level": level,
                         **{k: round(float(val[k]), 3) for k in ("A_raw", "A_dm", "B_raw", "B_dm", "mean_dm")},
                         "test_mean": round(float(values.mean()), 1),
                         "weights": json.dumps(dict(zip(cols, np.round(model.info["weights"], 4).tolist()))),
                         "config": key})
    rows = pd.DataFrame(rows)
    rows.to_csv(out / "manifest.csv", index=False)
    means = d["test"].mean().round(1).to_dict()
    print(rows.drop(columns=["config", "weights"]).to_string(index=False))
    print("growth/month", round(d["growth"], 5), "| base test means", means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*STAGES, "finalize"])
    name = parser.parse_args().stage
    if name == "finalize":
        finalize()
    else:
        STAGES[name](Stage(name))
    print("STAGE DONE", name)
