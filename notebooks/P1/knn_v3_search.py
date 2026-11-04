"""P1: KNN search with auxiliary features; each stage appends rolling results to knn_v3/<stage>.csv.

Run from the repo root:  python notebooks/P1/knn_v3_search.py <stage>
Stage order: contrib, greedy_ridge, greedy_knn -> combine -> hyper_a, hyper_b -> refine -> polish
-> dropcheck, final_check; `deliver` writes the polish config to results/P1/ for fusion.
Search runs go to a temp dir. A rerun skips configs already in the stage CSV, so an interrupted stage
resumes where it stopped. Greedy stages also write knn_v3/<stage>.best.json.

A config is a patch on DEFAULT (config A). Patch keys add_knn / drop_knn, add_ridge / drop_ridge
and add_ridge_cat / drop_ridge_cat edit feature lists, set_weights merges into feature_weights; any
other key replaces the value. Later stages start from the lowest-mean config of an earlier stage.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

import src.features.p1_aux  # noqa: F401  (registers p1_aux, p1_aux_lag0)
from src.config import ROOT_DIR
from src.experiment import run_experiment
from src.features.p1_aux import AUX_COLUMNS, AUX_GROUPS
from src.models.knn import KNN_COLUMNS, rent_knn_model

OUT_DIR = Path(__file__).with_name("knn_v3")
A_KNN = [*KNN_COLUMNS, "FLAT_AGE", "MAX_FLOOR"]
DEFAULT = {
    "aux": "p1_aux", "knn_features": A_KNN,
    "ridge_numeric": ["FLOOR_AREA_SQM", "REMAINING_LEASE"],
    "ridge_categorical": ["FLAT_TYPE", "FLAT_MODEL", "TOWN"],
    "base": "ridge", "alpha": 1.0, "n_neighbors": 80, "weights": "distance", "clip_month": False,
    "geo_km": False, "level": "none", "by_flat_type": False, "feature_weights": None,
    "target_transform": "none",
}
LISTS = {"knn": "knn_features", "ridge": "ridge_numeric", "ridge_cat": "ridge_categorical"}
SPATIAL = [c for c in AUX_COLUMNS if c not in AUX_GROUPS["macro"]]


def apply(config, patch):
    config = dict(config)
    for key, value in patch.items():
        action, _, target = key.partition("_")
        if action in ("add", "drop") and target in LISTS:
            current = list(config[LISTS[target]])
            config[LISTS[target]] = (current + [v for v in value if v not in current] if action == "add"
                                     else [v for v in current if v not in value])
        elif key == "set_weights":
            config["feature_weights"] = {**(config["feature_weights"] or {}), **value}
        else:
            config[key] = value
    return config


def make_model(c):
    weights = {k: v for k, v in (c["feature_weights"] or {}).items() if k in c["knn_features"]} or None
    return rent_knn_model(
        c["target_transform"], features=tuple(c["knn_features"]), base=c["base"], alpha=c["alpha"],
        ridge_numeric=tuple(c["ridge_numeric"]), ridge_categorical=tuple(c["ridge_categorical"]),
        n_neighbors=c["n_neighbors"], weights=c["weights"], clip_month=c["clip_month"],
        geo_km=c["geo_km"], level=c["level"], by_flat_type=c["by_flat_type"], feature_weights=weights,
    )


class Stage:
    def __init__(self, name):
        self.path = OUT_DIR / f"{name}.csv"
        self.tmp = Path(tempfile.mkdtemp(prefix=f"p1_knn_v3_{name}_"))
        self.cache = {}
        if self.path.exists():
            for row in pd.read_csv(self.path).to_dict("records"):
                self.cache[row["config"]] = row

    def evaluate(self, label, config, final=False):
        config = apply(DEFAULT, {}) | config
        key = json.dumps(config, sort_keys=True)
        if key in self.cache and not final:
            return self.cache[key]
        start = time.time()
        result = run_experiment(f"knn_v3 {label}", "P1", ["basic", "block", config["aux"]], make_model(config),
                                split="rolling", final=final, output_dir=self.tmp)
        row = {"label": label, **result.fold_rmse, "mean_rmse": result.mean_rmse,
               "seconds": round(time.time() - start, 1), "config": key}
        if final:
            row["submission"] = str(result.submission_path)
            return row
        shutil.rmtree(result.run_dir, ignore_errors=True)
        OUT_DIR.mkdir(exist_ok=True)
        pd.DataFrame([row]).to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        self.cache[key] = row
        print(f"{label:48s} {row['holdout']:.2f} {row['rolling_2024_04']:.2f} mean={row['mean_rmse']:.2f}", flush=True)
        return row

    def batch(self, base, patches):
        start = apply(DEFAULT, base)
        return {label: self.evaluate(label, apply(start, patch)) for label, patch in patches.items()}

    def greedy(self, base, candidates, min_gain=0.3):
        """Best-first forward selection; accept only if the mean gains min_gain and both folds improve."""
        config = apply(DEFAULT, base)
        current = self.evaluate("greedy base", config)
        remaining, accepted = dict(candidates), []
        while remaining:
            trials = {name: self.evaluate(f"step{len(accepted) + 1} {name}", apply(config, patch))
                      for name, patch in remaining.items()}
            name, row = min(trials.items(), key=lambda item: item[1]["mean_rmse"])
            if not (row["mean_rmse"] <= current["mean_rmse"] - min_gain
                    and row["holdout"] < current["holdout"] and row["rolling_2024_04"] < current["rolling_2024_04"]):
                break
            config, current = apply(config, remaining.pop(name)), row
            accepted.append({"step": name, **{k: row[k] for k in ("holdout", "rolling_2024_04", "mean_rmse")}})
            print(f"ACCEPT {name} -> mean {row['mean_rmse']:.2f}", flush=True)
        self.path.with_suffix(".best.json").write_text(json.dumps(
            {"accepted": accepted, "config": config, "mean_rmse": current["mean_rmse"]}, indent=2) + "\n")
        return config


STAGES = {}


def stage(fn):
    STAGES[fn.__name__] = fn
    return fn


@stage
def contrib(s):
    """Full aux run, then each aux group added to KNN only, Ridge only and both."""
    patches = {"A": {}, "all aux knn+ridge": {"add_knn": AUX_COLUMNS, "add_ridge": AUX_COLUMNS},
               "all aux knn": {"add_knn": AUX_COLUMNS}, "all aux ridge": {"add_ridge": AUX_COLUMNS},
               "spatial knn+ridge": {"add_knn": SPATIAL, "add_ridge": SPATIAL},
               "spatial knn": {"add_knn": SPATIAL}, "spatial ridge": {"add_ridge": SPATIAL}}
    for group, columns in AUX_GROUPS.items():
        patches |= {f"{group} knn": {"add_knn": columns}, f"{group} ridge": {"add_ridge": columns},
                    f"{group} knn+ridge": {"add_knn": columns, "add_ridge": columns}}
    patches |= {f"macro lag0 {where}": {**patch, "aux": "p1_aux_lag0"} for where, patch in (
        ("knn", {"add_knn": AUX_GROUPS["macro"]}), ("ridge", {"add_ridge": AUX_GROUPS["macro"]}))}
    s.batch({}, patches)


@stage
def greedy_ridge(s):
    """Forward selection of Ridge base features; KNN stays as in A."""
    candidates = {f"+{c}": {"add_ridge": [c]} for c in [*AUX_COLUMNS, "FLAT_AGE", "MAX_FLOOR"]}
    candidates |= {f"+cat {c}": {"add_ridge_cat": [c]} for c in ("PLANNING_AREA", "SUBZONE", "REGION")}
    s.greedy({}, candidates)


@stage
def greedy_knn(s):
    """Forward selection over adding aux features to and dropping A features from the KNN distance."""
    candidates = {f"+{c}": {"add_knn": [c]} for c in AUX_COLUMNS}
    candidates |= {f"-{c}": {"drop_knn": [c]} for c in A_KNN}
    s.greedy({}, candidates)


def lowest(name):
    rows = pd.read_csv(OUT_DIR / f"{name}.csv")
    return json.loads(rows.loc[rows["mean_rmse"].idxmin(), "config"])


def best(name):
    return json.loads((OUT_DIR / f"{name}.best.json").read_text())["config"]


@stage
def combine(s):
    """Greedy Ridge features together with greedy KNN features."""
    ridge = {k: best("greedy_ridge")[k] for k in ("ridge_numeric", "ridge_categorical")}
    knn = {"knn_features": best("greedy_knn")["knn_features"]}
    s.batch({}, {"ridge best": ridge, "knn best": knn, "ridge best + knn best": ridge | knn})


# Single changes on top of the combined config; refine later combines them greedily.
HYPER_B = {
    "alpha 0.1": {"alpha": 0.1}, "alpha 10": {"alpha": 10.0}, "alpha 100": {"alpha": 100.0},
    "geo_km": {"geo_km": True}, "level market": {"level": "market"}, "by_flat_type": {"by_flat_type": True},
    "per_sqm": {"target_transform": "per_sqm"}, "macro lag0": {"aux": "p1_aux_lag0"},
    "month w0.5": {"set_weights": {"MONTH_INDEX": 0.5}}, "month w2": {"set_weights": {"MONTH_INDEX": 2.0}},
    "geo w0.5": {"set_weights": {"LATITUDE": 0.5, "LONGITUDE": 0.5}},
    "geo w2": {"set_weights": {"LATITUDE": 2.0, "LONGITUDE": 2.0}},
}


@stage
def hyper_a(s):
    """k x neighbour weighting x month clipping on the combined config."""
    s.batch(lowest("combine"), {f"k{k} {w}{' clip' if clip else ''}": {"n_neighbors": k, "weights": w, "clip_month": clip}
                                for k in (40, 60, 80, 120, 160) for w in ("distance", "uniform") for clip in (False, True)})


@stage
def hyper_b(s):
    s.batch(lowest("combine"), HYPER_B)


@stage
def refine(s):
    """Greedy over every hyperparameter change (the hyper_a / hyper_b scans) from the combined config."""
    base = lowest("combine")
    uses_macro = any(c in AUX_GROUPS["macro"] for c in base["knn_features"] + base["ridge_numeric"])
    candidates = {k: v for k, v in HYPER_B.items() if k != "macro lag0" or uses_macro}
    candidates |= {f"k{k}": {"n_neighbors": k} for k in (40, 60, 80, 120, 160) if k != base["n_neighbors"]}
    candidates |= {f"weights {w}": {"weights": w} for w in ("distance", "uniform") if w != base["weights"]}
    candidates |= {f"clip {c}": {"clip_month": c} for c in (False, True) if c != base["clip_month"]}
    s.greedy(base, candidates)


@stage
def polish(s):
    """Second feature pass after tuning: add or drop features in either component."""
    base = best("refine")
    spatial = [*SPATIAL, "FLAT_AGE", "MAX_FLOOR"]
    candidates = {f"-knn {c}": {"drop_knn": [c]} for c in base["knn_features"] if c != "MONTH_INDEX"}
    candidates |= {f"+knn {c}": {"add_knn": [c]} for c in SPATIAL if c not in base["knn_features"]}
    candidates |= {f"-ridge {c}": {"drop_ridge": [c]} for c in base["ridge_numeric"]}
    candidates |= {f"+ridge {c}": {"add_ridge": [c]} for c in spatial if c not in base["ridge_numeric"]}
    candidates |= {f"-ridge cat {c}": {"drop_ridge_cat": [c]} for c in base["ridge_categorical"]}
    candidates |= {f"+ridge cat {c}": {"add_ridge_cat": [c]} for c in ("PLANNING_AREA", "SUBZONE", "REGION")
                   if c not in base["ridge_categorical"]}
    s.greedy(base, candidates)


@stage
def dropcheck(s):
    """Contribution of each feature in the final config: drop it alone."""
    base = best("polish")
    patches = {"final": {}}
    patches |= {f"-knn {c}": {"drop_knn": [c]} for c in base["knn_features"]}
    patches |= {f"-ridge {c}": {"drop_ridge": [c]} for c in base["ridge_numeric"]}
    patches |= {f"-ridge cat {c}": {"drop_ridge_cat": [c]} for c in base["ridge_categorical"]}
    s.batch(base, patches)


@stage
def final_check(s):
    """Full train + test prediction for the final config; test monthly means go to final_test_months.csv."""
    from src.data_cleaning import load_test
    row = s.evaluate("final", best("polish"), final=True)
    months = load_test()["RENT_APPROVAL_DATE"].astype(str).to_numpy()
    test = pd.read_csv(row["submission"]).assign(month=months)
    test.groupby("month")["Predicted"].mean().round(1).to_csv(OUT_DIR / "final_test_months.csv")
    print(f"final {row['holdout']:.2f} {row['rolling_2024_04']:.2f} mean={row['mean_rmse']:.2f}")
    print(test.groupby("month")["Predicted"].mean().round(0).to_string())


@stage
def deliver(s):
    """Final config on rolling + submission into results/P1 (fusion delivery, at most 3 runs there)."""
    config = best("polish")
    result = run_experiment("knn_v3 aux", "P1", ["basic", "block", config["aux"]], make_model(config),
                            split="rolling", final=True, output_dir=ROOT_DIR / "results" / "P1")
    print(result.fold_rmse, result.mean_rmse)
    print("run_id:", result.run_id)
    print("submission:", result.submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES))
    name = parser.parse_args().stage
    STAGES[name](Stage(name))
    print("STAGE DONE", name)
