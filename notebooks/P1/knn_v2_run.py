"""P1: KNN improvements, tested one at a time and then combined greedily.

Run from the repo root:
  python notebooks/P1/knn_v2_run.py ablation  # rolling runs in a temp dir -> knn_v2_ablation.csv
  python notebooks/P1/knn_v2_run.py deliver   # BEST on rolling + submission -> results/P1/
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import ROOT_DIR
from src.experiment import run_experiment
from src.models.knn import KNN_COLUMNS, KNN_FEATURES, rent_knn_model

# Each change is (estimator params, target transform or None to keep the current one).
CHANGES = {
    "clip_month": ({"clip_month": True}, None),
    "geo_km": ({"geo_km": True}, None),
    "market_level": ({"level": "market"}, None),
    "add_age_floor": ({"features": (*KNN_COLUMNS, "FLAT_AGE", "MAX_FLOOR")}, None),
    "per_sqm": ({}, "per_sqm"),
    "by_flat_type": ({"by_flat_type": True}, None),
    "ridge_residual": ({"base": "ridge"}, None),
    "uniform": ({"weights": "uniform"}, None),
}
TUNE = ({"tune": True}, None)
TABLE_PATH = Path(__file__).with_name("knn_v2_ablation.csv")
BEST = ["ridge_residual", "add_age_floor"]  # greedy choice in knn_v2_ablation.csv


def combine(names, extra=()):
    params, target = {}, "none"
    for change, transform in [CHANGES[n] for n in names] + list(extra):
        params |= change
        target = transform or target
    return params, target


def evaluate(output_dir, stage, names, extra=()):
    params, target = combine(names, extra)
    label = " + ".join(names + (["tune"] if extra else [])) or "basic"
    result = run_experiment(f"knn_v2 {label}", "P1", KNN_FEATURES, rent_knn_model(target, **params),
                            split="rolling", output_dir=output_dir)
    return {"stage": stage, "config": label, **result.fold_rmse, "mean_rmse": result.mean_rmse}


def ablation():
    output_dir = tempfile.mkdtemp(prefix="p1_knn_v2_")
    rows = [evaluate(output_dir, "baseline", [])]
    base = rows[0]["mean_rmse"]
    single = {name: evaluate(output_dir, "single", [name]) for name in CHANGES}
    rows += single.values()
    rows.append(evaluate(output_dir, "single", [], [TUNE]))
    # Greedy: add changes in order of their single gain, keep one only if the rolling mean improves.
    chosen, best = [], base
    for name in sorted(CHANGES, key=lambda n: single[n]["mean_rmse"]):
        if single[name]["mean_rmse"] >= base:
            continue
        row = single[name] if not chosen else evaluate(output_dir, "greedy", chosen + [name])
        if chosen:
            rows.append(row)
        if row["mean_rmse"] < best:
            chosen, best = chosen + [name], row["mean_rmse"]
    rows.append(evaluate(output_dir, "final", chosen, [TUNE]))
    # Clipped months create exact matches that distance weights collapse onto; recheck with uniform.
    for names in (["clip_month", "uniform"], chosen + ["clip_month", "uniform"]):
        rows.append(evaluate(output_dir, "follow_up", names))
    table = pd.DataFrame(rows)
    table["delta_vs_basic"] = table["mean_rmse"] - base
    table.round(3).to_csv(TABLE_PATH, index=False)
    print(table.round(2).to_string(index=False))
    print("greedy choice:", chosen)


def deliver():
    params, target = combine(BEST)
    result = run_experiment("knn_v2 " + " + ".join(BEST), "P1", KNN_FEATURES, rent_knn_model(target, **params),
                            split="rolling", final=True, output_dir=ROOT_DIR / "results" / "P1")
    print(result.fold_rmse, result.mean_rmse)
    print("run_id:", result.run_id)
    print("submission:", result.submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ablation", "deliver"])
    {"ablation": ablation, "deliver": deliver}[parser.parse_args().mode]()
