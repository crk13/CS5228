"""Reproduce three finalized afternoon models with the public delivery API."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The user's project fixes seed 42. Do not edit the shared configuration file.
import src.config as project_config
project_config.SEED = 42

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from threadpoolctl import threadpool_limits
import src.experiment as experiment
experiment.SEED = 42
import src.features.distance
import src.features.macro
from src.cv import DATE_COL, get_splits
from src.data_cleaning import TARGET, load_train, load_test
from src.features import FeaturePipeline
from src.features.distance import DISTANCE_COLUMNS
from src.features.macro import COE_COLUMNS
from src.models.trees import tree_model

OUT = ROOT / "results/Yaqi"
SELECTED = {"catboost": "location_coe", "xgboost": "base", "lightgbm": "location_coe"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hashes():
    paths = sorted((ROOT / "data").rglob("*.csv")) + sorted((ROOT / "data_cleaned").rglob("*.csv"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def blocks(kind):
    return ["basic", "block"] + ([] if SELECTED[kind] == "base" else ["distance", "macro_coe"])


def build(kind):
    numeric = () if SELECTED[kind] == "base" else tuple(DISTANCE_COLUMNS + COE_COLUMNS)
    return tree_model(kind, months=24, address=True, threads=4, extra_numeric=numeric)


class TrainingPredictionRecorder(RegressorMixin, BaseEstimator):
    """Record diagnostic training predictions without changing fitted parameters."""

    def __init__(self, estimator, cache_dir):
        self.estimator = estimator
        self.cache_dir = cache_dir

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        prediction = self.estimator_.predict(X)
        dates = pd.PeriodIndex(X[DATE_COL], freq="M")
        end = str(dates.max())
        labels = {"2023-11": "holdout", "2024-03": "rolling_2024_04", "2025-03": "final"}
        label = labels.get(end, "training_" + end)
        used = (dates >= pd.Period(self.estimator_.fit_start_, freq="M")) & (
            dates <= pd.Period(self.estimator_.fit_end_, freq="M"))
        assert int(used.sum()) == self.estimator_.fit_rows_
        assert np.isfinite(prediction).all()
        folder = ROOT / self.cache_dir
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row_index": X.index.to_numpy(), "fold": label,
                      DATE_COL: X[DATE_COL].to_numpy(), TARGET: np.asarray(y),
                      "Predicted": prediction, "used_for_model_fit": used}).to_csv(folder / f"{label}.csv", index=False)
        write_json(folder / f"{label}.json", {"fold": label, "input_rows": len(X),
                   "actual_fit_rows": self.estimator_.fit_rows_, "fit_start": self.estimator_.fit_start_,
                   "fit_end": self.estimator_.fit_end_})
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=list(SELECTED))
    args = parser.parse_args()
    kind = args.kind
    name = f"{kind}_{SELECTED[kind]}"
    before = hashes()
    original = json.loads((ROOT / f"notebooks/Yaqi/original_configs/{name}.json").read_text(encoding="utf-8"))
    recipe = experiment._json_value(experiment._new_model(build(kind)).get_params(deep=False))
    assert recipe == original["parameters"], "Afternoon model parameters must remain unchanged"
    if (OUT / "experiments/log.csv").exists():
        previous = pd.read_csv(OUT / "experiments/log.csv")
        if previous.name.eq("yaqi_final_" + name).any():
            raise RuntimeError(f"{name} is already delivered; do not add duplicate runs")
    cache = f"outputs/yaqi_delivery_cache/{kind}"
    recorder = TrainingPredictionRecorder(build(kind), cache)
    print(f"Delivery training: {name}; seed=42, recent=24 months, threads=4", flush=True)
    with threadpool_limits(limits=4):
        result = experiment.run_experiment("yaqi_final_" + name, "Yaqi", blocks(kind), recorder,
                                           split="rolling", final=True, output_dir=OUT)
    np.testing.assert_allclose([result.fold_rmse[n] for n in ("holdout", "rolling_2024_04")],
                               [original["fold_rmse"][n] for n in ("holdout", "rolling_2024_04")],
                               rtol=0, atol=1e-6)
    cache_folder = ROOT / cache
    shutil.copy2(cache_folder / "final.csv", result.run_dir / "training_predictions.csv")
    fold_training = pd.concat([pd.read_csv(cache_folder / f"{name}.csv")
                               for name in ("holdout", "rolling_2024_04")], ignore_index=True)
    fold_training.to_csv(result.run_dir / "fold_training_predictions.csv", index=False)
    fitting = [json.loads((cache_folder / f"{name}.json").read_text(encoding="utf-8"))
               for name in ("holdout", "rolling_2024_04", "final")]
    train, test = load_train(), load_test()
    np.testing.assert_array_equal(train.index, np.arange(len(train)))
    audit = []
    for label, part in [(f.definition.name, train.iloc[f.train_idx]) for f in get_splits(train, "rolling")] + [("final", train)]:
        pipeline = FeaturePipeline(blocks(kind)).fit(part)
        audit.append({"fold": label, "features": {key: block.audit_ for key, block in pipeline.blocks
                                                  if hasattr(block, "audit_")}})
    config_path = result.run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["submission_path"] = result.submission_path.relative_to(ROOT).as_posix()
    config["delivery"] = {"kind": kind, "scenario": SELECTED[kind], "source_afternoon_run_id": original["run_id"],
                          "afternoon_scores_reproduced": True, "estimator_recipe_unchanged": True,
                          "training_window_months": 24, "actual_fit_windows": fitting,
                          "training_predictions_role": "diagnostic; fitted-model predictions, not OOF",
                          "validation_predictions_role": "held-out chronological predictions for meta-model training",
                          "test_predictions_role": "features for final meta-model inference",
                          "feature_audit": audit, "weights_included": False,
                          "versions": {n: importlib.metadata.version(n) for n in
                                       ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost")}}
    write_json(config_path, config)
    assert before == hashes(), "Data files were modified"
    write_json(OUT / f"{kind}_receipt.json", {"kind": kind, "run_id": result.run_id,
               "run_dir": result.run_dir.relative_to(ROOT).as_posix(), "mean_rmse": result.mean_rmse,
               "fold_rmse": result.fold_rmse, "train_rows": len(train), "test_rows": len(test),
               "datasets_unchanged": True, "data_hashes": before, "source_run_id": original["run_id"]})
    print(f"Completed delivery: {kind} -> {result.run_id}", flush=True)


if __name__ == "__main__":
    main()
