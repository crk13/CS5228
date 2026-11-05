"""Train neural models with the unchanged shared experiment framework."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from threadpoolctl import threadpool_limits

from src.config import SEED
from src.experiment import run_experiment
from src.models import group_median_model, hist_gradient_boosting_model
from src.models.yuxin_nn import MarketHybridRegressor, TabularNNRegressor
from src.models.yuxin_nn.ensemble import NeuralFamilyEnsemble

MODELS = ("mlp", "tabular_transformer", "market_carry", "market_rnn", "market_lstm", "market_transformer")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes():
    paths = sorted((ROOT / "data").rglob("*.csv")) + sorted((ROOT / "data_cleaned").rglob("*.csv"))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def build_model(name, *, smoke=False, verbose=False, config_dir=None):
    if name == "baseline_median":
        return group_median_model(), []
    if name == "baseline_hist":
        return hist_gradient_boosting_model(), ["basic", "block"]
    config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
    recipe = json.loads((config_dir / f"{name}.json").read_text())
    classes = {"tabular": TabularNNRegressor, "market": MarketHybridRegressor, "ensemble": NeuralFamilyEnsemble}
    if recipe["class"] not in classes:
        raise ValueError("Unknown neural configuration class")
    cls = classes[recipe["class"]]
    params = dict(recipe["parameters"])
    if cls is not NeuralFamilyEnsemble:
        params["verbose"] = verbose
    if smoke:
        if cls is NeuralFamilyEnsemble:
            raise ValueError("Use individual-member smoke tests before ensemble delivery")
        params.update(epochs=2, inner_months=1) if cls is TabularNNRegressor else params.update(
            property_epochs=2, temporal_epochs=3, inner_months=1, temporal_inner_months=2)
    return cls(**params), ["basic", "block"]


class AuditRecorder(RegressorMixin, BaseEstimator):
    """Persist fold-local training evidence without changing predictions or fit inputs."""

    def __init__(self, estimator, audit_dir):
        self.estimator = estimator
        self.audit_dir = audit_dir

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        origin = X["RENT_APPROVAL_DATE"].max()
        evidence = {"origin": origin, "train_rows": len(X),
                    "audit": getattr(self.estimator_, "audit_", {}),
                    "selection_history": getattr(self.estimator_, "selection_history_", []),
                    "training_history": getattr(self.estimator_, "training_history_", [])}
        if hasattr(self.estimator_, "property_"):
            evidence.update(property_selection_history=self.estimator_.property_.selection_history_,
                            property_training_history=self.estimator_.property_.training_history_,
                            market_selection_history=self.estimator_.market_.selection_history_,
                            market_training_history=getattr(self.estimator_.market_, "training_history_", []))
            market = self.estimator_.market_
            months = pd.period_range(market.origin_, market.origin_ + 16, freq="M").astype(str)
            evidence["market_forecast"] = dict(zip(months, market.predict_months(months).tolist()))
        # Check inference does not depend on query order, batching or stray label columns.
        probe = X.iloc[:min(len(X), 31)].copy()
        direct = self.estimator_.predict(probe)
        reversed_prediction = self.estimator_.predict(probe.iloc[::-1])[::-1]
        batched = np.concatenate([self.estimator_.predict(probe.iloc[:7]), self.estimator_.predict(probe.iloc[7:])])
        changed = probe.assign(MONTHLY_RENT=1e9, Id=-999, row_index=-999)
        deltas = [np.max(np.abs(direct - other), initial=0.0) for other in
                  (reversed_prediction, batched, self.estimator_.predict(changed))]
        if max(deltas) > 0.01:
            raise AssertionError(f"Order/batch/label inference invariance failed: {deltas}")
        evidence["inference_probe_max_abs_delta_sgd"] = float(max(deltas))
        write_json(Path(self.audit_dir) / f"{origin}.json", evidence)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def run_one(name, output, *, split="rolling", final=False, smoke=False, verbose=False, config_dir=None):
    before = source_hashes()
    model, features = build_model(name, smoke=smoke, verbose=verbose, config_dir=config_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep transient audit files/checkpoints out of Git. Final evidence is attached below.
    with tempfile.TemporaryDirectory(prefix="yuxin_audit_") as audit_dir:
        start = time.monotonic()
        recorder = AuditRecorder(model, audit_dir)
        # Do not mutate both Conda sklearn and PyTorch OpenMP runtimes here.
        # Neural estimators own their torch thread setting (CPU default: one).
        with threadpool_limits(limits=2, user_api="blas"):
            result = run_experiment(f"yuxin_{name}", "Yuxin", features, recorder,
                                    split=split, final=final, output_dir=output)
        audits = [json.loads(p.read_text()) for p in sorted(Path(audit_dir).glob("*.json"))]
    config_path = result.run_dir / "config.json"
    config = json.loads(config_path.read_text())
    if result.submission_path is not None:
        config["submission_path"] = "submissions/" + result.submission_path.name
    # The cache path is not an inference dependency, nor part of a reproducible recipe.
    config["parameters"]["audit_dir"] = "temporary; training evidence is in training_audit.json"
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "torch", "optuna")}
    config["yuxin"] = {"model_name": name, "seconds": time.monotonic()-start,
                       "versions": versions, "python": platform.python_version(),
                       "platform": platform.platform(), "device": "cpu", "seed": SEED,
                       "torch_deterministic": True, "smoke_only": smoke,
                       "features": features, "data_hashes": before,
                       "model_source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted((ROOT / "src/models/yuxin_nn").glob("*.py"))}}
    if name in MODELS:
        recipe_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        recipe_file = recipe_dir / f"{name}.json"
        config["yuxin"]["recipe"] = json.loads(recipe_file.read_text())
        config["yuxin"]["recipe_sha256"] = hashlib.sha256(recipe_file.read_bytes()).hexdigest()
    write_json(config_path, config)
    write_json(result.run_dir / "training_audit.json", audits)
    if before != source_hashes():
        raise AssertionError("Input data changed during training")
    print(json.dumps({"name": name, "run_id": result.run_id, "mean_rmse": result.mean_rmse,
                      "seconds": config["yuxin"]["seconds"], "directory": str(result.run_dir)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", choices=[*MODELS, "baseline_median", "baseline_hist", "all"], default=["mlp"])
    parser.add_argument("--split", choices=["holdout", "rolling"], default="holdout")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-dir", type=Path, help="Optional personal recipes; defaults are unchanged")
    parser.add_argument("--deliver", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="two-epoch pipeline smoke, not a selected model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    names = list(MODELS) if "all" in args.model else args.model
    if len(names) != len(set(names)):
        parser.error("Duplicate model names")
    if args.deliver:
        if args.output or args.smoke or args.split != "rolling" or not args.final:
            parser.error("--deliver requires --split rolling --final; no --output/--smoke")
        output = ROOT / "results/Yuxin"
        existing = list((output / "experiments").glob("*/config.json"))
        if len(existing) + len(names) > 3:
            parser.error("Delivery limited to three runs; use a development --output instead")
    else:
        output = args.output or Path(tempfile.mkdtemp(prefix="yuxin_experiments_"))
    print(f"Output root: {output}", flush=True)
    for name in names:
        run_one(name, output, split=args.split, final=args.final, smoke=args.smoke, verbose=args.verbose,
                config_dir=args.config_dir)


if __name__ == "__main__":
    main()
