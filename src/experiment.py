"""Comparable time-based experiments, validation predictions, and submissions."""

import csv
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone

from src.config import ROOT_DIR, SEED
from src.cv import DATE_COL, get_splits
from src.data_cleaning import load_train, load_test, TARGET, ID_COL
from src.features import FeaturePipeline

LOG_COLUMNS = [
    "timestamp", "run_id", "name", "author", "features", "model", "parameters",
    "split", "seed", "final", "fold_rmse", "mean_rmse",
]


@dataclass
class ExperimentResult:
    run_id: str
    fold_rmse: dict[str, float]
    mean_rmse: float
    run_dir: Path
    submission_path: Path | None = None


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _json_value(value):
    if isinstance(value, BaseEstimator):
        return {"class": _class_name(value), "parameters": _json_value(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)


def _new_model(model):
    fitted = clone(model)
    params = fitted.get_params(deep=True)
    fitted.set_params(**{key: SEED for key in params if key.split("__")[-1] == "random_state"})
    return fitted


def _predictions(model, X):
    values = np.asarray(model.predict(X), dtype=float)
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise ValueError("predict must return one finite original-scale rent per row")
    return values


def write_submission(test: pd.DataFrame, prediction, path: Path) -> Path:
    """Preserve the loader's test row order; reject malformed competition output."""
    values = np.asarray(prediction, dtype=float)
    if len(test) != 50000 or not np.array_equal(test[ID_COL].to_numpy(), np.arange(50000)):
        raise ValueError("Submission requires 50000 test rows with Id 0..49999 in order")
    if values.shape != (50000,) or not np.isfinite(values).all():
        raise ValueError("Submission requires 50000 finite predictions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test[ID_COL].to_numpy(), "Predicted": values}).to_csv(path, index=False)
    return path


def _append_log(path, row):
    # An exclusive lock prevents simultaneous local runs from racing on the header.
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + 30
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Experiment log is locked: {lock}")
            time.sleep(0.1)
    try:
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            with path.open(encoding="utf-8", newline="") as stream:
                if next(csv.reader(stream)) != LOG_COLUMNS:
                    raise ValueError(f"Unexpected experiment log columns: {path}")
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LOG_COLUMNS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    finally:
        lock.unlink()


def run_experiment(name, author, features: list[str], model, split="holdout", final=False,
                   *, output_dir=None) -> ExperimentResult:
    """Clone per fold; fit feature statistics only on that fold's training data.

    model must be sklearn-cloneable and predict original-scale rent. output_dir
    defaults to the repository root and can isolate artifacts in tests.
    """
    train = load_train()
    folds = get_splits(train, split)
    prototype = FeaturePipeline(features)
    model_template = _new_model(model)
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    root = Path(output_dir) if output_dir is not None else ROOT_DIR
    run_dir = root / "experiments" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "timestamp": timestamp.isoformat(), "run_id": run_id, "name": name, "author": author,
        "features": features, "feature_parameters": {
            key: _json_value(block.get_params(deep=False)) for key, block in prototype.blocks
        },
        "model": _class_name(model_template),
        "parameters": _json_value(model_template.get_params(deep=False)),
        "split": split, "seed": SEED, "final": final,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "folds": [],
    }
    scores = {}
    validation_predictions = []
    for fold in folds:
        _seed()
        training = train.iloc[fold.train_idx].copy()
        validation = train.iloc[fold.valid_idx].copy()
        pipeline = FeaturePipeline(features)
        X_train = pipeline.fit_transform(training)
        X_valid = pipeline.transform(validation)
        fitted_model = _new_model(model_template)
        fitted_model.fit(X_train, training[TARGET])
        prediction = _predictions(fitted_model, X_valid)
        rmse = float(np.sqrt(np.mean((validation[TARGET].to_numpy() - prediction) ** 2)))
        fold_name = fold.definition.name
        scores[fold_name] = rmse
        config["folds"].append({**asdict(fold.definition), "train_rows": len(training),
                                "valid_rows": len(validation), "rmse": rmse,
                                "feature_columns": list(X_train.columns)})
        validation_predictions.append(pd.DataFrame({
            "row_index": fold.valid_idx, "fold": fold_name,
            DATE_COL: validation[DATE_COL].to_numpy(),
            TARGET: validation[TARGET].to_numpy(), "Predicted": prediction,
        }))
        print(f"{name} [{split}/{fold_name}] RMSE = {rmse:.3f}", flush=True)
    pd.concat(validation_predictions, ignore_index=True).to_csv(run_dir / "validation_predictions.csv", index=False)
    mean_rmse = float(np.mean(list(scores.values())))
    submission_path = None
    if final:
        _seed()
        test = load_test()
        pipeline = FeaturePipeline(features)
        X_train = pipeline.fit_transform(train)
        X_test = pipeline.transform(test)
        fitted_model = _new_model(model_template)
        fitted_model.fit(X_train, train[TARGET])
        submission_path = write_submission(test, _predictions(fitted_model, X_test),
                                           root / "submissions" / f"{run_id}.csv")
        config["final_train_rows"] = len(train)
        config["test_rows"] = len(test)
        config["submission_path"] = str(submission_path)
    config["fold_rmse"] = scores
    config["mean_rmse"] = mean_rmse
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + "\n", encoding="utf-8")
    row = {key: config[key] for key in LOG_COLUMNS}
    for key in ("features", "parameters", "fold_rmse"):
        row[key] = json.dumps(row[key], ensure_ascii=False, allow_nan=False)
    _append_log(root / "experiments" / "log.csv", row)
    return ExperimentResult(run_id, scores, mean_rmse, run_dir, submission_path)
