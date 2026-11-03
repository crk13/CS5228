"""FXY: market-ratio Ridge through the shared experiment framework (rolling split + submission).

Run from anywhere (project root recommended):
  python notebooks/FXY/p2_framework_run.py              # dry run into a temporary directory
  python notebooks/FXY/p2_framework_run.py --deliver    # delivery run into results/FXY/

The delivery run also writes training_predictions.csv and fold_training_predictions.csv
(diagnostics only) next to validation_predictions.csv of the new run.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from sklearn.base import clone

import src.features.mrt_distance  # noqa: F401  (registers "mrt_distance")
from src.config import SEED
from src.cv import DATE_COL, get_splits
from src.data_cleaning import TARGET, load_train
from src.experiment import run_experiment
from src.features import FeaturePipeline
from src.models import ModelWrapper
from src.models.market_ratio import MarketRatioRidge

NAME = "ridge_market_ratio_carry"
AUTHOR = "FXY"
FEATURES = ["block", "mrt_distance"]
COLUMNS = ["SUBZONE", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM",
           "RENT_YEAR", "LEASE_COMMENCE_DATE", "MRT_KM", "RENT_APPROVAL_DATE"]


def build_model():
    return ModelWrapper(MarketRatioRidge(), columns=COLUMNS, target_transform="none")


def _fit_and_predict_training(model, training):
    """Fit features and model on `training` (as the framework does) and predict those same rows."""
    pipeline = FeaturePipeline(FEATURES)
    X_train = pipeline.fit_transform(training)
    fitted = clone(model)
    fitted.set_params(**{k: SEED for k in fitted.get_params(deep=True) if k.split("__")[-1] == "random_state"})
    fitted.fit(X_train, training[TARGET])
    return fitted.predict(X_train)


def write_training_predictions(run_dir, model):
    """Final model (all train rows) and each fold's model predicting their own training rows."""
    train = load_train()
    out = lambda idx, fold, pred: pd.DataFrame({
        "row_index": idx, "fold": fold,
        DATE_COL: train[DATE_COL].to_numpy()[idx], TARGET: train[TARGET].to_numpy()[idx],
        "Predicted": pred, "used_for_model_fit": True,
    })
    all_idx = range(len(train))
    final = out(list(all_idx), "final", _fit_and_predict_training(model, train))
    final.to_csv(run_dir / "training_predictions.csv", index=False)
    parts = []
    for fold in get_splits(train, "rolling"):
        pred = _fit_and_predict_training(model, train.iloc[fold.train_idx].copy())
        parts.append(out(fold.train_idx, fold.definition.name, pred))
    pd.concat(parts, ignore_index=True).to_csv(run_dir / "fold_training_predictions.csv", index=False)


def relative_submission_path(run_dir):
    """Store the submission path relative to the repository root, not as a local absolute path."""
    path = run_dir / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["submission_path"] = Path(config["submission_path"]).relative_to(ROOT_DIR).as_posix()
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None, help="dry run: where experiments/ and submissions/ are written")
    parser.add_argument("--deliver", action="store_true", help="write the delivery run into results/FXY/")
    args = parser.parse_args()
    if args.deliver:
        output_dir = ROOT_DIR / "results" / "FXY"
    else:
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="p2_framework_")
    model = build_model()
    result = run_experiment(NAME, AUTHOR, FEATURES, model, split="rolling", final=True, output_dir=output_dir)
    if args.deliver:
        write_training_predictions(result.run_dir, model)
        relative_submission_path(result.run_dir)
    print(result.fold_rmse, result.mean_rmse)
    print("run_id:", result.run_id)
    print("output:", output_dir)
    print("submission:", result.submission_path)


if __name__ == "__main__":
    main()
