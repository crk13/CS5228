"""Stage 5: aligned rolling errors and fusion diagnostics from committed results.

Fixed 50:50 blends do not fit weights. A separate forward-transfer diagnostic fits
one clipped least-squares weight on A's Dec 2023--Mar 2024 only and evaluates B's
Apr 2024--Mar 2025. Base models were already development-selected: neither
diagnostic is an untouched nested estimate or a Kaggle score.
"""

import hashlib
import io
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.cv import get_splits
from src.data_cleaning import load_train, load_test
from notebooks.Yuxin.run_experiments import source_hashes, write_json

OUT = ROOT / "notebooks/Yuxin/stage5"
TEAM = (
    ("ridge", "6e7ffcb0ccf20bacba42ca9d471084a3ab3aec71", "FXY", "20261103T110030459326Z_838f1908"),
    ("catboost", "3dae63c3ba9e5919fadb9af54474fee20d86f751", "Yaqi", "20261103T102108961798Z_15d233d7"),
    ("xgboost", "3dae63c3ba9e5919fadb9af54474fee20d86f751", "Yaqi", "20261103T102107575560Z_03bcf0a4"),
    ("lightgbm", "3dae63c3ba9e5919fadb9af54474fee20d86f751", "Yaqi", "20261103T102108253957Z_ef409512"),
)


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(prediction))**2)))


def align_prediction(frame, canonical):
    """One-to-one key join; fail on duplicates, missing rows, changed labels/dates."""
    keys = ["fold", "row_index"]
    if frame.duplicated(keys).any() or canonical.duplicated(keys).any():
        raise ValueError("Duplicate validation key")
    if len(frame) != len(canonical):
        raise ValueError("Validation population mismatch")
    joined = canonical.merge(frame, on=keys, how="left", validate="one_to_one", sort=False,
                             suffixes=("", "_source"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Missing validation keys")
    for column in ("RENT_APPROVAL_DATE", "MONTHLY_RENT"):
        np.testing.assert_array_equal(joined[column], joined[f"{column}_source"])
    prediction = joined.Predicted.to_numpy(float)
    if not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError("Predictions must be finite positive SGD")
    return prediction


def calibration_weight(y, neural, teammate):
    """Minimise calibration MSE of w*neural+(1-w)*teammate; no evaluation input."""
    difference = np.asarray(neural)-np.asarray(teammate)
    denominator = float(difference @ difference)
    return float(np.clip(difference @ (np.asarray(y)-np.asarray(teammate))/denominator, 0, 1)) if denominator else .5


def score_rows(frame, name, prediction):
    result = []
    for fold in frame.fold.unique():
        mask = frame.fold.eq(fold).to_numpy()
        error = prediction[mask]-frame.loc[mask, "MONTHLY_RENT"].to_numpy()
        result.append(dict(model=name, fold=fold, rows=int(mask.sum()),
                           rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                           mae=float(np.abs(error).mean())))
    return result


def group_rows(frame, name, prediction):
    rows = []
    for dimension in ("RENT_APPROVAL_DATE", "REGION", "TOWN", "FLAT_TYPE", "horizon", "horizon_bin"):
        for (fold, value), part in frame.groupby(["fold", dimension], dropna=False, sort=True):
            error = prediction[part.index]-part.MONTHLY_RENT.to_numpy()
            rows.append(dict(model=name, fold=fold, dimension=dimension, group=str(value), rows=len(part),
                             rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                             mae=float(np.abs(error).mean())))
    return rows


def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load_train(), load_test()
    parts = []
    for fold in get_splits(train, "rolling"):
        valid = train.iloc[fold.valid_idx]
        origin = pd.Period(train.iloc[fold.train_idx].RENT_APPROVAL_DATE.max(), freq="M")
        part = valid[["RENT_APPROVAL_DATE", "MONTHLY_RENT", "REGION", "TOWN", "FLAT_TYPE"]].copy()
        part.insert(0, "row_index", fold.valid_idx)
        part.insert(0, "fold", fold.definition.name)
        part["horizon"] = pd.PeriodIndex(part.RENT_APPROVAL_DATE, freq="M").asi8-origin.ordinal
        part["horizon_bin"] = pd.cut(part.horizon, [0, 4, 8, 12, 16], labels=["1-4", "5-8", "9-12", "13-16"]).astype(str)
        parts.append(part)
    canonical = pd.concat(parts, ignore_index=True)
    assert len(canonical) == 82329 and not canonical.duplicated(["fold", "row_index"]).any()
    predictions, submissions, sources = {}, {}, []
    for path in sorted((ROOT / "results/Yuxin/experiments").glob("*/config.json")):
        config = json.loads(path.read_text())
        assert config["split"] == "rolling" and config["final"]
        name = config["yuxin"]["model_name"]
        valid_path = path.parent / "validation_predictions.csv"
        test_path = ROOT / "results/Yuxin" / config["submission_path"]
        # Fusion only consumes results committed to the current branch.
        for file in (path, valid_path, test_path):
            assert git_bytes("HEAD", file.relative_to(ROOT).as_posix()) == file.read_bytes()
        result_commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", path.relative_to(ROOT).as_posix()],
            cwd=ROOT, text=True).strip()
        assert git_bytes(result_commit, valid_path.relative_to(ROOT).as_posix()) == valid_path.read_bytes()
        predictions[name] = align_prediction(pd.read_csv(valid_path), canonical)
        submitted = pd.read_csv(test_path)
        np.testing.assert_array_equal(submitted.Id, test.Id)
        submissions[name] = submitted.Predicted.to_numpy()
        sources.append(dict(model=name, commit=result_commit,
                            run_id=config["run_id"], validation_path=valid_path.relative_to(ROOT).as_posix(),
                            validation_sha256=hashlib.sha256(valid_path.read_bytes()).hexdigest(),
                            test_path=test_path.relative_to(ROOT).as_posix(),
                            test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(), seed=config["seed"]))
    neural_names = list(predictions)
    for name, commit, member, run_id in TEAM:
        folder = f"results/{member}"
        valid_path = f"{folder}/experiments/{run_id}/validation_predictions.csv"
        test_path = f"{folder}/submissions/{run_id}.csv"
        raw_valid, raw_test = git_bytes(commit, valid_path), git_bytes(commit, test_path)
        config = json.loads(git_bytes(commit, f"{folder}/experiments/{run_id}/config.json"))
        assert config["split"] == "rolling" and config["final"]
        predictions[name] = align_prediction(pd.read_csv(io.BytesIO(raw_valid)), canonical)
        submitted = pd.read_csv(io.BytesIO(raw_test))
        assert list(submitted) == ["Id", "Predicted"] and len(submitted) == 50000
        np.testing.assert_array_equal(submitted.Id, test.Id)
        assert np.isfinite(submitted.Predicted).all() and submitted.Predicted.gt(0).all()
        submissions[name] = submitted.Predicted.to_numpy()
        for score in score_rows(canonical, name, predictions[name]):
            np.testing.assert_allclose(score["rmse"], config["fold_rmse"][score["fold"]], rtol=0, atol=1e-8)
        sources.append(dict(model=name, commit=commit, run_id=run_id, validation_path=valid_path,
                            validation_sha256=hashlib.sha256(raw_valid).hexdigest(), test_path=test_path,
                            test_sha256=hashlib.sha256(raw_test).hexdigest(), seed=config["seed"]))
    formal_input_models = list(predictions)
    # Controls enter error analysis only, never cross-family fusion inputs.
    control_paths = [ROOT / "notebooks/Yuxin/tuning/final/experiments/20261103T175001571910Z_a7a4588e/config.json"]
    temporal_paths = list((ROOT / "notebooks/Yuxin/stage4/full/experiments").glob("*/config.json"))
    assert len(temporal_paths) == 1, "Complete the stage 4 independent full run before final analysis"
    control_paths.extend(temporal_paths)
    for path in control_paths:
        config = json.loads(path.read_text())
        assert config["split"] == "rolling" and config["final"]
        assert config["yuxin"]["data_hashes"] == source_hashes()
        name = config["yuxin"]["model_name"]
        file = path.parent / "validation_predictions.csv"
        predictions[name] = align_prediction(pd.read_csv(file), canonical)
        sources.append(dict(model=name, scope="error-analysis-only development control; excluded from fusion",
                            run_id=config["run_id"], validation_path=file.relative_to(ROOT).as_posix(),
                            validation_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), seed=config["seed"]))
    baseline_names = list(predictions)
    correlations = []
    for fold in canonical.fold.unique():
        keep = canonical.fold.eq(fold).to_numpy()
        y = canonical.loc[keep, "MONTHLY_RENT"].to_numpy()
        for a, b in itertools.combinations(baseline_names, 2):
            correlations.append(dict(fold=fold, model_a=a, model_b=b, rows=int(keep.sum()),
                                     residual_pearson=float(np.corrcoef(predictions[a][keep]-y, predictions[b][keep]-y)[0, 1])))
    blends, transfer = [], []
    calibration = (canonical.fold.eq("holdout") & canonical.RENT_APPROVAL_DATE.between("2023-12", "2024-03")).to_numpy()
    evaluation = canonical.fold.eq("rolling_2024_04").to_numpy()
    assert canonical.loc[calibration, "RENT_APPROVAL_DATE"].max() < canonical.loc[evaluation, "RENT_APPROVAL_DATE"].min()
    y = canonical.MONTHLY_RENT.to_numpy()
    for neural in neural_names:
        for teammate, *_ in TEAM:
            name = f"half_{neural}_{teammate}"
            predictions[name] = .5*predictions[neural]+.5*predictions[teammate]
            fold_scores = score_rows(canonical, name, predictions[name])
            a, b = [r["rmse"] for r in fold_scores]
            nmean = np.mean([r["rmse"] for r in score_rows(canonical, neural, predictions[neural])])
            tmean = np.mean([r["rmse"] for r in score_rows(canonical, teammate, predictions[teammate])])
            blends.append(dict(model=name, neural=neural, teammate=teammate, neural_weight=.5,
                               holdout_rmse=a, rolling_2024_04_rmse=b, mean_rmse=(a+b)/2,
                               neural_mean_rmse=nmean, teammate_mean_rmse=tmean,
                               improvement_vs_better_component=min(nmean, tmean)-(a+b)/2))
            weight = calibration_weight(y[calibration], predictions[neural][calibration], predictions[teammate][calibration])
            transferred = weight*predictions[neural][evaluation]+(1-weight)*predictions[teammate][evaluation]
            transfer.append(dict(neural=neural, teammate=teammate, neural_weight=weight,
                                 calibration_rows=int(calibration.sum()), calibration_start="2023-12", calibration_end="2024-03",
                                 evaluation_rows=int(evaluation.sum()), evaluation_start="2024-04", evaluation_end="2025-03",
                                 evaluation_rmse=rmse(y[evaluation], transferred),
                                 neural_evaluation_rmse=rmse(y[evaluation], predictions[neural][evaluation]),
                                 teammate_evaluation_rmse=rmse(y[evaluation], predictions[teammate][evaluation])))
    grouped, scores = [], []
    for name, prediction in predictions.items():
        grouped.extend(group_rows(canonical, name, prediction))
        scores.extend(score_rows(canonical, name, prediction))
    grouped = pd.DataFrame(grouped)
    for (_, fold, _), rows in grouped.groupby(["model", "fold", "dimension"]):
        assert rows.rows.sum() == canonical.fold.eq(fold).sum()
    grouped.to_csv(OUT / "group_errors.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "fold_scores.csv", index=False)
    pd.DataFrame(correlations).to_csv(OUT / "residual_correlations.csv", index=False)
    pd.DataFrame(blends).sort_values("mean_rmse").to_csv(OUT / "fixed_blends.csv", index=False)
    pd.DataFrame(transfer).to_csv(OUT / "forward_transfer.csv", index=False)
    # Prespecified illustrative pair, not another formal model or a tuned best blend.
    saved_name = "half_market_rnn_catboost"
    valid_output = canonical[["row_index", "fold", "RENT_APPROVAL_DATE", "MONTHLY_RENT"]].copy()
    valid_output["Predicted"] = predictions[saved_name]
    valid_output.to_csv(OUT / "diagnostic_blend_validation.csv", index=False)
    test_output = pd.DataFrame(dict(Id=test.Id, Predicted=.5*submissions["market_rnn"]+.5*submissions["catboost"]))
    assert len(test_output) == 50000 and np.isfinite(test_output.Predicted).all() and test_output.Predicted.gt(0).all()
    test_output.to_csv(OUT / "diagnostic_blend_test.csv", index=False)
    write_json(OUT / "verification.json", dict(status="passed", validation_rows=82329, test_rows=50000,
               aligned_models=len(baseline_names), formal_fusion_input_models=formal_input_models,
               fixed_blends=len(blends), forward_diagnostics=len(transfer),
               all_keys_dates_labels_exact=True, group_populations_reconciled=True, rolling_folds_kept_separate=True,
               no_training_fitted_predictions_used=True, transfer_evaluation_labels_used_for_weights=False,
               saved_blend=saved_name, data_hashes=source_hashes(), sources=sources,
               limitations=["Base-model hyperparameters already selected on outer folds; not untouched nested evaluation",
                            "A/B windows overlap; never merge their rows into one RMSE",
                            "Team seeds/environments/features differ", "12 pairs reported, not one cherry-picked winner",
                            "KNN unavailable in inspected branches; no invented correlation",
                            "Diagnostic blend is not an additional formal results/ run or a Kaggle upload"]))
    print(pd.DataFrame(blends).sort_values("mean_rmse").to_string(index=False))
    print(pd.DataFrame(transfer).to_string(index=False))


if __name__ == "__main__":
    main()
