"""Recheck stages 4-6 and lock the three formal candidates without overwriting runs."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import ROOT, write_json
from verify_delivery import verify_root, verify_shared_property

OUT = ROOT / "notebooks/Yuxin/stage4"
ANALYSIS = ROOT / "notebooks/Yuxin/stage5"
DELIVERY = ROOT / "results/Yuxin"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_temporal():
    roots = [ROOT / "notebooks/Yuxin/tuning/final", OUT / "full", DELIVERY]
    reports = [(root, verify_root(root, delivery=root == DELIVERY)) for root in roots]
    shared = verify_shared_property(reports)
    assert shared["checked"] and "market_transformer" in shared["max_abs_ratio_deltas"]
    controlled = json.loads((OUT / "controlled_comparison.json").read_text())
    selected = next(c for c in controlled["candidates"] if c["candidate"] == controlled["selected"])
    full_record = reports[1][1]["checked"][0]
    full_folder = OUT / "full/experiments" / full_record["run_id"]
    full_config = json.loads((full_folder / "config.json").read_text())
    assert sha(OUT / "configs/market_transformer.json") == full_config["yuxin"]["recipe_sha256"]
    for file, expected in full_config["yuxin"]["model_source_hashes"].items():
        assert sha(ROOT / file) == expected, "New temporal run must match its recorded model source"
    full = pd.read_csv(full_folder / "validation_predictions.csv")
    carry_folder = ROOT / controlled["property_source"]
    assert sha(carry_folder / "validation_predictions.csv") == controlled["property_validation_sha256"]
    carry = pd.read_csv(carry_folder / "validation_predictions.csv")
    old_audits = json.loads((carry_folder / "training_audit.json").read_text())
    comparison = pd.read_csv(OUT / "comparison.csv").set_index("candidate")
    assert len(controlled["candidates"]) == len(comparison) == 12
    for candidate in controlled["candidates"]:
        scores = []
        for fit in candidate["fits"]:
            old = carry.loc[carry.fold.eq(fit["fold"])]
            before = next(a for a in old_audits if a["origin"] == fit["audit"]["origin"])
            ratio = old.Predicted.to_numpy()/old.RENT_APPROVAL_DATE.map(before["market_forecast"]).to_numpy()
            prediction = ratio*old.RENT_APPROVAL_DATE.map(fit["market_forecast"]).to_numpy()
            score = np.linalg.norm(prediction-old.MONTHLY_RENT.to_numpy())/np.sqrt(len(old))
            np.testing.assert_allclose(score, candidate[fit["fold"]], rtol=0, atol=1e-8)
            np.testing.assert_allclose(score, comparison.loc[candidate["candidate"], fit["fold"]], rtol=0, atol=1e-8)
            scores.append(score)
        np.testing.assert_allclose(np.mean(scores), candidate["mean_rmse"], rtol=0, atol=1e-8)
    deltas = []
    for fit in selected["fits"]:
        rows = full.loc[full.fold.eq(fit["fold"])]
        old = carry.loc[carry.fold.eq(fit["fold"])]
        np.testing.assert_array_equal(rows.row_index, old.row_index)
        origin = fit["audit"]["origin"]
        before = next(a for a in old_audits if a["origin"] == origin)
        ratio = old.Predicted.to_numpy()/old.RENT_APPROVAL_DATE.map(before["market_forecast"]).to_numpy()
        expected = ratio*rows.RENT_APPROVAL_DATE.map(fit["market_forecast"]).to_numpy()
        np.testing.assert_allclose(rows.Predicted, expected, atol=1e-8, rtol=0)
        deltas.append(float(np.max(np.abs(rows.Predicted-expected))))
    np.testing.assert_allclose(full_record["mean_rmse"], selected["mean_rmse"], atol=1e-8, rtol=0)
    write_json(OUT / "verification.json", dict(status="passed", roots={str(p.relative_to(ROOT)): r for p, r in reports},
               shared_property=shared, independent_retrain_matches_controlled_predictions=True,
               maximum_prediction_delta_sgd=max(deltas), all_12_controlled_scores_recomputed=True,
               selected_candidate=controlled["selected"],
               temporal_transformer=full_record))
    return full_record


def check_analysis():
    receipt = json.loads((ANALYSIS / "verification.json").read_text())
    assert receipt["status"] == "passed" and receipt["aligned_models"] == 9
    source = {s["model"]: s for s in receipt["sources"]}
    def load(name, test=False):
        s = source[name]
        path_key = "test_path" if test else "validation_path"
        if "commit" in s:
            raw = subprocess.check_output(["git", "show", s["commit"]+":"+s[path_key]], cwd=ROOT)
        else:
            raw = (ROOT / s[path_key]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == s["test_sha256" if test else "validation_sha256"]
        import io
        return pd.read_csv(io.BytesIO(raw))
    a, b = load("market_rnn"), load("catboost")
    # Independent join and direct vector arithmetic, not the reporting helpers.
    aligned = a.merge(b, on=["fold", "row_index"], suffixes=("_n", "_t"), validate="one_to_one")
    assert len(aligned) == 82329
    np.testing.assert_array_equal(aligned.MONTHLY_RENT_n, aligned.MONTHLY_RENT_t)
    np.testing.assert_array_equal(aligned.RENT_APPROVAL_DATE_n, aligned.RENT_APPROVAL_DATE_t)
    saved = pd.read_csv(ANALYSIS / "diagnostic_blend_validation.csv")
    joined = saved.merge(aligned, on=["fold", "row_index"], validate="one_to_one")
    np.testing.assert_allclose(joined.Predicted, .5*(joined.Predicted_n+joined.Predicted_t), rtol=0, atol=1e-9)
    blend_table = pd.read_csv(ANALYSIS / "fixed_blends.csv")
    base_scores = pd.read_csv(ANALYSIS / "fold_scores.csv")
    cache = {name: load(name) for name in receipt["formal_fusion_input_models"]}
    correlation_cache = {s["model"]: load(s["model"]) for s in receipt["sources"]}
    correlations = pd.read_csv(ANALYSIS / "residual_correlations.csv")
    assert len(correlations) == 72
    for row in correlations.itertuples():
        left, right = correlation_cache[row.model_a], correlation_cache[row.model_b]
        pair = left.loc[left.fold.eq(row.fold)].merge(right.loc[right.fold.eq(row.fold)],
                   on=["fold", "row_index"], suffixes=("_a", "_b"), validate="one_to_one")
        assert len(pair) == row.rows
        ea = pair.Predicted_a.to_numpy()-pair.MONTHLY_RENT_a.to_numpy()
        eb = pair.Predicted_b.to_numpy()-pair.MONTHLY_RENT_b.to_numpy()
        ea, eb = ea-ea.mean(), eb-eb.mean()
        pearson = np.dot(ea, eb)/np.sqrt(np.dot(ea, ea)*np.dot(eb, eb))
        np.testing.assert_allclose(pearson, row.residual_pearson, rtol=0, atol=1e-12)
    for row in blend_table.itertuples():
        left, right = cache[row.neural], cache[row.teammate]
        pair = left.merge(right, on=["fold", "row_index"], validate="one_to_one", suffixes=("_a", "_b"))
        scores = []
        for fold, part in pair.groupby("fold", sort=False):
            ea = part.Predicted_a.to_numpy()-part.MONTHLY_RENT_a.to_numpy()
            eb = part.Predicted_b.to_numpy()-part.MONTHLY_RENT_b.to_numpy()
            # MSE identity independently checks both bias and covariance terms.
            mse = (np.dot(ea, ea)+np.dot(eb, eb)+2*np.dot(ea, eb))/(4*len(part))
            actual = float(np.sqrt(mse))
            expected = row.holdout_rmse if fold == "holdout" else row.rolling_2024_04_rmse
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)
            scores.append(actual)
        np.testing.assert_allclose(np.mean(scores), row.mean_rmse, atol=1e-8, rtol=0)
    group = pd.read_csv(ANALYSIS / "group_errors.csv")
    assert len(group.model.unique()) == 21
    # Aggregated group squared errors must recover each full-fold MSE.
    for (name, fold, _), part in group.groupby(["model", "fold", "dimension"]):
        reference = base_scores.loc[base_scores.model.eq(name) & base_scores.fold.eq(fold)].iloc[0]
        assert part.rows.sum() == reference.rows
        np.testing.assert_allclose(np.sqrt(np.dot(part.rows, part.rmse**2)/part.rows.sum()), reference.rmse,
                                   atol=1e-8, rtol=0)
    for row in pd.read_csv(ANALYSIS / "forward_transfer.csv").itertuples():
        pair = cache[row.neural].merge(cache[row.teammate], on=["fold", "row_index"], validate="one_to_one", suffixes=("_a", "_b"))
        calibration = pair.loc[pair.fold.eq("holdout") & pair.RENT_APPROVAL_DATE_a.between("2023-12", "2024-03")]
        evaluation = pair.loc[pair.fold.eq("rolling_2024_04")]
        assert len(calibration) == 12013 and len(evaluation) == 35158
        assert calibration.RENT_APPROVAL_DATE_a.max() < evaluation.RENT_APPROVAL_DATE_a.min()
        direction = calibration.Predicted_a-calibration.Predicted_b
        weight = float(np.clip(np.dot(direction, calibration.MONTHLY_RENT_a-calibration.Predicted_b)/np.dot(direction, direction), 0, 1))
        np.testing.assert_allclose(weight, row.neural_weight, atol=1e-12, rtol=0)
        error = weight*evaluation.Predicted_a+(1-weight)*evaluation.Predicted_b-evaluation.MONTHLY_RENT_a
        np.testing.assert_allclose(np.sqrt(np.mean(error**2)), row.evaluation_rmse, atol=1e-8, rtol=0)
    test_a, test_b = load("market_rnn", test=True), load("catboost", test=True)
    submitted = pd.read_csv(ANALYSIS / "diagnostic_blend_test.csv")
    np.testing.assert_array_equal(submitted.Id, test_a.Id)
    np.testing.assert_array_equal(submitted.Id, test_b.Id)
    np.testing.assert_allclose(submitted.Predicted, .5*(test_a.Predicted+test_b.Predicted), atol=1e-9, rtol=0)
    write_json(ANALYSIS / "independent_verification.json", dict(status="passed", all_12_fixed_blends_recomputed=True,
               group_squared_errors_reconcile=True, all_72_correlations_recomputed=True,
               all_12_transfer_weights_and_scores_recomputed=True,
               saved_validation_and_test_blend_exact=True, hash_pinned_sources_checked=True,
               scope="arithmetic and temporal-weight boundary verified; base-model selection bias remains"))


def lock_delivery(temporal):
    records = verify_root(DELIVERY, delivery=True)["checked"]
    assert {r["model"] for r in records} == {"market_rnn", "market_lstm", "tabular_transformer"}
    # Predeclared promotion gate: a new time encoder must beat the same-property carry.
    carry = json.loads((ROOT / "notebooks/Yuxin/tuning/final/experiments/20261103T175001571910Z_a7a4588e/config.json").read_text())
    assert temporal["mean_rmse"] >= carry["mean_rmse"], "Reconsider selection: new encoder beat carry"
    selected = []
    for row in records:
        folder = DELIVERY / "experiments" / row["run_id"]
        config = json.loads((folder / "config.json").read_text())
        recipe_path = ROOT / "notebooks/Yuxin/tuning/configs" / f"{row['model']}.json"
        assert sha(recipe_path) == config["yuxin"]["recipe_sha256"]
        assert json.loads(recipe_path.read_text()) == config["yuxin"]["recipe"]
        matching_commits = []
        for commit in ("aba5078", "55d2acb"):
            if all(hashlib.sha256(subprocess.check_output(["git", "show", commit+":"+file], cwd=ROOT)).hexdigest() == expected
                   for file, expected in config["yuxin"]["model_source_hashes"].items()):
                matching_commits.append(subprocess.check_output(["git", "rev-parse", commit], cwd=ROOT, text=True).strip())
        assert matching_commits, "Training source snapshot must be recoverable exactly from Git"
        epochs = []
        for evidence in json.loads((folder / "training_audit.json").read_text()):
            state = evidence["audit"]
            epochs.append(dict(origin=evidence["origin"], available_rows=evidence["train_rows"],
                               property_selected_epochs=[m["selected_epochs"] for m in state["property"]["members"]],
                               market_selected_epochs=state["market"].get("selected_epochs", 0)))
        artifacts = [folder / "config.json", folder / "training_audit.json", folder / "validation_predictions.csv",
                     DELIVERY / config["submission_path"], recipe_path]
        selected.append(dict(**row, frozen_recipe=recipe_path.relative_to(ROOT).as_posix(),
                             exact_training_source_commit=matching_commits[0], actual_epochs=epochs,
                             artifact_sha256={f.relative_to(ROOT).as_posix(): sha(f) for f in artifacts}))
    write_json(DELIVERY / "selection.json", dict(status="locked_and_verified", max_formal_runs=3, selected=selected,
               policy="Retain the three verified Transformer/RNN/LSTM families; do not add a fourth run",
               epoch_policy="Frozen training hyperparameters and train-prefix time early stopping; fresh full-fold refit for selected epochs. Never choose epochs from outer validation.",
               temporal_transformer_not_promoted=dict(**temporal, reason="Did not beat matched-property carry; complete negative experiment retained in notebooks/Yuxin/stage4/full"),
               carry_control_mean_rmse=carry["mean_rmse"],
               fusion_diagnostics="notebooks/Yuxin/stage5; fixed-weight validation only, not an extra formal candidate",
               caveat="LSTM is nearly redundant with RNN, retained for assigned-family comparison rather than claimed unique fusion value; hidden Kaggle score unknown"))


if __name__ == "__main__":
    temporal = check_temporal()
    check_analysis()
    lock_delivery(temporal)
    print("Stages 4-6 verified; exactly three formal candidates locked.")
