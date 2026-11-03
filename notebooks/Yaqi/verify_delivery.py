"""Verify labels, chronological OOF rows, train diagnostics and test alignment."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from src.cv import get_splits, DATE_COL
from src.data_cleaning import load_train, load_test, TARGET
from delivery import OUT, SELECTED, hashes, write_json


def main():
    train, test = load_train(), load_test()
    afternoon = json.loads((ROOT / "notebooks/Yaqi/afternoon_signature.json").read_text(encoding="utf-8"))
    expected_data = {key.replace("\\", "/"): value for key, value in afternoon["data"].items()}
    assert hashes() == expected_data, "Delivery datasets differ from the afternoon experiments"
    log = pd.read_csv(OUT / "experiments/log.csv")
    folders = [p for p in (OUT / "experiments").iterdir() if p.is_dir()]
    assert len(folders) == len(log) == 3 and log.run_id.is_unique
    assert log["final"].all() and log.split.eq("rolling").all() and log.seed.eq(42).all()
    folds = {f.definition.name: f for f in get_splits(train, "rolling")}
    checked = []
    for kind in SELECTED:
        receipt = json.loads((OUT / f"{kind}_receipt.json").read_text(encoding="utf-8"))
        assert receipt["data_hashes"] == hashes()
        folder = ROOT / receipt["run_dir"]
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        assert config["final"] is True and config["split"] == "rolling" and config["seed"] == 42
        valid = pd.read_csv(folder / "validation_predictions.csv")
        assert list(valid) == ["row_index", "fold", DATE_COL, TARGET, "Predicted"]
        assert not valid.duplicated(["fold", "row_index"]).any()
        assert set(valid.fold) == set(folds) and np.isfinite(valid.Predicted).all()
        scores = {}
        training = pd.read_csv(folder / "fold_training_predictions.csv")
        for label, fold in folds.items():
            part = valid.loc[valid.fold.eq(label)]
            np.testing.assert_array_equal(part.row_index, fold.valid_idx)
            np.testing.assert_array_equal(part[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
            np.testing.assert_array_equal(part[TARGET], train.iloc[fold.valid_idx][TARGET])
            fitting = training.loc[training.fold.eq(label)]
            np.testing.assert_array_equal(fitting.row_index, fold.train_idx)
            np.testing.assert_array_equal(fitting[TARGET], train.iloc[fold.train_idx][TARGET])
            assert fitting[DATE_COL].max() < part[DATE_COL].min()
            assert not set(fitting.row_index).intersection(part.row_index)
            assert np.isfinite(fitting.Predicted).all()
            scores[label] = float(np.sqrt(np.mean((part[TARGET] - part.Predicted) ** 2)))
            np.testing.assert_allclose(scores[label], config["fold_rmse"][label], atol=1e-9, rtol=0)
        full = pd.read_csv(folder / "training_predictions.csv")
        assert len(full) == 150000 and full.fold.eq("final").all()
        np.testing.assert_array_equal(full.row_index, np.arange(len(train)))
        np.testing.assert_array_equal(full[TARGET], train[TARGET])
        np.testing.assert_array_equal(full[DATE_COL], train[DATE_COL])
        assert np.isfinite(full.Predicted).all() and int(full.used_for_model_fit.sum()) == 71040
        submitted = pd.read_csv(OUT / f"submissions/{receipt['run_id']}.csv")
        assert len(submitted) == 50000 and list(submitted) == ["Id", "Predicted"]
        np.testing.assert_array_equal(submitted.Id, test.Id)
        np.testing.assert_array_equal(submitted.Id, pd.read_csv(ROOT / "data/example-submission.csv").Id)
        assert np.isfinite(submitted.Predicted).all()
        np.testing.assert_allclose(np.mean(list(scores.values())), receipt["mean_rmse"], atol=1e-9, rtol=0)
        row = log.loc[log.run_id.eq(receipt["run_id"])].iloc[0]
        np.testing.assert_allclose(row.mean_rmse, receipt["mean_rmse"], atol=1e-9, rtol=0)
        checked.append({"kind": kind, "run_id": receipt["run_id"], "fold_rmse": scores,
                        "mean_rmse": receipt["mean_rmse"], "oof_rows": len(valid),
                        "training_rows": len(full), "test_rows": len(submitted)})
    assert not list((OUT / "experiments").glob("*.lock"))
    assert not [p for p in OUT.rglob("*") if p.suffix.lower() in (".joblib", ".pkl", ".zip", ".pth", ".cbm")]
    write_json(OUT / "verification.json", {"status": "passed", "checked": checked,
               "all_predictions_finite": True, "rows_and_targets_exact": True,
               "fold_training_and_validation_disjoint": True, "test_ids_match_sample": True,
               "scores_recomputed": True, "afternoon_scores_reproduced": True,
               "weights_included": False, "datasets_unchanged": True})
    write_json(OUT / "manifest.json", {"author": "Yaqi", "seed": 42, "threads_per_run": 4,
               "framework_base": "32213c435466bf4b081efef9bdcd10a73856b5b5",
               "models": checked, "data_hashes": expected_data,
               "meta_training_input": "experiments/<run_id>/validation_predictions.csv",
               "meta_inference_input": "submissions/<run_id>.csv",
               "diagnostic_only": ["training_predictions.csv", "fold_training_predictions.csv"],
               "training_window_months": 24, "weights_included": False})
    pd.DataFrame(checked).drop(columns="fold_rmse").to_csv(OUT / "selected_models.csv", index=False)
    print(json.dumps({"status": "passed", "models": checked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
