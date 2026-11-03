"""Recompute scores and verify every saved prediction against the public contract."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.config import SEED
from src.cv import DATE_COL, get_splits
from src.data_cleaning import TARGET, load_train, load_test
from run_experiments import source_hashes, write_json

VALID_COLUMNS = ["row_index", "fold", DATE_COL, TARGET, "Predicted"]


def verify_root(output, *, delivery=False):
    output = Path(output)
    train, test = load_train(), load_test()
    splits = {f.definition.name: f for f in get_splits(train, "rolling")}
    fit_frames = {train.iloc[f.train_idx][DATE_COL].max(): train.iloc[f.train_idx] for f in splits.values()}
    fit_frames[train[DATE_COL].max()] = train
    current_hashes = source_hashes()
    configs = sorted((output / "experiments").glob("*/config.json"))
    if not configs or (delivery and len(configs) > 3):
        raise AssertionError("Expected 1-3 delivery runs, or at least one development run")
    log = pd.read_csv(output / "experiments/log.csv")
    assert len(log) == len(configs) and log.run_id.is_unique
    checked = []
    for path in configs:
        config = json.loads(path.read_text())
        assert config["split"] == "rolling" and config["final"] is True
        assert config["seed"] == SEED
        assert not config["yuxin"]["smoke_only"]
        assert config["yuxin"]["data_hashes"] == current_hashes
        valid = pd.read_csv(path.parent / "validation_predictions.csv")
        assert list(valid) == VALID_COLUMNS
        assert len(valid) == 82329 and set(valid.fold) == set(splits)
        assert not valid.duplicated(["fold", "row_index"]).any()
        assert np.isfinite(valid.Predicted).all() and valid.Predicted.gt(0).all()
        scores = {}
        for name, fold in splits.items():
            rows = valid.loc[valid.fold.eq(name)]
            np.testing.assert_array_equal(rows.row_index, fold.valid_idx)
            np.testing.assert_array_equal(rows[DATE_COL], train.iloc[fold.valid_idx][DATE_COL])
            np.testing.assert_array_equal(rows[TARGET], train.iloc[fold.valid_idx][TARGET])
            assert train.iloc[fold.train_idx][DATE_COL].max() < rows[DATE_COL].min()
            scores[name] = float(np.sqrt(np.mean((rows[TARGET] - rows.Predicted) ** 2)))
            np.testing.assert_allclose(scores[name], config["fold_rmse"][name], rtol=0, atol=1e-8)
        mean = float(np.mean(list(scores.values())))
        np.testing.assert_allclose(mean, config["mean_rmse"], rtol=0, atol=1e-8)
        row = log.loc[log.run_id.eq(config["run_id"])].iloc[0]
        np.testing.assert_allclose(mean, row.mean_rmse, rtol=0, atol=1e-8)
        submission = output / config["submission_path"]
        submitted = pd.read_csv(submission)
        assert list(submitted) == ["Id", "Predicted"] and len(submitted) == 50000
        np.testing.assert_array_equal(submitted.Id, test.Id)
        np.testing.assert_array_equal(submitted.Id, np.arange(50000))
        assert np.isfinite(submitted.Predicted).all() and submitted.Predicted.gt(0).all()
        audits = json.loads((path.parent / "training_audit.json").read_text())
        assert [a["origin"] for a in audits] == ["2023-11", "2024-03", "2025-03"]
        assert [a["train_rows"] for a in audits] == [102829, 114842, 150000]
        for audit in audits:
            state = audit["audit"]
            assert audit["inference_probe_max_abs_delta_sgd"] < 0.01
            if not state:  # public sklearn baselines
                continue
            assert state["origin"] == audit["origin"] and state["future_actuals_used"] is False
            prop = state.get("property", state)
            members = prop.get("members", [prop])
            if "members" in prop:
                assert prop["n_members"] == len(members) and prop["aggregation"] == "equal_weight_mean"
                assert prop["n_members"] == len(config["yuxin"]["recipe"]["parameters"]["members"])
            for member in members:
                inner = member["inner"]
                assert inner["enabled"] and inner["train_end"] < inner["valid_start"] <= inner["valid_end"]
                assert inner["valid_end"] <= audit["origin"]
                assert member["origin"] == audit["origin"] and member["future_actuals_used"] is False
                assert member["selected_epochs"] >= 1
                assert not {TARGET, "Id", "row_index", DATE_COL}.intersection(member["numerical"] + member["categorical"])
                fit_dates = pd.PeriodIndex(fit_frames[audit["origin"]][DATE_COL], freq="M")
                months = member.get("train_months")
                keep = np.ones(len(fit_dates), dtype=bool) if months is None else fit_dates >= fit_dates.max()-(months-1)
                assert member["train_rows"] == int(keep.sum())
                assert inner["train_rows"] + inner["valid_rows"] == member["train_rows"]
                if "available_rows" in member:
                    assert member["available_rows"] == audit["train_rows"]
                    assert member["fit_start"] == str(fit_dates[keep].min())
            if "market" in state:
                market = state["market"]
                assert market["origin"] == audit["origin"] and not market["future_actuals_used"]
                if market["kind"] in ("rnn", "lstm", "transformer"):
                    assert market["last_input_month"] < market["last_target_month"] <= audit["origin"]
                    assert market["inner_train_end"] < market["inner_valid_start"]
                    assert market["windows"] == market["months"] - market["lookback"]
                    if market["kind"] == "transformer":
                        assert market["attention_mask"] == "causal"
                        assert market["position_encoding"] == "window_relative_sinusoidal"
                        assert market["d_model"] % market["nhead"] == 0
        checked.append({"model": config["yuxin"]["model_name"], "run_id": config["run_id"],
                        "fold_rmse": scores, "mean_rmse": mean,
                        "validation_rows": len(valid), "test_rows": len(submitted),
                        "seconds": config["yuxin"]["seconds"]})
    assert not list(output.rglob("*.lock"))
    assert not [p for p in output.rglob("*") if p.suffix in (".pt", ".pth", ".pkl", ".joblib")]
    report = {"status": "passed", "seed": SEED, "delivery": delivery, "checked": checked,
              "all_validation_rows_labels_and_dates_exact": True, "scores_recomputed": True,
              "test_ids_exact": True, "all_predictions_finite_positive_sgd": True,
              "fold_local_training_audits_checked": True, "data_hashes_unchanged": True,
              "weights_included": False}
    write_json(output / "verification.json", report)
    return report


def verify_shared_property(reports):
    """Same static model across carry/RNN/LSTM: isolate the market encoder effect."""
    records = {}
    for output, report in reports:
        for record in report["checked"]:
            records[record["model"]] = (Path(output), record)
    if not {"market_carry", "market_rnn", "market_lstm"} <= records.keys():
        return {"checked": False, "reason": "need all three market models"}
    carry_root, carry_record = records["market_carry"]
    carry_dir = carry_root / "experiments" / carry_record["run_id"]
    reference = pd.read_csv(carry_dir / "validation_predictions.csv")
    reference_audits = json.loads((carry_dir / "training_audit.json").read_text())
    deltas = {}
    names = [name for name in ("market_rnn", "market_lstm", "market_transformer") if name in records]
    for name in names:
        output, record = records[name]
        folder = output / "experiments" / record["run_id"]
        rows = pd.read_csv(folder / "validation_predictions.csv")
        audits = json.loads((folder / "training_audit.json").read_text())
        for a, b in zip(reference_audits, audits):
            assert a["audit"]["property"] == b["audit"]["property"]
            assert a["property_training_history"] == b["property_training_history"]
        ratio_deltas = []
        for fold, origin in (("holdout", "2023-11"), ("rolling_2024_04", "2024-03")):
            a = next(a for a in reference_audits if a["origin"] == origin)
            b = next(a for a in audits if a["origin"] == origin)
            r = reference.loc[reference.fold.eq(fold)]
            m = rows.loc[rows.fold.eq(fold)]
            np.testing.assert_array_equal(r.row_index, m.row_index)
            ratio_a = r.Predicted.to_numpy() / r[DATE_COL].map(a["market_forecast"]).to_numpy()
            ratio_b = m.Predicted.to_numpy() / m[DATE_COL].map(b["market_forecast"]).to_numpy()
            np.testing.assert_allclose(ratio_a, ratio_b, rtol=0, atol=1e-10)
            ratio_deltas.append(float(np.max(np.abs(ratio_a-ratio_b))))
        test_months = load_test()[DATE_COL]
        a, b = reference_audits[-1], audits[-1]
        r = pd.read_csv(carry_root / "submissions" / f"{carry_record['run_id']}.csv")
        m = pd.read_csv(output / "submissions" / f"{record['run_id']}.csv")
        np.testing.assert_array_equal(r.Id, m.Id)
        ratio_a = r.Predicted.to_numpy() / test_months.map(a["market_forecast"]).to_numpy()
        ratio_b = m.Predicted.to_numpy() / test_months.map(b["market_forecast"]).to_numpy()
        np.testing.assert_allclose(ratio_a, ratio_b, rtol=0, atol=1e-10)
        ratio_deltas.append(float(np.max(np.abs(ratio_a-ratio_b))))
        deltas[name] = max(ratio_deltas)
    return {"checked": True, "same_property_training_and_predictions": True,
            "max_abs_ratio_deltas": deltas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, default=[ROOT / "results/Yuxin"])
    parser.add_argument("--summary-output", type=Path, default=Path(__file__).parent / "verification.json")
    args = parser.parse_args()
    reports = [(p, verify_root(p, delivery=p.resolve() == (ROOT / "results/Yuxin").resolve())) for p in args.roots]
    shared = verify_shared_property(reports)
    summary = {"status": "passed", "roots": {str(p): r for p, r in reports}, "shared_property": shared}
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
