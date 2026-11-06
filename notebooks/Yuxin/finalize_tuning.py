"""Materialize searched recipes and verify independent full-framework retraining."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiments import ROOT, write_json
from tune_models import OUT, TREE_TARGET, code_signature
from verify_delivery import verify_root, verify_shared_property

NAMES = {"carry": "market_carry", "transformer": "tabular_transformer",
         "rnn": "market_rnn", "lstm": "market_lstm"}


def prepare(phases=None):
    phases = phases or list(NAMES)
    best = {p: json.loads((OUT / f"{'selected' if p in ('carry', 'transformer') else 'best'}_{p}.json").read_text())
            for p in set(phases) | {"carry"}}
    assert all(b["signature"] == code_signature() for b in best.values())
    prop = best["carry"]["parameters"]
    identity = hashlib.sha256(json.dumps(prop, sort_keys=True).encode()).hexdigest()
    assert all(best[p]["property_identity"] == identity for p in phases if p in ("rnn", "lstm"))
    for phase in phases:
        name = NAMES[phase]
        if phase == "transformer":
            params = dict(best[phase]["parameters"], n_jobs=4)
        else:
            params = dict(prop, encoder=phase)
            if phase != "carry":
                params["temporal_parameters"] = best[phase]["parameters"]
        recipe = {"class": "ensemble", "parameters": params}
        recipe["search"] = {"phase": phase, "trial": best[phase]["trial"],
                            "development_mean_rmse": best[phase]["mean_rmse"],
                            "selection": best[phase].get("selection", "best COMPLETE temporal trial on frozen property model"),
                            "global_optimum_proven": False}
        write_json(OUT / "configs" / f"{name}.json", recipe)
    print("Prepared searched recipes for", phases, "in", OUT / "configs")


def check():
    root = OUT / "final"
    report = verify_root(root)
    shared = verify_shared_property([(root, report)])
    assert shared["checked"]
    records = {r["model"]: r for r in report["checked"]}
    assert set(records) == set(NAMES.values())
    rows = []
    for phase, name in NAMES.items():
        prefix = "selected" if phase in ("carry", "transformer") else "best"
        selected = json.loads((OUT / f"{prefix}_{phase}.json").read_text())
        result = records[name]
        np.testing.assert_allclose(result["mean_rmse"], selected["mean_rmse"], atol=1e-8, rtol=0)
        prediction = pd.read_csv(root / "experiments" / result["run_id"] / "validation_predictions.csv")
        deltas = []
        for fold, file in zip(("holdout", "rolling_2024_04"), selected["cache_files"]):
            cached = np.load(file, allow_pickle=False)
            current = prediction.loc[prediction.fold.eq(fold)]
            np.testing.assert_array_equal(current.row_index, cached["row_index"])
            delta = float(np.max(np.abs(current.Predicted.to_numpy()-cached["prediction"])))
            assert delta < 1e-8, (phase, fold, delta)
            deltas.append(delta)
        rows.append({"phase": phase, **result, "search_trial": selected["trial"],
                     "best_tree_rmse": TREE_TARGET, "beats_best_tree": result["mean_rmse"] < TREE_TARGET,
                     "improvement_vs_tree": TREE_TARGET-result["mean_rmse"],
                     "retraining_max_prediction_delta": max(deltas)})
    write_json(OUT / "final_verification.json", {"status": "passed", "shared_property": shared,
               "all_search_predictions_reproduced": True, "models": rows,
               "scope": "development folds only; Kaggle hidden-test score unknown"})
    pd.DataFrame(rows).sort_values("mean_rmse").to_csv(OUT / "comparison.csv", index=False)
    print(pd.DataFrame(rows)[["model", "mean_rmse", "improvement_vs_tree", "beats_best_tree"]].to_string(index=False))
    return rows


def publish():
    """Recoverably archive the known initial delivery, then copy the three neural families."""
    rows = check()  # Do not move the existing delivery unless all four new runs pass.
    delivery = ROOT / "results/Yuxin"
    initial_ids = {"20261103T165254207013Z_e7366de1", "20261103T165254212024Z_ae71e28f",
                   "20261103T165436355139Z_16342354"}
    assert {p.parent.name for p in (delivery / "experiments").glob("*/config.json")} == initial_ids
    verify_root(delivery, delivery=True)
    archive = ROOT / "notebooks/Yuxin/archive/stage1_3"
    assert not archive.exists(), "Archive exists: refusing to overwrite prior evidence"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.csv", "monthly_errors.csv", "verification.json"):
        shutil.copy2(ROOT / "notebooks/Yuxin" / name, archive / name)
    shutil.move(str(delivery), str(archive / "results_Yuxin"))
    (delivery / "experiments").mkdir(parents=True)
    (delivery / "submissions").mkdir()
    selected = [r for r in rows if r["model"] != "market_carry"]
    assert len(selected) == 3
    for record in selected:
        run_id = record["run_id"]
        shutil.copytree(OUT / "final/experiments" / run_id, delivery / "experiments" / run_id)
        shutil.copy2(OUT / "final/submissions" / f"{run_id}.csv", delivery / "submissions" / f"{run_id}.csv")
    log = pd.read_csv(OUT / "final/experiments/log.csv")
    log.loc[log.run_id.isin([r["run_id"] for r in selected])].to_csv(delivery / "experiments/log.csv", index=False)
    verify_root(delivery, delivery=True)
    write_json(OUT / "delivery_index.json", {"delivered": selected,
               "carry_control": next(r for r in rows if r["model"] == "market_carry"),
               "initial_delivery_archive": str(archive.relative_to(ROOT)),
               "reason": "three-run team limit; Transformer/RNN/LSTM delivered, carry control retained in tuning/final"})
    print("Archived initial results recoverably to", archive)
    print("Published Transformer/RNN/LSTM; carry remains in", OUT / "final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--publish", action="store_true", help="Archive known initial results and publish three verified families")
    parser.add_argument("--models", nargs="+", choices=list(NAMES), help="Prepare only these families")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.models)
    if args.check:
        check()
    if args.publish:
        publish()
