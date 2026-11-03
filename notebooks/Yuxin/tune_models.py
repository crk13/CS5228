"""TPE search on public chronological folds; no temporal Transformer or future inputs."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import SEED
from src.cv import get_splits
from src.data_cleaning import TARGET, load_train
from src.features import FeaturePipeline
from src.models.yuxin_nn import TabularNNRegressor
from src.models.yuxin_nn.market import MarketForecaster, monthly_market
from run_experiments import source_hashes, write_json

OUT = ROOT / "notebooks/Yuxin/tuning"
TREE_TARGET = 484.52140406824947
TREE_REFERENCE = {
    "commit": "3dae63c3ba9e5919fadb9af54474fee20d86f751",
    "path": "results/Yaqi/experiments/20261103T102108961798Z_15d233d7/config.json",
    "model": "CatBoost", "holdout": 492.87431277402214,
    "rolling_2024_04": 476.1684953624768, "seed": 42,
    "caveat": "same chronological folds; different features, random seed and environment; not retrained here",
}


def code_signature():
    # The new wrapper does not change the base learners that produced these caches.
    # Full-run provenance separately fingerprints every module, including ensemble.py.
    files = sorted(p for p in (ROOT / "src/models/yuxin_nn").glob("*.py") if p.name != "ensemble.py")
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def frames():
    raw = load_train()
    result = []
    for fold in get_splits(raw, "rolling"):
        a, b = raw.iloc[fold.train_idx].copy(), raw.iloc[fold.valid_idx].copy()
        pipeline = FeaturePipeline(["basic", "block"])
        result.append((fold, a, b, pipeline.fit_transform(a), pipeline.transform(b)))
    return result


def tabular_recipe(trial, phase):
    parameters = {
        "kind": "mlp" if phase == "carry" else "transformer",
        "market_normalize": True,
        "epochs": 60 if phase == "carry" else 24,
        "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]) if phase == "carry" else 128,
        "embedding_size": trial.suggest_categorical("embedding_size", [0, 8, 16]) if phase == "carry" else 0,
        "d_model": trial.suggest_categorical("d_model", [16, 32]) if phase == "transformer" else 32,
        "nhead": 4, "layers": 1,
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.05, 0.1]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
        "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.004, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.02, log=True),
        "min_frequency": trial.suggest_categorical("min_frequency", [1, 3, 5, 10]),
        "inner_months": trial.suggest_categorical("inner_months", [3, 6]),
        "patience": 7,
        "train_months": trial.suggest_categorical("train_months", [0, 18, 24, 36]) or None,
        "loss_weighting": trial.suggest_categorical("loss_weighting", ["uniform", "market"]),
        "threads": 1, "random_state": SEED,
    }
    return parameters


def temporal_recipe(trial, phase):
    return {
        "kind": phase, "lookback": trial.suggest_categorical("lookback", [3, 6, 12]),
        "hidden_size": trial.suggest_categorical("hidden_size", [8, 16, 32]),
        "epochs": trial.suggest_categorical("epochs", [80, 160, 240]),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01, log=True),
        "damping": trial.suggest_categorical("damping", [0.1, 0.3, 0.5, 0.8, 0.95]),
        "forecast_strength": trial.suggest_categorical("forecast_strength", [0.1, 0.25, 0.5, 1.0]),
        "inner_months": trial.suggest_categorical("inner_months", [6, 12]),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2]),
        "weight_decay": trial.suggest_float("weight_decay", 0.000001, 0.01, log=True),
        "random_state": SEED, "threads": 1,
    }


def queued_trials(phase):
    if phase in ("carry", "transformer"):
        values = [
            dict(dropout=0.0, batch_size=1024, learning_rate=0.002, weight_decay=0.0001,
                 min_frequency=3, inner_months=6, train_months=24, loss_weighting="uniform"),
            dict(dropout=0.05, batch_size=1024, learning_rate=0.0015, weight_decay=0.001,
                 min_frequency=1, inner_months=6, train_months=0, loss_weighting="market"),
            dict(dropout=0.05, batch_size=512, learning_rate=0.001, weight_decay=0.001,
                 min_frequency=5, inner_months=3, train_months=36, loss_weighting="uniform"),
            dict(dropout=0.0, batch_size=1024, learning_rate=0.003, weight_decay=0.00001,
                 min_frequency=1, inner_months=3, train_months=0, loss_weighting="uniform"),
        ]
        for i, value in enumerate(values):
            value.update(hidden_size=128 if i != 2 else 256, embedding_size=8 if i % 2 == 0 else 16) if phase == "carry" else value.update(d_model=16)
        return values
    return [dict(lookback=lookback, hidden_size=16, epochs=160, learning_rate=0.005,
                 damping=damping, forecast_strength=strength, inner_months=6,
                 dropout=0.1, weight_decay=0.01)
            for lookback, damping, strength in [(6, 0.1, 0.1), (3, 0.3, 0.5), (6, 0.5, 1.0), (12, 0.8, 0.5)]]


def export_study(study, phase, signature):
    records = []
    for trial in study.trials:
        if trial.state.name == "WAITING":
            continue
        records.append({"trial": trial.number, "state": trial.state.name, "value": trial.value,
                        "proposal": ("replay" if "replay_of" in trial.user_attrs else
                                     "queued" if trial.system_attrs.get("fixed_params") else "TPE"),
                        "suggested": trial.params, **trial.user_attrs})
    write_json(OUT / f"{phase}_trials.json", {"signature": signature, "seed": SEED,
               "method": "Optuna queued warm/local candidates + unqueued TPE; first-fold median pruning; inner temporal early stopping",
               "tree_target": TREE_TARGET, "tree_reference": TREE_REFERENCE,
               "data_hashes": study.user_attrs.get("data_hashes"),
               "property_identity": study.user_attrs.get("property_identity"),
               "selection_bias": "both public folds are development data",
               "trials": records})
    pd.DataFrame([{k:v for k,v in r.items() if k in ("trial", "state", "value", "seconds", "rmse_A", "rmse_B")}
                  for r in records]).to_csv(OUT / f"{phase}_trials.csv", index=False)
    complete = [t for t in study.trials if t.state.name == "COMPLETE"]
    if complete:
        best = min(complete, key=lambda t:t.value)
        write_json(OUT / f"best_{phase}.json", {"phase": phase, "trial": best.number,
                   "mean_rmse": best.value, "beats_best_tree": best.value < TREE_TARGET,
                   "parameters": best.user_attrs["recipe"], "fold_rmse": {
                       "holdout": best.user_attrs["rmse_A"], "rolling_2024_04": best.user_attrs["rmse_B"]},
                   "cache_files": best.user_attrs["cache_files"], "signature": signature,
                   "data_hashes": study.user_attrs.get("data_hashes"),
                   "property_identity": study.user_attrs.get("property_identity")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["carry", "transformer", "rnn", "lstm"], required=True)
    parser.add_argument("--trials", type=int, required=True, help="total non-waiting trial budget, resumable")
    parser.add_argument("--complete-pruned", action="store_true", help="Replay pruned temporal recipes without pruning")
    args = parser.parse_args()
    if args.complete_pruned and args.phase not in ("rnn", "lstm"):
        parser.error("Completing pruned recipes is reserved for the cheap temporal searches")
    OUT.mkdir(parents=True, exist_ok=True)
    signature = code_signature()
    cache = Path("/tmp") / ("yuxin_hpo_" + signature[:12])
    cache.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///" + str(cache / "studies.sqlite")
    study = optuna.create_study(study_name=args.phase, direction="minimize", storage=storage,
                               load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=4),
                               pruner=(optuna.pruners.NopPruner() if args.complete_pruned else
                                       optuna.pruners.MedianPruner(n_startup_trials=4)))
    if not study.trials:
        for recipe in queued_trials(args.phase):
            study.enqueue_trial(recipe)
    data = frames()
    before = source_hashes()
    if study.user_attrs.get("data_hashes", before) != before:
        raise ValueError("Input data changed since this study was created; start a new study")
    study.set_user_attr("data_hashes", before)
    property_best = None
    if args.phase in ("rnn", "lstm"):
        selected = OUT / "selected_carry.json"
        property_best = json.loads((selected if selected.exists() else OUT / "best_carry.json").read_text())
        if property_best["signature"] != signature:
            raise ValueError("Property cache source differs from current model code")
        if property_best.get("data_hashes") != before:
            raise ValueError("Property cache data hashes differ; re-export/rebuild the carry study")
        identity = hashlib.sha256(json.dumps(property_best["parameters"], sort_keys=True).encode()).hexdigest()
        if study.user_attrs.get("property_identity", identity) != identity:
            raise ValueError("Carry recipe changed; do not reuse a temporal study tied to another property model")
        study.set_user_attr("property_identity", identity)

    def objective(trial):
        recipe = tabular_recipe(trial, args.phase) if args.phase in ("carry", "transformer") else temporal_recipe(trial, args.phase)
        trial.set_user_attr("recipe", recipe)
        scores, caches, audits = [], [], []
        start = time.monotonic()
        for i, (fold, raw_a, raw_b, X_a, X_b) in enumerate(data):
            if property_best is None:
                with threadpool_limits(limits=2, user_api="blas"):
                    model = TabularNNRegressor(**recipe).fit(X_a, raw_a[TARGET])
                ratio = model.predict_ratio(X_b)
                prediction = model.predict(X_b)
                audit = model.audit_
            else:
                stored = np.load(property_best["cache_files"][i], allow_pickle=False)
                np.testing.assert_array_equal(stored["row_index"], fold.valid_idx)
                ratio = stored["ratio"]
                market = MarketForecaster(**recipe).fit(monthly_market(X_a, raw_a[TARGET]))
                prediction = ratio * market.predict_months(X_b["RENT_APPROVAL_DATE"])
                audit = market.audit_
            score = float(np.sqrt(np.mean((raw_b[TARGET].to_numpy()-prediction)**2)))
            scores.append(score)
            trial.set_user_attr("rmse_A" if i == 0 else "rmse_B", score)
            file = cache / f"{args.phase}_{trial.number}_{i}.npz"
            np.savez_compressed(file, prediction=prediction, ratio=ratio, row_index=fold.valid_idx)
            caches.append(str(file))
            audits.append(audit)
            trial.set_user_attr("cache_files", caches)
            trial.set_user_attr("audits", audits)
            trial.set_user_attr("seconds", time.monotonic()-start)
            print(f"{args.phase} trial={trial.number} fold={fold.definition.name} RMSE={score:.4f}", flush=True)
            if i == 0:
                trial.report(score, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        if source_hashes() != before:
            raise AssertionError("Data changed during hyperparameter search")
        return float(np.mean(scores))

    def callback(current, trial):
        export_study(current, args.phase, signature)

    done = sum(t.state.name not in ("WAITING", "RUNNING") for t in study.trials)
    remaining = max(0, args.trials-done)
    if args.complete_pruned:
        complete_params = {json.dumps(t.params, sort_keys=True) for t in study.trials if t.state.name == "COMPLETE"}
        pending = [t for t in study.trials if t.state.name == "PRUNED" and
                   json.dumps(t.params, sort_keys=True) not in complete_params]
        if any(t.state.name in ("RUNNING", "WAITING") for t in study.trials):
            raise ValueError("Finish existing trials before replaying pruned configurations")
        for trial in pending:
            study.enqueue_trial(trial.params, user_attrs={"replay_of": trial.number})
        remaining = len(pending)
        study.set_user_attr("temporal_pruned_replay_without_pruning", True)
    study.optimize(objective, n_trials=remaining, callbacks=[callback], gc_after_trial=True)
    export_study(study,args.phase,signature)
    print((OUT / f"best_{args.phase}.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
