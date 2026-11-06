"""P1 fusion inputs: every delivered run, read from the member's branch at a pinned commit.

Results are not merged into this branch; each source is read with `git show <commit>:<path>`, so
the inputs stay fixed even if a branch moves. Fetch the branches first (git fetch origin).

Run from the repo root:  python notebooks/P1/fusion_inputs.py   # load, check and summarise
"""

import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# member -> (branch it was delivered on, pinned commit, results directory)
SOURCES = {
    "P1": ("p1_knn", "976b403e7581cecae570d7a28092edbaf45ce378", "results/P1"),
    "FXY": ("p2-linear", "3a732dd67f71ca31087f2fd3f84d72fb056c9b74", "results/FXY"),
    "Yaqi": ("yaqi-results-20261103", "20632f60fa5988a29010d58c325babbd0e35bcd2", "results/Yaqi"),
    "Yuxin": ("main", "3fa9345aae90a1fb8ecca9301fd15640d9db591f", "results/Yuxin"),
}
KEY = ["fold", "row_index"]
FOLDS = ("holdout", "rolling_2024_04")


def _show(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", check=True).stdout


def _csv(commit, path):
    return pd.read_csv(io.StringIO(_show(commit, path)))


def list_runs():
    """One row per delivered run: member, run_id, name, logged fold scores and source commit."""
    rows = []
    for member, (branch, commit, folder) in SOURCES.items():
        for run in _csv(commit, f"{folder}/experiments/log.csv").to_dict("records"):
            rows.append({"label": f"{member}:{run['name']}", "member": member, "run_id": run["run_id"],
                         "name": run["name"], "branch": branch, "commit": commit[:7], "folder": folder,
                         **json.loads(run["fold_rmse"]), "mean_rmse": run["mean_rmse"]})
    return pd.DataFrame(rows)


def load_inputs(runs=None):
    """Validation predictions wide by (fold, row_index), plus targets/dates and test predictions by Id."""
    runs = list_runs() if runs is None else runs
    valid, test, base = {}, {}, None
    for run in runs.itertuples():
        commit = SOURCES[run.member][1]
        v = _csv(commit, f"{run.folder}/experiments/{run.run_id}/validation_predictions.csv").set_index(KEY).sort_index()
        if base is None:
            base = v[["RENT_APPROVAL_DATE", "MONTHLY_RENT"]]
        elif not (v.index.equals(base.index) and v["MONTHLY_RENT"].equals(base["MONTHLY_RENT"])):
            raise ValueError(f"{run.label}: validation rows or targets differ from the other runs")
        valid[run.label] = v["Predicted"]
        s = _csv(commit, f"{run.folder}/submissions/{run.run_id}.csv")
        if list(s.columns) != ["Id", "Predicted"] or not np.array_equal(s["Id"], np.arange(50000)):
            raise ValueError(f"{run.label}: submission must be Id 0..49999 then Predicted")
        test[run.label] = s["Predicted"].to_numpy()
    valid, test = pd.DataFrame(valid), pd.DataFrame(test)
    if not (np.isfinite(valid.to_numpy()).all() and np.isfinite(test.to_numpy()).all()):
        raise ValueError("Non-finite predictions in the inputs")
    return valid, base, test


def check(runs, valid, base):
    """Recomputed fold RMSE must match each run's own log."""
    y = base["MONTHLY_RENT"]
    for run in runs.itertuples():
        for fold in FOLDS:
            rmse = float(np.sqrt(((y.loc[fold] - valid[run.label].loc[fold]) ** 2).mean()))
            if not np.isclose(rmse, getattr(run, fold), atol=1e-6):
                raise ValueError(f"{run.label} {fold}: recomputed {rmse:.6f} != logged {getattr(run, fold):.6f}")


if __name__ == "__main__":
    runs = list_runs()
    valid, base, test = load_inputs(runs)
    check(runs, valid, base)
    print(runs[["label", "run_id", "branch", "commit", *FOLDS, "mean_rmse"]].round(3).to_string(index=False))
    print(f"\nvalidation rows: {base.groupby(level='fold').size().to_dict()} | test rows: {len(test)} | runs: {len(runs)}")
    print("All inputs aligned; recomputed RMSE matches every log.")
