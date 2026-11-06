"""P1: basic KNN through the shared experiment framework.

Run from the repo root:
  python notebooks/P1/knn_run.py grid     # holdout grid in a temp dir -> knn_grid_holdout.csv
  python notebooks/P1/knn_run.py deliver  # BEST on rolling + submission -> results/P1/
"""

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.config import ROOT_DIR
from src.experiment import run_experiment
from src.models.knn import KNN_FEATURES, knn_model

GRID = {
    "n_neighbors": [5, 10, 20, 40, 80],
    "weights": ["uniform", "distance"],
    "target_transform": ["none", "log"],
}
GRID_PATH = Path(__file__).with_name("knn_grid_holdout.csv")
BEST = {"n_neighbors": 80, "weights": "distance", "target_transform": "none"}  # lowest holdout RMSE in knn_grid_holdout.csv


def grid():
    output_dir = tempfile.mkdtemp(prefix="p1_knn_grid_")
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        result = run_experiment("knn_grid", "P1", KNN_FEATURES, knn_model(**params),
                                split="holdout", output_dir=output_dir)
        rows.append({**params, "holdout_rmse": result.mean_rmse})
    table = pd.DataFrame(rows).sort_values("holdout_rmse")
    table.to_csv(GRID_PATH, index=False)
    print(table.to_string(index=False))


def deliver():
    result = run_experiment("knn_basic", "P1", KNN_FEATURES, knn_model(**BEST),
                            split="rolling", final=True, output_dir=ROOT_DIR / "results" / "P1")
    print(result.fold_rmse, result.mean_rmse)
    print("run_id:", result.run_id)
    print("submission:", result.submission_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["grid", "deliver"])
    {"grid": grid, "deliver": deliver}[parser.parse_args().mode]()
