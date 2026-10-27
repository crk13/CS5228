"""P2: market-ratio Ridge through the shared experiment framework (rolling split + submission).

Run from the repo root:  python notebooks/p2_framework_run.py [--output-dir DIR]
Without --output-dir, artifacts go to a temporary directory.
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.features.mrt_distance  # noqa: F401  (registers "mrt_distance")
from src.experiment import run_experiment
from src.models import ModelWrapper
from src.models.market_ratio import MarketRatioRidge

COLUMNS = ["SUBZONE", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM",
           "RENT_YEAR", "LEASE_COMMENCE_DATE", "MRT_KM", "RENT_APPROVAL_DATE"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None, help="where experiments/ and submissions/ are written")
    args = parser.parse_args()
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="p2_framework_")
    model = ModelWrapper(MarketRatioRidge(), columns=COLUMNS, target_transform="none")
    result = run_experiment("ridge_market_ratio_carry", "P2", ["block", "mrt_distance"], model,
                            split="rolling", final=True, output_dir=output_dir)
    print(result.fold_rmse, result.mean_rmse)
    print("output:", output_dir)
    print("submission:", result.submission_path)


if __name__ == "__main__":
    main()
