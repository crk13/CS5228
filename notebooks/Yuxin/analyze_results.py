"""Build a compact, inspectable score table from saved prediction files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    records, monthly = [], []
    for root in args.roots:
        for path in sorted((root / "experiments").glob("*/config.json")):
            config = json.loads(path.read_text())
            valid = pd.read_csv(path.parent / "validation_predictions.csv")
            scores = {}
            for fold, part in valid.groupby("fold", sort=False):
                scores[fold] = float(np.sqrt(np.mean((part.MONTHLY_RENT-part.Predicted)**2)))
                for month, group in part.groupby("RENT_APPROVAL_DATE"):
                    monthly.append({"model": config["yuxin"]["model_name"], "fold": fold, "month": month,
                                    "rows": len(group), "rmse": float(np.sqrt(np.mean((group.MONTHLY_RENT-group.Predicted)**2))),
                                    "bias": float(np.mean(group.Predicted-group.MONTHLY_RENT))})
            records.append({"model": config["yuxin"]["model_name"], "run_id": config["run_id"],
                            "holdout_rmse": scores["holdout"], "rolling_2024_04_rmse": scores.get("rolling_2024_04"),
                            "mean_rmse": float(np.mean(list(scores.values()))),
                            "seconds": config["yuxin"]["seconds"], "output_root": str(root)})
    output = Path(__file__).parent
    pd.DataFrame(records).sort_values("mean_rmse").to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_errors.csv", index=False)
    print(pd.DataFrame(records).sort_values("mean_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
