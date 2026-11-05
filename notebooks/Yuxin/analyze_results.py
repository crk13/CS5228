"""Build a compact, inspectable score table from saved prediction files."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional report directory; useful for historical reproductions")
    args = parser.parse_args()
    records, monthly = [], []
    for root in args.roots:
        for path in sorted((root / "experiments").glob("*/config.json")):
            config = json.loads(path.read_text())
            variant = ("tuned_family_ensemble" if config["yuxin"].get("recipe", {}).get("class") == "ensemble"
                       else "initial")
            if variant == "tuned_family_ensemble" and root.name == "pre_replay":
                variant = "tuned_before_temporal_replay"
            valid = pd.read_csv(path.parent / "validation_predictions.csv")
            scores = {}
            for fold, part in valid.groupby("fold", sort=False):
                scores[fold] = float(np.sqrt(np.mean((part.MONTHLY_RENT-part.Predicted)**2)))
                for month, group in part.groupby("RENT_APPROVAL_DATE"):
                    monthly.append({"model": config["yuxin"]["model_name"], "variant": variant,
                                    "run_id": config["run_id"], "output_root": str(root),
                                    "fold": fold, "month": month,
                                    "rows": len(group), "rmse": float(np.sqrt(np.mean((group.MONTHLY_RENT-group.Predicted)**2))),
                                    "bias": float(np.mean(group.Predicted-group.MONTHLY_RENT))})
            records.append({"model": config["yuxin"]["model_name"], "variant": variant, "run_id": config["run_id"],
                            "holdout_rmse": scores["holdout"], "rolling_2024_04_rmse": scores.get("rolling_2024_04"),
                            "mean_rmse": float(np.mean(list(scores.values()))),
                            "seconds": config["yuxin"]["seconds"], "output_root": str(root)})
    output = args.output or Path(__file__).parent
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("mean_rmse").to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(monthly).to_csv(output / "monthly_errors.csv", index=False)
    print(pd.DataFrame(records).sort_values("mean_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
