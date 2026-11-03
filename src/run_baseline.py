"""Run both reference models on shared splits and produce one submission."""

from threadpoolctl import threadpool_limits

from src.experiment import run_experiment
from src.models import group_median_model, hist_gradient_boosting_model


def main():
    results = []
    with threadpool_limits(limits=4):
        for name, features, model in (
            ("recent_group_median", [], group_median_model()),
            ("hist_gradient_boosting", ["basic", "block"], hist_gradient_boosting_model()),
        ):
            for split in ("holdout", "rolling"):
                result = run_experiment(name, "baseline", features, model, split=split,
                                        final=name == "hist_gradient_boosting" and split == "rolling")
                results.append((name, split, result))
    print("\nBaseline summary (original rent scale, SGD):")
    for name, split, result in results:
        print(f"{name:24s} {split:8s} mean RMSE = {result.mean_rmse:.3f}")
        if result.submission_path:
            print(f"Submission: {result.submission_path}")


if __name__ == "__main__":
    main()
