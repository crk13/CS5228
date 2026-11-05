"""Build the neural-network evaluation figure for the report (P4 / Yuxin).

Two panels, both on the shared rolling folds:
  (a) observed monthly median rent versus the market level forecast by the four
      encoder families (carry, LSTM, RNN, temporal Transformer), fold B origin;
  (b) mean RMSE of the twelve controlled encoder candidates against carry.

Run from the repository root with a python that has matplotlib and pandas:
    python doc/figures/make_p4_encoder_figure.py
Inputs are the frozen run artefacts, nothing is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LSTM_RUN = "results/Yuxin/experiments/20261103T180702083101Z_b8e1e8c0"
RNN_RUN = "results/Yuxin/experiments/20261103T180702097497Z_0c01ed2c"
TRF_RUN = "notebooks/Yuxin/stage4/full/experiments/20261103T190543556826Z_a02818be"
COMPARISON = "notebooks/Yuxin/stage4/comparison.csv"
ORIGIN = "2024-03"


def forecast(path: Path, origin: str = ORIGIN) -> dict[str, float]:
    for item in json.loads(path.read_text(encoding="utf-8")):
        if item["origin"] == origin:
            return item["market_forecast"]
    raise LookupError(f"no origin {origin} in {path}")


def observed_median() -> pd.Series:
    frame = pd.read_csv(ROOT / "data_cleaned" / "train.csv", low_memory=False)
    month = pd.to_datetime(frame["RENT_APPROVAL_DATE"]).dt.to_period("M")
    return frame.groupby(month)["MONTHLY_RENT"].median()


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.5,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
    })

    observed = observed_median()
    carry_level = float(observed[pd.Period(ORIGIN, freq="M")])
    paths = {
        "LSTM": forecast(ROOT / LSTM_RUN / "training_audit.json"),
        "RNN": forecast(ROOT / RNN_RUN / "training_audit.json"),
        "Transformer": forecast(ROOT / TRF_RUN / "training_audit.json"),
    }
    months = sorted({m for path in paths.values() for m in path})
    validation = [m for m in months if m > ORIGIN][:12]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.35, 3.55), constrained_layout=True)

    observed_window = observed.loc[[pd.Period(m, freq="M") for m in validation]]
    ax1.plot(range(len(validation)), observed_window.to_numpy(), color="black",
             marker="o", markersize=2.2, linewidth=1.1, label="observed")
    ax1.plot(range(len(validation)), [carry_level] * len(validation),
             color="0.55", linestyle="--", linewidth=1.0, label="carry")
    for (name, path), colour in zip(paths.items(), ["#1f5fa8", "#2f7d32", "#c0501e"]):
        ax1.plot(range(len(validation)), [path[m] for m in validation],
                 color=colour, linewidth=1.0, label=name)
    ax1.set_ylabel("monthly median rent (SGD)")
    ax1.set_xticks(range(len(validation)))
    ax1.set_xticklabels([m[2:] for m in validation], rotation=60)
    ax1.set_title("(a) fold B market level, fixed origin", fontsize=7.5, loc="left")
    ax1.set_ylim(3085, 3275)
    ax1.legend(ncol=3, frameon=False, loc="upper center", handlelength=1.3,
               columnspacing=0.7, handletextpad=0.4)

    comparison = pd.read_csv(ROOT / COMPARISON).sort_values("mean_rmse")
    labels = (comparison["candidate"]
              .str.replace("matched_", "", regex=False)
              .str.replace("transformer_", "trf ", regex=False)
              .str.replace("_", " ", regex=False))
    ax2.barh(range(len(comparison)), comparison["mean_rmse"], height=0.62,
             color="#4a6fa5", label="candidate")
    ax2.axvline(comparison["mean_rmse"].iloc[0], visible=False)
    carry_rmse = float(comparison.loc[comparison["candidate"] == "matched_carry", "mean_rmse"].iloc[0])
    ax2.axvline(carry_rmse, color="0.35", linestyle="--", linewidth=1.0, label="carry")
    ax2.set_yticks(range(len(comparison)))
    ax2.set_yticklabels(labels, fontsize=6.0)
    ax2.set_xlim(carry_rmse - 0.15, comparison["mean_rmse"].max() + 0.08)
    ax2.set_xlabel("mean RMSE over folds A and B (SGD)")
    ax2.set_title("(b) twelve controlled encoder candidates", fontsize=7.5, loc="left")
    ax2.legend(frameon=False, loc="lower right", handlelength=1.4)

    for ax in (ax1, ax2):
        ax.tick_params(width=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.savefig(OUT / "p4_encoder_evaluation.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "p4_encoder_evaluation.png", dpi=220, bbox_inches="tight", pad_inches=0.02)

    spread = max(
        abs(path[m] - carry_level) for path in paths.values() for m in validation
    )
    print(f"carry level at {ORIGIN}: {carry_level:.0f}")
    print(f"observed range over fold B: {observed_window.min():.0f}-{observed_window.max():.0f}")
    print(f"largest encoder deviation from carry: {spread:.1f} SGD")
    print(f"carry mean RMSE: {carry_rmse:.4f}; worst candidate: {comparison['mean_rmse'].max():.4f}")


if __name__ == "__main__":
    main()
