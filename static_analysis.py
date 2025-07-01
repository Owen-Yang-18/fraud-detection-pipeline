#!/usr/bin/env python3
"""
Per-class summary statistics + multi-subplot histograms for a CSV dataset.

Key points
----------
* Assumes the **last column** in the CSV is the class label.
* All other numeric columns are treated as features.
* For each feature, one figure (PNG) is produced:
    └─ contains a subplot histogram for every class.
* Bins are automatic (`numpy.histogram_bin_edges(..., bins='auto')`) unless
  you supply `--bins N`.

Usage examples
--------------
python class_histograms.py data.csv
python class_histograms.py data.csv --bins 40 --outdir plots --show
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class histograms & stats")
    p.add_argument("csv", help="Path to input CSV file")
    p.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Manual bin count (omit for automatic bins per class)",
    )
    p.add_argument(
        "--outdir",
        default="plots",
        help="Directory to write histogram PNGs (default: plots/)",
    )
    p.add_argument(
        "--stats_file",
        default=None,
        help="Where to save mean±std table (default: <outdir>/stats.csv)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display figures instead of saving PNGs",
    )
    p.add_argument(
        "--max_cols",
        type=int,
        default=3,
        help="Maximum subplot columns in the grid (default: 3)",
    )
    return p.parse_args()


def automatic_bins(data: np.ndarray, manual_bins: int | None) -> np.ndarray | int:
    if manual_bins is not None:
        return manual_bins
    return np.histogram_bin_edges(data, bins="auto")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- Load data ----------
    df = pd.read_csv(args.csv)
    class_col = df.columns[-1]
    class_values = sorted(df[class_col].dropna().unique())
    n_classes = len(class_values)
    if n_classes == 0:
        raise ValueError("No class labels found in the last column.")

    # Identify numeric features
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if class_col in numeric_cols:  # exclude label even if numeric
        numeric_cols.remove(class_col)
    if not numeric_cols:
        raise ValueError("No numeric feature columns detected.")

    # ---------- Compute mean & std ----------
    stats = (
        df.groupby(class_col)[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
        .sort_index()
    )
    stats_path = Path(args.stats_file) if args.stats_file else outdir / "stats.csv"
    stats.to_csv(stats_path)
    print(f"\nSaved summary statistics → {stats_path.resolve()}\n")
    print(stats)

    # ---------- Histogram figures ----------
    for col in numeric_cols:
        # Determine subplot grid shape
        ncols = min(n_classes, args.max_cols)
        nrows = math.ceil(n_classes / ncols)

        # Create figure
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 3.5 * nrows),
            squeeze=False,
        )
        fig.suptitle(f"{col}: distributions by class", fontsize=14)

        for idx, cls in enumerate(class_values):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
            data = df.loc[df[class_col] == cls, col].dropna().values
            if data.size == 0:
                ax.axis("off")
                ax.set_title(f"Class {cls} (no data)")
                continue

            bins = automatic_bins(data, args.bins)
            ax.hist(data, bins=bins)
            ax.set_title(f"Class {cls}")
            ax.set_xlabel(col)
            ax.set_ylabel("Freq")

        # Hide any empty subplots (when n_classes % ncols != 0)
        for j in range(n_classes, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle

        if args.show:
            plt.show()
        else:
            out_path = outdir / f"{col}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved figure → {out_path}")


if __name__ == "__main__":
    main()
