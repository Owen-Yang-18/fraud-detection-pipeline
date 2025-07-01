#!/usr/bin/env python3
"""
Per-class summary statistics, multi-subplot histograms, and feature-separation
ranking for a CSV dataset.

Assumptions
-----------
* The CSV's **first row** holds column names.
* The **last column** is the class label; all other numeric columns are features.
"""

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ───────────────────────────────────────── CLI ──────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class histograms & stats")
    p.add_argument("csv", help="Path to input CSV file")

    p.add_argument("--bins", type=int, default=None,
                   help="Manual bin count (omit for automatic bins)")
    p.add_argument("--metric", choices=["js", "kl", "tv"], default="js",
                   help="js (default) = Jensen–Shannon, kl = symmetric KL, tv = Total Variation")
    p.add_argument("--outdir", default="plots",
                   help="Directory to write PNGs + CSVs (default: plots/)")
    p.add_argument("--stats_file", default=None,
                   help="Custom path for mean±std CSV (default: <outdir>/stats.csv)")
    p.add_argument("--rank_file", default=None,
                   help="Custom path for feature-ranking CSV (default: <outdir>/feature_ranking.csv)")
    p.add_argument("--show", action="store_true",
                   help="Display figures instead of saving PNGs")
    p.add_argument("--max_cols", type=int, default=3,
                   help="Max subplot columns in a grid (default: 3)")
    return p.parse_args()


# ───────────────────────────── distance helpers (JS / KL / TV) ──────────────────
_EPS = 1e-12  # avoid log(0)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()


def symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))


_METRIC_FN = {"tv": tv_distance, "kl": symmetric_kl, "js": js_distance}


# ─────────────────────────────────────── main ───────────────────────────────────
def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load CSV (header row gives column names) ──
    df = pd.read_csv(args.csv, header=0)
    class_col = df.columns[-1]                       # label column
    class_values = sorted(df[class_col].dropna().unique())
    if not class_values:
        raise ValueError("No class labels found in the last column.")

    # ── Identify numeric features ──
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if class_col in numeric_cols:
        numeric_cols.remove(class_col)
    if not numeric_cols:
        raise ValueError("No numeric feature columns detected.")

    print(f"\nFeatures ({len(numeric_cols)}): {numeric_cols}")
    print(f"Classes  ({len(class_values)}): {class_values}\n")

    # ── Per-class mean ± std ──
    stats = (
        df.groupby(class_col)[numeric_cols]
        .agg(["mean", "std"])
        .round(4)
        .sort_index()
    )
    stats_path = Path(args.stats_file) if args.stats_file else outdir / "stats.csv"
    stats.to_csv(stats_path)
    print(f"Saved summary statistics → {stats_path.resolve()}\n")

    # ── Histograms + separation metric ──
    ranking = []
    metric_fn = _METRIC_FN[args.metric]

    for col in numeric_cols:
        data_all = df[col].dropna().values
        bins = (np.histogram_bin_edges(data_all, bins="auto")
                if args.bins is None
                else np.histogram_bin_edges(data_all, bins=args.bins))

        # Histograms as probability vectors
        hist_prob = {}
        for cls in class_values:
            hist, _ = np.histogram(df.loc[df[class_col] == cls, col].dropna(), bins=bins)
            hist = hist.astype(float) + _EPS
            hist /= hist.sum()
            hist_prob[cls] = hist

        # Average pairwise distance
        dists = [metric_fn(hist_prob[c1], hist_prob[c2])
                 for c1, c2 in itertools.combinations(class_values, 2)]
        ranking.append((col, float(np.mean(dists))))

        # Figure with subplots
        n_cls = len(class_values)
        ncols = min(n_cls, args.max_cols)
        nrows = math.ceil(n_cls / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.5 * nrows),
                                 squeeze=False)
        fig.suptitle(f"{col} — per-class distributions", fontsize=14)

        for i, cls in enumerate(class_values):
            r, c = divmod(i, ncols)
            ax = axes[r][c]
            ax.hist(df.loc[df[class_col] == cls, col].dropna(), bins=bins if args.bins is None else args.bins)
            ax.set_title(f"Class {cls}")
            ax.set_xlabel(col)
            ax.set_ylabel("Freq")

        # Hide unused subplots
        for j in range(n_cls, nrows * ncols):
            axes[divmod(j, ncols)].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if args.show:
            plt.show()
        else:
            fig_path = outdir / f"{col}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Saved figure → {fig_path}")

    # ── Ranking CSV ──
    ranking.sort(key=lambda t: t[1], reverse=True)
    rank_df = pd.DataFrame(ranking, columns=["feature", f"avg_{args.metric}"])
    rank_path = Path(args.rank_file) if args.rank_file else outdir / "feature_ranking.csv"
    rank_df.to_csv(rank_path, index=False)

    print("\nTop-ranked features by separation:")
    print(rank_df.head(20).to_string(index=False))
    print(f"\nSaved full ranking → {rank_path.resolve()}")


if __name__ == "__main__":
    main()
