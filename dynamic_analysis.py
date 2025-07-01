#!/usr/bin/env python3
"""
compare_json_lists.py
---------------------
Cross-compare JSON files listed in two-or-more TXT “lists”, rank sequence
dissimilarity per name, and tally how often each name is rank-#1, #2, …, #y.

For each rank position i in 1…y_top, the script prints and saves the top-N
names (default 10) that appear most frequently at that rank.

Distance metrics: l2 (default), dtw, corr.

Example
-------
python compare_json_lists.py listA.txt listB.txt \
       --metric dtw --y_top 3 --top_n 10 --outdir results
"""

import argparse
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ────────────────────────── extraction helpers ──────────────────────────
_LETTERS_ONLY = re.compile(r"[A-Za-z]+")


def is_all_upper_alpha(s: str) -> bool:
    return s.isupper()


def ts_int(ts_val: Any) -> int:
    """Drop milliseconds: int(ts_ms)."""
    if isinstance(ts_val, (int, float)):
        return int(ts_val)
    return int("".join(ch for ch in str(ts_val) if ch.isdigit()))


def extract_json(json_path: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """Return (sorted_ts, name→freq_list) for one JSON file."""
    with open(json_path, "r", encoding="utf-8") as fh:
        root = json.load(fh)
    events = root.get("dynamic", {}).get("host", [])
    if not isinstance(events, list):
        raise ValueError(f"{json_path}: root['dynamic']['host'] is not a list")

    tuples: List[Tuple[str, int]] = []
    for ev in events:
        if not isinstance(ev, dict) or "class" not in ev or "low" not in ev:
            continue
        low0 = (ev["low"] or [{}])[0]
        if "ts" not in low0:
            continue
        ts = ts_int(low0["ts"])
        cls = str(ev["class"])

        if cls == "BINDER":
            name = str(ev.get("method", ""))
        elif cls == "SYSCALL":
            name = str(low0.get("sysname", ""))
        elif is_all_upper_alpha(cls):
            name = cls
        else:
            continue

        if name:
            tuples.append((name, ts))

    if not tuples:
        raise ValueError(f"{json_path}: no valid tuples")

    ts_sorted = sorted({t for _, t in tuples})
    index = {t: i for i, t in enumerate(ts_sorted)}
    freq: Dict[str, List[int]] = defaultdict(lambda: [0] * len(ts_sorted))
    for name, t in tuples:
        freq[name][index[t]] += 1

    return ts_sorted, dict(freq)


# ────────────────────────── distance helpers ────────────────────────────
def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def corr_dist(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 1.0
    return float(1 - np.corrcoef(a, b)[0, 1])


def dtw_dist(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


_METRIC_FN = {"l2": l2_dist, "corr": corr_dist, "dtw": dtw_dist}


def align(ts_union: List[int], ts: List[int], freq: List[int]) -> np.ndarray:
    """Pad `freq` onto `ts_union` timeline."""
    idx = {t: i for i, t in enumerate(ts)}
    out = np.zeros(len(ts_union), dtype=float)
    for i, t in enumerate(ts_union):
        j = idx.get(t)
        if j is not None:
            out[i] = freq[j]
    return out


# ─────────────────────────────── CLI ────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare JSON lists & rank names")
    p.add_argument("list_files", nargs="+", help="TXT files (≥2) listing JSON paths")
    p.add_argument("--metric", choices=["l2", "dtw", "corr"], default="l2",
                   help="Distance metric (default: l2)")
    p.add_argument("--y_top", type=int, default=3,
                   help="Track ranks 1…y_top (default 3)")
    p.add_argument("--top_n", type=int, default=10,
                   help="Print/save top_n names for each rank (default 10)")
    p.add_argument("--outdir", default="results", help="Output directory")
    return p.parse_args()


# ─────────────────────────────────── main ─────────────────────────────────
def main() -> None:
    args = parse_args()
    if len(args.list_files) < 2:
        sys.exit("Need at least two TXT list files")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metric_fn = _METRIC_FN[args.metric]

    # Cache extracted data
    cache: Dict[str, Tuple[List[int], Dict[str, List[int]]]] = {}

    def load_json(path: str):
        if path not in cache:
            cache[path] = extract_json(path)
        return cache[path]

    # Read TXT lists
    lists: List[List[str]] = []
    for txt in args.list_files:
        with open(txt, "r", encoding="utf-8") as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
        if not paths:
            sys.exit(f"{txt} is empty")
        lists.append(paths)

    # counts[name][rank_idx]  (rank_idx 0 == rank-1)
    counts = defaultdict(lambda: [0] * args.y_top)

    # Compare every unordered pair of list-files
    for idx_a, idx_b in itertools.combinations(range(len(lists)), 2):
        for json_a in lists[idx_a]:
            for json_b in lists[idx_b]:
                ts_a, freq_a = load_json(json_a)
                ts_b, freq_b = load_json(json_b)

                ts_union = sorted(set(ts_a) | set(ts_b))
                names = set(freq_a) | set(freq_b)

                dist_pairs = []
                for name in names:
                    seq_a = align(ts_union, ts_a, freq_a.get(name, [0] * len(ts_a)))
                    seq_b = align(ts_union, ts_b, freq_b.get(name, [0] * len(ts_b)))
                    dist_pairs.append((name, metric_fn(seq_a, seq_b)))

                dist_pairs.sort(key=lambda t: t[1], reverse=True)
                for rank_idx, (name, _) in enumerate(dist_pairs[: args.y_top]):
                    counts[name][rank_idx] += 1

    # Build full counts DataFrame
    rank_cols = [f"rank_{i+1}" for i in range(args.y_top)]
    df_counts = pd.DataFrame(
        [(n, *counts[n]) for n in counts], columns=["name", *rank_cols]
    )
    df_counts.sort_values(rank_cols, ascending=False, inplace=True, ignore_index=True)
    df_counts.to_csv(outdir / "rank_counts.csv", index=False)
    print(f"Saved full rank counts → {outdir/'rank_counts.csv'}")

    # Per-rank top-N summaries
    for r in range(args.y_top):
        col = f"rank_{r+1}"
        sub = df_counts[["name", col]].copy()
        sub = sub[sub[col] > 0].nlargest(args.top_n, col)
        if sub.empty:
            continue
        csv_path = outdir / f"top_rank_{r+1}.csv"
        sub.to_csv(csv_path, index=False)

        print(f"\nTop {args.top_n} names for rank {r+1} (by appearances):")
        print(sub.to_string(index=False))
        print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
