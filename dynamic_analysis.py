#!/usr/bin/env python3
"""
compare_json_lists.py   (with graceful skip-on-error)
"""

import argparse, itertools, json, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ───────────────────────── extraction helpers ──────────────────────────
_LETTERS_ONLY = re.compile(r"[A-Za-z]+")


def is_all_upper_alpha(s: str) -> bool:
    return bool(s) and s.isupper()


def ts_int(ts_val: Any) -> int:  # drop ms
    return int(ts_val) if isinstance(ts_val, (int, float)) else int("".join(ch for ch in str(ts_val) if ch.isdigit()))


def extract_json(json_path: str) -> Tuple[List[int], Dict[str, List[int]]]:
    """Return (ts_sorted, name→freq) OR raise RuntimeError if anything goes wrong."""
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            root = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to read/parse {json_path}: {e}") from e

    events = root.get("dynamic", {}).get("host", [])
    if not isinstance(events, list):
        raise RuntimeError(f"{json_path}: root['dynamic']['host'] is not a list")

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
        raise RuntimeError(f"{json_path}: no valid tuples")

    ts_sorted = sorted({t for _, t in tuples})
    idx = {t: i for i, t in enumerate(ts_sorted)}
    freq: Dict[str, List[int]] = defaultdict(lambda: [0] * len(ts_sorted))
    for name, t in tuples:
        freq[name][idx[t]] += 1
    return ts_sorted, dict(freq)


# ───────────────────────── distance helpers ────────────────────────────
def l2_dist(a, b): return float(np.linalg.norm(a - b))
def corr_dist(a, b):
    sa, sb = np.std(a), np.std(b)
    return 1.0 if sa == 0 or sb == 0 else float(1 - np.corrcoef(a, b)[0, 1])
def dtw_dist(a, b):
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf); dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


_METRIC_FN = {"l2": l2_dist, "corr": corr_dist, "dtw": dtw_dist}


def align(ts_union: List[int], ts: List[int], freq: List[int]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(ts)}
    out = np.zeros(len(ts_union))
    for i, t in enumerate(ts_union):
        j = idx.get(t)
        if j is not None:
            out[i] = freq[j]
    return out


# ───────────────────────────── CLI ────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare JSON lists & rank names")
    p.add_argument("list_files", nargs="+", help="TXT files (≥2) listing JSON paths")
    p.add_argument("--metric", choices=["l2", "dtw", "corr"], default="l2")
    p.add_argument("--y_top", type=int, default=3)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--outdir", default="results")
    return p.parse_args()


# ───────────────────────────── main ───────────────────────────────
def main() -> None:
    args = parse_args()
    if len(args.list_files) < 2:
        sys.exit("Need at least two TXT list files")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metric_fn = _METRIC_FN[args.metric]
    cache: Dict[str, Tuple[List[int], Dict[str, List[int]]]] = {}
    bad_files: set[str] = set()

    def load_json(path: str):
        if path in bad_files:
            return None
        if path not in cache:
            try:
                cache[path] = extract_json(path)
            except RuntimeError as err:
                print(f"[WARN] {err}")
                bad_files.add(path)
                cache[path] = None
        return cache[path]

    # read lists
    lists: List[List[str]] = []
    for txt in args.list_files:
        with open(txt, "r", encoding="utf-8") as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
        if not paths:
            sys.exit(f"{txt} is empty")
        lists.append(paths)

    counts = defaultdict(lambda: [0] * args.y_top)

    # ------------------------------------------------------------------
    # 1) PRE-LOAD each JSON exactly once (skip bad ones)
    # ------------------------------------------------------------------
    data_lists: List[List[Tuple[str, Tuple[List[int], Dict[str, List[int]]]]]] = []
    for txt_paths in lists:
        cur: List[Tuple[str, Tuple[List[int], Dict[str, List[int]]]]] = []
        for path in txt_paths:
            data = load_json(path)          # already warns & marks bad_files
            if data is not None:            # only keep good files
                cur.append((path, data))
        if not cur:
            print(f"[WARN] All JSONs in one list were invalid → skipping that list")
        data_lists.append(cur)

    if len(data_lists) < 2 or any(len(dl) == 0 for dl in data_lists[:2]):
        sys.exit("Nothing to compare after pre-loading valid JSON files.")

    # ------------------------------------------------------------------
    # 2) PAIRWISE comparison using pre-loaded data
    #    (assume exactly two lists: data_lists[0] vs data_lists[1])
    # ------------------------------------------------------------------
    list0, list1 = data_lists[0], data_lists[1]
    total_pairs = len(list0) * len(list1)

    from tqdm import tqdm
    import itertools
    for (path_a, (ts_a, freq_a)), (path_b, (ts_b, freq_b)) in tqdm(
            itertools.product(list0, list1),
            total=total_pairs,
            desc="Comparing JSON pairs"):
        # build union timeline once per pair
        ts_union = sorted(set(ts_a) | set(ts_b))
        names = set(freq_a) | set(freq_b)

        # compute distances
        dist_pairs = []
        for name in names:
            seq_a = align(ts_union, ts_a, freq_a.get(name, [0] * len(ts_a)))
            seq_b = align(ts_union, ts_b, freq_b.get(name, [0] * len(ts_b)))
            dist_pairs.append((name, metric_fn(seq_a, seq_b)))

        dist_pairs.sort(key=lambda t: t[1], reverse=True)
        for rank_idx, (name, _) in enumerate(dist_pairs[: args.y_top]):
            counts[name][rank_idx] += 1

    # from tqdm import tqdm
    # import itertools

    # # assume exactly two sub-lists ⇒ lists[0], lists[1]
    # list0, list1 = lists[0], lists[1]
    # total_pairs = len(list0) * len(list1)

    # for json_a, json_b in tqdm(itertools.product(list0, list1),
    #                         total=total_pairs,
    #                         desc="Comparing JSON pairs"):
    # # # # compare every unordered pair of list-files
    # # # for idx_a, idx_b in itertools.combinations(range(len(lists)), 2):
    # #     for json_a in lists[idx_a]:
    # #         for json_b in lists[idx_b]:
    #         data_a = load_json(json_a)
    #         data_b = load_json(json_b)
    #         if data_a is None or data_b is None:
    #             continue  # skip if either file failed

    #         ts_a, freq_a = data_a
    #         ts_b, freq_b = data_b

    #         ts_union = sorted(set(ts_a) | set(ts_b))
    #         names = set(freq_a) | set(freq_b)

    #         dist_pairs = []
    #         for name in names:
    #             seq_a = align(ts_union, ts_a, freq_a.get(name, [0]*len(ts_a)))
    #             seq_b = align(ts_union, ts_b, freq_b.get(name, [0]*len(ts_b)))
    #             dist_pairs.append((name, metric_fn(seq_a, seq_b)))

    #         dist_pairs.sort(key=lambda t: t[1], reverse=True)
    #         for r, (name, _) in enumerate(dist_pairs[:args.y_top]):
    #             counts[name][r] += 1

    # output
    cols = [f"rank_{i+1}" for i in range(args.y_top)]
    df = pd.DataFrame([(n, *counts[n]) for n in counts], columns=["name", *cols])
    df.sort_values(cols, ascending=False, inplace=True, ignore_index=True)
    df.to_csv(outdir / "rank_counts.csv", index=False)
    print(f"Saved full rank counts → {outdir/'rank_counts.csv'}")

    for r in range(args.y_top):
        col = f"rank_{r+1}"
        top = df[["name", col]][df[col] > 0].nlargest(args.top_n, col)
        if top.empty:
            continue
        top.to_csv(outdir / f"top_rank_{r+1}.csv", index=False)
        print(f"\nTop {args.top_n} names for rank {r+1}:")
        print(top.to_string(index=False))
        print(f"Saved → {outdir/f'top_rank_{r+1}.csv'}")


if __name__ == "__main__":
    main()
