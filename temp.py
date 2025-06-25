#!/usr/bin/env python3
"""
Stream-sorted interval extraction.

Given a directory of pickle files, each containing a list (or stream) of
(timestamp, action) tuples sorted by timestamp, this script:

    1. Finds the global min/max timestamp.
    2. Splits that range into N uniform intervals.
    3. For every interval i:
         • re-opens each pickle,
         • streams only records in [edge_i, edge_{i+1}),
         • heap-merges those generators into a globally-sorted iterator,
         • writes interval_i.pkl to --output_dir (one frame per tuple).

The merge keeps at most (#open files) tuples in RAM and closes every file
promptly via context managers.

Author  : ChatGPT (o3) — 25 Jun 2025
Licence : MIT
"""
from __future__ import annotations
import argparse, heapq, logging, math, pickle, sys
from pathlib import Path
from operator import itemgetter
from typing  import Iterator, List, Tuple

try:
    import numpy as np   # optional, used for linspace
except ImportError:
    np = None

Record = Tuple[float, str]          # (timestamp, action)

# ─────────────────── streaming helpers ──────────────────── #

def stream_pickle(path: Path) -> Iterator[Record]:
    """Yield one record at a time; memo never inflates."""
    with path.open("rb") as fh:     # context mgr closes file automatically
        while True:
            try:
                yield pickle.load(fh)   # fresh Unpickler per call
            except EOFError:
                break

def pickle_stream_writer(stream: Iterator[Record], out_path: Path) -> None:
    """Write many tuples, one pickle frame each, to *out_path*."""
    with out_path.open("wb") as fh:
        pk = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL)
        for rec in stream:
            pk.dump(rec)
        pk.clear_memo()                 # drop writer memo

# ─────────────────── min / max discovery ─────────────────── #

def global_min_max(files: List[Path]) -> Tuple[float, float]:
    t_min, t_max = math.inf, -math.inf
    for p in files:
        for ts, _ in stream_pickle(p):
            t_min = min(t_min, ts)
            t_max = max(t_max, ts)
    return t_min, t_max

def make_edges(t_min: float, t_max: float, n: int) -> List[float]:
    if np:
        return np.linspace(t_min, t_max, n + 1).tolist()
    step = (t_max - t_min) / n
    return [t_min + i * step for i in range(n + 1)]

# ─────────────────────────── driver ──────────────────────── #

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Sort and write records per uniform time interval")
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--intervals",  type=int,   default=10,
                    help="Number of equal-width intervals (default 10)")
    args = ap.parse_args(argv)

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files:
        logging.error("No *.pkl in %s", args.input_dir); sys.exit(1)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")

    # 1️⃣  discover time span
    t_min, t_max = global_min_max(files)
    logging.info("Global span: %.6f → %.6f", t_min, t_max)

    # 2️⃣  build edges
    edges = make_edges(t_min, t_max, args.intervals)
    logging.info("Built %d equal-width intervals", args.intervals)

    # 3️⃣  loop over intervals
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key = itemgetter(0)

    for i in range(args.intervals):
        start, end = edges[i], edges[i + 1]
        logging.info("Interval %02d  [%.6f, %.6f)", i, start, end)

        # Build one generator per pickle that yields only records in window
        gens = []
        for p in files:
            def gen(path=p):                   # default-arg trick for closure
                for ts, act in stream_pickle(path):
                    if start <= ts < end:
                        yield (ts, act)
                    elif ts >= end:            # file is sorted: early break
                        break
            gens.append(gen())

        merged = heapq.merge(*gens, key=key)   # k-way merge

        out = args.output_dir / f"interval_{i:02d}.pkl"
        pickle_stream_writer(merged, out)
        logging.info("wrote → %s", out.name)

if __name__ == "__main__":
    main(sys.argv[1:])
