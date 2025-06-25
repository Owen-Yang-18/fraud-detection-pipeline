#!/usr/bin/env python3
"""
Stream-merge many per-file-sorted pickle logs into equal-frequency bins,
keeping the source *filename* inside every record.

Input
-----
Each *.pkl* in --input_dir contains **one list** of (timestamp, action) tuples
sorted by timestamp.

Output
------
interval_00.pkl … interval_09.pkl, each storing
    (timestamp, action, filename)
triples, sorted by timestamp, with ~equal numbers of records per file
(equal-frequency binning).

Author : ChatGPT (o3) — 25 Jun 2025 · MIT
"""
from __future__ import annotations
import argparse, collections, heapq, logging, math, pickle, sys, gc
from pathlib import Path
from operator import itemgetter
from typing  import Iterator, List, Tuple
from tqdm    import tqdm

RecordOut = Tuple[float, str, str]            # (timestamp, action, filename)
N_BINS    = 10

# ───────────────────────── helpers ───────────────────────── #

def stream_pickle(path: Path) -> Iterator[RecordOut]:
    """
    Yield *(timestamp, action, filename)* from a pickle that contains **one list**.
    The entire list is loaded once, then iterated; the list reference is
    released immediately afterwards to free memory.
    """
    fname = path.name
    with path.open("rb") as fh:
        records = pickle.load(fh)             # -> list[(ts, action)]
    for ts, act in records:
        yield (ts, act, fname)
    del records
    gc.collect()                              # encourage GC for large lists

def write_pickle_stream(stream: Iterator[RecordOut], out: Path) -> None:
    with out.open("wb") as fh:
        pk = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL)
        for rec in stream:
            pk.dump(rec)
        pk.clear_memo()

# ─────────────── Phase 1: scan & frequency counter ─────────────── #

def scan_stats(files: List[Path]) -> tuple[collections.Counter, float, float]:
    counter: collections.Counter[float] = collections.Counter()
    t_min, t_max = math.inf, -math.inf

    for p in tqdm(files, desc="Scanning pickles", unit="file"):
        for ts, *_ in stream_pickle(p):
            counter[ts] += 1
            t_min = min(t_min, ts)
            t_max = max(t_max, ts)
    return counter, t_min, t_max

def make_equalfreq_edges(counter: collections.Counter,
                         bins: int = N_BINS) -> List[float]:
    total  = sum(counter.values())
    target = total / bins
    edges: List[float] = []
    cum, goal = 0, target

    for ts in sorted(counter):
        cum += counter[ts]
        if cum >= goal and len(edges) < bins - 1:
            edges.append(ts)
            goal += target
    edges = [min(counter)] + edges + [max(counter)]
    return edges

# ─────────────── Phase 2: per-bin merge & write ─────────────── #

def merge_bin(files: List[Path], start: float, end: float) -> Iterator[RecordOut]:
    key, gens = itemgetter(0), []

    for p in files:
        def gen(path=p):
            for ts, act, fname in stream_pickle(path):
                if ts < start:
                    continue
                if ts >= end:
                    break                    # records are sorted; done with this file
                yield (ts, act, fname)
        gens.append(gen())
    return heapq.merge(*gens, key=key)       # one tuple per file in memory

# ───────────────────────────── driver ───────────────────────────── #

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Merge pickle logs into equal-frequency bins (with filename)")
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--bins",       type=int, default=N_BINS)
    args = ap.parse_args(argv)

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files:
        logging.error("No *.pkl files in %s", args.input_dir)
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Phase 1: collect statistics
    counter, t_min, t_max = scan_stats(files)
    logging.info("Scanned %d files · %d unique timestamps · span %.6f → %.6f",
                 len(files), len(counter), t_min, t_max)

    edges = make_equalfreq_edges(counter, bins=args.bins)
    logging.info("Equal-frequency edges: %s", edges)

    # Phase 2: process each bin
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.bins):
        start, end = edges[i], edges[i + 1]
        logging.info("Bin %02d  [%.6f, %.6f)", i, start, end)
        merged   = merge_bin(files, start, end)
        out_path = args.output_dir / f"interval_{i:02d}.pkl"
        write_pickle_stream(merged, out_path)
        logging.info("→ wrote %s", out_path.name)

if __name__ == "__main__":
    main(sys.argv[1:])
