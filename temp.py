#!/usr/bin/env python3
"""
Merge many per-file-sorted pickle logs into equal-frequency bins, keep
(timestamp, action, filename) triples, and write each bin either as:

    • batched Pickle  (default)     --format pickle
    • Arrow Feather   (requires pyarrow) --format feather
"""
from __future__ import annotations
import argparse, collections, gc, heapq, logging, math, pickle, sys
from pathlib import Path
from operator import itemgetter
from typing  import Iterator, List, Tuple
from tqdm    import tqdm                                    # progress bar

# optional Arrow
try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ModuleNotFoundError:
    pa = feather = None

RecordOut = Tuple[float, str, str]      # (timestamp, action, filename)
BATCH     = 100_000                     # records per pickle batch
N_BINS    = 10

# ─────────────── helpers ─────────────── #

def stream_pickle(path: Path) -> Iterator[RecordOut]:
    """Load the whole list once, then yield (ts, act, fname)."""
    fname = path.name
    with path.open("rb") as fh:
        records = pickle.load(fh)       # one list
    for ts, act in records:
        yield (ts, act, fname)
    del records; gc.collect()

def write_pickle_batched(stream: Iterator[RecordOut], out: Path,
                         batch: int = BATCH) -> None:
    with out.open("wb") as fh:
        pk = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL)
        buf: List[RecordOut] = []
        for rec in stream:
            buf.append(rec)
            if len(buf) >= batch:
                pk.dump(buf)
                buf.clear(); pk.clear_memo()
        if buf:
            pk.dump(buf)
        pk.clear_memo()

def write_feather(stream: Iterator[RecordOut], out: Path) -> None:
    if pa is None:
        raise RuntimeError("pyarrow not installed; cannot use feather")
    data = list(stream)                 # one materialisation
    tbl  = pa.Table.from_pylist([{'ts':ts, 'act':a, 'file':f} for ts,a,f in data])
    feather.write_feather(tbl, out, compression="lz4")

def write_records(stream: Iterator[RecordOut], out: Path, fmt: str) -> None:
    if fmt == "pickle":
        write_pickle_batched(stream, out)
    elif fmt == "feather":
        write_feather(stream, out)
    else:
        raise ValueError("format must be 'pickle' or 'feather'")

# ─────────────── phase-1: stats ─────────────── #

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
    total, target = sum(counter.values()), 0
    edges, acc, quota = [], 0, total / bins
    for ts in sorted(counter):
        acc += counter[ts]
        if acc >= quota and len(edges) < bins - 1:
            edges.append(ts); quota += total / bins
    return [min(counter)] + edges + [max(counter)]

# ─────────────── phase-2: per-bin merge ─────────────── #

def merge_bin(files: List[Path], lo: float, hi: float) -> Iterator[RecordOut]:
    gens = []
    for p in files:
        def gen(path=p):
            for ts, act, fname in stream_pickle(path):
                if ts < lo:  continue
                if ts >= hi: break
                yield (ts, act, fname)
        gens.append(gen())
    return heapq.merge(*gens, key=itemgetter(0))

# ─────────────── driver ─────────────── #

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--format",     choices=["pickle", "feather"], default="pickle")
    args = ap.parse_args(argv)

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files:
        logging.error("no *.pkl in %s", args.input_dir); sys.exit(1)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    counter, t_min, t_max = scan_stats(files)
    edges = make_equalfreq_edges(counter)
    logging.info("Equal-frequency bins built (%.0f total records)", sum(counter.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i+1]
        logging.info("Bin %02d  [%.6f, %.6f)", i, lo, hi)
        merged = merge_bin(files, lo, hi)
        out    = args.output_dir / f"interval_{i:02d}.{ 'pkl' if args.format=='pickle' else 'feather'}"
        write_records(merged, out, args.format)
        logging.info("→ wrote %s", out.name)

if __name__ == "__main__":
    main(sys.argv[1:])