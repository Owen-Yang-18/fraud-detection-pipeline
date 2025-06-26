#!/usr/bin/env python3
"""
Parallel version: 8 worker processes build the 10 equal-frequency bins
concurrently.  See original header comments for context.
"""
from __future__ import annotations
import argparse, collections, gc, heapq, logging, math, pickle, sys
from pathlib import Path
from operator import itemgetter
from typing   import Iterator, List, Tuple

from tqdm import tqdm                              # progress bar
from multiprocessing import Pool, cpu_count        # parallelism  ★docs : :contentReference[oaicite:2]{index=2}

# ---------- optional Arrow handling ----------------------------------------------------
try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ModuleNotFoundError:
    pa = feather = None

RecordOut = Tuple[float, str, str]     # (ts, act, fname)
BATCH      = 100_000                   # records per Pickle dump
N_BINS     = 10
N_WORKERS  = 8                         # <= cpu_count()  :contentReference[oaicite:3]{index=3}

# -------------------------- streaming helpers (unchanged) -----------------------------
def stream_pickle(path: Path) -> Iterator[RecordOut]:
    fname = path.name
    with path.open("rb") as fh:
        recs = pickle.load(fh)
    for ts, act in recs:
        yield (ts, act, fname)
    del recs; gc.collect()

def write_pickle_batched(stream: Iterator[RecordOut], out: Path,
                         batch: int = BATCH) -> None:
    with out.open("wb") as fh:
        pk, buf = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL), []
        for rec in stream:
            buf.append(rec)
            if len(buf) >= batch:
                pk.dump(buf); buf.clear(); pk.clear_memo()
        if buf: pk.dump(buf)
        pk.clear_memo()

def write_feather(stream: Iterator[RecordOut], out: Path) -> None:
    if pa is None:
        raise RuntimeError("pyarrow not installed")
    tbl = pa.Table.from_pylist(
        [{'ts': ts, 'act': a, 'file': f} for ts, a, f in stream])
    feather.write_feather(tbl, out, compression="lz4")

def write_records(stream: Iterator[RecordOut], out: Path, fmt: str) -> None:
    write_pickle_batched(stream, out) if fmt == "pickle" else write_feather(stream, out)

# -------------------------- stats / edges (unchanged) ----------------------------------
def scan_stats(files: List[Path]) -> tuple[collections.Counter, float, float]:
    ctr, tmin, tmax = collections.Counter(), math.inf, -math.inf
    for p in tqdm(files, desc="Scanning pickles", unit="file"):
        for ts, *_ in stream_pickle(p):
            ctr[ts] += 1;  tmin = min(tmin, ts);  tmax = max(tmax, ts)
    return ctr, tmin, tmax

def make_equalfreq_edges(counter: collections.Counter, bins: int = N_BINS) -> List[float]:
    tot, acc, quota = sum(counter.values()), 0, sum(counter.values()) / bins
    edges: List[float] = []
    for ts in sorted(counter):
        acc += counter[ts]
        if acc >= quota and len(edges) < bins - 1:
            edges.append(ts); quota += tot / bins
    return [min(counter)] + edges + [max(counter)]

def merge_bin(files: List[Path], lo: float, hi: float) -> Iterator[RecordOut]:
    gens = []
    for p in files:
        def gen(path=p):
            for ts, act, fname in stream_pickle(path):
                if ts < lo: continue
                if ts >= hi: break
                yield (ts, act, fname)
        gens.append(gen())
    return heapq.merge(*gens, key=itemgetter(0))          # one-tuple buffer  :contentReference[oaicite:4]{index=4}

# ========================== multiprocessing section ====================================

# globals each worker will reuse (read-only)
_WORK_FILES: List[Path] = []
_OUT_DIR: Path          = Path()
_FMT: str               = "pickle"

def _init_pool(files: List[Path], out_dir: Path, fmt: str) -> None:
    """Pool initializer: set globals so each worker avoids pickling big args."""
    global _WORK_FILES, _OUT_DIR, _FMT
    _WORK_FILES, _OUT_DIR, _FMT = files, out_dir, fmt           # ★pattern : :contentReference[oaicite:5]{index=5}

def _process_one_bin(args) -> str:
    """Worker: build one bin and return output filename for progress bar."""
    idx, lo, hi = args
    merged = merge_bin(_WORK_FILES, lo, hi)
    suffix = "pkl" if _FMT == "pickle" else "feather"
    out    = _OUT_DIR / f"interval_{idx:02d}.{suffix}"
    write_records(merged, out, _FMT)
    return out.name

# -------------------------- driver ------------------------------------------------------
def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--format",     choices=["pickle", "feather"], default="pickle")
    args = ap.parse_args(argv)

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files: logging.error("no *.pkl files found"); sys.exit(1)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    counter, *_ = scan_stats(files)
    edges = make_equalfreq_edges(counter)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # prepare work list
    bins = [(i, edges[i], edges[i+1]) for i in range(N_BINS)]

    with Pool(processes=N_WORKERS,
              initializer=_init_pool,
              initargs=(files, args.output_dir, args.format)) as pool:   # ★docs : :contentReference[oaicite:6]{index=6}
        for out_name in tqdm(pool.imap_unordered(_process_one_bin, bins),
                             total=N_BINS, desc="Merging bins"):          # ★recipe : :contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}
            logging.info("→ wrote %s", out_name)

if __name__ == "__main__":
    main(sys.argv[1:])
