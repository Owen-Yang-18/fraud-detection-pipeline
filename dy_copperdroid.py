#!/usr/bin/env python3
"""
Memory-safe external merge-sort for many per-file-sorted pickle logs.

Two modes
---------
(1) **Normal run**  : give --input_dir; the script creates batch_XXXXX.pkl
    files in a temp directory, then merges them into chunk_XXXXX.pkl.

(2) **Resume merge**: give --batches_dir to *skip* the run-generation phase
    and start the final k-way merge directly from the existing batch files.

Each input pickle (or batch pickle) stores *one object per frame* so that
`Unpickler.load()` returns just a single (timestamp, action) tuple at a time.
This keeps resident memory O(#open files) rather than O(dataset size).

Raise your fd-limit with `ulimit -n 4096` if you hit “Too many open files”.

Author : ChatGPT (o3) — 25 Jun 2025 — MIT
"""
from __future__ import annotations

import argparse, gc, heapq, logging, pickle, shutil, sys, tempfile
from pathlib import Path
from operator import itemgetter
from typing import Iterator, List, Tuple

Record = Tuple[float, str]        # (timestamp, action)

# ───────────────────────────── Streaming helpers ───────────────────────────── #

def load_pickle_iter(path: Path) -> Iterator[Record]:
    """Yield one tuple at a time from a multi-object pickle file."""
    with path.open("rb") as fh:
        up = pickle.Unpickler(fh)
        while True:
            try:
                yield up.load()                     # one record per call
            except EOFError:
                break

def write_pickle_stream(iterable: Iterator[Record], path: Path) -> None:
    """Write many records, one pickle frame each, to *path*."""
    with path.open("wb") as fh:
        pk = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL)
        for rec in iterable:
            pk.dump(rec)
        pk.clear_memo()                            # free the memo table

def write_chunk(chunk: List[Record], out_dir: Path, idx: int) -> None:
    """Atomically dump *chunk* to chunk_{idx:05d}.pkl inside *out_dir*."""
    tmp   = out_dir / f"chunk_{idx:05d}.part"
    final = out_dir / f"chunk_{idx:05d}.pkl"
    with tmp.open("wb") as fh:
        pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(final)
    logging.info("wrote %s (%d records)", final.name, len(chunk))

# ───────────────────────────── Sort / merge logic ──────────────────────────── #

def batch_merge(files: List[Path], tmp_dir: Path, batch_size: int) -> List[Path]:
    """Phase-1: stream-merge ≤batch_size pickles at a time → batch_XXXXX.pkl."""
    key = itemgetter(0)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    outfiles: List[Path] = []

    for idx, start in enumerate(range(0, len(files), batch_size)):
        group = files[start : start + batch_size]
        logging.info("batch %03d: merging %d files", idx, len(group))
        merged = heapq.merge(*(load_pickle_iter(p) for p in group), key=key)
        out_path = tmp_dir / f"batch_{idx:05d}.pkl"
        write_pickle_stream(merged, out_path)
        outfiles.append(out_path)
        gc.collect()                                # keep RSS flat
    return outfiles

def final_merge(batches: List[Path], out_dir: Path, chunk_size: int) -> None:
    """Phase-2: k-way merge all *batches* → chunk_XXXXX.pkl."""
    key = itemgetter(0)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = heapq.merge(*(load_pickle_iter(p) for p in batches), key=key)

    chunk, cid = [], 0
    for rec in merged:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            write_chunk(chunk, out_dir, cid)
            chunk.clear()
            cid += 1
    if chunk:
        write_chunk(chunk, out_dir, cid)

# ──────────────────────────────── CLI driver ───────────────────────────────── #

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="External merge-sort large pickle logs (streaming)"
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_dir",   type=Path,
                   help="Directory of PER-FILE-sorted pickles (phase-1 + 2).")
    g.add_argument("--batches_dir", type=Path,
                   help="Directory of batch_XXXXX.pkl to resume phase-2 only.")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Where chunk_XXXXX.pkl are written.")
    ap.add_argument("--batch_size", type=int, default=100,
                    help="Files loaded simultaneously during phase-1.")
    ap.add_argument("--chunk_size", type=int, default=500_000,
                    help="Number of records per output chunk.")
    ap.add_argument("--keep_batches", action="store_true",
                    help="Do NOT delete batch files in normal run.")
    return ap.parse_args(argv)

def main(argv: List[str]) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S")

    # ── Phase selection ────────────────────────────────────────────────────── #
    if args.batches_dir:                           # resume mode
        batch_files = sorted(args.batches_dir.glob("batch_*.pkl"))
        if not batch_files:
            logging.error("no batch_*.pkl in %s", args.batches_dir)
            sys.exit(1)
        final_merge(batch_files, args.output_dir, args.chunk_size)
        return                                     # finished

    # else: full run from raw input pickles
    input_files = sorted(args.input_dir.glob("*.pkl"))
    if not input_files:
        logging.error("no *.pkl files in %s", args.input_dir)
        sys.exit(1)

    tmp_dir = Path(tempfile.mkdtemp(prefix="extsort_"))
    try:
        batch_files = batch_merge(input_files, tmp_dir, args.batch_size)
        final_merge(batch_files, args.output_dir, args.chunk_size)
    finally:
        if not args.keep_batches:
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main(sys.argv[1:])
