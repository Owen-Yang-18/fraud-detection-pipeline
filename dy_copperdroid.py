from __future__ import annotations

import argparse, heapq, itertools, logging, os, pickle, shutil, sys, tempfile
from pathlib import Path
from typing import Iterator, List, Tuple
from operator import itemgetter

Record = Tuple[float, str]          # (timestamp, action)

# --------------------------------------------------------------------------- #
#                          Core streaming utilities                           #
# --------------------------------------------------------------------------- #

def load_pickle_iter(path: Path) -> Iterator[Record]:
    """Yield items from a pickle file that contains a *list* of records."""
    with path.open("rb") as fh:
        for rec in pickle.load(fh):
            yield rec

def write_chunk(chunk: List[Record], out_dir: Path, chunk_id: int) -> None:
    """Pickle-dump `chunk` to disk using an atomic rename."""
    tmp = out_dir / f"chunk_{chunk_id:05d}.part"
    final = out_dir / f"chunk_{chunk_id:05d}.pkl"
    with tmp.open("wb") as fh:
        pickle.dump(chunk, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(final)   # atomic on POSIX
    logging.info("Wrote %s (%d records)", final.name, len(chunk))

# --------------------------------------------------------------------------- #
#                          Two-phase external merge                           #
# --------------------------------------------------------------------------- #

def batch_merge(
    files: List[Path], tmp_dir: Path, batch_size: int
) -> List[Path]:
    """First pass: merge `batch_size` input pickles at a time."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: List[Path] = []
    key = itemgetter(0)  # compare on timestamp

    for batch_idx, start in enumerate(range(0, len(files), batch_size)):
        batch = files[start : start + batch_size]
        logging.info("Batch %d: merging %d files", batch_idx, len(batch))
        iters = [load_pickle_iter(p) for p in batch]
        merged_iter = heapq.merge(*iters, key=key)
        outfile = tmp_dir / f"batch_{batch_idx:05d}.pkl"
        with outfile.open("wb") as fh:
            pickle.dump(list(merged_iter), fh, protocol=pickle.HIGHEST_PROTOCOL)
        batch_paths.append(outfile)
        logging.info("   saved -> %s", outfile.name)
    return batch_paths

def final_streaming_merge(
    batches: List[Path],
    out_dir: Path,
    chunk_size: int,
) -> None:
    """Second pass: k-way merge the batch files, emitting chunked pickles."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key = itemgetter(0)
    iters = [load_pickle_iter(p) for p in batches]
    merged = heapq.merge(*iters, key=key)

    chunk: List[Record] = []
    chunk_id = 0
    for rec in merged:
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            write_chunk(chunk, out_dir, chunk_id)
            chunk.clear()
            chunk_id += 1

    if chunk:  # residual
        write_chunk(chunk, out_dir, chunk_id)

# --------------------------------------------------------------------------- #
#                           CLI / argument parsing                            #
# --------------------------------------------------------------------------- #

def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="External merge-sort over many per-file-sorted pickles."
    )
    ap.add_argument("--input_dir", required=True, type=Path,
                    help="Directory containing the  *.pkl  input files.")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Where to write chunk_XXXXX.pkl outputs.")
    ap.add_argument("--batch_size", type=int, default=100,
                    help="How many pickles to load and merge at once.")
    ap.add_argument("--chunk_size", type=int, default=500_000,
                    help="Number of records per output chunk pickle.")
    ap.add_argument("--tmp_dir", type=Path, default=None,
                    help="Optional explicit temp directory for batch files.")
    return ap.parse_args(argv)

def main(argv: List[str]) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files:
        logging.error("No *.pkl files found in %s", args.input_dir)
        sys.exit(1)

    tmp_dir = args.tmp_dir or Path(tempfile.mkdtemp(prefix="extsort_"))
    try:
        # -------- Phase 1: partial merges ---------------------------------- #
        batch_files = batch_merge(files, tmp_dir, args.batch_size)

        # -------- Phase 2: streaming k-way merge to final chunks ----------- #
        final_streaming_merge(batch_files, args.output_dir, args.chunk_size)
        logging.info("Done!  Sorted data written to %s", args.output_dir)

    finally:
        # Remove intermediate batch files unless user kept them
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main(sys.argv[1:])
