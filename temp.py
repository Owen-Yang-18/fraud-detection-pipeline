from __future__ import annotations
import argparse, collections, heapq, logging, math, pickle, sys
from pathlib import Path
from operator import itemgetter
from typing  import Iterator, List, Tuple

from tqdm import tqdm                                     # progress bar

Record = Tuple[float, str]                                # (timestamp, action)
N_BINS = 10                                               # default # intervals

# ────────────────── Streaming helpers ────────────────── #

def stream_pickle(path: Path) -> Iterator[Record]:
    """Yield one record at a time; closes file immediately on EOF."""
    with path.open("rb") as fh:
        while True:
            try:
                yield pickle.load(fh)                     # fresh Unpickler
            except EOFError:
                break

def write_pickle_stream(stream: Iterator[Record], out: Path) -> None:
    """Write many records, one pickle frame each."""
    with out.open("wb") as fh:
        pk = pickle.Pickler(fh, protocol=pickle.HIGHEST_PROTOCOL)
        for rec in stream:
            pk.dump(rec)
        pk.clear_memo()

# ────────────────── Phase 1 – global stats ───────────── #

def scan_stats(files: List[Path]) -> tuple[collections.Counter, float, float]:
    """First pass: return (Counter, t_min, t_max) with tqdm progress."""
    counter: collections.Counter[float] = collections.Counter()
    t_min, t_max = math.inf, -math.inf

    for p in tqdm(files, desc="Scanning pickles", unit="file"):
        for ts, _ in stream_pickle(p):
            counter[ts] += 1
            t_min = min(t_min, ts)
            t_max = max(t_max, ts)
    return counter, t_min, t_max

# ────────────────── Build equal-frequency edges ──────── #

def make_equalfreq_edges(counter: collections.Counter,
                         bins: int = N_BINS) -> List[float]:
    """Compute edges so each bin has ≈ total/bins occurrences."""
    total = sum(counter.values())
    target = total / bins
    edges: List[float] = []
    cum, goal = 0, target

    for ts in sorted(counter):                            # ascending timestamps
        cum += counter[ts]
        if cum >= goal and len(edges) < bins - 1:         # record edge(1…bins-1)
            edges.append(ts)
            goal += target
    edges = [min(counter)] + edges + [max(counter)]       # e₀..e_N inclusive
    return edges

# ────────────────── Phase 2 – per-bin merge ──────────── #

def merge_bin(files: List[Path], start: float, end: float) -> Iterator[Record]:
    """Return lazy iterator of globally-sorted records in [start, end)."""
    key = itemgetter(0)
    gens = []

    for p in files:
        def gen(path=p):                                  # closure default
            for ts, act in stream_pickle(path):
                if float(ts) < start:
                    continue
                if float(ts) >= end:
                    break                                 # file sorted, done
                yield (float(ts), act)
        gens.append(gen())
    return heapq.merge(*gens, key=key)

# ────────────────── Driver ────────────────────────────── #

def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(
        description="Merge pickle logs into equal-frequency time bins")
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--bins",       type=int, default=N_BINS)
    args = ap.parse_args(argv)

    files = sorted(args.input_dir.glob("*.pkl"))
    if not files:
        logging.error("No *.pkl in %s", args.input_dir); sys.exit(1)

    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    # Phase 1: stats + Counter
    counter, t_min, t_max = scan_stats(files)
    logging.info("Scanned %d pickles · %d unique timestamps · span %.6f → %.6f",
                 len(files), len(counter), t_min, t_max)

    # Build equal-frequency edges
    edges = make_equalfreq_edges(counter, bins=args.bins)
    logging.info("Built %d equal-frequency bins", args.bins)

    # Phase 2: per-bin processing
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.bins):
        start, end = edges[i], edges[i + 1]
        logging.info("Bin %02d  [%f, %f)", i, start, end)
        merged_iter = merge_bin(files, start, end)
        out = args.output_dir / f"interval_{i:02d}.pkl"
        write_pickle_stream(merged_iter, out)
        logging.info("→ wrote %s", out.name)

if __name__ == "__main__":
    main(sys.argv[1:])
