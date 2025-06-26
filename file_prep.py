from __future__ import annotations

import argparse
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import torch

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------

def action_case_code(action: str) -> int:
    """Return 0 (all upper), 1 (all lower), 2 (mixed) ignoring non‑letters."""
    letters = [c for c in action if c.isalpha()]
    if not letters:
        return 2  # Edge‑case: treat empty or non‑alpha as mixed
    if all(c.isupper() for c in letters):
        return 0
    if all(c.islower() for c in letters):
        return 1
    return 2


def parse_time_from_filename(filename: str, pattern: re.Pattern[str]) -> int:
    """Extract the <time> integer from a file name like 'interval_123.pkl'."""
    m = pattern.match(filename)
    if not m:
        raise ValueError(f"File name '{filename}' does not match expected pattern")
    return int(m.group(1))


# -------------------------------------------------------------
# Core loader
# -------------------------------------------------------------

def load_directory(dir_path: Path) -> Tuple[List[str], List[str], List[int], List[int]]:
    """Walk *dir_path* in sorted order and accumulate src, dst, t, msg lists."""
    if not dir_path.is_dir():
        raise FileNotFoundError(dir_path)

    file_re = re.compile(r"^interval_(\d{3})\.pkl$")

    src: List[str] = []
    dst: List[str] = []
    t:   List[int] = []
    msg: List[int] = []

    for entry in sorted(dir_path.iterdir(), key=lambda p: p.name):
        if not entry.is_file():
            continue
        try:
            time_val = parse_time_from_filename(entry.name, file_re)
        except ValueError:
            continue  # skip files that do not match the pattern

        with entry.open("rb") as fh:
            triples = pickle.load(fh)

        if not isinstance(triples, list):
            raise TypeError(f"{entry}: expected a list, got {type(triples).__name__}")

        for ts, action, filename in triples:
            # Append values according to the rule set
            t.append(time_val)
            src.append(str(filename))
            dst.append(str(action))
            msg.append(action_case_code(str(action)))

    return src, dst, t, msg


# -------------------------------------------------------------
# Mapping utilities
# -------------------------------------------------------------

def build_id_map(strings: List[str], offset: int = 0) -> Dict[str, int]:
    """Map each unique string in *strings* to a unique integer starting at *offset*."""
    unique_sorted = sorted(set(strings))
    return {s: i + offset for i, s in enumerate(unique_sorted)}


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert interval PKLs to torch tensors")
    parser.add_argument("directory", type=Path, help="Directory containing interval_*.pkl files")
    parser.add_argument("--out", type=Path, default=Path("data_tensors.pt"), help="Output .pt file")
    args = parser.parse_args()

    src_str, dst_str, t_list, msg_list = load_directory(args.directory)

    # Build mappings
    filename2id = build_id_map(src_str, offset=0)
    action2id   = build_id_map(dst_str, offset=len(filename2id))

    # Integer‑encode src & dst lists
    src_ids = [filename2id[s] for s in src_str]
    dst_ids = [action2id[d]   for d in dst_str]

    # Convert to tensors
    src_tensor = torch.tensor(src_ids, dtype=torch.long)
    dst_tensor = torch.tensor(dst_ids, dtype=torch.long)
    t_tensor   = torch.tensor(t_list, dtype=torch.float)
    msg_tensor = torch.tensor(msg_list, dtype=torch.long)

    # Save everything as a single torch bundle
    torch.save({
        "src": src_tensor,
        "dst": dst_tensor,
        "t":   t_tensor,
        "msg": msg_tensor,
        "filename2id": filename2id,
        "action2id":   action2id,
    }, args.out)

    print(f"Saved tensors to {args.out.resolve()}")


if __name__ == "__main__":
    main()
