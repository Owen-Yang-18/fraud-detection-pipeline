#!/usr/bin/env python3
"""
triples2hetero_edgeclass.py
---------------------------
Read each *.pkl of (timestamp, action, filename) triples, class-ify the
action, build a DGL heterograph with **edge types that include the action
class**, add the reciprocal edge, and save to --graph_dir/<stem>.bin.

Node types
----------
file        : every unique filename
syscall     : action  .islower()  (lower-case or specials)  :contentReference[oaicite:1]{index=1}
composite   : action  .isupper()  (upper-case or specials)  :contentReference[oaicite:2]{index=2}
binder      : everything else

Edge types
----------
("file",      "calls_<atype>",        <atype>)
(<atype>,     "<atype>_called_by",    "file")

DGL concept reference: canonical edge triplets  :contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.
"""
from __future__ import annotations
import argparse, collections, gc, pickle, sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import dgl                                    # heterograph API  :contentReference[oaicite:5]{index=5}

Triplet = Tuple[float, str, str]              # (ts, action, filename)

# ── helpers ────────────────────────────────────────────────────────────── #
def classify(act: str) -> str:
    lo, up = any(c.islower() for c in act), any(c.isupper() for c in act)
    if lo and not up:
        return "syscall"
    if up and not lo:
        return "composite"
    return "binder"

def build_graph(triples: List[Triplet]) -> dgl.DGLHeteroGraph:
    edict: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = collections.defaultdict(list)
    nid : Dict[str, Dict[str, int]] = collections.defaultdict(dict)

    def idx(ntype: str, key: str) -> int:
        tbl = nid[ntype]
        if key not in tbl:
            tbl[key] = len(tbl)
        return tbl[key]

    # --- populate ------------------------------------------------------- #
    for _, act, fname in triples:
        atype  = classify(act)
        f_id   = idx("file", fname)
        a_id   = idx(atype,  act)

        # forward edge
        et_fwd = f"calls_{atype}"
        edict[("file", et_fwd, atype)].append((f_id, a_id))

        # reverse edge
        et_rev = f"{atype}_called_by"
        edict[(atype, et_rev, "file")].append((a_id, f_id))

    # --- convert to tensors & heterograph ------------------------------- #
    data_dict = {trip: (torch.tensor([u for u, _ in pairs]),
                        torch.tensor([v for _, v in pairs]))
                 for trip, pairs in edict.items()}

    num_nodes = {ntype: len(tbl) for ntype, tbl in nid.items()}
    return dgl.heterograph(data_dict, num_nodes_dict=num_nodes)  # API spec  :contentReference[oaicite:6]{index=6}

# ── driver ─────────────────────────────────────────────────────────────── #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build heterographs w/ class-aware edge types")
    ap.add_argument("--input_dir",  required=True, type=Path)
    ap.add_argument("--graph_dir",  required=True, type=Path)
    args = ap.parse_args()

    args.graph_dir.mkdir(parents=True, exist_ok=True)
    pkls = sorted(args.input_dir.glob("*.pkl"))
    if not pkls:
        print("No *.pkl files found.", file=sys.stderr); sys.exit(1)

    for pkl in pkls:
        with pkl.open("rb") as fh:
            triples: List[Triplet] = pickle.load(fh)

        g = build_graph(triples)
        out = args.graph_dir / f"{pkl.stem}.bin"
        dgl.save_graphs(str(out), [g])                     # persist  :contentReference[oaicite:7]{index=7}

        print(f"{pkl.name:<25} → {out.name:<20}  "
              f"ntypes={g.ntypes},  etypes={g.etypes}, "
              f"#edges={g.num_edges()}")
        del triples, g; gc.collect()

if __name__ == "__main__":
    main()
