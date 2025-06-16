"""
zettir_kg.py
Builds a heterogeneous knowledge-graph per MAC address and visualises it.

Requirements
------------
pip install dgl-cu102  # or dgl==X if CPU only
pip install cudf-cu11  # optional GPU DataFrame
pip install pyvis pandas numpy

Usage
-----
python zettir_kg.py \
    --mac 6c:ac:c2:68:07:fd \
    --process_csv TAG_APP_PROCESS_START.csv \
    --socket_csv SOCKET_CREATION.csv \
    --html_out mac-graph.html
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import dgl                     # DGL heterographs :contentReference[oaicite:1]{index=1}
from pyvis.network import Network  # interactive html graphs :contentReference[oaicite:2]{index=2}

# -------------------------------------------------------------------------
# 1. Try cuDF; fall back to pandas (they share a similar API subset)
# -------------------------------------------------------------------------
try:
    import cudf as _pd         # GPU DataFrame, RAPIDS cuDF :contentReference[oaicite:3]{index=3}
    GPU_ENABLED = True
except ImportError:
    import pandas as _pd       # CPU DataFrame
    GPU_ENABLED = False

# -------------------------------------------------------------------------
# 2. Placeholder: convert sine/cosine features -> integer triple (m,d,h)
#    You supply real inverse-transform later.
# -------------------------------------------------------------------------
import math

def sine_cosine_to_mdh(
        sin_m: float, cos_m: float,
        sin_h: float, cos_h: float,
        sin_d: float, cos_d: float
) -> Tuple[int, int, int]:
    """
    Recover discrete month-of-year [1-12], day-of-week [0-6], and hour-of-day [0-23]
    from their sine / cosine encodings.

    The forward encoding was:
        sin = np.sin(value * 2π / period)
        cos = np.cos(value * 2π / period)

    We invert by:
        angle  = atan2(sin, cos)               -> range (-π, π]
        angle += 2π if angle < 0               -> range [0, 2π)
        value = round(angle * period / 2π) % period
    """
    def decode(sin_v: float, cos_v: float, period: int) -> int:
        angle = math.atan2(sin_v, cos_v)           # (-π, π]
        if angle < 0:                              # map to [0, 2π)
            angle += 2 * math.pi
        val = int(round(angle * period / (2 * math.pi))) % period
        return val

    month = decode(sin_m, cos_m, 12) + 1          # 1-12 for readability
    hour  = decode(sin_h, cos_h, 24)              # 0-23
    dow   = decode(sin_d, cos_d, 7) + 1               # 0-6 (Mon=0 if that was your forward def)

    return month, dow, hour

# -------------------------------------------------------------------------
# 3. Graph-building helper
# -------------------------------------------------------------------------
def build_graph_for_mac(
        mac: str,
        df_proc, df_sock) -> dgl.DGLHeteroGraph:
    """
    Build a heterogeneous DGL graph for *one* MAC address.
    """
    # Filter rows for this device
    df_proc = df_proc[df_proc["mac"] == mac]
    df_sock = df_sock[df_sock["mac"] == mac]

    # Dictionaries mapping raw identifiers -> node IDs for each ntype
    node_maps: Dict[str, Dict[str, int]] = defaultdict(dict)
    node_feat  = defaultdict(list)   # store properties if needed

    # Helper to fetch/create node IDs
    def node_id(ntype: str, key: str):
        mapping = node_maps[ntype]
        if key not in mapping:
            mapping[key] = len(mapping)
        return mapping[key]

    # Edge containers
    edge_src, edge_dst, edge_etype = [], [], []

    # -----------------------------------------------------------------
    # Device node (single per MAC)
    device_nid = node_id("Device", mac)

    # -----------------------------------------------------------------
    # ProcessEvent nodes + edges
    for idx, row in df_proc.iterrows():
        event_key = f"proc_{idx}"
        e_nid = node_id("ProcessEvent", event_key)

        # Add basic edges
        edge_src += [e_nid]
        edge_dst += [device_nid]
        edge_etype += ["ON_DEVICE"]

        add_on = row["addon_content__pkgName"]
        addon_nid = node_id("AddOn", add_on)
        edge_src += [e_nid]
        edge_dst += [addon_nid]
        edge_etype += ["USES_ADDON"]

        # Date-time node
        m, d, h = sine_cosine_to_mdh(
            row["month_of_year_sine"],   row["month_of_year_cosine"],
            row["hour_of_day_sine"],     row["hour_of_day_cosine"],
            row["day_of_week_sine"],     row["day_of_week_cosine"]
        )
        dt_key = f"{m}-{d}-{h}"
        dt_nid = node_id("DateTime", dt_key)
        edge_src += [e_nid]
        edge_dst += [dt_nid]
        edge_etype += ["AT_TIME"]

    # -----------------------------------------------------------------
    # SocketEvent nodes + edges
    for idx, row in df_sock.iterrows():
        event_key = f"sock_{idx}"
        e_nid = node_id("SocketEvent", event_key)

        # edges to device / add-on
        edge_src += [e_nid, e_nid]
        edge_dst += [device_nid,
                     node_id("AddOn", row["addon_content__pkgName"])]
        edge_etype += ["ON_DEVICE", "USES_ADDON"]

        # Host node (full IP)
        ip = ".".join(str(int(row[f"remote_octet_{i}"])) for i in range(1, 5))
        host_nid = node_id("Host", ip)
        edge_src.append(e_nid); edge_dst.append(host_nid); edge_etype.append("TO_HOST")

        # Date-time node
        m, d, h = sine_cosine_to_mdh(
            row["month_of_year_sine"],   row["month_of_year_cosine"],
            row["hour_of_day_sine"],     row["hour_of_day_cosine"],
            row["day_of_week_sine"],     row["day_of_week_cosine"]
        )
        dt_nid = node_id("DateTime", f"{m}-{d}-{h}")
        edge_src.append(e_nid); edge_dst.append(dt_nid); edge_etype.append("AT_TIME")

    # -----------------------------------------------------------------
    # Build DGL heterograph
    # Convert lists -> tensors
    edge_src = torch.tensor(edge_src)
    edge_dst = torch.tensor(edge_dst)
    edge_etype = np.array(edge_etype)

    # Gather edges per canonical etype (src_ntype, rel, dst_ntype)
    etype_dict = defaultdict(lambda: ([], []))  # (srcs, dsts)
    # Map node-id back to ntype
    nid2ntype = {}
    for ntype, mapping in node_maps.items():
        for k, nid in mapping.items():
            nid2ntype[nid] = ntype

    for s, t, rel in zip(edge_src, edge_dst, edge_etype):
        k = (nid2ntype[int(s)], rel, nid2ntype[int(t)])
        etype_dict[k][0].append(int(s))
        etype_dict[k][1].append(int(t))

    # tensor-ise
    edge_data = {k: (torch.tensor(v[0]), torch.tensor(v[1]))
                 for k, v in etype_dict.items()}

    G = dgl.heterograph(edge_data)   # build heterograph :contentReference[oaicite:4]{index=4}
    return G, node_maps


# -------------------------------------------------------------------------
# 4. Visualise with PyVis
# -------------------------------------------------------------------------
COLOR_MAP = {
    "Device":       "#2ca02c",
    "AddOn":        "#ff7f0e",
    "Host":         "#d62728",
    "DateTime":     "#9467bd",
    "ProcessEvent": "#1f77b4",
    "SocketEvent":  "#e377c2",
}

EDGE_COLOR = {
    "ON_DEVICE":  "grey",
    "USES_ADDON": "black",
    "TO_HOST":    "red",
    "AT_TIME":    "purple",
}

def visualise_pyvis(G: dgl.DGLHeteroGraph,
                    node_maps: Dict[str, Dict[str, int]],
                    html_path="graph.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.toggle_physics(True)

    # Add nodes with colours
    for ntype, mapping in node_maps.items():
        for key, nid in mapping.items():
            net.add_node(
                nid,
                label=f"{ntype}:{key}" if len(str(key)) < 25 else f"{ntype}",
                color=COLOR_MAP.get(ntype, "gray"),
                title=key
            )

    # Add edges
    for (src_ntype, rel, dst_ntype) in G.canonical_etypes:
        src, dst = G.edges(etype=(src_ntype, rel, dst_ntype))
        for s, t in zip(src.tolist(), dst.tolist()):
            net.add_edge(
                s, t,
                color=EDGE_COLOR.get(rel, "black"),
                title=rel
            )

    net.save_graph(html_path)      # save as interactive HTML :contentReference[oaicite:5]{index=5}
    print(f"PyVis graph saved to {html_path}")

# -------------------------------------------------------------------------
# 5. Command-line wrapper
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mac", required=True, help="MAC address to visualise")
    ap.add_argument("--process_csv", required=True)
    ap.add_argument("--socket_csv",  required=True)
    ap.add_argument("--html_out",    default="device_graph.html")
    args = ap.parse_args()

    # Read CSVs (cuDF or pandas) :contentReference[oaicite:6]{index=6}
    read_csv = _pd.read_csv
    df_proc  = read_csv(args.process_csv)
    df_sock  = read_csv(args.socket_csv)

    # Build graph
    G, node_maps = build_graph_for_mac(args.mac, df_proc, df_sock)
    print(f"Graph for {args.mac}: {G}")

    # Visualise
    visualise_pyvis(G, node_maps, html_path=args.html_out)

if __name__ == "__main__":
    main()