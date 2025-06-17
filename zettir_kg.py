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

# zitter_graph_no_device.py
"""
Build per-MAC heterogeneous graph *without* a Device node and visualise it.

Node types :  ProcessEvent , SocketEvent , AddOn , DateTime , Host
Edge types  :  (see table above)

pip install dgl-cu118  # or dgl for CPU only
pip install pyvis pandas numpy
"""

import argparse, math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch, dgl                                   # DGL heterographs :contentReference[oaicite:4]{index=4}
from pyvis.network import Network                   # interactive HTML :contentReference[oaicite:5]{index=5}
try:
    import cudf as gd                               # optional GPU frame
    GPU = True
except ImportError:
    import pandas as gd
    GPU = False

# ─────────────────────────────────────────────────────────────────────
# 1. exact inverse of your trig transform
# ─────────────────────────────────────────────────────────────────────
def sine_cosine_to_mdh(sin_m, cos_m, sin_h, cos_h, sin_d, cos_d) -> Tuple[int, int, int]:
    def decode(s, c, period):
        ang = math.atan2(float(s), float(c))
        if ang < 0: ang += 2*math.pi
        return int(round(ang * period / (2*math.pi))) % period
    month = decode(sin_m, cos_m, 12) + 1   # 1-12
    hour  = decode(sin_h, cos_h, 24)       # 0-23
    dow   = decode(sin_d, cos_d, 7)        # 0-6
    return month, dow, hour

# ─────────────────────────────────────────────────────────────────────
# 2. build heterograph for a single MAC
# ─────────────────────────────────────────────────────────────────────
def build_graph(mac: str, df_proc, df_sock):
    # a) keep rows for target MAC and move to pandas for easy iterrows
    df_proc = df_proc[df_proc["mac"] == mac].to_pandas()
    df_sock = df_sock[df_sock["mac"] == mac].to_pandas()

    node_maps: Dict[str, Dict[str, int]] = defaultdict(dict)
    def nid(ntype, key):
        mp = node_maps[ntype]
        if key not in mp: mp[key] = len(mp)
        return mp[key]

    src, dst, rel = [], [], []

    # ---------- ProcessEvent nodes ----------
    for idx, row in df_proc.iterrows():
        pe = nid("ProcessEvent", f"proc_{idx}")
        addon = nid("AddOn", row["addon_content__pkgName"])
        m, d, h = sine_cosine_to_mdh(row["month_of_year_sine"],  row["month_of_year_cosine"],
                                     row["hour_of_day_sine"],    row["hour_of_day_cosine"],
                                     row["day_of_week_sine"],    row["day_of_week_cosine"])
        dt = nid("DateTime", f"{m}-{d}-{h}")

        # USES_ADDON (forward & reverse)
        src += [pe, addon];  dst += [addon, pe];  rel += ["USES_ADDON", "USED_BY"]
        # AT_TIME     (forward & reverse)
        src += [pe, dt];     dst += [dt, pe];     rel += ["AT_TIME", "OCCURED"]

    # ---------- SocketEvent nodes ----------
    for idx, row in df_sock.iterrows():
        se = nid("SocketEvent", f"sock_{idx}")
        addon = nid("AddOn", row["addon_content__pkgName"])
        ip = ".".join(str(int(row[f"remote_octet_{i}"])) for i in range(1,5))
        host = nid("Host", ip)
        m, d, h = sine_cosine_to_mdh(row["month_of_year_sine"],  row["month_of_year_cosine"],
                                     row["hour_of_day_sine"],    row["hour_of_day_cosine"],
                                     row["day_of_week_sine"],    row["day_of_week_cosine"])
        dt = nid("DateTime", f"{m}-{d}-{h}")

        # USES_ADDON (fwd & rev)
        src += [se, addon];  dst += [addon, se];  rel += ["USES_ADDON", "USED_BY"]
        # TO_HOST     (fwd & rev)
        src += [se, host];   dst += [host, se];   rel += ["TO_HOST", "HOSTED"]
        # AT_TIME     (fwd & rev)
        src += [se, dt];     dst += [dt, se];     rel += ["AT_TIME", "OCCURED"]

    # ---------- group edges by canonical etype ----------
    nid2t = {nid: t for t, mp in node_maps.items() for nid in mp.values()}
    buckets = defaultdict(lambda: ([], []))        # (srcs, dsts)
    for s, t, r in zip(src, dst, rel):
        buckets[(nid2t[s], r, nid2t[t])][0].append(s)
        buckets[(nid2t[s], r, nid2t[t])][1].append(t)

    edge_dict = {k: (torch.tensor(v[0]), torch.tensor(v[1])) for k,v in buckets.items()}
    G = dgl.heterograph(edge_dict, num_nodes_dict={t: len(mp) for t,mp in node_maps.items()})
    return G, node_maps

# ─────────────────────────────────────────────────────────────────────
# 3. PyVis visualisation
# ─────────────────────────────────────────────────────────────────────
NODE_COLOR = dict(ProcessEvent="#1f77b4", SocketEvent="#e377c2",
                  AddOn="#ff7f0e", DateTime="#9467bd", Host="#d62728")
EDGE_COLOR = dict(USES_ADDON="black", USED_BY="gray",
                  TO_HOST="red", HOSTED="orange",
                  AT_TIME="purple", OCCURED="green")

def visualise(G, mp, html="graph.html"):
    net = Network(height="750px", width="100%", directed=True)
    net.toggle_physics(True)
    for ntype, m in mp.items():
        for key, nid in m.items():
            lab = key if len(str(key))<25 else key[:22]+"…"
            net.add_node(nid, label=lab, color=NODE_COLOR.get(ntype,"grey"),
                         title=f"{ntype}: {key}")
    for (snt, rel, dnt) in G.canonical_etypes:
        s, d = G.edges(etype=(snt, rel, dnt))
        for a,b in zip(s.tolist(), d.tolist()):
            net.add_edge(a,b,color=EDGE_COLOR.get(rel,"black"),title=rel)
    net.save_graph(html); print("PyVis saved →", html)

# ─────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────
def main():
    ag = argparse.ArgumentParser()
    ag.add_argument("--mac", required=True)
    ag.add_argument("--process_csv", required=True)
    ag.add_argument("--socket_csv", required=True)
    ag.add_argument("--html_out", default="graph.html")
    a = ag.parse_args()

    df_p = gd.read_csv(a.process_csv)
    df_s = gd.read_csv(a.socket_csv)

    G, mp = build_graph(a.mac, df_p, df_s)
    print(G)                                              # quick summary
    visualise(G, mp, a.html_out)

if __name__ == "__main__":
    main()

    
# Several improvements. First, remove the ‘Device’ node, since we only have one. Also, we need to clarify the edge types. firstly, AddOn ndoes are connected to ProcessEvent via relationships (processevent, uses addon, addon) and (addon, used by, processevent). secondly, AddOn nodes are connected to SocketEvent via relationships (socketevent, uses addon, addon) and (addon, used by, socketevent). thirdly, SocketEvent nodes are connected to Host nodes via relationships (socketevent, to host, host) and (host, hosted, socketevent). finally, DateTime nodes are connected to ProcessEvent and SocketEvent nodes via relationships (processevent, at time, datetime), (datetime, occured, processevent), (socketevent, at time, datetime), and (datetime, occured, socketevent). please update the code to reflect these changes.
