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
import torch, dgl                                   # DGL heterographs
from pyvis.network import Network                   # interactive HTML
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
# --------------------------------------------------------------------
# 1.  Build heterograph for a single MAC (no Device node)
# --------------------------------------------------------------------
def build_graph(mac: str, df_proc, df_sock):
    # ── filter rows for this device & move to pandas for simple iterrows
    df_proc = df_proc[df_proc["mac"] == mac].to_pandas()
    df_sock = df_sock[df_sock["mac"] == mac].to_pandas()

    # ── per-type node dictionaries
    node_maps: Dict[str, Dict[str, int]] = defaultdict(dict)

    def nid(ntype: str, key: str) -> int:
        """Return (or create) a node-id local to its node-type."""
        mp = node_maps[ntype]
        if key not in mp:
            mp[key] = len(mp)
        return mp[key]

    # ── edge collector: dict[(src_ntype, rel, dst_ntype)] → (src_ids, dst_ids)
    edge_dict: Dict[Tuple[str, str, str], Tuple[list, list]] = defaultdict(lambda: ([], []))

    def add_edge(s_type, s_id, rel, d_type, d_id):
        l = edge_dict[(s_type, rel, d_type)]
        l[0].append(s_id)
        l[1].append(d_id)

    # ── ProcessEvent nodes & edges
    for idx, row in df_proc.iterrows():
        pe = nid("ProcessEvent", f"proc_{idx}")
        addon = nid("AddOn",      str(row["addon_content__pkgName"]))

        m, d, h = sine_cosine_to_mdh(row["month_of_year_sine"],  row["month_of_year_cosine"],
                                     row["hour_of_day_sine"],    row["hour_of_day_cosine"],
                                     row["day_of_week_sine"],    row["day_of_week_cosine"])
        dt = nid("DateTime",      f"{m}-{d}-{h}")

        # ProcessEvent ↔ AddOn
        add_edge("ProcessEvent", pe, "USES_ADDON", "AddOn", addon)
        add_edge("AddOn",        addon, "USED_BY",   "ProcessEvent", pe)
        # ProcessEvent ↔ DateTime
        add_edge("ProcessEvent", pe, "AT_TIME",     "DateTime", dt)
        add_edge("DateTime",     dt, "OCCURED",     "ProcessEvent", pe)

    # ── SocketEvent nodes & edges
    for idx, row in df_sock.iterrows():
        se = nid("SocketEvent", f"sock_{idx}")
        addon = nid("AddOn",     str(row["addon_content__pkgName"]))
        ip = ".".join(str(int(row[f"remote_octet_{i}"])) for i in range(1, 5))
        host = nid("Host", ip)

        m, d, h = sine_cosine_to_mdh(row["month_of_year_sine"],  row["month_of_year_cosine"],
                                     row["hour_of_day_sine"],    row["hour_of_day_cosine"],
                                     row["day_of_week_sine"],    row["day_of_week_cosine"])
        dt = nid("DateTime", f"{m}-{d}-{h}")

        # SocketEvent ↔ AddOn
        add_edge("SocketEvent", se, "USES_ADDON", "AddOn", addon)
        add_edge("AddOn",       addon, "USED_BY",  "SocketEvent", se)

        # SocketEvent ↔ Host
        add_edge("SocketEvent", se, "TO_HOST", "Host", host)
        add_edge("Host",        host, "HOSTED", "SocketEvent", se)

        # SocketEvent ↔ DateTime
        add_edge("SocketEvent", se, "AT_TIME", "DateTime", dt)
        add_edge("DateTime",    dt, "OCCURED", "SocketEvent", se)

    # ── build DGL heterograph
    dgl_edges = {
        k: (torch.tensor(v[0]), torch.tensor(v[1]))
        for k, v in edge_dict.items()
    }
    num_nodes = {ntype: len(mp) for ntype, mp in node_maps.items()}
    G = dgl.heterograph(dgl_edges, num_nodes_dict=num_nodes)   # canonical-etype safe

    return G, node_maps

# --------------------------------------------------------------------
#  PyVis visualisation with global-unique node IDs  (FIXED)
# --------------------------------------------------------------------
NODE_COLOR = dict(ProcessEvent="#1f77b4", SocketEvent="#e377c2",
                  AddOn="#ff7f0e", DateTime="#9467bd", Host="#d62728")
EDGE_COLOR = dict(USES_ADDON="black", USED_BY="gray",
                  TO_HOST="red", HOSTED="orange",
                  AT_TIME="purple", OCCURED="green")

def visualise(G, node_maps, html="graph.html"):
    """
    Render the heterogeneous graph in PyVis.

    *Every node gets a unique global ID*  (= type_offset + local_id),
    so different node-types never collide inside PyVis.
    """
    net = Network(height="750px", width="100%", directed=True)
    net.toggle_physics(True)

    # 1. allocate a disjoint integer ID range for each node-type
    type_offset: Dict[str, int] = {}
    next_id = 0
    for ntype, mapping in node_maps.items():
        type_offset[ntype] = next_id
        next_id += len(mapping)

    # 2. add nodes with global IDs
    for ntype, mapping in node_maps.items():
        off = type_offset[ntype]
        for local_key, local_id in mapping.items():
            gid = off + local_id                          # global id
            label = (str(local_key) if len(str(local_key)) < 25
                     else str(local_key)[:22] + "…")
            net.add_node(
                gid,
                label=label,
                title=f"{ntype}: {local_key}",
                color=NODE_COLOR.get(ntype, "grey")
            )

    # 3. add edges, remapping local IDs → global IDs on the fly
    for (snt, rel, dnt) in G.canonical_etypes:
        s_local, d_local = G.edges(etype=(snt, rel, dnt))
        s_off, d_off = type_offset[snt], type_offset[dnt]
        for s, d in zip(s_local.tolist(), d_local.tolist()):
            net.add_edge(
                s_off + s,
                d_off + d,
                title=rel,
                color=EDGE_COLOR.get(rel, "black")
            )

    net.save_graph(html)
    print(f"PyVis HTML saved → {html}")

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

# 1. event id: like log id, like 82864
# 2. time stamp: in the form of yyyy-mm-dd hh:mm:ss.miliseconds, like 2023-10-01 12:34:56.789
# 3. version: like 1
# 4. name: event name like SOCKET_CREATION, SOCKET_TERMINATION, PROCESS_CREATION, PROCESS_TERMINATION, SOCKET_UPDATE, ACCESSIBILITY_EVENT, TAG_APP_PROCESS_START, TAG_APP_PROCESS_END, PROCESS_PERMISSION_MODIFICATION, etc.
# 5. severity: like low, medium, high
# 6. maturity: like RAW_DATA
# 7. source: like ANDROID_SECURITY_LOG, EBPF
# 8. private: like PUBLIC, PRIVATE
# 9. tag_table: like generic_events, socket_events, process_events, etc.
# 10. mitre_attack_technique: like T1059, T1071, T1105, etc, but mostly are None
# 11. server_ack: like 0.0
# 12. mac: like 6c:ac:c2:68:07:fd
# 13: model: like SM-S921U
# 14: addon_content__result: mostly NaN
# 15: addon_content__keyAlias: mostly nan
# 16. addon_content_uid: most NaN
# 17. addon_content__timestamp: like 1.729533e+12
# 18. addon_content_timeZone: like America/Los_Angeles
# 19. addon_content__locale: like en_US
# 20. tid: like 4566.0 or 12219.0
# 21. pid: like 4566.0 or 12098.0
# 22. ppid: like 2.0 or 1756.0
# 23. uid: like 10335.0
# 24. gid: like 10229.0
# 25. exit_code: like 0.0 or nan
# 26. syscall: like 221.0, 0.0, or nan
# 27. path: like /system/bin/app_process64, /system/bin/sdcard, etc, or nan
# 28. family: like 2.0, 10.0, or nan
# 29. type: like 1.0 or nan
# 30. protocol: like 6.0, 17.0, or nan
# 31. local_address: like ::ffff:10.0.0.77
# 32. remote_address: like ::ffff:44.216.98.239
# 33. local_port: like 55544.0
# 34. remote_port: like 443.0
# 35. fingerprint: like a hash or nan or None
# 36. addon_content__unknownSource: like False, True, or nan
# 37. addon_content__pkgName: like com.android.chrome, com.google.android.youtube
# 38. cwd: like nan or /
# 39. cmdline: like /system/bin/app_process64 /system/bin com/xingin/tiny/daemon/e
# 40. euid: mostly 10340.0 or NaN
# 41. egid: mostly 10340.0 or NaN
# 42. fsuid: mostly 10340.0 or NaN
# 43. fsgid: mostly 10340.0 or NaN
# 44. suid: mostly 10340.0 or NaN
# 45. sgid: mostly 10340.0 or NaN
# 46. owner_uid: mostly 0.0 or NaN
# 47. owner_gid: mostly 0.0 or NaN
# 48. atime: like 1.729533e+12 or NaN
# 49. mtime: like 1.729533e+12 or NaN
# 50. ctime: like 1.729533e+12 or NaN
# 51. package_name
# 52. accessibility_api
# 53. restricted_permission
# 54. addon_content__startTime
# 55. addon_content__pid
# 56. addon_content__seTag
# 57. addon_content__hash
# 58. timestamp_unix: like 1.729533e+09
# 59. event_date: yyyy-mm-dd like 2023-10-01
# 60. addon_content__strong
# 61. addon_content__inetFamily
# 62. addon_content__eventType
# 63. addon_content__interfaceName
# 64. addon_content__protocol
# 65. addon_content__remoteAddress
# 66. addon_content__remotePort
# 67. addon_content__sourceAddress
# 68. addon_content__sourcePort