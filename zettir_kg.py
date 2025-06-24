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


import pandas as pd
import numpy as np
import dgl
import torch
import os

def create_dummy_frequency_csv(num_samples=50):
    """
    Generates a dummy CSV file that matches your description, including the
    specific column naming conventions and 'Class' for the label.
    """
    print(f"Creating a dummy CSV file with {num_samples} samples...")
    
    # Define feature names that follow the user's rules
    syscall_features = ['read', 'write', 'openat', 'execve', 'chmod', 'futex']
    binder_features = ['sendSMS', 'getDeviceId', 'startActivity', 'queryContentProviders']
    composite_features = ['NETWORK_WRITE_EXEC', 'READ_CONTACTS(D)', 'DYNAMIC_CODE_LOADING']
    
    features = syscall_features + binder_features + composite_features
    
    # Create random frequency data
    data = np.random.randint(0, 30, size=(num_samples, len(features)))
    df = pd.DataFrame(data, columns=features)
    
    # Make some data zero to ensure not all nodes are connected
    df.loc[df.sample(frac=0.3).index, np.random.choice(df.columns, 3)] = 0
    
    # Add an application ID column
    df.insert(0, 'app_id', [f'app_{i}' for i in range(num_samples)])
    
    # Generate labels based on a simple rule and name the column 'Class'
    labels = []
    for i, row in df.iterrows():
        malicious_score = (
            row['execve'] + row['sendSMS'] * 2 + row['NETWORK_WRITE_EXEC'] * 3
        )
        labels.append('malware' if malicious_score > 70 else 'benign')
            
    df['Class'] = labels # The label column is now named 'Class'
    
    file_path = 'app_behavior_frequencies.csv'
    df.to_csv(file_path, index=False)
    print(f"Dummy data saved to '{file_path}'")
    return file_path

def classify_feature_name(name):
    """Classifies a column name as syscall, binder, or composite."""
    if name.islower():
        return 'syscall'
    if not any(c.islower() for c in name):
        return 'composite'
    return 'binder'

def create_heterogeneous_graph(csv_path):
    """
    Processes a frequency-based CSV into a DGL heterogeneous graph with
    reciprocal edges and NO node features.
    """
    print(f"\n--- Starting Heterogeneous Graph Construction from {csv_path} ---")

    # 1. Load Data
    print("\nStep 1: Loading data...")
    df = pd.read_csv(csv_path)

    # 2. Identify Node Types and Create Mappings
    print("Step 2: Identifying node types and creating ID mappings...")
    app_ids = df['app_id'].unique()
    app_map = {name: i for i, name in enumerate(app_ids)}
    
    # Get action columns, ignoring app_id and the new 'Class' label column
    action_cols = [col for col in df.columns if col not in ['app_id', 'Class']]
    syscall_nodes = [col for col in action_cols if classify_feature_name(col) == 'syscall']
    binder_nodes = [col for col in action_cols if classify_feature_name(col) == 'binder']
    composite_nodes = [col for col in action_cols if classify_feature_name(col) == 'composite']

    syscall_map = {name: i for i, name in enumerate(syscall_nodes)}
    binder_map = {name: i for i, name in enumerate(binder_nodes)}
    composite_map = {name: i for i, name in enumerate(composite_nodes)}
    
    print(f"- Found {len(app_map)} 'application' nodes.")
    print(f"- Found {len(syscall_map)} 'syscall' nodes.")
    print(f"- Found {len(binder_map)} 'binder' nodes.")
    print(f"- Found {len(composite_map)} 'composite_behavior' nodes.")

    # 3. Prepare Edge Lists and Edge Features
    print("Step 3: Building edge lists and feature weights...")
    app_src, syscall_dst, syscall_freq = [], [], []
    app_src_b, binder_dst, binder_freq = [], [], []
    app_src_c, composite_dst, composite_freq = [], [], []

    for _, row in df.iterrows():
        current_app_id = app_map[row['app_id']]
        for action_name in syscall_nodes:
            if row[action_name] > 0:
                app_src.append(current_app_id)
                syscall_dst.append(syscall_map[action_name])
                syscall_freq.append(row[action_name])
        for action_name in binder_nodes:
            if row[action_name] > 0:
                app_src_b.append(current_app_id)
                binder_dst.append(binder_map[action_name])
                binder_freq.append(row[action_name])
        for action_name in composite_nodes:
            if row[action_name] > 0:
                app_src_c.append(current_app_id)
                composite_dst.append(composite_map[action_name])
                composite_freq.append(row[action_name])

    # 4. Prepare Node Labels (No Node Features)
    print("Step 4: Preparing node labels (no node features)...")
    # Convert 'malware'/'benign' labels from 'Class' column to a numeric tensor (1/0)
    app_labels = torch.tensor([1 if label == 'malware' else 0 for label in df['Class']])
    print("- ALL node types will have no input features ('feat').")
    print("- 'application' nodes will have a ground-truth 'label' for training/evaluation.")


    # 5. Construct the Graph with Reciprocal Edges
    print("Step 5: Assembling the DGL graph with reciprocal edges...")
    graph_data = {
        # Forward edges (application -> action)
        ('application', 'uses', 'syscall'): (app_src, syscall_dst),
        ('application', 'uses', 'binder'): (app_src_b, binder_dst),
        ('application', 'exhibits', 'composite_behavior'): (app_src_c, composite_dst),
        
        # Reciprocal edges (action -> application)
        ('syscall', 'used_by', 'application'): (syscall_dst, app_src),
        ('binder', 'used_by', 'application'): (binder_dst, app_src_b),
        ('composite_behavior', 'exhibited_by', 'application'): (composite_dst, app_src_c)
    }
    
    g = dgl.heterograph(graph_data, num_nodes_dict={
        'application': len(app_map),
        'syscall': len(syscall_map),
        'binder': len(binder_map),
        'composite_behavior': len(composite_map)
    })

    # 6. Add Data to Graph (Labels and Edge Weights)
    print("Step 6: Adding ground-truth labels and edge weights to graph...")
    # Add ground-truth labels to application nodes for training/evaluation
    g.nodes['application'].data['label'] = app_labels
    
    # Add frequency weights to forward edges
    g.edges['uses', 'syscall'].data['frequency'] = torch.tensor(syscall_freq, dtype=torch.float32)
    g.edges['uses', 'binder'].data['frequency'] = torch.tensor(binder_freq, dtype=torch.float32)
    g.edges['exhibits', 'composite_behavior'].data['frequency'] = torch.tensor(composite_freq, dtype=torch.float32)

    # Add frequency weights to reciprocal edges
    g.edges['used_by', 'syscall'].data['frequency'] = torch.tensor(syscall_freq, dtype=torch.float32)
    g.edges['used_by', 'binder'].data['frequency'] = torch.tensor(binder_freq, dtype=torch.float32)
    g.edges['exhibited_by', 'application'].data['frequency'] = torch.tensor(composite_freq, dtype=torch.float32)

    print("\nGraph construction complete!")
    return g

if __name__ == '__main__':
    data_file_path = create_dummy_frequency_csv()
    hetero_graph = create_heterogeneous_graph(data_file_path)

    # --- Verification ---
    if hetero_graph:
        print("\n--- Graph Summary ---")
        print(hetero_graph)
        
        # Verify node features are now gone
        print("\n--- Node Feature Inspection ---")
        for ntype in hetero_graph.ntypes:
            # Check if the .data dictionary is empty or only contains 'label'
            if not hetero_graph.nodes[ntype].data or list(hetero_graph.nodes[ntype].data.keys()) == ['label']:
                 print(f"Node type '{ntype}' correctly has no input features ('feat').")
            else:
                 print(f"Node type '{ntype}' INCORRECTLY has features: {hetero_graph.nodes[ntype].data.keys()}")
        
        print("\nGround truth label for first application node:", hetero_graph.nodes['application'].data['label'][0])
        print("Frequency weight on first ('application', 'uses', 'syscall') edge:", hetero_graph.edges['uses', 'syscall'].data['frequency'][0])

        os.remove(data_file_path)
        print("\nCleaned up dummy CSV file.")
