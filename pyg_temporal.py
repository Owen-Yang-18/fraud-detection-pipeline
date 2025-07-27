import os
import torch
import numpy as np
from datetime import datetime
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.attention import STConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.data.temporal_signal import temporal_signal_split
from torch_geometric_temporal.signal.static_graph_temporal_signal import temporal_signal_split as static_split

# 1. Load .pt files and build DynamicGraphTemporalSignal per file
def load_dynamic_graphs(folder, snapshot_interval: str = "D"):
    graphs = []
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        data = torch.load(os.path.join(folder, fname))
        src, dst, t, w = data.src, data.dst, data.t, data.msg
        df = np.rec.fromarrays([
            src.numpy(), dst.numpy(), t.numpy(), w.numpy()
        ], names=['src','dst','t','w'])
        # time binning
        df.sort(order='t')
        times = np.unique(np.floor(df['t'] / (24*3600)))  # daily bins
        edge_indices, edge_weights = [], []
        for ti in times:
            mask = (df['t'] // (24*3600)) == ti
            e = np.vstack([df['src'][mask], df['dst'][mask]])
            edge_indices.append(e)
            edge_weights.append(df['w'][mask])
        # dummy node features and targets (e.g. all ones)
        num_nodes = int(max(df['src'].max(), df['dst'].max()) + 1)
        features = [np.ones((num_nodes, 1), dtype=float)] * len(times)
        targets  = [np.zeros((num_nodes,), dtype=int)] * len(times)
        graphs.append(DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets))
    return graphs

# 2. Create model wrapper
class STConvClassifier(torch.nn.Module):
    def __init__(self, node_features=1, hidden_channels=16, kernel_size=3, num_classes=2):
        super().__init__()
        self.stconv = STConv(in_channels=node_features, out_channels=hidden_channels,
                             K=2, kernel_size=kernel_size)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_weight):
        h = self.stconv(x, edge_index, edge_weight)
        # global mean pool
        h = h.mean(dim=1)  # (batch, features)
        return self.lin(h)

# 3. Train on all graphs
def train_all(graphs, graph_labels, epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = graphs  # list of DynamicGraphTemporalSignal
    assert len(dataset) == len(graph_labels)
    all_data = []
    for signal, g_label in zip(dataset, graph_labels):
        # convert each snapshot into Data, attach graph-level label
        seq = []
        for edge_index, edge_weight, x in zip(signal.edge_indices,
                                               signal.edge_weights,
                                               signal.features):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            data.y_graph = torch.tensor([g_label], dtype=torch.long)
            seq.append(data)
        all_data.append(seq)

    model = STConvClassifier(num_classes=max(graph_labels)+1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for seq in all_data:
            opt.zero_grad()
            # process sequence
            h_seq = []
            for data in seq:
                data = data.to(device)
                out = model(data.x.unsqueeze(0), data.edge_index, data.edge_weight)
                h_seq.append(out)
            logits = torch.stack(h_seq, dim=0).mean(dim=0)
            loss = F.cross_entropy(logits, seq[0].y_graph.to(device))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:3d}, loss: {total_loss/len(all_data):.4f}")
    return model

if __name__ == "__main__":
    folder = "/path/to/pt/folder"
    graphs = load_dynamic_graphs(folder)
    labels = [0 if "classA" in fname else 1 for fname in os.listdir(folder) if fname.endswith(".pt")]
    model = train_all(graphs, labels)


import os

# Load class-to-prefix mappings from .txt files
class_prefixes = {}  # class_name -> set of file prefixes

txt_folder = "/path/to/txt_folder"
for fname in os.listdir(txt_folder):
    if not fname.endswith(".txt"):
        continue
    class_name = os.path.splitext(fname)[0]  # filename without .txt
    prefixes = set()
    with open(os.path.join(txt_folder, fname), "r") as f:
        for line in f:
            prefix = line.strip().split(".json")[0]
            prefixes.add(prefix)
    class_prefixes[class_name] = prefixes

# Rename .pt files with class prefix
pt_folder = "/path/to/pt_folder"
for pt in os.listdir(pt_folder):
    if not pt.endswith(".pt"):
        continue
    prefix = os.path.splitext(pt)[0]
    assigned = None
    for cls, prefixes in class_prefixes.items():
        if prefix in prefixes:
            assigned = cls
            break

    if assigned:
        src_path = os.path.join(pt_folder, pt)
        new_name = f"{assigned}_{pt}"
        dst_path = os.path.join(pt_folder, new_name)
        # Optionally check for collisions
        if os.path.exists(dst_path):
            raise FileExistsError(f"{dst_path} already exists")
        os.rename(src_path, dst_path)
        print(f"Renamed {pt} → {new_name}")
    else:
        print(f"No class match for {pt}, skipping.")



import os
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.transforms import LaplacianLambdaMax

# 1. Load your dynamic graphs (.pt files) and labels
def load_graph_sequences(pt_folder, graph_labels, snapshot_interval=24*3600):
    sequences = []
    for pt_file, label in graph_labels.items():
        data = torch.load(os.path.join(pt_folder, pt_file))
        df = np.rec.fromarrays([
            data.src.numpy(), data.dst.numpy(),
            data.t.numpy(), data.msg.numpy()
        ], names=['src','dst','t','w'])
        df.sort(order='t')
        times = np.unique(df['t'] // snapshot_interval)

        edge_indices, edge_weights, features = [], [], []
        for ti in times:
            mask = (df['t'] // snapshot_interval) == ti
            s, d = df['src'][mask], df['dst'][mask]
            edge_indices.append(np.vstack([s, d]))
            edge_weights.append(df['w'][mask])
            nodes = np.unique(np.concatenate([s, d]))
            N = nodes.max() + 1
            features.append(np.ones((N, 1), dtype=float))

        label_arr = [label] * len(times)
        sequences.append(DynamicGraphTemporalSignal(
            edge_indices, edge_weights, features, label_arr))
    return sequences

# 2. Model for per-sequence classification
class SequenceGConvLSTM(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, num_classes):
        super().__init__()
        self.recurrent = GConvLSTM(in_channels, out_channels, K)
        self.lin = torch.nn.Linear(out_channels, num_classes)

    def forward(self, x_seq, edge_idx_seq, edge_w_seq, lambda_max):
        H, C = None, None
        for x, eidx, ew in zip(x_seq, edge_idx_seq, edge_w_seq):
            H, C = self.recurrent(
                x, eidx, ew, H=H, C=C, lambda_max=lambda_max)
        hg = H.mean(dim=0, keepdim=True)
        return self.lin(hg)

# 3. Training pipeline
pt_folder = "/path/to/pt"
graph_labels = {"a.pt": 0, "b.pt": 1, ...}

sequences = load_graph_sequences(pt_folder, graph_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SequenceGConvLSTM(in_channels=1, out_channels=16, K=3,
                          num_classes=max(graph_labels.values()) + 1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lap = LaplacianLambdaMax()

for epoch in range(1, 51):
    total_loss = 0
    for signal in sequences:
        xs = [torch.tensor(x, dtype=torch.float).to(device)
              for x in signal.features]
        eidxs = [torch.tensor(e, dtype=torch.long).to(device)
                 for e in signal.edge_indices]
        ews = [torch.tensor(w, dtype=torch.float).to(device)
               for w in signal.edge_weights]
        lmax = lap(torch.tensor(eidxs[0]))['lambda_max'].to(device)

        logits = model(xs, eidxs, ews, lambda_max=lmax)
        y = torch.tensor([signal.targets[0]], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}, Loss: {total_loss/len(sequences):.4f}")


from typing import List, Tuple
import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def make_batches_grouped_by_length(
    signals: List[DynamicGraphTemporalSignal],
    lengths: List[int],
    labels: List[int],
    batch_size: int
) -> Tuple[
    List[List[DynamicGraphTemporalSignal]],
    List[List[int]],
    List[int]
]:
    # Sort indices by increasing length
    idx_sorted = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, batch_labels, batch_lengths = [], [], []

    for i in range(0, len(signals), batch_size):
        idx_batch = idx_sorted[i : i + batch_size]
        # Determine min length in this batch
        min_len = min(lengths[j] for j in idx_batch)
        batch = []
        lbls = []
        for j in idx_batch:
            sig = signals[j]
            # Trim trailing snapshots beyond min_len
            trimmed = DynamicGraphTemporalSignal(
                edge_indices = sig.edge_indices[:min_len],
                edge_weights = sig.edge_weights[:min_len],
                features = sig.features[:min_len],
                targets = sig.targets[:min_len]
            )
            batch.append(trimmed)
            lbls.append(labels[j])
        batches.append(batch)
        batch_labels.append(lbls)
        batch_lengths.append(min_len)

    return batches, batch_labels, batch_lengths

for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for batch_signals, batch_lbls, batch_T in zip(signals_batches, labels_batches, lengths_per_batch):
        B = len(batch_signals)

        # Stack features: shape (B, T, N_max, F)
        features = []
        edge_idx = []
        edge_w = []
        Ns = []
        for sig in batch_signals:
            Ns.append(sig.features[0].shape[0])
        N_max = max(Ns)

        for sig in batch_signals:
            # pad features to (T, N_max, F)
            feat = torch.stack([torch.tensor(x, dtype=torch.float) for x in sig.features[:batch_T]], dim=0)
            pad_N = N_max - feat.shape[1]
            if pad_N > 0:
                feat = torch.nn.functional.pad(feat, (0, 0, 0, pad_N))
            features.append(feat)

            # collect per-snapshot edges & weights trimmed to batch_T
            edge_idx.append([torch.tensor(e, dtype=torch.long) for e in sig.edge_indices[:batch_T]])
            edge_w.append([torch.tensor(w, dtype=torch.float) for w in sig.edge_weights[:batch_T]])

        features = torch.stack(features, dim=0).to(device)  # (B, T, N_max, F)
        edge_w = [[w.to(device) for w in seq] for seq in edge_w]
        edge_idx = [[e.to(device) for e in seq] for seq in edge_idx]
        y = torch.tensor(batch_lbls, dtype=torch.long).to(device)

        # Compute lambda_max once (or individually if needed)
        lmax = lap(edge_idx[0][0])['lambda_max'].to(device)

        optimizer.zero_grad()
        logits = model(features, edge_idx, edge_w, lambda_max=lmax)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


from typing import List, Tuple
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

def make_batches_grouped_by_length(
    signals: List[DynamicGraphTemporalSignal],
    lengths: List[int],
    labels: List[int],
    batch_size: int
) -> Tuple[
    List[List[DynamicGraphTemporalSignal]],
    List[List[int]],
    List[int]
]:
    # sort by length
    idx_sorted = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, batch_labels, batch_lengths = [], [], []

    for i in range(0, len(signals), batch_size):
        idx_batch = idx_sorted[i : i + batch_size]
        min_len = min(lengths[j] for j in idx_batch)
        batch, lbls = [], []
        for j in idx_batch:
            sig = signals[j]
            trimmed = DynamicGraphTemporalSignal(
                edge_indices=sig.edge_indices[:min_len],
                edge_weights=sig.edge_weights[:min_len],
                features=sig.features[:min_len],
                targets=sig.targets[:min_len]
            )
            batch.append(trimmed)
            lbls.append(labels[j])
        batches.append(batch)
        batch_labels.append(lbls)
        batch_lengths.append(min_len)

    return batches, batch_labels, batch_lengths


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric.nn import global_mean_pool

class HybridClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, K, lstm_hidden, num_classes):
        super().__init__()
        self.gc_lstm = GConvLSTM(in_channels, hidden_channels, K)
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden, batch_first=True)
        self.lin = nn.Linear(lstm_hidden, num_classes)

    def forward(self, batch_signals: List, batch_labels):
        B = len(batch_signals)
        T = len(batch_signals[0].edge_indices)
        device = next(self.parameters()).device

        # Collect per-time-step hidden vectors for each graph
        h_seq = torch.zeros(B, T, self.gc_lstm.out_channels, device=device)

        for t in range(T):
            # Build a PyG mini-batch across B graphs at time t
            data_list = []
            for sig in batch_signals:
                data_list.append(
                    torch_geometric.data.Data(
                        x=torch.tensor(sig.features[t], dtype=torch.float),
                        edge_index=torch.tensor(sig.edge_indices[t], dtype=torch.long),
                        edge_weight=torch.tensor(sig.edge_weights[t], dtype=torch.float)
                    )
                )
            mini_batch = Batch.from_data_list(data_list).to(device)

            # Run GConvLSTM step across batched graphs
            H_out, _ = self.gc_lstm(
                mini_batch.x, mini_batch.edge_index, mini_batch.edge_weight,
                H=None, C=None, lambda_max=None
            )
            # pool node features to graph-level embedding
            h_graph = global_mean_pool(H_out, mini_batch.batch)
            h_seq[:, t, :] = h_graph

        # Use PyTorch LSTM to process sequence of embeddings per graph
        _, (h_last, _) = self.lstm(h_seq)
        h_last = h_last.squeeze(0)  # shape [B, lstm_hidden]
        return self.lin(h_last)


model = HybridClassifier(in_channels=1, hidden_channels=16, K=3,
                         lstm_hidden=32, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    for batch_signals, batch_labels, _ in zip(batches, labels_batches, batch_lengths):
        logits = model(batch_signals, batch_labels)
        y = torch.tensor(batch_labels, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch:02d}, avg loss: {total_loss/len(batches):.4f}")


import os
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric_temporal.signal import DynamicHeteroGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import HeteroGCLSTM
from torch_geometric.nn import global_mean_pool

# ─── 1. Load and invert mapping pickle ────────────────────────────────────
def invert_mapping(pkl_path):
    with open(pkl_path, 'rb') as f:
        name_to_id = pickle.load(f)
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    return id_to_name

# ─── 2. Load .pt files and build sequences ─────────────────────────────────
def load_sequences(pt_folder, mapping_pkl, k, node_type='node'):
    id_to_name = invert_mapping(mapping_pkl)
    sequences, labels = [], []
    metadata = None

    for fname in sorted(os.listdir(pt_folder)):
        if not fname.endswith('.pt'):
            continue
        td = torch.load(os.path.join(pt_folder, fname))
        src = td.src.numpy(); dst = td.dst.numpy()
        t = td.t.numpy(); msg = td.msg.numpy()
        times = np.unique(t)
        T = len(times)
        if T < k:
            continue

        edge_index_dicts, edge_weight_dicts = [], []
        feature_dicts, target_dicts = [], []

        all_nodes = np.unique(np.concatenate([src, dst]))
        N = int(all_nodes.max()) + 1

        for ti in times[-k:]:
            mask = t == ti
            s = src[mask]; d = dst[mask]; w = msg[mask]
            ei = torch.tensor([s, d], dtype=torch.long)
            ew = torch.tensor(w, dtype=torch.float)

            # placeholder: you may build node features via CSV or other source
            x_dict = {node_type: torch.ones((N, 1), dtype=torch.float)}  # or your actual features
            # placeholder: target per node type, e.g. from td.y or external label source
            y_arr = td.y.numpy() if hasattr(td, 'y') else np.zeros((N,), dtype=int)
            y_dict = {node_type: torch.tensor(y_arr, dtype=torch.long)}

            edge_index_dicts.append({(node_type, 'to', node_type): ei})
            edge_weight_dicts.append({(node_type, 'to', node_type): ew})
            feature_dicts.append(x_dict)
            target_dicts.append(y_dict)

        seq = DynamicHeteroGraphTemporalSignal(
            edge_index_dicts, edge_weight_dicts,
            feature_dicts, target_dicts
        )
        sequences.append(seq)
        labels.append(0)  # placeholder: assign sequence label here
        metadata = seq.metadata

    return sequences, labels, metadata

# ─── 3. Model definition ───────────────────────────────────────────────────
class HeteroSeqClassifier(nn.Module):
    def __init__(self,
                 in_channels_dict,
                 out_channels,
                 metadata,
                 num_classes,
                 node_type='node'):
        super().__init__()
        self.gc = HeteroGCLSTM(in_channels_dict=in_channels_dict,
                                out_channels=out_channels,
                                metadata=metadata)
        self.lin = nn.Linear(out_channels, num_classes)
        self.node_type = node_type

    def forward(self, batch_signals):
        B = len(batch_signals)
        k = len(batch_signals[0].feature_dicts)
        device = next(self.parameters()).device
        last_emb = torch.zeros(B, self.gc.out_channels, device=device)

        for t in range(k):
            data_list = [sig[t] for sig in batch_signals]
            batch = Batch.from_data_list(data_list).to(device)

            H_dict, _ = self.gc(batch.x_dict, batch.edge_index_dict)
            h_nt = H_dict[self.node_type]
            pooled = global_mean_pool(h_nt, batch.batch_dict[self.node_type])
            last_emb = pooled

        return self.lin(last_emb)

# ─── 4. Training loop ──────────────────────────────────────────────────────
def train(
    pt_folder,
    mapping_pkl,
    k=5,
    batch_size=4,
    num_epochs=20,
    lr=1e-3
):
    sequences, labels, metadata = load_sequences(pt_folder, mapping_pkl, k)
    in_dict = {metadata[0]: 1}  # placeholder: your in_channels per node_type
    num_classes = max(labels) + 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroSeqClassifier(in_dict, out_channels=16,
                                metadata=metadata,
                                num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    batches = [
        sequences[i:i+batch_size]
        for i in range(0, len(sequences), batch_size)
    ]
    label_batches = [
        labels[i:i+batch_size]
        for i in range(0, len(labels), batch_size)
    ]

    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        model.train()
        for batch_sigs, batch_lbls in zip(batches, label_batches):
            logits = model(batch_sigs)
            y = torch.tensor(batch_lbls, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}, avg loss: {total_loss/len(batches):.4f}")

    return model

# ─── Example entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    pt_folder = "/path/to/pt"
    mapping_pkl = "/path/to/map.pkl"
    trained_model = train(pt_folder, mapping_pkl, k=5, batch_size=8, num_epochs=50)


def forward(self, batch_signals):
    B = len(batch_signals)
    k = len(batch_signals[0].feature_dicts)
    device = next(self.parameters()).device
    h_dict, c_dict = None, None
    final_embeddings = None

    for t in range(k):
        data_list = [sig[t] for sig in batch_signals]
        batch = Batch.from_data_list(data_list).to(device)

        h_dict, c_dict = self.gc(batch.x_dict,
                                 batch.edge_index_dict,
                                 H=h_dict, C=c_dict)

        h_nt = h_dict[self.node_type]
        pooled = global_mean_pool(h_nt, batch.batch_dict[self.node_type])

        # allocate once
        if final_embeddings is None:
            final_embeddings = torch.zeros((B, self.gc.out_channels), device=device)
        final_embeddings = pooled

    return self.lin(final_embeddings)
