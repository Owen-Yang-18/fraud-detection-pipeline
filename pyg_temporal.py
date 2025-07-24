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

