"""
Unsupervised HeteroGraphSAGE **with edge‑type–stratified K‑fold splitting**
===========================================================================
This version shows **how to integrate a user‑supplied function**
`stratified_edge_kfold(data, k)` that returns per‑fold edge splits respecting
edge‑type proportions.  For each fold we
1.  **Prune** validation/test edges from the training graph to avoid
    information leakage.
2.  Build a `NeighborLoader` from the pruned training graph.
3.  Evaluate on *all* edges but restrict metrics to the held‑out sets.

The key helper is `make_fold_graph(data, kept_edges)` that clones the
original `HeteroData` and keeps only the specified edges for each canonical
relation.  A working stub for `stratified_edge_kfold` is included so you can
swap in your own implementation.
"""

from __future__ import annotations
import math, os
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter, negative_sampling

# ---------------------------------------------------------------------------
# 0  K‑fold edge splitter stub  (REPLACE with your own implementation)
# ---------------------------------------------------------------------------

def stratified_edge_kfold(data: HeteroData, k: int = 5):
    """Return a list of *k* dicts with train/val/test edge‑index arrays.

    Each dict has keys: {'train', 'val', 'test'}, each maps canonical edge
    type → 1‑D **tensor of edge indices** (row positions).
    The stub simply uses a 80/10/10 random split per edge type while keeping
    the same proportion across types.  Replace this with your stratified
    sampling logic.
    """
    splits = []
    rng = np.random.default_rng(0)
    for fold in range(k):
        fold_split = {s: {} for s in ("train", "val", "test")}
        for et in data.edge_types:
            n = data[et].edge_index.size(1)
            idx = rng.permutation(n)
            n_train = int(0.8 * n)
            n_val = int(0.1 * n)
            fold_split["train"][et] = torch.as_tensor(idx[:n_train])
            fold_split["val"][et] = torch.as_tensor(idx[n_train : n_train + n_val])
            fold_split["test"][et] = torch.as_tensor(idx[n_train + n_val :])
        splits.append(fold_split)
    return splits

# ---------------------------------------------------------------------------
# 1  Synthetic CSV (same as before)
# ---------------------------------------------------------------------------

def create_dummy_frequency_csv(n: int = 500) -> Path:
    syscall = ["read","write","openat","execve","chmod","futex","clone","mmap","close"]
    binder  = ["sendSMS","getDeviceId","startActivity","queryContentProviders","getAccounts"]
    comp    = ["NETWORK_WRITE_EXEC","READ_CONTACTS(D)","DYNAMIC_CODE_LOADING","CRYPTO_API_USED"]
    feats   = syscall + binder + comp
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0,30,(n,len(feats))), columns=feats)
    df.loc[df.sample(frac=.3,random_state=1).index, rng.choice(df.columns,3)] = 0
    def lbl(r): return int((r.execve*1.5+r.sendSMS*2+r.NETWORK_WRITE_EXEC*3+r.getDeviceId)//40)+1
    df["Class"] = df.apply(lbl, axis=1)
    out = Path("app_behavior_frequencies.csv"); df.to_csv(out,index=False); return out

# ---------------------------------------------------------------------------
# 2  HeteroData + symmetric weight norm  (unchanged)
# ---------------------------------------------------------------------------

def _cat(name:str)->str: return "syscall" if name.islower() else ("composite_behavior" if name.upper()==name else "binder")

def _sym_norm(edge_index:torch.Tensor,w:torch.Tensor,n_s:int,n_d:int):
    s,d=edge_index
    deg_s=scatter(w,s,dim_size=n_s,reduce="sum"); deg_d=scatter(w,d,dim_size=n_d,reduce="sum")
    return w*deg_s.clamp(min=1).pow(-.5)[s]*deg_d.clamp(min=1).pow(-.5)[d]

def create_heterodata(csv:Path)->HeteroData:
    df=pd.read_csv(csv); data=HeteroData(); n_app=len(df); data["application"].num_nodes=n_app
    cols=[c for c in df.columns if c!="Class"]; types={t:[] for t in ["syscall","binder","composite_behavior"]}
    for c in cols: types[_cat(c)].append(c)
    for nt,lst in types.items(): data[nt].num_nodes=len(lst)
    for nt,lst in types.items():
        src,dst,w=[],[],[]
        for app in df.index:
            for local,f in enumerate(lst):
                c=int(df.iloc[app][f]);
                if c: src.append(app); dst.append(local); w.append(c)
        eidx=torch.tensor([src,dst]); ew=torch.tensor(w,dtype=torch.float32)
        data[("application",f"uses_{nt}",nt)].edge_index=eidx
        data[("application",f"uses_{nt}",nt)].edge_weight=_sym_norm(eidx,ew,n_app,len(lst))
    return data

# ---------------------------------------------------------------------------
# 3  Model (unchanged)
# ---------------------------------------------------------------------------

class HeteroGraphSAGE(nn.Module):
    def __init__(self, metadata, n_nodes, emb_dim=64, hid_dim=64):
        super().__init__(); self.emb=nn.ModuleDict({nt:nn.Embedding(n,emb_dim) for nt,n in n_nodes.items()})
        self.conv1=HeteroConv({et:SAGEConv(emb_dim,hid_dim,aggr="mean",normalize=True) for et in metadata[1]}, aggr="sum")
        self.conv2=HeteroConv({et:SAGEConv(hid_dim,hid_dim,aggr="mean",normalize=True) for et in metadata[1]}, aggr="sum")
        self.dec_exist=nn.Sequential(nn.Linear(hid_dim*2,hid_dim),nn.ReLU(),nn.Linear(hid_dim,1))
        self.dec_weight=nn.Sequential(nn.Linear(hid_dim*2,hid_dim),nn.ReLU(),nn.Linear(hid_dim,1))

    def _encode(self,x,eidx,ew):
        h=self.conv1(x,eidx,edge_weight_dict=ew); h={k:F.relu(v) for k,v in h.items()}
        return self.conv2(h,eidx,edge_weight_dict=ew)

    def full_emb(self,data,device):
        x={nt:self.emb[nt].weight.to(device) for nt in data.node_types}
        return self._encode(x,data.edge_index_dict,data.edge_weight_dict)

    def forward(self,batch,device):
        x={nt:self.emb[nt](batch[nt].n_id.to(device)) for nt in batch.node_types}
        h=self._encode(x,batch.edge_index_dict,{et:batch[et].edge_weight for et in batch.edge_types})
        return h

# ---------------------------------------------------------------------------
# 4  Utility: prune HeteroData to keep only specific edges
# ---------------------------------------------------------------------------

def make_fold_graph(base: HeteroData, kept_edges: dict[str, torch.Tensor]) -> HeteroData:
    """Return a **deep‑copied** graph that keeps only the edges whose row
    indices are listed in *kept_edges[etype]* for every canonical relation.
    Node stores are shared to save memory; edge stores are sliced.
    """
    g = deepcopy(base)
    for et, keep_idx in kept_edges.items():
        ei = g[et].edge_index[:, keep_idx]
        g[et].edge_index = ei
        for key, val in list(g[et].items()):  # slice all edge attributes
            if val.size(0) == keep_idx.numel():
                g[et][key] = val[keep_idx]
    return g

# ---------------------------------------------------------------------------
# 5  Loss + metric helpers (unchanged)
# ---------------------------------------------------------------------------

def edge_losses(h_dict,batch,lam=0.1,device="cpu"):
    bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss(); l_exist=l_mse=0.
    for et in [e for e in batch.edge_types if e[0]=="application"]:
        ei=batch[et].edge_index.to(device); w = batch[et].edge_weight[: ei.size(1)].to(device)  # shapes now match pos_pred
        src, dst = h_dict[et[0]][ei[0]], h_dict[et[2]][ei[1]]
        pair = torch.cat([src,dst],1)
        pos_log = batch.model.dec_exist(pair).squeeze(); pos_pred = batch.model.dec_weight(pair).squeeze()
        neg_ei = negative_sampling(ei, size=ei.size(1), device=device,
                                   num_nodes=(h_dict[et[0]].size(0), h_dict[et[2]].size(0)))
        neg_pair = torch.cat([h_dict[et[0]][neg_ei[0]], h_dict[et[2]][neg_ei[1]]],1)
        neg_log = batch.model.dec_exist(neg_pair).squeeze()
        l_exist += bce(torch.cat([pos_log,neg_log]), torch.cat([torch.ones_like(pos_log), torch.zeros_like(neg_log)]))
        l_mse   += mse(pos_pred, w)
    return l_exist + lam*l_mse


def evaluate(model,data,edge_mask_dict,device):
    model.eval(); aucs=[]; rmses=[]
    with torch.no_grad():
        h=model.full_emb(data,device)
        for et, mask in edge_mask_dict.items():
            ei=data[et].edge_index[:,mask].to(device); w=data[et].edge_weight[mask].to(device)
            pair=torch.cat([h[et[0]][ei[0]], h[et[2]][ei[1]]],1)
            prob = torch.sigmoid(model.dec_exist(pair).squeeze()).cpu().numpy()
            aucs.append(roc_auc_score(np.ones_like(prob), prob))
            pred_w=model.dec_weight(pair).squeeze().cpu().numpy(); rmses.append(math.sqrt(mean_squared_error(w.cpu().numpy(), pred_w)))
    return float(np.mean(aucs)), float(np.mean(rmses))

# ---------------------------------------------------------------------------
# 6  Main – edge‑type stratified K‑fold
# ---------------------------------------------------------------------------

def main(k=5):
    csv=create_dummy_frequency_csv(); base_data=create_heterodata(csv)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds=stratified_edge_kfold(base_data,k)
    for fold_idx, split in enumerate(folds,1):
        print(f"\n── Fold {fold_idx}/{k} ──────────────────")
        train_graph = make_fold_graph(base_data, split["train"])
        train_loader = NeighborLoader(train_graph, batch_size=256, shuffle=True,
                                      num_neighbors=[4,4], input_nodes=("application", None))
        model = HeteroGraphSAGE(base_data.metadata(), {nt: base_data[nt].num_nodes for nt in base_data.node_types}).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1,16):
            model.train(); tot_l=0.
            for batch in train_loader:
                batch=batch.to(device); batch.model=model
                h=model(batch,device)
                loss=edge_losses(h,batch,device=device)
                optim.zero_grad(); loss.backward(); optim.step(); tot_l+=loss.item()
            if epoch%5==0:
                val_auc,val_rmse=evaluate(model, base_data.to(device), split["val"], device)
                print(f"  epoch {epoch:02d}  trainL {tot_l/len(train_loader):.4f}  valAUC {val_auc:.4f}  valRMSE {val_rmse:.4f}")
        test_auc,test_rmse=evaluate(model, base_data.to(device), split["test"], device)
        print(f">> Test AUROC {test_auc:.4f}  RMSE {test_rmse:.4f}")
    os.remove(csv)

if __name__=="__main__":
    main()


pareto = study.best_trials

rmse = np.array([t.values[0] for t in pareto])
auc  = np.array([t.values[1] for t in pareto])

rmse_n = (rmse - rmse.min()) / (rmse.max() - rmse.min() + 1e-12)
auc_n  = (auc.max() - auc   ) / (auc.max() - auc.min()  + 1e-12)

best_idx = np.argmin(np.sqrt(rmse_n**2 + auc_n**2))  # distance-to-utopia
best_trial = pareto[best_idx]

print("Balanced champion → Trial", best_trial.number,
      "(RMSE =", best_trial.values[0], ", AUC =", best_trial.values[1], ")")
params = best_trial.params 