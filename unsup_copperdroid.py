"""
Unsupervised **edge‑prediction + edge‑weight regression** variant of the
HeteroGraphSAGE example (PyTorch Geometric).
---------------------------------------------------------
This refactor removes the downstream node‑classification head and trains
solely from the graph structure:

* **Link existence** – binary cross‑entropy on real vs. negative edges for
  each canonical relation (forward direction only).
* **Edge‑weight regression** – mean‑squared error on the *positive* edges,
  predicting the original frequency count.

> Total loss  
> `L = Σ_rel  ( BCE_posneg  +  λ · MSE_pos )`,  with `λ = 0.1` by default.

Negative edges are generated on‑the‑fly via
`torch_geometric.utils.negative_sampling` for each minibatch.

Tested with:
    • Python 3.11.4, torch 2.3 + CUDA 12.4
    • torch‑geometric 2.5.1, scikit‑learn 1.5

Run the script to see the average positive‑edge AUROC and RMSE over 5‑fold
edge splits.
"""

from __future__ import annotations
import math, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter, negative_sampling

# ---------------------------------------------------------------------------
# 1  Synthetic CSV (unchanged)
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
# 2  HeteroData construction + symmetric weight norm (unchanged)
# ---------------------------------------------------------------------------

def _cat(name:str)->str: return "syscall" if name.islower() else ("composite_behavior" if name.upper()==name else "binder")

def _sym_norm(edge_index:Tensor,w:Tensor,n_s:int,n_d:int):
    s,d=edge_index; deg_s=scatter(w,s,dim_size=n_s,reduce="sum"); deg_d=scatter(w,d,dim_size=n_d,reduce="sum")
    return w*deg_s.clamp(min=1).pow(-.5)[s]*deg_d.clamp(min=1).pow(-.5)[d]

def create_heterodata(csv:Path)->HeteroData:
    df=pd.read_csv(csv); data=HeteroData(); n_app=len(df); data["application"].num_nodes=n_app
    cols=[c for c in df.columns if c!="Class"]; types={t:[] for t in ["syscall","binder","composite_behavior"]}
    for c in cols: types[_cat(c)].append(c)
    for nt,ls in types.items(): data[nt].num_nodes=len(ls)
    for nt,ls in types.items():
        src,dst,w=[],[],[]
        for app in df.index:
            for local,f in enumerate(ls):
                c=int(df.iloc[app][f]);
                if c: src.append(app); dst.append(local); w.append(c)
        eidx=torch.tensor([src,dst]); ew=torch.tensor(w,dtype=torch.float32); ew=_sym_norm(eidx,ew,n_app,len(ls))
        data[("application",f"uses_{nt}",nt)].edge_index=eidx; data[("application",f"uses_{nt}",nt)].edge_weight=ew
    return data

# ---------------------------------------------------------------------------
# 3  Unsupervised HeteroGraphSAGE encoder + edge decoders
# ---------------------------------------------------------------------------

class HeteroGraphSAGE(nn.Module):
    def __init__(self, metadata, n_nodes, emb_dim=64, hid_dim=64):
        super().__init__(); self.emb=nn.ModuleDict({nt:nn.Embedding(n,emb_dim) for nt,n in n_nodes.items()})
        self.conv1=HeteroConv({et:SAGEConv(emb_dim,hid_dim,aggr="mean",normalize=True) for et in metadata[1]}, aggr="sum")
        self.conv2=HeteroConv({et:SAGEConv(hid_dim,hid_dim,aggr="mean",normalize=True) for et in metadata[1]}, aggr="sum")
        # shared decoder (concat -> hidden -> score, weight)
        self.dec_exist=nn.Sequential(nn.Linear(hid_dim*2,hid_dim),nn.ReLU(),nn.Linear(hid_dim,1))
        self.dec_weight=nn.Sequential(nn.Linear(hid_dim*2,hid_dim),nn.ReLU(),nn.Linear(hid_dim,1))

    def _encode(self,x_dict,eidx_dict,ew_dict):
        h=self.conv1(x_dict,eidx_dict,edge_weight_dict=ew_dict); h={k:F.relu(v) for k,v in h.items()}
        h=self.conv2(h,eidx_dict,edge_weight_dict=ew_dict); return h

    def full_embeddings(self,data: HeteroData, device):
        x={nt:self.emb[nt].weight.to(device) for nt in data.node_types}
        return self._encode(x,data.edge_index_dict,data.edge_weight_dict)

    # -----------------------------------------------------------------
    def forward(self, batch, device):
        x={nt:self.emb[nt](batch[nt].n_id.to(device)) for nt in batch.node_types}
        h=self._encode(x,batch.edge_index_dict,{et:batch[et].edge_weight for et in batch.edge_types})
        return h  # dict of node embeddings keyed by ntype

# ---------------------------------------------------------------------------
# 4  Loss helpers
# ---------------------------------------------------------------------------

def edge_losses(h_dict,batch,lambda_mse=0.1,device="cpu"):
    bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss(); total_exist=0.; total_mse=0.
    for et in [e for e in batch.edge_types if e[0]=="application"]:  # forward only
        ei=batch[et].edge_index.to(device); w=batch[et].edge_weight.to(device)
        src_h=h_dict[et[0]][ei[0]]; dst_h=h_dict[et[2]][ei[1]]; pair=torch.cat([src_h,dst_h],dim=1)
        pos_logit=batch.model.dec_exist(pair).squeeze(); pos_pred=batch.model.dec_weight(pair).squeeze()
        # negative sampling (same size)
        neg_ei=negative_sampling(ei,size=ei.size(1),num_nodes=(h_dict[et[0]].size(0),h_dict[et[2]].size(0)),device=device)
        neg_src=h_dict[et[0]][neg_ei[0]]; neg_dst=h_dict[et[2]][neg_ei[1]]; neg_pair=torch.cat([neg_src,neg_dst],1)
        neg_logit=batch.model.dec_exist(neg_pair).squeeze()
        exist_loss=bce(torch.cat([pos_logit,neg_logit]), torch.cat([torch.ones_like(pos_logit),torch.zeros_like(neg_logit)]))
        mse_loss=mse(pos_pred, w)
        total_exist+=exist_loss; total_mse+=mse_loss
    return total_exist+lambda_mse*total_mse, total_exist.item(), total_mse.item()

# ---------------------------------------------------------------------------
# 5  Training & evaluation (edge‑level metrics)
# ---------------------------------------------------------------------------

def train_epoch(model,loader,opt,device):
    model.train(); total=0.
    for batch in loader:
        batch=batch.to(device); batch.model=model  # pass pointer for loss util
        h=model(batch,device)
        loss,_,_=edge_losses(h,batch,device=device)
        opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()
    return total/len(loader)


def evaluate(model,data,device):
    model.eval(); with torch.no_grad():
        h=model.full_embeddings(data,device)
        aucs=[]; rmses=[]
        for et in [e for e in data.edge_types if e[0]=="application"]:
            ei=data[et].edge_index.to(device); w=data[et].edge_weight.to(device)
            src_h=h[et[0]][ei[0]]; dst_h=h[et[2]][ei[1]]; pair=torch.cat([src_h,dst_h],1)
            logit=model.dec_exist(pair).squeeze().cpu(); prob=torch.sigmoid(logit)
            aucs.append(roc_auc_score(np.ones(len(prob)), prob.numpy()))
            pred_w=model.dec_weight(pair).squeeze().cpu().numpy(); rmses.append(math.sqrt(mean_squared_error(w.cpu().numpy(),pred_w)))
        return float(np.mean(aucs)), float(np.mean(rmses))

# ---------------------------------------------------------------------------
# 6  Main routine – unsupervised edge prediction
# ---------------------------------------------------------------------------

def main():
    csv=create_dummy_frequency_csv(); data=create_heterodata(csv)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader=NeighborLoader(data,num_neighbors=[4,4],batch_size=256,shuffle=True,input_nodes=("application",None))
    model=HeteroGraphSAGE(data.metadata(),{nt:data[nt].num_nodes for nt in data.node_types}).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(1,21):
        l=train_epoch(model,loader,opt,device)
        if epoch%5==0:
            auc,rmse=evaluate(model,data.to(device),device)
            print(f"epoch {epoch:02d} loss {l:.4f}  AUROC {auc:.4f}  RMSE {rmse:.4f}")
    os.remove(csv)

if __name__=="__main__":
    main()
