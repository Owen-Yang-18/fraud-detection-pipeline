# train_hetero6_mean_inductive.py
# ------------------------------------------------------------
# Hetero GNN (dict-less), 2 layers, MEAN aggregation.
# Inductive evaluation: train with train-only edges, test with test-only edges.
# 5-way classification on application nodes with stratified 80/20 split.

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- IDs for readability ----------------
APP, SYS, BND, CMP = 0, 1, 2, 3

# --------------- Bipartite neighbor-only conv (MEAN) ---------------
class BiMeanNeighbor(nn.Module):
    """
    Bipartite neighbor aggregation from src -> dst with WEIGHTED MEAN:

        agg_dst[d] = ( sum_{(s->d) in E} w[s->d] * (W_nei * x_src[s]) )
                     / ( sum_{(s->d) in E} w[s->d] ), with 0-safe denom.

    This layer does *not* apply any transform to x_dst; a separate self linear
    is applied once per node type in the hetero layer (no double-counting).
    """
    def __init__(self, in_src: int, out_dim: int):
        super().__init__()
        self.lin_nei = nn.Linear(in_src, out_dim, bias=False)

    def forward(
        self,
        x_src: torch.Tensor,          # [N_src, F_src]
        dst_len: int,                 # number of dst nodes
        edge_index_sd: torch.Tensor,  # [2, E], row0=src (local), row1=dst (local)
        edge_weight: torch.Tensor     # [E]
    ) -> torch.Tensor:
        if edge_index_sd.numel() == 0:
            return torch.zeros(dst_len, self.lin_nei.out_features, device=x_src.device)

        src_idx = edge_index_sd[0]
        dst_idx = edge_index_sd[1]

        msg = self.lin_nei(x_src).index_select(0, src_idx)  # [E, F_out]
        msg = msg * edge_weight.view(-1, 1)

        agg = torch.zeros(dst_len, msg.size(1), device=x_src.device)
        agg.index_add_(0, dst_idx, msg)

        denom = torch.zeros(dst_len, device=x_src.device, dtype=edge_weight.dtype)
        denom.index_add_(0, dst_idx, edge_weight)
        denom = denom.clamp_min(1e-12).view(-1, 1)

        return agg / denom

# ---------------- One hetero layer (MEAN) ----------------
class Hetero6Layer(nn.Module):
    """
    One layer with:
      - a single self Linear per node type (applied once),
      - one neighbor-only bipartite conv per relation,
      - sum of incoming relation messages per destination type.
    Layer is parameterized by per-type input dims -> shared hidden dim.
    """
    def __init__(self, in_app: int, in_sys: int, in_bnd: int, in_cmp: int, hidden: int):
        super().__init__()
        # Self transforms (applied once per type)
        self.self_app = nn.Linear(in_app, hidden)
        self.self_sys = nn.Linear(in_sys, hidden)
        self.self_bnd = nn.Linear(in_bnd, hidden)
        self.self_cmp = nn.Linear(in_cmp, hidden)

        # Neighbor-only MEAN convs
        self.app_to_sys = BiMeanNeighbor(in_app, hidden)  # app -> sys
        self.sys_to_app = BiMeanNeighbor(in_sys, hidden)  # sys -> app
        self.app_to_bnd = BiMeanNeighbor(in_app, hidden)  # app -> bnd
        self.bnd_to_app = BiMeanNeighbor(in_bnd, hidden)  # bnd -> app
        self.app_to_cmp = BiMeanNeighbor(in_app, hidden)  # app -> cmp
        self.cmp_to_app = BiMeanNeighbor(in_cmp, hidden)  # cmp -> app

    def forward(
        self,
        x_app: torch.Tensor, x_sys: torch.Tensor, x_bnd: torch.Tensor, x_cmp: torch.Tensor,
        ei_app_sys: torch.Tensor, ew_app_sys: torch.Tensor,   # app -> sys
        ei_sys_app: torch.Tensor, ew_sys_app: torch.Tensor,   # sys -> app
        ei_app_bnd: torch.Tensor, ew_app_bnd: torch.Tensor,   # app -> bnd
        ei_bnd_app: torch.Tensor, ew_bnd_app: torch.Tensor,   # bnd -> app
        ei_app_cmp: torch.Tensor, ew_app_cmp: torch.Tensor,   # app -> cmp
        ei_cmp_app: torch.Tensor, ew_cmp_app: torch.Tensor,   # cmp -> app
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # self terms
        out_app = self.self_app(x_app)
        out_sys = self.self_sys(x_sys)
        out_bnd = self.self_bnd(x_bnd)
        out_cmp = self.self_cmp(x_cmp)

        # neighbor messages (MEAN) summed per destination type
        out_app = out_app \
            + self.sys_to_app(x_sys, x_app.size(0), ei_sys_app, ew_sys_app) \
            + self.bnd_to_app(x_bnd, x_app.size(0), ei_bnd_app, ew_bnd_app) \
            + self.cmp_to_app(x_cmp, x_app.size(0), ei_cmp_app, ew_cmp_app)

        out_sys = out_sys + self.app_to_sys(x_app, x_sys.size(0), ei_app_sys, ew_app_sys)
        out_bnd = out_bnd + self.app_to_bnd(x_app, x_bnd.size(0), ei_app_bnd, ew_app_bnd)
        out_cmp = out_cmp + self.app_to_cmp(x_app, x_cmp.size(0), ei_app_cmp, ew_app_cmp)
        return out_app, out_sys, out_bnd, out_cmp

# ---------------- Whole network (2 layers + head on apps) ----------------
class Hetero6Net(nn.Module):
    """
    Two Hetero6Layer blocks.
    First layer maps (F_app, F_sys, F_bnd, F_cmp) -> hidden.
    Second layer maps (hidden, hidden, hidden, hidden) -> hidden.
    Head: logits for application nodes only.
    """
    def __init__(self, f_app: int, f_sys: int, f_bnd: int, f_cmp: int, hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.l1 = Hetero6Layer(f_app, f_sys, f_bnd, f_cmp, hidden)
        self.l2 = Hetero6Layer(hidden, hidden, hidden, hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        x_app, x_sys, x_bnd, x_cmp,
        ei_app_sys, ew_app_sys,
        ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd,
        ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp,
        ei_cmp_app, ew_cmp_app,
    ):
        a, s, b, c = self.l1(
            x_app, x_sys, x_bnd, x_cmp,
            ei_app_sys, ew_app_sys,
            ei_sys_app, ew_sys_app,
            ei_app_bnd, ew_app_bnd,
            ei_bnd_app, ew_bnd_app,
            ei_app_cmp, ew_app_cmp,
            ei_cmp_app, ew_cmp_app,
        )
        a, s, b, c = F.relu(a), F.relu(s), F.relu(b), F.relu(c)
        a, s, b, c = self.dropout(a), self.dropout(s), self.dropout(b), self.dropout(c)

        a, s, b, c = self.l2(
            a, s, b, c,
            ei_app_sys, ew_app_sys,
            ei_sys_app, ew_sys_app,
            ei_app_bnd, ew_app_bnd,
            ei_bnd_app, ew_bnd_app,
            ei_app_cmp, ew_app_cmp,
            ei_cmp_app, ew_cmp_app,
        )
        a = F.relu(a)
        return self.head(a)  # [N_app, out_dim]

# ---------------- Synthetic data (replace with your tensors) ----------------
@torch.no_grad()
def build_from_counts(
    n_app: int, n_sys: int, n_bnd: int, n_cmp: int,
    e_app_sys: int, e_sys_app: int, e_app_bnd: int, e_bnd_app: int, e_app_cmp: int, e_cmp_app: int,
    num_classes: int = 5,
):
    # features are arange scalars per your spec (shape [N_type, 1])
    x_app = torch.arange(n_app, dtype=torch.float32).unsqueeze(1)
    x_sys = torch.arange(n_sys, dtype=torch.float32).unsqueeze(1)
    x_bnd = torch.arange(n_bnd, dtype=torch.float32).unsqueeze(1)
    x_cmp = torch.arange(n_cmp, dtype=torch.float32).unsqueeze(1)

    def rnd_edges(n_src, n_dst, e):
        if e == 0:
            return torch.zeros(2,0,dtype=torch.long), torch.zeros(0)
        src = torch.randint(0, n_src, (e,), dtype=torch.long)
        dst = torch.randint(0, n_dst, (e,), dtype=torch.long)
        ei = torch.stack([src, dst], dim=0)
        ew = torch.ones(e, dtype=torch.float32)
        return ei, ew

    ei_app_sys, ew_app_sys = rnd_edges(n_app, n_sys, e_app_sys)
    ei_sys_app, ew_sys_app = rnd_edges(n_sys, n_app, e_sys_app)
    ei_app_bnd, ew_app_bnd = rnd_edges(n_app, n_bnd, e_app_bnd)
    ei_bnd_app, ew_bnd_app = rnd_edges(n_bnd, n_app, e_bnd_app)
    ei_app_cmp, ew_app_cmp = rnd_edges(n_app, n_cmp, e_app_cmp)
    ei_cmp_app, ew_cmp_app = rnd_edges(n_cmp, n_app, e_cmp_app)

    # labels for application nodes
    y_app = torch.randint(0, num_classes, (n_app,), dtype=torch.long)

    return (
        x_app, x_sys, x_bnd, x_cmp,
        ei_app_sys, ew_app_sys,
        ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd,
        ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp,
        ei_cmp_app, ew_cmp_app,
        y_app,
    )

# ---------------- Stratified split over application nodes ----------------
def stratified_split_indices(y: torch.Tensor, train_ratio: float = 0.8, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    classes = torch.unique(y)
    train_idx, test_idx = [], []
    for c in classes.tolist():
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        perm = idx[torch.randperm(idx.numel(), generator=g)]
        n_train = int(math.floor(train_ratio * perm.numel()))
        train_idx.append(perm[:n_train])
        test_idx.append(perm[n_train:])
    return torch.cat(train_idx), torch.cat(test_idx)

# ---------------- Build train-only and test-only edge sets ----------------
@torch.no_grad()
def split_edges_by_app(
    n_app: int,
    ei_app_sys, ew_app_sys,
    ei_sys_app, ew_sys_app,
    ei_app_bnd, ew_app_bnd,
    ei_bnd_app, ew_bnd_app,
    ei_app_cmp, ew_app_cmp,
    ei_cmp_app, ew_cmp_app,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
):
    # boolean masks for app indices
    mask_train = torch.zeros(n_app, dtype=torch.bool)
    mask_train[train_idx] = True
    mask_test = torch.zeros(n_app, dtype=torch.bool)
    mask_test[test_idx] = True

    def filter_src_app(ei, ew, mask_app):
        if ei.numel() == 0:
            return ei, ew
        keep = mask_app[ei[0]]
        return ei[:, keep], ew[keep]

    def filter_dst_app(ei, ew, mask_app):
        if ei.numel() == 0:
            return ei, ew
        keep = mask_app[ei[1]]
        return ei[:, keep], ew[keep]

    # Train edges: keep edges whose APP endpoint is in train set
    tr_ei_app_sys, tr_ew_app_sys = filter_src_app(ei_app_sys, ew_app_sys, mask_train)  # app->sys (src is app)
    tr_ei_sys_app, tr_ew_sys_app = filter_dst_app(ei_sys_app, ew_sys_app, mask_train)  # sys->app (dst is app)
    tr_ei_app_bnd, tr_ew_app_bnd = filter_src_app(ei_app_bnd, ew_app_bnd, mask_train)  # app->bnd
    tr_ei_bnd_app, tr_ew_bnd_app = filter_dst_app(ei_bnd_app, ew_bnd_app, mask_train)  # bnd->app
    tr_ei_app_cmp, tr_ew_app_cmp = filter_src_app(ei_app_cmp, ew_app_cmp, mask_train)  # app->cmp
    tr_ei_cmp_app, tr_ew_cmp_app = filter_dst_app(ei_cmp_app, ew_cmp_app, mask_train)  # cmp->app

    # Test edges: analogous for test set
    te_ei_app_sys, te_ew_app_sys = filter_src_app(ei_app_sys, ew_app_sys, mask_test)
    te_ei_sys_app, te_ew_sys_app = filter_dst_app(ei_sys_app, ew_sys_app, mask_test)
    te_ei_app_bnd, te_ew_app_bnd = filter_src_app(ei_app_bnd, ew_app_bnd, mask_test)
    te_ei_bnd_app, te_ew_bnd_app = filter_dst_app(ei_bnd_app, ew_bnd_app, mask_test)
    te_ei_app_cmp, te_ew_app_cmp = filter_src_app(ei_app_cmp, ew_app_cmp, mask_test)
    te_ei_cmp_app, te_ew_cmp_app = filter_dst_app(ei_cmp_app, ew_cmp_app, mask_test)

    train_edges = (
        tr_ei_app_sys, tr_ew_app_sys,
        tr_ei_sys_app, tr_ew_sys_app,
        tr_ei_app_bnd, tr_ew_app_bnd,
        tr_ei_bnd_app, tr_ew_bnd_app,
        tr_ei_app_cmp, tr_ew_app_cmp,
        tr_ei_cmp_app, tr_ew_cmp_app,
    )
    test_edges = (
        te_ei_app_sys, te_ew_app_sys,
        te_ei_sys_app, te_ew_sys_app,
        te_ei_app_bnd, te_ew_app_bnd,
        te_ei_bnd_app, te_ew_bnd_app,
        te_ei_app_cmp, te_ew_app_cmp,
        te_ei_cmp_app, te_ew_cmp_app,
    )
    return train_edges, test_edges

# ---------------- Metrics ----------------
@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(-1) == y).float().mean().item()

# ---------------- Training loop ----------------
def train_model(
    model: nn.Module,
    x_app, x_sys, x_bnd, x_cmp,
    train_edges, test_edges,
    y_app, train_idx, test_idx,
    epochs=50, lr=1e-3, weight_decay=5e-4, device="cpu",
):
    # move tensors to device
    tensors = [x_app, x_sys, x_bnd, x_cmp, y_app, train_idx, test_idx] + list(train_edges) + list(test_edges)
    tensors = [t.to(device) for t in tensors]
    (x_app, x_sys, x_bnd, x_cmp, y_app, train_idx, test_idx, *others) = tensors
    tr_edges = tuple(others[:12])
    te_edges = tuple(others[12:])

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits_tr = model(x_app, x_sys, x_bnd, x_cmp, *tr_edges)         # use train-only edges
        loss = loss_fn(logits_tr.index_select(0, train_idx), y_app.index_select(0, train_idx))
        loss.backward()
        opt.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                train_acc = accuracy(logits_tr.index_select(0, train_idx), y_app.index_select(0, train_idx))
                logits_te = model(x_app, x_sys, x_bnd, x_cmp, *te_edges) # evaluate with test-only edges
                test_acc  = accuracy(logits_te.index_select(0, test_idx),  y_app.index_select(0, test_idx))
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

# ---------------- Main (demo) ----------------
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Toy sizes (replace with your real tensors)
    n_app, n_sys, n_bnd, n_cmp = 1000, 4000, 600, 900
    e0, e1, e2, e3, e4, e5 = 5000, 5000, 2500, 2500, 5000, 5000

    (x_app, x_sys, x_bnd, x_cmp,
     ei_app_sys, ew_app_sys,
     ei_sys_app, ew_sys_app,
     ei_app_bnd, ew_app_bnd,
     ei_bnd_app, ew_bnd_app,
     ei_app_cmp, ew_app_cmp,
     ei_cmp_app, ew_cmp_app,
     y_app) = build_from_counts(
        n_app, n_sys, n_bnd, n_cmp,
        e0, e1, e2, e3, e4, e5,
        num_classes=5,
    )

    # Stratified split on application labels (80/20)
    train_idx, test_idx = stratified_split_indices(y_app, train_ratio=0.8, seed=42)

    # Build train-only and test-only edge sets (no edges touching held-out apps leak into training)
    train_edges, test_edges = split_edges_by_app(
        n_app,
        ei_app_sys, ew_app_sys,
        ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd,
        ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp,
        ei_cmp_app, ew_cmp_app,
        train_idx, test_idx,
    )

    # Features are 1-D (arange), so f_* = 1
    model = Hetero6Net(f_app=1, f_sys=1, f_bnd=1, f_cmp=1, hidden=128, out_dim=5, dropout=0.2)

    train_model(
        model,
        x_app, x_sys, x_bnd, x_cmp,
        train_edges, test_edges,
        y_app, train_idx, test_idx,
        epochs=50, lr=1e-3, weight_decay=5e-4, device=device
    )

if __name__ == "__main__":
    main()
