# simple_hetero6_execu.py
# ------------------------------------------------------------
# Four node types: application, syscall, binder, composite_behavior
# Six edge relations (reciprocal pairs):
#   0) application -> syscall            (uses_sys)
#   1) syscall     -> application        (uses_sys_by)
#   2) application -> binder             (uses_binder)
#   3) binder      -> application        (uses_binder_by)
#   4) application -> composite_behavior (uses_composite)
#   5) composite_behavior -> application (uses_composite_by)
#
# - Two "hetero" layers (sum aggregation across relations).
# - MLP head outputs logits for APPLICATION nodes only.
# - No dicts in forward; ExecuTorch-friendly.
#
# Requirements (user install or venv is fine):
#   pip install "torch>=2.3" "executorch>=0.6"

import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, Dim

# ---------- IDs for readability ----------
APP, SYS, BND, CMP = 0, 1, 2, 3

# ---------- Bipartite GraphConv (edge-weighted) ----------
class BiGraphConv(nn.Module):
    """
    Bipartite GraphConv: from src->dst
      out_dst = W_self * x_dst + sum_{(s,d) in E} w[s->d] * (W_nei * x_src[s])
    edge_index_sd: [2, E] with row0=src idx (local to src set), row1=dst idx (local to dst set)
    """
    def __init__(self, in_src: int, in_dst: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_nei  = nn.Linear(in_src, out_dim, bias=False)  # neighbor transform
        self.lin_self = nn.Linear(in_dst, out_dim, bias=bias)   # self transform (on dst)

    def forward(
        self,
        x_src: torch.Tensor,          # [N_src, F_src]
        x_dst: torch.Tensor,          # [N_dst, F_dst]
        edge_index_sd: torch.Tensor,  # [2, E], long
        edge_weight: torch.Tensor     # [E], float
    ) -> torch.Tensor:
        src_idx = edge_index_sd[0]
        dst_idx = edge_index_sd[1]

        # self term on destination nodes
        out = self.lin_self(x_dst)                    # [N_dst, F_out]

        # neighbor messages
        msg = self.lin_nei(x_src).index_select(0, src_idx)  # [E, F_out]
        msg = msg * edge_weight.view(-1, 1)

        # aggregate by sum into destinations
        out.index_add_(0, dst_idx, msg)
        return out

# ---------- One hetero layer with 6 relations ----------
class Hetero6Layer(nn.Module):
    """
    For each node type we keep a self Linear (size: hidden->hidden).
    For each relation we keep one BiGraphConv (hidden->hidden).
    """
    def __init__(self, hidden: int):
        super().__init__()
        # self transforms per node type
        self.self_app = nn.Linear(hidden, hidden)
        self.self_sys = nn.Linear(hidden, hidden)
        self.self_bnd = nn.Linear(hidden, hidden)
        self.self_cmp = nn.Linear(hidden, hidden)

        # six relations (src->dst), all in hidden dim
        self.app_to_sys = BiGraphConv(hidden, hidden, hidden)  # 0
        self.sys_to_app = BiGraphConv(hidden, hidden, hidden)  # 1
        self.app_to_bnd = BiGraphConv(hidden, hidden, hidden)  # 2
        self.bnd_to_app = BiGraphConv(hidden, hidden, hidden)  # 3
        self.app_to_cmp = BiGraphConv(hidden, hidden, hidden)  # 4
        self.cmp_to_app = BiGraphConv(hidden, hidden, hidden)  # 5

    def forward(
        self,
        x_app: torch.Tensor, x_sys: torch.Tensor, x_bnd: torch.Tensor, x_cmp: torch.Tensor,
        ei_app_sys: torch.Tensor, ew_app_sys: torch.Tensor,
        ei_sys_app: torch.Tensor, ew_sys_app: torch.Tensor,
        ei_app_bnd: torch.Tensor, ew_app_bnd: torch.Tensor,
        ei_bnd_app: torch.Tensor, ew_bnd_app: torch.Tensor,
        ei_app_cmp: torch.Tensor, ew_app_cmp: torch.Tensor,
        ei_cmp_app: torch.Tensor, ew_cmp_app: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # start from self terms
        out_app = self.self_app(x_app)
        out_sys = self.self_sys(x_sys)
        out_bnd = self.self_bnd(x_bnd)
        out_cmp = self.self_cmp(x_cmp)

        # add incoming messages per destination type
        # application receives from syscall, binder, composite_behavior
        out_app = out_app + self.sys_to_app(x_sys, x_app, ei_sys_app, ew_sys_app)
        out_app = out_app + self.bnd_to_app(x_bnd, x_app, ei_bnd_app, ew_bnd_app)
        out_app = out_app + self.cmp_to_app(x_cmp, x_app, ei_cmp_app, ew_cmp_app)

        # syscall receives from application
        out_sys = out_sys + self.app_to_sys(x_app, x_sys, ei_app_sys, ew_app_sys)

        # binder receives from application
        out_bnd = out_bnd + self.app_to_bnd(x_app, x_bnd, ei_app_bnd, ew_app_bnd)

        # composite_behavior receives from application
        out_cmp = out_cmp + self.app_to_cmp(x_app, x_cmp, ei_app_cmp, ew_app_cmp)

        return out_app, out_sys, out_bnd, out_cmp

# ---------- Whole network: 2 layers + MLP head on application ----------
class Hetero6Net(nn.Module):
    """
    Inputs:
      x_app: [N_app, F_app], x_sys: [N_sys, F_sys], x_bnd: [N_bnd, F_bnd], x_cmp: [N_cmp, F_cmp]
      For each relation r, edge_index_r: [2, E_r], edge_weight_r: [E_r]
    Output:
      logits for APPLICATION nodes only: [N_app, out_dim]
    """
    def __init__(self, f_app: int, f_sys: int, f_bnd: int, f_cmp: int,
                 hidden: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        # per-type input projection to shared hidden dim
        self.in_app = nn.Linear(f_app, hidden)
        self.in_sys = nn.Linear(f_sys, hidden)
        self.in_bnd = nn.Linear(f_bnd, hidden)
        self.in_cmp = nn.Linear(f_cmp, hidden)

        self.layer1 = Hetero6Layer(hidden)
        self.layer2 = Hetero6Layer(hidden)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
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
        # project inputs
        x_app = F.relu(self.in_app(x_app))
        x_sys = F.relu(self.in_sys(x_sys))
        x_bnd = F.relu(self.in_bnd(x_bnd))
        x_cmp = F.relu(self.in_cmp(x_cmp))

        # layer 1
        a, s, b, c = self.layer1(
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

        # layer 2
        a, s, b, c = self.layer2(
            a, s, b, c,
            ei_app_sys, ew_app_sys,
            ei_sys_app, ew_sys_app,
            ei_app_bnd, ew_app_bnd,
            ei_bnd_app, ew_bnd_app,
            ei_app_cmp, ew_app_cmp,
            ei_cmp_app, ew_cmp_app,
        )
        a = F.relu(a)  # only need a for the head

        # MLP head on application nodes
        logits_app = self.mlp(a)  # [N_app, out_dim]
        return logits_app

# ---------- Build your data tensors ----------
@torch.no_grad()
def build_from_counts(
    n_app: int, n_sys: int, n_bnd: int, n_cmp: int,  # node counts
    e_app_sys: int, e_sys_app: int, e_app_bnd: int, e_bnd_app: int, e_app_cmp: int, e_cmp_app: int,
):
    # Features are torch.arange per your note (shape [N_type, 1])
    x_app = torch.arange(n_app, dtype=torch.float32).unsqueeze(1)
    x_sys = torch.arange(n_sys, dtype=torch.float32).unsqueeze(1)
    x_bnd = torch.arange(n_bnd, dtype=torch.float32).unsqueeze(1)
    x_cmp = torch.arange(n_cmp, dtype=torch.float32).unsqueeze(1)

    def rnd_edges(n_src, n_dst, e):
        src = torch.randint(0, n_src, (e,), dtype=torch.long)
        dst = torch.randint(0, n_dst, (e,), dtype=torch.long)
        ei = torch.stack([src, dst], dim=0)  # [2, e]
        ew = torch.ones(e, dtype=torch.float32)
        return ei, ew

    ei_app_sys, ew_app_sys = rnd_edges(n_app, n_sys, e_app_sys)
    ei_sys_app, ew_sys_app = rnd_edges(n_sys, n_app, e_sys_app)
    ei_app_bnd, ew_app_bnd = rnd_edges(n_app, n_bnd, e_app_bnd)
    ei_bnd_app, ew_bnd_app = rnd_edges(n_bnd, n_app, e_bnd_app)
    ei_app_cmp, ew_app_cmp = rnd_edges(n_app, n_cmp, e_app_cmp)
    ei_cmp_app, ew_cmp_app = rnd_edges(n_cmp, n_app, e_cmp_app)

    return (
        x_app, x_sys, x_bnd, x_cmp,
        ei_app_sys, ew_app_sys,
        ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd,
        ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp,
        ei_cmp_app, ew_cmp_app,
    )

# ---------- Export to .pte ----------
def export_to_pte(model: nn.Module, example_inputs: Tuple[torch.Tensor, ...], pte_path="hetero6_app_only.pte"):
    (x_app, x_sys, x_bnd, x_cmp,
     ei_app_sys, ew_app_sys,
     ei_sys_app, ew_sys_app,
     ei_app_bnd, ew_app_bnd,
     ei_bnd_app, ew_bnd_app,
     ei_app_cmp, ew_app_cmp,
     ei_cmp_app, ew_cmp_app) = example_inputs

    # dynamic shape hints (first dims and E_r vary)
    dyn = {
        "x_app":      {0: Dim("N_app", min=1, max=int(x_app.shape[0]))},
        "x_sys":      {0: Dim("N_sys", min=1, max=int(x_sys.shape[0]))},
        "x_bnd":      {0: Dim("N_bnd", min=1, max=int(x_bnd.shape[0]))},
        "x_cmp":      {0: Dim("N_cmp", min=1, max=int(x_cmp.shape[0]))},

        "ei_app_sys": {1: Dim("E0", min=1, max=int(ei_app_sys.shape[1]))},
        "ew_app_sys": {0: Dim("E0", min=1, max=int(ew_app_sys.shape[0]))},

        "ei_sys_app": {1: Dim("E1", min=1, max=int(ei_sys_app.shape[1]))},
        "ew_sys_app": {0: Dim("E1", min=1, max=int(ew_sys_app.shape[0]))},

        "ei_app_bnd": {1: Dim("E2", min=1, max=int(ei_app_bnd.shape[1]))},
        "ew_app_bnd": {0: Dim("E2", min=1, max=int(ew_app_bnd.shape[0]))},

        "ei_bnd_app": {1: Dim("E3", min=1, max=int(ei_bnd_app.shape[1]))},
        "ew_bnd_app": {0: Dim("E3", min=1, max=int(ew_bnd_app.shape[0]))},

        "ei_app_cmp": {1: Dim("E4", min=1, max=int(ei_app_cmp.shape[1]))},
        "ew_app_cmp": {0: Dim("E4", min=1, max=int(ew_app_cmp.shape[0]))},

        "ei_cmp_app": {1: Dim("E5", min=1, max=int(ei_cmp_app.shape[1]))},
        "ew_cmp_app": {0: Dim("E5", min=1, max=int(ew_cmp_app.shape[0]))},
    }

    exp = export(model.eval(), example_inputs, dynamic_shapes=dyn)

    # Lower to ExecuTorch (XNNPACK delegate) and write .pte
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    et_prog = to_edge_transform_and_lower(exp, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(pte_path, "wb") as f:
        f.write(et_prog.buffer)
    return os.path.abspath(pte_path)

# ---------- Run with ExecuTorch runtime ----------
def run_with_executorch(pte_path: str, example_inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    from executorch.runtime import Runtime, Verification
    rt = Runtime.get()
    prog = rt.load_program(pte_path, verification=Verification.Minimal)
    meth = prog.load_method("forward")
    outs = meth.execute(list(example_inputs))  # returns [tensor]
    return outs[0]

# ---------- Demo ----------
def main():
    torch.manual_seed(0)

    # Node counts and edges per relation (toy)
    n_app, n_sys, n_bnd, n_cmp = 100, 400, 50, 80
    e0, e1, e2, e3, e4, e5 = 1200, 1200, 600, 600, 1200, 1200

    inputs = build_from_counts(n_app, n_sys, n_bnd, n_cmp, e0, e1, e2, e3, e4, e5)

    # Features are 1-D arange scalars → F_app=F_sys=F_bnd=F_cmp=1
    model = Hetero6Net(f_app=1, f_sys=1, f_bnd=1, f_cmp=1, hidden=128, out_dim=7, dropout=0.2).eval()

    pte = export_to_pte(model, inputs, pte_path="hetero6_app_only.pte")
    print("Saved:", pte)

    logits = run_with_executorch(pte, inputs)
    print("ExecuTorch logits shape:", tuple(logits.shape))  # [N_app, out_dim]

if __name__ == "__main__":
    main()
