# dictless_hetero_app_sys_binder_comp.py
# ------------------------------------------------------------
# Hetero GNN for your schema (no dicts in forward), export to .pte,
# then load and run in ExecuTorch.
#
# Node types:
#   0=application, 1=syscall, 2=binder, 3=composite_behavior
# Edge relations (R=6) in fixed order:
#   0: application -> syscall           (uses_sys)
#   1: syscall     -> application       (uses_sys_by)
#   2: application -> binder            (uses_binder)
#   3: binder      -> application       (uses_binder_by)
#   4: application -> composite_behavior(uses_composite)
#   5: composite_behavior -> application(uses_composite_by)
#
# Each node feature = torch.arange(count) as a single scalar feature.
# The forward takes only tensors (ExecuTorch-friendly).

import os
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, Dim

# -----------------------------
# Constants (node / relation id)
# -----------------------------
APP, SYS, BND, CMP = 0, 1, 2, 3
REL_USES_SYS, REL_USES_SYS_BY, REL_USES_BND, REL_USES_BND_BY, REL_USES_CMP, REL_USES_CMP_BY = range(6)

# -------------------------------
# Minimal edge-weighted GraphConv
# -------------------------------
class SimpleGraphConv(nn.Module):
    """
    x'_i = W_self x_i + sum_{j in N(i)} w_(j->i) * (W_nei x_j)
    edge_index: [2, E] with [src; dst]
    edge_weight: [E]
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_nei  = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_self = self.lin_self(x)                         # [N, Fout]
        msg = self.lin_nei(x).index_select(0, src)        # [E, Fout]
        msg = msg * edge_weight.view(-1, 1)               # weight messages
        out = torch.zeros_like(h_self)
        out.index_add_(0, dst, msg)                       # sum into dst
        return out + h_self

# --------------------------------------
# Dict-less hetero layer (sum over R rels)
# --------------------------------------
class DictlessHeteroLayer(nn.Module):
    """
    Owns one SimpleGraphConv per relation; SUMs their outputs.
    Edges for relation r live in slice [rel_ptr[r]:rel_ptr[r+1]) of global edge tensors.
    """
    def __init__(self, hidden_dim: int, num_relations: int = 6):
        super().__init__()
        self.num_rel = num_relations
        self.convs = nn.ModuleList([SimpleGraphConv(hidden_dim, hidden_dim) for _ in range(num_relations)])

    def forward(self,
                x: torch.Tensor,               # [N, H]
                edge_index: torch.Tensor,      # [2, E]
                edge_weight: torch.Tensor,     # [E]
                rel_ptr: torch.Tensor) -> torch.Tensor:   # [R+1]
        out = torch.zeros_like(x)
        for r in range(self.num_rel):
            b = int(rel_ptr[r].item())
            e = int(rel_ptr[r + 1].item())
            if e > b:
                ei = edge_index[:, b:e]
                ew = edge_weight[b:e]
                out = out + self.convs[r](x, ei, ew)
        return out

# ---------------------------------------------------------
# Two hetero layers + MLP head (classify application nodes)
# ---------------------------------------------------------
class DictlessHeteroGNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_node_types: int = 4,
                 num_relations: int = 6,
                 type_emb_dim: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        self.type_emb = nn.Embedding(num_node_types, type_emb_dim) if type_emb_dim > 0 else None
        in_plus = in_dim + (type_emb_dim if self.type_emb is not None else 0)

        self.lin_in = nn.Linear(in_plus, hidden_dim)
        self.layer1 = DictlessHeteroLayer(hidden_dim, num_relations)
        self.layer2 = DictlessHeteroLayer(hidden_dim, num_relations)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self,
                x: torch.Tensor,               # [N, Fin]
                edge_index: torch.Tensor,      # [2, E]
                edge_weight: torch.Tensor,     # [E]
                rel_ptr: torch.Tensor,         # [R+1]
                node_type: torch.Tensor,       # [N]
                target_index: torch.Tensor) -> torch.Tensor:  # [M, out_dim]
        if self.type_emb is not None:
            x = torch.cat([x, self.type_emb(node_type)], dim=-1)

        h = F.relu(self.lin_in(x))
        h = self.layer1(h, edge_index, edge_weight, rel_ptr)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(h, edge_index, edge_weight, rel_ptr)
        h = F.relu(h)
        logits = self.mlp(h)                                # [N, out_dim]
        return logits.index_select(0, target_index)         # [M, out_dim] (applications only)

# ------------------------------------------------------------
# Packing: hetero → homogeneous tensors for this schema
# ------------------------------------------------------------
@torch.no_grad()
def _concat_node_features(n_app: int, n_sys: int, n_bnd: int, n_cmp: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build x (scalar arange feature per node) and node_type vector.
    """
    feat_app = torch.arange(n_app, dtype=torch.float32).unsqueeze(1)
    feat_sys = torch.arange(n_sys, dtype=torch.float32).unsqueeze(1)
    feat_bnd = torch.arange(n_bnd, dtype=torch.float32).unsqueeze(1)
    feat_cmp = torch.arange(n_cmp, dtype=torch.float32).unsqueeze(1)
    x = torch.cat([feat_app, feat_sys, feat_bnd, feat_cmp], dim=0)     # [N, 1]

    node_type = torch.cat([
        torch.full((n_app,), APP, dtype=torch.long),
        torch.full((n_sys,), SYS, dtype=torch.long),
        torch.full((n_bnd,), BND, dtype=torch.long),
        torch.full((n_cmp,), CMP, dtype=torch.long),
    ], dim=0)  # [N]
    return x, node_type

@torch.no_grad()
def _offsets(n_app: int, n_sys: int, n_bnd: int, n_cmp: int) -> Tuple[int, int, int, int]:
    base_app = 0
    base_sys = n_app
    base_bnd = n_app + n_sys
    base_cmp = n_app + n_sys + n_bnd
    return base_app, base_sys, base_bnd, base_cmp

@torch.no_grad()
def _pack_edges_with_rel(
    src_local: torch.Tensor, dst_local: torch.Tensor, weight: Optional[torch.Tensor],
    src_base: int, dst_base: int, rel_id: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map local IDs to global, build edge_index, weights, and edge_type vec for one relation."""
    assert src_local.dtype == torch.long and dst_local.dtype == torch.long
    if weight is None:
        weight = torch.ones(src_local.numel(), dtype=torch.float32)
    ei = torch.stack([src_local + src_base, dst_local + dst_base], dim=0)   # [2, E_rel]
    et = torch.full((src_local.numel(),), rel_id, dtype=torch.long)
    return ei, weight.to(torch.float32), et

@torch.no_grad()
def _sort_by_relation(edge_index: torch.Tensor,
                      edge_weight: torch.Tensor,
                      edge_type: torch.Tensor,
                      R: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort edges by relation and build rel_ptr."""
    perm = torch.argsort(edge_type)
    ei = edge_index[:, perm]
    ew = edge_weight[perm]
    et = edge_type[perm]
    counts = torch.bincount(et, minlength=R)
    rel_ptr = torch.zeros(R + 1, dtype=torch.long)
    rel_ptr[1:] = torch.cumsum(counts, dim=0)
    return ei, ew, rel_ptr

@torch.no_grad()
def build_inputs_from_hetero(
    # node counts
    n_app: int, n_sys: int, n_bnd: int, n_cmp: int,
    # edges for each relation (local ids, 0-based within its type)
    app_uses_sys_src: torch.Tensor, app_uses_sys_dst: torch.Tensor, w_app_sys: Optional[torch.Tensor],
    sys_uses_by_src: torch.Tensor,  sys_uses_by_dst: torch.Tensor,  w_sys_app: Optional[torch.Tensor],
    app_uses_bnd_src: torch.Tensor, app_uses_bnd_dst: torch.Tensor, w_app_bnd: Optional[torch.Tensor],
    bnd_uses_by_src: torch.Tensor,  bnd_uses_by_dst: torch.Tensor,  w_bnd_app: Optional[torch.Tensor],
    app_uses_cmp_src: torch.Tensor, app_uses_cmp_dst: torch.Tensor, w_app_cmp: Optional[torch.Tensor],
    cmp_uses_by_src: torch.Tensor,  cmp_uses_by_dst: torch.Tensor,  w_cmp_app: Optional[torch.Tensor],
    # target type to classify (default: application)
    target_type_id: int = APP,
):
    x, node_type = _concat_node_features(n_app, n_sys, n_bnd, n_cmp)
    base_app, base_sys, base_bnd, base_cmp = _offsets(n_app, n_sys, n_bnd, n_cmp)

    # Pack each relation
    ei0, ew0, et0 = _pack_edges_with_rel(app_uses_sys_src, app_uses_sys_dst, w_app_sys, base_app, base_sys, REL_USES_SYS)
    ei1, ew1, et1 = _pack_edges_with_rel(sys_uses_by_src, sys_uses_by_dst, w_sys_app, base_sys, base_app, REL_USES_SYS_BY)
    ei2, ew2, et2 = _pack_edges_with_rel(app_uses_bnd_src, app_uses_bnd_dst, w_app_bnd, base_app, base_bnd, REL_USES_BND)
    ei3, ew3, et3 = _pack_edges_with_rel(bnd_uses_by_src,  bnd_uses_by_dst,  w_bnd_app, base_bnd, base_app, REL_USES_BND_BY)
    ei4, ew4, et4 = _pack_edges_with_rel(app_uses_cmp_src, app_uses_cmp_dst, w_app_cmp, base_app, base_cmp, REL_USES_CMP)
    ei5, ew5, et5 = _pack_edges_with_rel(cmp_uses_by_src,  cmp_uses_by_dst,  w_cmp_app, base_cmp, base_app, REL_USES_CMP_BY)

    edge_index = torch.cat([ei0, ei1, ei2, ei3, ei4, ei5], dim=1)            # [2, E]
    edge_weight = torch.cat([ew0, ew1, ew2, ew3, ew4, ew5], dim=0)            # [E]
    edge_type   = torch.cat([et0, et1, et2, et3, et4, et5], dim=0)            # [E]

    # Sort by relation and build rel_ptr
    edge_index, edge_weight, rel_ptr = _sort_by_relation(edge_index, edge_weight, edge_type, R=6)

    # Target nodes = the chosen type
    target_index = (node_type == target_type_id).nonzero(as_tuple=False).view(-1)
    if target_index.numel() == 0:
        target_index = torch.tensor([0], dtype=torch.long)

    return (x, edge_index, edge_weight, rel_ptr, node_type, target_index)

# ------------------------------------------------------------
# Export → Lower → Save .pte → Load & Run (ExecuTorch)
# ------------------------------------------------------------
def export_to_pte(model: nn.Module,
                  example_inputs: Tuple[torch.Tensor, ...],
                  pte_path: str = "hetero_app_model.pte") -> str:
    x, edge_index, edge_weight, rel_ptr, node_type, target_index = example_inputs

    dynamic_shapes = {
        "x":            {0: Dim("N", min=1, max=int(x.shape[0]))},
        "edge_index":   {1: Dim("E", min=1, max=int(edge_index.shape[1]))},
        "edge_weight":  {0: Dim("E", min=1, max=int(edge_weight.shape[0]))},
        # rel_ptr is length R+1 (fixed since R=6)
        "node_type":    {0: Dim("N", min=1, max=int(node_type.shape[0]))},
        "target_index": {0: Dim("M", min=1, max=max(1, int(target_index.shape[0])) )},
    }

    aten_graph = export(model.eval(), example_inputs, dynamic_shapes=dynamic_shapes)

    # Lower with XNNPACK delegate and serialize to .pte
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    et_prog = to_edge_transform_and_lower(
        aten_graph,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with open(pte_path, "wb") as f:
        f.write(et_prog.buffer)
    return os.path.abspath(pte_path)

def run_with_executorch(pte_path: str, example_inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    from executorch.runtime import Runtime, Verification
    et_runtime = Runtime.get()
    program = et_runtime.load_program(pte_path, verification=Verification.Minimal)
    method = program.load_method("forward")
    outputs = method.execute(list(example_inputs))  # list of tensors, same order as forward
    return outputs[0]  # logits for target nodes

# ------------------------------------------------------------
# Demo main: build a toy graph matching your schema
# ------------------------------------------------------------
def main():
    torch.manual_seed(0)

    # Node counts
    n_app, n_sys, n_bnd, n_cmp = 100, 400, 50, 80
    Fin, hidden, num_classes = 1, 128, 7   # Fin=1 because features are arange scalars

    # Create some random local edges for each relation (replace with real tensors in your pipeline)
    def rnd_edges(n_src, n_dst, e):
        return torch.randint(0, n_src, (e,), dtype=torch.long), torch.randint(0, n_dst, (e,), dtype=torch.long)

    e_each = 1000
    a_s_src, a_s_dst = rnd_edges(n_app, n_sys, e_each)
    s_a_src, s_a_dst = rnd_edges(n_sys, n_app, e_each)
    a_b_src, a_b_dst = rnd_edges(n_app, n_bnd, e_each // 2)
    b_a_src, b_a_dst = rnd_edges(n_bnd, n_app, e_each // 2)
    a_c_src, a_c_dst = rnd_edges(n_app, n_cmp, e_each)
    c_a_src, c_a_dst = rnd_edges(n_cmp, n_app, e_each)

    # Optional edge weights (here all ones)
    ones = lambda n: torch.ones(n, dtype=torch.float32)

    inputs = build_inputs_from_hetero(
        n_app, n_sys, n_bnd, n_cmp,
        a_s_src, a_s_dst, ones(a_s_src.numel()),
        s_a_src, s_a_dst, ones(s_a_src.numel()),
        a_b_src, a_b_dst, ones(a_b_src.numel()),
        b_a_src, b_a_dst, ones(b_a_src.numel()),
        a_c_src, a_c_dst, ones(a_c_src.numel()),
        c_a_src, c_a_dst, ones(c_a_src.numel()),
        target_type_id=APP,  # classify application nodes
    )

    # Build model
    model = DictlessHeteroGNN(
        in_dim=Fin, hidden_dim=hidden, out_dim=num_classes,
        num_node_types=4, num_relations=6, type_emb_dim=8, dropout=0.2
    ).eval()

    # Export → .pte
    pte_path = export_to_pte(model, inputs, pte_path="hetero_app_model.pte")
    print("Saved:", pte_path)

    # Load & run in ExecuTorch
    logits = run_with_executorch(pte_path, inputs)
    print("ExecuTorch logits shape:", tuple(logits.shape))  # [#application_nodes, num_classes]

if __name__ == "__main__":
    main()
