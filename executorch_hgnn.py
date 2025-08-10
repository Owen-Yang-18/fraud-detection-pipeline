# dictless_hetero_executorch_demo.py
# ------------------------------------------------------------
# Two-layer dict-less "hetero" GNN + MLP head, export to .pte,
# then load and run with ExecuTorch runtime.
#
# Inputs to forward (all tensors):
#   x            : [N, Fin]            node features
#   edge_index   : [2, E]              concatenated edges across relations
#   edge_weight  : [E]                 per-edge weights (can be all ones)
#   rel_ptr      : [R+1]               edge segments per relation
#   node_type    : [N] (long)          node type id in [0..num_node_types-1]
#   target_index : [M] (long)          indices of nodes of the target type
#
# This file is self-contained and runnable.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export, Dim

# -------------------------------
# Minimal GraphConv (edge-weighted)
# -------------------------------
class SimpleGraphConv(nn.Module):
    """
    GraphConv with edge weights (source -> destination):

        h_out[i] = W_self x[i] + sum_{j in N(i)} w[j->i] * (W_nei x[j])

    edge_index: [2, E] with row=src, col=dst (0-based)
    edge_weight: [E] or None
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_nei  = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,               # [N, Fin]
        edge_index: torch.Tensor,      # [2, E], long
        edge_weight: torch.Tensor | None,  # [E]
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_self = self.lin_self(x)                       # [N, Fout]
        msg = self.lin_nei(x).index_select(0, src)     # [E, Fout]
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)        # weight messages
        out = torch.zeros_like(h_self)
        out.index_add_(0, dst, msg)                    # sum into destinations
        return out + h_self


# --------------------------------------
# Dict-less "hetero" layer: per-relation
# --------------------------------------
class DictlessHeteroLayer(nn.Module):
    """
    Owns one SimpleGraphConv per relation, and SUMs their outputs.
    Edges for relation r are in the slice rel_ptr[r]:rel_ptr[r+1].
    """
    def __init__(self, hidden_dim: int, num_relations: int):
        super().__init__()
        self.num_rel = num_relations
        self.convs = nn.ModuleList(
            [SimpleGraphConv(hidden_dim, hidden_dim) for _ in range(num_relations)]
        )

    def forward(
        self,
        x: torch.Tensor,               # [N, H]
        edge_index: torch.Tensor,      # [2, E]
        edge_weight: torch.Tensor | None,  # [E]
        rel_ptr: torch.Tensor,         # [R+1], long
    ) -> torch.Tensor:
        out = torch.zeros_like(x)
        # iterate relations by slicing the global edge list
        for r in range(self.num_rel):
            b = int(rel_ptr[r].item())
            e = int(rel_ptr[r + 1].item())
            if e > b:
                ei = edge_index[:, b:e]
                ew = edge_weight[b:e] if edge_weight is not None else None
                out = out + self.convs[r](x, ei, ew)
        return out


# ---------------------------------------------------------
# Two-layer dict-less hetero GNN + MLP head for target type
# ---------------------------------------------------------
class DictlessHeteroGNN(nn.Module):
    """
    Two DictlessHeteroLayer blocks and an MLP classifier applied
    only to nodes indexed by `target_index`.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_node_types: int = 4,
        num_relations: int = 6,
        type_emb_dim: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_rel = num_relations
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

    def forward(
        self,
        x: torch.Tensor,               # [N, Fin]
        edge_index: torch.Tensor,      # [2, E]
        edge_weight: torch.Tensor,     # [E]
        rel_ptr: torch.Tensor,         # [R+1]
        node_type: torch.Tensor,       # [N]
        target_index: torch.Tensor,    # [M]
    ) -> torch.Tensor:                 # [M, out_dim]
        if self.type_emb is not None:
            x = torch.cat([x, self.type_emb(node_type)], dim=-1)

        h = F.relu(self.lin_in(x))
        h = self.layer1(h, edge_index, edge_weight, rel_ptr)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(h, edge_index, edge_weight, rel_ptr)
        h = F.relu(h)

        logits = self.mlp(h)  # [N, out_dim]
        return logits.index_select(0, target_index)


# ------------------------------------------------------------
# Utilities: pack edges by relation; build a synthetic test set
# ------------------------------------------------------------
@torch.no_grad()
def pack_relptr(
    edge_index: torch.Tensor,   # [2, E]
    edge_type: torch.Tensor,    # [E] in [0..R-1]
    edge_weight: torch.Tensor,  # [E]
    num_relations: int,
):
    # sort edges by relation to get contiguous segments
    perm = torch.argsort(edge_type)
    ei = edge_index[:, perm]
    ew = edge_weight[perm]
    et = edge_type[perm]
    counts = torch.bincount(et, minlength=num_relations)
    rel_ptr = torch.zeros(num_relations + 1, dtype=torch.long)
    rel_ptr[1:] = torch.cumsum(counts, dim=0)
    return ei, ew, rel_ptr


@torch.no_grad()
def build_synthetic_inputs(
    N=4000, E=30000, Fin=32, R=6, T=4, target_type_id=2, num_classes=5, device="cpu"
):
    x = torch.randn(N, Fin, device=device)
    edge_index = torch.randint(0, N, (2, E), dtype=torch.long, device=device)
    edge_weight = torch.rand(E, device=device)
    edge_type = torch.randint(0, R, (E,), dtype=torch.long, device=device)
    node_type = torch.randint(0, T, (N,), dtype=torch.long, device=device)

    edge_index, edge_weight, rel_ptr = pack_relptr(edge_index, edge_type, edge_weight, R)

    target_index = (node_type == target_type_id).nonzero(as_tuple=False).view(-1)
    if target_index.numel() == 0:
        # make sure we have at least one target for the demo
        target_index = torch.tensor([0], dtype=torch.long, device=device)

    return (x, edge_index, edge_weight, rel_ptr, node_type, target_index), num_classes


# ------------------------------------------------------------
# Export → Lower → Save .pte → Load & Run with ExecuTorch runtime
# ------------------------------------------------------------
def export_to_pte(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    pte_path: str = "hetero_model.pte",
):
    # 1) torch.export (set dynamic shape bounds for N, E, M)
    x, edge_index, edge_weight, rel_ptr, node_type, target_index = example_inputs

    dynamic_shapes = {
        # x: [N, Fin]  -> dim 0 varies
        "x": {0: Dim("N", min=1, max=max(1, int(x.shape[0])) )},
        # edge_index: [2, E] -> dim 1 varies
        "edge_index": {1: Dim("E", min=1, max=max(1, int(edge_index.shape[1])) )},
        # edge_weight: [E]   -> dim 0 varies
        "edge_weight": {0: Dim("E", min=1, max=max(1, int(edge_weight.shape[0])) )},
        # rel_ptr: [R+1]     -> keep fixed for a fixed R
        # node_type: [N]     -> dim 0 varies
        "node_type": {0: Dim("N", min=1, max=max(1, int(node_type.shape[0])) )},
        # target_index: [M]  -> dim 0 varies
        "target_index": {0: Dim("M", min=1, max=max(1, int(target_index.shape[0])) )},
    }

    exported = export(
        model.eval(),
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    # 2) Lower to ExecuTorch with XNNPACK backend
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    et_program = to_edge_transform_and_lower(
        exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    # 3) Save .pte
    with open(pte_path, "wb") as f:
        f.write(et_program.buffer)
    return os.path.abspath(pte_path)


def run_with_executorch(pte_path: str, example_inputs: tuple[torch.Tensor, ...]):
    # Load the program and execute "forward" with ExecuTorch runtime
    from executorch.runtime import Runtime, Verification

    runtime = Runtime.get()
    program = runtime.load_program(pte_path, verification=Verification.Minimal)
    method = program.load_method("forward")

    # ExecuTorch expects a list/tuple of tensors (same order as forward)
    outputs = method.execute(list(example_inputs))
    return outputs


def main():
    # 0) Build a toy batch and the model
    (x, edge_index, edge_weight, rel_ptr, node_type, target_index), num_classes = \
        build_synthetic_inputs()

    model = DictlessHeteroGNN(
        in_dim=x.shape[1],
        hidden_dim=128,
        out_dim=num_classes,
        num_node_types=4,
        num_relations=6,
        type_emb_dim=8,
        dropout=0.2,
    ).eval()

    # 1) Export → Lower → Save
    pte_path = export_to_pte(
        model,
        (x, edge_index, edge_weight, rel_ptr, node_type, target_index),
        pte_path="hetero_model.pte",
    )
    print(f"Saved: {pte_path}")

    # 2) Load & run with ExecuTorch runtime
    outputs = run_with_executorch(pte_path, (x, edge_index, edge_weight, rel_ptr, node_type, target_index))
    # ExecuTorch returns a list; our forward has one tensor output
    logits = outputs[0]
    print("ExecuTorch output shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
