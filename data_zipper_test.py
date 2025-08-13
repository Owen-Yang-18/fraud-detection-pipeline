# data_zip.py — build a compact, test-only app subgraph from full edges.
# - Filters edges to those touching test_idx app nodes (on the app side)
# - Re-indexes app nodes to 0..N_test-1
# - Writes compact x_app, optional y_app/app_ids, and the 6 remapped edge sets
# - Leaves x_sys/x_bnd/x_cmp unchanged (you can add similar compaction if desired)

import json, zipfile, numpy as np, torch
from typing import List, Optional, Dict, Tuple

# ----------------- low-level utils -----------------

def to_le_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        return t.numpy().astype(np.float32).tobytes(order="C")
    if t.dtype == torch.int64:
        return t.numpy().astype(np.int64).tobytes(order="C")
    raise ValueError(f"Unsupported dtype {t.dtype}")

def dtype_str(t: torch.Tensor) -> str:
    return {torch.float32: "float32", torch.int64: "int64"}[t.dtype]

def build_old2new_from_keep(keep: torch.Tensor, N_app: int) -> torch.Tensor:
    """
    keep: int64 [N_test] indices into the original app set.
    returns old2new: int64 [N_app], -1 for dropped, new id for kept.
    """
    assert keep.dtype == torch.int64
    old2new = torch.full((N_app,), -1, dtype=torch.int64, device=keep.device)
    old2new[keep] = torch.arange(keep.numel(), dtype=torch.int64, device=keep.device)
    return old2new

def select_edges_by_app_src(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, old2new: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep edges with SOURCE app in keep-set and remap src:
      edge_index: int64 [2, E], app is row-0
      edge_weight: float32 [E]
    """
    assert edge_index.dtype == torch.int64 and edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_weight.dtype == torch.float32 and edge_weight.dim() == 1 and edge_weight.size(0) == edge_index.size(1)

    src_old = edge_index[0]
    src_new = old2new[src_old]                         # -1 for dropped
    mask = src_new >= 0
    if mask.sum() == 0:
        return edge_index.new_empty((2,0)), edge_weight.new_empty((0,))
    ei2 = edge_index[:, mask]
    ew2 = edge_weight[mask]
    ei2 = torch.stack([src_new[mask], ei2[1]], dim=0)  # remap src
    return ei2.contiguous(), ew2.contiguous()

def select_edges_by_app_dst(
    edge_index: torch.Tensor, edge_weight: torch.Tensor, old2new: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep edges with DEST app in keep-set and remap dst:
      edge_index: int64 [2, E], app is row-1
      edge_weight: float32 [E]
    """
    assert edge_index.dtype == torch.int64 and edge_index.dim() == 2 and edge_index.size(0) == 2
    assert edge_weight.dtype == torch.float32 and edge_weight.dim() == 1 and edge_weight.size(0) == edge_index.size(1)

    dst_old = edge_index[1]
    dst_new = old2new[dst_old]
    mask = dst_new >= 0
    if mask.sum() == 0:
        return edge_index.new_empty((2,0)), edge_weight.new_empty((0,))
    ei2 = edge_index[:, mask]
    ew2 = edge_weight[mask]
    ei2 = torch.stack([ei2[0], dst_new[mask]], dim=0)  # remap dst
    return ei2.contiguous(), ew2.contiguous()

# ----------------- main compaction from full edges -----------------

def export_zip_compact_app_from_full_edges(
    path_zip: str,
    # node features
    x_app: torch.Tensor, x_sys: torch.Tensor, x_bnd: torch.Tensor, x_cmp: torch.Tensor,
    # FULL edges (unfiltered):
    ei_app_sys: torch.Tensor, ew_app_sys: torch.Tensor,   # app -> sys
    ei_sys_app: torch.Tensor, ew_sys_app: torch.Tensor,   # sys -> app
    ei_app_bnd: torch.Tensor, ew_app_bnd: torch.Tensor,   # app -> binder
    ei_bnd_app: torch.Tensor, ew_bnd_app: torch.Tensor,   # binder -> app
    ei_app_cmp: torch.Tensor, ew_app_cmp: torch.Tensor,   # app -> composite
    ei_cmp_app: torch.Tensor, ew_cmp_app: torch.Tensor,   # composite -> app
    *,
    class_names: Optional[List[str]] = None,
    app_ids: Optional[List[str]] = None,          # length == x_app.size(0)
    y_app: Optional[torch.Tensor] = None,         # int64 1..5, length == x_app.size(0)
    label_map: Optional[Dict[int,str]] = None,
    test_idx: torch.Tensor,                       # int64 indices into original apps to KEEP
    write_orig_idx: bool = True,
):
    """
    Build a test-only app subgraph from FULL edges using test_idx (no shuffle).
    - x_app/y_app/app_ids are compacted to test_idx order.
    - All six app-* edge sets are filtered to edges touching kept apps and re-indexed on the app side.
    - x_sys/x_bnd/x_cmp are unchanged.
    """
    assert test_idx.dtype == torch.int64
    N_app = x_app.size(0)
    old2new = build_old2new_from_keep(test_idx, N_app)       # [-1 or new_id]

    # compact app features / labels / ids (order = test_idx)
    x_app_c = x_app.index_select(0, test_idx)                # torch.index_select :contentReference[oaicite:2]{index=2}
    y_app_c = y_app.index_select(0, test_idx) if y_app is not None else None
    app_ids_c = [app_ids[i.item()] for i in test_idx.cpu()] if app_ids is not None else None

    # filter + reindex edges where app is src (row-0)
    ei_app_sys_c, ew_app_sys_c = select_edges_by_app_src(ei_app_sys, ew_app_sys, old2new)
    ei_app_bnd_c, ew_app_bnd_c = select_edges_by_app_src(ei_app_bnd, ew_app_bnd, old2new)
    ei_app_cmp_c, ew_app_cmp_c = select_edges_by_app_src(ei_app_cmp, ew_app_cmp, old2new)

    # filter + reindex edges where app is dst (row-1)
    ei_sys_app_c, ew_sys_app_c = select_edges_by_app_dst(ei_sys_app, ew_sys_app, old2new)
    ei_bnd_app_c, ew_bnd_app_c = select_edges_by_app_dst(ei_bnd_app, ew_bnd_app, old2new)
    ei_cmp_app_c, ew_cmp_app_c = select_edges_by_app_dst(ei_cmp_app, ew_cmp_app, old2new)

    # pack for manifest
    names = [
        "x_app","x_sys","x_bnd","x_cmp",
        "ei_app_sys","ew_app_sys",
        "ei_sys_app","ew_sys_app",
        "ei_app_bnd","ew_app_bnd",
        "ei_bnd_app","ew_bnd_app",
        "ei_app_cmp","ew_app_cmp",
        "ei_cmp_app","ew_cmp_app",
    ]
    arrays = [
        x_app_c, x_sys, x_bnd, x_cmp,
        ei_app_sys_c, ew_app_sys_c,
        ei_sys_app_c, ew_sys_app_c,
        ei_app_bnd_c, ew_app_bnd_c,
        ei_bnd_app_c, ew_bnd_app_c,
        ei_app_cmp_c, ew_app_cmp_c,
        ei_cmp_app_c, ew_cmp_app_c,
    ]
    files = [f"{n}.bin" for n in names]

    manifest = {
        "tensors":[
            {"name":n, "dtype":dtype_str(t), "shape":list(t.shape), "file":fn}
            for n,t,fn in zip(names, arrays, files)
        ],
        "class_names": class_names,
        "app_ids": app_ids_c,
    }
    if label_map is not None:
        manifest["label_map"] = {str(k): v for k,v in label_map.items()}

    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for n,t,fn in zip(names, arrays, files):
            z.writestr(fn, to_le_bytes(t))
        if y_app_c is not None:
            if y_app_c.dtype != torch.int64: y_app_c = y_app_c.to(torch.int64)
            z.writestr("y_app.bin", to_le_bytes(y_app_c))
        if write_orig_idx:
            z.writestr("orig_app_idx.bin", to_le_bytes(test_idx.to(torch.int64)))

# ----------------- example usage -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # toy sizes
    N_app, N_sys, N_bnd, N_cmp = 10, 6, 4, 5
    F = 1
    x_app = torch.arange(N_app, dtype=torch.float32).view(-1, F)
    x_sys = torch.arange(N_sys, dtype=torch.float32).view(-1, F)
    x_bnd = torch.arange(N_bnd, dtype=torch.float32).view(-1, F)
    x_cmp = torch.arange(N_cmp, dtype=torch.float32).view(-1, F)

    def rnd_e(n_src, n_dst, e):
        src = torch.randint(n_src, (e,), dtype=torch.int64)
        dst = torch.randint(n_dst, (e,), dtype=torch.int64)
        w = torch.ones(e, dtype=torch.float32)
        return torch.stack([src, dst], dim=0), w

    # FULL edges (unfiltered)
    ei_app_sys, ew_app_sys = rnd_e(N_app, N_sys, 20)
    ei_sys_app, ew_sys_app = rnd_e(N_sys, N_app, 20)
    ei_app_bnd, ew_app_bnd = rnd_e(N_app, N_bnd, 10)
    ei_bnd_app, ew_bnd_app = rnd_e(N_bnd, N_app, 10)
    ei_app_cmp, ew_app_cmp = rnd_e(N_app, N_cmp, 12)
    ei_cmp_app, ew_cmp_app = rnd_e(N_cmp, N_app, 12)

    # labels and ids
    y_app = torch.randint(1, 6, (N_app,), dtype=torch.int64)
    app_ids = [f"pkg.{i}" for i in range(N_app)]
    label_map = {1:"adware", 2:"banking", 3:"sms", 4:"riskware", 5:"benign"}

    # choose a test subset of app nodes (no shuffle of underlying arrays here)
    test_idx = torch.tensor([0,2,5,7], dtype=torch.int64)

    export_zip_compact_app_from_full_edges(
        "data_test_only.zip",
        x_app, x_sys, x_bnd, x_cmp,
        ei_app_sys, ew_app_sys, ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd, ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp, ei_cmp_app, ew_cmp_app,
        class_names=["adware","banking","sms","riskware","benign"],
        app_ids=app_ids,
        y_app=y_app,
        label_map=label_map,
        test_idx=test_idx,
        write_orig_idx=True,
    )
    print("Wrote data_test_only.zip")
