# data_zip_subsets.py
# Build compact, test-only zips from FULL edges + test_idx, with shuffle and size subsets.
# Requires: torch, numpy

import json, zipfile, numpy as np, torch
from typing import List, Optional, Dict, Tuple

# ----------------- binary I/O helpers -----------------

def _to_le_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        return t.numpy().astype(np.float32).tobytes(order="C")
    if t.dtype == torch.int64:
        return t.numpy().astype(np.int64).tobytes(order="C")
    raise ValueError(f"Unsupported dtype {t.dtype}")

def _dtype_str(t: torch.Tensor) -> str:
    return {torch.float32: "float32", torch.int64: "int64"}[t.dtype]

# ----------------- edge filtering / remapping -----------------

def _build_old2new_from_keep(keep: torch.Tensor, n_app: int) -> torch.Tensor:
    """keep: int64 [N_keep] (original app indices) -> old2new map [-1 or new_id]."""
    assert keep.dtype == torch.int64
    old2new = torch.full((n_app,), -1, dtype=torch.int64, device=keep.device)
    old2new[keep] = torch.arange(keep.numel(), dtype=torch.int64, device=keep.device)
    return old2new

def _select_edges_by_app_src(ei: torch.Tensor, ew: torch.Tensor, old2new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep edges with SOURCE app in keep-set and remap src (row 0)."""
    if ei.numel() == 0:
        return ei, ew
    src_old = ei[0]
    src_new = old2new[src_old]                 # -1 means drop
    mask = src_new >= 0
    if mask.sum() == 0:
        return ei.new_empty((2, 0)), ew.new_empty((0,))
    ei2 = torch.stack([src_new[mask], ei[1, mask]], dim=0)
    ew2 = ew[mask]
    return ei2.contiguous(), ew2.contiguous()

def _select_edges_by_app_dst(ei: torch.Tensor, ew: torch.Tensor, old2new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep edges with DEST app in keep-set and remap dst (row 1)."""
    if ei.numel() == 0:
        return ei, ew
    dst_old = ei[1]
    dst_new = old2new[dst_old]
    mask = dst_new >= 0
    if mask.sum() == 0:
        return ei.new_empty((2, 0)), ew.new_empty((0,))
    ei2 = torch.stack([ei[0, mask], dst_new[mask]], dim=0)
    ew2 = ew[mask]
    return ei2.contiguous(), ew2.contiguous()

def _filter_edges_to_first_k(ei: torch.Tensor, ew: torch.Tensor, k: int, app_on_src: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """After remap to 0..Ntest-1, keep only edges whose app index < k on the app side."""
    if ei.numel() == 0:
        return ei, ew
    idx = ei[0] if app_on_src else ei[1]
    mask = idx < k
    if mask.sum() == 0:
        return ei.new_empty((2, 0)), ew.new_empty((0,))
    return ei[:, mask].contiguous(), ew[mask].contiguous()

# ----------------- writer -----------------

def _write_zip(
    path_zip: str,
    tensors: Dict[str, torch.Tensor],
    class_names: Optional[List[str]] = None,
    app_ids: Optional[List[str]] = None,
    y_app: Optional[torch.Tensor] = None,           # int64 [N_app_subset], values 1..K
    label_map: Optional[Dict[int, str]] = None,
    orig_app_idx: Optional[torch.Tensor] = None     # int64 [N_app_subset] -> original app row
):
    """Write tensors + manifest.json (+ optional y_app.bin, orig_app_idx.bin)."""
    names = [
        "x_app","x_sys","x_bnd","x_cmp",
        "ei_app_sys","ew_app_sys",
        "ei_sys_app","ew_sys_app",
        "ei_app_bnd","ew_app_bnd",
        "ei_bnd_app","ew_bnd_app",
        "ei_app_cmp","ew_app_cmp",
        "ei_cmp_app","ew_cmp_app",
    ]
    files = [f"{n}.bin" for n in names]
    arrays = [tensors[n] for n in names]

    manifest = {
        "tensors": [
            {"name": n, "dtype": _dtype_str(t), "shape": list(t.shape), "file": fn}
            for n, t, fn in zip(names, arrays, files)
        ],
        "class_names": class_names,
        "app_ids": app_ids,
    }
    if label_map is not None:
        manifest["label_map"] = {str(k): v for k, v in label_map.items()}

    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for n, t, fn in zip(names, arrays, files):
            z.writestr(fn, _to_le_bytes(t))
        if y_app is not None:
            if y_app.dtype != torch.int64:
                y_app = y_app.to(torch.int64)
            z.writestr("y_app.bin", _to_le_bytes(y_app))
        if orig_app_idx is not None:
            z.writestr("orig_app_idx.bin", _to_le_bytes(orig_app_idx.to(torch.int64)))

# ----------------- main API -----------------

def build_test_subsets_and_zip(
    # FULL graph inputs (torch tensors)
    x_app: torch.Tensor, x_sys: torch.Tensor, x_bnd: torch.Tensor, x_cmp: torch.Tensor,
    ei_app_sys: torch.Tensor, ew_app_sys: torch.Tensor,    # app -> sys
    ei_sys_app: torch.Tensor, ew_sys_app: torch.Tensor,    # sys -> app
    ei_app_bnd: torch.Tensor, ew_app_bnd: torch.Tensor,    # app -> binder
    ei_bnd_app: torch.Tensor, ew_bnd_app: torch.Tensor,    # binder -> app
    ei_app_cmp: torch.Tensor, ew_app_cmp: torch.Tensor,    # app -> composite
    ei_cmp_app: torch.Tensor, ew_cmp_app: torch.Tensor,    # composite -> app
    *,
    test_idx: torch.Tensor,                                # int64 indices into ORIGINAL app set
    base_out: str,                                         # path prefix, e.g. "out/data_test"
    class_names: Optional[List[str]] = None,
    app_ids: Optional[List[str]] = None,                   # length = original N_app
    y_app: Optional[torch.Tensor] = None,                  # int64 [N_app] labels 1..K
    label_map: Optional[Dict[int, str]] = None,
    shuffle_seed: int = 0,
    subset_sizes: Optional[List[int]] = None               # e.g., [32,16,8,1]; full set always written
):
    """
    Steps:
      - Filter FULL edges to the chosen apps and remap app indices to 0..Ntest-1.
      - Shuffle the test apps deterministically (seed).
      - Write one ZIP for the full shuffled test set.
      - Then write additional ZIPs for the first K apps (K in subset_sizes),
        filtering edges to indices < K (app side only).
    """
    assert test_idx.dtype == torch.int64
    n_app = x_app.size(0)

    # 1) Choose test order and reindex mapping
    order = test_idx.detach().clone()
    if shuffle_seed is not None:
        g = torch.Generator(device=order.device).manual_seed(shuffle_seed)
        perm = torch.randperm(order.numel(), generator=g, device=order.device)
        order = order.index_select(0, perm)
    old2new = _build_old2new_from_keep(order, n_app)  # [-1 or new_id], new_id in 0..Ntest-1

    # 2) Compact app features / labels / ids to "order"
    x_app_c = x_app.index_select(0, order)
    y_app_c = y_app.index_select(0, order) if y_app is not None else None
    app_ids_c = [app_ids[i.item()] for i in order.cpu()] if app_ids is not None else None
    orig_app_idx = order.clone()  # for traceability

    # 3) Filter and remap all six edge sets (app side only)
    e_app_sys, w_app_sys = _select_edges_by_app_src(ei_app_sys, ew_app_sys, old2new)
    e_sys_app, w_sys_app = _select_edges_by_app_dst(ei_sys_app, ew_sys_app, old2new)
    e_app_bnd, w_app_bnd = _select_edges_by_app_src(ei_app_bnd, ew_app_bnd, old2new)
    e_bnd_app, w_bnd_app = _select_edges_by_app_dst(ei_bnd_app, ew_bnd_app, old2new)
    e_app_cmp, w_app_cmp = _select_edges_by_app_src(ei_app_cmp, ew_app_cmp, old2new)
    e_cmp_app, w_cmp_app = _select_edges_by_app_dst(ei_cmp_app, ew_cmp_app, old2new)

    # Pack tensors for the full shuffled test set
    tensors_all = {
        "x_app": x_app_c, "x_sys": x_sys, "x_bnd": x_bnd, "x_cmp": x_cmp,
        "ei_app_sys": e_app_sys, "ew_app_sys": w_app_sys,
        "ei_sys_app": e_sys_app, "ew_sys_app": w_sys_app,
        "ei_app_bnd": e_app_bnd, "ew_app_bnd": w_app_bnd,
        "ei_bnd_app": e_bnd_app, "ew_bnd_app": w_bnd_app,
        "ei_app_cmp": e_app_cmp, "ew_app_cmp": w_app_cmp,
        "ei_cmp_app": e_cmp_app, "ew_cmp_app": w_cmp_app,
    }

    # 4) Write full shuffled test set
    path_full = f"{base_out}_all.zip"
    _write_zip(path_full, tensors_all, class_names, app_ids_c, y_app_c, label_map, orig_app_idx)
    print("Wrote", path_full)

    # 5) Additional subset sizes
    ntest = x_app_c.size(0)
    sizes = subset_sizes or []
    # always include a unique, valid set of sizes <= ntest, descending order (optional)
    sizes = [k for k in sizes if isinstance(k, int) and 1 <= k <= ntest]
    seen = set()
    sizes = [k for k in sizes if (k not in seen and not seen.add(k))]

    for k in sizes:
        # slice x_app/y/app_ids
        x_app_k = x_app_c[:k]
        y_app_k = y_app_c[:k] if y_app_c is not None else None
        app_ids_k = app_ids_c[:k] if app_ids_c is not None else None
        orig_idx_k = orig_app_idx[:k]

        # filter edges to indices < k (app side only)
        e_app_sys_k, w_app_sys_k = _filter_edges_to_first_k(e_app_sys, w_app_sys, k, app_on_src=True)
        e_sys_app_k, w_sys_app_k = _filter_edges_to_first_k(e_sys_app, w_sys_app, k, app_on_src=False)
        e_app_bnd_k, w_app_bnd_k = _filter_edges_to_first_k(e_app_bnd, w_app_bnd, k, app_on_src=True)
        e_bnd_app_k, w_bnd_app_k = _filter_edges_to_first_k(e_bnd_app, w_bnd_app, k, app_on_src=False)
        e_app_cmp_k, w_app_cmp_k = _filter_edges_to_first_k(e_app_cmp, w_app_cmp, k, app_on_src=True)
        e_cmp_app_k, w_cmp_app_k = _filter_edges_to_first_k(e_cmp_app, w_cmp_app, k, app_on_src=False)

        tensors_k = {
            "x_app": x_app_k, "x_sys": x_sys, "x_bnd": x_bnd, "x_cmp": x_cmp,
            "ei_app_sys": e_app_sys_k, "ew_app_sys": w_app_sys_k,
            "ei_sys_app": e_sys_app_k, "ew_sys_app": w_sys_app_k,
            "ei_app_bnd": e_app_bnd_k, "ew_app_bnd": w_app_bnd_k,
            "ei_bnd_app": e_bnd_app_k, "ew_bnd_app": w_bnd_app_k,
            "ei_app_cmp": e_app_cmp_k, "ew_app_cmp": w_app_cmp_k,
            "ei_cmp_app": e_cmp_app_k, "ew_cmp_app": w_cmp_app_k,
        }

        path_k = f"{base_out}_{k}.zip"
        _write_zip(path_k, tensors_k, class_names, app_ids_k, y_app_k, label_map, orig_idx_k)
        print("Wrote", path_k)

# ----------------- demo (remove in production) -----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # toy full graph
    N_app, N_sys, N_bnd, N_cmp = 100, 400, 60, 80
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

    # FULL edges
    ei_app_sys, ew_app_sys = rnd_e(N_app, N_sys, 2000)
    ei_sys_app, ew_sys_app = rnd_e(N_sys, N_app, 2000)
    ei_app_bnd, ew_app_bnd = rnd_e(N_app, N_bnd,  900)
    ei_bnd_app, ew_bnd_app = rnd_e(N_bnd, N_app,  900)
    ei_app_cmp, ew_app_cmp = rnd_e(N_app, N_cmp, 1600)
    ei_cmp_app, ew_cmp_app = rnd_e(N_cmp, N_app, 1600)

    y_app = torch.randint(1, 6, (N_app,), dtype=torch.int64)
    app_ids = [f"pkg.{i}" for i in range(N_app)]
    label_map = {1:"adware", 2:"banking", 3:"sms", 4:"riskware", 5:"benign"}

    # choose arbitrary test set (no shuffle here; shuffling is inside the function)
    test_idx = torch.tensor(sorted(np.random.choice(N_app, size=48, replace=False)), dtype=torch.int64)

    build_test_subsets_and_zip(
        x_app, x_sys, x_bnd, x_cmp,
        ei_app_sys, ew_app_sys, ei_sys_app, ew_sys_app,
        ei_app_bnd, ew_app_bnd, ei_bnd_app, ew_bnd_app,
        ei_app_cmp, ew_app_cmp, ei_cmp_app, ew_cmp_app,
        test_idx=test_idx,
        base_out="data_test",
        class_names=["adware","banking","sms","riskware","benign"],
        app_ids=app_ids,
        y_app=y_app,
        label_map=label_map,
        shuffle_seed=42,
        subset_sizes=[32, 16, 8, 1],
    )
