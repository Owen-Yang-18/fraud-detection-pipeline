# export_to_zip.py
import json, zipfile, io, numpy as np, torch

# tensors in *this* exact order:
# x_app, x_sys, x_bnd, x_cmp,
# ei_app_sys, ew_app_sys,
# ei_sys_app, ew_sys_app,
# ei_app_bnd, ew_app_bnd,
# ei_bnd_app, ew_bnd_app,
# ei_app_cmp, ew_app_cmp,
# ei_cmp_app, ew_cmp_app,
def to_le_bytes(t: torch.Tensor):
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.float32:
        return t.numpy().astype(np.float32).tobytes(order="C")
    if t.dtype == torch.int64:
        return t.numpy().astype(np.int64).tobytes(order="C")
    raise ValueError(f"Unsupported dtype {t.dtype}")

def dtype_str(t: torch.Tensor):
    return {torch.float32: "float32", torch.int64: "int64"}[t.dtype]

def export_zip(path_zip, tensors, class_names=None, app_ids=None):
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
    manifest = {
        "tensors":[
            {"name":n, "dtype":dtype_str(t), "shape":list(t.shape), "file":fn}
            for n,t,fn in zip(names, tensors, files)
        ],
        "class_names": class_names,
        "app_ids": app_ids,
    }
    with zipfile.ZipFile(path_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for n,t,fn in zip(names, tensors, files):
            z.writestr(fn, to_le_bytes(t))
    print("Wrote", path_zip)

# Example (replace these with your *real* tensors)
if __name__ == "__main__":
    # toy shapes:
    x_app = torch.arange(100, dtype=torch.float32).view(-1,1)
    x_sys = torch.arange(400, dtype=torch.float32).view(-1,1)
    x_bnd = torch.arange(50, dtype=torch.float32).view(-1,1)
    x_cmp = torch.arange(80, dtype=torch.float32).view(-1,1)

    def rnd_e(n_src, n_dst, e):
        src = torch.randint(n_src, (e,), dtype=torch.int64)
        dst = torch.randint(n_dst, (e,), dtype=torch.int64)
        return torch.stack([src, dst], dim=0), torch.ones(e, dtype=torch.float32)

    ei_app_sys, ew_app_sys = rnd_e(100, 400, 1200)
    ei_sys_app, ew_sys_app = rnd_e(400, 100, 1200)
    ei_app_bnd, ew_app_bnd = rnd_e(100, 50, 600)
    ei_bnd_app, ew_bnd_app = rnd_e(50, 100, 600)
    ei_app_cmp, ew_app_cmp = rnd_e(100, 80, 1200)
    ei_cmp_app, ew_cmp_app = rnd_e(80, 100, 1200)

    tensors = [x_app,x_sys,x_bnd,x_cmp,
               ei_app_sys,ew_app_sys, ei_sys_app,ew_sys_app,
               ei_app_bnd,ew_app_bnd, ei_bnd_app,ew_bnd_app,
               ei_app_cmp,ew_app_cmp, ei_cmp_app,ew_cmp_app]

    class_names = ["benign","class1","class2","class3","class4"]
    app_ids = [f"app_{i}" for i in range(x_app.size(0))]

    export_zip("data.zip", tensors, class_names, app_ids)
