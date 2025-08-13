# quantize_and_export.py
import os, torch

# PT2E quantization API (torchao >= 0.3); fall back to torch.ao on older installs
try:
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
except Exception:
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e  # older path

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

# ---- helpers ----
def param_stats(m: torch.nn.Module):
    n_params = sum(p.numel() for p in m.parameters())
    bytes_ = sum(p.numel() * p.element_size() for p in m.parameters())
    return n_params, bytes_ / (1024**2)  # (count, MB)

def save_pte(ep, out_path: str):
    et_prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    with open(out_path, "wb") as f:
        et_prog.write_to_file(f)
    return os.path.getsize(out_path) / (1024**2)  # MB

# ---- your model & inputs ----
# Replace with your hetero-GNN module and a tuple of representative Tensors
model = ...                 # torch.nn.Module, eval() ready
sample_inputs = (...)       # e.g., (x_app, x_sys, x_bnd, x_cmp, ei_app_sys, ew_app_sys, ...)

model.eval()

# (A) float32 export (optional, for baseline size comparison)
ep_float = torch.export.export(model, sample_inputs)   # torch 2.x export
pte_float = "model_fp32_xnnpack.pte"
pte_float_mb = save_pte(ep_float, pte_float)

n_float, mb_float = param_stats(model)
print(f"[FP32] params={n_float:,}  param_mem≈{mb_float:.2f} MB  .pte={pte_float_mb:.2f} MB")

# (B) PT2E static 8-bit quantization for XNNPACK (weights int8; activations int8 with calibration)
qparams = get_symmetric_quantization_config(is_per_channel=True)   # per-channel weights
quantizer = XNNPACKQuantizer()
quantizer.set_global(qparams)

# 1) capture for training graph, as required by PT2E
training_ep = torch.export.export_for_training(model, sample_inputs).module()  # ExecuTorch docs use .module() 
prepared = prepare_pt2e(training_ep, quantizer)                                 # insert observers/annotations 

# 2) calibration: run representative inputs through 'prepared'
#    Do this over a small calibration set that reflects production data.
def calibrate(prepared_module, reps):
    prepared_module.eval()
    with torch.inference_mode():
        for ins in reps:   # 'reps' yields tuples shaped like sample_inputs
            prepared_module(*ins)

# Example: single-batch calibration with the same shapes (replace with a real loader)
calibrate(prepared, [sample_inputs])

# 3) convert to a quantized model graph (still a PyTorch module)
qmodel = convert_pt2e(prepared)                                                  # 
qmodel.eval()

# 4) export the quantized model and lower to ExecuTorch/XNNPACK
ep_q = torch.export.export(qmodel, sample_inputs)
pte_int8 = "model_int8_xnnpack.pte"
pte_int8_mb = save_pte(ep_q, pte_int8)

# 5) print sizes
n_q, mb_q = param_stats(qmodel)
print(f"[INT8] params={n_q:,}  param_mem≈{mb_q:.2f} MB  .pte={pte_int8_mb:.2f} MB")

# Optional sanity check: run both models in eager to compare logits numerically
with torch.inference_mode():
    y_fp32 = model(*sample_inputs)
    y_int8 = qmodel(*sample_inputs)
    # print a quick cosine similarity if they are tensors
    if torch.is_tensor(y_fp32) and torch.is_tensor(y_int8):
        cs = torch.nn.functional.cosine_similarity(y_fp32.flatten(), y_int8.flatten(), dim=0).item()
        print(f"cosine_sim(fp32, int8) = {cs:.4f}")
