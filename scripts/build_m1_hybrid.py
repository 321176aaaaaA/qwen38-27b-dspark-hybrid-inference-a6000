#!/usr/bin/env python3
"""Build model_m1_hybrid: Q4 checkpoint with all 64 down_proj upgraded to 8bit.

Strategy: hardlink-copy model_mtp_opt, requant down_proj from model_bf16
to int8 g128 (replacing 4bit triples), insert config_group group_d8
(targets re:.*down_proj$, num_bits=8) FIRST in config_groups dict order
(vLLM find_matched_target: first match in dict order wins).
"""
import glob, json, math, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/opt/qwen38_27b/model_mtp_opt"
BF16 = "/opt/qwen38_27b/model_bf16"
DST = "/opt/qwen38_27b/model_m1_hybrid"
BITS, GROUP, QMAX = 8, 128, 127

def pack_to_int32(value, num_bits):
    offset = 1 << (num_bits - 1)
    value = (value + offset).to(torch.uint8)
    pf = 32 // num_bits
    rows, cols = value.shape
    pad = math.ceil(cols / pf) * pf - cols
    if pad:
        value = torch.nn.functional.pad(value, (0, pad))
    r = value.view(rows, (cols + pad) // pf, pf).to(torch.int32)
    shifts = torch.arange(pf, dtype=torch.int32) * num_bits
    return (r << shifts).sum(dim=2, dtype=torch.int32)

def quant8(w):
    w = w.to(torch.float32)
    out_f, in_f = w.shape
    assert in_f % GROUP == 0, w.shape
    g = w.reshape(out_f, in_f // GROUP, GROUP)
    scale = torch.clamp(g.abs().amax(dim=-1, keepdim=True) / QMAX, min=1e-10)
    q = torch.clamp(torch.round(g / scale), -QMAX - 1, QMAX).to(torch.int8).reshape(out_f, in_f)
    deq = (q.reshape(out_f, -1, GROUP).float() * scale).reshape(out_f, in_f)
    err = ((deq - w).norm() / w.norm()).item()
    return pack_to_int32(q, BITS).contiguous(), scale.squeeze(-1).half().contiguous(), \
           torch.tensor([out_f, in_f], dtype=torch.int64), err

# 1. hardlink copy
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    s, d = os.path.join(SRC, f), os.path.join(DST, f)
    if not os.path.isfile(s):
        continue  # skip dirs like .cache
    if os.path.exists(d):
        os.remove(d)
    os.link(s, d)  # hardlink; rewritten shards will be replaced (new inode)
print("hardlinked", flush=True)

# 2. locate down_proj 4bit triples and bf16 sources
idx = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
wm = idx["weight_map"]
bidx = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]
bases = sorted(t[:-len(".weight_packed")] for t in wm
               if t.endswith(".down_proj.weight_packed") and not t.startswith("mtp."))
print("down_proj count:", len(bases), flush=True)
assert len(bases) == 64

by_dst_shard = {}
del_set = {}
for b in bases:
    for suf in ("weight_packed", "weight_scale", "weight_shape"):
        del_set.setdefault(wm[b + "." + suf], set()).add(b + "." + suf)
    by_dst_shard.setdefault(wm[b + ".weight_packed"], []).append(b)

# 3. rewrite each dst shard: drop 4bit triple, add 8bit triple (from bf16)
bf16_cache = {}
def read_bf16(name):
    f = bidx[name]
    if f not in bf16_cache:
        bf16_cache[f] = safe_open(os.path.join(BF16, f), framework="pt")
    return bf16_cache[f].get_tensor(name)

# quantize all 64 first (single bf16 read each)
quant_results = {}
for b in bases:
    w = read_bf16(b + ".weight")
    packed, scale, shape, err = quant8(w)
    quant_results[b] = (packed, scale, shape)
    print(f"  {b}: err={err:.3e}", flush=True)

shards = sorted(set(by_dst_shard) | set(del_set))
for shard in shards:
    path = os.path.join(DST, shard)
    with safe_open(path, framework="pt") as h:
        tensors = {k: h.get_tensor(k) for k in h.keys()}
    for k in del_set.get(shard, ()):
        del tensors[k]
    for b in by_dst_shard.get(shard, ()):
        packed, scale, shape = quant_results[b]
        tensors[b + ".weight_packed"] = packed
        tensors[b + ".weight_scale"] = scale
        tensors[b + ".weight_shape"] = shape
        for suf in ("weight_packed", "weight_scale", "weight_shape"):
            wm[b + "." + suf] = shard
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, path)
    print(f"rewrote {shard}", flush=True)

for fname in ("model.safetensors.index.json", "config.json"):
    p = os.path.join(DST, fname)
    if os.path.exists(p):
        os.remove(p)  # break hardlink before writing
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)

# 4. config.json: insert group_d8 first
cfg = json.load(open(os.path.join(SRC, "config.json")))
qc = cfg["quantization_config"]
g0 = qc["config_groups"]["group_0"]
import copy
gd8 = copy.deepcopy(g0)
gd8["targets"] = ["re:.*down_proj$"]
gd8["weights"]["num_bits"] = 8
new_groups = {"group_d8": gd8}
new_groups.update(qc["config_groups"])
qc["config_groups"] = new_groups
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)  # hardlink already broken above
print("config updated; groups order:", list(new_groups), flush=True)
print("M1_DONE", flush=True)
