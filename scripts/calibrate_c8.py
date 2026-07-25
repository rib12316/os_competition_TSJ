#!/usr/bin/env python3
"""修正版 C8 自校准：hook **post-RoPE** K（= 真·会被 cache 的）+ v_proj V。

旧 calibrate_f1_c8.py 的 bug：hook k_proj（pre-RoPE），但 cache 存 post-RoPE K；
RoPE+norm 把早期层 K 放大上百倍 → pre-RoPE scale 严重饱和 → 反量化乱码。
（verify_rope.py 实测：L0 post/pre=435×, pre-scale roundtrip err=84.5%, sat=68%；
 post-scale err≈0.5%。）

修法：post-RoPE K 用 monkeypatch apply_rotary_pos_emb 抓；V 无 RoPE，v_proj 直采。
产出：模型 annotate + 注入正确 scale。NPU 需启动。
"""
from __future__ import annotations
import sys, json, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as q3m

sys.path.insert(0, "/tmp/f1-c8-wt/agent-mem/src")
from agent_mem.kv.c8 import annotate_model  # noqa: E402

MODEL = Path("/data/os_competition_TSJ/models/Qwen3-0.6B")
dev = "npu" if torch.npu.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(str(MODEL))
m = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to(dev).eval()
NK, HS, NL = m.config.num_key_value_heads, m.config.head_dim, m.config.num_hidden_layers
CH = NK * HS
print(f"[model] layers={NL} kv_heads={NK} head_dim={HS} ch={CH} dev={dev}")

k_max = {i: torch.zeros(NK, HS) for i in range(NL)}   # post-RoPE K per-(head,dim) maxabs
v_max = {i: torch.zeros(NK, HS) for i in range(NL)}    # V (v_proj) per-(head,dim) maxabs


def to_hsd(t):
    t = t.detach().float()
    if t.dim() == 3:
        B, S, _ = t.shape
        return t.reshape(B, S, NK, HS).reshape(B * S, NK, HS)
    if t.dim() == 4:
        B, a, b, c = t.shape
        if a == NK and b != NK:
            return t.permute(0, 2, 1, 3).reshape(B * b, NK, HS)
        return t.reshape(-1, NK, HS)
    raise ValueError(t.shape)


for i, layer in enumerate(m.model.layers):
    def mk(idx):
        def hv(_m, _i, out):                       # v_proj → V（无 RoPE，直接采）
            cur = to_hsd(out).abs().amax(0)
            v_max[idx] = torch.maximum(v_max[idx], cur.cpu())
        return hv
    layer.self_attn.v_proj.register_forward_hook(mk(i))

_ctr = {"i": 0}
_orig = q3m.apply_rotary_pos_emb


def _reset(_model, _inp):
    _ctr["i"] = 0


def patched_arpe(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig(q, k, cos, sin, *a, **kw)       # k2 = post-RoPE K
    li = _ctr["i"] % NL
    _ctr["i"] += 1
    cur = to_hsd(k2).abs().amax(0)
    k_max[li] = torch.maximum(k_max[li], cur.cpu())
    return q2, k2


q3m.apply_rotary_pos_emb = patched_arpe
m.register_forward_pre_hook(_reset)

_CALIB = [
    "You are a helpful retail assistant. A customer wants to return a shirt bought last week with a receipt. Walk them through the return policy and process.",
    "Summarize the key steps for processing a refund in an e-commerce system, including order lookup, condition check, and payment reversal.",
    "As an airline agent, rebook a passenger whose flight was cancelled. Offer two alternatives and confirm seat preferences.",
    "List the tools available to a shopping agent: search_products, get_order_status, initiate_return, update_address. Describe when to use each.",
    "Explain in plain language: items can be returned within 30 days with a receipt for a full refund, or within 60 days for store credit.",
    "A user asks about their order #12345 status. Draft a polite response and indicate you will check the tracking system.",
    "Compare two laptops for a customer focused on battery life and weight, then recommend one with justification.",
    "Write a short follow-up message to a customer after resolving their billing dispute, confirming the credit was applied.",
]
N = 32
prompts = (_CALIB * ((N // len(_CALIB)) + 1))[:N]
for j, p in enumerate(prompts):
    ids = tok(p, return_tensors="pt").to(dev)
    with torch.no_grad():
        m(**ids, use_cache=False)
    if (j + 1) % 8 == 0:
        print(f"[calib] {j+1}/{N} done")

# 注入正确 scale（post-RoPE K, v_proj V）
from safetensors.torch import save_file, load_file
scales = {}
for i in range(NL):
    scales[f"model.layers.{i}.self_attn.k_proj.kv_cache_scale"] = (k_max[i] / 127.0).clamp(min=1e-8).reshape(CH).to(torch.float32)
    scales[f"model.layers.{i}.self_attn.v_proj.kv_cache_scale"] = (v_max[i] / 127.0).clamp(min=1e-8).reshape(CH).to(torch.float32)

annotate_model(MODEL, overwrite=True)
sf = MODEL / "model.safetensors"
if not (MODEL / "model.safetensors.stock.bak").exists():
    shutil.copy2(sf, MODEL / "model.safetensors.stock.bak")
allw = load_file(str(sf))
allw.update(scales)
save_file(allw, str(sf))
print(f"[inject] merged {len(scales)} scales into model.safetensors (post-RoPE K + V)")
for i in (0, NL // 2, NL - 1):
    print(f"  L{i:2d} k_scale[{k_max[i].min()/127:.2e},{k_max[i].max()/127:.2e}]  v_scale[{v_max[i].min()/127:.2e},{v_max[i].max()/127:.2e}]")
print("[done] 起 C8 引擎重测：FULL_DECODE_ONLY + 正确 scale")
