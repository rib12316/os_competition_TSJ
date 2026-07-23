#!/usr/bin/env python3
"""Qwen2.5 C8 自校准（post-RoPE K + v_proj V）—— calibrate_c8.py 的 Qwen2 版。

Qwen2 与 Qwen3 的差异：无 q_norm/k_norm；rotary 走 transformers.models.qwen2.modeling_qwen2。
校准原理不变：cache 存 post-RoPE K，所以 K 必须 hook post-RoPE（monkeypatch
apply_rotary_pos_emb），V 无 RoPE 走 v_proj。scale = maxabs/127。需 NPU。
Qwen2.5-7B：num_kv_heads=4, head_dim=128 → ch=512。
"""
from __future__ import annotations
import sys, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen2.modeling_qwen2 as q2m

sys.path.insert(0, "/tmp/f1-c8-wt/agent-mem/src")
from agent_mem.kv.c8 import annotate_model  # noqa: E402

MODEL = Path("/data/os_competition_TSJ/models/Qwen2.5-7B-Instruct")
dev = "npu" if torch.npu.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(str(MODEL))
m = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16).to(dev).eval()
NK, HS, NL = (
    m.config.num_key_value_heads,
    getattr(m.config, "head_dim", None) or m.config.hidden_size // m.config.num_attention_heads,
    m.config.num_hidden_layers,
)
CH = NK * HS
print(f"[model] {m.config.model_type} layers={NL} kv_heads={NK} head_dim={HS} ch={CH} dev={dev}")

k_max = {i: torch.zeros(NK, HS) for i in range(NL)}
v_max = {i: torch.zeros(NK, HS) for i in range(NL)}


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
        def hv(_m, _i, out):                       # v_proj → V（无 RoPE）
            cur = to_hsd(out).abs().amax(0)
            v_max[idx] = torch.maximum(v_max[idx], cur.cpu())
        return hv
    layer.self_attn.v_proj.register_forward_hook(mk(i))

_ctr = {"i": 0}
_orig = q2m.apply_rotary_pos_emb


def _reset(_model, _inp):
    _ctr["i"] = 0


def patched_arpe(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig(q, k, cos, sin, *a, **kw)       # k2 = post-RoPE K
    li = _ctr["i"] % NL
    _ctr["i"] += 1
    cur = to_hsd(k2).abs().amax(0)
    k_max[li] = torch.maximum(k_max[li], cur.cpu())
    return q2, k2


q2m.apply_rotary_pos_emb = patched_arpe
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

from safetensors.torch import save_file, load_file
scales = {}
for i in range(NL):
    scales[f"model.layers.{i}.self_attn.k_proj.kv_cache_scale"] = (k_max[i] / 127.0).clamp(min=1e-8).reshape(CH).to(torch.float32)
    scales[f"model.layers.{i}.self_attn.v_proj.kv_cache_scale"] = (v_max[i] / 127.0).clamp(min=1e-8).reshape(CH).to(torch.float32)

annotate_model(MODEL, overwrite=True)
sf = MODEL / "model.safetensors"
bak = MODEL / "model.safetensors.stock.bak"
if sf.exists():
    if not bak.exists():
        shutil.copy2(sf, bak)
    allw = load_file(str(sf)); allw.update(scales); save_file(allw, str(sf))
    print(f"[inject] single-file: merged {len(scales)} scales (ch={CH})")
else:
    # sharded
    SCALES_FILE = "kv_cache_scales.safetensors"
    save_file(scales, str(MODEL / SCALES_FILE))
    idx = MODEL / "model.safetensors.index.json"
    idx_bak = MODEL / "model.safetensors.index.json.stock.bak"
    if not idx_bak.exists():
        shutil.copy2(idx, idx_bak)
    d = json.loads(idx.read_text()) if False else __import__("json").loads(idx.read_text())
    for kk in scales:
        d["weight_map"][kk] = SCALES_FILE
    idx.write_text(__import__("json").dumps(d, indent=2))
    print(f"[inject] sharded: wrote {SCALES_FILE} + updated index")

for i in (0, NL // 2, NL - 1):
    print(f"  L{i:2d} k_scale[{k_max[i].min()/127:.2e},{k_max[i].max()/127:.2e}]  v_scale[{v_max[i].min()/127:.2e},{v_max[i].max()/127:.2e}]")
print("[done] 起 Qwen2.5 C8 引擎: python scripts/serve_c8_qwen2.py --model ... --quantization ascend --compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}'")
