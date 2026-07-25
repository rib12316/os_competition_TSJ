#!/usr/bin/env python3
"""验证：C8 校准 hook pre-RoPE 的 k_proj，但 cache 存 post-RoPE K → scale 对不上？

直接测：拿真·post-RoPE K（会被 cache 的），分别用 pre-RoPE maxabs scale（当前校准）
和 post-RoPE maxabs scale（oracle）做 quant→dequant roundtrip，比相对误差 + 饱和率。
post-RoPE K 通过 monkeypatch apply_rotary_pos_emb 捕获（按 layer 顺序）。
"""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as q3m

MODEL = "/data/os_competition_TSJ/models/Qwen3-0.6B"
dev = "npu" if torch.npu.is_available() else "cpu"
print(f"[dev] {dev}")

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
NK = m.config.num_key_value_heads   # 8
HS = m.config.head_dim              # 128
NL = m.config.num_hidden_layers     # 28
print(f"[model] layers={NL} kv_heads={NK} head_dim={HS}")

pre_max, post_max, post_samp = {}, {}, {}


def to_nhsd(t):
    t = t.detach().float()
    if t.dim() == 3:                          # (B,S,NK*HS)
        B, S, _ = t.shape
        return t.reshape(B, S, NK, HS).reshape(B * S, NK, HS)
    if t.dim() == 4:
        B, a, b, c = t.shape
        if a == NK and b != NK:               # (B,NK,S,HS)
            return t.permute(0, 2, 1, 3).reshape(B * b, NK, HS)
        return t.reshape(-1, NK, HS)
    raise ValueError(t.shape)


for i, layer in enumerate(m.model.layers):
    def mk(idx):
        def hk(_m, _i, out):                   # k_proj 输出 = pre-norm, pre-RoPE（当前校准采的）
            cur = to_nhsd(out).abs().amax(0)
            pre_max[idx] = torch.maximum(pre_max[idx], cur) if idx in pre_max else cur
        return hk
    layer.self_attn.k_proj.register_forward_hook(mk(i))

_ctr = {"i": 0}
_orig = q3m.apply_rotary_pos_emb


def _reset(_model, _inp):
    _ctr["i"] = 0


def patched_arpe(q, k, cos, sin, *a, **kw):
    q2, k2 = _orig(q, k, cos, sin, *a, **kw)   # k2 = post-RoPE K = 真·会被 cache 的
    li = _ctr["i"] % NL
    _ctr["i"] += 1
    n = to_nhsd(k2)
    cur = n.abs().amax(0)
    post_max[li] = torch.maximum(post_max[li], cur) if li in post_max else cur
    if li not in post_samp:
        post_samp[li] = n.clone()
    return q2, k2


q3m.apply_rotary_pos_emb = patched_arpe
m.register_forward_pre_hook(_reset)

PROMPTS = [
    "You are a helpful retail assistant. Walk a customer through returning a shirt with a receipt.",
    "Summarize the steps to process a refund: order lookup, condition check, payment reversal.",
    "As an airline agent, rebook a passenger whose flight was cancelled; offer two alternatives.",
    "List the tools a shopping agent has and describe when to use each.",
    "Explain: items can be returned within 30 days with a receipt for a full refund.",
    "Compare two laptops for a customer focused on battery life and weight, then recommend one.",
]
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").to(dev)
    with torch.no_grad():
        m(**ids, use_cache=False)
print(f"[hooks] collected pre/post maxabs over {len(PROMPTS)} prompts, layers seen: {sorted(post_max)}")


def roundtrip(K, scale):
    inv = 1.0 / scale.clamp(min=1e-8)
    q = torch.clamp(torch.round(K * inv), -128, 127)
    recon = q * scale
    rel = (recon - K).abs().mean() / K.abs().mean().clamp(min=1e-8)
    sat = (K * inv).abs().gt(127).float().mean()
    return rel.item(), sat.item()


print("\n=== quant→dequant roundtrip on REAL post-RoPE K (what gets cached) ===")
print(f"{'L':>3} {'post/pre maxabs':>16} {'pre-scale err':>14} {'pre-scale sat':>14} {'post-scale err':>15}")
worst_pre = 0.0
for li in [0, 7, 14, 21, 27]:
    spre = pre_max[li] / 127
    spost = post_max[li] / 127
    ratio = (post_max[li] / pre_max[li].clamp(min=1e-8)).amax().item()
    K = post_samp[li]
    rpre, spresat = roundtrip(K, spre)
    rpost, _ = roundtrip(K, spost)
    worst_pre = max(worst_pre, rpre)
    print(f"{li:>3} {ratio:>16.2f} {rpre:>14.4f} {spresat:>13.2%} {rpost:>15.4f}")

print(f"\nworst pre-scale roundtrip rel-err = {worst_pre:.4f}")
print("pre err 大 & 高饱和 → RoPE 校准 bug（我方 hook pre-RoPE，cache 是 post-RoPE），可修")
print("pre/post 都大       → per-channel MinMax scheme 本身有问题")
print("pre err 小          → RoPE 不是元凶，另查")
