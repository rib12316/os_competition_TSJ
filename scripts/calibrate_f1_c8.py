#!/usr/bin/env python3
"""F1 C8 自校准 MinMax（P1，**需 NPU**）——产真 per-channel scale 替代 placeholder 0.05。

假说：placeholder scale=0.05 的垃圾数值在 graph 捕获期 forward 产生病态→挂；真 scale 或可解挂。
用法（NPU 已启动）：
  PYTHONPATH=agent-mem/src:$PYTHONPATH .venv/bin/python scripts/calibrate_f1_c8.py --model models/Qwen3-0.6B
产出：模型被 annotate + 注入真 scale。随后：
  PYTHONPATH=agent-mem/src:$PYTHONPATH .venv/bin/python scripts/probe_f1_c8.py --model models/Qwen3-0.6B --skip-prep
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import probe_f1_c8 as P  # 复用 annotate / _cfg / _chans / 常量

# tau-bench 风格多轮 agent 文本（覆盖常见 token 分布）
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


def calibrate(model_dir: Path, prompts: list[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = P._cfg(model_dir)
    ch, nh, hs = P._chans(cfg)
    n = int(cfg["num_hidden_layers"])
    print(f"[calib] layers={n} kv_heads={nh} head_size={hs} ch={ch}")

    dev = "npu" if torch.npu.is_available() else "cpu"
    print(f"[calib] device={dev}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    maxabs = {i: {"k": torch.zeros(ch), "v": torch.zeros(ch)} for i in range(n)}

    def hook(i, kv):
        def _h(_m, _x, out):
            x = out.detach().float().reshape(-1, ch)
            cur = x.abs().amax(dim=0).cpu()
            maxabs[i][kv] = torch.maximum(maxabs[i][kv], cur)
        return _h

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook(i, "k")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook(i, "v")))

    with torch.no_grad():
        for j, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").to(dev)
            try:
                model(**ids, use_cache=False)
            except Exception as e:
                print(f"[calib] skip sample {j}: {e}")
            if (j + 1) % 8 == 0:
                print(f"[calib] {j+1}/{len(prompts)} done")
    for h in handles:
        h.remove()
    del model
    return maxabs, ch, n


def inject_real(model_dir: Path, maxabs, ch, n):
    import json
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    scales = {}
    for i in range(n):
        for kv in ("k", "v"):
            v = maxabs[i][kv]
            scales[f"model.layers.{i}.self_attn.{kv}_proj.kv_cache_scale"] = (v / 127.0).clamp(min=1e-8).to(torch.float32)

    idx = model_dir / P.INDEX
    if idx.exists():
        save_file(scales, str(model_dir / P.SCALES_FILE))
        bak = model_dir / P.INDEX_BAK
        if not bak.exists():
            shutil.copy2(idx, bak)
        d = json.loads(idx.read_text())
        for k in scales:
            d["weight_map"][k] = P.SCALES_FILE
        idx.write_text(json.dumps(d, indent=2) + "\n")
        print(f"[calib] sharded: wrote {P.SCALES_FILE} + index ({len(scales)} scales)")
    else:
        sf = model_dir / "model.safetensors"
        if not (model_dir / P.SF_BAK).exists():
            shutil.copy2(sf, model_dir / P.SF_BAK)
        allw = load_file(str(sf))
        allw.update(scales)
        save_file(allw, str(sf))
        print(f"[calib] single-file: merged {len(scales)} scales (ch={ch})")

    for i in (0, n // 2, n - 1):
        for kv in ("k", "v"):
            v = maxabs[i][kv]
            print(f"  L{i:2d} {kv}: max_abs[{v.min():.3f},{v.max():.3f}] scale[{(v/127).min():.2e},{(v/127).max():.2e}]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--num-samples", type=int, default=32)
    args = ap.parse_args()
    m = Path(args.model)

    P.annotate(m)
    prompts = (_CALIB * ((args.num_samples // len(_CALIB)) + 1))[: args.num_samples]
    maxabs, ch, n = calibrate(m, prompts)
    inject_real(m, maxabs, ch, n)
    print("[calib] done → 跑探针: python scripts/probe_f1_c8.py --model", args.model, "--skip-prep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
