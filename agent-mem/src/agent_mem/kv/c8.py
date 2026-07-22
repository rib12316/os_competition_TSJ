"""缝A · F1 C8 int8 KV 量化 — 模型 quant_description 注入工具。

vllm-ascend 的真 int8 KV 叫 **C8**，激活**不**靠 ``--kv-cache-dtype int8``
（在 0.22.1rc1 已证为 no-op——后端 ``attention_v1.py:1309`` 硬断言 ``scales==1.0``，
全仓无 int8→C8 路由）。正确激活靠：

1. 模型目录写一份 ``quant_model_description.json``（``"kv_cache_type":"C8"`` +
   每层 ``k/v_proj.kv_cache_scale`` 标记），触发
   ``AscendC8KVCacheAttentionMethod``（上游 ``vllm_ascend/quantization/methods/kv_c8.py``）；
2. 引擎加 ``--quantization ascend``。

本模块把"标注模型 → 注入 scale → 还原 stock"做成**可测纯函数 + CLI**，无需 NPU。
激活机制与上游 PR `#7474 <https://github.com/vllm-project/vllm-ascend/pull/7474>`_
一致；上游调研见 ``docs/F1-c8-upstream-survey.md``。

C8 需要 **per-channel scale**（``num_kv_heads*head_size`` 维），默认 ``ones(1)``
会在 ``_prepare_c8_scales`` 的 ``.view(h,1,d)`` 崩 → 探针阶段用占位 scale
（:func:`inject_placeholder_scales`，精度垃圾但能验证路径）；真 scale 由自校准产出
（见 ``feat/f1-c8-ascend-investigation`` 的 ``scripts/calibrate_f1_c8.py``）。

⚠️ 910B 真机坑：C8 decode 在 **FULL ACL-graph capture** 确定性死锁（详见
``docs/F1-c8-injection.md`` 的 caveat）。本模块只负责"注入"，可跑性是另一层。

风格对齐 :mod:`agent_mem.kv.lmcache`：纯函数 + 重依赖 lazy import + 可单测。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# ---- 常量（对齐 quant_model_description.json / vllm-ascend 约定）----
QUANT_DESC_FILENAME = "quant_model_description.json"
SCALES_FILE = "kv_cache_scales.safetensors"
INDEX = "model.safetensors.index.json"
INDEX_BAK = "model.safetensors.index.json.stock.bak"
SF_BAK = "model.safetensors.stock.bak"

# quant_model_description.json 里的 dtype token
FLOAT = "FLOAT"  # 普通 bf16/fp16 权重
C8 = "C8"  # int8 KV cache（scale 张量 + kv_cache_type）

_KV_PROJS = ("k_proj", "v_proj")


def _read_config(model_dir: Path) -> dict:
    p = model_dir / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}（非 HF 模型目录？）")
    return json.loads(p.read_text(encoding="utf-8"))


def _num_hidden_layers(cfg: dict) -> int:
    return int(cfg["num_hidden_layers"])


def _weight_names(model_dir: Path) -> list[str]:
    """所有 ``*.weight`` 参数名（index 优先；否则读单文件 safetensors keys）。"""
    idx = model_dir / INDEX
    if idx.exists():
        wm = json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
        return sorted({k for k in wm if k.endswith(".weight")})
    # 单文件：lazy import safetensors（纯函数测试不走这条）
    from safetensors import safe_open

    sf = model_dir / "model.safetensors"
    with safe_open(str(sf), framework="pt") as f:
        return sorted({k for k in f.keys() if k.endswith(".weight")})


def _kv_scale_keys(num_layers: int) -> list[str]:
    return [
        f"model.layers.{i}.self_attn.{proj}.kv_cache_scale"
        for i in range(num_layers)
        for proj in _KV_PROJS
    ]


def build_c8_quant_description(model_dir: str | Path) -> dict[str, str]:
    """构造 C8 的 quant_model_description（内存 dict，**不落盘**）。

    - 所有 ``.weight`` → ``"FLOAT"``（缺了 vllm 加载会 ``KeyError``）；
    - ``"kv_cache_type"`` → ``"C8"``（触发 ``enable_c8_quant``）；
    - 每层 ``k/v_proj.kv_cache_scale`` → ``"C8"``（填 ``c8_quant_layers``）。

    纯函数：仅需 ``config.json`` + index/safetensors，无 torch/NPU。
    """
    model_dir = Path(model_dir)
    cfg = _read_config(model_dir)
    n = _num_hidden_layers(cfg)
    desc: dict[str, str] = {w: FLOAT for w in _weight_names(model_dir)}
    desc["kv_cache_type"] = C8
    for k in _kv_scale_keys(n):
        desc[k] = C8
    return desc


def is_annotated(model_dir: str | Path) -> bool:
    """模型目录是否已注入 C8 quant_description。"""
    return (Path(model_dir) / QUANT_DESC_FILENAME).exists()


def annotate_model(model_dir: str | Path, *, overwrite: bool = False) -> Path:
    """写 C8 ``quant_model_description.json``，返回其路径。

    已存在且 ``overwrite=False`` → 抛 :class:`FileExistsError`（防误覆盖）。
    删该文件即恢复 stock（无 quant_description 时 C8 路径不会激活）。
    """
    model_dir = Path(model_dir)
    out = model_dir / QUANT_DESC_FILENAME
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} 已存在；传 overwrite=True 覆盖，或先 restore_model()"
        )
    desc = build_c8_quant_description(model_dir)
    out.write_text(
        json.dumps(desc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def _kv_channels(cfg: dict) -> tuple[int, int, int]:
    """per-channel 维度 = ``num_kv_heads * head_size``。

    head_size 优先取 config 显式 ``head_dim``（Qwen3-0.6B=128，≠ hidden//heads），
    否则回退 ``hidden_size // num_attention_heads``（Qwen2.5-7B 无 head_dim → 128）。
    返回 ``(channels, kv_heads, head_size)``。
    """
    nh = int(cfg["num_key_value_heads"])
    hs = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))
    return nh * hs, nh, hs


def inject_placeholder_scales(model_dir: str | Path, value: float = 0.05) -> None:
    """给模型注入**占位** per-channel KV scale（探针用，精度垃圾）。

    sharded 模型：写独立 ``kv_cache_scales.safetensors`` + 更新 index（备份 stock）；
    单文件模型：合并进 ``model.safetensors``（备份 stock）。
    需 torch + safetensors（lazy import），故不进单测。

    真精度 scale 用自校准产出（见 ``scripts/calibrate_f1_c8.py``）。
    """
    import torch
    from safetensors.torch import load_file, save_file

    model_dir = Path(model_dir)
    cfg = _read_config(model_dir)
    ch, _, _ = _kv_channels(cfg)
    n = _num_hidden_layers(cfg)
    scales = {
        f"model.layers.{i}.self_attn.{p}.kv_cache_scale": torch.full(
            (ch,), float(value), dtype=torch.float32
        )
        for i in range(n)
        for p in _KV_PROJS
    }

    idx = model_dir / INDEX
    if idx.exists():  # sharded：独立 scales 文件 + 更新 index（备份 stock）
        save_file(scales, str(model_dir / SCALES_FILE))
        bak = model_dir / INDEX_BAK
        if not bak.exists():
            shutil.copy2(idx, bak)
        data = json.loads(idx.read_text(encoding="utf-8"))
        for k in scales:
            data["weight_map"][k] = SCALES_FILE
        idx.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[c8] sharded: 写 {SCALES_FILE} + 更新 index（{len(scales)} scales, ch={ch}）")
    else:  # 单文件：合并进 model.safetensors（备份 stock）
        sf = model_dir / "model.safetensors"
        if not (model_dir / SF_BAK).exists():
            shutil.copy2(sf, model_dir / SF_BAK)
        allw = load_file(str(sf))
        allw.update(scales)
        save_file(allw, str(sf))
        print(f"[c8] 单文件: 合并 {len(scales)} scales 进 model.safetensors（ch={ch}）")


def restore_model(model_dir: str | Path) -> None:
    """还原 stock 模型：删 quant_description + scales + 还原 index/safetensors 备份。"""
    model_dir = Path(model_dir)
    for f in (QUANT_DESC_FILENAME, SCALES_FILE):
        p = model_dir / f
        if p.exists():
            p.unlink()
            print(f"[c8] 删除 {p.name}")
    bak = model_dir / INDEX_BAK
    if bak.exists():
        shutil.copy2(bak, model_dir / INDEX)
        bak.unlink()
        print("[c8] 还原 stock index")
    sfb = model_dir / SF_BAK
    if sfb.exists():
        shutil.copy2(sfb, model_dir / "model.safetensors")
        sfb.unlink()
        print("[c8] 还原 stock model.safetensors")


def main() -> int:
    """CLI：``python -m agent_mem.kv.c8 {annotate|inject-scales|restore} <model>``。"""
    ap = argparse.ArgumentParser(
        description="F1 C8 模型标注工具（annotate / inject-scales / restore）"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate", help="写 quant_model_description.json（触发 C8，无需 scale）")
    a.add_argument("model", help="HF 模型目录")
    a.add_argument("--overwrite", action="store_true", help="已存在则覆盖")

    s = sub.add_parser("inject-scales", help="注入占位 per-channel KV scale（探针用，需 torch）")
    s.add_argument("model", help="HF 模型目录")
    s.add_argument("--value", type=float, default=0.05, help="占位 scale 值（精度垃圾，仅探针）")

    r = sub.add_parser("restore", help="还原 stock 模型（删 quant_description + scales + 还原备份）")
    r.add_argument("model", help="HF 模型目录")

    args = ap.parse_args()
    if args.cmd == "annotate":
        out = annotate_model(args.model, overwrite=args.overwrite)
        print(f"[c8] wrote {out}")
    elif args.cmd == "inject-scales":
        inject_placeholder_scales(args.model, value=args.value)
    elif args.cmd == "restore":
        restore_model(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
