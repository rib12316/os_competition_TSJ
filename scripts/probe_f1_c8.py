#!/usr/bin/env python3
"""F1 C8 int8 KV 探针（Qwen3-native 版，**需 NPU**）。

历史教训（Qwen2.5-7B 上的 6 轮探针，见 memory f1-int8-c8-plan）：
- C8 不走 `--kv-cache-dtype`，走 `--quantization ascend` + 模型目录 quant_model_description.json。
- C8 强制 per-channel `(num_kv_heads*head_size,)` scale（默认 ones(1) 会崩 `.view(h,1,d)`）。
- C8 scale 加载 patch **只覆盖 Qwen3/Glm4Moe/MiniMaxM2**（不覆盖 Qwen2）→ 用 Qwen3。
- Qwen2 上 graph 模式非确定性挂起；Qwen3 是 C8 原生支持，graph 路径已测，应不挂。
- 设 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` 跳过 profiling 挂起。
- 探针失败会留孤儿 `VLLM::EngineCore` 子进程占显存 → 必须单独 kill。

本脚本合并 annotate + 注入 scale + 起引擎 + 检查 + 清理，支持 sharded 和单文件模型。

用法（NPU 已启动）：
  PYTHONPATH=agent-mem/src:$PYTHONPATH .venv/bin/python scripts/probe_f1_c8.py \
      --model models/Qwen3-0.6B --port 8001 [--scale 0.05] [--eager] [--restore]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

QUANT_DESC = "quant_model_description.json"
SCALES_FILE = "kv_cache_scales.safetensors"
INDEX = "model.safetensors.index.json"
INDEX_BAK = "model.safetensors.index.json.stock.bak"
SF_BAK = "model.safetensors.stock.bak"

C8_MARK = re.compile(r"AscendC8|enable_c8_quant|C8KVCache|kv_cache_torch_dtype.*int8|C8_KV", re.I)
FAIL_MARK = re.compile(r"EngineCore failed|RuntimeError|Traceback|KeyError|ValueError|AssertionError|OperationSetup", re.I)


def _cfg(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


def _chans(cfg: dict) -> tuple[int, int, int]:
    """per-channel 维度 = num_kv_heads * head_size。

    head_size 优先取 config 显式 ``head_dim``（Qwen3-0.6B=128，≠ hidden//num_heads=64），
    否则回退 ``hidden//num_attention_heads``（Qwen2.5-7B 无 head_dim → 128）。
    """
    nh = int(cfg["num_key_value_heads"])
    hs = int(cfg.get("head_dim") or int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))
    return nh * hs, nh, hs


def _weight_names(model_dir: Path) -> list[str]:
    """所有 *.weight 参数名（index 优先，否则读单文件 safetensors keys）。"""
    idx = model_dir / INDEX
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return sorted({k for k in wm if k.endswith(".weight")})
    from safetensors import safe_open
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        return sorted({k for k in f.keys() if k.endswith(".weight")})


def annotate(model_dir: Path) -> Path:
    cfg = _cfg(model_dir)
    n = int(cfg["num_hidden_layers"])
    desc = {w: "FLOAT" for w in _weight_names(model_dir)}
    desc["kv_cache_type"] = "C8"
    for i in range(n):
        for proj in ("k_proj", "v_proj"):
            desc[f"model.layers.{i}.self_attn.{proj}.kv_cache_scale"] = "C8"
    out = model_dir / QUANT_DESC
    out.write_text(json.dumps(desc, indent=2, ensure_ascii=False) + "\n")
    print(f"[probe] wrote {out} ({sum(1 for v in desc.values() if v=='C8')} C8 entries, {n} layers)")
    return out


def inject_scales(model_dir: Path, value: float) -> None:
    import torch
    from safetensors.torch import save_file, load_file

    cfg = _cfg(model_dir)
    ch, _, _ = _chans(cfg)
    n = int(cfg["num_hidden_layers"])
    scales = {
        f"model.layers.{i}.self_attn.{p}.kv_cache_scale": torch.full((ch,), float(value), dtype=torch.float32)
        for i in range(n) for p in ("k_proj", "v_proj")
    }

    idx = model_dir / INDEX
    if idx.exists():
        # sharded：独立 scales 文件 + 更新 index（备份 stock）
        save_file(scales, str(model_dir / SCALES_FILE))
        bak = model_dir / INDEX_BAK
        if not bak.exists():
            import shutil
            shutil.copy2(idx, bak)
        data = json.loads(idx.read_text())
        for k in scales:
            data["weight_map"][k] = SCALES_FILE
        idx.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"[probe] sharded: wrote {SCALES_FILE} + updated index ({len(scales)} scales, ch={ch})")
    else:
        # 单文件：把 scale 直接加进 model.safetensors（备份 stock）
        import shutil
        sf = model_dir / "model.safetensors"
        if not (model_dir / SF_BAK).exists():
            shutil.copy2(sf, model_dir / SF_BAK)
        allw = load_file(str(sf))
        allw.update(scales)
        save_file(allw, str(sf))
        print(f"[probe] single-file: merged {len(scales)} scales into model.safetensors (ch={ch})")


def restore(model_dir: Path) -> None:
    import shutil
    for f in (QUANT_DESC, SCALES_FILE):
        p = model_dir / f
        if p.exists():
            p.unlink()
            print(f"[probe] removed {p}")
    bak = model_dir / INDEX_BAK
    if bak.exists():
        shutil.copy2(bak, model_dir / INDEX)
        bak.unlink()
        print("[probe] restored stock index")
    sfb = model_dir / SF_BAK
    if sfb.exists():
        shutil.copy2(sfb, model_dir / "model.safetensors")
        sfb.unlink()
        print("[probe] restored stock model.safetensors")


def wait_ready(port: int, timeout: float) -> bool:
    import httpx
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--scale", type=float, default=0.05, help="占位 scale 值（探针用，精度垃圾）")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--startup-timeout", type=int, default=600)
    ap.add_argument("--extra", default="", help="额外 vLLM 参数，如 '--max-num-seqs 32'")
    ap.add_argument("--eager", action="store_true", help="加 --enforce-eager（跳过 graph 捕获）")
    ap.add_argument("--restore", action="store_true", help="还原 stock 模型")
    ap.add_argument("--skip-prep", action="store_true", help="跳过 annotate+注入（模型已由 calibrate 准备好真 scale）")
    ap.add_argument("--no-probe-env", action="store_true", help="不设 VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0")
    args = ap.parse_args()

    model_dir = Path(args.model)
    if args.restore:
        restore(model_dir)
        return 0

    if not args.skip_prep:
        annotate(model_dir)
        inject_scales(model_dir, args.scale)

    Path("logs-f1-c8").mkdir(exist_ok=True)
    log_path = Path("logs-f1-c8") / f"probe-{args.port}.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model, "--served-model-name", "probe",
        "--quantization", "ascend", "--host", "127.0.0.1", "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
    ]
    if args.eager:
        cmd.append("--enforce-eager")
    if args.extra:
        import shlex
        cmd += shlex.split(args.extra)
    print("[probe] launch:", " ".join(cmd), "| log:", log_path)

    env = os.environ.copy()
    if not args.no_probe_env:
        env.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

    log_file = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    verdict = "FAIL"
    try:
        ready = wait_ready(args.port, args.startup_timeout)
        text = log_path.read_text(errors="replace")
        lines = text.splitlines()
        c8 = [ln.strip()[:160] for ln in lines if C8_MARK.search(ln)]
        fail = [ln.strip()[:160] for ln in lines if FAIL_MARK.search(ln)]
        reply = None
        tput = ""
        if ready:
            import httpx
            try:
                t0 = time.time()
                r = httpx.post(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                               json={"model": "probe", "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 8},
                               timeout=120)
                reply = r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"<status {r.status_code}>"
                tput = f"{8/(time.time()-t0):.1f} tok/s(含首token)"
            except Exception as e:
                reply = f"<request error: {e}>"
        # 找 graph 捕获是否 100%
        graph_ok = any("Capturing CUDA graphs" in ln and "100%" in ln for ln in lines)

        print("\n========== F1 C8 PROBE RESULT ==========")
        print(f"engine ready   : {ready}")
        print(f"graph captured : {graph_ok}")
        print(f"C8 markers     : {len(c8)}")
        for ln in c8[:6]:
            print("    +", ln)
        print(f"fail markers   : {len(fail)}")
        for ln in fail[:6]:
            print("    !", ln)
        print(f"probe reply    : {reply!r}  {tput}")
        if ready and c8 and not fail and reply and not str(reply).startswith("<"):
            verdict = "PASS"
        print(f"\nVERDICT: {verdict}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        # 清理孤儿 EngineCore 子进程（pkill 父进程杀不死它）
        subprocess.run("pkill -9 -f 'vllm.entrypoints'; for p in $(pgrep -f EngineCore); do kill -9 $p; done",
                       shell=True, stderr=subprocess.DEVNULL)
        log_file.close()
        print(f"[probe] server stopped; log: {log_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
