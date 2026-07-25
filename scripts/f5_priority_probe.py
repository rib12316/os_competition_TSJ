#!/usr/bin/env python
"""F5 priority 调度探针（真机用）——验证 ``--scheduling-policy priority`` 非 no-op。

目的：在 vllm-ascend 上确认 priority 调度真的"低优先被 preempt"（避免重蹈 F1 int8
flag 假生效覆辙），并观察 #41951（被踢的请求重入队行为）。

前提（NPU 由用户启动）：
1. 起一个**低压**引擎（``gpu_memory_utilization`` 小，如 0.3）+ priority 调度。可用
   ``configs/f5-priority-static.yaml``（已带 priority_scheduling=true）::

       python -m agent_mem.server.vllm_server --config configs/f5-priority-static.yaml \\
         --model-path models/Qwen2.5-7B-Instruct --tool-call-parser hermes

2. 跑本探针（指向该引擎）::

       python scripts/f5_priority_probe.py --engine-url http://localhost:8000/v1 \\
         --model Qwen2.5-7B-Instruct

机制：并发发两个长输出请求——一个 priority=0（受保护）、一个 priority=100（可被踢）。
HBM 压力下 vLLM 应先 preempt priority=100 的（recompute）。观察：各自墙钟延迟 +
``/metrics`` 里 preempt 相关计数器的增量。

判读：若 ``priority=low`` 的 wall 明显大于 ``priority=high`` 且 preempt 计数 >0，
则 priority 调度**真生效**；若两者延迟相近、preempt 计数为 0 → 可能 no-op 或压力不够
（调小 ``--gpu-memory-utilization`` / 加大 ``--max-tokens`` / 增并发）。
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# bootstrap agent_mem（脚本从仓库根跑时确保 src 在 path）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "agent-mem" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stream_one(client, *, model, prompt, max_tokens, priority):
    """发一个流式请求（带 priority），返回 (priority, wall_s, n_chunks)。"""
    t0 = time.monotonic()
    n = 0
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        extra_body={
            "priority": priority,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    for _ in stream:
        n += 1
    return priority, time.monotonic() - t0, n


_PREEMPT_RE = re.compile(r"^(vllm:[^\s{#]*preempt[^\s{#]*)\s*(?:\{[^}]*\}\s*)?(\S+)", re.M | re.I)


def _preempt_counters(text: str) -> dict[str, float]:
    """从 /metrics 文本里抽 preempt 相关计数器（名→累加值）。找不到返回 {}。"""
    out: dict[str, float] = {}
    for name, val in _PREEMPT_RE.findall(text):
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            continue
    return out


def _scrape(base_url: str) -> str:
    from agent_mem.bench import vllm_metrics

    return vllm_metrics.scrape(base_url)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="F5 priority 调度探针（真机）")
    p.add_argument("--engine-url", required=True, help="引擎 OpenAI base_url")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="stub")
    p.add_argument("--prompt", default="请把 1 到 300 每个整数的英文单词逐行列出。")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--priority-high", type=int, default=0, help="受保护（默认 0）")
    p.add_argument("--priority-low", type=int, default=100, help="可被踢（默认 100）")
    args = p.parse_args(argv)

    from openai import OpenAI

    client = OpenAI(base_url=args.engine_url, api_key=args.api_key)

    try:
        base = _preempt_counters(_scrape(args.engine_url))
    except Exception as e:  # noqa: BLE001
        print(f"[probe] 跑前抓 /metrics 失败（继续，无基线）：{e}", file=sys.stderr)
        base = {}

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {
            ex.submit(_stream_one, client, model=args.model, prompt=args.prompt,
                      max_tokens=args.max_tokens, priority=args.priority_high): "high",
            ex.submit(_stream_one, client, model=args.model, prompt=args.prompt,
                      max_tokens=args.max_tokens, priority=args.priority_low): "low",
        }
        results: dict[str, tuple] = {}
        for f in as_completed(futs):
            tag = futs[f]
            try:
                results[tag] = f.result()
            except Exception as e:  # noqa: BLE001
                results[tag] = ("ERR", repr(e))

    try:
        after = _preempt_counters(_scrape(args.engine_url))
    except Exception:
        after = {}
    delta = {k: after.get(k, 0.0) - base.get(k, 0.0) for k in set(base) | set(after)}

    print("=== F5 priority 探针结果 ===")
    for tag in ("high", "low"):
        r = results.get(tag)
        if r and r[0] == "ERR":
            print(f"  [{tag}] 失败: {r[1]}")
        elif r:
            pr, wall, n = r
            print(f"  [{tag}] priority={pr} wall={wall:.2f}s chunks~{n}")
    print(f"  preempt 计数器增量: {delta or '（未抓到 preempt 指标）'}")
    print(
        "判读: priority=low 的 wall 明显 > priority=high 且 preempt 增量>0 → priority 调度真生效；\n"
        "      否则可能 no-op 或压力不足（调小 --gpu-memory-utilization / 加大并发或 max-tokens）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
