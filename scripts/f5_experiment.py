#!/usr/bin/env python
"""F5 A/B/C 实验编排（真机用）——依次跑三组并发 τ-bench 对照，出 before/after。

组（见 ``configs/f5-*.yaml`` + ``docs/F5-experiment-design.md``）：
  A native         : vllm 原生 FCFS + APC（无 priority、无应用层调度）
  B priority-static: ``--scheduling-policy priority``（固定优先级）
  C evict-dynamic  : priority + :class:`ConcurrentSessionDriver`（动态 priority + HBM 准入）

每组：起引擎（该组 config）→ 跑并发 benchmark（``runs=3`` 取中位数）→ 停引擎。
最后用 ``benchmarks/runner.py --compare`` 聚合三组 run 目录出对照报告。

前提：**NPU 已由用户启动**。用法示例::

    # 全自动（脚本起/停引擎）
    python scripts/f5_experiment.py --model-path models/Qwen2.5-7B-Instruct \\
        --max-concurrency 6 --device npu [--user-api-base ... --user-api-key ...]

    # 引擎自己起好了（只跑 benchmark）
    python scripts/f5_experiment.py --skip-engine --model-path models/Qwen2.5-7B-Instruct \\
        --max-concurrency 6 --device npu

    # 只跑某一组
    python scripts/f5_experiment.py --only C --model-path models/Qwen2.5-7B-Instruct ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "agent-mem" / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_GROUPS: list[tuple[str, str, str]] = [
    ("A", "f5-native",          "vllm 原生 FCFS + APC"),
    ("B", "f5-priority-static", "--scheduling-policy priority（固定）"),
    ("C", "f5-evict-dynamic",   "动态 priority + HBM 准入（ConcurrentSessionDriver）"),
    ("D", "f5-evict-dynamic-offload", "C + 无损 KV offload（SimpleCPUOffloadConnector→Ascend 变体）"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F5 A/B/C 实验编排（真机）")
    ap.add_argument("--model-path", required=True, help="模型权重路径（vllm --model）")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-concurrency", type=int, default=6, help="并发 task 数")
    ap.add_argument("--max-tasks", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--device", default="npu", choices=["npu", "cuda"])
    ap.add_argument("--tool-call-parser", default="hermes")
    ap.add_argument("--api-key", default="stub")
    ap.add_argument("--user-model", default=None)
    ap.add_argument("--user-api-base", default=None)
    ap.add_argument("--user-api-key", default=None)
    ap.add_argument("--only", default=None, help="只跑某组（A/B/C）")
    ap.add_argument("--skip-engine", action="store_true",
                    help="引擎已外部起好（--port），只跑 benchmark")
    args = ap.parse_args(argv)

    from agent_mem.config import load_config
    from agent_mem.server.vllm_server import start_engine, stop_engine, wait_for_engine

    base_url = f"http://127.0.0.1:{args.port}/v1"
    run_py = str(_ROOT / "agent-mem" / "benchmarks" / "runner.py")
    cfg_dir = _ROOT / "agent-mem" / "configs"
    logs_dir = _ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    groups = [g for g in _GROUPS if args.only is None or g[0] == args.only]
    if not groups:
        print(f"[F5] --only={args.only!r} 无匹配组（可选 A/B/C）", file=sys.stderr)
        return 2

    for tag, cfg_stem, desc in groups:
        cfg_path = cfg_dir / f"{cfg_stem}.yaml"
        print(f"\n===== 组 {tag} [{cfg_stem}] {desc} =====", flush=True)

        proc = None
        if not args.skip_engine:
            cfg = load_config(str(cfg_path))
            log_path = str(logs_dir / f"f5_engine_{tag}.log")
            print(f"[F5] 起引擎（{cfg_path.name}，日志 {log_path}）…", flush=True)
            proc, _ = start_engine(
                cfg, model_path=args.model_path, port=args.port,
                tool_call_parser=args.tool_call_parser, log_file=log_path,
            )
            try:
                wait_for_engine(base_url, timeout=900)
            except TimeoutError as e:
                print(f"[F5] 组 {tag} 引擎未就绪，跳过：{e}", file=sys.stderr)
                stop_engine(proc)
                continue

        cmd = [
            py, run_py, "--config", str(cfg_path), "--runner", "qwen-agent",
            "--engine-url", base_url,
            "--max-concurrency", str(args.max_concurrency),
            "--max-tasks", str(args.max_tasks), "--max-steps", str(args.max_steps),
            "--device", args.device, "--api-key", args.api_key,
        ]
        if args.user_model:
            cmd += ["--user-model", args.user_model]
        if args.user_api_base:
            cmd += ["--user-api-base", args.user_api_base]
        if args.user_api_key:
            cmd += ["--user-api-key", args.user_api_key]
        print(f"[F5] 跑 benchmark：{' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd, cwd=str(_ROOT))
        print(f"[F5] 组 {tag} benchmark rc={rc}", flush=True)

        if proc is not None:
            print("[F5] 停引擎…", flush=True)
            stop_engine(proc)
            time.sleep(3)  # 让 HBM 释放干净再起下一组

    print("\n[F5] 全部组完成。聚合对照报告：", flush=True)
    print(f"  {py} {run_py} --compare --study f5-dynamic-reclaim", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
