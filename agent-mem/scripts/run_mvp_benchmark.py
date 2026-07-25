"""MVP benchmark 驱动：每档起引擎 → τ-bench study（N 次取中位）→ compare。

端到端跑通新框架的完整 benchmark，作为后续 F1–F9 的对照基线。每档用对应 config 起
vllm-ascend 引擎 → :class:`QwenAgentRunner` 跑 τ-bench → 采集 6 指标 → 落 run 目录；
全部档跑完后 :func:`compare_runs` 出对照报告。

用法（注意 PYTHONPATH 追加，勿覆盖——否则引擎子进程丢 acl，见 memory）::

    PYTHONPATH="src:$PYTHONPATH" .venv/bin/python scripts/run_mvp_benchmark.py \\
        --tiers baseline prefix_cache --model Qwen2.5-7B-Instruct \\
        --max-tasks 5 --runs 3
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import yaml

from agent_mem.bench.runner import run_study
from agent_mem.bench.runners.qwen_agent import QwenAgentRunner
from agent_mem.config import load_config
from agent_mem.middleware import middlewares_from_config
from agent_mem.repro import repo_root
from agent_mem.server.vllm_server import start_engine, stop_engine, wait_for_engine

REPO = repo_root()
CONFIGS = REPO / "agent-mem" / "configs"
MODEL_PATHS = {
    "Qwen2.5-7B-Instruct": str(REPO / "models" / "Qwen2.5-7B-Instruct"),
    "Qwen3-0.6B": str(REPO / "models" / "Qwen3-0.6B"),
}
# 两档共用的 serve flag（max-model-len 要够大装 τ-bench retail wiki ~7681 token）
SERVE_EXTRA = ["--max-model-len 32768", "--gpu-memory-utilization 0.9"]


def _print(*a, **kw):
    kw.setdefault("flush", True)
    print(*a, **kw)


def run_tier(tier: str, model: str, max_tasks: int, runs: int, log_root: Path) -> dict | None:
    cfg = load_config(CONFIGS / f"{tier}.yaml")
    cfg.engine.model = model
    cfg.benchmark.runs = runs
    # 注入共用 serve flag（保留档自有 flag，如 baseline 的 --no-enable-prefix-caching）
    cfg.engine.extra_args = list(cfg.engine.extra_args) + SERVE_EXTRA
    model_path = MODEL_PATHS[model]
    # config_text 反映实际（mutated）cfg，保证 run 目录的 config.yaml 快照准确
    config_text = yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True)

    _print(f"\n===== TIER {tier}  (model={model} tasks={max_tasks} runs={runs}) =====")
    _print(f"[{tier}] serve extra: {cfg.engine.extra_args}")
    proc, base_url = start_engine(
        cfg, model_path=model_path, tool_call_parser="hermes",
        port=8000, log_file=f"/tmp/bench_{tier}.log",
    )
    _print(f"[{tier}] engine PID={proc.pid} → {base_url}")
    try:
        t0 = time.monotonic()
        wait_for_engine(base_url, timeout=600)
        _print(f"[{tier}] engine ready in {time.monotonic()-t0:.0f}s")
        runner = QwenAgentRunner(
            engine_url=base_url, model=model, max_tasks=max_tasks, max_steps=30,
            api_key="stub", middlewares=middlewares_from_config(cfg).middlewares,
        )
        t1 = time.monotonic()
        all_m, median = run_study(
            cfg, runner, run_root=log_root, config_text=config_text,
            device="npu", engine_url=base_url,
        )
        _print(
            f"[{tier}] done {len(all_m)} runs in {time.monotonic()-t1:.0f}s | "
            f"median: kv_hit={median.get('kv_cache_hit_rate',0):.3f} "
            f"lat_p50={median.get('e2e_latency_p50_ms',0):.0f}ms "
            f"lat_p95={median.get('e2e_latency_p95_ms',0):.0f}ms "
            f"mem_peak={median.get('mem_peak_mb',0):.0f}MB "
            f"ttft={median.get('ttft_ms',0):.0f}ms "
            f"success={median.get('task_success_rate',0):.2f} "
            f"qps={median.get('qps',0):.3f}"
        )
        return median
    except Exception as e:  # noqa: BLE001 — 单档失败不杀整个 benchmark
        _print(f"[{tier}] FAILED: {e!r}", file=sys.stderr)
        traceback.print_exc()
        return None
    finally:
        stop_engine(proc)
        time.sleep(8)  # 让 NPU HBM 释放，避免下档起不来


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MVP benchmark driver（新框架）")
    p.add_argument("--tiers", nargs="+", default=["baseline", "prefix_cache"])
    p.add_argument("--model", default="Qwen2.5-7B-Instruct", choices=list(MODEL_PATHS))
    p.add_argument("--max-tasks", type=int, default=5)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--log-root", default=str(REPO / "logs" / "mvp-newframework"))
    args = p.parse_args(argv)

    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    _print(f"[bench] log_root={log_root}  tiers={args.tiers}")

    t0 = time.monotonic()
    for tier in args.tiers:
        run_tier(tier, args.model, args.max_tasks, args.runs, log_root)

    # 对照报告（仅聚合本次 log_root 下的 run，干净）
    from agent_mem.bench.compare import compare_runs, find_run_dirs

    dirs = find_run_dirs(log_root)
    if not dirs:
        _print("[compare] 未找到 run 目录，跳过对照。", file=sys.stderr)
        return 1
    md = compare_runs(dirs, study="mvp-new-framework", log_root=log_root)
    _print(f"\n[compare] 聚合 {len(dirs)} 个 run 目录 → {md}")
    _print(f"[bench] TOTAL {time.monotonic()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
