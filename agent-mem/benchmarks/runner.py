"""Benchmark 主入口（薄 CLI）。

驱动一次 study（同 config 跑 N 次取中位数）：

    python benchmarks/runner.py --config configs/baseline.yaml
    python benchmarks/runner.py --config configs/baseline.yaml --runner tau-bench
    python benchmarks/runner.py --config configs/prefix_cache.yaml --runs 3 --engine vllm

day-1 默认 ``--runner dry-run``（指标全 0，纯 CPU 跑通落盘）；``--runner tau-bench``
惰性加载 τ-bench 验证 import 链路。MVP 时新增 ``qwen-agent`` runner 接 P2 CLI。
命名规范见 ``dev-guide/log-naming-convention.md``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_mem.bench.runner import RUNNERS, get_runner, run_study
from agent_mem.config import load_config
from agent_mem.repro import repo_root

# 中位数指标打印顺序（数值类）
_PRINT_FIELDS = (
    "e2e_latency_p50_ms",
    "e2e_latency_p95_ms",
    "qps",
    "mem_peak_mb",
    "kv_cache_hit_rate",
    "task_success_rate",
    "ttft_ms",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="agent-mem benchmark runner")
    p.add_argument("--config", default=None, help="configs/*.yaml 路径（run 模式必填）")
    p.add_argument(
        "--runner",
        default="dry-run",
        choices=list(RUNNERS) + ["qwen-agent"],
        help="任务执行器（dry-run/tau-bench 无参；qwen-agent 需 --engine-url）",
    )
    p.add_argument("--engine", default=None, help="覆盖 config 中的 engine.backend")
    p.add_argument("--runs", type=int, default=None, help="覆盖 config 中的 benchmark.runs")
    p.add_argument("--domain", default=None, help="覆盖 config 中的 benchmark.domain")
    p.add_argument(
        "--model-name",
        default=None,
        help="覆盖 engine.model（同时作 agent 调用名 + metrics 标签）",
    )
    p.add_argument(
        "--log-root",
        default=None,
        help="run 目录根（默认 <repo>/logs）；compare 模式下也是读取源",
    )
    # 真采集（P1 引擎就绪后用）
    p.add_argument(
        "--engine-url",
        default=None,
        help="引擎 OpenAI base_url（如 http://localhost:8000/v1），给定时抓 /metrics 算 KV 命中率",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cuda", "npu"],
        help="显存采样设备；不给则不采显存（NPU 默认停着，真跑前暂停等用户启动）",
    )
    # qwen-agent 真跑参数
    p.add_argument("--max-tasks", type=int, default=None, help="限制真跑任务数（qwen-agent）")
    p.add_argument("--max-steps", type=int, default=30, help="单任务最大 tool-calling 步数")
    p.add_argument("--priority", type=int, default=0, help="agent 调度优先级（0=最高，值越大越先被驱逐，F5）")
    p.add_argument("--api-key", default="stub", help="引擎 API key（vLLM 不校验，占位即可）")
    # user-simulator 扩充选项（默认走本地引擎；给 --user-api-base 切外部 OpenAI 兼容 API）
    p.add_argument(
        "--user-model", default=None,
        help="user-sim 模型名（默认=agent 本地模型；外部如 gpt-4o / abab-... ）",
    )
    p.add_argument(
        "--user-provider", default="openai",
        help="user-sim 的 litellm provider（openai|anthropic|…；非 openai 系请自行 export 对应 env）",
    )
    p.add_argument(
        "--user-api-base", default=None,
        help="user-sim 外部端点（OpenAI 兼容，如 https://api.minimaxi.com/v1）；不给则 user-sim 走本地引擎",
    )
    p.add_argument("--user-api-key", default=None, help="user-sim 外部端点的 API key")
    # compare 模式
    p.add_argument(
        "--compare",
        action="store_true",
        help="对照模式：不跑，聚合 --log-root 下已有 run 目录，生成 comparison.md",
    )
    p.add_argument("--study", default="mvp-three-tier", help="对照报告 study 名（文件名用）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    run_root = Path(args.log_root) if args.log_root else repo_root() / "logs"
    run_root.mkdir(parents=True, exist_ok=True)

    # ---- compare 模式：聚合已有 run 目录出对照报告 ----
    if args.compare:
        from agent_mem.bench.compare import compare_runs, find_run_dirs

        dirs = find_run_dirs(run_root)
        if not dirs:
            print(f"[compare] {run_root} 下未找到 run 目录。", file=sys.stderr)
            return 1
        p = compare_runs(dirs, study=args.study, log_root=run_root)
        print(f"[compare] 生成对照报告：{p}", flush=True)
        print(f"[compare] 聚合了 {len(dirs)} 个 run 目录。", flush=True)
        return 0

    # ---- run 模式 ----
    if not args.config:
        print("[runner] run 模式需要 --config（或用 --compare 聚合已有 run）。", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.engine:
        cfg.engine.backend = args.engine
    if args.runs is not None:
        cfg.benchmark.runs = args.runs
    if args.domain:
        cfg.benchmark.domain = args.domain
    if args.model_name:
        cfg.engine.model = args.model_name  # 影响 agent 调用名 + metrics 标签

    config_text = Path(args.config).read_text(encoding="utf-8")

    if args.runner == "qwen-agent":
        if not args.engine_url:
            print("[runner] --runner qwen-agent 需要 --engine-url。", file=sys.stderr)
            return 2
        from agent_mem.bench.runners.qwen_agent import QwenAgentRunner
        from agent_mem.middleware import middlewares_from_config

        mw_stack = middlewares_from_config(cfg)
        runner = QwenAgentRunner(
            engine_url=args.engine_url,
            model=cfg.engine.model,
            max_tasks=args.max_tasks,
            max_steps=args.max_steps,
            api_key=args.api_key,
            user_model=args.user_model,
            user_provider=args.user_provider,
            user_api_base=args.user_api_base,
            user_api_key=args.user_api_key,
            priority=args.priority,
            middlewares=mw_stack.middlewares,  # 缝D：cfg.middleware 激活的中间件
        )
    else:
        runner = get_runner(args.runner)
    print(
        f"[runner] engine={cfg.engine.backend} model={cfg.engine.model} "
        f"config={cfg.config_name} runner={runner.name()} runs={cfg.benchmark.runs} "
        f"domain={cfg.benchmark.domain}/{cfg.benchmark.split} → {run_root}",
        flush=True,
    )

    try:
        all_metrics, median = run_study(
            cfg,
            runner,
            run_root=run_root,
            config_text=config_text,
            device=args.device,
            engine_url=args.engine_url,
        )
    except ModuleNotFoundError as e:
        print(
            f"[runner] 缺少依赖，无法用 {args.runner!r} 跑（{e.name}）。"
            "day-1 请用默认 --runner dry-run。",
            file=sys.stderr,
        )
        return 2

    print(f"\n[runner] 完成 {len(all_metrics)} 次 run（取中位数）：", flush=True)
    for m in all_metrics:
        print(f"  - {m.run_id}", flush=True)
    print("\n中位数指标：", flush=True)
    for f in _PRINT_FIELDS:
        print(f"  {f:>22} = {median.get(f, 0.0)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
