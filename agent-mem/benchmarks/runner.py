"""Benchmark 主入口（占位实现）。

三档对照：baseline → prefix_cache → optimized。
指标定义见 configs/*.yaml；命名规范见 dev-guide/log-naming-convention.md。

TODO（按 roadmap 推进）:
- 接入 tau-bench retail/airline 任务集
- 采集 6 大指标（延迟 p50/p95、QPS、显存峰值、KV 命中率、成功率、TTFT）
- 抓取 vLLM /metrics（vllm:gpu_prefix_cache_hits_total 等）
- 显存按时间采样写入 mem_timeseries.csv
- 3 次取中位数，产出 logs/_summaries/<date>_<study>_comparison.md
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="agent-mem benchmark runner")
    parser.add_argument("--config", required=True, help="configs/*.yaml 路径")
    parser.add_argument("--engine", default=None, help="覆盖 config 中的 engine.backend")
    args = parser.parse_args()
    # 占位：实际实现在 roadmap MVP 阶段补全
    raise NotImplementedError(
        f"benchmark runner 尚未实现（config={args.config}, engine={args.engine}）。"
        "见 dev-guide/roadmap.md。"
    )


if __name__ == "__main__":
    main()
