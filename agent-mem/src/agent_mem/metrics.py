"""``metrics.json`` schema + 序列化。

字段严格对齐 ``dev-guide/log-naming-convention.md`` 的 schema；
字段顺序即 JSON 输出顺序。骨架阶段所有数值指标默认 0（真值由 MVP 采集）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RunMetrics:
    """单次 run 的结构化指标结果（6 大指标 + run 元信息）。"""

    run_id: str
    engine: str
    model: str
    config: str
    e2e_latency_p50_ms: float = 0.0
    e2e_latency_p95_ms: float = 0.0
    qps: float = 0.0
    mem_peak_mb: int = 0
    kv_cache_hit_rate: float = 0.0
    kv_cache_usage_perc: float = 0.0  # KV pool 利用率（vllm:kv_cache_usage_perc gauge；F5 准入信号，旧日志缺则为 0）
    task_success_rate: float = 0.0
    ttft_ms: float = 0.0
    # F5 动态资源回收指标（仅 session.strategy=priority-evict 的 dynamic 路径采集）
    evictions: int = 0  # 回收次数（PriorityEvictionStrategy 的 priority 抬升计数）
    idle_hits: int = 0  # 其中命中 idle session 的次数
    idle_hit_rate: float = 0.0  # eviction 命中 idle 比例（0~1），越高说明"回收闲的"越准
    seed: int = 0
    started_at: str = ""


# 字段顺序与 log-naming-convention.md 的 schema 严格一致（取 dataclass 声明顺序）
METRIC_FIELDS: tuple[str, ...] = tuple(RunMetrics.__dataclass_fields__.keys())


def default_metrics(
    *,
    run_id: str,
    engine: str,
    model: str,
    config: str,
    seed: int = 0,
    started_at: str = "",
) -> RunMetrics:
    """构造一个全 0 的占位 :class:`RunMetrics`（骨架阶段用）。"""
    return RunMetrics(
        run_id=run_id,
        engine=engine,
        model=model,
        config=config,
        seed=seed,
        started_at=started_at,
    )


def to_json(m: RunMetrics, *, indent: int = 2) -> str:
    """序列化为 JSON 字符串（字段顺序固定）。"""
    return json.dumps(asdict(m), ensure_ascii=False, indent=indent)


def write(m: RunMetrics, path: str | Path) -> Path:
    """写入 ``metrics.json``，返回写入路径。"""
    p = Path(path)
    p.write_text(to_json(m) + "\n", encoding="utf-8")
    return p
