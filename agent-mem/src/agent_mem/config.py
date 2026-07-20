"""Benchmark 配置 schema + YAML 加载/校验。

分段结构（对齐 ``configs/*.yaml``）::

    engine:     推理引擎（backend / model / extra_args / lmcache）   缝A/B/C
    benchmark:  任务集（suite / domain / runs / seed / split）
    metrics:    待采集指标名列表
    middleware: 缝D 中间件激活列表 + 选项（F2/F3）
    session:    缝E 生命周期策略（F5/F6）

每个 F 功能有自己的 yaml preset，只开关自己对应的缝段（见 ``configs/f*.yaml``）。

设计：dataclass + 手写 validate（不引入 omegaconf/pydantic，保持零运行时依赖）。
配置示例见 ``configs/{baseline,prefix_cache,optimized,f*.}.yaml``；
命名规范见 ``dev-guide/log-naming-convention.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 白名单（与 log-naming-convention.md 词汇表对齐）
# 注：sglang 在 v2 作废（不支持 Ascend），保留在白名单仅为向后兼容旧 config，真机不可用。
_BACKENDS = ("vllm", "sglang", "vllm-ascend")
_DOMAINS = ("retail", "airline")
_SUITES = ("tau-bench", "agentbench")
_SPLITS = ("train", "test", "dev")
# 缝E 策略名（对齐 scheduler.strategies 的类 name）
_SESSION_STRATEGIES = ("noop", "idle-evict", "checkpoint")

# 6 大必采指标（赛题硬指标）
DEFAULT_METRICS: tuple[str, ...] = (
    "e2e_latency_p50",
    "e2e_latency_p95",
    "qps",
    "mem_peak_mb",
    "kv_cache_hit_rate",
    "task_success_rate",
    "ttft",
)


class ConfigError(ValueError):
    """配置解析/校验失败。"""


@dataclass
class LmCacheConfig:
    """LMCache 分层存储配置（F4 · 缝C，optimized / f4-lmcache 档用）。"""

    enabled: bool = False
    config_file: str | None = None


@dataclass
class MiddlewareConfig:
    """缝D 中间件激活配置（F2 压缩 / F3 lazy-load）。

    - ``active``：注册表里的中间件名（如 ``["compress"]``）；空 = 不启用（identity）。
    - ``options``：每个中间件的构造 kwargs，键为名字。
    名字→类的解析在 :func:`agent_mem.middleware.build_middlewares`（动态注册表），
    配置层只校验是非空字符串，不耦合中间件包。
    """

    active: list[str] = field(default_factory=list)
    options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SessionConfig:
    """缝E session 生命周期配置（F5 idle eviction / F6 checkpoint）。

    - ``strategy``：策略名（``noop`` / ``idle-evict`` / ``checkpoint``）。
    - ``idle_timeout_s``：F5 的 idle 阈值（秒）；策略由 scheduler 消费。
    - ``options``：策略构造的额外 kwargs（如 checkpoint 路径）。
    机制（offload/save 回调）由运行时注入，不在配置里。
    """

    strategy: str = "noop"
    idle_timeout_s: float = 60.0
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    backend: str = "vllm"
    model: str = ""
    extra_args: list[str] = field(default_factory=list)
    lmcache: LmCacheConfig = field(default_factory=LmCacheConfig)


@dataclass
class BenchmarkConfig:
    suite: str = "tau-bench"
    domain: str = "retail"
    runs: int = 3
    seed: int = 42
    split: str = "test"


@dataclass
class MetricsConfig:
    collect: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))


@dataclass
class AppConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    # 从 yaml 文件 stem 填（如 "prefix_cache"），用于 run 目录命名
    config_name: str = ""


def _build_engine(data: dict[str, Any]) -> EngineConfig:
    lm_data = data.get("lmcache") or {}
    return EngineConfig(
        backend=data.get("backend", "vllm"),
        model=data.get("model", ""),
        extra_args=list(data.get("extra_args") or []),
        lmcache=LmCacheConfig(
            enabled=bool(lm_data.get("enabled", False)),
            config_file=lm_data.get("config_file"),
        ),
    )


def _build_benchmark(data: dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig(
        suite=data.get("suite", "tau-bench"),
        domain=data.get("domain", "retail"),
        runs=int(data.get("runs", 3)),
        seed=int(data.get("seed", 42)),
        split=data.get("split", "test"),
    )


def _build_metrics(data: Any) -> MetricsConfig:
    # yaml 里 metrics 是顶层 list；缺省取 DEFAULT_METRICS
    if data is None:
        return MetricsConfig(collect=list(DEFAULT_METRICS))
    if isinstance(data, list):
        return MetricsConfig(collect=[str(x) for x in data])
    if isinstance(data, dict) and "collect" in data:
        return MetricsConfig(collect=[str(x) for x in (data.get("collect") or [])])
    raise ConfigError(
        f"metrics 段格式无法识别：{type(data).__name__}（应为 list 或含 collect 的 dict）"
    )


def _build_middleware(data: Any) -> MiddlewareConfig:
    """解析 ``middleware`` 段。缺省 → 空激活（identity）。"""
    if data is None:
        return MiddlewareConfig()
    if not isinstance(data, dict):
        raise ConfigError(
            f"middleware 段必须是 mapping，得到 {type(data).__name__}"
        )
    active = data.get("active") or []
    if not isinstance(active, list):
        raise ConfigError("middleware.active 必须是字符串列表")
    options = data.get("options") or {}
    if not isinstance(options, dict):
        raise ConfigError("middleware.options 必须是 mapping")
    return MiddlewareConfig(
        active=[str(x) for x in active],
        options={str(k): dict(v) for k, v in options.items()},
    )


def _build_session(data: Any) -> SessionConfig:
    """解析 ``session`` 段。缺省 → noop 策略。"""
    if data is None:
        return SessionConfig()
    if not isinstance(data, dict):
        raise ConfigError(
            f"session 段必须是 mapping，得到 {type(data).__name__}"
        )
    return SessionConfig(
        strategy=str(data.get("strategy", "noop")),
        idle_timeout_s=float(data.get("idle_timeout_s", 60.0)),
        options=dict(data.get("options") or {}),
    )


def validate(cfg: AppConfig) -> None:
    """校验配置（白名单 + 取值范围）。失败抛 :class:`ConfigError`。"""
    e = cfg.engine
    if e.backend not in _BACKENDS:
        raise ConfigError(f"engine.backend={e.backend!r} 不在白名单 {_BACKENDS}")
    if not e.model:
        raise ConfigError("engine.model 不能为空")
    if e.extra_args and not all(isinstance(a, str) for a in e.extra_args):
        raise ConfigError("engine.extra_args 必须是字符串列表")

    b = cfg.benchmark
    if b.suite not in _SUITES:
        raise ConfigError(f"benchmark.suite={b.suite!r} 不在白名单 {_SUITES}")
    if b.domain not in _DOMAINS:
        raise ConfigError(f"benchmark.domain={b.domain!r} 不在白名单 {_DOMAINS}")
    if b.runs < 1:
        raise ConfigError(f"benchmark.runs={b.runs} 必须 >= 1")
    if b.seed < 0:
        raise ConfigError(f"benchmark.seed={b.seed} 必须 >= 0")
    if b.split not in _SPLITS:
        raise ConfigError(f"benchmark.split={b.split!r} 不在白名单 {_SPLITS}")

    if not cfg.metrics.collect:
        raise ConfigError("metrics.collect 不能为空")

    # 缝D：中间件激活名（只校验非空字符串；名字→类解析在 middleware 包）
    for name in cfg.middleware.active:
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"middleware.active 含空名：{cfg.middleware.active!r}")

    # 缝E：session 策略白名单 + idle 阈值
    s = cfg.session
    if s.strategy not in _SESSION_STRATEGIES:
        raise ConfigError(
            f"session.strategy={s.strategy!r} 不在白名单 {_SESSION_STRATEGIES}"
        )
    if s.idle_timeout_s < 0:
        raise ConfigError(f"session.idle_timeout_s={s.idle_timeout_s} 必须 >= 0")


def load_config(path: str | Path) -> AppConfig:
    """从 YAML 文件加载配置，构造 :class:`AppConfig` 并校验。

    ``config_name`` 取自文件 stem（如 ``prefix_cache.yaml`` → ``"prefix_cache"``）。
    """
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"配置根必须是 mapping，得到 {type(raw).__name__}: {p}")
    cfg = AppConfig(
        engine=_build_engine(raw.get("engine") or {}),
        benchmark=_build_benchmark(raw.get("benchmark") or {}),
        metrics=_build_metrics(raw.get("metrics")),
        middleware=_build_middleware(raw.get("middleware")),
        session=_build_session(raw.get("session")),
        config_name=p.stem,
    )
    validate(cfg)
    return cfg
