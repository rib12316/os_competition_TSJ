"""Benchmark 配置 schema + YAML 加载/校验。

分段结构（对齐 ``configs/*.yaml``）::

    engine:     推理引擎（backend / model / extra_args / kv_transfer） 缝A/B/C
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
_SUITES = ("tau-bench", "agentbench", "longbench")
_SPLITS = ("train", "test", "dev")
# 缝E 策略名（对齐 scheduler.strategies 的类 name）
_SESSION_STRATEGIES = ("noop", "idle-evict", "checkpoint", "priority-evict", "progress-evict", "combined-evict")

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


# 缝C KV connector 白名单（vllm-ascend KVConnectorFactory 注册表中 Ascend 适配的）
_KV_CONNECTORS = ("", "LMCacheAscendConnector", "SimpleCPUOffloadConnector")
_KV_ROLES = ("kv_both", "kv_producer", "kv_consumer")


@dataclass
class KVTransferConfig:
    """缝C · KV transfer connector 配置（F4 LMCache Ascend / native CPU offload）。

    vllm-ascend 0.22.1rc1 已内置 LMCacheAscendConnector（factory 注册），
    只需通过 ``--kv-transfer-config`` JSON 激活；``lmcache_ascend`` 包需在
    NPU 上另行安装（见 ``docs/F4-lmcache-ascend.md``）。
    """

    connector: str = ""
    role: str = "kv_both"
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class C8Config:
    """缝A F1：vllm-ascend 原生 **C8 int8 KV 量化**配置。

    - ``enabled``：开则 :func:`build_serve_args` 注入 ``--quantization ascend`` +
      ``--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}``。FULL decode 在 910B
      上确定性死锁（见 ``docs/F1-c8-injection.md``），故强制 FULL_DECODE_ONLY（decode eager）。
      **前置**：模型须先 annotate + 校准（post-RoPE per-channel MinMax scale），见
      ``python -m agent_mem.kv.c8`` 与 ``scripts/calibrate_c8_qwen2.py``；否则 C8 激活但输出垃圾。
    - ``patch_qwen2``：Qwen2.5（Qwen2 架构）不在 vllm-ascend 的 ``patch_gqa_c8`` 覆盖里
      （只 Qwen3/Glm4Moe/MiniMaxM2），需经 ``agent_mem/kv/c8patch/sitecustomize.py`` 在
      EngineCore 子进程给 ``Qwen2ForCausalLM.load_weights`` 打补丁，KV scale 才会加载
      （否则默认 ones(1) → ``_prepare_c8_scales`` 的 ``.view`` 崩）。Qwen3 等原生覆盖的模型设 False。
    """

    enabled: bool = False
    patch_qwen2: bool = False


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

    - ``strategy``：策略名（``noop`` / ``idle-evict`` / ``checkpoint`` / ``priority-evict``）。
      ``priority-evict`` = F5 动态优先级回收（Phase 1 lossy：idle→抬 priority，让 vLLM 先抢）。
    - ``idle_timeout_s``：F5 的 idle 阈值（秒）；策略由 scheduler 消费。
    - ``options``：策略构造的额外 kwargs（如 checkpoint 路径）。
    机制（offload/save 回调）由运行时注入，不在配置里。
    """

    strategy: str = "noop"
    idle_timeout_s: float = 60.0
    target_lo: int = 70
    target_hi: int = 85
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineConfig:
    backend: str = "vllm"
    model: str = ""
    extra_args: list[str] = field(default_factory=list)
    # 缝C（统一槽）：KV transfer connector —— yaml 二选一后端
    #   LMCacheAscendConnector (F4，需编译 lmcache_ascend，三级 NPU/CPU/Disk)
    #   SimpleCPUOffloadConnector (F5-Phase2，零依赖，单级 lazy offload)
    kv_transfer: KVTransferConfig = field(default_factory=KVTransferConfig)
    # 缝A（F1）：C8 int8 KV 量化（Ascend 专属）
    c8: C8Config = field(default_factory=C8Config)
    # F5（缝A）：priority 调度 —— HBM 满时 vLLM 先踢低优先级请求
    priority_scheduling: bool = False


@dataclass
class BenchmarkConfig:
    suite: str = "tau-bench"
    domain: str = "retail"
    runs: int = 3
    seed: int = 42
    split: str = "test"
    data_zip: str = ""  # longbench: THUDM/LongBench 数据 zip 路径（suite=longbench 必填）
    options: dict[str, Any] = field(default_factory=dict)  # suite 专属旋钮（longbench: start/limit/...）


@dataclass
class MetricsConfig:
    collect: list[str] = field(default_factory=lambda: list(DEFAULT_METRICS))


@dataclass
class UserSimConfig:
    """tau-bench user-simulator 配置（对话的 user 侧）。

    agent 侧始终走本地引擎（被测/被优化对象）；user-sim 默认也走本地引擎（开源可复现），
    但可切外部 OpenAI 兼容 API（如 mimo / GPT-4o）获得更强、更真实的对话。

    - ``api_base`` 为空 → user-sim 走本地引擎（向后兼容旧行为）。
    - ``api_key_env``：key 从该环境变量读（**不写进仓库**），如 ``MIMO_KEY``。
    """

    model: str = ""
    provider: str = "openai"
    api_base: str = ""
    api_key_env: str = ""


@dataclass
class AppConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    middleware: MiddlewareConfig = field(default_factory=MiddlewareConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    user_sim: UserSimConfig = field(default_factory=UserSimConfig)
    # 从 yaml 文件 stem 填（如 "prefix_cache"），用于 run 目录命名
    config_name: str = ""


def _build_engine(data: dict[str, Any]) -> EngineConfig:
    kvt = data.get("kv_transfer") or {}
    c8_data = data.get("c8") or {}
    return EngineConfig(
        backend=data.get("backend", "vllm"),
        model=data.get("model", ""),
        extra_args=list(data.get("extra_args") or []),
        kv_transfer=KVTransferConfig(
            connector=str(kvt.get("connector", "")),
            role=str(kvt.get("role", "kv_both")),
            extra_config=dict(kvt.get("extra_config") or {}),
        ),
        c8=C8Config(
            enabled=bool(c8_data.get("enabled", False)),
            patch_qwen2=bool(c8_data.get("patch_qwen2", False)),
        ),
        priority_scheduling=bool(data.get("priority_scheduling", False)),
    )


def _build_benchmark(data: dict[str, Any]) -> BenchmarkConfig:
    return BenchmarkConfig(
        suite=data.get("suite", "tau-bench"),
        domain=data.get("domain", "retail"),
        runs=int(data.get("runs", 3)),
        seed=int(data.get("seed", 42)),
        split=data.get("split", "test"),
        data_zip=str(data.get("data_zip", "")),
        options=dict(data.get("options") or {}),
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
        target_lo=int(data.get("target_lo", 70)),
        target_hi=int(data.get("target_hi", 85)),
        options=dict(data.get("options") or {}),
    )


def _build_user_sim(data: Any) -> UserSimConfig:
    """解析 ``user_sim`` 段。缺省 → 空（user-sim 走本地引擎，向后兼容）。"""
    if data is None:
        return UserSimConfig()
    if not isinstance(data, dict):
        raise ConfigError(
            f"user_sim 段必须是 mapping，得到 {type(data).__name__}"
        )
    return UserSimConfig(
        model=str(data.get("model", "")),
        provider=str(data.get("provider", "openai")),
        api_base=str(data.get("api_base", "")),
        api_key_env=str(data.get("api_key_env", "")),
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
    # 缝A（F1）：C8 仅 Ascend 可用
    if e.c8.enabled and e.backend != "vllm-ascend":
        raise ConfigError("engine.c8.enabled 需要 backend=vllm-ascend（C8 是 Ascend 专属）")
    # 缝C（统一）：kv_transfer connector/role 白名单（LMCache / SimpleCPUOffload 二选一）
    kvt = e.kv_transfer
    if kvt.connector and kvt.connector not in _KV_CONNECTORS:
        raise ConfigError(f"engine.kv_transfer.connector={kvt.connector!r} 不在白名单 {_KV_CONNECTORS}")
    if kvt.role not in _KV_ROLES:
        raise ConfigError(f"engine.kv_transfer.role={kvt.role!r} 不在白名单 {_KV_ROLES}")

    b = cfg.benchmark
    if b.suite not in _SUITES:
        raise ConfigError(f"benchmark.suite={b.suite!r} 不在白名单 {_SUITES}")
    # longbench：domain 是数据集名（如 2wikimqa），不在 retail/airline 白名单。
    # 仅 tau-bench/agentbench 校验 domain 白名单；longbench 的 data_zip 由 adapter 运行时校验
    # （这样 preset 可留空 data_zip、由 CLI --data-zip 覆盖，不卡 load_config）。
    if b.suite != "longbench" and b.domain not in _DOMAINS:
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
    if s.target_lo < 0 or s.target_lo > 100:
        raise ConfigError(f"session.target_lo={s.target_lo} 必须在 0..100")
    if s.target_hi < s.target_lo:
        raise ConfigError(f"session.target_hi={s.target_hi} 必须 >= target_lo={s.target_lo}")


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
        user_sim=_build_user_sim(raw.get("user_sim")),
        config_name=p.stem,
    )
    validate(cfg)
    return cfg
