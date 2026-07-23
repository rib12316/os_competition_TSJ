"""config schema 加载/校验测试（覆盖三档真实 yaml）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_mem.config import (
    AppConfig,
    BenchmarkConfig,
    ConfigError,
    EngineConfig,
    LmCacheConfig,
    MetricsConfig,
    MiddlewareConfig,
    SessionConfig,
    load_config,
    validate,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


# ---- 真实三档 yaml 解析 ----


def test_load_baseline():
    cfg = load_config(CONFIGS_DIR / "baseline.yaml")
    assert cfg.config_name == "baseline"
    assert cfg.engine.backend == "vllm"
    assert cfg.engine.model == "Qwen2.5-7B-Instruct"
    assert "--no-enable-prefix-caching" in cfg.engine.extra_args
    assert cfg.engine.lmcache.enabled is False
    assert cfg.benchmark.domain == "retail"
    assert cfg.benchmark.runs == 3
    assert cfg.benchmark.seed == 42
    assert "task_success_rate" in cfg.metrics.collect


def test_load_prefix_cache_has_no_disable_flag():
    cfg = load_config(CONFIGS_DIR / "prefix_cache.yaml")
    assert cfg.engine.extra_args == []
    assert cfg.engine.lmcache.enabled is False


def test_load_optimized_enables_lmcache_and_int8():
    cfg = load_config(CONFIGS_DIR / "optimized.yaml")
    assert cfg.engine.lmcache.enabled is True
    assert cfg.engine.lmcache.config_file is not None
    # C8 int8 KV（Ascend-only）：--quantization ascend（非 no-op 的 --kv-cache-dtype int8）
    assert cfg.engine.backend == "vllm-ascend"
    assert any("quantization ascend" in a for a in cfg.engine.extra_args)
    assert not any("kv-cache-dtype" in a for a in cfg.engine.extra_args)
    assert not any("fp8" in a for a in cfg.engine.extra_args)


# ---- F 系列独立 yaml 预设（每个只开关自己的缝）----


def test_load_f1_int8():
    cfg = load_config(CONFIGS_DIR / "f1-int8.yaml")
    assert cfg.engine.backend == "vllm-ascend"
    assert cfg.engine.c8.enabled is True
    assert cfg.engine.c8.patch_qwen2 is True
    # --quantization ascend 由 build_serve_args 从 c8.enabled 注入，不在 extra_args
    assert not any("quantization" in a for a in cfg.engine.extra_args)
    assert not any("kv-cache-dtype" in a for a in cfg.engine.extra_args)


def test_c8_enabled_requires_ascend_backend():
    # c8.enabled + backend=vllm 应校验失败
    from agent_mem.config import AppConfig, validate, ConfigError
    bad = AppConfig(
        engine=EngineConfig(backend="vllm", model="m"),
        benchmark=BenchmarkConfig(), metrics=MetricsConfig(), config_name="x",
    )
    bad.engine.c8.enabled = True
    with pytest.raises(ConfigError):
        validate(bad)


def test_load_f2_compress_toggles_middleware():
    cfg = load_config(CONFIGS_DIR / "f2-compress.yaml")
    assert cfg.middleware.active == ["compress"]
    assert "compress" in cfg.middleware.options


def test_load_f3_lazyload_toggles_middleware():
    cfg = load_config(CONFIGS_DIR / "f3-lazyload.yaml")
    assert cfg.middleware.active == ["lazyload"]


def test_load_f4_lmcache_toggles_seam_c():
    cfg = load_config(CONFIGS_DIR / "f4-lmcache.yaml")
    assert cfg.engine.lmcache.enabled is True


def test_load_f5_evict_toggles_session():
    cfg = load_config(CONFIGS_DIR / "f5-evict.yaml")
    assert cfg.session.strategy == "idle-evict"
    assert cfg.session.idle_timeout_s == 60


def test_load_f6_checkpoint_toggles_session():
    cfg = load_config(CONFIGS_DIR / "f6-checkpoint.yaml")
    assert cfg.session.strategy == "checkpoint"


def test_load_f8_multi_hw_ascend_backend():
    cfg = load_config(CONFIGS_DIR / "f8-multi-hw.yaml")
    assert cfg.engine.backend == "vllm-ascend"


def test_default_split_is_test():
    # 现有 yaml 不含 split，应取默认 "test"
    for name in ("baseline", "prefix_cache", "optimized"):
        cfg = load_config(CONFIGS_DIR / f"{name}.yaml")
        assert cfg.benchmark.split == "test"


# ---- 校验白名单/取值范围 ----


def _valid_cfg() -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm", model="Qwen2.5-7B-Instruct"),
        benchmark=BenchmarkConfig(),
        metrics=MetricsConfig(),
    )


@pytest.mark.parametrize("backend", ["sglang", "vllm-ascend"])
def test_validate_accepts_known_backends(backend):
    cfg = _valid_cfg()
    cfg.engine.backend = backend
    validate(cfg)  # 不抛


def test_validate_rejects_unknown_backend():
    cfg = _valid_cfg()
    cfg.engine.backend = "triton"
    with pytest.raises(ConfigError, match="backend"):
        validate(cfg)


def test_validate_rejects_empty_model():
    cfg = _valid_cfg()
    cfg.engine.model = ""
    with pytest.raises(ConfigError, match="model"):
        validate(cfg)


def test_validate_rejects_bad_domain():
    cfg = _valid_cfg()
    cfg.benchmark.domain = "finance"
    with pytest.raises(ConfigError, match="domain"):
        validate(cfg)


def test_validate_rejects_runs_lt_one():
    cfg = _valid_cfg()
    cfg.benchmark.runs = 0
    with pytest.raises(ConfigError, match="runs"):
        validate(cfg)


def test_validate_rejects_bad_split():
    cfg = _valid_cfg()
    cfg.benchmark.split = "holdout"
    with pytest.raises(ConfigError, match="split"):
        validate(cfg)


def test_validate_rejects_empty_metrics():
    cfg = _valid_cfg()
    cfg.metrics.collect = []
    with pytest.raises(ConfigError, match="collect"):
        validate(cfg)


def test_validate_rejects_non_mapping_root(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


def test_lmcache_defaults():
    lm = LmCacheConfig()
    assert lm.enabled is False
    assert lm.config_file is None


# ---- 缝D/缝E 新增段：middleware / session ----


def test_defaults_no_middleware_no_session_strategy():
    cfg = _valid_cfg()
    assert cfg.middleware.active == []
    assert cfg.session.strategy == "noop"
    assert cfg.session.idle_timeout_s == 60.0


def test_middleware_section_parsed(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(
        "engine: {backend: vllm, model: Qwen3-0.6B}\n"
        "middleware:\n"
        "  active: [compress]\n"
        "  options: {compress: {ratio: 0.5}}\n"
        "session:\n"
        "  strategy: idle-evict\n"
        "  idle_timeout_s: 30\n",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.middleware.active == ["compress"]
    assert cfg.middleware.options == {"compress": {"ratio": 0.5}}
    assert cfg.session.strategy == "idle-evict"
    assert cfg.session.idle_timeout_s == 30


def test_validate_rejects_bad_session_strategy():
    cfg = _valid_cfg()
    cfg.session.strategy = "bogus"
    with pytest.raises(ConfigError, match="session.strategy"):
        validate(cfg)


def test_validate_rejects_negative_idle_timeout():
    cfg = _valid_cfg()
    cfg.session.strategy = "idle-evict"
    cfg.session.idle_timeout_s = -1
    with pytest.raises(ConfigError, match="idle_timeout_s"):
        validate(cfg)


def test_validate_accepts_all_known_strategies():
    for strat in ("noop", "idle-evict", "checkpoint"):
        cfg = _valid_cfg()
        cfg.session.strategy = strat
        validate(cfg)  # 不抛


def test_middleware_options_defaults():
    mw = MiddlewareConfig()
    assert mw.active == []
    assert mw.options == {}


def test_session_defaults():
    s = SessionConfig()
    assert s.strategy == "noop"
    assert s.idle_timeout_s == 60.0
