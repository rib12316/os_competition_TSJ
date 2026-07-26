"""suite registry 测试（get_adapter dispatch，纯函数）。"""

from __future__ import annotations

import pytest

from agent_mem.bench.tasks.registry import (
    RunContext,
    SuiteAdapter,
    get_adapter,
    registered_suites,
)
from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, MetricsConfig


def test_get_adapter_returns_tau():
    a = get_adapter("tau-bench")
    assert isinstance(a, SuiteAdapter)
    assert a.suite == "tau-bench"


def test_get_adapter_returns_longbench():
    a = get_adapter("longbench")
    assert a.suite == "longbench"


def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError, match="未知 suite"):
        get_adapter("bogus")


def test_registered_suites_has_both():
    suites = registered_suites()
    assert "tau-bench" in suites and "longbench" in suites


def test_run_context_carries_tau_and_longbench_fields():
    cfg = AppConfig(
        engine=EngineConfig(backend="vllm", model="m"),
        benchmark=BenchmarkConfig(suite="longbench"),
        metrics=MetricsConfig(),
    )
    ctx = RunContext(
        cfg=cfg, engine_url="http://x/v1", model="m",
        user_model="gpt-4o",          # tau
        longbench_system_prompt="X",  # longbench
        priority_fn=lambda: 5,        # F5
    )
    assert ctx.user_model == "gpt-4o"
    assert ctx.longbench_system_prompt == "X"
    assert ctx.priority_fn() == 5
    assert ctx.engine_url == "http://x/v1"
