"""τ-bench 适配器测试。

纯函数（is_successful / aggregate_success / run_task_stub）不需要 τ-bench，始终跑；
``list_tasks`` 等真正访问 τ-bench 的用 ``importorskip`` 守门，CI 没装 τ-bench 时跳过。
"""

from __future__ import annotations

import pytest

from agent_mem.bench.tasks.tau_bench_adapter import (
    SUPPORTED_ENVS,
    SUPPORTED_SPLITS,
    aggregate_success,
    is_successful,
    list_tasks,
    run_task_stub,
)

# ---- 纯函数（无需 tau_bench）----


def test_is_successful_boundaries():
    assert is_successful(1.0) is True
    assert is_successful(1.0 + 1e-7) is True  # 容差内
    assert is_successful(1.0 - 1e-7) is True
    assert is_successful(0.0) is False
    assert is_successful(0.5) is False


def test_aggregate_success():
    from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult

    rs = [
        TaskRunResult(0, 1.0, True, 0.0, 0, None),
        TaskRunResult(1, 0.0, False, 0.0, 0, None),
        TaskRunResult(2, 1.0, True, 0.0, 0, None),
    ]
    assert aggregate_success(rs) == pytest.approx(2 / 3)


def test_aggregate_success_empty():
    assert aggregate_success([]) == 0.0


def test_run_task_stub_returns_placeholder():
    r = run_task_stub(7, domain="retail", split="test")
    assert r.task_id == 7
    assert r.reward == 0.0
    assert r.success is False
    assert r.error is None


def test_run_task_stub_rejects_bad_domain():
    with pytest.raises(ValueError, match="domain"):
        run_task_stub(0, domain="finance", split="test")


def test_run_task_stub_rejects_bad_split():
    with pytest.raises(ValueError, match="split"):
        run_task_stub(0, domain="retail", split="holdout")


def test_supported_lists():
    assert "retail" in SUPPORTED_ENVS and "airline" in SUPPORTED_ENVS
    assert "test" in SUPPORTED_SPLITS


# ---- 真 τ-bench 访问（importorskip 守门）----


def test_list_tasks_retail():
    pytest.importorskip("tau_bench")
    tasks = list_tasks("retail", "test")
    assert len(tasks) > 0
    t0 = tasks[0]
    assert t0.task_id == 0
    assert t0.domain == "retail"
    assert t0.split == "test"
    assert isinstance(t0.instruction, str) and t0.instruction
    assert isinstance(t0.action_names, tuple)


def test_n_tasks_matches_list():
    pytest.importorskip("tau_bench")
    from agent_mem.bench.tasks.tau_bench_adapter import n_tasks

    assert n_tasks("retail", "test") == len(list_tasks("retail", "test"))
