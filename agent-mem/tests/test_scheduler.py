"""缝E Session 生命周期测试（纯 Python，无 NPU；机制用 fake 回调）。"""

from __future__ import annotations

import pytest

from agent_mem.scheduler import (
    CheckpointStrategy,
    IdleEvictionStrategy,
    NoOpStrategy,
    SessionManager,
    SessionState,
)


class _Clock:
    """可控单调时钟（测试用）。"""

    def __init__(self, t0=0.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---- Session / SessionManager ----


def test_register_and_get():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    s = mgr.register("s1", user="alice")
    assert s.session_id == "s1"
    assert s.state is SessionState.ACTIVE
    assert mgr.get("s1") is s
    assert "s1" in mgr
    assert len(mgr) == 1


def test_idle_seconds_with_clock():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    mgr.register("s1")
    clk.advance(5)
    assert mgr.idle_seconds("s1") == 5.0
    mgr.touch("s1")
    assert mgr.idle_seconds("s1") == 0.0
    assert mgr.idle_seconds("missing") == float("inf")


def test_idle_sessions_filter():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    mgr.register("a")
    clk.advance(2)
    mgr.register("b")  # b 刚活跃
    clk.advance(10)
    idle = mgr.idle_sessions(idle_timeout_s=5)
    ids = {s.session_id for s in idle}
    assert ids == {"a", "b"}  # 都 idle >= 5（a=12, b=10）


# ---- NoOpStrategy ----


def test_noop_sweep_changes_nothing():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    mgr.register("s1")
    touched = mgr.sweep(NoOpStrategy())
    assert touched == []


# ---- IdleEvictionStrategy (F5) ----


def test_idle_evict_offloads_after_timeout():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    s = mgr.register("s1")
    offloaded = []

    strat = IdleEvictionStrategy(
        idle_timeout_s=10, offload_fn=lambda sess: offloaded.append(sess.session_id) or "handle-1"
    )
    # 刚注册（idle=0）→ 不搬
    mgr.sweep(strat)
    assert s.state is SessionState.ACTIVE
    assert offloaded == []

    clk.advance(9)  # 还没到 10
    mgr.sweep(strat)
    assert s.state is SessionState.ACTIVE

    clk.advance(2)  # idle=11 ≥ 10 → 搬
    touched = mgr.sweep(strat)
    assert s.state is SessionState.OFFLOADED
    assert s.kv_handle == "handle-1"
    assert offloaded == ["s1"]
    assert touched == [s]


def test_idle_evict_does_not_re_offload():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    s = mgr.register("s1")
    calls = []
    strat = IdleEvictionStrategy(idle_timeout_s=1, offload_fn=lambda sess: calls.append(1))
    clk.advance(5)
    mgr.sweep(strat)
    assert s.state is SessionState.OFFLOADED
    mgr.sweep(strat)  # 再 sweep，已是 OFFLOADED
    mgr.sweep(strat)
    assert len(calls) == 1  # 只搬一次


def test_idle_evict_restore():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    s = mgr.register("s1")
    restored = []
    strat = IdleEvictionStrategy(
        idle_timeout_s=1,
        offload_fn=lambda sess: "h",
        restore_fn=lambda h: restored.append(h),
    )
    clk.advance(5)
    mgr.sweep(strat)
    assert s.state is SessionState.OFFLOADED
    assert strat.restore(s) is True
    assert s.state is SessionState.ACTIVE
    assert restored == ["h"]
    # 再次 restore（已 ACTIVE）→ False
    assert strat.restore(s) is False


def test_idle_evict_rejects_bad_args():
    with pytest.raises(ValueError):
        IdleEvictionStrategy(idle_timeout_s=-1, offload_fn=lambda s: None)
    with pytest.raises(ValueError):
        IdleEvictionStrategy(idle_timeout_s=1, offload_fn=None)


# ---- CheckpointStrategy (F6) ----


def test_checkpoint_save_and_restore():
    saved = []
    strat = CheckpointStrategy(
        save_fn=lambda sess: saved.append(sess.session_id) or "kv-blob",
        load_fn=lambda sid: f"loaded-{sid}",
    )
    from agent_mem.scheduler import Session

    s = Session(session_id="s1")
    handle = strat.checkpoint(s)
    assert handle == "kv-blob"
    assert s.state is SessionState.CHECKPOINTED
    assert saved == ["s1"]
    # 反向加载
    assert strat.restore("s1") == "loaded-s1"


def test_checkpoint_restore_without_load_fn_raises():
    strat = CheckpointStrategy(save_fn=lambda s: "h")
    with pytest.raises(RuntimeError, match="load_fn"):
        strat.restore("s1")
