"""缝E Session 生命周期测试（纯 Python，无 NPU；机制用 fake 回调）。"""

from __future__ import annotations

import threading

import pytest

from agent_mem.bench.mem_sampler import FakeBackend
from agent_mem.scheduler import (
    AdmissionController,
    CheckpointStrategy,
    EvictionTracker,
    IdleEvictionStrategy,
    NoOpStrategy,
    PriorityEvictionStrategy,
    SessionManager,
    SessionState,
)
from agent_mem.scheduler.admission import hbm_pct_from_backend
from agent_mem.scheduler.driver import ConcurrentSessionDriver


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


# ---- EvictionTracker (F5) ----


def test_eviction_tracker_counts_and_rate():
    t = EvictionTracker()
    t.record_eviction("a", was_idle=True)
    t.record_eviction("b", was_idle=False, reason="manual")
    t.record_eviction("c", was_idle=True)
    snap = t.snapshot()
    assert snap["evictions"] == 3
    assert snap["idle_hits"] == 2
    assert snap["idle_hit_rate"] == 2 / 3
    assert len(t.events()) == 3


# ---- SessionManager 线程安全 ----


def test_session_manager_concurrent_touch_is_safe():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    n_threads, n_each = 8, 200

    def worker(i):
        for _ in range(n_each):
            mgr.touch(f"s{i}")
            mgr.idle_seconds(f"s{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # 8 个 session 都注册成功（线程安全下没有丢）
    assert len(mgr) == n_threads
    assert all(f"s{i}" in mgr for i in range(n_threads))


# ---- PriorityEvictionStrategy (F5 动态优先级回收) ----


def test_priority_evict_raises_priority_when_idle():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    tr = EvictionTracker()
    s = mgr.register("s1")
    strat = PriorityEvictionStrategy(idle_timeout_s=10, tracker=tr)

    mgr.sweep(strat)  # active → 不抬、不计
    assert "priority" not in s.metadata
    assert tr.snapshot()["evictions"] == 0

    clk.advance(11)
    mgr.sweep(strat)  # idle → 抬到 100，计 1 次
    assert s.metadata["priority"] == 100
    assert tr.snapshot()["evictions"] == 1

    mgr.sweep(strat)  # 仍 idle → 不重复计数
    assert tr.snapshot()["evictions"] == 1

    mgr.touch("s1")  # 重新活跃
    strat.mark_active(mgr.get("s1"))
    clk.advance(1)
    mgr.sweep(strat)  # active → 回落 0
    assert s.metadata["priority"] == 0


def test_priority_evict_re_idling_counts_again():
    clk = _Clock()
    mgr = SessionManager(clock=clk)
    tr = EvictionTracker()
    s = mgr.register("s1")
    strat = PriorityEvictionStrategy(idle_timeout_s=5, tracker=tr)

    clk.advance(6)
    mgr.sweep(strat)
    assert tr.snapshot()["evictions"] == 1
    mgr.touch("s1")
    strat.mark_active(s)
    clk.advance(6)
    mgr.sweep(strat)
    assert tr.snapshot()["evictions"] == 2  # 再次 idle 再计一次


def test_priority_evict_rejects_bad_args():
    with pytest.raises(ValueError):
        PriorityEvictionStrategy(idle_timeout_s=-1)
    with pytest.raises(ValueError):
        PriorityEvictionStrategy(idle_timeout_s=1, idle_priority=0, active_priority=0)


# ---- AdmissionController（注入 HBM 读取 + tracker）----


def test_admission_controller_gating_with_injected_hbm():
    hbm = [0.0]
    ctrl = AdmissionController(target_lo=70, target_hi=85, max_workers=4, hbm_pct_fn=lambda: hbm[0])
    assert ctrl.should_admit() is True       # 低 HBM，有空位
    ctrl.admit("s1")
    hbm[0] = 90.0
    assert ctrl.should_admit() is False      # 高 HBM 拒绝
    hbm[0] = 75.0
    assert ctrl.should_admit() is False      # 中间 hold（不增不减）


def test_admission_controller_evict_fires_tracker():
    tr = EvictionTracker()
    ctrl = AdmissionController(idle_timeout_s=0.0, hbm_pct_fn=lambda: 0.0, tracker=tr)
    ctrl.admit("s1")
    import time
    time.sleep(0.01)
    assert ctrl.evict_idle() == "s1"
    assert tr.snapshot()["idle_hits"] == 1


def test_hbm_pct_from_backend_wraps_fakebackend():
    fn = hbm_pct_from_backend(FakeBackend([32768]), 65536.0)
    assert fn() == 50.0
    assert hbm_pct_from_backend(FakeBackend([32768]), 0.0)() == -1.0  # total<=0 → -1


# ---- ConcurrentSessionDriver（fake runner + fake HBM，端到端接线）----


def test_driver_runs_all_and_injects_callbacks():
    seen: dict[int, int] = {}

    def runner(tid, on_turn, pfn):
        on_turn()              # 模拟 agent 每轮开始
        seen[tid] = pfn()      # on_turn 后应读到 active priority 0
        return f"r{tid}"

    d = ConcurrentSessionDriver(
        max_workers=4, idle_timeout_s=30, hbm_pct_fn=lambda: 10.0, sweep_interval=10
    )
    out = d.run([0, 1, 2, 3], runner)
    assert out == ["r0", "r1", "r2", "r3"]
    assert seen == {0: 0, 1: 0, 2: 0, 3: 0}
    assert d.snapshot()["admits"] == 4


def test_driver_admission_throttles_under_high_hbm():
    order: list[int] = []

    def runner(tid, on_turn, pfn):
        on_turn()
        order.append(tid)
        return tid

    d = ConcurrentSessionDriver(
        max_workers=4, idle_timeout_s=30, hbm_pct_fn=lambda: 90.0, sweep_interval=10
    )
    out = d.run([0, 1, 2, 3], runner)
    assert out == [0, 1, 2, 3]        # 结果仍按原序
    assert order == [0, 1, 2, 3]      # 高 HBM → 串行（pool 空时保底放一个，防死锁）
