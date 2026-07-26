"""F5 用户活跃度仿真测试（think-time + EWMA recency + combined priority）。

纯 Python、无 NPU：用 fake clock + sleep recorder 验证 ConcurrentSessionDriver
的频率仿真逻辑（从 f5 WIP 移植到 v1）。
"""

from __future__ import annotations

from agent_mem.scheduler.driver import ConcurrentSessionDriver


class _Clock:
    """可控单调时钟（测试用）。"""

    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---- think-time 注入 ----


def test_think_time_fires_only_after_first_turn():
    sleeps: list[float] = []
    d = ConcurrentSessionDriver(
        max_workers=1, priority_mode="idle",
        think_time_profiles=[(1.0, 3.0)], sleep_fn=sleeps.append,
    )
    on_turn, _ = d._make_callbacks(0)
    on_turn(step=1)              # 首轮不 sleep（inter-turn 才注入）
    assert sleeps == []
    on_turn(step=2)              # 第二轮起注入 think-time
    assert len(sleeps) == 1 and 1.0 <= sleeps[0] <= 3.0


def test_think_time_round_robin_by_task_id():
    sleeps: list[float] = []
    d = ConcurrentSessionDriver(
        max_workers=2, priority_mode="idle",
        think_time_profiles=[(1.0, 2.0), (5.0, 6.0)], sleep_fn=sleeps.append,
    )
    on0, _ = d._make_callbacks(0)   # → profile[0]
    on1, _ = d._make_callbacks(1)   # → profile[1]
    on2, _ = d._make_callbacks(2)   # → profile[0]（2 % 2 == 0，round-robin 回绕）
    on0(step=2)
    on1(step=2)
    on2(step=2)
    assert 1.0 <= sleeps[0] <= 2.0   # task 0
    assert 5.0 <= sleeps[1] <= 6.0   # task 1
    assert 1.0 <= sleeps[2] <= 2.0   # task 2 → profile[0]


# ---- EWMA recency 信号 ----


def test_ewma_gap_tracks_interaction_interval():
    clk = _Clock(0.0)
    d = ConcurrentSessionDriver(
        max_workers=1, priority_mode="combined", think_time_profiles=[(1.0, 1.0)],
        sleep_fn=lambda _: None, clock=clk,
    )
    on_turn, _ = d._make_callbacks(0)
    on_turn(step=1)              # last_req_time = 0（首次，无 gap）
    clk.advance(10.0)
    on_turn(step=2)              # gap=10 → ewma = 0.7*10 + 0.3*10 = 10
    clk.advance(5.0)
    on_turn(step=3)              # gap=5  → ewma = 0.7*10 + 0.3*5 = 8.5
    s = d.mgr.get("tau-0")
    assert abs(s.metadata["ewma_gap"] - 8.5) < 1e-9
    assert s.metadata["last_req_time"] == 15.0


# ---- combined priority（recency + progress，SRTF）----


def test_combined_priority_is_recency_plus_progress():
    clk = _Clock(0.0)
    d = ConcurrentSessionDriver(
        max_workers=1, priority_mode="combined", think_time_profiles=[(1.0, 1.0)],
        sleep_fn=lambda _: None, clock=clk,
    )
    on_turn, pfn = d._make_callbacks(0)
    on_turn(step=1)
    clk.advance(10.0)
    on_turn(step=2)
    clk.advance(5.0)
    on_turn(step=3)              # ewma_gap=8.5, step=3
    # recency = min(70, int(8.5*3.5)) = 29 ; progress = max(0, 30 - int(3*1.2)) = 27
    assert pfn() == 29 + 27


def test_combined_protects_active_near_done_session():
    """EWMA gap 短（活跃）+ 步数多（近完成）→ 低分（受保护）。"""
    clk = _Clock(0.0)
    d = ConcurrentSessionDriver(
        max_workers=1, priority_mode="combined", sleep_fn=lambda _: None, clock=clk,
    )
    on_turn, pfn = d._make_callbacks(0)
    on_turn(step=1)
    clk.advance(2.0)             # 短 gap=活跃
    on_turn(step=20)             # 多步=近完成
    score = pfn()
    assert score < 30            # 活跃 + 近完成 → 低分（保护）
