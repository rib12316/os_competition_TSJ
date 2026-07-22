"""F5 · 会话准入控制器 —— HBM 驱动的动态并发 + idle 驱逐。

基于 HBM 使用率调整并发 session 数，HBM 超过阈值时驱逐 idle 最久的 session。
"""

from __future__ import annotations

import subprocess
import time


def _read_hbm_pct() -> float:
    """读 npu-smi 获取 NPU HBM 使用率（%）。无 NPU 返回 -1。"""
    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "usages", "-i", "0", "-c", "0"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.split("\n"):
            if "HBM Usage Rate" in line:
                return float(line.split(":")[-1].strip().replace("%", ""))
    except Exception:
        pass
    return -1.0


class SessionRecord:
    __slots__ = ("session_id", "last_active", "created_at")
    def __init__(self, sid: str):
        self.session_id = sid
        self.last_active = time.monotonic()
        self.created_at = self.last_active


class AdmissionController:
    """HBM 驱动的动态并发控制 + idle 驱逐。

    用法::

        ctrl = AdmissionController(target_lo=70, target_hi=85, max_workers=6)
        while tasks_remaining:
            # 准入决策
            if ctrl.should_admit():
                worker = ctrl.admit("session-1")
                run_one_task(worker)  # 实际跑
            else:
                # 驱逐 idle 最久的 session
                sid = ctrl.evict_idle()
                if sid:
                    print(f"evicted {sid}")

            time.sleep(ctrl.interval)
    """

    def __init__(
        self,
        target_lo: int = 70,
        target_hi: int = 85,
        min_workers: int = 1,
        max_workers: int = 6,
        idle_timeout_s: float = 30.0,
        interval: float = 3.0,
    ):
        self.target_lo = target_lo
        self.target_hi = target_hi
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.idle_timeout_s = idle_timeout_s
        self.interval = interval

        self._sessions: dict[str, SessionRecord] = {}
        self._current_workers = 0
        self._last_hbm: float = 0.0
        self._eviction_count = 0
        self._admit_count = 0

    # ---- HBM 读取 ----

    @property
    def hbm_pct(self) -> float:
        self._last_hbm = _read_hbm_pct()
        return self._last_hbm

    # ---- session 注册 / 活跃追踪 ----

    def admit(self, session_id: str) -> str:
        """准入一个 session，记录活跃时间。"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionRecord(session_id)
        self._sessions[session_id].last_active = time.monotonic()
        self._current_workers += 1
        self._admit_count += 1
        return session_id

    def release(self, session_id: str) -> None:
        """session 执行完毕，从控制器注销。"""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if self._current_workers > 0:
            self._current_workers -= 1

    def touch(self, session_id: str) -> None:
        """标记 session 活跃（中间件每步调用）。"""
        if session_id in self._sessions:
            self._sessions[session_id].last_active = time.monotonic()

    # ---- 准入决策 ----

    def should_admit(self) -> bool:
        """是否允许新 session 进来。"""
        hbm = self.hbm_pct
        if hbm < 0:  # 读不到 HBM → 放行
            return self._current_workers < self.max_workers
        if hbm > self.target_hi:
            return False  # HBM 太高，不放
        if hbm < self.target_lo:
            return self._current_workers < self.max_workers  # 有空间，放
        # HBM 在中间 → 保持当前并发度，不增不减
        return self._current_workers < self._current_workers  # always False

    @property
    def current_workers(self) -> int:
        return self._current_workers

    # ---- idle 驱逐 ----

    def _idle_seconds(self, sid: str) -> float:
        rec = self._sessions.get(sid)
        if rec is None:
            return float("inf")
        return max(0.0, time.monotonic() - rec.last_active)

    def evict_idle(self) -> str | None:
        """驱逐 idle 最久的 session，返回其 id；无可驱逐返回 None。"""
        if not self._sessions:
            return None
        # 找 idle 最久且超过阈值的
        most_idle_sid: str | None = None
        most_idle_secs: float = 0.0
        for sid in list(self._sessions.keys()):
            idle_s = self._idle_seconds(sid)
            if idle_s >= self.idle_timeout_s and idle_s > most_idle_secs:
                most_idle_secs = idle_s
                most_idle_sid = sid
        if most_idle_sid is not None:
            del self._sessions[most_idle_sid]
            if self._current_workers > 0:
                self._current_workers -= 1
            self._eviction_count += 1
            return most_idle_sid
        return None

    # ---- 统计 ----

    @property
    def stats(self) -> dict:
        return {
            "current_workers": self._current_workers,
            "total_sessions": len(self._sessions),
            "evictions": self._eviction_count,
            "admits": self._admit_count,
            "hbm_pct": self._last_hbm,
        }
