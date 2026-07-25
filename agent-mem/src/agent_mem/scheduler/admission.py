"""F5 · 会话准入控制器 —— HBM 驱动的动态并发 + idle 驱逐。

基于 HBM 使用率调整并发 session 数，HBM 超过阈值时驱逐 idle 最久的 session。

设计要点（v2 升级）：
- **HBM 读取可注入**：``hbm_pct_fn``（返回 HBM 已用 %，读不到返回负数）。缺省走
  :func:`_read_hbm_pct`（npu-smi）；测试可注入固定序列。也可用
  :func:`hbm_pct_from_backend` 把 :class:`agent_mem.bench.mem_sampler.MemBackend`
  包成百分数读取器（复用现成 TorchNpu/NpuSmi/Fake 后端，不再硬编码 npu-smi）。
- **线程安全**：``RLock`` 保护注册表与计数（并发 task 线程 touch/release + driver
  线程 admit/evict）。HBM 读取（可能 subprocess）在锁外做。
- **EvictionTracker 上报**：``evict_idle`` 命中时上报一次回收（``was_idle=True``）。
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable

from agent_mem.scheduler.eviction import EvictionTracker


def _read_hbm_pct() -> float:
    """读 npu-smi 获取 NPU HBM 使用率（%）。无 NPU 返回 -1。"""
    try:
        out = subprocess.check_output(
            ["npu-smi", "info", "-t", "usages", "-i", "0", "-c", "0"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.split("\n"):
            if "HBM Usage rate" in line:
                return float(line.split(":")[-1].strip().replace("%", ""))
    except Exception:
        pass
    return -1.0


def hbm_pct_from_backend(backend: object, total_mb: float) -> Callable[[], float]:
    """把 ``MemBackend``（返回已用 MB）包成"HBM 已用 %"读取器。

    复用 :mod:`agent_mem.bench.mem_sampler` 的后端（TorchNpuBackend / NpuSmiBackend /
    FakeBackend），免得本模块硬编码 npu-smi。设备读失败时返回 -1.0（与
    :func:`_read_hbm_pct` 语义一致）。
    """

    def _read() -> float:
        try:
            used = float(backend.used_mb())  # type: ignore[attr-defined]
        except Exception:
            return -1.0
        if total_mb <= 0:
            return -1.0
        return max(0.0, min(100.0, used / total_mb * 100.0))

    return _read


class SessionRecord:
    __slots__ = ("session_id", "last_active", "created_at")

    def __init__(self, sid: str):
        self.session_id = sid
        self.last_active = time.monotonic()
        self.created_at = self.last_active


class AdmissionController:
    """HBM 驱动的动态并发控制 + idle 驱逐。线程安全。

    用法::

        ctrl = AdmissionController(target_lo=70, target_hi=85, max_workers=6)
        while tasks_remaining:
            if ctrl.should_admit():
                worker = ctrl.admit("session-1")
                run_one_task(worker)
            else:
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
        *,
        hbm_pct_fn: Callable[[], float] | None = None,
        tracker: EvictionTracker | None = None,
    ):
        self.target_lo = target_lo
        self.target_hi = target_hi
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.idle_timeout_s = idle_timeout_s
        self.interval = interval
        self._hbm_pct_fn = hbm_pct_fn
        self._tracker = tracker

        self._sessions: dict[str, SessionRecord] = {}
        self._current_workers = 0
        self._last_hbm: float = 0.0
        self._eviction_count = 0
        self._admit_count = 0
        self._lock = threading.RLock()

    # ---- HBM 读取 ----

    @property
    def hbm_pct(self) -> float:
        """读 HBM 使用率（%）；读不到返回 -1。读取在锁外（可能 subprocess）。"""
        fn = self._hbm_pct_fn or _read_hbm_pct
        try:
            return float(fn())
        except Exception:
            return -1.0

    # ---- session 注册 / 活跃追踪 ----

    def admit(self, session_id: str) -> str:
        """准入一个 session，记录活跃时间。"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionRecord(session_id)
            self._sessions[session_id].last_active = time.monotonic()
            self._current_workers += 1
            self._admit_count += 1
            return session_id

    def release(self, session_id: str) -> None:
        """session 执行完毕，从控制器注销。"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if self._current_workers > 0:
                self._current_workers -= 1

    def touch(self, session_id: str) -> None:
        """标记 session 活跃（中间件每步调用）。"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_active = time.monotonic()

    # ---- 准入决策 ----

    def should_admit(self) -> bool:
        """是否允许新 session 进来。"""
        hbm = self.hbm_pct  # 锁外读（可能 subprocess）
        with self._lock:
            self._last_hbm = hbm
            if hbm < 0:  # 读不到 HBM → 放行（受 max_workers 上限约束）
                return self._current_workers < self.max_workers
            if hbm > self.target_hi:
                return False  # HBM 太高，不放
            if hbm < self.target_lo:
                return self._current_workers < self.max_workers  # 有空间，放
            # HBM 在中间 → 保持当前并发度，不增不减
            return False

    @property
    def current_workers(self) -> int:
        with self._lock:
            return self._current_workers

    # ---- idle 驱逐 ----

    def _idle_seconds(self, sid: str) -> float:
        rec = self._sessions.get(sid)
        if rec is None:
            return float("inf")
        return max(0.0, time.monotonic() - rec.last_active)

    def evict_idle(self) -> str | None:
        """驱逐 idle 最久的 session，返回其 id；无可驱逐返回 None。"""
        with self._lock:
            if not self._sessions:
                return None
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
                if self._tracker is not None:
                    self._tracker.record_eviction(
                        most_idle_sid, was_idle=True, reason="admit-evict"
                    )
                return most_idle_sid
            return None

    # ---- 统计 ----

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "current_workers": self._current_workers,
                "total_sessions": len(self._sessions),
                "evictions": self._eviction_count,
                "admits": self._admit_count,
                "hbm_pct": self._last_hbm,
            }
