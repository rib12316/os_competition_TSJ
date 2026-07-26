"""缝E · F5 并发驱动 —— HBM 准入控制 + 后台 sweep（动态 priority）+ 回调注入。

把 N 个并发 task 当作 N 个 agent session：

- 提交前 :class:`AdmissionController` 闸门（HBM 高就暂缓提交，动态降并发，防 preempt
  抖动）；
- 放行后注册 session，把 ``priority_fn`` / ``on_turn_start`` 注入该 task 的 agent
  （agent 每轮 touch 标记活跃、读当前 priority 透传给 vLLM）；
- 后台线程周期 ``sweep`` 跑 :class:`PriorityEvictionStrategy`：idle session 的 priority
  被抬高 → 开了 ``--scheduling-policy priority`` 的 vLLM 在 HBM 吃紧时**先抢占它**
  （= 动态资源回收：回收闲的、保护忙的）。

与引擎/τ-bench 解耦：``task_runner(task_id, on_turn_start, priority_fn)`` 是注入回调
（生产里是 :func:`agent_mem.bench.tasks.tau_bench_adapter.run_task`），故本类可脱离 NPU
用 fake runner + fake HBM 单测。本驱动默认**不**用 ``AdmissionController.evict_idle`` 的
"回队重跑"（会双跑 task）；真正的 KV 回收交给 vLLM priority 抢占（Phase 1 真机）。
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from agent_mem.scheduler.admission import AdmissionController
from agent_mem.scheduler.eviction import EvictionTracker
from agent_mem.scheduler.session import SessionManager
from agent_mem.scheduler.strategies import PriorityEvictionStrategy


class ConcurrentSessionDriver:
    """F5 并发驱动：HBM 准入闸门 + 后台 sweep 抬 idle priority + agent 回调注入。"""

    def __init__(
        self,
        *,
        max_workers: int,
        idle_timeout_s: float = 30.0,
        target_lo: int = 70,
        target_hi: int = 85,
        sweep_interval: float = 3.0,
        hbm_pct_fn: Callable[[], float] | None = None,
        idle_priority: int = 100,
        active_priority: int = 0,
        priority_mode: str = "idle",
        max_steps: int = 25,
        think_time_profiles: list[tuple[float, float]] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.tracker = EvictionTracker()
        self.mgr = SessionManager(clock=clock)
        self.strategy = PriorityEvictionStrategy(
            idle_timeout_s=idle_timeout_s, idle_priority=idle_priority,
            active_priority=active_priority, tracker=self.tracker,
        )
        self.ctrl = AdmissionController(
            target_lo=target_lo, target_hi=target_hi, min_workers=1,
            max_workers=max_workers, idle_timeout_s=idle_timeout_s,
            interval=sweep_interval, hbm_pct_fn=hbm_pct_fn, tracker=self.tracker,
        )
        self.priority_mode = priority_mode  # "idle" / "progress" / "combined"
        self.max_steps = max_steps
        self._think_time_profiles = think_time_profiles or []  # [(lo,hi),...] 用户活跃度 profile（round-robin 分配）
        self._sleep_fn = sleep_fn  # think-time 注入（默认 time.sleep；测试可注入 recorder）
        self._clock = clock  # EWMA 时间戳（生产=time.monotonic；测试可注入 fake clock）
        self._stop = threading.Event()
        self._sweep_thread: threading.Thread | None = None

    @staticmethod
    def _sid(task_id: int) -> str:
        return f"tau-{task_id}"  # 与 TauBenchAgent 的 MiddlewareContext session_id 对齐

    def _start_sweep(self) -> None:
        interval = self.ctrl.interval

        def _loop() -> None:
            while not self._stop.wait(interval):
                try:
                    self.mgr.sweep(self.strategy)
                except Exception:  # noqa: BLE001 — sweep 失败不杀驱动
                    pass

        self._sweep_thread = threading.Thread(target=_loop, daemon=True, name="f5-sweep")
        self._sweep_thread.start()

    def _make_callbacks(self, task_id: int) -> tuple[Callable[..., None], Callable[[], int]]:
        sid = self._sid(task_id)
        active = self.strategy.active_priority
        max_steps = max(self.max_steps, 1)
        # 用户活跃度 profile（round-robin 分配）：模拟真实用户的异质交互节奏
        think_range = None
        if self._think_time_profiles:
            think_range = self._think_time_profiles[task_id % len(self._think_time_profiles)]

        def on_turn_start(step: int = 0) -> None:
            # 模拟用户 think-time（step>1 时注入 inter-turn 间隔；sleep 在 touch 之前）
            if step > 1 and think_range:
                self._sleep_fn(random.uniform(*think_range))
            s = self.mgr.touch(sid)
            s.metadata["step"] = step
            # EWMA 交互间隔（recency 信号）：不随 touch 清零，记录 session 一贯节奏
            if self.priority_mode in ("combined", "idle"):
                now = self._clock()
                last_req = s.metadata.get("last_req_time")
                if last_req is not None:
                    gap = max(0.0, now - last_req)
                    prev = s.metadata.get("ewma_gap", gap)
                    s.metadata["ewma_gap"] = 0.7 * prev + 0.3 * gap
                s.metadata["last_req_time"] = now
            if self.priority_mode == "idle":
                self.strategy.mark_active(s)

        def priority_fn() -> int:
            s = self.mgr.get(sid)
            if s is None:
                return active
            if self.priority_mode == "combined":
                # Recency 分量 (0-70)：EWMA gap 短=活跃=低分（保护）
                ewma = s.metadata.get("ewma_gap", 0.0)
                recency = min(70, int(ewma * 3.5))
                # Progress 分量 (0-30)：步数多=近完成=低分（保护/SRTF）
                step = s.metadata.get("step", 0)
                progress = max(0, 30 - int(step * 1.2))
                return recency + progress
            if self.priority_mode == "progress":
                step = s.metadata.get("step", 0)
                return max(0, round((1 - step / max_steps) * 100))
            return s.metadata.get("priority", active)

        return on_turn_start, priority_fn

    def run(self, task_ids: list[int], task_runner: Callable[..., object]) -> list:
        """并发跑 tasks，带 HBM 准入闸门 + 后台 sweep。

        ``task_runner(task_id, on_turn_start, priority_fn) -> result`` 由调用方注入
        （生产用 run_task；测试用 fake）。返回结果按 ``task_ids`` 原序（均会跑完）。
        """
        results: dict[int, object] = {}
        pending = list(task_ids)
        futures: dict = {}
        interval = self.ctrl.interval
        self._start_sweep()
        try:
            with ThreadPoolExecutor(max_workers=self.ctrl.max_workers) as ex:
                while pending or futures:
                    # 准入提交：HBM 允许→放；HBM 高但池空→保底放一个（防死锁）
                    while pending:
                        pool_empty = not futures
                        if not self.ctrl.should_admit() and not pool_empty:
                            break
                        tid = pending.pop(0)
                        sid = self._sid(tid)
                        self.mgr.register(sid)
                        self.ctrl.admit(sid)
                        on_turn, pfn = self._make_callbacks(tid)
                        fut = ex.submit(task_runner, tid, on_turn, pfn)
                        futures[fut] = tid
                    # 等任一完成（最多 interval），让闸门/sweep 周期推进
                    if futures:
                        done, _ = wait(list(futures), timeout=interval, return_when=FIRST_COMPLETED)
                        for fut in done:
                            tid = futures.pop(fut)
                            try:
                                results[tid] = fut.result()
                            except Exception as e:  # noqa: BLE001 — 单任务异常不杀驱动
                                results[tid] = e
                            self.ctrl.release(self._sid(tid))
                    elif pending:
                        # 全被闸门挡且无在途 → 短等重试（HBM 降下后再放）
                        self._stop.wait(min(interval, 0.5))
        finally:
            self._stop.set()
            if self._sweep_thread is not None:
                self._sweep_thread.join(timeout=2.0)

        return [results[tid] for tid in task_ids if tid in results]

    def snapshot(self) -> dict:
        """供 metrics / 实时监控消费的瞬时快照（并发度、HBM、eviction 命中率）。"""
        return {**self.ctrl.stats, "eviction_tracker": self.tracker.snapshot()}
