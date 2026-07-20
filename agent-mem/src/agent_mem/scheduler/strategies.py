"""缝E · Session 生命周期策略（F5 idle eviction / F6 KV checkpoint-恢复）。

策略 vs 机制的边界（对齐总方案 §7 的「干净分层」）：

- **策略**（本模块）：决定**何时**搬 KV（idle 多久？何时 checkpoint？）。
- **机制**（缝C 的 offload connector）：决定**怎么**搬（NPU↔CPU 的真传输，
  vllm-ascend 的 ``simple_kv_offload`` / ``SimpleCPUOffloadConnector``）。

机制以**回调**注入策略，故策略可脱离 NPU 单测（fake 回调记录调用即可）。真机联调
时把 ``offload_fn`` 换成 connector 的 ``save_kv_layer`` 即可——策略代码不动。

F5 / F6 各自独立类、独立 yaml、独立 before/after（见 ``configs/f5-evict.yaml`` /
``configs/f6-checkpoint.yaml``）。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from agent_mem.scheduler.session import Session, SessionManager, SessionState

# 机制回调签名（由 缝C connector 注入；策略不解释其内部）：
#   offload_fn(session) -> handle     把显存 KV 搬到 CPU/盘，返回句柄
#   restore_fn(handle) -> Any         按句柄把 KV 换回显存
#   save_fn(session) -> handle        进程退出前落盘（F6）
#   load_fn(session_id) -> handle     冷启动按 id 反向加载（F6）
OffloadFn = Callable[[Session], Any]
RestoreFn = Callable[[Any], Any]
SaveFn = Callable[[Session], Any]
LoadFn = Callable[[str], Any]


class SessionStrategy(Protocol):
    """缝E 策略契约。实现者继承 :class:`BaseStrategy` 更省事。"""

    name: str

    def on_sweep(self, session: Session, mgr: SessionManager) -> None:
        """SessionManager.sweep 每轮对每个 session 调用一次。"""
        ...


class BaseStrategy:
    """策略基类：默认 on_sweep 不做事。F5/F6 子类化它。"""

    name = "base"

    def on_sweep(self, session: Session, mgr: SessionManager) -> None:
        return None


class NoOpStrategy(BaseStrategy):
    """identity 策略：什么都不做（无 F5/F6 时的零成本默认）。"""

    name = "noop"


class IdleEvictionStrategy(BaseStrategy):
    """F5 · session idle 淘汰：idle 超阈值 → 把显存 KV offload 到 CPU。

    只写**策略**：idle 判定 + 状态迁移 + 调机制回调。真 NPU↔CPU 传输由注入的
    ``offload_fn`` 完成（真机换 connector 的 save_kv_layer）。已 offload 的 session
    不重复搬。

    ``restore_fn`` 可选：给定时，调用方可在 session 再被访问前调 :meth:`restore`
    把 KV 换回显存（本策略不在 sweep 里自动 restore——避免抖动；由 driver 显式触发）。
    """

    name = "idle-evict"

    def __init__(
        self,
        *,
        idle_timeout_s: float,
        offload_fn: OffloadFn,
        restore_fn: RestoreFn | None = None,
    ):
        if idle_timeout_s < 0:
            raise ValueError("idle_timeout_s 必须 >= 0")
        if offload_fn is None:
            raise ValueError("offload_fn 不能为 None（机制回调必填）")
        self.idle_timeout_s = idle_timeout_s
        self.offload_fn = offload_fn
        self.restore_fn = restore_fn

    def on_sweep(self, session: Session, mgr: SessionManager) -> None:
        # 只对仍在显存（ACTIVE）且 idle 超阈值的 session 搬迁
        if session.state is not SessionState.ACTIVE:
            return
        if mgr.idle_seconds(session.session_id) < self.idle_timeout_s:
            return
        session.kv_handle = self.offload_fn(session)
        session.state = SessionState.OFFLOADED

    def restore(self, session: Session) -> bool:
        """把已 offload 的 session 的 KV 换回显存（driver 在复用前显式调）。

        无 ``restore_fn`` 或 session 未 offload 时返回 False，否则 True。
        """
        if self.restore_fn is None or session.state is not SessionState.OFFLOADED:
            return False
        self.restore_fn(session.kv_handle)
        session.kv_handle = None
        session.state = SessionState.ACTIVE
        session.touch()
        return True


class CheckpointStrategy(BaseStrategy):
    """F6 · KV checkpoint/恢复：进程退出前落盘，重启冷启动按 session_id 反向加载。

    与 F5 共机制（都是搬 KV），但**功能独立**：F5 是运行时淘汰，F6 是跨重启持久化。
    本类不在 sweep 里自动 checkpoint——checkpoint 时机由 driver 显式触发
    （如 benchmark 结束 / 进程 SIGTERM）。

    落盘/加载动作由注入的 ``save_fn`` / ``load_fn`` 完成（真机换 connector 的
    ``save_kv_layer`` / SharedStorage 思路）。
    """

    name = "checkpoint"

    def __init__(self, *, save_fn: SaveFn, load_fn: LoadFn | None = None):
        if save_fn is None:
            raise ValueError("save_fn 不能为 None（机制回调必填）")
        self.save_fn = save_fn
        self.load_fn = load_fn

    def checkpoint(self, session: Session) -> Any:
        """把 session 的 KV 落盘，返回句柄并标记 CHECKPOINTED。"""
        handle = self.save_fn(session)
        session.kv_handle = handle
        session.state = SessionState.CHECKPOINTED
        return handle

    def restore(self, session_id: str) -> Any:
        """冷启动按 session_id 反向加载 KV；无 ``load_fn`` 时抛 RuntimeError。"""
        if self.load_fn is None:
            raise RuntimeError("未配置 load_fn，无法 restore")
        return self.load_fn(session_id)
