"""suite-agnostic 任务 DTO（tau-bench / longbench 共享）。

刻意不 import tau_bench / openai，保持轻量——import 本模块是廉价的
（与 tau_bench_adapter 的惰性 import 纪律一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskInfo:
    """一个 benchmark 任务的声明（跨 suite）。

    - ``task_id``：序号，驱动 session_id 与结果对齐（runner 只读这个字段）。
    - ``suite``：``"tau-bench"`` | ``"longbench"``（dispatch / 报告用）。
    - ``domain``：tau→``retail``|``airline``；longbench→数据集名（如 ``2wikimqa``）。
    - ``split``：``test``|``train``|``dev``。
    - ``payload``：opaque——tau=None（按 task_id 重新加载）；longbench=该 JSONL example dict
      （因为 longbench 的任务*就是*数据行，不像 tau 可按 id 重载）。
    """

    task_id: int
    suite: str = ""
    domain: str = ""
    split: str = "test"
    payload: Any = None
