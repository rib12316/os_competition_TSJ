"""缝D 参考实现：演示如何子类化 :class:`BaseMiddleware`。

这里的 :class:`ToolResultTruncator` 是一个**真实但极简**的工具结果截断器——
把超长工具返回值截到 ``max_chars`` 并附 ``...[truncated]``。它：

- 不依赖任何重模型 / NPU（纯字符串，可单测）；
- **不进默认注册表**（见 :mod:`agent_mem.middleware.__init__`），避免与 F3 的正式
  lazy-load 设计冲突——它只证明 ``intercept_tool_result`` 钩子端到端可用；
- 可作为 F2/F3 实现者的复制模板：另起一个 ``compress.py`` / ``lazyload.py``，
  ``register("compress", CompressMiddleware)`` 即被 yaml 激活。

正式功能（F2/F3）由各自 owner 在独立 feature 分支实现。
"""

from __future__ import annotations

from typing import Any

from agent_mem.middleware.base import BaseMiddleware, MiddlewareContext


class ToolResultTruncator(BaseMiddleware):
    """参考中间件：截断超长工具结果（演示 ``intercept_tool_result`` 钩子）。

    只动 ``intercept_tool_result``，``transform_messages`` 走默认 identity。
    """

    name = "truncator"

    def __init__(self, max_chars: int = 200):
        if max_chars <= 0:
            raise ValueError("max_chars 必须 > 0")
        self.max_chars = max_chars

    def intercept_tool_result(
        self, name: str, args: dict[str, Any], result: str, ctx: MiddlewareContext
    ) -> str:
        if len(result) <= self.max_chars:
            return result
        return result[: self.max_chars] + "...[truncated]"
