"""缝D · 上下文中间件接口（F2 Prompt 压缩 / F3 工具数据 lazy-load 的挂载点）。

请求变换与响应观察钩子覆盖 agent loop 里的 messages、tools、模型响应和工具结果：

- :meth:`Middleware.transform_messages` —— 发引擎**前**变换 messages（F2 压缩冷历史、
  F3 注入引用占位）。**只变换发给引擎的副本，不改 agent 的正典历史**——压缩类优化
  因此可在不丢信息的前提下送短 prompt，正典历史仍完整。
- :meth:`Middleware.after_model_call` —— 引擎响应后观察本次真实
  ``usage.prompt_tokens``。只用于记账/观测，不改写模型响应。
- :meth:`Middleware.intercept_tool_result` —— 工具返回值回灌**前**拦截（F3 把长
  HTML/JSON 存外部 store，context 只留 ``<doc id=.. summary=..>``）。**会改写进入
  正典历史的内容**——这正是 lazy-load 的目的。

设计约束（实现者必读）
^^^^^^^^^^^^^^^^^^^^^^

- ``transform_messages`` 若删某条带 ``tool_calls`` 的 assistant 消息，必须连带处理
  对应 ``role=tool`` 结果（OpenAI API 要求 tool 结果引用的 tool_call 必须存在），
  否则引擎 400。NoOp 默认安全；F2 实现时自行保证配对完整。
- 中间件**不得**阻塞或重试引擎调用——只做无副作用的变换。需要状态时写进
  :attr:`MiddlewareContext.scratch`（每个 session 一份）。
- 链式组合：多个中间件按注册顺序串联（pipeline），前一个的输出是后一个的输入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class MiddlewareContext:
    """单次 ReAct 步的中间件上下文（每 session 一份，跨步复用 scratch）。

    - ``session_id``：全链路透传（为 缝E 的 F5/F6 预留）。
    - ``step``：当前 ReAct 步号（每次引擎调用 +1），中间件可据此只压冷历史。
    - ``scratch``：中间件私有状态桶（按 ``middleware.name`` 取键），避免互相污染。
    """

    session_id: str
    step: int = 0
    scratch: dict[str, Any] = field(default_factory=dict)

    def bump_step(self) -> int:
        """步号 +1，返回新步号。"""
        self.step += 1
        return self.step


@runtime_checkable
class Middleware(Protocol):
    """缝D 中间件契约。实现者继承 :class:`BaseMiddleware` 更省事（带默认 no-op）。"""

    name: str

    def transform_messages(
        self, messages: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        """发引擎前变换 messages（返回新列表，不改入参正典历史）。"""
        ...

    def transform_tools(
        self, tools: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        """发引擎前变换工具 schema（返回新列表，不改 canonical tools）。"""
        ...

    def transform_request(
        self, messages: list[dict], tools: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], list[dict]]:
        """联合变换 messages/tools；需要跨区域去重的中间件可覆盖。"""
        ...

    def intercept_tool_result(
        self, name: str, args: dict[str, Any], result: str, ctx: MiddlewareContext
    ) -> str:
        """拦工具返回值，返回回灌进正典历史的（可能改写的）结果文本。"""
        ...

    def after_model_call(
        self, prompt_tokens: int | None, ctx: MiddlewareContext
    ) -> None:
        """观察引擎返回的真实 prompt token；服务端未返回 usage 时为 ``None``。"""
        ...


class BaseMiddleware:
    """中间件基类：默认全 no-op（identity）。F2/F3 子类化它，覆盖需要的钩子。

    用法::

        class CompressMiddleware(BaseMiddleware):
            name = "compress"
            def transform_messages(self, messages, ctx):
                return compress(messages)   # 只覆盖需要的钩子
    """

    name: str = "base"

    def transform_messages(
        self, messages: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        return messages

    def transform_tools(
        self, tools: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        return tools

    def transform_request(
        self, messages: list[dict], tools: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], list[dict]]:
        return self.transform_messages(messages, ctx), self.transform_tools(tools, ctx)

    def intercept_tool_result(
        self, name: str, args: dict[str, Any], result: str, ctx: MiddlewareContext
    ) -> str:
        return result

    def after_model_call(
        self, prompt_tokens: int | None, ctx: MiddlewareContext
    ) -> None:
        return None


class MiddlewareStack:
    """有序中间件链：把多个中间件串成 pipeline。

    - :meth:`transform_messages`：依次套用，前一个输出喂后一个；入参不被改。
    - :meth:`after_model_call`：把引擎返回的真实 prompt token 广播给各中间件。
    - :meth:`intercept_tool_result`：依次套用，前一个输出喂后一个。
    - 空栈 = identity（直接返回入参），agent loop 可无脑调用。
    """

    def __init__(self, middlewares: list[Middleware] | None = None):
        self._mw: list[Middleware] = list(middlewares or [])

    @property
    def middlewares(self) -> list[Middleware]:
        return list(self._mw)

    @property
    def names(self) -> list[str]:
        return [getattr(m, "name", type(m).__name__) for m in self._mw]

    def is_empty(self) -> bool:
        return not self._mw

    def transform_messages(
        self, messages: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        out = list(messages)  # 不改入参正典历史
        for mw in self._mw:
            out = mw.transform_messages(out, ctx)
        return out

    def transform_request(
        self, messages: list[dict], tools: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], list[dict]]:
        out_messages = list(messages)
        out_tools = list(tools)
        for mw in self._mw:
            hook = getattr(mw, "transform_request", None)
            if hook is not None:
                out_messages, out_tools = hook(out_messages, out_tools, ctx)
            else:
                out_messages = mw.transform_messages(out_messages, ctx)
                transform_tools = getattr(mw, "transform_tools", None)
                if transform_tools is not None:
                    out_tools = transform_tools(out_tools, ctx)
        return out_messages, out_tools

    def intercept_tool_result(
        self, name: str, args: dict[str, Any], result: str, ctx: MiddlewareContext
    ) -> str:
        out = result
        for mw in self._mw:
            out = mw.intercept_tool_result(name, args, out, ctx)
        return out

    def after_model_call(
        self, prompt_tokens: int | None, ctx: MiddlewareContext
    ) -> None:
        for mw in self._mw:
            # 兼容已有的 duck-typed middleware；BaseMiddleware 子类天然具备该钩子。
            hook = getattr(mw, "after_model_call", None)
            if hook is not None:
                hook(prompt_tokens, ctx)
