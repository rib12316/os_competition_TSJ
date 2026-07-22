"""缝D · 上下文中间件包（F2 Prompt 压缩 / F3 工具数据 lazy-load）。

核心契约在 :mod:`agent_mem.middleware.base`：::

    Middleware.transform_messages()    # 发引擎前变换 messages（F2 压缩）
    Middleware.intercept_tool_result() # 拦工具返回值（F3 lazy-load）

本包提供：

- :class:`MiddlewareContext` / :class:`Middleware` / :class:`BaseMiddleware` /
  :class:`MiddlewareStack`（见 ``base.py``）
- :class:`NoOpMiddleware`：identity 默认实现（agent loop 无中间件时的零成本占位）
- **注册表** :func:`register` / :func:`build_middlewares`：按 yaml 里的名字构造
  中间件链。F2/F3 实现各自子类后 ``register("compress", CompressMiddleware)``
  即可被 ``middleware.active: [compress]`` 激活，**不改 agent loop 代码**。

参考实现见 :mod:`agent_mem.middleware.examples`（一个工具结果截断器，演示子类化
模式，不进默认注册表，避免与 F3 的正式设计冲突）。
"""

from __future__ import annotations

from typing import Any

from agent_mem.middleware.base import (
    BaseMiddleware,
    Middleware,
    MiddlewareContext,
    MiddlewareStack,
)
from agent_mem.middleware.compress import CompressMiddleware

__all__ = [
    "BaseMiddleware",
    "Middleware",
    "MiddlewareContext",
    "MiddlewareStack",
    "NoOpMiddleware",
    "CompressMiddleware",
    "registry",
    "register",
    "unregister",
    "build_middlewares",
    "middlewares_from_config",
]


class NoOpMiddleware(BaseMiddleware):
    """identity 中间件：两个钩子都原样返回。空 ``MiddlewareStack`` 的等价物。"""

    name = "noop"


# ---- 注册表：名字 → 中间件类（F2/F3 实现后在此注册）----
# F2 Prompt 压缩已注册；F3 lazy-load 待补 "lazyload"。
_REGISTRY: dict[str, type[BaseMiddleware]] = {
    "noop": NoOpMiddleware,
    "compress": CompressMiddleware,
}


def registry() -> dict[str, type[BaseMiddleware]]:
    """返回注册表快照（名字 → 类）。"""
    return dict(_REGISTRY)


def register(name: str, cls: type[BaseMiddleware]) -> None:
    """注册一个中间件类到 ``name``（覆盖同名）。供 F2/F3 feature 模块调用。"""
    if not name:
        raise ValueError("中间件注册名不能为空")
    _REGISTRY[name] = cls


def unregister(name: str) -> None:
    """从注册表移除 ``name``（不存在则 no-op）。测试隔离用。"""
    _REGISTRY.pop(name, None)


def build_middlewares(
    active: list[str] | None,
    options: dict[str, dict[str, Any]] | None = None,
) -> MiddlewareStack:
    """按名字列表构造 :class:`MiddlewareStack`。

    - ``active``：yaml ``middleware.active`` 里的名字（如 ``["compress"]``）；
      ``None`` / 空列表 → 空 stack（identity，agent loop 无开销）。
    - ``options``：每个中间件的构造 kwargs，键为名字，值为 dict；未给出用默认构造。

    未知名抛 :class:`KeyError`（并列出可选名），让配错尽早暴露。
    """
    if not active:
        return MiddlewareStack([])
    options = options or {}
    mws: list[Middleware] = []
    for n in active:
        n = n.strip()
        if n not in _REGISTRY:
            raise KeyError(
                f"未知中间件 {n!r}，已注册：{sorted(_REGISTRY)}。"
                f"（F2/F3 实现后用 register() 注册）"
            )
        mws.append(_REGISTRY[n](**options.get(n, {})))
    return MiddlewareStack(mws)


def middlewares_from_config(cfg: Any) -> MiddlewareStack:
    """从 :class:`AppConfig` 的 ``middleware`` 段构造 stack（薄封装）。

    ``cfg.middleware.active`` 为空 → 空 stack（identity）。供 benchmark runner /
    CLI 在启动 agent 前一次性构造。
    """
    mw = cfg.middleware
    return build_middlewares(mw.active, mw.options)
