"""缝D 中间件接口测试（纯 Python，无 openai / 无 NPU）。

覆盖：Context/Stack 语义、注册表/工厂、参考实现、以及 run_react 接线
（transform 发副本不改正典历史、intercept 改写进历史的内容）。
"""

from __future__ import annotations

import types

import pytest

from agent_mem.agent.react import run_react
from agent_mem.middleware import (
    BaseMiddleware,
    MiddlewareContext,
    MiddlewareStack,
    NoOpMiddleware,
    build_middlewares,
    register,
    registry,
    unregister,
)
from agent_mem.middleware.examples import ToolResultTruncator

# ---- Context / Base / Stack ----


def test_context_bump_step():
    ctx = MiddlewareContext(session_id="s1")
    assert ctx.step == 0
    assert ctx.bump_step() == 1
    assert ctx.bump_step() == 2
    assert ctx.step == 2
    assert ctx.scratch == {}


def test_base_middleware_is_identity():
    mw = BaseMiddleware()
    msgs = [{"role": "user", "content": "hi"}]
    assert mw.transform_messages(msgs, MiddlewareContext("s")) is msgs
    assert mw.intercept_tool_result("x", {}, "r", MiddlewareContext("s")) == "r"


def test_empty_stack_is_identity_and_does_not_mutate():
    stack = MiddlewareStack()
    assert stack.is_empty()
    msgs = [{"role": "user", "content": "hi"}]
    out = stack.transform_messages(msgs, MiddlewareContext("s"))
    assert out == msgs
    assert out is not msgs  # 返回副本，不改入参


def test_stack_chains_in_order():
    class _Add(BaseMiddleware):
        name = "add"

        def __init__(self, suffix):
            self.suffix = suffix

        def transform_messages(self, messages, ctx):
            return [{**m, "content": m.get("content", "") + self.suffix} for m in messages]

    stack = MiddlewareStack([_Add("-a"), _Add("-b")])
    out = stack.transform_messages(
        [{"role": "user", "content": "x"}], MiddlewareContext("s")
    )
    assert out[0]["content"] == "x-a-b"  # 顺序串联


def test_stack_intercept_pipeline():
    class _Upper(BaseMiddleware):
        name = "up"

        def intercept_tool_result(self, name, args, result, ctx):
            return result.upper()

    class _Tag(BaseMiddleware):
        name = "tag"

        def intercept_tool_result(self, name, args, result, ctx):
            return f"[{result}]"

    stack = MiddlewareStack([_Upper(), _Tag()])
    assert stack.intercept_tool_result("t", {}, "hi", MiddlewareContext("s")) == "[HI]"


# ---- 注册表 / 工厂 ----


def test_registry_has_noop_and_register():
    assert "noop" in registry()
    register("custom", NoOpMiddleware)
    try:
        assert "custom" in registry()
    finally:
        unregister("custom")


def test_build_middlewares_empty_returns_empty_stack():
    assert build_middlewares(None).is_empty()
    assert build_middlewares([]).is_empty()


def test_build_middlewares_unknown_raises():
    with pytest.raises(KeyError, match="未知中间件"):
        build_middlewares(["does-not-exist"])


def test_build_middlewares_constructs_with_options():
    register("trunc", ToolResultTruncator)
    try:
        stack = build_middlewares(["trunc"], options={"trunc": {"max_chars": 5}})
        # 注册名是 yaml key；.name 是中间件自报身份（类的 name 属性）
        assert len(stack.names) == 1
        out = stack.intercept_tool_result("t", {}, "abcdefg", MiddlewareContext("s"))
        assert out == "abcde...[truncated]"  # 证明用 max_chars=5 构造成功
    finally:
        # 不污染其他测试：恢复注册表
        unregister("trunc")


# ---- 参考实现 ----


def test_tool_result_truncator_passes_short_through():
    mw = ToolResultTruncator(max_chars=10)
    assert mw.intercept_tool_result("t", {}, "short", MiddlewareContext("s")) == "short"


def test_tool_result_truncator_rejects_bad_max():
    with pytest.raises(ValueError):
        ToolResultTruncator(max_chars=0)


# ---- run_react 接线 ----


def _msg(content=None, tool_calls=None):
    return types.SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)


def _tc(name, args):
    return types.SimpleNamespace(
        id="c1", function=types.SimpleNamespace(name=name, arguments=args)
    )


def _resp(msg):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeClient:
    """按脚本依次返回 response；记录每次 create 收到的 messages。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


def test_run_react_middleware_intercept_lands_in_history():
    """intercept_tool_result 的返回值进入正典历史。"""
    from agent_mem.agent import tools

    class _Upper(BaseMiddleware):
        name = "up"

        def intercept_tool_result(self, name, args, result, ctx):
            return result.upper()

    client = _FakeClient(
        [_resp(_msg(None, [_tc("search", '{"query":"x"}')])), _resp(_msg("done"))]
    )
    res = run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=5,
        middlewares=[_Upper()], session_id="s1",
    )
    # search 工具的 stub 结果被大写后回灌
    tool_msg = next(m for m in res.messages if m["role"] == "tool")
    assert tool_msg["content"].isupper()


def test_run_react_transform_sends_copy_canonical_intact():
    """transform_messages 只影响发给引擎的副本，正典历史保持完整。"""
    from agent_mem.agent import tools

    class _Shorten(BaseMiddleware):
        name = "short"

        def transform_messages(self, messages, ctx):
            # 只把最后一条发给引擎（模拟压缩）
            return [messages[-1]] if messages else []

    client = _FakeClient(
        [_resp(_msg(None, [_tc("search", '{"query":"x"}')])), _resp(_msg("done"))]
    )
    res = run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=5,
        middlewares=[_Shorten()],
    )
    # 引擎每次只收到 1 条（被压缩的副本）
    assert all(len(c["messages"]) == 1 for c in client.calls)
    # 但正典历史完整：user, assistant(tool), tool, assistant(done)
    roles = [m["role"] for m in res.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert res.final_text == "done"


def test_run_react_middleware_context_step_increments():
    from agent_mem.agent import tools

    seen_steps = []

    class _Probe(BaseMiddleware):
        name = "probe"

        def transform_messages(self, messages, ctx):
            seen_steps.append(ctx.step)
            return messages

    client = _FakeClient(
        [_resp(_msg(None, [_tc("search", "{}")])), _resp(_msg("done"))]
    )
    run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=5,
        middlewares=[_Probe()], session_id="s9",
    )
    assert seen_steps == [1, 2]  # 每步引擎调用前 bump
