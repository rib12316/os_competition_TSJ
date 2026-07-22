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
    CompressMiddleware,
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


# ---- F2 CompressMiddleware（fake compressor，不依赖 llmlingua/真模型）----


class _FakeCompressor:
    """记录调用、返回可断言的压缩结果（替身 llmlingua.PromptCompressor）。"""

    def __init__(self):
        self.calls: list[dict] = []

    def compress_prompt(self, prompt_list, **kw):
        self.calls.append(kw)
        joined = (
            " ".join(prompt_list) if isinstance(prompt_list, (list, tuple)) else str(prompt_list)
        )
        return {
            "compressed_prompt": f"CMP({len(joined)})",
            "origin_tokens": len(joined),
            "compressed_tokens": 10,
            "ratio": "10x",
        }


def _compress_mw(**kw) -> CompressMiddleware:
    """构造一个绑了 fake 压缩器的 CompressMiddleware（绕过懒加载/真模型）。"""
    mw = CompressMiddleware(**kw)
    mw._compressor = _FakeCompressor()
    return mw


def _assert_no_orphan_tool(messages: list[dict]) -> None:
    """不变量：每条 role=tool 消息，前面必有 assistant 的 tool_calls 含其 tool_call_id。"""
    seen: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    seen.add(tc["id"])
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen, (
                f"孤立 tool 消息 tool_call_id={m.get('tool_call_id')!r} 无对应 assistant tool_call"
            )


def test_compress_bad_method_raises():
    with pytest.raises(ValueError):
        CompressMiddleware(method="nope")


def test_compress_bad_rate_raises():
    with pytest.raises(ValueError):
        CompressMiddleware(rate=0)
    with pytest.raises(ValueError):
        CompressMiddleware(rate=1.5)


def test_compress_gate_skips_short_history():
    """冷历史 < trigger_tokens：原样放行，压缩器不调用。"""
    mw = _compress_mw(trigger_tokens=100_000)  # 永不触发
    msgs = [{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}]
    out = mw.transform_messages(msgs, MiddlewareContext("s"))
    assert out == msgs
    assert mw._compressor.calls == []


def test_compress_skips_when_history_shorter_than_keep_hot():
    """非 system 消息数 ≤ keep_hot：全留。"""
    mw = _compress_mw(keep_hot=10, trigger_tokens=1)
    msgs = [{"role": "user", "content": "x" * 1000}, {"role": "assistant", "content": "y"}]
    out = mw.transform_messages(msgs, MiddlewareContext("s"))
    assert out == msgs
    assert mw._compressor.calls == []


def test_compress_compresses_cold_history_and_keeps_hot():
    mw = _compress_mw(keep_hot=2, trigger_tokens=1)  # 默认 method=longllmlingua, rate=0.5
    msgs = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "do task " * 200},
        {"role": "assistant", "content": "thinking " * 200},
        {"role": "user", "content": "more " * 200},
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "final"},
    ]
    out = mw.transform_messages(msgs, MiddlewareContext("s"))
    # system 原样
    assert out[0] == {"role": "system", "content": "policy"}
    # 紧跟一条压缩历史
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith("[compressed history]")
    # 热尾原样保留（最后 2 条）
    assert out[-2:] == [
        {"role": "user", "content": "latest question"},
        {"role": "assistant", "content": "final"},
    ]
    # 压缩器以 question-aware 调用，question = 最近一条 user
    assert mw._compressor.calls[0]["question"] == "latest question"
    assert mw._compressor.calls[0]["rate"] == 0.5
    assert mw._compressor.calls[0]["rank_method"] == "longllmlingua"
    _assert_no_orphan_tool(out)


def test_compress_snaps_hot_boundary_to_tool_group():
    """热尾从 tool 消息开始时，snap 把 caller assistant 一起拉进热尾，避免孤立。"""
    mw = _compress_mw(keep_hot=1, trigger_tokens=1)
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "q" * 400},  # cold
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r" * 400},  # keep_hot=1 时本是 hot[0]
    ]
    out = mw.transform_messages(msgs, MiddlewareContext("s"))
    assert mw._compressor.calls, "应触发压缩（cold=user 一块）"
    # assistant(tool_calls)+tool 对完整保留在热尾
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "c1" for m in out)
    _assert_no_orphan_tool(out)


def test_compress_never_produces_orphan_tool():
    """混合轨迹（热尾含完整 tool 组）压缩后无孤立 tool。"""
    mw = _compress_mw(keep_hot=2, trigger_tokens=1)
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "x" * 500},
        {"role": "assistant", "content": "t" * 500},
        {"role": "user", "content": "cold q " * 100},
        {"role": "assistant", "content": "plan",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "a", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r" * 200},
        {"role": "assistant", "content": "done"},
    ]
    out = mw.transform_messages(msgs, MiddlewareContext("s"))
    _assert_no_orphan_tool(out)


def test_compress_registered_via_build_middlewares():
    """yaml active:[compress] 能构造 CompressMiddleware 并透传 options（不需真模型）。"""
    stack = build_middlewares(
        ["compress"], options={"compress": {"rate": 0.3, "method": "llmlingua2"}}
    )
    assert stack.names == ["compress"]
    mw = stack.middlewares[0]
    assert isinstance(mw, CompressMiddleware)
    assert mw.rate == 0.3
    assert mw.method == "llmlingua2"


def test_compress_bad_backend_raises():
    with pytest.raises(ValueError):
        CompressMiddleware(backend="nope")


def test_compress_subprocess_backend_requires_worker_venv():
    """backend=subprocess 但没配 worker_venv → 友好报错（不拉起进程）。"""
    mw = CompressMiddleware(backend="subprocess", worker_venv="")
    with pytest.raises(ValueError, match="worker_venv"):
        mw._get_compressor()


def test_compress_worker_script_defaults_to_packaged():
    """worker_script 默认指向随包的 _compress_worker.py。"""
    import os

    mw = CompressMiddleware(backend="subprocess", worker_venv="/x/python")
    assert mw.worker_script.endswith("_compress_worker.py")
    assert os.path.exists(mw.worker_script)


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
