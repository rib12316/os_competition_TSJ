"""ReAct 核心引擎测试（fake client + stub server 联调）+ 工具测试。"""

from __future__ import annotations

import types

import pytest

from agent_mem.agent import tools
from agent_mem.agent.react import run_react
from agent_mem.server import stub_openai


def _msg(content=None, tool_calls=None):
    return types.SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)


def _tc(name, args, cid="c1"):
    return types.SimpleNamespace(
        id=cid,
        function=types.SimpleNamespace(name=name, arguments=args),
    )


def _resp(msg):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeClient:
    """按脚本依次返回 response；记录每次 create 的入参。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


def test_run_react_one_tool_then_finish():
    client = _FakeClient([
        _resp(_msg(None, [_tc("search", '{"query":"x"}')])),
        _resp(_msg("done")),
    ])
    res = run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=5,
    )
    assert res.final_text == "done"
    assert res.n_steps == 2
    assert res.tool_calls_made == 1
    assert not res.truncated
    # messages: user, assistant(tool_call), tool(obs), assistant(done)
    roles = [m["role"] for m in res.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "search" in res.messages[2]["content"] or "stub" in res.messages[2]["content"]


def test_run_react_max_steps_truncation():
    # 始终返回 tool_call → 应在 max_steps 截断
    client = _FakeClient([_resp(_msg(None, [_tc("search", "{}")]))] * 10)
    res = run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=3,
    )
    assert res.truncated
    assert res.n_steps == 3
    assert res.tool_calls_made == 3


def test_run_react_bad_tool_args_recovers():
    # arguments 不是合法 JSON → args={}，工具仍被调用
    client = _FakeClient([
        _resp(_msg(None, [_tc("search", "not-json")])),
        _resp(_msg("ok")),
    ])
    res = run_react(
        client, "m", [{"role": "user", "content": "hi"}],
        tools.TOOLS, tools.execute_tool, max_steps=5,
    )
    assert res.final_text == "ok"
    assert res.tool_calls_made == 1


def test_run_react_against_stub_server():
    """真 HTTP：stub 带 tools 返回 tool_call，run_react 能解析并执行工具。"""
    openai = pytest.importorskip("openai")  # 未装 openai 时跳过（不崩收集）
    srv, base_url = stub_openai.start_background(port=0)
    try:
        client = openai.OpenAI(base_url=base_url, api_key="stub")
        res = run_react(
            client, "Qwen3-0.6B", [{"role": "user", "content": "search for x"}],
            tools.TOOLS, tools.execute_tool, max_steps=2,
        )
        # stub 总是回 tool_call，2 步内应至少执行 1 次工具（截断）
        assert res.tool_calls_made >= 1
        assert res.n_steps == 2
    finally:
        srv.shutdown()
        srv.server_close()


# ---- 工具 ----


def test_python_safe_arith():
    assert tools.python("1 + 2 * 3") == "7"
    assert tools.python("(1 + 2) * 3") == "9"
    assert tools.python("2 ** 10") == "1024"


def test_python_rejects_unsafe():
    with pytest.raises(ValueError):
        tools.python("__import__('os')")  # 不是合法算术表达式
    with pytest.raises(ValueError):
        tools.python("open('x')")


def test_search_canned():
    assert "stub" in tools.search("anything")


def test_execute_tool_dispatch():
    assert "7" in tools.execute_tool("python", {"expression": "3+4"})
    assert "stub" in tools.execute_tool("search", {"query": "q"})
    with pytest.raises(ValueError):
        tools.execute_tool("nope", {})


# ---- stream_chat_with_ttft ----


def _chunk(content=None, tool_calls=None, finish=None):
    import types

    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(content=content, tool_calls=tool_calls), finish_reason=finish)])


def _tc_delta(index, name=None, args=None, cid=None):
    import types

    return types.SimpleNamespace(
        index=index, id=cid,
        function=types.SimpleNamespace(name=name, arguments=args))


def test_stream_chat_with_ttft_reconstructs_content_and_tool_calls():
    from agent_mem.agent.react import stream_chat_with_ttft

    class _C:
        def __init__(self, chunks):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: iter(chunks))
            )

    chunks = [
        _chunk(content="hel"),
        _chunk(content="lo"),
        _chunk(tool_calls=[_tc_delta(0, name="search", args='{"q":"')]),
        _chunk(tool_calls=[_tc_delta(0, args='x"}')], finish="tool_calls"),
    ]
    clock_vals = iter([10.0, 10.02])  # t0, first-chunk
    msg, ttft = stream_chat_with_ttft(
        _C(chunks), model="m", messages=[], tools=[],
        clock=lambda: next(clock_vals),
    )
    assert msg["content"] == "hello"
    assert msg["tool_calls"][0]["function"]["name"] == "search"
    # arguments 跨两片拼接
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"q":"x"}'
    assert ttft == pytest.approx(0.02)


def test_stream_chat_with_ttft_empty_stream():
    from agent_mem.agent.react import stream_chat_with_ttft

    class _C:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kw: iter([]))
            )

    clock_vals = iter([5.0, 5.5])
    msg, ttft = stream_chat_with_ttft(
        _C(), model="m", messages=[], clock=lambda: next(clock_vals)
    )
    assert msg["content"] is None
    assert ttft == pytest.approx(0.5)
