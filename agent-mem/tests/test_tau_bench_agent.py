"""τ-bench agent 测试（fake 流式 client + fake env，importorskip tau_bench 守门）。"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("tau_bench")

from agent_mem.agent.tau_bench_agent import TauBenchAgent  # noqa: E402


def _content_chunks(text):
    return [types.SimpleNamespace(choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")])]


def _tool_chunks(name, args, cid="c1"):
    return [types.SimpleNamespace(choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(
            content=None,
            tool_calls=[types.SimpleNamespace(
                index=0, id=cid, function=types.SimpleNamespace(name=name, arguments=args))]),
        finish_reason="tool_calls")])]


class _FakeStreamClient:
    """按脚本依次返回 chunk 流；create 返回迭代器。"""

    def __init__(self, chunk_lists):
        self._lists = list(chunk_lists)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kw):
        assert kw.get("stream") is True
        return iter(self._lists.pop(0))


class _FakeEnv:
    wiki = "You are a retail agent."
    tools_info = [
        {"type": "function", "function": {
            "name": "calculate", "description": "calc",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
        }}
    ]

    def reset(self, task_index=None):
        return types.SimpleNamespace(
            observation="请帮我取消订单 #W0001",
            info=types.SimpleNamespace(model_dump=lambda: {"task_id": task_index}),
        )

    def step(self, action):
        if action.name == "respond":
            return types.SimpleNamespace(
                observation="再见", reward=1.0, done=True,
                info=types.SimpleNamespace(model_dump=lambda: {}),
            )
        return types.SimpleNamespace(
            observation="tool ok", reward=0.0, done=False,
            info=types.SimpleNamespace(model_dump=lambda: {}),
        )


def test_solve_tool_then_respond_gets_reward_and_ttft():
    env = _FakeEnv()
    client = _FakeStreamClient([
        _tool_chunks("calculate", '{"expression":"1+1"}'),
        _content_chunks("您的订单已取消"),
    ])
    # 注入确定性 clock：首 chunk 时 +0.05s
    times = iter([0.0, 0.05, 0.05, 0.10])
    agent = TauBenchAgent(client, "Qwen3-0.6B")

    # monkeypatch stream_chat_with_ttft 的 clock（经 tau_bench_agent 调用）
    import agent_mem.agent.react as react

    orig = react.stream_chat_with_ttft

    def patched(client, **kw):
        kw["clock"] = lambda: next(times)
        return orig(client, **kw)

    react.stream_chat_with_ttft = patched
    try:
        out = agent.solve(env, task_index=0, max_num_steps=10)
    finally:
        react.stream_chat_with_ttft = orig

    assert out.reward == 1.0
    assert out.n_steps == 2
    assert len(out.ttft_ms_list) == 2
    assert all(t > 0 for t in out.ttft_ms_list)  # TTFT 被采集
    assert any(m.get("role") == "tool" and m.get("content") == "tool ok" for m in out.messages)


def test_solve_truncates_at_max_steps_without_done():
    env = _FakeEnv()
    client = _FakeStreamClient([_tool_chunks("calculate", "{}")] * 20)
    agent = TauBenchAgent(client, "Qwen3-0.6B")
    out = agent.solve(env, task_index=0, max_num_steps=3)
    assert out.n_steps == 3
    assert out.reward == 0.0
    assert len(out.ttft_ms_list) == 3
