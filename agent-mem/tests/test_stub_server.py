"""stub OpenAI server 测试（httpx + 真 openai SDK 双断言）。"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_mem.server import stub_openai

# 真 openai SDK（若环境没装则跳过本组 SDK 断言）
openai = pytest.importorskip("openai")


@pytest.fixture
def server():
    srv, base_url = stub_openai.start_background(port=0)
    yield base_url
    srv.shutdown()
    srv.server_close()


# ---- httpx 直连断言 ----


def test_chat_completions_with_tools_returns_tool_call(server):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "calc",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    resp = httpx.post(
        f"{server}/chat/completions",
        json={"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
        timeout=10,
    )
    assert resp.status_code == 200
    obj = resp.json()
    assert obj["object"] == "chat.completion"
    choice = obj["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "calculate"  # 回显请求里的第一个 tool 名
    # arguments 必须是合法 JSON 字符串（openai SDK 约定）
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {}
    assert obj["usage"]["total_tokens"] == obj["usage"]["prompt_tokens"] + obj["usage"]["completion_tokens"]


def test_chat_completions_without_tools_returns_text(server):
    resp = httpx.post(
        f"{server}/chat/completions",
        json={"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "hi"}]},
        timeout=10,
    )
    obj = resp.json()
    assert obj["choices"][0]["finish_reason"] == "stop"
    assert obj["choices"][0]["message"]["content"] == "ok"


def test_models_endpoint(server):
    resp = httpx.get(f"{server}/models", timeout=10)
    assert resp.status_code == 200
    obj = resp.json()
    assert obj["object"] == "list"
    assert obj["data"][0]["id"] == "Qwen2.5-7B-Instruct"


def test_metrics_endpoint(server):
    resp = httpx.get(f"{server.replace('/v1', '')}/metrics", timeout=10)
    assert resp.status_code == 200
    assert "vllm:prefix_cache_hits_total" in resp.text
    assert "vllm:prefix_cache_queries_total" in resp.text


def test_unknown_path_404(server):
    resp = httpx.get(f"{server.replace('/v1', '')}/nope", timeout=10)
    assert resp.status_code == 404


# ---- 真 openai SDK 断言（最强：证明 P2 day-1 可用）----


def test_openai_sdk_parses_tool_call(server):
    client = openai.OpenAI(base_url=server, api_key="stub")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    completion = client.chat.completions.create(
        model="Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "run a tool"}],
        tools=tools,
    )
    msg = completion.choices[0].message
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == "search"
    json.loads(msg.tool_calls[0].function.arguments)  # SDK 已保证是 JSON 字符串


def test_openai_sdk_text_path(server):
    client = openai.OpenAI(base_url=server, api_key="stub")
    completion = client.chat.completions.create(
        model="Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert completion.choices[0].message.content == "ok"
    assert completion.choices[0].finish_reason == "stop"
