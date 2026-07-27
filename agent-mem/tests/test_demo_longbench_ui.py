"""LongBench interactive task runner tests without an engine or NPU."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

from agent_mem.context_telemetry import ContextEventBuffer
from agent_mem.demo.longbench_ui import run_longbench_task_streaming
from agent_mem.middleware import LazyLoadMiddleware, MiddlewareStack
from agent_mem.middleware.tool_synopsis import artifact_id_from_reference


def _write_longbench_zip(tmp_path):
    capital_body = "France's capital is Paris. " * 180
    eiffel_body = "The Eiffel Tower is in Paris. " * 180
    example = {
        "_id": "demo-0",
        "context": (
            f"Passage 1:\nCapital\n{capital_body}\n"
            f"Passage 2:\nEiffel\n{eiffel_body}"
        ),
        "input": "What is the capital of France?",
        "answers": ["Paris"],
    }
    path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data/2wikimqa.jsonl", json.dumps(example))
    return path


def test_longbench_streams_tool_trace_prompt_events_and_verdict(tmp_path, monkeypatch):
    data_zip = _write_longbench_zip(tmp_path)
    responses = iter([
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "retrieve_documents",
                        "arguments": json.dumps({"query": "capital of France"}),
                    },
                }],
            },
            0.01,
            100,
        ),
        ({"role": "assistant", "content": "Paris"}, 0.02, 140),
    ])
    monkeypatch.setattr(
        "agent_mem.agent.react.stream_chat_with_ttft",
        lambda *args, **kwargs: next(responses),
    )
    measurements = iter([
        {
            "original_prompt_tokens": 110,
            "transformed_prompt_tokens": 100,
            "saved_tokens": 10,
            "saved_percent": 9.09,
        },
        {
            "original_prompt_tokens": 160,
            "transformed_prompt_tokens": 140,
            "saved_tokens": 20,
            "saved_percent": 12.5,
        },
    ])
    monkeypatch.setattr(
        "agent_mem.agent.usage_log.measure_prompt_pair",
        lambda **kwargs: next(measurements),
    )
    buffer = ContextEventBuffer()

    updates = list(run_longbench_task_streaming(
        data_zip=str(data_zip),
        task_id=0,
        engine_url="http://unused/v1",
        model="test-model",
        max_steps=4,
        middlewares=MiddlewareStack(),
        context_event_sink=buffer,
        client=SimpleNamespace(),
    ))

    assert len(updates) == 3
    assert "候选文档 2 篇" in updates[0][1]
    assert any("retrieve_documents" in item["content"] for item in updates[1][0])
    assert any("工具结果" in item["content"] for item in updates[1][0])
    assert "成功" in updates[-1][1]
    assert "gold=Paris" in updates[-1][1]
    prompt_events = [
        event for event in buffer.events(session_id="longbench-0")
        if event["event"] == "prompt.completed"
    ]
    assert len(prompt_events) == 2
    assert prompt_events[-1]["data"]["transformed_prompt_tokens"] == 140


def test_longbench_stream_handles_f3_externalize_and_fetch(tmp_path, monkeypatch):
    data_zip = _write_longbench_zip(tmp_path)
    call_index = 0

    def fake_stream(*args, **kwargs):
        nonlocal call_index
        call_index += 1
        if call_index == 1:
            return ({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "retrieve-1",
                    "type": "function",
                    "function": {"name": "retrieve_documents", "arguments": "{}"},
                }],
            }, 0.01, 100)
        if call_index == 2:
            reference = str(kwargs["messages"][-1]["content"])
            result_id = artifact_id_from_reference(reference)
            assert result_id
            return ({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "fetch-1",
                    "type": "function",
                    "function": {
                        "name": "fetch_tool_result",
                        "arguments": json.dumps({
                            "result_id": result_id,
                            "json_pointer": "/documents/0",
                        }),
                    },
                }],
            }, 0.01, 120)
        return ({"role": "assistant", "content": "Paris"}, 0.01, 130)

    monkeypatch.setattr("agent_mem.agent.react.stream_chat_with_ttft", fake_stream)
    monkeypatch.setattr(
        "agent_mem.agent.usage_log.measure_prompt_pair",
        lambda **kwargs: {
            "original_prompt_tokens": 200,
            "transformed_prompt_tokens": 100,
            "saved_tokens": 100,
            "saved_percent": 50.0,
        },
    )
    buffer = ContextEventBuffer()
    stack = MiddlewareStack([LazyLoadMiddleware(
        store="memory",
        externalize_trigger_tokens=1,
        tokenizer_model="",
    )])

    updates = list(run_longbench_task_streaming(
        data_zip=str(data_zip),
        task_id=0,
        engine_url="http://unused/v1",
        model="test-model",
        max_steps=4,
        middlewares=stack,
        context_event_sink=buffer,
        client=SimpleNamespace(),
    ))

    event_names = [event["event"] for event in buffer.events(session_id="longbench-0")]
    assert "f3.tool_result_externalized" in event_names
    assert "f3.fetch_finished" in event_names
    assert "成功" in updates[-1][1]
