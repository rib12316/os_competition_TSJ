"""Frontend-neutral F2/F3 context telemetry tests."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

from agent_mem.agent import usage_log
from agent_mem.agent.react import run_react
from agent_mem.context_telemetry import (
    CompositeContextEventSink,
    ContextEventBuffer,
    JsonlContextEventSink,
    messages_preview,
    text_preview,
)
from agent_mem.middleware import BaseMiddleware, MiddlewareContext
from agent_mem.middleware.artifact_store import MemoryArtifactStore
from agent_mem.middleware.compress import CompressMiddleware
from agent_mem.middleware.lazyload import LazyLoadMiddleware


class _FakeCompressor:
    def __init__(self, buffer=None):
        self.buffer = buffer

    def compress_prompt(self, prompt, **kwargs):
        if self.buffer is not None:
            assert self.buffer.snapshot("tau-7")["f2"]["phase"] == "compressing"
        return {
            "compressed_prompt": "compressed cold history",
            "origin_tokens": 120,
            "compressed_tokens": 24,
            "ratio": "5x",
        }


class _ShortenMiddleware(BaseMiddleware):
    name = "shorten"

    def transform_messages(self, messages, ctx):
        return [
            {**message, "content": str(message.get("content") or "")[:4]}
            for message in messages
        ]


def test_preview_helpers_bound_frontend_content():
    preview = text_preview("x" * 20, limit=8)
    assert preview == {"text": "x" * 8, "chars": 20, "truncated": True}

    messages = messages_preview(
        [{"role": "user", "content": "abcdefghij"}],
        content_limit=4,
    )
    assert messages["message_count"] == 1
    assert messages["messages"][0]["content"]["text"] == "abcd"
    assert messages["messages"][0]["content"]["truncated"] is True


def test_f2_exports_cold_history_waiting_and_compressed_history():
    buffer = ContextEventBuffer()
    ctx = MiddlewareContext("tau-7", event_sink=buffer)
    ctx.bump_step()
    middleware = CompressMiddleware(
        method="longllmlingua",
        trigger_tokens=1,
        keep_hot=1,
        backend="inprocess",
    )
    middleware._compressor = _FakeCompressor(buffer)

    output = middleware.transform_messages(
        [
            {"role": "user", "content": "cold history " * 40},
            {"role": "assistant", "content": "hot tail"},
        ],
        ctx,
    )

    names = [event["event"] for event in buffer.events(session_id="tau-7")]
    assert names[:3] == [
        "f2.history_ready",
        "f2.compress_started",
        "f2.compress_finished",
    ]
    assert output[0]["content"].startswith("[compressed history]")
    snapshot = buffer.snapshot("tau-7")
    assert snapshot["f2"]["phase"] == "compressed"
    assert snapshot["f2"]["cold_before"]["message_count"] == 1
    assert snapshot["f2"]["cold_before"]["messages"][0]["content"]["text"]
    assert snapshot["f2"]["cold_after"]["compressed_text"]["text"] == (
        "compressed cold history"
    )
    assert snapshot["f2"]["origin_tokens"] == 120
    assert snapshot["f2"]["compressed_tokens"] == 24


def test_f3_exports_original_tool_data_and_structured_reference():
    buffer = ContextEventBuffer()
    ctx = MiddlewareContext(
        "longbench-3",
        event_sink=buffer,
        tool_call_id="call-retrieve",
        tool_call_index=0,
    )
    ctx.bump_step()
    middleware = LazyLoadMiddleware(
        externalize_trigger_tokens=1,
        artifact_store=MemoryArtifactStore(),
    )
    raw = json.dumps({
        "documents": [
            {"title": f"Document {index}", "text": "evidence " * 40}
            for index in range(20)
        ]
    })

    reference = middleware.intercept_tool_result(
        "retrieve_documents",
        {"query": "question"},
        raw,
        ctx,
    )

    parsed_reference = json.loads(reference)
    assert parsed_reference["_agent_mem"] == "external_tool_result"
    snapshot = buffer.snapshot("longbench-3")
    latest = snapshot["f3"]["latest"]
    assert latest["phase"] == "externalized"
    assert latest["tool_call_id"] == "call-retrieve"
    assert latest["arguments"] == {"query": "question"}
    assert latest["original"]["preview"]["text"]
    assert latest["externalized"]["synopsis"]["kind"] == "json"
    assert latest["externalized"]["reference"]["text"] == reference
    assert latest["externalized"]["reference_tokens"] < latest["original"]["tokens"]

    ctx.bump_step()
    middleware.handle_internal_tool_call(
        "fetch_tool_result",
        {
            "result_id": parsed_reference["result_id"],
            "json_pointer": "/documents",
            "match_field": "title",
            "match_value": "Document 3",
            "match_mode": "exact",
        },
        ctx,
    )
    fetch = buffer.snapshot("longbench-3")["f3"]["fetches"][-1]
    assert fetch["event"] == "f3.fetch_finished"
    assert fetch["selector"]["match_value"] == "Document 3"
    assert fetch["fetch_tokens"] > 0


def test_prompt_pair_and_incremental_events_are_frontend_ready():
    buffer = ContextEventBuffer()
    ctx = MiddlewareContext("tau-2", event_sink=buffer)
    ctx.bump_step()
    ctx.emit("prompt.measured", {
        "original_prompt_tokens": 1000,
        "transformed_prompt_tokens": 600,
        "saved_tokens": 400,
        "saved_percent": 40.0,
    })
    first_sequence = buffer.events()[-1]["sequence"]
    ctx.emit("prompt.completed", {"prompt_tokens": 600})

    incremental = buffer.events(after_sequence=first_sequence)
    assert [event["event"] for event in incremental] == ["prompt.completed"]
    prompt = buffer.snapshot("tau-2")["prompt"]
    assert prompt["original_prompt_tokens"] == 1000
    assert prompt["transformed_prompt_tokens"] == 600
    assert prompt["prompt_tokens"] == 600


def test_event_buffer_is_thread_safe_for_concurrent_benchmark():
    buffer = ContextEventBuffer(max_events=1000)

    def publish(task_id):
        ctx = MiddlewareContext(f"tau-{task_id}", event_sink=buffer)
        for step in range(1, 51):
            ctx.step = step
            ctx.emit("prompt.completed", {"prompt_tokens": step})

    threads = [threading.Thread(target=publish, args=(task_id,)) for task_id in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    events = buffer.events()
    assert len(events) == 400
    assert len({event["sequence"] for event in events}) == 400
    assert buffer.snapshot("tau-3")["prompt"]["prompt_tokens"] == 50


def test_composite_sink_writes_jsonl_without_coupling_frontend(tmp_path):
    buffer = ContextEventBuffer()
    jsonl = tmp_path / "context-events.jsonl"
    sink = CompositeContextEventSink(buffer, JsonlContextEventSink(jsonl))
    ctx = MiddlewareContext("s", event_sink=sink)
    ctx.emit("f2.skipped", {"phase": "skipped", "reason": "below_trigger"})

    assert buffer.snapshot("s")["f2"]["reason"] == "below_trigger"
    written = json.loads(jsonl.read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert written["event"] == "f2.skipped"


def test_run_react_emits_real_prompt_pair_without_prompt_log(monkeypatch):
    monkeypatch.delenv("PROMPT_TOKEN_LOG", raising=False)
    monkeypatch.setattr(usage_log, "_get_tokenizer", lambda model: object())
    monkeypatch.setattr(usage_log, "_resolve_tokenizer_path", lambda model: "fake-tokenizer")
    monkeypatch.setattr(
        usage_log,
        "_count_chat_tokens",
        lambda tokenizer, messages, tools, kwargs: sum(
            len(str(message.get("content") or "")) for message in messages
        ),
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=4),
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="done", tool_calls=[]),
        )],
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response),
        )
    )
    buffer = ContextEventBuffer()

    run_react(
        client,
        "model",
        [{"role": "user", "content": "a much longer canonical request"}],
        [],
        lambda name, args: "",
        middlewares=[_ShortenMiddleware()],
        session_id="react-prompt",
        context_event_sink=buffer,
    )

    prompt = buffer.snapshot("react-prompt")["prompt"]
    assert prompt["original_prompt_tokens"] == 31
    assert prompt["transformed_prompt_tokens"] == 4
    assert prompt["saved_tokens"] == 27
    assert prompt["prompt_tokens"] == 4
