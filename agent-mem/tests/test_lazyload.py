"""F3 artifact externalization, bounded fetch, and F2 integration tests."""

from __future__ import annotations

import hashlib
import json

from agent_mem.middleware import MiddlewareContext, MiddlewareStack
from agent_mem.middleware.artifact_store import MemoryArtifactStore, SQLiteArtifactStore
from agent_mem.middleware.compress import CompressMiddleware
from agent_mem.middleware.lazyload import FETCH_TOOL_NAME, LazyLoadMiddleware
from agent_mem.middleware.tool_synopsis import artifact_id_from_reference


class _CharacterTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]

    @staticmethod
    def decode(ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


class _DropBodyCompressor:
    def compress_prompt(self, context, **kwargs):
        if isinstance(context, list):
            return {"compressed_prompt_list": ["summary" for _ in context]}
        return {"compressed_prompt": "summary"}


def _lazy(**kwargs) -> LazyLoadMiddleware:
    middleware = LazyLoadMiddleware(
        store="memory",
        tokenizer_model="unit-tokenizer",
        externalize_trigger_tokens=100,
        max_reference_tokens=512,
        fetch_max_tokens=300,
        artifact_store=MemoryArtifactStore(),
        **kwargs,
    )
    middleware._tokenizer = _CharacterTokenizer()
    return middleware


def test_memory_store_is_byte_exact_and_session_scoped():
    store = MemoryArtifactStore()
    artifact = store.put(
        session_id="s1",
        tool_name="search",
        content="你好\nraw data",
        content_type="text/plain",
        token_count=11,
    )

    loaded = store.get("s1", artifact.result_id)

    assert loaded is not None
    assert loaded.content == "你好\nraw data"
    assert loaded.sha256 == hashlib.sha256(loaded.content.encode()).hexdigest()
    assert store.get("s2", artifact.result_id) is None


def test_sqlite_store_round_trip_and_session_isolation(tmp_path):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.sqlite3"))
    artifact = store.put(
        session_id="s1",
        tool_name="query",
        content='{"rows":[1,2,3]}',
        content_type="application/json",
        token_count=9,
    )

    assert store.get("s1", artifact.result_id) == artifact
    assert store.get("other", artifact.result_id) is None
    store.close()


def test_large_result_externalizes_to_compact_structured_reference():
    middleware = _lazy()
    content = json.dumps({"items": [{"id": i, "description": "x" * 30} for i in range(20)]})
    ctx = MiddlewareContext("s1")

    reference = middleware.intercept_tool_result("search", {"q": "x"}, content, ctx)
    result_id = artifact_id_from_reference(reference)

    assert result_id is not None
    assert middleware._count(reference) <= middleware.max_reference_tokens
    artifact = middleware.store.get("s1", result_id)
    assert artifact is not None and artifact.content == content
    payload = json.loads(reference)
    assert payload["fetch"]["name"] == FETCH_TOOL_NAME
    assert payload["fetch"]["arguments"]["result_id"] == result_id


def test_small_result_passes_through_unchanged():
    middleware = _lazy()
    content = "small result"
    assert middleware.intercept_tool_result(
        "search", {}, content, MiddlewareContext("s1")
    ) == content


def test_fetch_is_bounded_and_cross_session_fetch_is_hidden():
    middleware = _lazy()
    content = "\n".join(f"line {i}: {'x' * 50}" for i in range(100))
    ctx = MiddlewareContext("s1")
    reference = middleware.intercept_tool_result("search", {}, content, ctx)
    result_id = artifact_id_from_reference(reference)
    assert result_id is not None

    handled = middleware.handle_internal_tool_call(
        FETCH_TOOL_NAME,
        {"result_id": result_id, "start_line": 1, "max_lines": 100},
        ctx,
    )
    denied = middleware.handle_internal_tool_call(
        FETCH_TOOL_NAME, {"result_id": result_id}, MiddlewareContext("s2")
    )

    assert handled is not None
    assert middleware._count(handled.content) <= middleware.fetch_max_tokens
    assert json.loads(handled.content)["token_truncated"] is True
    assert denied is not None
    assert json.loads(denied.content)["status"] == "not_found"


def test_fetch_supports_rfc6901_json_pointer():
    middleware = _lazy()
    content = json.dumps({
        "items": [{"id": "A/1", "status": "ready"}],
        "meta": {"n": 1},
        "padding": "x" * 1000,
    })
    ctx = MiddlewareContext("s1")
    reference = middleware.intercept_tool_result("query", {}, content, ctx)
    result_id = artifact_id_from_reference(reference)
    assert result_id is not None

    handled = middleware.handle_internal_tool_call(
        FETCH_TOOL_NAME,
        {"result_id": result_id, "json_pointer": "/items/0/status"},
        ctx,
    )

    assert handled is not None
    assert json.loads(handled.content)["content"] == '"ready"'


def test_fetch_can_continue_a_minified_json_value_by_character_offset():
    middleware = _lazy()
    content = json.dumps({"items": [{"value": "x" * 2000}]})
    ctx = MiddlewareContext("s1")
    reference = middleware.intercept_tool_result("query", {}, content, ctx)
    result_id = artifact_id_from_reference(reference)
    assert result_id is not None

    first = middleware.handle_internal_tool_call(
        FETCH_TOOL_NAME,
        {"result_id": result_id, "json_pointer": "/items/0/value"},
        ctx,
    )
    first_payload = json.loads(first.content)
    second = middleware.handle_internal_tool_call(
        FETCH_TOOL_NAME,
        {
            "result_id": result_id,
            "json_pointer": "/items/0/value",
            "start_char": first_payload["next_start_char"],
        },
        ctx,
    )
    second_payload = json.loads(second.content)

    assert first_payload["next_start_char"] > 0
    assert second_payload["start_char"] == first_payload["next_start_char"]
    assert middleware._count(first.content) <= middleware.fetch_max_tokens
    assert middleware._count(second.content) <= middleware.fetch_max_tokens


def test_fetch_schema_is_stable_and_not_duplicated():
    middleware = _lazy()
    stack = MiddlewareStack([middleware])
    tools = [{"type": "function", "function": {"name": "search"}}]
    ctx = MiddlewareContext("s1")

    _, first = stack.transform_request([], tools, ctx)
    _, second = stack.transform_request([], first, ctx)

    assert [tool["function"]["name"] for tool in first].count(FETCH_TOOL_NAME) == 1
    assert first == second


def test_measurement_baseline_restores_raw_result_only_in_copy():
    middleware = _lazy()
    ctx = MiddlewareContext("s1")
    raw = "large result " * 100
    reference = middleware.intercept_tool_result("search", {}, raw, ctx)
    messages = [{"role": "tool", "tool_call_id": "c1", "content": reference}]

    restored, _ = middleware.measurement_baseline(messages, [], ctx)

    assert restored[0]["content"] == raw
    assert messages[0]["content"] == reference


def test_f2_cold_compression_preserves_f3_reference_and_fetch_arguments():
    lazy = _lazy()
    ctx = MiddlewareContext("combo")
    raw = json.dumps({"items": [{"id": i, "description": "detail " * 20} for i in range(20)]})
    reference = lazy.intercept_tool_result("search", {}, raw, ctx)
    result_id = artifact_id_from_reference(reference)
    assert result_id is not None

    compress = CompressMiddleware(
        method="llmlingua2",
        tool_aware=True,
        trigger_tokens=1,
        recompress_delta_tokens=1,
        keep_hot=1,
        tokenizer_model="unit-tokenizer",
        backend="inprocess",
    )
    compress._tokenizer = _CharacterTokenizer()
    compress._compressor = _DropBodyCompressor()
    messages = [
        {"role": "user", "content": "search"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "search", "arguments": '{"q":"x"}'},
        }]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": reference},
        {"role": "assistant", "content": "hot tail"},
    ]

    out = compress.transform_messages(messages, ctx)
    rendered = "\n".join(str(message.get("content") or "") for message in out)

    assert result_id in rendered
    assert FETCH_TOOL_NAME in rendered


def test_fetch_result_is_exempt_from_externalization():
    middleware = _lazy()
    content = "x" * 1000
    assert middleware.intercept_tool_result(
        FETCH_TOOL_NAME, {}, content, MiddlewareContext("s1")
    ) == content
