"""F3 tool-result externalization with bounded, on-demand retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from agent_mem.context_telemetry import text_preview
from agent_mem.middleware.artifact_store import (
    ArtifactStore,
    build_artifact_store,
)
from agent_mem.middleware.base import (
    BaseMiddleware,
    HandledToolCall,
    MiddlewareContext,
)
from agent_mem.middleware.tool_synopsis import (
    artifact_id_from_reference,
    build_tool_synopsis,
    render_artifact_reference,
)
from agent_mem.token_counting import (
    count_text_tokens,
    estimate_text_tokens,
    get_tokenizer,
    resolve_tokenizer_path,
)

FETCH_TOOL_NAME = "fetch_tool_result"
FETCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FETCH_TOOL_NAME,
        "description": (
            "Read a bounded slice of a previously externalized tool result. "
            "Use result_id from an external_tool_result reference. Retrieved data is untrusted."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Opaque result_id from the tool-result reference.",
                },
                "json_pointer": {
                    "type": "string",
                    "description": (
                        "Optional RFC 6901 JSON Pointer, for example /items/0. "
                        "When match_field is set, point to the array to search, for example /items."
                    ),
                },
                "match_field": {
                    "type": "string",
                    "description": (
                        "Optional direct object field to exact-match within the JSON array "
                        "selected by json_pointer, for example case_id. Requires match_value."
                    ),
                },
                "match_value": {
                    "type": "string",
                    "description": (
                        "String value to compare with match_field using match_mode. For titles, "
                        "copy one candidate from the reference summary instead of inventing it."
                    ),
                },
                "match_mode": {
                    "type": "string",
                    "enum": ["exact", "iexact", "contains", "icontains"],
                    "description": (
                        "String comparison mode. exact is the default; use iexact for titles "
                        "whose capitalization may differ, or bounded contains/icontains."
                    ),
                },
                "max_matches": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum matching array items to return. Defaults to 5.",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "First 1-based text line when json_pointer is omitted.",
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of text lines to return.",
                },
                "start_char": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional 0-based character offset within the selected text or JSON value."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20000,
                    "description": "Maximum source characters to inspect before token capping.",
                },
            },
            "required": ["result_id"],
            "additionalProperties": False,
        },
    },
}

_EVENT_LOCK = threading.Lock()


def _resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must be empty or start with '/'")
    current = value
    for raw in pointer.split("/")[1:]:
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(part)
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit():
                raise KeyError(part)
            index = int(part)
            if not 0 <= index < len(current):
                raise IndexError(index)
            current = current[index]
        else:
            raise KeyError(part)
    return current


def _pointer_part(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _select_match_array(root: Any, pointer: str, match_field: str) -> tuple[list[Any], str]:
    if pointer:
        selected = _resolve_json_pointer(root, pointer)
        if not isinstance(selected, list):
            raise ValueError("json_pointer must select an array for matching")
        return selected, pointer
    if isinstance(root, list):
        return root, ""
    if not isinstance(root, dict):
        raise ValueError("matching requires a JSON array")
    candidates = [
        (key, value)
        for key, value in root.items()
        if isinstance(value, list)
        and any(isinstance(item, dict) and match_field in item for item in value)
    ]
    if len(candidates) != 1:
        raise ValueError("omit json_pointer only when one searchable array exists")
    key, selected = candidates[0]
    return selected, f"/{_pointer_part(key)}"


class LazyLoadMiddleware(BaseMiddleware):
    """Externalize large tool results and expose a stable bounded fetch tool."""

    name = "lazyload"

    def __init__(
        self,
        *,
        store: str = "sqlite",
        store_path: str = "",
        ttl_seconds: float = 3600.0,
        externalize_trigger_tokens: int = 4000,
        max_reference_tokens: int = 512,
        fetch_max_tokens: int = 768,
        fetch_default_lines: int = 20,
        fetch_max_lines: int = 100,
        max_parse_bytes: int = 5_000_000,
        on_store_error: str = "passthrough",
        fallback_head_tokens: int = 512,
        fallback_tail_tokens: int = 256,
        exempt_tools: list[str] | None = None,
        tool_overrides: dict[str, int] | None = None,
        tokenizer_model: str = "",
        event_log: str | None = None,
        artifact_store: ArtifactStore | None = None,
        telemetry_preview_chars: int = 4000,
    ) -> None:
        if externalize_trigger_tokens <= 0:
            raise ValueError("externalize_trigger_tokens must be > 0")
        if max_reference_tokens <= 0 or fetch_max_tokens <= 0:
            raise ValueError("reference/fetch token budgets must be > 0")
        if fetch_default_lines <= 0 or fetch_max_lines <= 0:
            raise ValueError("fetch line limits must be > 0")
        if on_store_error not in {"passthrough", "head_tail", "raise"}:
            raise ValueError("on_store_error must be passthrough, head_tail, or raise")
        self.externalize_trigger_tokens = int(externalize_trigger_tokens)
        self.max_reference_tokens = int(max_reference_tokens)
        self.fetch_max_tokens = int(fetch_max_tokens)
        self.fetch_default_lines = min(int(fetch_default_lines), int(fetch_max_lines))
        self.fetch_max_lines = int(fetch_max_lines)
        self.max_parse_bytes = int(max_parse_bytes)
        self.on_store_error = on_store_error
        self.fallback_head_tokens = int(fallback_head_tokens)
        self.fallback_tail_tokens = int(fallback_tail_tokens)
        self.exempt_tools = set(exempt_tools or [FETCH_TOOL_NAME])
        self.exempt_tools.add(FETCH_TOOL_NAME)
        self.tool_overrides = dict(tool_overrides or {})
        self.tokenizer_model = tokenizer_model
        self._tokenizer: Any = None
        self._tokenizer_lock = threading.Lock()
        self._count_cache: dict[bytes, int] = {}
        self._count_lock = threading.Lock()
        if artifact_store is None:
            if store == "sqlite" and not store_path:
                store_path = f"/tmp/agent-mem-f3-{os.getpid()}.sqlite3"
            artifact_store = build_artifact_store(
                store, path=store_path, ttl_seconds=ttl_seconds
            )
        self.store = artifact_store
        self._event_log_path = event_log or os.environ.get("F3_EVENT_LOG", "")
        self.telemetry_preview_chars = max(200, int(telemetry_preview_chars))

    def prepare(self) -> None:
        if self.tokenizer_model:
            self._get_tokenizer()

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            with self._tokenizer_lock:
                if self._tokenizer is None:
                    self._tokenizer = get_tokenizer(self.tokenizer_model)
        return self._tokenizer

    @property
    def token_count_source(self) -> str:
        if self.tokenizer_model:
            return f"tokenizer:{resolve_tokenizer_path(self.tokenizer_model)}"
        return "unicode_heuristic"

    def _count(self, text: str) -> int:
        if not text:
            return 0
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        with self._count_lock:
            cached = self._count_cache.get(digest)
        if cached is not None:
            return cached
        count = (
            count_text_tokens(self._get_tokenizer(), text)
            if self.tokenizer_model
            else estimate_text_tokens([text])
        )
        with self._count_lock:
            if len(self._count_cache) >= 4096:
                self._count_cache.pop(next(iter(self._count_cache)))
            self._count_cache[digest] = count
        return count

    def transform_tools(
        self, tools: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        if any(
            tool.get("function", {}).get("name") == FETCH_TOOL_NAME
            for tool in tools
            if isinstance(tool, dict)
        ):
            return list(tools)
        return [*tools, FETCH_TOOL_SCHEMA]

    @staticmethod
    def _next_operation_id(ctx: MiddlewareContext) -> str:
        counter = int(ctx.scratch.get("lazyload:operation_counter", 0)) + 1
        ctx.scratch["lazyload:operation_counter"] = counter
        return f"{ctx.session_id}:{ctx.step}:{counter}"

    def intercept_tool_result(
        self, name: str, args: dict[str, Any], result: str, ctx: MiddlewareContext
    ) -> str:
        if name in self.exempt_tools:
            return result
        telemetry = ctx.telemetry_enabled
        operation_id = self._next_operation_id(ctx) if telemetry else ""
        threshold = int(self.tool_overrides.get(name, self.externalize_trigger_tokens))
        started = time.monotonic()
        original_tokens = self._count(result)
        event_base: dict[str, Any] = {}
        if telemetry:
            original = {
                "preview": text_preview(result, limit=self.telemetry_preview_chars),
                "tokens": original_tokens,
                "byte_count": len(result.encode("utf-8")),
            }
            event_base = {
                "operation_id": operation_id,
                "phase": "observed",
                "tool_call_id": ctx.tool_call_id,
                "tool_call_index": ctx.tool_call_index,
                "tool_name": name,
                "arguments": args,
                "threshold_tokens": threshold,
                "original": original,
            }
            ctx.emit("f3.tool_result_observed", event_base)
        if threshold <= 0:
            ctx.emit("f3.tool_result_passthrough", {
                **event_base,
                "phase": "passthrough",
                "reason": "tool_disabled",
            })
            return result
        if original_tokens < threshold:
            ctx.emit("f3.tool_result_passthrough", {
                **event_base,
                "phase": "passthrough",
                "reason": "below_trigger",
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            self._log_event(ctx, {
                "action": "passthrough",
                "tool_name": name,
                "original_tokens": original_tokens,
                "token_count_source": self.token_count_source,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            return result
        content_type, synopsis = build_tool_synopsis(
            result, max_parse_bytes=self.max_parse_bytes
        )
        if telemetry:
            ctx.emit("f3.externalize_started", {
                **event_base,
                "phase": "storing",
                "content_type": content_type,
                "synopsis": synopsis,
            })
        try:
            store_started = time.monotonic()
            artifact = self.store.put(
                session_id=ctx.session_id,
                tool_name=name,
                content=result,
                content_type=content_type,
                token_count=original_tokens,
            )
            store_ms = (time.monotonic() - store_started) * 1000
        except Exception as exc:
            if self.on_store_error == "raise":
                ctx.emit("f3.externalize_failed", {
                    **event_base,
                    "phase": "failed",
                    "error": repr(exc),
                })
                raise
            if self.on_store_error == "head_tail":
                fallback = self._head_tail_fallback(result)
                ctx.emit("f3.externalize_failed", {
                    **event_base,
                    "phase": "fallback",
                    "reason": "store_error_head_tail",
                    "error": repr(exc),
                    "fallback": text_preview(
                        fallback,
                        limit=self.telemetry_preview_chars,
                    ),
                })
                return fallback
            ctx.emit("f3.externalize_failed", {
                **event_base,
                "phase": "passthrough",
                "reason": "store_error_passthrough",
                "error": repr(exc),
            })
            self._log_event(ctx, {
                "action": "store_error_passthrough",
                "tool_name": name,
                "original_tokens": original_tokens,
                "token_count_source": self.token_count_source,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            return result

        reference = render_artifact_reference(artifact, synopsis)
        reference_tokens = self._count(reference)
        if reference_tokens > self.max_reference_tokens:
            reference = render_artifact_reference(
                artifact,
                {"kind": synopsis.get("kind", "unknown"), "summary_omitted": True},
            )
            reference_tokens = self._count(reference)
        if reference_tokens >= original_tokens or reference_tokens > self.max_reference_tokens:
            self.store.delete(ctx.session_id, artifact.result_id)
            ctx.emit("f3.tool_result_passthrough", {
                **event_base,
                "phase": "passthrough",
                "reason": "reference_not_smaller",
                "reference_tokens": reference_tokens,
            })
            return result

        ctx.scratch.setdefault("lazyload:result_ids", set()).add(artifact.result_id)
        if telemetry:
            externalized = {
                "result_id": artifact.result_id,
                "content_type": content_type,
                "byte_count": artifact.byte_count,
                "sha256": artifact.sha256,
                "synopsis": synopsis,
                "reference": text_preview(
                    reference,
                    limit=self.telemetry_preview_chars,
                ),
                "reference_tokens": reference_tokens,
            }
            ctx.emit("f3.tool_result_externalized", {
                **event_base,
                "phase": "externalized",
                "externalized": externalized,
                "saved_tokens": original_tokens - reference_tokens,
                "saved_percent": round(
                    (original_tokens - reference_tokens) / max(1, original_tokens) * 100,
                    4,
                ),
                "store_ms": round(store_ms, 3),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
        self._log_event(ctx, {
            "action": "externalize",
            "tool_name": name,
            "result_id": artifact.result_id,
            "content_type": content_type,
            "byte_count": artifact.byte_count,
            "original_tokens": original_tokens,
            "reference_tokens": reference_tokens,
            "saved_tokens": original_tokens - reference_tokens,
            "saved_percent": round(
                (original_tokens - reference_tokens) / max(1, original_tokens) * 100, 4
            ),
            "store_ms": round(store_ms, 3),
            "token_count_source": self.token_count_source,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        })
        return reference

    def handle_internal_tool_call(
        self, name: str, args: dict[str, Any], ctx: MiddlewareContext
    ) -> HandledToolCall | None:
        if name != FETCH_TOOL_NAME:
            return None
        started = time.monotonic()
        result_id = str(args.get("result_id") or "")
        telemetry = ctx.telemetry_enabled
        fetch_base: dict[str, Any] = {}
        if telemetry:
            fetch_id = self._next_operation_id(ctx)
            selector = {
                key: args[key]
                for key in (
                    "json_pointer",
                    "match_field",
                    "match_value",
                    "match_mode",
                    "max_matches",
                    "start_line",
                    "max_lines",
                    "start_char",
                    "max_chars",
                )
                if key in args
            }
            fetch_base = {
                "fetch_id": fetch_id,
                "phase": "fetching",
                "tool_call_id": ctx.tool_call_id,
                "tool_call_index": ctx.tool_call_index,
                "result_id": result_id,
                "selector": selector,
            }
            ctx.emit("f3.fetch_started", fetch_base)
        artifact = self.store.get(ctx.session_id, result_id)
        if artifact is None:
            content = json.dumps(
                {"status": "not_found", "result_id": result_id},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            ctx.emit("f3.fetch_failed", {
                **fetch_base,
                "phase": "failed",
                "status": "not_found",
                "response": text_preview(content, limit=self.telemetry_preview_chars),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            return HandledToolCall(content=content, status="error")
        if hashlib.sha256(artifact.content.encode("utf-8")).hexdigest() != artifact.sha256:
            content = json.dumps(
                {"status": "integrity_error", "result_id": result_id},
                separators=(",", ":"),
            )
            ctx.emit("f3.fetch_failed", {
                **fetch_base,
                "phase": "failed",
                "status": "integrity_error",
                "response": text_preview(content, limit=self.telemetry_preview_chars),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            return HandledToolCall(content=content, status="error")

        pointer = str(args.get("json_pointer") or "")
        match_field = str(args.get("match_field") or "")
        match_value = args.get("match_value")
        match_mode = str(args.get("match_mode") or "exact")
        matched_items: list[Any] | None = None
        try:
            if match_field:
                if not isinstance(match_value, str):
                    raise ValueError("match_value is required with match_field")
                if match_mode not in {"exact", "iexact", "contains", "icontains"}:
                    raise ValueError("invalid match_mode")
                if not match_value and match_mode in {"contains", "icontains"}:
                    raise ValueError("contains match_value must not be empty")
                if artifact.byte_count > self.max_parse_bytes:
                    content = json.dumps(
                        {"status": "selector_too_large", "result_id": result_id},
                        separators=(",", ":"),
                    )
                    ctx.emit("f3.fetch_failed", {
                        **fetch_base,
                        "phase": "failed",
                        "status": "selector_too_large",
                        "response": text_preview(
                            content,
                            limit=self.telemetry_preview_chars,
                        ),
                        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                    })
                    return HandledToolCall(content=content, status="error")
                selected, pointer = _select_match_array(
                    json.loads(artifact.content), pointer, match_field
                )
                max_matches = min(10, max(1, int(args.get("max_matches") or 5)))
                indexed_matches = [
                    (index, item)
                    for index, item in enumerate(selected)
                    if isinstance(item, dict)
                    and isinstance(item.get(match_field), str)
                    and self._string_matches(item[match_field], match_value, match_mode)
                ]
                returned_pairs = indexed_matches[:max_matches]
                returned = [item for _, item in returned_pairs]
                matched_items = returned
                text = json.dumps(returned, ensure_ascii=False, separators=(",", ":"))
                meta = {
                    "json_pointer": pointer,
                    "match_field": match_field,
                    "match_value": match_value,
                    "match_mode": match_mode,
                    "scanned_items": len(selected),
                    "matches_found": len(indexed_matches),
                    "matches_returned": len(returned),
                    "match_pointers": [
                        f"{pointer}/{index}" if pointer else f"/{index}"
                        for index, _ in returned_pairs
                    ],
                }
                source_truncated = len(indexed_matches) > len(returned)
            elif pointer:
                selected = _resolve_json_pointer(json.loads(artifact.content), pointer)
                text = json.dumps(selected, ensure_ascii=False, separators=(",", ":"))
                meta = {"json_pointer": pointer}
                source_truncated = False
            else:
                start_line = max(1, int(args.get("start_line") or 1))
                max_lines = min(
                    self.fetch_max_lines,
                    max(1, int(args.get("max_lines") or self.fetch_default_lines)),
                )
                lines = artifact.content.splitlines() or [artifact.content]
                start_index = min(len(lines), start_line - 1)
                end_index = min(len(lines), start_index + max_lines)
                text = "\n".join(lines[start_index:end_index])
                source_truncated = end_index < len(lines)
                meta = {
                    "start_line": start_index + 1,
                    "end_line": end_index,
                    "next_start_line": end_index + 1 if source_truncated else None,
                    "line_count": len(lines),
                }
        except (TypeError, ValueError, KeyError, IndexError, json.JSONDecodeError):
            content = json.dumps(
                {"status": "invalid_selector", "result_id": result_id},
                separators=(",", ":"),
            )
            ctx.emit("f3.fetch_failed", {
                **fetch_base,
                "phase": "failed",
                "status": "invalid_selector",
                "response": text_preview(content, limit=self.telemetry_preview_chars),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
            return HandledToolCall(content=content, status="error")

        start_char = max(0, int(args.get("start_char") or 0))
        max_chars = min(20_000, max(1, int(args.get("max_chars") or 12_000)))
        char_end = min(len(text), start_char + max_chars)
        selected_text = text[start_char:char_end]
        char_truncated = char_end < len(text)
        meta["start_char"] = start_char

        if matched_items is not None and start_char == 0:
            meta.pop("start_char", None)
            content, token_truncated = self._bounded_match_response(
                result_id=result_id,
                items=matched_items,
                meta=meta,
                source_truncated=source_truncated,
            )
        else:
            content, token_truncated = self._bounded_fetch_response(
                result_id=result_id,
                text=selected_text,
                meta=meta,
                source_truncated=source_truncated or char_truncated,
                source_start_char=start_char,
                has_more_chars=char_truncated,
            )
        fetch_tokens = self._count(content)
        if telemetry:
            ctx.emit("f3.fetch_finished", {
                **fetch_base,
                "phase": "fetched",
                "status": "ok",
                "response": text_preview(content, limit=self.telemetry_preview_chars),
                "fetch_tokens": fetch_tokens,
                "source_truncated": source_truncated,
                "token_truncated": token_truncated,
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            })
        self._log_event(ctx, {
            "action": "fetch",
            "tool_name": name,
            "result_id": result_id,
            "fetch_tokens": fetch_tokens,
            "source_truncated": source_truncated,
            "token_truncated": token_truncated,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "token_count_source": self.token_count_source,
        })
        return HandledToolCall(content=content)

    @staticmethod
    def _string_matches(candidate: str, query: str, mode: str) -> bool:
        if mode == "exact":
            return candidate == query
        if mode == "iexact":
            return candidate.casefold() == query.casefold()
        if mode == "contains":
            return query in candidate
        return query.casefold() in candidate.casefold()

    def _bounded_match_response(
        self,
        *,
        result_id: str,
        items: list[Any],
        meta: dict[str, Any],
        source_truncated: bool,
    ) -> tuple[str, bool]:
        """Fit exact-match results by whole JSON records, never partial JSON text."""
        def render(selected: list[Any], token_truncated: bool) -> str:
            selected_meta = dict(meta)
            pointers = selected_meta.get("match_pointers")
            if isinstance(pointers, list):
                selected_meta["match_pointers"] = pointers[: len(selected)]
            payload = {
                "status": "ok",
                "result_id": result_id,
                **selected_meta,
                "matches_returned": len(selected),
                "source_truncated": source_truncated or token_truncated,
                "token_truncated": token_truncated,
                "content": json.dumps(
                    selected, ensure_ascii=False, separators=(",", ":")
                ),
            }
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if not items:
            rendered = render([], False)
            if self._count(rendered) <= self.fetch_max_tokens:
                return rendered, False

        for take in range(len(items), 0, -1):
            token_truncated = take < len(items)
            rendered = render(items[:take], token_truncated)
            if self._count(rendered) <= self.fetch_max_tokens:
                return rendered, token_truncated

        if items and isinstance(items[0], dict):
            preview = {
                key: value
                if isinstance(value, (str, int, float, bool)) or value is None
                else f"<{type(value).__name__} omitted>"
                for key, value in items[0].items()
            }
            for _ in range(12):
                rendered = render([preview], True)
                count = self._count(rendered)
                if count <= self.fetch_max_tokens:
                    return rendered, True
                strings = sorted(
                    (
                        (len(value), key, value)
                        for key, value in preview.items()
                        if isinstance(value, str) and len(value) > 16
                    ),
                    reverse=True,
                )
                if not strings:
                    break
                _, key, value = strings[0]
                ratio = max(0.1, min(0.8, self.fetch_max_tokens / max(1, count) * 0.8))
                keep = max(16, int(len(value) * ratio))
                preview[key] = value[: max(13, keep - 3)] + "..."
        fallback = {
            "status": "ok",
            "result_id": result_id,
            "matches_found": int(meta.get("matches_found") or 0),
            "matches_returned": 0,
            "source_truncated": True,
            "token_truncated": True,
            "content": "[]",
        }
        pointers = meta.get("match_pointers")
        if isinstance(pointers, list) and pointers:
            fallback["match_pointers"] = pointers[:1]
        return json.dumps(fallback, separators=(",", ":")), True

    def measurement_baseline(
        self, messages: list[dict], tools: list[dict], ctx: MiddlewareContext
    ) -> tuple[list[dict], list[dict]]:
        out: list[dict] = []
        for message in messages:
            content = str(message.get("content") or "")
            result_id = artifact_id_from_reference(content)
            if message.get("role") != "tool" or result_id is None:
                out.append(message)
                continue
            artifact = self.store.get(ctx.session_id, result_id)
            out.append(
                {**message, "content": artifact.content}
                if artifact is not None else message
            )
        return out, tools

    def _bounded_fetch_response(
        self,
        *,
        result_id: str,
        text: str,
        meta: dict[str, Any],
        source_truncated: bool,
        source_start_char: int,
        has_more_chars: bool,
    ) -> tuple[str, bool]:
        base = {
            "status": "ok",
            "result_id": result_id,
            **meta,
            "source_truncated": source_truncated,
            "token_truncated": False,
            "content": "",
        }
        token_truncated = False
        candidate = text
        if self.tokenizer_model:
            tokenizer = self._get_tokenizer()
            ids = tokenizer.encode(text, add_special_tokens=False)
            overhead = self._count(json.dumps(base, ensure_ascii=False, separators=(",", ":")))
            take = min(len(ids), max(0, self.fetch_max_tokens - overhead - 16))
            token_truncated = take < len(ids)
            candidate = tokenizer.decode(ids[:take], skip_special_tokens=True)
        else:
            char_limit = max(0, (self.fetch_max_tokens - 64) * 4)
            token_truncated = len(text) > char_limit
            candidate = text[:char_limit]

        for _ in range(8):
            base["content"] = candidate
            base["token_truncated"] = token_truncated
            rendered = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
            count = self._count(rendered)
            if count <= self.fetch_max_tokens:
                base["next_start_char"] = (
                    source_start_char + len(candidate)
                    if token_truncated or has_more_chars else None
                )
                rendered = json.dumps(base, ensure_ascii=False, separators=(",", ":"))
                if self._count(rendered) <= self.fetch_max_tokens:
                    return rendered, token_truncated
                base.pop("next_start_char", None)
            token_truncated = True
            if not candidate:
                return rendered, token_truncated
            keep = max(0, int(len(candidate) * (self.fetch_max_tokens / count) * 0.9))
            candidate = candidate[:keep]
        base["content"] = ""
        base["token_truncated"] = True
        return json.dumps(base, ensure_ascii=False, separators=(",", ":")), True

    def _head_tail_fallback(self, content: str) -> str:
        if self.tokenizer_model:
            tokenizer = self._get_tokenizer()
            ids = tokenizer.encode(content, add_special_tokens=False)
            if len(ids) <= self.fallback_head_tokens + self.fallback_tail_tokens:
                return content
            head = tokenizer.decode(
                ids[: self.fallback_head_tokens], skip_special_tokens=True
            )
            tail = tokenizer.decode(
                ids[-self.fallback_tail_tokens :], skip_special_tokens=True
            )
        else:
            head = content[: self.fallback_head_tokens * 4]
            tail = content[-self.fallback_tail_tokens * 4 :]
        return f"{head}\n[... tool result omitted: store unavailable ...]\n{tail}"

    def _log_event(self, ctx: MiddlewareContext, payload: dict[str, Any]) -> None:
        if not self._event_log_path:
            return
        event = {"session_id": ctx.session_id, "step": ctx.step, "ts": time.time(), **payload}
        try:
            Path(self._event_log_path).parent.mkdir(parents=True, exist_ok=True)
            with _EVENT_LOCK:
                with open(self._event_log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            return

    def close(self) -> None:
        self.store.close()
