"""Realtime, frontend-neutral telemetry for F2/F3 context transformations."""

from __future__ import annotations

import copy
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContextEvent:
    """One versioned context-optimization event."""

    event: str
    session_id: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class ContextEventSink(Protocol):
    """Consumer used by Agent loops and middlewares to publish context events."""

    def emit(self, event: ContextEvent) -> None: ...


def text_preview(value: Any, *, limit: int = 4000) -> dict[str, Any]:
    """Return bounded display text plus its original size."""
    text = "" if value is None else str(value)
    limit = max(0, int(limit))
    truncated = len(text) > limit
    return {
        "text": text[:limit] if truncated else text,
        "chars": len(text),
        "truncated": truncated,
    }


def messages_preview(
    messages: list[dict[str, Any]],
    *,
    max_messages: int = 30,
    content_limit: int = 800,
) -> dict[str, Any]:
    """Build a bounded, protocol-aware preview of canonical or transformed history."""
    shown = messages[: max(0, int(max_messages))]
    items: list[dict[str, Any]] = []
    for message in shown:
        item: dict[str, Any] = {
            "role": message.get("role"),
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
            "content": text_preview(message.get("content"), limit=content_limit),
        }
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.get("id"),
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": text_preview(
                        (call.get("function") or {}).get("arguments"),
                        limit=content_limit,
                    ),
                }
                for call in tool_calls
                if isinstance(call, dict)
            ]
        items.append(item)
    return {
        "message_count": len(messages),
        "shown_count": len(items),
        "messages_truncated": len(messages) > len(items),
        "messages": items,
    }


class ContextEventBuffer:
    """Thread-safe event buffer with incremental reads and merged UI snapshots."""

    def __init__(self, *, max_events: int = 2000) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max(1, int(max_events)))
        self._sequence = 0
        self._lock = threading.RLock()

    def emit(self, event: ContextEvent) -> None:
        payload = event.to_dict()
        with self._lock:
            self._sequence += 1
            payload["sequence"] = self._sequence
            self._events.append(payload)

    def events(
        self,
        *,
        session_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return copied events suitable for direct frontend consumption."""
        with self._lock:
            events = [
                event
                for event in self._events
                if int(event.get("sequence", 0)) > int(after_sequence)
                and (session_id is None or event.get("session_id") == session_id)
            ]
            if limit is not None:
                events = events[-max(0, int(limit)) :]
            return copy.deepcopy(events)

    def snapshot(self, session_id: str | None = None) -> dict[str, Any]:
        """Merge recent F2/F3/Prompt events into a stable frontend snapshot."""
        with self._lock:
            all_events = list(self._events)
        if session_id is None and all_events:
            session_id = str(all_events[-1].get("session_id") or "")
        events = [event for event in all_events if event.get("session_id") == session_id]
        if not events:
            return {
                "schema_version": SCHEMA_VERSION,
                "sequence": 0,
                "session_id": session_id,
                "step": 0,
                "f2": {},
                "f3": {"latest": {}, "fetches": []},
                "prompt": {},
                "recent_events": [],
            }

        f2_events = [event for event in events if str(event.get("event", "")).startswith("f2.")]
        f2: dict[str, Any] = {}
        if f2_events:
            f2_step = max(int(event.get("step", 0)) for event in f2_events)
            for event in f2_events:
                if int(event.get("step", 0)) != f2_step:
                    continue
                f2.update(copy.deepcopy(event.get("data") or {}))
                f2["event"] = event.get("event")
                f2["step"] = f2_step

        operations: dict[str, dict[str, Any]] = {}
        operation_sequence: dict[str, int] = {}
        fetches: list[dict[str, Any]] = []
        for event in events:
            event_name = str(event.get("event", ""))
            if not event_name.startswith("f3."):
                continue
            data = copy.deepcopy(event.get("data") or {})
            if event_name.startswith("f3.fetch_"):
                fetches.append({"event": event_name, "step": event.get("step"), **data})
                continue
            operation_id = str(data.get("operation_id") or event.get("sequence"))
            current = operations.setdefault(operation_id, {"operation_id": operation_id})
            current.update(data)
            current["event"] = event_name
            current["step"] = event.get("step")
            operation_sequence[operation_id] = int(event.get("sequence", 0))
        latest_f3 = (
            operations[max(operation_sequence, key=operation_sequence.get)]
            if operation_sequence else {}
        )

        prompt: dict[str, Any] = {}
        for event in events:
            if str(event.get("event", "")).startswith("prompt."):
                prompt.update(copy.deepcopy(event.get("data") or {}))
                prompt["event"] = event.get("event")
                prompt["step"] = event.get("step")

        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": int(events[-1].get("sequence", 0)),
            "session_id": session_id,
            "step": max(int(event.get("step", 0)) for event in events),
            "f2": f2,
            "f3": {"latest": latest_f3, "fetches": fetches[-20:]},
            "prompt": prompt,
            "recent_events": copy.deepcopy(events[-50:]),
        }


class JsonlContextEventSink:
    """Append the same event contract consumed by the live buffer to JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: ContextEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class CompositeContextEventSink:
    """Fan out events to a live buffer, JSONL sink, or other consumers."""

    def __init__(self, *sinks: ContextEventSink) -> None:
        self.sinks = tuple(sinks)

    def emit(self, event: ContextEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)
