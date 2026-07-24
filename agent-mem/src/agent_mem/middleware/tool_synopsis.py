"""Bounded, deterministic synopsis generation for F3 tool results.

The design is adapted from ByteDance DeerFlow's tool-output synopsis, but is
kept dependency-free and rendered as F2-safe structured JSON.
"""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import Any

from agent_mem.middleware.artifact_store import ToolArtifact

_MAX_KEYS = 12
_MAX_HEADINGS = 8
_MAX_EXCERPT_CHARS = 240


def _clip(value: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: max(0, limit - 3)] + "..."


def _json_shape(value: Any, depth: int = 0) -> str:
    if isinstance(value, dict):
        keys = list(value)[:_MAX_KEYS]
        suffix = f" keys={keys}" if keys else ""
        return f"object({len(value)}{suffix})"
    if isinstance(value, list):
        child = _json_shape(value[0], depth + 1) if value and depth < 2 else ""
        suffix = f" first={child}" if child else ""
        return f"array({len(value)}{suffix})"
    if value is None:
        return "null"
    return type(value).__name__


def _json_synopsis(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    out: dict[str, Any] = {"kind": "json", "shape": _json_shape(value)}
    if isinstance(value, dict):
        out["keys"] = [str(key) for key in list(value)[:_MAX_KEYS]]
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        out["item_keys"] = [str(key) for key in list(value[0])[:_MAX_KEYS]]
    return out


class _HTMLSynopsisParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._capture = ""
        self.title = ""
        self.headings: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"title", "h1", "h2", "h3"}:
            self._capture = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture:
            self._capture = ""

    def handle_data(self, data: str) -> None:
        value = _clip(data, 160)
        if not value:
            return
        if self._capture == "title" and not self.title:
            self.title = value
        elif self._capture.startswith("h") and len(self.headings) < _MAX_HEADINGS:
            self.headings.append(value)


def _html_synopsis(content: str) -> dict[str, Any] | None:
    sample = content.lstrip()[:2048].lower()
    if not any(marker in sample for marker in ("<html", "<!doctype html", "<body", "<h1")):
        return None
    parser = _HTMLSynopsisParser()
    try:
        parser.feed(content)
    except Exception:
        return None
    return {
        "kind": "html",
        "title": parser.title,
        "headings": parser.headings,
        "line_count": len(content.splitlines()),
    }


def _csv_synopsis(content: str) -> dict[str, Any] | None:
    lines = [line for line in content.splitlines()[:20] if line.strip()]
    if len(lines) < 3:
        return None
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",\t")
        rows = list(csv.reader(io.StringIO("\n".join(lines)), dialect=dialect))
    except (csv.Error, UnicodeError):
        return None
    if len(rows) < 3 or len(rows[0]) < 2:
        return None
    width = len(rows[0])
    if sum(1 for row in rows[1:] if len(row) == width) < 2:
        return None
    return {
        "kind": "csv" if dialect.delimiter == "," else "tsv",
        "columns": [_clip(cell, 80) for cell in rows[0][:_MAX_KEYS]],
        "line_count": len(content.splitlines()),
    }


def infer_content_type(content: str, *, max_parse_bytes: int = 5_000_000) -> str:
    raw_size = len(content.encode("utf-8"))
    if raw_size <= max_parse_bytes:
        if _json_synopsis(content) is not None:
            return "application/json"
        if _html_synopsis(content) is not None:
            return "text/html"
        table = _csv_synopsis(content)
        if table is not None:
            return "text/csv" if table["kind"] == "csv" else "text/tab-separated-values"
    return "text/plain"


def build_tool_synopsis(
    content: str,
    *,
    max_parse_bytes: int = 5_000_000,
) -> tuple[str, dict[str, Any]]:
    raw_size = len(content.encode("utf-8"))
    if raw_size > max_parse_bytes:
        return "text/plain", {
            "kind": "oversized",
            "line_count": len(content.splitlines()),
            "opening": _clip(content[:_MAX_EXCERPT_CHARS]),
        }
    for content_type, parser in (
        ("application/json", _json_synopsis),
        ("text/html", _html_synopsis),
        ("text/csv", _csv_synopsis),
    ):
        synopsis = parser(content)
        if synopsis is not None:
            if synopsis.get("kind") == "tsv":
                content_type = "text/tab-separated-values"
            return content_type, synopsis
    lines = content.splitlines()
    return "text/plain", {
        "kind": "text",
        "line_count": len(lines),
        "opening": _clip(content[:_MAX_EXCERPT_CHARS]),
        "closing": _clip(content[-120:]) if len(content) > _MAX_EXCERPT_CHARS else "",
    }


def render_artifact_reference(artifact: ToolArtifact, synopsis: dict[str, Any]) -> str:
    payload = {
        "_agent_mem": "external_tool_result",
        "result_id": artifact.result_id,
        "status": "stored",
        "tool_name": artifact.tool_name,
        "content_type": artifact.content_type,
        "byte_count": artifact.byte_count,
        "token_count": artifact.token_count,
        "summary": synopsis,
        "fetch": {
            "name": "fetch_tool_result",
            "arguments": {"result_id": artifact.result_id},
        },
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def artifact_id_from_reference(content: str) -> str | None:
    try:
        value = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("_agent_mem") != "external_tool_result":
        return None
    result_id = value.get("result_id")
    return result_id if isinstance(result_id, str) and result_id.startswith("tr_") else None
