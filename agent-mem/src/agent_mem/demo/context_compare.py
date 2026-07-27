"""Readable, bounded HTML comparisons for F2/F3 context telemetry."""

from __future__ import annotations

import difflib
import html
import json
import re
from collections import Counter
from typing import Any

_CONTENT_LIMIT = 8_000
_RAW_LIMIT = 16_000
_DIFF_LIMIT = 7_000
_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
_LEX_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[^\W_]+|_|[^\w\s]",
    re.UNICODE,
)
_ALIGNMENT_JUNK = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "with",
    ",", ".", ":", ";", "!", "?", "-", "_", "(", ")", "[", "]", "{", "}",
}

_STYLE = """
<style>
.ctx-pane{border:1px solid #d0d7de;border-radius:6px;background:#fff;color:#1f2328;
  min-height:420px;overflow:hidden;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.ctx-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;
  padding:12px 14px 10px;border-bottom:1px solid #d8dee4;background:#f6f8fa}
.ctx-head h3{font-size:15px;line-height:1.3;margin:2px 0 0;letter-spacing:0}
.ctx-kicker{font-size:11px;color:#57606a;font-weight:700;letter-spacing:0}
.ctx-token{flex:none;font-size:13px;font-weight:700;color:#0969da;background:#ddf4ff;
  border:1px solid #b6e3ff;border-radius:999px;padding:4px 9px;font-variant-numeric:tabular-nums}
.ctx-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-bottom:1px solid #d8dee4}
.ctx-metric{padding:9px 12px;border-right:1px solid #d8dee4;min-width:0}
.ctx-metric:last-child{border-right:0}.ctx-metric b{display:block;font-size:14px;
  font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.ctx-metric span{font-size:11px;color:#57606a}
.ctx-meta{padding:8px 14px;color:#57606a;font-size:12px;border-bottom:1px solid #d8dee4;
  overflow-wrap:anywhere}.ctx-content{height:300px;overflow:auto;margin:0;padding:12px 14px;
  background:#fff;color:#24292f;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;
  white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}
.ctx-content.ctx-empty{color:#6e7781;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.ctx-section{padding:10px 14px;border-top:1px solid #d8dee4;background:#f6f8fa}
.ctx-section-title{display:flex;align-items:center;justify-content:space-between;gap:8px;
  font-size:12px;font-weight:700;margin-bottom:7px}.ctx-legend{font-size:11px;font-weight:400;color:#57606a}
.ctx-diff{max-height:230px;overflow:auto;margin:0;padding:10px;border:1px solid #d8dee4;
  border-radius:4px;background:#fff;color:#24292f;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;
  white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}
.ctx-del{background:#ffebe9;color:#82071e;text-decoration:line-through;text-decoration-thickness:1px}
.ctx-ins{background:#dafbe1;color:#116329}.ctx-fetch{padding:9px 10px;border-left:3px solid #bf8700;
  background:#fff8c5;font-size:12px;overflow-wrap:anywhere}.ctx-fetch pre{max-height:140px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font:11px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.ctx-raw{border-top:1px solid #d8dee4;padding:8px 14px;background:#f6f8fa;font-size:12px}
.ctx-raw summary{cursor:pointer;color:#57606a}.ctx-raw pre{max-height:260px;overflow:auto;
  white-space:pre-wrap;word-break:break-word;font:11px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
.ctx-state{display:inline-block;border-radius:999px;padding:2px 7px;background:#fff8c5;
  color:#7d4e00;border:1px solid #d4a72c;font-size:11px;font-weight:700}
.ctx-diff-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px}
.ctx-diff-track{min-width:0}.ctx-diff-track b{display:block;font-size:11px;margin:0 0 5px;color:#57606a}
.ctx-align-summary{font-size:11px;color:#57606a;font-weight:400}
@media(max-width:700px){.ctx-head{flex-direction:column}.ctx-metrics{grid-template-columns:1fr}
  .ctx-metric{border-right:0;border-bottom:1px solid #d8dee4}.ctx-metric:last-child{border-bottom:0}
  .ctx-diff-grid{grid-template-columns:1fr}}
@media(prefers-color-scheme:dark){.ctx-pane{background:#171717;color:#f0f0f0;border-color:#454545}
  .ctx-head,.ctx-section,.ctx-raw{background:#222;border-color:#454545}.ctx-head{border-color:#454545}
  .ctx-kicker,.ctx-meta,.ctx-metric span,.ctx-legend,.ctx-raw summary,.ctx-diff-track b,
  .ctx-align-summary{color:#b6b6b6}
  .ctx-token{color:#9ecbff;background:#12324a;border-color:#245b82}.ctx-metrics,.ctx-meta,
  .ctx-metric,.ctx-section,.ctx-raw{border-color:#454545}.ctx-content,.ctx-diff{background:#151515;
  color:#ededed;border-color:#454545}.ctx-content.ctx-empty{color:#aaa}.ctx-del{background:#4a1d20;color:#ffb3b8}
  .ctx-ins{background:#153b24;color:#9be9a8}.ctx-fetch{background:#3b3215;color:#f1df9b}}
</style>
"""

_REASONS = {
    "history_shorter_than_keep_hot": "历史仍在热窗口",
    "below_trigger": "未达到触发阈值",
    "first_compression": "首次压缩",
    "recompress_delta_reached": "新增量达到重压阈值",
    "cached_compression_reused": "复用上次压缩结果",
    "tool_disabled": "该工具已禁用外置",
    "reference_not_smaller": "短引用没有 token 收益",
    "store_error_passthrough": "存储失败，原文直通",
}


def _safe(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _bounded(value: Any, limit: int = _CONTENT_LIMIT) -> tuple[str, bool]:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text, False
    return text[:limit] + f"\n\n[展示截断，完整内容共 {len(text):,} 字符]", True


def _preview_text(value: Any) -> tuple[str, bool]:
    if isinstance(value, dict) and "text" in value:
        text = str(value.get("text") or "")
        return text, bool(value.get("truncated"))
    return ("" if value is None else str(value)), False


def _pretty_text(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _messages_text(preview: dict[str, Any] | None) -> tuple[str, bool]:
    if not preview:
        return "", False
    blocks: list[str] = []
    truncated = bool(preview.get("messages_truncated"))
    for index, message in enumerate(preview.get("messages") or [], 1):
        role = str(message.get("role") or "unknown").upper()
        qualifiers = [
            str(value)
            for value in (message.get("name"), message.get("tool_call_id"))
            if value
        ]
        heading = f"[{index:02d}] {role}"
        if qualifiers:
            heading += " · " + " · ".join(qualifiers)
        content, content_truncated = _preview_text(message.get("content"))
        truncated = truncated or content_truncated
        parts = [heading]
        if content:
            parts.append(_pretty_text(content))
        for tool_call in message.get("tool_calls") or []:
            arguments, args_truncated = _preview_text(tool_call.get("arguments"))
            truncated = truncated or args_truncated
            parts.append(
                f"TOOL CALL · {tool_call.get('name') or '?'} · {tool_call.get('id') or '?'}\n"
                f"{_pretty_text(arguments)}"
            )
        blocks.append("\n".join(parts))
    text, bounded = _bounded("\n\n".join(blocks))
    return text, truncated or bounded


def _messages_payload_text(
    preview: dict[str, Any] | None,
    *,
    strip_compressed_prefix: bool = False,
) -> str:
    """Flatten only user/model/tool payloads, excluding UI-only role/index headings."""
    if not preview:
        return ""
    blocks: list[str] = []
    for message in preview.get("messages") or []:
        content, _ = _preview_text(message.get("content"))
        if content:
            if strip_compressed_prefix and content.startswith("[compressed history]\n"):
                content = content.removeprefix("[compressed history]\n")
            blocks.append(_pretty_text(content))
        for tool_call in message.get("tool_calls") or []:
            arguments, _ = _preview_text(tool_call.get("arguments"))
            if arguments:
                blocks.append(_pretty_text(arguments))
    return _bounded("\n\n".join(blocks), _DIFF_LIMIT)[0]


def _fmt_tokens(value: Any) -> str:
    return f"{value:,}" if isinstance(value, int) else "--"


def _state_text(action: Any, reason: Any) -> str:
    action_text = {
        "compress": "已压缩",
        "reuse": "复用压缩",
        "skip": "原文直通",
    }.get(str(action), str(action or "等待事件"))
    reason_text = _REASONS.get(str(reason), str(reason or ""))
    return f"{action_text} · {reason_text}" if reason_text else action_text


def _metrics_html(before: Any, after: Any) -> str:
    before_int = before if isinstance(before, int) else None
    after_int = after if isinstance(after, int) else None
    saved = before_int - after_int if before_int is not None and after_int is not None else None
    percent = saved / max(1, before_int) * 100 if saved is not None else None
    saved_text = (
        f"{saved:,}"
        if saved is not None and saved >= 0
        else f"增加 {-saved:,}"
        if saved is not None
        else "--"
    )
    percent_text = f"{percent:.2f}%" if percent is not None else "--"
    return (
        '<div class="ctx-metrics">'
        f'<div class="ctx-metric"><b>{_fmt_tokens(before_int)}</b><span>before token</span></div>'
        f'<div class="ctx-metric"><b>{_fmt_tokens(after_int)}</b><span>after token</span></div>'
        f'<div class="ctx-metric"><b>{saved_text} · {percent_text}</b><span>节省 token · 缩减率</span></div>'
        "</div>"
    )


def _diff_html(before: str, after: str) -> str:
    before, _ = _bounded(before, _DIFF_LIMIT)
    after, _ = _bounded(after, _DIFF_LIMIT)
    if not before and not after:
        return '<pre class="ctx-diff">等待可比较内容</pre>'
    before_tokens = _TOKEN_RE.findall(before)
    after_tokens = _TOKEN_RE.findall(after)
    matcher = difflib.SequenceMatcher(None, before_tokens, after_tokens, autojunk=False)
    pieces: list[str] = []
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
        if opcode == "equal":
            pieces.append(_safe("".join(before_tokens[a0:a1])))
        elif opcode == "delete":
            pieces.append(f'<span class="ctx-del">{_safe("".join(before_tokens[a0:a1]))}</span>')
        elif opcode == "insert":
            pieces.append(f'<span class="ctx-ins">{_safe("".join(after_tokens[b0:b1]))}</span>')
        else:
            pieces.append(f'<span class="ctx-del">{_safe("".join(before_tokens[a0:a1]))}</span>')
            pieces.append(f'<span class="ctx-ins">{_safe("".join(after_tokens[b0:b1]))}</span>')
    return f'<pre class="ctx-diff">{"".join(pieces)}</pre>'


def _lex_units(text: str) -> tuple[list[tuple[str, str]], str]:
    """Split lexical tokens while retaining their leading whitespace for rendering."""
    units: list[tuple[str, str]] = []
    cursor = 0
    for match in _LEX_RE.finditer(text):
        units.append((text[cursor:match.start()], match.group(0)))
        cursor = match.end()
    return units, text[cursor:]


def _compression_alignment(before: str, after: str) -> tuple[set[int], set[int]]:
    """Align stable lexical anchors without letting whitespace/common repeats drive matches."""
    before_units, _ = _lex_units(before)
    after_units, _ = _lex_units(after)
    before_keys = [token.casefold() for _, token in before_units]
    after_keys = [token.casefold() for _, token in after_units]
    frequencies = Counter([*before_keys, *after_keys])
    popularity_cutoff = max(3, min(len(before_keys), len(after_keys)) // 50)

    def is_junk(token: str) -> bool:
        return token in _ALIGNMENT_JUNK or frequencies[token] > popularity_cutoff

    matcher = difflib.SequenceMatcher(
        is_junk,
        before_keys,
        after_keys,
        autojunk=True,
    )
    matched_before: set[int] = set()
    matched_after: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched_before.update(range(block.a, block.a + block.size))
        matched_after.update(range(block.b, block.b + block.size))

    # Compression can move or collapse message boundaries. Exact unique phrases remain
    # retained even when a global monotonic diff chooses a different repeated anchor.
    ngram_size = 4
    before_ngrams: dict[tuple[str, ...], list[int]] = {}
    after_ngrams: dict[tuple[str, ...], list[int]] = {}
    for index in range(max(0, len(before_keys) - ngram_size + 1)):
        key = tuple(before_keys[index:index + ngram_size])
        before_ngrams.setdefault(key, []).append(index)
    for index in range(max(0, len(after_keys) - ngram_size + 1)):
        key = tuple(after_keys[index:index + ngram_size])
        after_ngrams.setdefault(key, []).append(index)
    for key, before_starts in before_ngrams.items():
        after_starts = after_ngrams.get(key) or []
        if len(before_starts) != 1 or len(after_starts) != 1:
            continue
        before_start = before_starts[0]
        after_start = after_starts[0]
        matched_before.update(range(before_start, before_start + ngram_size))
        matched_after.update(range(after_start, after_start + ngram_size))
    return matched_before, matched_after


def _render_alignment_track(
    text: str,
    matched: set[int],
    *,
    changed_class: str,
) -> str:
    units, tail = _lex_units(text)
    pieces: list[str] = []
    for index, (prefix, token) in enumerate(units):
        pieces.append(_safe(prefix))
        escaped = _safe(token)
        pieces.append(
            escaped
            if index in matched
            else f'<span class="{changed_class}">{escaped}</span>'
        )
    pieces.append(_safe(tail))
    return "".join(pieces)


def _is_word_token(token: str) -> bool:
    return bool(token) and (token[0].isalnum() or token[0] == "_")


def _compression_diff_html(before: str, after: str) -> str:
    """Render two non-ambiguous F2 tracks instead of pairing unrelated replacements."""
    before, _ = _bounded(before, _DIFF_LIMIT)
    after, _ = _bounded(after, _DIFF_LIMIT)
    if not before and not after:
        return '<pre class="ctx-diff">等待可比较内容</pre>'
    matched_before, matched_after = _compression_alignment(before, after)
    before_units, _ = _lex_units(before)
    after_units, _ = _lex_units(after)
    before_words = {i for i, (_, token) in enumerate(before_units) if _is_word_token(token)}
    after_words = {i for i, (_, token) in enumerate(after_units) if _is_word_token(token)}
    retained = len(before_words & matched_before)
    removed = len(before_words - matched_before)
    added = len(after_words - matched_after)
    summary = f"匹配保留 {retained} · 删除 {removed} · 新增/改写 {added} 个词"
    return (
        f'<div class="ctx-align-summary">{summary}</div>'
        '<div class="ctx-diff-grid">'
        '<div class="ctx-diff-track"><b>原文保留 / 删除</b>'
        f'<pre class="ctx-diff">{_render_alignment_track(before, matched_before, changed_class="ctx-del")}</pre></div>'
        '<div class="ctx-diff-track"><b>压缩后保留 / 新增或改写</b>'
        f'<pre class="ctx-diff">{_render_alignment_track(after, matched_after, changed_class="ctx-ins")}</pre></div>'
        "</div>"
    )


def _raw_details(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    raw, truncated = _bounded(raw, _RAW_LIMIT)
    suffix = "（展示已截断）" if truncated else ""
    return (
        '<details class="ctx-raw"><summary>原始 telemetry 数据'
        f'{suffix}</summary><pre>{_safe(raw)}</pre></details>'
    )


def waiting_panel(title: str) -> str:
    return (
        _STYLE
        + '<div class="ctx-pane"><div class="ctx-head"><div>'
        + '<span class="ctx-kicker">WAITING</span>'
        + f'<h3>{_safe(title)}</h3></div><span class="ctx-state">等待任务</span></div>'
        + '<pre class="ctx-content ctx-empty">运行任务后将在这里显示真实内容和完整 token 计数。</pre>'
        + "</div>"
    )


def render_f2_panels(f2: dict[str, Any], *, enabled: bool) -> tuple[str, str]:
    """Render F2 canonical/sent content with full-token metrics and bounded word diff."""
    before_preview = f2.get("cold_before") or {}
    after_preview = f2.get("cold_after") or {}
    before_text, before_truncated = _messages_text(before_preview)
    after_text, after_truncated = _messages_text(after_preview)
    before_diff_text = _messages_payload_text(before_preview)
    after_diff_text = _messages_payload_text(after_preview, strip_compressed_prefix=True)
    if not after_text and after_preview.get("compressed_text"):
        after_text, compressed_truncated = _preview_text(after_preview.get("compressed_text"))
        after_text, bounded_after = _bounded(_pretty_text(after_text))
        after_truncated = after_truncated or compressed_truncated or bounded_after
    compressed_diff_text, _ = _preview_text(after_preview.get("compressed_text"))
    if compressed_diff_text and not after_diff_text:
        after_diff_text = _bounded(_pretty_text(compressed_diff_text), _DIFF_LIMIT)[0]
    action = f2.get("action")
    reason = f2.get("reason")
    before_tokens = before_preview.get("tokens") if before_preview else None
    after_tokens = after_preview.get("tokens") if after_preview else None
    if enabled and action == "skip" and before_preview and not after_preview:
        after_text = before_text
        after_diff_text = before_diff_text
        after_tokens = before_tokens
    state = _state_text(action, reason) if enabled else "F2 未启用"
    before_note = "内容为有界 preview；token 对应完整冷历史" if before_truncated else "token 对应完整冷历史"
    after_note = "发送 preview 已截断；token 对应完整发送副本" if after_truncated else "token 对应完整发送副本"
    if not before_text:
        before_text = "当前步骤尚无冷历史。" if enabled else "当前模式未启用 F2。"
    if not after_text:
        after_text = "尚未产生动态压缩副本。"

    before_html = (
        _STYLE
        + '<div class="ctx-pane"><div class="ctx-head"><div><span class="ctx-kicker">F2 · BEFORE</span>'
        + '<h3>待压缩冷历史</h3></div>'
        + f'<span class="ctx-token">{_fmt_tokens(before_tokens)} tokens</span></div>'
        + f'<div class="ctx-meta"><span class="ctx-state">{_safe(state)}</span> · {_safe(before_note)}</div>'
        + f'<pre class="ctx-content{(" ctx-empty" if not before_preview else "")}">{_safe(before_text)}</pre>'
        + _raw_details(before_preview)
        + "</div>"
    )
    method = f2.get("method") or "--"
    compressor_tokens = ""
    if isinstance(f2.get("origin_tokens"), int) and isinstance(f2.get("compressed_tokens"), int):
        compressor_tokens = (
            f" · 压缩器正文 {_fmt_tokens(f2['origin_tokens'])} → "
            f"{_fmt_tokens(f2['compressed_tokens'])} tokens"
        )
    after_html = (
        _STYLE
        + '<div class="ctx-pane"><div class="ctx-head"><div><span class="ctx-kicker">F2 · AFTER</span>'
        + '<h3>实际发送的冷历史副本</h3></div>'
        + f'<span class="ctx-token">{_fmt_tokens(after_tokens)} tokens</span></div>'
        + _metrics_html(before_tokens, after_tokens)
        + f'<div class="ctx-meta">method={_safe(method)} · {_safe(after_note + compressor_tokens)}</div>'
        + f'<pre class="ctx-content{(" ctx-empty" if not after_preview and action != "skip" else "")}">{_safe(after_text)}</pre>'
        + '<div class="ctx-section"><div class="ctx-section-title"><span>原文保留与压缩后改写</span>'
        + '<span class="ctx-legend"><span class="ctx-del">原文删除</span> · <span class="ctx-ins">新增/改写</span> · 无底色=匹配保留</span></div>'
        + _compression_diff_html(
            before_diff_text if before_preview else "",
            after_diff_text if after_preview or action == "skip" else "",
        )
        + "</div>"
        + _raw_details(after_preview or {"action": action, "reason": reason})
        + "</div>"
    )
    return before_html, after_html


def render_f3_panels(f3_state: dict[str, Any], *, enabled: bool) -> tuple[str, str]:
    """Render F3 original tool output versus the short reference sent to the model."""
    latest = f3_state.get("latest") or {}
    original = latest.get("original") or {}
    externalized = latest.get("externalized") or {}
    before_text, before_truncated = _preview_text(original.get("preview"))
    before_text, bounded_before = _bounded(_pretty_text(before_text))
    before_truncated = before_truncated or bounded_before
    after_text, after_truncated = _preview_text(externalized.get("reference"))
    after_text, bounded_after = _bounded(_pretty_text(after_text))
    after_truncated = after_truncated or bounded_after
    before_tokens = original.get("tokens") if original else None
    after_tokens = externalized.get("reference_tokens") if externalized else None
    phase = latest.get("phase")
    reason = latest.get("reason")
    if enabled and phase == "passthrough" and original and not externalized:
        after_text = before_text
        after_tokens = before_tokens
    state = "F3 未启用" if not enabled else {
        "externalized": "已外置为短引用",
        "passthrough": "原工具结果直通",
        "observed": "已观察工具结果",
        "storing": "正在写入 ArtifactStore",
        "failed": "外置失败",
    }.get(str(phase), str(phase or "等待工具结果"))
    reason_text = _REASONS.get(str(reason), str(reason or ""))
    if not before_text:
        before_text = "等待业务工具返回数据。" if enabled else "当前模式未启用 F3。"
    if not after_text:
        after_text = "尚未产生外置短引用。"

    tool_name = latest.get("tool_name") or "--"
    before_note = f"tool={tool_name}"
    if before_truncated:
        before_note += " · preview 已截断，token 仍对应完整原文"
    else:
        before_note += " · token 对应完整原文"
    before_html = (
        _STYLE
        + '<div class="ctx-pane"><div class="ctx-head"><div><span class="ctx-kicker">F3 · BEFORE</span>'
        + '<h3>业务工具原始结果</h3></div>'
        + f'<span class="ctx-token">{_fmt_tokens(before_tokens)} tokens</span></div>'
        + f'<div class="ctx-meta"><span class="ctx-state">{_safe(state)}</span> · {_safe(before_note)}</div>'
        + f'<pre class="ctx-content{(" ctx-empty" if not original else "")}">{_safe(before_text)}</pre>'
        + _raw_details({"tool_name": tool_name, "arguments": latest.get("arguments"), "original": original})
        + "</div>"
    )

    synopsis = externalized.get("synopsis") or {}
    result_id = str(externalized.get("result_id") or "")
    result_short = result_id[:12] + ("…" if len(result_id) > 12 else "")
    after_note = (
        f"type={externalized.get('content_type') or '--'} · result_id={result_short or '--'}"
    )
    if reason_text:
        after_note += f" · {reason_text}"
    if after_truncated:
        after_note += " · reference preview 已截断"
    fetches = f3_state.get("fetches") or []
    fetch_html = ""
    if fetches:
        latest_fetch = fetches[-1]
        response, _ = _preview_text(latest_fetch.get("response"))
        selector = json.dumps(latest_fetch.get("selector") or {}, ensure_ascii=False)
        fetch_html = (
            '<div class="ctx-section"><div class="ctx-fetch"><b>最近一次按需取回</b> · '
            f'{_fmt_tokens(latest_fetch.get("fetch_tokens"))} tokens · selector={_safe(selector)}'
            f'<pre>{_safe(_bounded(_pretty_text(response), 3_000)[0])}</pre></div></div>'
        )
    after_html = (
        _STYLE
        + '<div class="ctx-pane"><div class="ctx-head"><div><span class="ctx-kicker">F3 · AFTER</span>'
        + '<h3>进入 Agent 历史的 synopsis / 短引用</h3></div>'
        + f'<span class="ctx-token">{_fmt_tokens(after_tokens)} tokens</span></div>'
        + _metrics_html(before_tokens, after_tokens)
        + f'<div class="ctx-meta">{_safe(after_note)}</div>'
        + f'<pre class="ctx-content{(" ctx-empty" if not externalized and phase != "passthrough" else "")}">{_safe(after_text)}</pre>'
        + '<div class="ctx-section"><div class="ctx-section-title"><span>原工具结果 → 短引用的词级差异</span>'
        + '<span class="ctx-legend"><span class="ctx-del">移出 Prompt</span> · <span class="ctx-ins">短引用新增</span></span></div>'
        + _diff_html(before_text if original else "", after_text if externalized or phase == "passthrough" else "")
        + "</div>"
        + fetch_html
        + _raw_details({"externalized": externalized, "synopsis": synopsis, "fetches": fetches[-3:]})
        + "</div>"
    )
    return before_html, after_html
