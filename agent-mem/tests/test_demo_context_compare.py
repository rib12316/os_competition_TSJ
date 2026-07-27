"""Readable F2/F3 telemetry comparison rendering tests."""

from __future__ import annotations

from agent_mem.demo.context_compare import (
    render_f2_panels,
    render_f3_panels,
    waiting_panel,
)


def _content(text: str, *, truncated: bool = False) -> dict:
    return {"text": text, "chars": len(text), "truncated": truncated}


def test_f2_panels_show_content_tokens_and_word_diff():
    f2 = {
        "action": "compress",
        "reason": "first_compression",
        "method": "llmlingua2",
        "origin_tokens": 120,
        "compressed_tokens": 55,
        "cold_before": {
            "tokens": 120,
            "message_count": 1,
            "messages": [{
                "role": "assistant",
                "content": _content(
                    "Order A-100 is pending. The battery lasts ten hours. Remove this detail."
                ),
            }],
        },
        "cold_after": {
            "tokens": 55,
            "message_count": 1,
            "messages": [{
                "role": "system",
                "content": _content("Order A-100 pending. Battery lasts ten hours."),
            }],
            "compressed_text": _content("Order A-100 pending. Battery lasts ten hours."),
        },
    }

    before, after = render_f2_panels(f2, enabled=True)

    assert "Order A-100 is pending" in before
    assert "120 tokens" in before
    assert "55 tokens" in after
    assert "65 · 54.17%" in after
    assert "Order A-100 pending" in after
    assert 'class="ctx-del"' in after
    assert 'class="ctx-ins"' in after
    assert "压缩器正文 120 → 55 tokens" in after
    after_markup = after.split("</style>", 1)[1]
    assert '<pre class="ctx-content' not in after_markup
    assert '<details class="ctx-raw"' not in after_markup


def test_f2_skip_renders_identical_sent_copy():
    f2 = {
        "action": "skip",
        "reason": "below_trigger",
        "cold_before": {
            "tokens": 80,
            "messages": [{"role": "assistant", "content": _content("unchanged text")}],
        },
    }

    before, after = render_f2_panels(f2, enabled=True)

    assert "原文直通" in before
    assert "unchanged text" in after
    assert "80 tokens" in after
    assert "0 · 0.00%" in after


def test_f2_alignment_does_not_pair_unrelated_words_after_line_drift():
    f2 = {
        "action": "compress",
        "reason": "first_compression",
        "cold_before": {
            "tokens": 60,
            "messages": [{
                "role": "assistant",
                "content": _content(
                    "Shared opening sentence.\nMechanical Keyboard details and repeated repeated text."
                ),
            }],
        },
        "cold_after": {
            "tokens": 20,
            "messages": [
                {
                    "role": "system",
                    "content": _content(
                        "[compressed history]\nShared opening sentence.\nNext Steps are concise."
                    ),
                },
                {
                    "role": "assistant",
                    "content": _content("New cold fact appended verbatim."),
                },
            ],
            "compressed_text": _content("Shared opening sentence.\nNext Steps are concise."),
        },
    }

    _, after = render_f2_panels(f2, enabled=True)

    assert "原文保留 / 删除" in after
    assert "压缩后保留 / 新增或改写" in after
    assert '<span class="ctx-del">Mechanical</span>' in after
    assert '<span class="ctx-del">Keyboard</span>' in after
    assert '<span class="ctx-ins">Next</span>' in after
    assert '<span class="ctx-ins">Steps</span>' in after
    for word in ("New", "cold", "fact", "appended", "verbatim"):
        assert f'<span class="ctx-ins">{word}</span>' in after
    assert 'ctx-del">Mechanical</span><span class="ctx-ins">Next' not in after
    assert "匹配保留" in after and "新增/改写" in after


def test_f2_alignment_keeps_exact_unique_sentence_when_blocks_move():
    opening = "Hello Yusuf, I can help discuss the exchange for your delivered order today."
    exact = (
        "The order details for #W2378156 show that it was delivered on March 15th. "
        "You received the following items:"
    )
    f2 = {
        "action": "compress",
        "cold_before": {
            "tokens": 80,
            "messages": [{
                "role": "assistant",
                "content": _content(f"{opening}\nMechanical Keyboard details.\n{exact}"),
            }],
        },
        "cold_after": {
            "tokens": 35,
            "messages": [{
                "role": "system",
                "content": _content(f"[compressed history]\n{exact}\n{opening}"),
            }],
            "compressed_text": _content(f"{exact}\n{opening}"),
        },
    }

    _, after = render_f2_panels(f2, enabled=True)
    diff_tracks = after.split('<div class="ctx-diff-grid">', 1)[1]

    assert diff_tracks.count(exact) >= 2
    assert '<span class="ctx-del">The</span> order details' not in diff_tracks
    assert '<span class="ctx-ins">The</span> order details' not in diff_tracks


def test_f3_panels_show_reference_savings_fetch_and_escape_content():
    raw = '{"documents":[{"title":"Unsafe <script>alert(1)</script>","text":"evidence"}]}'
    reference = '{"_agent_mem":"external_tool_result","result_id":"abcdef1234567890"}'
    f3 = {
        "latest": {
            "phase": "externalized",
            "tool_name": "retrieve_documents",
            "arguments": {"query": "question"},
            "original": {"preview": _content(raw), "tokens": 9000},
            "externalized": {
                "result_id": "abcdef1234567890",
                "content_type": "json",
                "reference": _content(reference),
                "reference_tokens": 200,
                "synopsis": {"kind": "json", "top_level_keys": ["documents"]},
            },
        },
        "fetches": [{
            "event": "f3.fetch_finished",
            "fetch_tokens": 120,
            "selector": {"match_field": "title", "match_value": "Unsafe"},
            "response": _content('{"status":"ok","content":"evidence"}'),
        }],
    }

    before, after = render_f3_panels(f3, enabled=True)

    assert "9,000 tokens" in before
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in before
    assert "<script>alert(1)</script>" not in before
    assert "200 tokens" in after
    assert "8,800 · 97.78%" in after
    assert "abcdef123456…" in after
    assert "最近一次按需取回" not in after
    assert 'class="ctx-del"' in after and 'class="ctx-ins"' in after
    after_markup = after.split("</style>", 1)[1]
    assert '<pre class="ctx-content' not in after_markup
    assert '<details class="ctx-raw"' not in after_markup


def test_waiting_panel_is_stable_and_readable():
    panel = waiting_panel("F2 待压缩冷历史")
    assert "WAITING" in panel
    assert "等待任务" in panel
    assert "F2 待压缩冷历史" in panel
