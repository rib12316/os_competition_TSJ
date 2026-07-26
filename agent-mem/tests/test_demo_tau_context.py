"""Tests for the tau-bench context panels and mode mapping."""

from __future__ import annotations

from agent_mem.context_telemetry import ContextEventBuffer
from agent_mem.demo.chat_app import (
    _tau_context_stack,
    _tau_context_view,
    _tau_user_sim_settings,
)
from agent_mem.demo.tau_bench_ui import run_tau_task_streaming
from agent_mem.middleware import MiddlewareContext


def test_tau_context_modes_build_expected_middleware_order():
    expected = {
        "baseline": [],
        "F2": ["compress"],
        "F3": ["lazyload"],
        "F2+F3": ["lazyload", "compress"],
    }
    for mode, names in expected.items():
        stack = _tau_context_stack(mode, "Qwen2.5-7B-Instruct")
        assert stack.names == names


def test_tau_frontend_defaults_to_mimo_user_simulator():
    settings = _tau_user_sim_settings()
    assert settings["model"] == "mimo-v2.5-pro"
    assert settings["provider"] == "openai"
    assert settings["api_base"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert settings["api_key_env"] == "MIMO_KEY"
    assert run_tau_task_streaming.__kwdefaults__["user_model"] == "mimo-v2.5-pro"


def test_tau_context_view_exposes_prompt_f2_and_f3_fields():
    buffer = ContextEventBuffer()
    ctx = MiddlewareContext("tau-0", event_sink=buffer)
    ctx.bump_step()
    ctx.emit("f2.history_ready", {
        "phase": "ready",
        "action": "compress",
        "cold_before": {"tokens": 9000, "messages": []},
    })
    ctx.emit("f2.compress_finished", {
        "phase": "compressed",
        "cold_after": {"tokens": 5000, "compressed_text": {"text": "short"}},
    })
    ctx.emit("f3.tool_result_externalized", {
        "operation_id": "tau-0:1:1",
        "phase": "externalized",
        "tool_name": "retrieve_documents",
        "original": {"tokens": 9000},
        "externalized": {"reference_tokens": 200},
    })
    ctx.emit("prompt.completed", {
        "original_prompt_tokens": 10000,
        "transformed_prompt_tokens": 1500,
        "saved_tokens": 8500,
    })

    prompt, before, after, f3_before, f3_after = _tau_context_view(
        buffer,
        session_id="tau-0",
        mode="F2+F3",
        middleware_names=["lazyload", "compress"],
    )
    assert "10,000" not in prompt  # Markdown uses plain integer formatting.
    assert "10000" in prompt and "1500" in prompt
    assert before["tokens"] == 9000
    assert after["tokens"] == 5000
    assert f3_before["tool_name"] == "retrieve_documents"
    assert f3_after["reference_tokens"] == 200
