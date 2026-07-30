"""F5 tau-bench runtime tests without network or an engine."""

from __future__ import annotations

import os

import litellm
import pytest

from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.demo import tau_bench_ui


def test_scripted_user_completion_is_task_specific_then_deterministic():
    first = tau_bench_ui._scripted_user_completion(
        messages=[
            {
                "role": "system",
                "content": "Instruction: Cancel order #W1.\nRules:\n- one line",
            },
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
    )
    assert first.choices[0].message.content == "Cancel order #W1."
    assert first.choices[0].message.model_dump() == {
        "role": "assistant",
        "content": "Cancel order #W1.",
    }

    later = tau_bench_ui._scripted_user_completion(
        messages=[
            {"role": "system", "content": "Instruction: Cancel order #W1.\nRules:"},
            {"role": "assistant", "content": "Cancel order #W1."},
            {"role": "user", "content": "What can I help with?"},
        ]
    )
    assert later.choices[0].message.content.startswith("Please continue")


@pytest.mark.parametrize(
    ("user_api_base", "user_api_key", "expected_base", "expected_key", "expected_model"),
    [
        ("https://mimo.example/v1", "mimo-key", "https://mimo.example/v1", "mimo-key", "mimo"),
        ("https://mimo.example/v1", None, "http://engine/v1", "engine-key", "local-model"),
    ],
)
def test_concurrent_runner_scopes_litellm_endpoint(
    monkeypatch,
    user_api_base,
    user_api_key,
    expected_base,
    expected_key,
    expected_model,
):
    observed: list[tuple[str, str, str, str, str]] = []

    def fake_run(tid, convo_store, **kwargs):
        observed.append((
            os.environ["OPENAI_API_BASE"],
            os.environ["OPENAI_API_KEY"],
            litellm.api_base,
            litellm.api_key,
            kwargs["user_model"],
        ))
        return TaskRunResult(
            task_id=tid,
            reward=1.0,
            success=True,
            latency_ms=10.0,
            n_steps=1,
            error=None,
            prompt_tokens=12,
        )

    monkeypatch.setattr(tau_bench_ui, "run_task_into_convo", fake_run)
    monkeypatch.delenv("MIMO_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://previous.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "previous-env-key")
    previous_base = litellm.api_base
    previous_key = litellm.api_key
    litellm.api_base = "https://previous-global.example/v1"
    litellm.api_key = "previous-global-key"
    try:
        updates = list(tau_bench_ui.run_concurrent_streaming(
            domain="retail",
            split="test",
            task_ids=[0],
            concurrency=1,
            engine_url="http://engine/v1",
            model="local-model",
            convo_store={},
            api_key="engine-key",
            user_model="mimo",
            user_api_base=user_api_base,
            user_api_key=user_api_key,
            max_steps=1,
        ))

        assert observed == [(
            expected_base,
            expected_key,
            expected_base,
            expected_key,
            expected_model,
        )]
        assert updates[-1][2][0]["success"] is True
        assert os.environ["OPENAI_API_BASE"] == "https://previous.example/v1"
        assert os.environ["OPENAI_API_KEY"] == "previous-env-key"
        assert litellm.api_base == "https://previous-global.example/v1"
        assert litellm.api_key == "previous-global-key"
    finally:
        litellm.api_base = previous_base
        litellm.api_key = previous_key
