"""F5 judge-demo runtime tests without vLLM, Gradio, or an accelerator."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from agent_mem.demo.f5_runtime import (
    RequestAdmissionController,
    UserSimResponseCache,
    build_f5_tiers,
)


def test_build_f5_tiers_baseline_vs_admission():
    tiers = build_f5_tiers()

    assert [tier.key for tier in tiers] == ["baseline", "ours"]
    # baseline = FCFS（仅 prefix-cache），对齐 configs/f5-native.yaml
    assert tiers[0].features == ("prefix-cache",)
    assert tiers[0].priority_strategy == "fcfs"
    assert tiers[0].adaptive_admission is False
    assert tiers[0].offload_label == "关闭"
    # ours = KV-pool 准入控制（priority 协同 + 准入开、无 offload），对齐 configs/f5-evict-dynamic.yaml
    assert tiers[1].features == ("prefix-cache", "priority")
    assert tiers[1].priority_strategy == "idle"
    assert tiers[1].adaptive_admission is True
    assert tiers[1].offload_label == "关闭"


def test_build_f5_tiers_ignores_legacy_kv_mode():
    # kv_mode 已弃用（progress 不依赖 KV offload 层）；任意传/不传都返回同一组 tier、不报错。
    assert build_f5_tiers() == build_f5_tiers("SimpleCPU lazy offload")
    assert build_f5_tiers("anything") == build_f5_tiers()


def test_user_sim_cache_replays_identical_requests_and_forces_temperature_zero():
    times = iter([10.0, 12.5])
    sleeps: list[float] = []
    cache = UserSimResponseCache(
        clock=lambda: next(times), sleeper=sleeps.append
    )
    calls: list[dict] = []

    def delegate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=len(calls))

    first = cache.complete(
        delegate,
        model="mimo",
        messages=[
            {"role": "system", "content": "Instruction: exchange order"},
            {"role": "assistant", "content": "I need an exchange."},
            {"role": "user", "content": "Which item?"},
        ],
    )
    second = cache.complete(
        delegate,
        model="mimo",
        messages=[
            {"role": "system", "content": "Instruction: exchange order"},
            {"role": "assistant", "content": "I need an exchange."},
            {"role": "user", "content": "Please provide the item identifier."},
        ],
    )

    assert first.value == second.value == 1
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
    assert sleeps == [2.5]
    assert cache.snapshot() == {"entries": 1, "hits": 1, "misses": 1}


def test_fixed_request_cap_defers_then_releases_waiter():
    controller = RequestAdmissionController(
        max_requests=1,
        adaptive=False,
        kv_pct_fn=lambda: 10.0,
        poll_interval_s=0.02,
    )
    controller.before_request("tau-0", step=1, priority=0)
    admitted = threading.Event()

    def wait_for_slot() -> None:
        controller.before_request("tau-1", step=1, priority=0)
        admitted.set()

    worker = threading.Thread(target=wait_for_slot)
    worker.start()
    time.sleep(0.08)
    assert not admitted.is_set()
    assert controller.snapshot()["deferred_requests"] == 1

    controller.after_request("tau-0")
    worker.join(timeout=1.0)
    assert admitted.is_set()
    assert controller.snapshot()["max_active_requests"] == 1
    controller.after_request("tau-1")


def test_adaptive_controller_contracts_and_recovers_limit():
    kv = [90.0]
    controller = RequestAdmissionController(
        max_requests=3,
        adaptive=True,
        kv_pct_fn=lambda: kv[0],
        poll_interval_s=0.02,
        adjustment_interval_s=0.0,
    )
    controller.before_request("tau-0", step=1, priority=20)
    assert controller.snapshot()["current_limit"] == 2
    controller.after_request("tau-0")

    kv[0] = 50.0
    time.sleep(0.06)
    controller.before_request("tau-1", step=1, priority=10)
    snapshot = controller.snapshot()
    assert snapshot["current_limit"] == 3
    assert snapshot["kv_peak_pct"] == 90.0
    controller.after_request("tau-1")


def test_adaptive_controller_respects_initial_and_minimum_limit():
    controller = RequestAdmissionController(
        max_requests=8,
        min_requests=4,
        initial_requests=4,
        adaptive=True,
        kv_pct_fn=lambda: 99.0,
        adjustment_interval_s=0.0,
    )

    controller.before_request("tau-0", step=1, priority=0)
    snapshot = controller.snapshot()
    assert snapshot["current_limit"] == 4
    assert snapshot["min_requests"] == 4
    controller.after_request("tau-0")


def test_kv_reader_is_shared_across_concurrent_admission_checks():
    calls = [0]

    def read_kv() -> float:
        calls[0] += 1
        return 40.0

    controller = RequestAdmissionController(
        max_requests=2,
        adaptive=False,
        kv_pct_fn=read_kv,
        poll_interval_s=0.5,
    )
    controller.before_request("tau-0", step=1, priority=0)
    controller.before_request("tau-1", step=1, priority=0)
    assert calls == [1]
    controller.after_request("tau-0")
    controller.after_request("tau-1")


def test_cancel_unblocks_deferred_request():
    controller = RequestAdmissionController(
        max_requests=1,
        adaptive=False,
        poll_interval_s=0.02,
    )
    controller.before_request("tau-0", step=1, priority=0)
    errors: list[str] = []

    def wait_for_slot() -> None:
        try:
            controller.before_request("tau-1", step=1, priority=0)
        except RuntimeError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=wait_for_slot)
    worker.start()
    time.sleep(0.05)
    controller.cancel()
    worker.join(timeout=1.0)
    assert errors == ["F5 运行已停止"]


def test_tau_task_uses_request_boundary_and_collects_prompt_tokens(monkeypatch):
    import openai
    import tau_bench.envs

    from agent_mem.agent import react, tau_bench_agent
    from agent_mem.demo.tau_bench_ui import run_task_into_convo

    class FakeEnv:
        wiki = "wiki"
        tools_info = []

        def reset(self, task_index):
            return SimpleNamespace(observation=f"task-{task_index}")

        def step(self, action):
            return SimpleNamespace(reward=1.0, observation="done", done=True)

    monkeypatch.setattr(tau_bench.envs, "get_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        react,
        "stream_chat_with_ttft",
        lambda *args, **kwargs: ({"role": "assistant", "content": "done"}, 0.1, 123),
    )
    monkeypatch.setattr(
        tau_bench_agent,
        "_message_to_action",
        lambda *args, **kwargs: SimpleNamespace(name="respond", kwargs={"content": "done"}),
    )
    controller = RequestAdmissionController(max_requests=1, adaptive=False)

    result = run_task_into_convo(
        0,
        {},
        domain="retail",
        split="test",
        engine_url="http://engine/v1",
        model="model",
        max_steps=1,
        request_controller=controller,
    )

    assert result.success is True
    assert result.prompt_tokens == 123
    snapshot = controller.snapshot()
    assert snapshot["admitted_requests"] == 1
    assert snapshot["active_requests"] == 0
    assert snapshot["sessions"]["tau-0"]["state"] == "DONE"
