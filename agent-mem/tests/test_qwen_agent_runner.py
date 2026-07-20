"""QwenAgentRunner 测试（monkeypatch adapter，不触真 τ-bench/LLM）。"""

from __future__ import annotations

from agent_mem.bench.runners.qwen_agent import QwenAgentRunner
from agent_mem.bench.tasks import tau_bench_adapter as adapter
from agent_mem.bench.tasks.tau_bench_adapter import TaskInfo, TaskRunResult, _resolve_user_sim
from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, MetricsConfig


def _cfg() -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm", model="Qwen3-0.6B"),
        benchmark=BenchmarkConfig(domain="retail", split="test"),
        metrics=MetricsConfig(),
        config_name="baseline",
    )


def test_run_all_delegates_to_run_task(monkeypatch):
    calls: list[tuple] = []

    def fake_list_tasks(domain, split="test"):
        return [TaskInfo(i, "u", "instr", (), 0, domain, split) for i in range(5)]

    def fake_run_task(task_id, **kw):
        calls.append((task_id, kw.get("engine_url"), kw.get("model")))
        return TaskRunResult(task_id, 1.0, True, 10.0, 1, None)

    monkeypatch.setattr(adapter, "list_tasks", fake_list_tasks)
    monkeypatch.setattr(adapter, "run_task", fake_run_task)

    r = QwenAgentRunner(engine_url="http://x:8000/v1", model="Qwen3-0.6B")
    results = r.run_all(_cfg())

    assert len(results) == 5
    assert all(x.success and x.reward == 1.0 for x in results)
    # 第一个任务用传入的 engine_url/model 调 run_task
    assert calls[0] == (0, "http://x:8000/v1", "Qwen3-0.6B")


def test_run_all_respects_max_tasks(monkeypatch):
    monkeypatch.setattr(
        adapter, "list_tasks",
        lambda d, s="test": [TaskInfo(i, "u", "i", (), 0, d, s) for i in range(10)],
    )
    monkeypatch.setattr(
        adapter, "run_task",
        lambda tid, **kw: TaskRunResult(tid, 0.0, False, 0.0, 0, None),
    )

    r = QwenAgentRunner(engine_url="http://x:8000/v1", model="m", max_tasks=3)
    results = r.run_all(_cfg())
    assert len(results) == 3
    assert [x.task_id for x in results] == [0, 1, 2]


def test_name():
    assert QwenAgentRunner(engine_url="x", model="m").name() == "qwen-agent"


# ---- _resolve_user_sim（user-sim 端点解析，纯函数）----


def test_user_sim_default_local():
    """默认：user-sim 走本地引擎（合规 + 可复现）。"""
    um, up, env = _resolve_user_sim(
        engine_url="http://localhost:8000/v1", model="Qwen2.5-7B-Instruct",
        user_model=None, user_provider="openai", user_api_base=None,
        user_api_key=None, api_key="stub",
    )
    assert um == "Qwen2.5-7B-Instruct"
    assert up == "openai"
    assert env == {"OPENAI_API_BASE": "http://localhost:8000/v1", "OPENAI_API_KEY": "stub"}


def test_user_sim_external_openai_compatible():
    """外部 OpenAI 兼容（GPT-4o 官方 / MiniMax / DeepSeek-API）：指向 user_api_base。"""
    um, up, env = _resolve_user_sim(
        engine_url="http://localhost:8000/v1", model="Qwen2.5-7B-Instruct",
        user_model="gpt-4o", user_provider="openai",
        user_api_base="https://api.openai.com/v1", user_api_key="sk-xxx",
        api_key="stub",
    )
    assert um == "gpt-4o"
    assert up == "openai"
    assert env["OPENAI_API_BASE"] == "https://api.openai.com/v1"
    assert env["OPENAI_API_KEY"] == "sk-xxx"
    # 不含本地 engine_url
    assert "localhost" not in env["OPENAI_API_BASE"]


def test_user_sim_external_minimax_style():
    """MiniMax 等 OpenAI 兼容端点：自定义 base + 模型名。"""
    um, up, env = _resolve_user_sim(
        engine_url="http://localhost:8000/v1", model="Qwen2.5-7B-Instruct",
        user_model="abab6.5-chat", user_provider="openai",
        user_api_base="https://api.minimaxi.com/v1", user_api_key="mm-key",
        api_key="stub",
    )
    assert um == "abab6.5-chat"
    assert env["OPENAI_API_BASE"] == "https://api.minimaxi.com/v1"


def test_user_sim_non_openai_provider():
    """非 openai 系（anthropic 等）：不动 OPENAI_*，用户自行 export provider env。"""
    um, up, env = _resolve_user_sim(
        engine_url="http://localhost:8000/v1", model="Qwen2.5-7B-Instruct",
        user_model="claude-3-5-sonnet", user_provider="anthropic",
        user_api_base="https://anything", user_api_key="k",
        api_key="stub",
    )
    assert um == "claude-3-5-sonnet"
    assert up == "anthropic"
    assert env == {}  # 不设 OPENAI_*，由用户 export ANTHROPIC_API_KEY


def test_user_sim_external_defaults_model_when_absent():
    um, _, _ = _resolve_user_sim(
        engine_url="http://x/v1", model="Qwen",
        user_model=None, user_provider="openai",
        user_api_base="https://api.openai.com/v1", user_api_key="sk",
        api_key="stub",
    )
    assert um == "gpt-4o"  # 外部但未指定 model → 默认 gpt-4o
