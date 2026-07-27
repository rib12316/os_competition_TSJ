"""LongBench adapter 测试（fake client + tmp zip，无 NPU / 无真 LLM）。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_mem.bench.tasks.longbench_adapter import list_tasks, load_task, run_task
from agent_mem.bench.tasks.types import TaskInfo
from agent_mem.config import AppConfig, BenchmarkConfig


def _write_zip(tmp_path: Path, examples: list[dict]) -> Path:
    z = tmp_path / "lb.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/2wikimqa.jsonl", "\n".join(json.dumps(e) for e in examples))
    return z


def _example(answer: str = "Paris") -> dict:
    return {
        "_id": "x",
        "context": "Passage 1:\nCapital\nFrance's capital is Paris.\nPassage 2:\nEiffel\nA tower in Paris.",
        "input": "What is the capital of France?",
        "answers": [answer],
    }


def _fake_client() -> SimpleNamespace:
    # run_react 被 monkeypatch，client.create 不会被真调；_RecordingClient 只需结构存在
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: None))
    )


def test_list_tasks_reads_zip(tmp_path):
    z = _write_zip(tmp_path, [_example(), _example("London")])
    cfg = AppConfig(
        benchmark=BenchmarkConfig(suite="longbench", data_zip=str(z), options={"limit": 2})
    )
    tasks = list_tasks(cfg)
    assert len(tasks) == 2
    assert all(t.suite == "longbench" and t.domain == "2wikimqa" for t in tasks)
    assert tasks[0].payload["answers"] == ["Paris"]
    assert tasks[1].payload["answers"] == ["London"]


def test_list_tasks_requires_data_zip():
    cfg = AppConfig(benchmark=BenchmarkConfig(suite="longbench"))
    with pytest.raises(ValueError, match="data_zip"):
        list_tasks(cfg)


def test_load_task_reads_one_dataset_index(tmp_path):
    z = _write_zip(tmp_path, [_example("Paris"), _example("London")])
    task = load_task(z, 1)
    assert task.task_id == 1
    assert task.suite == "longbench"
    assert task.domain == "2wikimqa"
    assert task.payload["answers"] == ["London"]


def test_load_task_rejects_missing_path_and_out_of_range(tmp_path):
    with pytest.raises(FileNotFoundError, match="不存在"):
        load_task(tmp_path / "missing.zip", 0)
    z = _write_zip(tmp_path, [_example()])
    with pytest.raises(IndexError, match="越界"):
        load_task(z, 2)


def test_run_task_success_when_answer_matches(monkeypatch):
    monkeypatch.setattr(
        "agent_mem.agent.react.run_react",
        lambda *a, **k: SimpleNamespace(final_text="The answer is Paris", n_steps=3, truncated=False),
    )
    task = TaskInfo(task_id=0, suite="longbench", domain="2wikimqa", payload=_example("Paris"))
    res = run_task(task, engine_url="http://x/v1", model="m", client=_fake_client())
    assert res.success is True
    assert res.reward == 1.0
    assert res.task_id == 0
    assert res.error is None


def test_run_task_failure_when_answer_wrong(monkeypatch):
    monkeypatch.setattr(
        "agent_mem.agent.react.run_react",
        lambda *a, **k: SimpleNamespace(final_text="London", n_steps=3, truncated=False),
    )
    task = TaskInfo(task_id=1, suite="longbench", domain="2wikimqa", payload=_example("Paris"))
    res = run_task(task, engine_url="http://x/v1", model="m", client=_fake_client())
    assert res.success is False
    assert res.reward == 0.0


def test_run_task_error_returns_failure(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("engine down")

    monkeypatch.setattr("agent_mem.agent.react.run_react", _boom)
    task = TaskInfo(task_id=2, suite="longbench", domain="2wikimqa", payload=_example())
    res = run_task(task, engine_url="http://x/v1", model="m", client=_fake_client())
    assert res.success is False
    assert res.reward == 0.0
    assert "engine down" in (res.error or "")
