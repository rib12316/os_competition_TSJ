"""run 目录命名规则 + 脚手架测试。"""

from __future__ import annotations

import re

import pytest

from agent_mem import repro
from agent_mem.bench.run_dir import (
    build_run_dir_name,
    create_run_dir,
    run_id_for,
    short_config_name,
    short_engine_name,
    short_model_name,
    timestamp_now,
    ts_to_iso,
    write_metrics,
)
from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, MetricsConfig
from agent_mem.metrics import default_metrics

_RUN_ID_RE = re.compile(
    r"^[0-9]{8}-[0-9]{6}_[a-z0-9-]+_[a-z0-9-]+_[a-z0-9-]+_run[0-9]+$"
)
# 单个短名段：仅小写字母/数字/连字符
_SHORT_RE = re.compile(r"^[a-z0-9-]+$")


# ---- 短名映射 ----


@pytest.mark.parametrize(
    "full,short",
    [
        ("Qwen2.5-7B-Instruct", "qwen25-7b"),
        ("Qwen3-0.6B", "qwen3-0-6b"),
        ("MiniCPM3-4B", "minicpm3-4b"),
        ("Qwen2.5-1.5B-Instruct", "qwen25-1-5b"),
    ],
)
def test_short_model_name_whitelist(full, short):
    assert short_model_name(full) == short


def test_short_model_name_fallback_strips_suffix_and_lowercases():
    assert short_model_name("SomeModel-7B-Instruct") == "somemodel-7b"
    # 点号转连字符（2.5 等已知模型走白名单，未知模型 fallback 用连字符保可读）
    assert short_model_name("Llama3.1-8B") == "llama3-1-8b"


def test_short_model_name_is_ascii_only():
    assert _SHORT_RE.fullmatch(short_model_name("Weird Model 1.0"))


def test_short_engine_name():
    assert short_engine_name("vLLM") == "vllm"
    assert short_engine_name("vllm-ascend") == "vllm-ascend"


def test_short_config_name_underscore_to_hyphen():
    assert short_config_name("prefix_cache") == "prefix-cache"
    assert short_config_name("baseline") == "baseline"
    assert _SHORT_RE.fullmatch(short_config_name("m3_session"))


# ---- 目录命名格式 ----


def test_build_run_dir_name_format():
    name = build_run_dir_name("20260718-143022", "vllm", "qwen25-7b", "baseline", 1)
    assert name == "20260718-143022_vllm_qwen25-7b_baseline_run1"
    assert _RUN_ID_RE.fullmatch(name)


def test_build_run_dir_name_ascend_prefix_cache():
    name = build_run_dir_name(
        "20260718-144510", "vllm-ascend", "qwen25-7b", "prefix-cache", 2
    )
    assert name == "20260718-144510_vllm-ascend_qwen25-7b_prefix-cache_run2"


def test_ts_to_iso():
    assert ts_to_iso("20260718-143022") == "2026-07-18T14:30:22"


def test_timestamp_now_format():
    assert re.fullmatch(r"[0-9]{8}-[0-9]{6}", timestamp_now())


# ---- 脚手架 ----


def _cfg(config_name: str = "prefix_cache") -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm-ascend", model="Qwen2.5-7B-Instruct"),
        benchmark=BenchmarkConfig(domain="retail", runs=3, seed=42),
        metrics=MetricsConfig(),
        config_name=config_name,
    )


def test_create_run_dir_writes_repro_and_placeholders(tmp_path):
    cfg = _cfg()
    d = create_run_dir(
        tmp_path, cfg, run_n=1, config_text="# raw config snapshot", ts="20260718-143022"
    )
    assert d.name == "20260718-143022_vllm-ascend_qwen25-7b_prefix-cache_run1"
    # 复现性三件套
    assert (d / "config.yaml").read_text(encoding="utf-8") == "# raw config snapshot"
    assert (d / "git_commit.txt").read_text(encoding="utf-8").strip()  # 非空
    env_txt = (d / "env.txt").read_text(encoding="utf-8")
    assert "python=" in env_txt
    # 日志占位
    assert (d / "engine.log").exists()
    assert (d / "agent.log").exists()
    assert (d / "mem_timeseries.csv").read_text() == "timestamp,used_mb\n"
    assert (d / "vllm_metrics.json").read_text().strip() == "{}"


def test_create_run_dir_config_text_fallback_dumps_cfg(tmp_path):
    cfg = _cfg("baseline")
    d = create_run_dir(tmp_path, cfg, run_n=3, ts="20260719-080000")
    text = (d / "config.yaml").read_text(encoding="utf-8")
    # 回退走 yaml.safe_dump(asdict(cfg))，应含关键字段
    assert "backend: vllm-ascend" in text
    assert "config_name: baseline" in text


def test_run_id_for_and_write_metrics(tmp_path):
    cfg = _cfg("baseline")
    run_id, started_at = run_id_for(cfg, run_n=1, ts="20260718-143022")
    assert run_id == "20260718-143022_vllm-ascend_qwen25-7b_baseline_run1"
    assert started_at == "2026-07-18T14:30:22"

    d = create_run_dir(tmp_path, cfg, run_n=1, ts="20260718-143022")
    m = default_metrics(
        run_id=run_id,
        engine="vllm-ascend",
        model="qwen25-7b",
        config="baseline",
        seed=42,
        started_at=started_at,
    )
    mp = write_metrics(d, m)
    assert mp.name == "metrics.json"
    import json

    obj = json.loads(mp.read_text())
    assert obj["run_id"] == run_id


# ---- repro 兜底分支 ----


def test_git_commit_fallback_on_failure(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("no git")

    monkeypatch.setattr(repro.subprocess, "check_output", _boom)
    assert repro.git_commit() == "unknown"


def test_collect_env_has_python():
    env = repro.collect_env()
    assert "python" in env
    assert env["python"]
