"""run 目录命名 + 脚手架。

命名严格对齐 ``dev-guide/log-naming-convention.md``::

    logs/<YYYYMMDD-HHMMSS>_<engine>_<model>_<config>_run<N>/

每个 run 目录内含复现性三件套（``config.yaml`` / ``git_commit.txt`` / ``env.txt``）
+ ``metrics.json`` + 日志占位文件（``engine.log`` / ``agent.log`` /
``mem_timeseries.csv`` / ``vllm_metrics.json``）。骨架阶段日志文件先留占位，
内容由 MVP 阶段的真实采集填充。
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from agent_mem.config import AppConfig
from agent_mem.metrics import RunMetrics
from agent_mem.metrics import write as _write_metrics_file
from agent_mem.repro import format_env_text, git_commit

# 模型全名 → 目录短名（对齐 log-naming-convention.md 词汇表）
MODEL_SHORT_NAMES: dict[str, str] = {
    "Qwen2.5-7B-Instruct": "qwen25-7b",
    "Qwen2.5-1.5B-Instruct": "qwen25-1-5b",
    "Qwen3-0.6B": "qwen3-0-6b",
    "MiniCPM3-4B": "minicpm3-4b",
}


def short_model_name(model: str) -> str:
    """模型全名 → 小写连字符短名（白名单优先，正则兜底）。"""
    if model in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model]
    s = re.sub(r"(?i)(-instruct|-chat)$", "", model.strip()).lower().replace(".", "-")
    return re.sub(r"[^a-z0-9-]", "", s)


def short_engine_name(backend: str) -> str:
    """引擎名规范化（``vllm`` / ``sglang`` / ``vllm-ascend`` 已是规范词）。"""
    return re.sub(r"[^a-z0-9-]", "", backend.strip().lower())


def short_config_name(config_name: str) -> str:
    """config 名规范化：下划线转连字符、全小写（``prefix_cache`` → ``prefix-cache``）。"""
    s = config_name.strip().lower().replace("_", "-")
    return re.sub(r"[^a-z0-9-]", "", s)


def build_run_dir_name(
    ts: str, engine: str, model: str, config: str, run_n: int
) -> str:
    """拼 ``<ts>_<engine>_<model>_<config>_run<N>``（字段间下划线）。"""
    return f"{ts}_{engine}_{model}_{config}_run{run_n}"


def timestamp_now() -> str:
    """当前本地时间戳 ``YYYYMMDD-HHMMSS``。"""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def ts_to_iso(ts: str) -> str:
    """``20260718-143022`` → ``2026-07-18T14:30:22``。"""
    t = _dt.datetime.strptime(ts, "%Y%m%d-%H%M%S")
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def write_repro_files(
    run_dir: Path, cfg: AppConfig, *, config_text: str | None = None
) -> None:
    """写复现性三件套：``config.yaml`` 快照 + ``git_commit.txt`` + ``env.txt``。"""
    if config_text is None:
        config_text = yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True)
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit() + "\n", encoding="utf-8")
    (run_dir / "env.txt").write_text(format_env_text(), encoding="utf-8")


def write_log_placeholders(run_dir: Path) -> None:
    """写日志占位文件（骨架阶段留空/占位，MVP 填真实内容）。"""
    (run_dir / "engine.log").write_text("", encoding="utf-8")
    (run_dir / "agent.log").write_text("", encoding="utf-8")
    (run_dir / "mem_timeseries.csv").write_text("timestamp,used_mb\n", encoding="utf-8")
    (run_dir / "vllm_metrics.json").write_text("{}\n", encoding="utf-8")


def create_run_dir(
    run_root: str | Path,
    cfg: AppConfig,
    run_n: int,
    *,
    config_text: str | None = None,
    ts: str | None = None,
) -> Path:
    """建 run 目录并写复现性三件套 + 日志占位。返回创建的目录路径。

    目录名 = :func:`build_run_dir_name`，engine/model/config 由 cfg 派生。
    """
    ts = ts or timestamp_now()
    engine = short_engine_name(cfg.engine.backend)
    model = short_model_name(cfg.engine.model)
    config = short_config_name(cfg.config_name)
    run_id = build_run_dir_name(ts, engine, model, config, run_n)
    run_dir = Path(run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_repro_files(run_dir, cfg, config_text=config_text)
    write_log_placeholders(run_dir)
    return run_dir


def write_metrics(run_dir: str | Path, metrics: RunMetrics) -> Path:
    """把 :class:`RunMetrics` 写入 run 目录的 ``metrics.json``（委托 :func:`agent_mem.metrics.write`）。"""
    return _write_metrics_file(metrics, Path(run_dir) / "metrics.json")


def run_id_for(
    cfg: AppConfig, run_n: int, *, ts: str | None = None
) -> tuple[str, str]:
    """返回 ``(run_id, started_at_iso)``，供 metrics 用。"""
    ts = ts or timestamp_now()
    run_id = build_run_dir_name(
        ts,
        short_engine_name(cfg.engine.backend),
        short_model_name(cfg.engine.model),
        short_config_name(cfg.config_name),
        run_n,
    )
    return run_id, ts_to_iso(ts)
