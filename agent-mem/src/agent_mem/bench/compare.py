"""三档对照 + ``comparison.md`` 生成。

把多个 run 目录的 ``metrics.json`` 聚合（按 config 分组、组内取中位数），再以
baseline 档为参照，对每档算显存/延迟降幅与成功率差，套门槛判定达标/标红：

- 显存峰值降幅 ≥ 30%（``mem_peak_mb`` 越低越好）
- e2e 延迟 p50 降幅 ≥ 20%（越低越好）
- 任务成功率差 ≤ 2 个百分点（成功率越高越好）

输出 ``logs/_summaries/<YYYYMMDD>_<study>_comparison.md``，**全 ASCII**
（``[PASS]`` / ``[FAIL]`` / ``[N/A]`` 标记，对齐日志规范）。baseline 值为 0
（骨架 dry-run）时该指标判 ``[N/A]``。
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from agent_mem.bench.stats import median

# 门槛
MEM_REDUCTION_MIN_PCT = 30.0
LATENCY_REDUCTION_MIN_PCT = 20.0
SUCCESS_DIFF_MAX_PP = 2.0

# 聚合 + 展示用的数值字段（顺序即表格列顺序）
NUMERIC_FIELDS = (
    "mem_peak_mb",
    "e2e_latency_p50_ms",
    "e2e_latency_p95_ms",
    "qps",
    "kv_cache_hit_rate",
    "task_success_rate",
    "ttft_ms",
)


def find_run_dirs(log_root: str | Path, *, config: str | None = None) -> list[Path]:
    """枚举 ``log_root`` 下的 run 目录（``*_run*``）。可按 config 短名过滤。"""
    root = Path(log_root)
    dirs = sorted(p for p in root.glob("*_run*") if p.is_dir())
    if config:
        dirs = [d for d in dirs if f"_{config}_run" in d.name]
    return dirs


def load_run_metrics(run_dirs: list[str | Path]) -> list[dict]:
    """从每个 run 目录读 ``metrics.json`` → dict 列表。缺文件跳过。"""
    out: list[dict] = []
    for d in run_dirs:
        p = Path(d) / "metrics.json"
        if p.exists():
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def median_by_config(metrics: list[dict]) -> dict[str, dict[str, float]]:
    """按 ``config`` 字段分组，组内对每个数值字段取中位数。"""
    groups: dict[str, list[dict]] = {}
    for m in metrics:
        groups.setdefault(m.get("config", "unknown"), []).append(m)
    result: dict[str, dict[str, float]] = {}
    for cfg, runs in groups.items():
        med: dict[str, float] = {}
        for f in NUMERIC_FIELDS:
            vals = [float(r.get(f, 0)) for r in runs]
            med[f] = median(vals)
        result[cfg] = med
    return result


def _reduction_row(metric: str, b: float, t: float, thr: float) -> dict:
    """越低越好：降幅 = (b - t)/b。"""
    if b <= 0:
        return {"metric": metric, "baseline": b, "target": t, "delta": "N/A",
                "threshold": f">= {thr:.0f}%", "verdict": "N/A"}
    red = (b - t) / b * 100
    return {"metric": metric, "baseline": b, "target": t, "delta": f"-{red:.1f}%",
            "threshold": f">= {thr:.0f}%", "verdict": "PASS" if red >= thr else "FAIL"}


def _success_row(b: float, t: float) -> dict:
    """越高越好：成功率差(pp) = (b - t)*100，正值表示 target 变差。"""
    diff_pp = (b - t) * 100
    return {"metric": "task_success_rate", "baseline": b, "target": t,
            "delta": f"{-diff_pp:+.2f}pp", "threshold": f"<= {SUCCESS_DIFF_MAX_PP:.0f}pp",
            "verdict": "PASS" if diff_pp <= SUCCESS_DIFF_MAX_PP else "FAIL"}


def compute_deltas(baseline: dict[str, float], target: dict[str, float]) -> list[dict]:
    """对照 baseline 算 target 的三行判定（显存/延迟/成功率）。"""
    return [
        _reduction_row("mem_peak_mb", baseline.get("mem_peak_mb", 0),
                       target.get("mem_peak_mb", 0), MEM_REDUCTION_MIN_PCT),
        _reduction_row("e2e_latency_p50_ms", baseline.get("e2e_latency_p50_ms", 0),
                       target.get("e2e_latency_p50_ms", 0), LATENCY_REDUCTION_MIN_PCT),
        _success_row(baseline.get("task_success_rate", 0), target.get("task_success_rate", 0)),
    ]


def _fmt(v: float) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def render_comparison_md(
    study: str,
    medians: dict[str, dict[str, float]],
    *,
    baseline_config: str = "baseline",
) -> str:
    """渲染对照报告 markdown（ASCII）。"""
    today = _dt.datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = [
        f"# Comparison: {study}",
        "",
        f"- date: {today}",
        f"- baseline: {baseline_config}",
        f"- thresholds: mem_peak >= {MEM_REDUCTION_MIN_PCT:.0f}% down, "
        f"e2e_latency_p50 >= {LATENCY_REDUCTION_MIN_PCT:.0f}% down, "
        f"task_success_rate <= {SUCCESS_DIFF_MAX_PP:.0f}pp diff",
        "",
        "## median metrics per config",
        "",
    ]

    # 中位数表
    configs = list(medians.keys())
    header = ["config"] + list(NUMERIC_FIELDS)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for cfg in configs:
        row = [cfg] + [_fmt(medians[cfg].get(f, 0.0)) for f in NUMERIC_FIELDS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 各档 vs baseline 判定
    baseline = medians.get(baseline_config)
    if baseline is None:
        lines.append(f"_(未找到 baseline 档 `{baseline_config}`，跳过判定。)_")
        return "\n".join(lines) + "\n"

    all_pass = True
    for cfg in configs:
        if cfg == baseline_config:
            continue
        lines.append(f"## {cfg} vs {baseline_config}")
        lines.append("")
        lines.append("| metric | baseline | target | delta | threshold | verdict |")
        lines.append("|---|---|---|---|---|---|")
        for r in compute_deltas(baseline, medians[cfg]):
            lines.append(
                f"| {r['metric']} | {_fmt(r['baseline'])} | {_fmt(r['target'])} "
                f"| {r['delta']} | {r['threshold']} | [{r['verdict']}] |"
            )
        rows = compute_deltas(baseline, medians[cfg])
        cfg_pass = all(r["verdict"] != "FAIL" for r in rows)
        all_pass = all_pass and cfg_pass
        lines.append("")
        lines.append(f"overall: {'[PASS]' if cfg_pass else '[FAIL]'}")
        lines.append("")

    lines.append(f"## all thresholds met: {'YES' if all_pass else 'NO'}")
    return "\n".join(lines) + "\n"


def write_comparison_md(
    log_root: str | Path,
    study: str,
    medians: dict[str, dict[str, float]],
    *,
    baseline_config: str = "baseline",
) -> Path:
    """写 ``logs/_summaries/<YYYYMMDD>_<study>_comparison.md``，返回路径。"""
    today = _dt.datetime.now().strftime("%Y%m%d")
    out_dir = Path(log_root) / "_summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{today}_{study}_comparison.md"
    path.write_text(render_comparison_md(study, medians, baseline_config=baseline_config),
                    encoding="utf-8")
    return path


def compare_runs(
    run_dirs: list[str | Path],
    *,
    study: str,
    log_root: str | Path,
    baseline_config: str = "baseline",
) -> Path:
    """端到端：读 metrics → 中位数分组 → 渲染写盘 → 返回 md 路径。"""
    medians = median_by_config(load_run_metrics(run_dirs))
    return write_comparison_md(log_root, study, medians, baseline_config=baseline_config)
