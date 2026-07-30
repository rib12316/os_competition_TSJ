"""通用改进页 · KV量化(F1) + LMCache三级分层(F4) —— 实时演示模块。

被 ``chat_app.py`` 的「通用改进」tab 调用。启动后，页面拉起所选 vLLM 档位并运行
一个真实 tau-bench 任务；``gr.Timer`` 把 ``LiveMonitor`` 从 vLLM/LMCache Prometheus
端点采到的 KV、CPU/磁盘层、吞吐和 TTFT 指标逐帧送进本模块渲染。
组合档按 C8 质量/容量探针、LMCache tau-bench 两阶段依次运行；当前 Ascend 上游把
``kv_both`` 视作 producer，无法与 int8 KV 稳定同时启用。

**模型参数均取自真机实测数据，不随意填充**：
- NPU 容量：fp16 775,936 / int8 1,556,992（F1 benchmark @ 41.5 GiB）
- CPU offload 预算：~4 GiB → ~73K tokens（``configs/f5-sweep-full-on.yaml`` cpu_bytes_to_use）
- Disk 预算：demo 引擎显式配置 2 GiB 本地 SSD 层
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_mem.demo.monitor import Sample

# ============================================================
# 容量（均取自真机实测数据，不留任意常数）
# ============================================================

NPU_CAP_FP16 = 775_936              # 真机 bf16 token 容量（~41.5 GiB KV 预算）
NPU_CAP_INT8 = 1_556_992            # 真机 int8 token 容量（2.007×）
NPU_KV_POOL_GIB = 41.5              # 真机 KV pool 预算
HBM_TOTAL_GIB = 64

# CPU offload 预算：取真机 sweep config cpu_bytes_to_use = 4 GiB
_CPU_BYTES = 4_294_967_296          # 真机 sweep config
_BYTES_PER_TOK_FP16 = (NPU_KV_POOL_GIB * 1024**3) / NPU_CAP_FP16  # ~56 KiB/tok
CPU_CAP_TOK = int(_CPU_BYTES / _BYTES_PER_TOK_FP16)  # ~73K tokens
CPU_GIB = 4                         # 真机 offload 预算（非系统总内存）

# Demo 引擎为 LMCache 显式分配 2 GiB 本地 SSD 层。
DISK_GIB = 2
DISK_CAP_TOK = int((DISK_GIB * 1024**3) / _BYTES_PER_TOK_FP16)

#: 展示档位（Radio）。quant=是否 KV量化；tier=是否 LMCache 三级分层。
MODES: dict[str, dict[str, object]] = {
    "baseline": {"label": "baseline（fp16 · 仅 NPU）", "quant": False, "tier": False},
    "quant": {"label": "+ KV量化 (int8)", "quant": True, "tier": False},
    "tier": {"label": "+ LMCache 三级分层", "quant": False, "tier": True},
    "full": {"label": "量化 → 分层（依次实测）", "quant": True, "tier": True},
}

_HIST_MAX = 120
_STATIC_RESULTS_PATH = Path(__file__).with_name("kv_experiment_results.json")


# ============================================================
# 实时模型状态
# ============================================================

@dataclass
class SimState:
    """通用改进页状态；所有运行时数值来自真实监控样本或由其换算。"""

    running: bool = False
    t: float = 0.0
    demand_tok: float = 0.0
    npu_tok: float = 0.0
    cpu_tok: float = 0.0
    disk_tok: float = 0.0
    preemptions: int = 0
    hit_rate: float | None = None
    history: list = field(default_factory=list)
    # 真机 /metrics 额外字段
    alive: bool = False
    npu_usage: float | None = None          # NPU KV 占用率 0~1
    hbm_mb: int | None = None               # NPU HBM 已用 MB
    throughput: float | None = None          # tok/s
    ttft: float | None = None               # ms
    running_reqs: int | None = None
    waiting_reqs: int | None = None
    phase: str = "idle"  # idle | starting | task | done | stopped | error
    message: str = ""
    source: str = ""
    lmcache_stored_tokens: float | None = None
    npu_capacity_tok: float | None = None
    quant_capacity_tok: float | None = None
    active_quant: bool = False
    active_tier: bool = False
    peak_npu_usage: float = 0.0
    peak_hbm_mb: int | None = None
    peak_cpu_tok: float = 0.0
    peak_disk_tok: float = 0.0
    max_running_reqs: int = 0
    max_waiting_reqs: int = 0
    current_stage: str = ""
    stage_results: dict[str, dict[str, object]] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> SimState:
        return cls()

    @classmethod
    def from_metrics(
        cls, *, kv_usage: float | None, kv_hit: float | None,
        hbm_mb: int | None = None, throughput: float | None = None,
        ttft: float | None = None, running: int | None = None, waiting: int | None = None,
        alive: bool = False, t: float = 0.0, quant: bool = False,
        cpu_bytes: float | None = None, disk_bytes: float | None = None,
        preemptions: float | None = None, stored_tokens: float | None = None,
        npu_capacity_tokens: int | None = None,
    ) -> SimState:
        """从真机 /metrics 快照构造状态（纯函数，可单测）。"""
        npu_cap = npu_capacity_tokens or (NPU_CAP_INT8 if quant else NPU_CAP_FP16)
        bytes_per_tok = _BYTES_PER_TOK_FP16 / (2 if quant else 1)
        npu_tok = max(0.0, kv_usage or 0.0) * npu_cap
        cpu_tok = max(0.0, cpu_bytes or 0.0) / bytes_per_tok
        disk_tok = max(0.0, disk_bytes or 0.0) / bytes_per_tok
        return cls(
            running=True, alive=alive, t=t,
            npu_usage=kv_usage,
            demand_tok=npu_tok + cpu_tok + disk_tok,
            npu_tok=npu_tok, cpu_tok=cpu_tok, disk_tok=disk_tok,
            hit_rate=kv_hit,
            hbm_mb=hbm_mb, throughput=throughput, ttft=ttft,
            running_reqs=running, waiting_reqs=waiting,
            preemptions=int(preemptions or 0), phase="task", source="Prometheus /metrics",
            lmcache_stored_tokens=stored_tokens,
            npu_capacity_tok=float(npu_cap),
        )


def _features_for(mode: str) -> list[str]:
    """档位 → engine_mgr.start() 用的 feature 列表。"""
    m = MODES.get(mode, MODES["full"])
    feats = ["prefix-cache"]
    if m["quant"]:
        feats.append("c8")
    if m["tier"]:
        feats.append("lmcache")
    return feats


def engine_stages_for(mode: str) -> list[list[str]]:
    """Return a real baseline followed by the requested feature stages."""
    stages = [_features_for("baseline")]
    if mode in {"quant", "full"}:
        stages.append(_features_for("quant"))
    if mode in {"tier", "full"}:
        stages.append(_features_for("tier"))
    return stages


def stage_name_for(features: list[str]) -> str:
    if "c8" in features:
        return "quant"
    if "lmcache" in features:
        return "tier"
    return "baseline"


def _last_value(values: list[float | None]) -> float | None:
    return next((value for value in reversed(values) if value is not None), None)


def _run_average(samples: list[Sample], value_attr: str, count_attr: str) -> float | None:
    usable = [
        sample for sample in samples
        if getattr(sample, value_attr) is not None and getattr(sample, count_attr) is not None
    ]
    if len(usable) < 2:
        return None
    delta_count = getattr(usable[-1], count_attr) - getattr(usable[0], count_attr)
    if delta_count <= 0:
        return None
    return (getattr(usable[-1], value_attr) - getattr(usable[0], value_attr)) / delta_count


def _run_throughput(samples: list[Sample]) -> float | None:
    usable = [sample for sample in samples if sample.gen_tokens is not None]
    if len(usable) < 2:
        return None
    elapsed = usable[-1].t - usable[0].t
    generated = usable[-1].gen_tokens - usable[0].gen_tokens
    return generated / elapsed if elapsed > 0 and generated >= 0 else None


def update_from_samples(
    sim: SimState,
    samples: list[Sample],
    mode: str,
    *,
    npu_capacity_tokens: int | None = None,
) -> SimState:
    """用 ``LiveMonitor`` 真机样本原地刷新状态，供 Gradio Timer 调用。"""
    if not samples:
        return sim
    from agent_mem.demo.monitor import compute_window_series

    latest = samples[-1]
    series = compute_window_series(samples, window_s=10.0)
    configured = MODES.get(mode, MODES["full"])
    quant = sim.active_quant if sim.phase != "idle" else bool(configured["quant"])
    requested = latest.lmcache_requested_tokens
    hits = latest.lmcache_hit_tokens
    lmcache_hit = (
        hits / requested if hits is not None and requested is not None and requested > 0 else None
    )
    window_hit = _last_value(series.kv_rate)
    fresh = SimState.from_metrics(
        kv_usage=latest.kv_usage_perc,
        kv_hit=lmcache_hit if lmcache_hit is not None else window_hit,
        hbm_mb=latest.mem_mb,
        throughput=_run_throughput(samples) or _last_value(series.throughput),
        ttft=(
            None
            if (run_ttft := _run_average(samples, "ttft_sum", "ttft_count")) is None
            else run_ttft * 1000.0
        ) or _last_value(series.ttft),
        running=latest.running,
        waiting=latest.waiting,
        alive=latest.kv_usage_perc is not None,
        t=latest.t,
        quant=quant,
        cpu_bytes=latest.lmcache_local_bytes,
        disk_bytes=latest.lmcache_disk_bytes,
        preemptions=latest.preemptions,
        stored_tokens=latest.lmcache_stored_tokens,
        npu_capacity_tokens=npu_capacity_tokens,
    )
    fresh.running = sim.running
    fresh.phase = sim.phase
    fresh.message = sim.message
    fresh.history = sim.history
    fresh.quant_capacity_tok = sim.quant_capacity_tok
    fresh.active_quant = sim.active_quant
    fresh.active_tier = sim.active_tier
    fresh.current_stage = sim.current_stage
    fresh.stage_results = {key: dict(value) for key, value in sim.stage_results.items()}
    fresh.peak_npu_usage = max(sim.peak_npu_usage, fresh.npu_usage or 0.0)
    hbm_values = [value for value in (sim.peak_hbm_mb, fresh.hbm_mb) if value is not None]
    fresh.peak_hbm_mb = max(hbm_values) if hbm_values else None
    fresh.peak_cpu_tok = max(sim.peak_cpu_tok, fresh.cpu_tok)
    fresh.peak_disk_tok = max(sim.peak_disk_tok, fresh.disk_tok)
    fresh.max_running_reqs = max(sim.max_running_reqs, fresh.running_reqs or 0)
    fresh.max_waiting_reqs = max(sim.max_waiting_reqs, fresh.waiting_reqs or 0)
    if fresh.current_stage:
        evidence = fresh.stage_results.setdefault(fresh.current_stage, {})
        evidence.update({
            "capacity_tokens": fresh.npu_capacity_tok,
            "kv_peak": fresh.peak_npu_usage,
            "hbm_peak_mb": fresh.peak_hbm_mb,
            "throughput": fresh.throughput,
            "ttft_ms": fresh.ttft,
            "hit_rate": fresh.hit_rate,
            "preemptions": fresh.preemptions,
            "cpu_peak_tok": fresh.peak_cpu_tok,
            "disk_peak_tok": fresh.peak_disk_tok,
            "stored_tokens": fresh.lmcache_stored_tokens,
            "max_running": fresh.max_running_reqs,
            "max_waiting": fresh.max_waiting_reqs,
        })
    fresh.history.append((
        round(fresh.t, 1),
        (fresh.npu_usage or 0.0) * 100,
        fresh.cpu_tok / max(1, CPU_CAP_TOK) * 100,
        fresh.disk_tok / max(1, DISK_CAP_TOK) * 100,
        None if fresh.hit_rate is None else fresh.hit_rate * 100,
    ))
    fresh.history = fresh.history[-_HIST_MAX:]
    sim.__dict__.update(fresh.__dict__)
    return sim


def tick_sim(sim: SimState, mode: str, dt: float = 1.0) -> SimState:
    """已弃用的兼容入口；真实演示必须调用 :func:`update_from_samples`。"""
    return sim


# ============================================================
# 格式化
# ============================================================

def _fmt_tokens(n: float) -> str:
    n = float(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:.0f}"


def _fmt_gib(gib: float) -> str:
    return f"{gib/1024:.1f} TiB" if gib >= 1024 else f"{gib:.0f} GiB"


def _status(sim: SimState) -> str:
    return {
        "idle": "未启动", "starting": "引擎启动中", "task": "任务运行中",
        "done": "任务已完成", "stopped": "已停止", "error": "运行失败",
    }.get(sim.phase, "运行中" if sim.running else "已停止")


# ============================================================
# 三级卡片 HTML
# ============================================================

_KV_CSS = """
.kv-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1a1a1a;line-height:1.5}
.kv-tiers{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:6px 0;min-height:168px}
.kv-tier{border:2px solid rgba(0,0,0,.1);border-radius:8px;padding:14px 12px;text-align:center;background:#fcfcfb;min-height:160px;box-sizing:border-box}
.kv-tier.npu{border-color:#2a78d6;background:linear-gradient(135deg,rgba(42,120,214,.05),rgba(42,120,214,.01))}
.kv-tier.cpu{border-color:#eb6834;background:linear-gradient(135deg,rgba(235,104,52,.05),rgba(235,104,52,.01))}
.kv-tier.disk{border-color:#1baf7a;background:linear-gradient(135deg,rgba(27,175,122,.05),rgba(27,175,122,.01))}
.kv-tier.off{opacity:.42;filter:grayscale(.7)}
.kv-tier .icon{font-size:1.4rem}
.kv-tier h4{margin:2px 0 0;font-size:.86rem;font-weight:700}
.kv-tier .cap{font-size:.66rem;color:#898781;margin:2px 0}
.kv-tier .bar-bg{height:9px;background:rgba(0,0,0,.08);border-radius:5px;overflow:hidden;margin:6px 0 3px}
.kv-tier .bar-fg{height:100%;border-radius:5px;transition:width .4s}
.kv-tier.npu .bar-fg{background:linear-gradient(90deg,#2a78d6,#5b9eed)}
.kv-tier.cpu .bar-fg{background:linear-gradient(90deg,#eb6834,#f0986a)}
.kv-tier.disk .bar-fg{background:linear-gradient(90deg,#1baf7a,#45d4a0)}
.kv-tier .frac{font-size:.74rem;color:#52514e;font-variant-numeric:tabular-nums}
.kv-tier .sub{font-size:.64rem;color:#898781;margin-top:2px}
.kv-tier .badge{display:inline-block;font-size:.62rem;font-weight:600;padding:1px 8px;border-radius:999px;margin-top:3px}
.kv-tier .badge.on{background:rgba(12,163,12,.13);color:#0a7a0a}
.kv-tier .badge.standby{background:rgba(42,120,214,.12);color:#2a78d6}
.kv-tier .badge.off{background:rgba(120,120,120,.14);color:#777}
@media(prefers-color-scheme:dark){
.kv-root{color:#e8e8e8}.kv-tier,.kv-tier.off{background:#1a1a19}
.kv-tier .bar-bg{background:rgba(255,255,255,.1)}
}
"""


def _tier_card(cls, icon, title, capacity, bar_pct, fraction, badge_html, subs, off=False):
    off_cls = " off" if off else ""
    pct = max(0.0, min(100.0, bar_pct))
    if pct > 0:
        pct = max(3.0, pct)
    return (
        f'<div class="kv-tier {cls}{off_cls}">'
        f'<div class="icon">{icon}</div>'
        f"<h4>{title}</h4>"
        f'<div class="cap">{capacity}</div>'
        f'<div class="bar-bg"><div class="bar-fg" style="width:{pct:.1f}%"></div></div>'
        f'<div class="fraction">{fraction}</div>'
        f"{badge_html}"
        + "".join(f'<div class="sub">{s}</div>' for s in subs)
        + "</div>"
    )


def render_tier_html(sim: SimState, mode: str) -> str:
    """渲染 NPU/CPU/Disk 三级卡片。"""
    m = MODES.get(mode, MODES["full"])
    quant = bool(m["quant"]) if sim.phase == "idle" else sim.active_quant
    tier = bool(m["tier"]) if sim.phase == "idle" else sim.active_tier
    density_label = "int8 · 2× 密度" if quant else "fp16 · 1× 密度"
    npu_cap = sim.npu_capacity_tok or (NPU_CAP_INT8 if quant else NPU_CAP_FP16)
    display_npu_usage = max(sim.peak_npu_usage, sim.npu_usage or 0.0)
    display_npu_tok = display_npu_usage * npu_cap
    display_cpu_tok = max(sim.peak_cpu_tok, sim.cpu_tok)
    display_disk_tok = max(sim.peak_disk_tok, sim.disk_tok)

    npu = _tier_card(
        "npu", "🖥️", "NPU HBM",
        f"{_fmt_gib(HBM_TOTAL_GIB)} HBM2e · KV 预算 {_fmt_gib(NPU_KV_POOL_GIB)}",
        display_npu_usage * 100,
        f"峰值 {_fmt_tokens(display_npu_tok)} / {_fmt_tokens(npu_cap)} tokens · {density_label}",
        '<span class="badge on">L0 · /metrics 实时</span>' if sim.alive else '<span class="badge standby">等待引擎</span>',
        [
            (
                f"本轮启动日志容量 {_fmt_tokens(npu_cap)} tok"
                if sim.npu_capacity_tok is not None
                else f"F1 实测容量 = baseline 的 {'2.0' if quant else '1.0'}×"
            ),
            f"HBM {_fmt_hbm(sim.hbm_mb)}",
        ],
    )
    if tier:
        cpu_budget_label = f"{_fmt_gib(CPU_GIB)}（真机 offload 预算 {_fmt_tokens(CPU_CAP_TOK)} tok）"
        cpu = _tier_card(
            "cpu", "💾", "CPU RAM", cpu_budget_label,
            display_cpu_tok / max(1, CPU_CAP_TOK) * 100,
            f"峰值 {_fmt_tokens(display_cpu_tok)} / {_fmt_tokens(CPU_CAP_TOK)} tokens",
            '<span class="badge standby">L1 · LMCache 实时</span>',
            ["Prometheus local_cache_usage"],
        )
        disk = _tier_card(
            "disk", "📀", "Disk (SSD)", f"{_fmt_gib(DISK_GIB)} demo 预算",
            display_disk_tok / max(1, DISK_CAP_TOK) * 100,
            f"峰值 {_fmt_tokens(display_disk_tok)} / {_fmt_tokens(DISK_CAP_TOK)} tokens",
            '<span class="badge standby">L2 · LMCache 实时</span>',
            ["Prometheus local_storage_usage"],
        )
    else:
        cpu = _tier_card(
            "cpu", "💾", "CPU RAM", f"{_fmt_gib(CPU_GIB)}", 0.0, "未启用",
            '<span class="badge off">关闭</span>', [], off=True,
        )
        disk = _tier_card(
            "disk", "📀", "Disk (SSD)", f"{_fmt_gib(DISK_GIB)}", 0.0, "未启用",
            '<span class="badge off">关闭</span>', [], off=True,
        )
    return f'<div class="kv-root"><div class="kv-tiers">{npu}{cpu}{disk}</div></div>'


def _fmt_hbm(val: int | None) -> str:
    if val is None:
        return "—"
    return f"{val:,}" + " MB"


# ============================================================
# 量化专版（同一实时需求下 fp16 vs int8 密度对比）
# ============================================================

def render_quant_html(sim: SimState, mode: str) -> str:
    """量化专版：固定 NPU 预算下直接比较 fp16 / int8 的容量密度。"""
    baseline = sim.stage_results.get("baseline") or {}
    quant = sim.stage_results.get("quant") or {}
    fp16_cap = float(baseline.get("capacity_tokens") or 0.0)
    int8_cap = float(quant.get("capacity_tokens") or 0.0)
    max_cap = max(1.0, fp16_cap, int8_cap)

    def _row(kind, label, cap, badge, detail):
        pct = cap / max_cap * 100
        act = f' <span class="act">{badge}</span>' if badge else ""
        return (
            f'<div class="row {kind}">'
            f'<div class="lab">{label}{act}</div>'
            f'<div class="bar-bg"><div class="bar-fg" style="width:{pct:.1f}%"></div></div>'
            f'<div class="st"><b>{_fmt_tokens(cap)} tokens</b> · {detail}</div>'
            f'</div>'
        )

    fp_badge = "正在实测" if sim.current_stage == "baseline" else ("本轮已实测" if fp16_cap else "等待启动")
    int8_badge = "正在实测" if sim.current_stage == "quant" else ("本轮已实测" if int8_cap else "等待启动")
    fp16 = _row("fp16", "fp16 (baseline)", fp16_cap, fp_badge, "16 bit/token")
    int8 = _row("int8", "int8 (C8)", int8_cap, int8_badge, "8 bit/token")
    if fp16_cap and int8_cap:
        ratio = int8_cap / fp16_cap
        foot = (
            f"本轮同配置预算实测：{_fmt_tokens(fp16_cap)} → {_fmt_tokens(int8_cap)} tokens，"
            f"C8 容量为 baseline 的 <b>{ratio:.2f}×</b>。"
        )
    else:
        foot = "等待本轮 baseline 与 C8 引擎依次启动；容量只读取本次引擎启动日志。"
    return (
        f'<div class="kv-root kv-q">'
        f'<div class="hd">KV量化 · 固定预算容量对比</div>'
        f'<div class="cap">{"未启动 · " if sim.phase == "idle" else ""}'
        f'NPU KV 预算 {_fmt_gib(NPU_KV_POOL_GIB)} · 条宽表示容量，不表示当前占用</div>'
        f"{fp16}{int8}"
        f'<div class="foot">{foot}</div>'
        f"</div>"
    )


_QUANT_CSS = """
.kv-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#1a1a1a;line-height:1.5}
.kv-q{margin:6px 0 2px;min-height:142px}
.kv-q .hd{font-size:.9rem;font-weight:700;margin-bottom:2px}
.kv-q .cap{font-size:.66rem;color:#898781;margin-bottom:6px}
.kv-q .row{display:flex;align-items:center;gap:10px;margin:5px 0}
.kv-q .lab{width:130px;font-weight:600;font-size:.8rem;flex-shrink:0}
.kv-q .bar-bg{flex:1;height:18px;background:rgba(0,0,0,.08);border-radius:6px;overflow:hidden}
.kv-q .bar-fg{height:100%;border-radius:6px;transition:width .4s}
.kv-q .fp16 .bar-fg{background:linear-gradient(90deg,#9ca3af,#c0c0c0)}
.kv-q .fp16.over .bar-fg{background:linear-gradient(90deg,#ef4444,#f87171)}
.kv-q .int8 .bar-fg{background:linear-gradient(90deg,#2a78d6,#5b9eed)}
.kv-q .int8.ok .bar-fg{background:linear-gradient(90deg,#1baf7a,#45d4a0)}
.kv-q .st{width:250px;font-size:.74rem;color:#52514e;flex-shrink:0}
.kv-q .act{display:inline-block;font-size:.6rem;font-weight:700;padding:0 6px;border-radius:999px;background:rgba(42,120,214,.15);color:#2a78d6;margin-left:4px}
.kv-q .foot{margin-top:8px;font-size:.8rem}
@media(prefers-color-scheme:dark){
.kv-q .cap,.kv-q .st{color:#a8a8a8}.kv-q .bar-bg{background:rgba(255,255,255,.1)}
}
"""

KV_PAGE_CSS = _KV_CSS + _QUANT_CSS


@lru_cache(maxsize=1)
def load_static_experiment_results() -> dict[str, object]:
    """Load the checked-in NPU experiment snapshot used by the final reveal."""
    with _STATIC_RESULTS_PATH.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("sections"), list):
        raise ValueError(f"invalid static experiment data: {_STATIC_RESULTS_PATH}")
    return data


def render_static_experiment_results_md() -> str:
    """Render the historical experiment snapshot without runtime calculations."""
    data = load_static_experiment_results()
    blocks = [
        f"### {data['title']}",
        str(data["subtitle"]),
        f"实验环境：`{data['environment']}`",
    ]
    for raw_section in data["sections"]:
        if not isinstance(raw_section, dict):
            continue
        columns = raw_section.get("columns") or []
        rows = raw_section.get("rows") or []
        table_lines = [
            f"#### {raw_section.get('title', '')}",
            "| " + " | ".join(str(item) for item in columns) + " |",
            "|" + "|".join("---" for _ in columns) + "|",
        ]
        for row in rows:
            table_lines.append("| " + " | ".join(str(item) for item in row) + " |")
        table_lines.append(f"\n数据来源：`{raw_section.get('source', '')}`")
        blocks.append("\n".join(table_lines))
    return "\n\n".join(blocks)


def should_reveal_static_results(
    mode: str, *, stage_index: int, stage_count: int, failed: bool,
) -> bool:
    """Reveal historical results only after the complete three-stage demo succeeds."""
    return mode == "full" and not failed and stage_count == 3 and stage_index == stage_count


def render_all(sim: SimState, mode: str) -> tuple[str, str]:
    """Gradio 事件用：状态 → (tier_html, quant_html)。"""
    return render_tier_html(sim, mode), render_quant_html(sim, mode)
