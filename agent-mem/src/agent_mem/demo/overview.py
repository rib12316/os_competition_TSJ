"""架构总览内容（静态，给评委一眼看全）—— demo T1 用。

结构化数据逐字取自 ``docs/agent-mem-v1-overview.html``（保持单一信息源）。
``overview_html()`` 渲染成自包含 HTML 字符串，喂给 ``gr.HTML``——评委一滚到底。
无 gradio 依赖（纯数据 + 字符串），可单测。
"""

from __future__ import annotations

# ---- 硬件栈 chips ----
STACK_CHIPS = [
    "Ascend 910B2C · 64GB HBM",
    "vllm-ascend 0.22.1rc1（CUDA-free）",
    "910B = BF16 → KV 走 int8/C8",
    "5 功能 F1–F5 · 7 条稳定缝",
    "统一 benchmark（tau-bench + longbench）",
]

# ---- 7 缝 ----
SEAMS = [
    {"id": "A", "name": "引擎 flag", "who": "F1"},
    {"id": "B", "name": "引擎后端", "who": "F8（未整合）"},
    {"id": "C", "name": "KV connector（统一槽，yaml 二选一）", "who": "F4 / F5-P2"},
    {"id": "D", "name": "上下文中间件", "who": "F2 / F3"},
    {"id": "E", "name": "Session 生命周期", "who": "F5"},
    {"id": "F", "name": "自带 workload", "who": "baseline"},
    {"id": "G", "name": "部署", "who": "F9（待补）"},
]

# ---- 痛点 → 功能 ----
PAIN_ROWS = [
    {"pain": "① 系统/工具 prompt 每轮重复", "conseq": "prefill 反复算前缀，TTFT 高", "sol": "prefix cache（MVP）+ F2 压缩", "sce": "β"},
    {"pain": "② 多轮 KV 持续累积，单 session 寿命长", "conseq": "HBM 吃紧 → 抢占重算 / OOM", "sol": "F5 动态回收 + F4 分层", "sce": "α"},
    {"pain": "③ 工具返回值（HTML/长 JSON）一次性进 context", "conseq": "context/KV 膨胀", "sol": "F3 lazy-load", "sce": "β"},
    {"pain": "④ KV 显存绝对占用大", "conseq": "并发上限低、易 OOM", "sol": "F1 C8 量化", "sce": "α"},
    {"pain": "⑤ 多 session 并发 + 价值异构（忙/闲不一）", "conseq": "抢占误伤活跃 session", "sol": "F5 会话感知调度", "sce": "α"},
]

# ---- 5 功能卡 ----
FEATURE_CARDS = [
    {"key": "F1", "title": "C8 int8 KV 量化", "seam": "缝A · 痛点④", "color": "#2a78d6",
     "mech": "--quantization ascend + c8.py post-RoPE 校准 + sitecustomize Qwen2 补丁，让 int8 KV 真生效",
     "kpis": [{"v": "2.0×", "l": "同 HBM token 容量"}, {"v": "+35%", "l": "并发16 吞吐"}],
     "necessity": "免费 --kv-cache-dtype int8 是 no-op；C8 才真砍显存"},
    {"key": "F2", "title": "Prompt 压缩", "seam": "缝D · 痛点①", "color": "#eb6834",
     "mech": "LLMLingua-2 BERT 压冷历史，热尾+system 原样；tool-aware 保护关键 JSON",
     "kpis": [{"v": "−19%", "l": "tau-bench prompt"}, {"v": "−27%", "l": "长上下文"}, {"v": "≤2.6pp", "l": "成功率(噪声内)"}],
     "necessity": "prefix cache 只免重复前缀重算，F2 进一步压冷历史正文"},
    {"key": "F3", "title": "工具数据 lazy-load", "seam": "缝D · 痛点③", "color": "#1baf7a",
     "mech": "超大工具返回值外化 SQLite，context 只留引用；fetch_tool_result 按需取有界片段",
     "kpis": [{"v": "−93%", "l": "长工具结果 context"}, {"v": "−66%", "l": "LongBench prompt"}, {"v": "0", "l": "成功率下降"}],
     "necessity": "vllm 无任何 flag 处理工具大数据进 KV — 纯自研"},
    {"key": "F4", "title": "LMCache 分层", "seam": "缝C · 痛点②", "color": "#eda100",
     "mech": "统一 kv_transfer 槽激活 LMCacheAscendConnector，KV 在 NPU↔CPU↔Disk 三级分层",
     "kpis": [{"v": "−21%", "l": "单 agent p50"}, {"v": "+32%", "l": "并发4 QPS"}],
     "necessity": "KV 超 HBM 即 OOM；分层让容量突破物理 HBM"},
    {"key": "F5", "title": "动态回收 + 会话感知调度", "seam": "缝E · 痛点②⑤", "color": "#e87ba4",
     "mech": "AdmissionController（KV-pool 准入闸门）+ priority 抢占 + combined（EWMA recency + progress/SRTF）+ think-time 用户活跃度",
     "kpis": [{"v": "0", "l": "抢占次数"}, {"v": "0.46→0.93", "l": "KV 命中率"}, {"v": "−21%", "l": "e2e p50"}, {"v": "3.5×", "l": "progress p50"}],
     "necessity": "免费 priority flag 单独 ≈FCFS；增益全来自我们准入+combined 信号"},
]

# ---- 三档递进 ----
TIERS = [
    {"step": "T0 · baseline", "title": "绝对零点", "desc": "prefix cache OFF，全功能关（仅画递进图最左端）", "cls": "t0"},
    {"step": "T1 · MVP（vllm 白送）", "title": "纯 flag 能拿到的", "desc": "prefix cache ON + 其它免费 flag（含实测 no-op 的 int8 / priority）", "cls": "t1"},
    {"step": "T2 · agent-mem v1（ours）", "title": "T1 + 我们写的代码", "desc": "F1 / F2 / F3 / F4 / F5 — 全部需自研或集成", "cls": "t2"},
]

NOOP_FLAGS = [
    {"flag": "prefix caching（V1 默认）", "effect": "真有用", "conclusion": "解决痛点①（prompt 重复）", "need": "作 MVP 基线保留"},
    {"flag": "--kv-cache-dtype int8", "effect": "no-op", "conclusion": "0.22.1rc1 后端硬断言 scales==1，不路由 C8", "need": "F1：c8.py 校准 + Qwen2 patch"},
    {"flag": "--scheduling-policy priority", "effect": "≈FCFS", "conclusion": "所有 session 同优先级时几乎无增益（A≈B）", "need": "F5：准入控制 + combined priority"},
]

# ---- 触发场景 ----
SCENARIOS = [
    {"family": "α 并发资源受限", "preset": "unified-tau-freq", "workload": "6 并发 τ-bench + 小 KV pool（util 0.27）+ think-time profiles", "triggers": "F5 / F1 / F4"},
    {"family": "β 长上下文 / 大工具", "preset": "unified-longbench", "workload": "LongBench 2WikiMQA（大 context 当工具结果）", "triggers": "F2 / F3"},
    {"family": "γ 集成（headline）", "preset": "综合 workload", "workload": "α 压力 + β 长context 同时造", "triggers": "全部"},
]

# ---- 核心指标 ----
METRICS_ROWS = [
    {"group": "资源", "metrics": "mem_peak_mb + KV token 容量", "claim": "显存↓（40 分主战场）", "who": "F1 / F4"},
    {"group": "延迟", "metrics": "e2e_p50/p95 + ttft + qps", "claim": "延迟↓", "who": "F2 / F5 / 全局"},
    {"group": "质量", "metrics": "task_success_rate", "claim": "成功率不掉（红线 ≤2pp）", "who": "全部"},
    {"group": "长生命周期", "metrics": "preemptions + kv_hit + idle_hit_rate", "claim": "动态资源回收", "who": "F5"},
    {"group": "压缩/工具", "metrics": "prompt_tokens + 工具结果节省", "claim": "上下文压缩 / 工具数据", "who": "F2 / F3"},
]

BANNER = ("⚠️ 数字来自各功能分支真机实测；v1 统一重跑待 NPU 启用。"
          "所有对照：同硬件/模型/prompt/seed，3 次取中位数。")


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_STYLE = """
<style>
.ov-wrap{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;line-height:1.55;max-width:980px;margin:0 auto}
.ov-chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.ov-chip{font-size:.74rem;padding:3px 9px;border:1px solid rgba(11,11,11,.12);border-radius:999px;color:#52514e;background:#fcfcfb}
.ov-banner{margin:8px 0 14px;padding:8px 12px;border-left:3px solid #fab219;background:#fcfcfb;border-radius:0 6px 6px 0;color:#52514e;font-size:.8rem}
.ov-h2{font-size:1.15rem;margin:18px 0 6px;border-bottom:1px solid #e1e0d9;padding-bottom:3px}
.ov-h3{font-size:.95rem;margin:12px 0 4px;color:#52514e;font-weight:600}
.ov-table{width:100%;border-collapse:collapse;font-size:.8rem;margin:4px 0}
.ov-table th{background:#f9f9f7;color:#52514e;font-weight:600;text-align:left;padding:5px 8px;border-bottom:1px solid #e1e0d9;font-size:.72rem;text-transform:uppercase}
.ov-table td{padding:5px 8px;border-bottom:1px solid #e1e0d9;vertical-align:top}
.ov-seams{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;margin:4px 0}
.ov-seam{border:1px solid rgba(11,11,11,.12);border-radius:8px;padding:8px 6px;text-align:center;background:#fcfcfb}
.ov-seam .id{font-size:.68rem;color:#898781;font-weight:700}.ov-seam .nm{font-size:.76rem;margin:2px 0;font-weight:600}.ov-seam .who{font-size:.7rem;color:#52514e}
.ov-cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:6px 0}
.ov-card{border:1px solid rgba(11,11,11,.12);border-radius:10px;padding:12px 14px;background:#fcfcfb}
.ov-card.full{grid-column:1/-1}
.ov-card h4{margin:0 0 4px;font-size:.92rem;display:flex;align-items:center;gap:8px}
.ov-dot{width:11px;height:11px;border-radius:3px;display:inline-block;flex:none}
.ov-tag{font-size:.68rem;color:#52514e;font-weight:600}
.ov-mech{color:#52514e;font-size:.78rem;margin:.2em 0 .5em}
.ov-kpis{display:flex;gap:14px;flex-wrap:wrap;margin:4px 0}
.ov-kpi .v{font-size:1.02rem;font-weight:700;font-variant-numeric:tabular-nums}.ov-kpi .l{font-size:.66rem;color:#898781;text-transform:uppercase}
.ov-nec{margin-top:6px;font-size:.74rem;color:#52514e;padding:5px 8px;background:#f9f9f7;border-radius:5px;border-left:2px solid #eda100}
.ov-tiers{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;margin:6px 0;border:1px solid rgba(11,11,11,.12);border-radius:10px;overflow:hidden}
.ov-tier{padding:12px 14px}.ov-tier.t0{background:#f9f9f7}.ov-tier.t1{background:rgba(134,182,239,.18)}.ov-tier.t2{background:rgba(57,135,229,.32);color:#fff}
.ov-tier .step{font-size:.66rem;text-transform:uppercase;letter-spacing:.05em;opacity:.85;font-weight:700}.ov-tier .tt{font-size:.92rem;font-weight:700;margin:3px 0}.ov-tier .dd{font-size:.74rem;opacity:.92}
.ov-pill{font-size:.68rem;padding:1px 7px;border-radius:999px;font-weight:600}.ov-pill.noop{background:rgba(208,59,59,.16);color:#d03b3b}.ov-pill.weak{background:rgba(236,131,90,.18);color:#ec835a}.ov-pill.ok{background:rgba(12,163,12,.16);color:#0ca30c}
@media(prefers-color-scheme:dark){.ov-wrap{color:#fff}.ov-chip,.ov-banner,.ov-seam,.ov-card,.ov-tier.t0{background:#1a1a19;color:#c3c2b7}.ov-table th{background:#0d0d0d}.ov-table td,.ov-h2{border-color:#2c2c2a}.ov-nec{background:#0d0d0d}.ov-tier.t1{background:rgba(85,152,231,.2)}.ov-tier.t2{background:rgba(57,135,229,.42);color:#fff}}
</style>
"""


def overview_html() -> str:
    """渲染整个总览 tab（一个 gr.HTML）。"""
    chips = "".join(f'<span class="ov-chip">{_esc(c)}</span>' for c in STACK_CHIPS)
    seams = "".join(
        f'<div class="ov-seam"><div class="id">缝 {s["id"]}</div><div class="nm">{_esc(s["name"])}</div><div class="who">{_esc(s["who"])}</div></div>'
        for s in SEAMS
    )
    pain = (
        '<table class="ov-table"><thead><tr><th>痛点（agent 内存特征）</th><th>不优化的后果</th><th>我们的解</th><th>场景</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{_esc(r["pain"])}</td><td>{_esc(r["conseq"])}</td><td>{_esc(r["sol"])}</td><td>{_esc(r["sce"])}</td></tr>'
            for r in PAIN_ROWS
        )
        + "</tbody></table>"
    )
    cards = []
    for f in FEATURE_CARDS:
        kpis = "".join(
            f'<div class="ov-kpi"><div class="v">{_esc(k["v"])}</div><div class="l">{_esc(k["l"])}</div></div>'
            for k in f["kpis"]
        )
        cls = "ov-card full" if f["key"] == "F5" else "ov-card"
        cards.append(
            f'<div class="{cls}"><h4><span class="ov-dot" style="background:{f["color"]}"></span>'
            f'{_esc(f["key"])} · {_esc(f["title"])} <span class="ov-tag">{_esc(f["seam"])}</span></h4>'
            f'<div class="ov-mech">{_esc(f["mech"])}</div>'
            f'<div class="ov-kpis">{kpis}</div>'
            f'<div class="ov-nec">必要性：{_esc(f["necessity"])}</div></div>'
        )
    tiers = "".join(
        f'<div class="ov-tier {t["cls"]}"><div class="step">{_esc(t["step"])}</div><div class="tt">{_esc(t["title"])}</div><div class="dd">{_esc(t["desc"])}</div></div>'
        for t in TIERS
    )
    noop = (
        '<table class="ov-table"><thead><tr><th>vllm 免费 flag</th><th>开了之后</th><th>结论</th><th>我们的回应</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td><code>{_esc(n["flag"])}</code></td><td><span class="ov-pill {"noop" if n["effect"]=="no-op" else "weak" if "FCFS" in n["effect"] else "ok"}">{_esc(n["effect"])}</span></td>'
            f'<td>{_esc(n["conclusion"])}</td><td>{_esc(n["need"])}</td></tr>'
            for n in NOOP_FLAGS
        )
        + "</tbody></table>"
    )
    sce = (
        '<table class="ov-table"><thead><tr><th>场景族</th><th>preset</th><th>workload</th><th>触发谁</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td><strong>{_esc(s["family"])}</strong></td><td><code>{_esc(s["preset"])}</code></td><td>{_esc(s["workload"])}</td><td>{_esc(s["triggers"])}</td></tr>'
            for s in SCENARIOS
        )
        + "</tbody></table>"
    )
    metrics = (
        '<table class="ov-table"><thead><tr><th>组</th><th>指标</th><th>命中评分主张</th><th>主要体现</th></tr></thead><tbody>'
        + "".join(
            f'<tr><td>{_esc(m["group"])}</td><td><code>{_esc(m["metrics"])}</code></td><td>{_esc(m["claim"])}</td><td>{_esc(m["who"])}</td></tr>'
            for m in METRICS_ROWS
        )
        + "</tbody></table>"
    )
    return (
        f"{_STYLE}<div class=\"ov-wrap\">"
        f'<div class="ov-chips">{chips}</div>'
        f'<div class="ov-banner">{_esc(BANNER)}</div>'
        f'<div class="ov-h2">1. 痛点 → 功能映射</div>{pain}'
        f'<div class="ov-h2">2. 架构：7 条稳定缝</div><div class="ov-seams">{seams}</div>'
        f'<div class="ov-h2">3. 我们干了什么（5 个功能）</div><div class="ov-cards">{"".join(cards)}</div>'
        f'<div class="ov-h2">4. 评估：baseline → MVP → ours 三档递进</div><div class="ov-tiers">{tiers}</div>'
        f'<div class="ov-h3">T1 里的 no-op flag（这是必要性证明，不是凑数）</div>{noop}'
        f'<div class="ov-h2">5. 测试场景：尊重每个功能的触发条件</div>{sce}'
        f'<div class="ov-h2">6. 核心指标</div>{metrics}'
        f"</div>"
    )
