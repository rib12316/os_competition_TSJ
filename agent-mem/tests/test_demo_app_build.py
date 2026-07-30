"""demo 应用构造测试（无 NPU / 无引擎 / 不 launch）。基于 demo 分支布局。"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from agent_mem.demo import overview  # noqa: E402
from agent_mem.demo.chat_app import build_app  # noqa: E402


def test_build_app_constructs_without_engine():
    """build_app 返回 gr.Blocks（demo 分支签名），不连引擎、不采样。"""
    demo = build_app(
        engine_url="http://127.0.0.1:8000/v1",
        model="Qwen2.5-7B-Instruct",
        model_path="models/Qwen2.5-7B-Instruct",
        history_dir="logs",
        interval=0.5,
    )
    assert isinstance(demo, gr.Blocks)
    assert hasattr(demo, "_agent_mem_monitor")
    tab_labels = {
        str((component.get("props") or {}).get("label") or "")
        for component in demo.config["components"]
        if component.get("type") == "tabitem"
    }
    assert "上下文优化任务" in tab_labels
    assert any("通用改进" in lbl for lbl in tab_labels), "缺「通用改进」tab"
    assert not any("优化对比" in lbl for lbl in tab_labels), "「优化对比」tab 应已移除"
    assert "🎯 τ-bench 任务" not in tab_labels
    assert "LongBench 任务" not in tab_labels
    assert "📊 统一 Benchmark" not in tab_labels
    labels = [
        str((component.get("props") or {}).get("label") or "")
        for component in demo.config["components"]
    ]
    assert "LongBench data zip" in labels
    assert "上下文优化 Agent 对话（自动路由）" in labels
    assert "baseline 对照场景" in labels
    assert labels.count("F2 待压缩冷历史（canonical preview）") == 1
    assert labels.count("F3 外置后的 synopsis/reference") == 1
    # F5 KV 后端 radio 已移除：ours 对齐到 progress（无 offload、准入关）
    assert "F5 KV 后端（完整F5固定启用）" not in labels
    assert any("任务数 (max_tasks" in lbl for lbl in labels)
    assert any("最大 active requests" in lbl for lbl in labels)
    assert "当前层 session 状态" in labels
    assert "场景（= bench preset：suite + middleware(F2/F3) + session(F5) 都编码在内）" not in labels
    context_components = [
        component
        for component in demo.config["components"]
        if (component.get("props") or {}).get("label")
        in {
            "F2 待压缩冷历史（canonical preview）",
            "F2 压缩后冷历史（发送副本）",
            "F3 待结构化存储的工具数据",
            "F3 外置后的 synopsis/reference",
        }
    ]
    assert len(context_components) == 4
    assert all(component["type"] == "html" for component in context_components)
    assert any(
        dependency.get("api_name") == "context_task"
        and len(dependency.get("inputs") or []) == 12
        and len(dependency.get("outputs") or []) == 7
        for dependency in demo.config["dependencies"]
    )
    # 两个独立按钮（baseline / progress），共享 f5_results State 累积对比；输出含 live_plot
    assert any(
        dependency.get("api_name") == "f5_run_baseline"
        and len(dependency.get("inputs") or []) == 5
        and len(dependency.get("outputs") or []) == 6
        for dependency in demo.config["dependencies"]
    )
    assert any(
        dependency.get("api_name") == "f5_run_ours"
        and len(dependency.get("inputs") or []) == 5
        and len(dependency.get("outputs") or []) == 6
        for dependency in demo.config["dependencies"]
    )
    assert not any(
        dependency.get("api_name") == "unified_bench"
        for dependency in demo.config["dependencies"]
    )
    # 通用改进 tab：启动/停止/重置 事件已接线
    assert any(
        dependency.get("api_name") == "kv_start"
        for dependency in demo.config["dependencies"]
    )
    assert any(
        dependency.get("api_name") in {"kv_stop", "kv_reset"}
        for dependency in demo.config["dependencies"]
    )
    rendered_text = "\n".join(
        str((component.get("props") or {}).get("value") or "")
        for component in demo.config["components"]
    )
    assert "引擎 HBM 峰值" not in rendered_text
    assert "本轮 C8 vs 本轮原始 BF16 baseline" not in rendered_text
    assert "本轮 LMCache vs 本轮原始 baseline" not in rendered_text
    static_results_component = next(
        component
        for component in demo.config["components"]
        if "kv-static-results" in ((component.get("props") or {}).get("elem_classes") or [])
    )
    assert static_results_component["props"]["visible"] is False
    assert static_results_component["props"]["value"] == ""
    kv_dependencies = {
        dependency.get("api_name"): dependency
        for dependency in demo.config["dependencies"]
        if dependency.get("api_name") in {"kv_start", "kv_stop", "kv_reset"}
    }
    assert len(kv_dependencies["kv_start"].get("outputs") or []) == 6
    assert len(kv_dependencies["kv_stop"].get("outputs") or []) == 5
    assert len(kv_dependencies["kv_reset"].get("outputs") or []) == 6
    assert "N 个 agent session 同时跑在一个引擎上" not in rendered_text
    assert "最终交付两层" not in rendered_text
    elem_classes = {
        elem_class
        for component in demo.config["components"]
        for elem_class in ((component.get("props") or {}).get("elem_classes") or [])
    }
    assert {
        "live-monitor-column",
        "live-monitor-sticky",
        "live-monitor-status",
        "live-monitor-plot",
    } <= elem_classes
    monitor_plot = next(
        component
        for component in demo.config["components"]
        if "live-monitor-plot" in ((component.get("props") or {}).get("elem_classes") or [])
    )
    assert monitor_plot["type"] == "plot"
    assert monitor_plot["props"]["show_label"] is True
    assert monitor_plot["props"]["label"] == "实时监控（窗口=10s）"
    assert ".live-monitor-status" in demo.css
    assert '.live-monitor-plot [data-testid="block-label"]' in demo.css
    assert "live-monitor-divider" not in demo.css
    assert ".live-monitor-sticky.is-docked" in demo.css
    assert "position: fixed !important" in demo.css
    assert "flex-shrink: 0 !important" in demo.css
    assert "updateMonitorDock" in demo.js
    assert "column.style.minHeight" in demo.js
    assert 'column.style.removeProperty("min-height")' in demo.js
    assert "savedPanelScrollTop" in demo.js
    assert "restorePanelScroll" in demo.js


def test_overview_html_has_all_features_and_seams():
    h = overview.overview_html()
    for f in ("F1", "F2", "F3", "F4", "F5"):
        assert f in h
    for sid in ("缝 A", "缝 B", "缝 C", "缝 D", "缝 E", "缝 F", "缝 G"):
        assert sid in h


def test_engine_control_has_v1_buttons():
    """v1 改造：引擎功能多选组合（FEATURE_FLAGS + flags_for）。"""
    from agent_mem.demo.engine_control import FEATURE_FLAGS, flags_for

    for f in ("c8", "lmcache", "priority"):
        assert f in FEATURE_FLAGS, f"缺引擎功能 {f}"
    # 多选组合：c8+priority → 含两套 flag
    both = flags_for(["c8", "priority"])
    assert "--quantization" in both and "--scheduling-policy" in both
    # 不选 prefix-cache = baseline → --no-enable-prefix-caching；选了 → 不带
    assert "--no-enable-prefix-caching" in flags_for([])
    assert "--no-enable-prefix-caching" not in flags_for(["prefix-cache"])


def test_kv_optimize_ui_uses_real_monitor_samples():
    """通用改进页只用监控样本驱动三层占用和性能读数。"""
    from agent_mem.demo.kv_optimize_ui import (
        CPU_CAP_TOK,
        DISK_CAP_TOK,
        NPU_CAP_FP16,
        SimState,
        render_all,
        render_quant_html,
        render_static_experiment_results_md,
        update_from_samples,
    )
    from agent_mem.demo.monitor import Sample

    # 真机 CPU 预算 = 4 GiB → ~73K token（不是 775K）
    assert CPU_CAP_TOK < 80_000

    # 未启动 → 空
    s0 = SimState.fresh()
    assert s0.running is False and s0.t == 0
    html, qhtml = render_all(s0, "full")
    assert "未启动" in qhtml
    assert "width:0.0%" in html

    def _sample(t, gen, **kwargs):
        values = dict(
            t=t, mem_mb=42000, kv_hits=0, kv_queries=0,
            ttft_sum=0.0 if t == 0 else 0.2, ttft_count=0 if t == 0 else 1,
            e2e_sum=None, e2e_count=None, inter_tok_sum=None, inter_tok_count=None,
            gen_tokens=gen, running=1, waiting=0, kv_usage_perc=0.25,
        )
        values.update(kwargs)
        return Sample(**values)

    s = SimState.fresh()
    s.running = True
    s.phase = "task"
    s.active_tier = True
    s.current_stage = "tier"
    s.stage_results = {
        "baseline": {"capacity_tokens": NPU_CAP_FP16, "throughput": 40.0, "ttft_ms": 250.0},
        "quant": {"capacity_tokens": 1_504_128, "throughput": 50.0, "ttft_ms": 200.0},
    }
    samples = [
        _sample(0.0, 0),
        _sample(
            2.0, 100, kv_usage_perc=0.5, preemptions=2,
            lmcache_local_bytes=1024**3, lmcache_disk_bytes=512 * 1024**2,
            lmcache_requested_tokens=1000, lmcache_hit_tokens=750,
            lmcache_stored_tokens=900,
        ),
    ]
    update_from_samples(s, samples, "full")

    assert s.alive is True and s.source == "Prometheus /metrics"
    assert s.npu_tok == NPU_CAP_FP16 * 0.5
    assert 0 < s.cpu_tok < CPU_CAP_TOK
    assert 0 < s.disk_tok < DISK_CAP_TOK
    assert s.hit_rate == pytest.approx(0.75)
    assert s.throughput == pytest.approx(50.0)
    assert s.ttft == pytest.approx(200.0)
    assert s.preemptions == 2

    # 量化专版：容量条固定形成约 1:2 对比，不受当前低驻留量影响。
    q = render_quant_html(s, "full")
    assert "775.9K tokens" in q and "1.50M tokens" in q
    assert "width:51.6%" in q and "width:100.0%" in q
    assert "本轮同配置预算实测" in q
    assert "fp16 (baseline)" in q

    # 任务结束后的瞬时 KV 可归零，动画仍保留任务区间峰值。
    s.phase = "done"
    s.running = False
    s.npu_usage = 0.0
    s.running_reqs = 0
    s.waiting_reqs = 0
    tier_html, quant_html = render_all(s, "full")
    assert "峰值 388.0K" in tier_html
    assert "本轮同配置预算实测" in quant_html

    static_results = render_static_experiment_results_md()
    assert "静态导入的既有实验数据" in static_results
    assert "775,936 tok" in static_results and "1,556,992 tok" in static_results
    assert "152.8 s" in static_results and "120.8 s" in static_results
    assert "| 同预算 KV 容量 | 775,936 tok | 1,556,992 tok | 2.007× |" in static_results


def test_full_kv_demo_runs_quant_and_tiering_as_separate_real_stages():
    from agent_mem.demo.kv_optimize_ui import (
        engine_stages_for,
        should_reveal_static_results,
    )

    assert engine_stages_for("full") == [
        ["prefix-cache"],
        ["prefix-cache", "c8"],
        ["prefix-cache", "lmcache"],
    ]
    assert not should_reveal_static_results(
        "full", stage_index=2, stage_count=3, failed=False,
    )
    assert not should_reveal_static_results(
        "full", stage_index=3, stage_count=3, failed=True,
    )
    assert not should_reveal_static_results(
        "quant", stage_index=2, stage_count=2, failed=False,
    )
    assert should_reveal_static_results(
        "full", stage_index=3, stage_count=3, failed=False,
    )
