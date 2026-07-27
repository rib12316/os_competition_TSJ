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
    assert "🎯 τ-bench 任务" not in tab_labels
    assert "LongBench 任务" not in tab_labels
    labels = [
        str((component.get("props") or {}).get("label") or "")
        for component in demo.config["components"]
    ]
    assert "LongBench data zip" in labels
    assert "上下文优化 Agent 对话（自动路由）" in labels
    assert "baseline 对照场景" in labels
    assert labels.count("F2 待压缩冷历史（canonical preview）") == 1
    assert labels.count("F3 外置后的 synopsis/reference") == 1
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
