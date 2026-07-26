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


def test_overview_html_has_all_features_and_seams():
    h = overview.overview_html()
    for f in ("F1", "F2", "F3", "F4", "F5"):
        assert f in h
    for sid in ("缝 A", "缝 B", "缝 C", "缝 D", "缝 E", "缝 F", "缝 G"):
        assert sid in h


def test_engine_control_has_v1_buttons():
    """v1 改造：引擎按钮行含 C8(F1) / LMCache(F4) / priority(F5) / 全开 档位（CONFIG_FLAGS）。"""
    from agent_mem.demo.engine_control import CONFIG_FLAGS

    for cfg in ("baseline", "prefix-cache", "c8", "lmcache", "priority", "all-engine"):
        assert cfg in CONFIG_FLAGS, f"缺引擎档位 {cfg}"
