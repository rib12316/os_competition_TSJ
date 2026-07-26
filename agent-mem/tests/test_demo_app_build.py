"""demo 应用构造测试（无 NPU / 无引擎 / 不 launch）。"""

from __future__ import annotations

from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from agent_mem.config import load_config  # noqa: E402
from agent_mem.demo import overview  # noqa: E402
from agent_mem.demo.chat_app import _enumerate_presets, _preview_config, build_app  # noqa: E402

CONFIGS_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_build_app_constructs_without_engine():
    """build_app 返回 gr.Blocks，不连引擎、不采样。"""
    demo = build_app(
        configs_dir=CONFIGS_DIR, model_path="models/Qwen2.5-7B-Instruct",
        history_dir="logs", interval=0.5, run_root="logs", device="npu",
    )
    assert isinstance(demo, gr.Blocks)
    assert hasattr(demo, "_agent_mem_monitor")


def test_enumerate_presets_filters_to_meaningful():
    presets = _enumerate_presets(CONFIGS_DIR)
    stems = [s for s, _ in presets]
    # 关键 preset 都在
    for must in ("baseline", "f1-bench-c8", "f4-lmcache", "unified-tau-freq", "unified-longbench"):
        assert must in stems, f"missing {must}"
    # 每个 preset 都能 load_config（schema 没漂）
    for _, path in presets:
        load_config(path)


def test_preview_config_renders():
    presets = dict(_enumerate_presets(CONFIGS_DIR))
    md = _preview_config("baseline", presets)
    assert "baseline" in md and "suite" in md


def test_overview_html_has_all_features_and_seams():
    h = overview.overview_html()
    for f in ("F1", "F2", "F3", "F4", "F5"):
        assert f in h
    for sid in ("缝 A", "缝 B", "缝 C", "缝 D", "缝 E", "缝 F", "缝 G"):
        assert sid in h
