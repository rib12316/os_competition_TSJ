"""EngineManager lifecycle tests without starting vLLM."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from agent_mem.demo.engine_control import EngineManager


def _manager(tmp_path) -> EngineManager:
    return EngineManager(
        model_path="model",
        log_file=str(tmp_path / "engine.log"),
        python_exe="python",
    )


def test_stop_reclaims_external_port_engine_without_owned_handle(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    calls: list[int] = []
    monkeypatch.setattr(manager, "_free_port", lambda: calls.append(manager.port) or True)

    assert manager.proc is None
    assert manager.stop() is True
    assert calls == [8000]


def test_stop_reports_no_engine_when_port_is_clear(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    monkeypatch.setattr(manager, "_free_port", lambda: False)

    assert manager.stop() is False


def _c8_model(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    stock_index = {"weight_map": {"model.embed.weight": "model-00001.safetensors"}}
    c8_index = {
        "weight_map": {
            **stock_index["weight_map"],
            "model.layers.0.self_attn.k_proj.kv_cache_scale": (
                "kv_cache_scales.safetensors"
            ),
        }
    }
    description = {
        "model.embed.weight": "FLOAT",
        "kv_cache_type": "C8",
        "model.layers.0.self_attn.k_proj.kv_cache_scale": "C8",
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(stock_index), encoding="utf-8"
    )
    (model_dir / "model.safetensors.index.json.c8bak").write_text(
        json.dumps(c8_index), encoding="utf-8"
    )
    (model_dir / "quant_model_description.json.c8bak").write_text(
        json.dumps(description), encoding="utf-8"
    )
    save_file(
        {
            "model.layers.0.self_attn.k_proj.kv_cache_scale": np.asarray(
                [0.01, 0.02], dtype=np.float32
            )
        },
        str(model_dir / "kv_cache_scales.safetensors.c8bak"),
    )
    return model_dir, stock_index, c8_index


def test_c8_runtime_activation_and_restore_preserve_calibrated_backups(tmp_path):
    model_dir, stock_index, c8_index = _c8_model(tmp_path)
    manager = EngineManager(
        model_path=str(model_dir),
        log_file=str(tmp_path / "engine.log"),
        python_exe="python",
    )

    manager.validate_c8_artifacts()
    manager._activate_c8_runtime()

    assert json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    ) == c8_index
    assert (model_dir / "quant_model_description.json").is_file()
    assert (model_dir / "kv_cache_scales.safetensors").is_file()
    assert (model_dir / "model.safetensors.index.json.demo-stock.bak").is_file()

    assert manager._restore_c8_runtime() is True
    assert json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    ) == stock_index
    assert not (model_dir / "quant_model_description.json").exists()
    assert not (model_dir / "kv_cache_scales.safetensors").exists()
    assert (model_dir / "quant_model_description.json.c8bak").is_file()
    assert (model_dir / "kv_cache_scales.safetensors.c8bak").is_file()


def test_c8_preflight_rejects_inconsistent_scale_metadata(tmp_path):
    model_dir, _, _ = _c8_model(tmp_path)
    (model_dir / "model.safetensors.index.json.c8bak").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )
    manager = EngineManager(
        model_path=str(model_dir),
        log_file=str(tmp_path / "engine.log"),
        python_exe="python",
    )

    with pytest.raises(RuntimeError, match="KV scale 映射不一致"):
        manager.validate_c8_artifacts()


def test_c8_preflight_rejects_constant_placeholder_scales(tmp_path):
    model_dir, _, _ = _c8_model(tmp_path)
    save_file(
        {
            "model.layers.0.self_attn.k_proj.kv_cache_scale": np.full(
                2, 0.05, dtype=np.float32
            )
        },
        str(model_dir / "kv_cache_scales.safetensors.c8bak"),
    )
    manager = EngineManager(
        model_path=str(model_dir),
        log_file=str(tmp_path / "engine.log"),
        python_exe="python",
    )

    with pytest.raises(RuntimeError, match="未校准的占位产物"):
        manager.validate_c8_artifacts()


def test_lmcache_engine_env_disables_conflicting_telemetry_and_matches_budget(tmp_path):
    manager = _manager(tmp_path)

    env = manager._engine_env(["lmcache"])

    assert env["VLLM_NO_USAGE_STATS"] == "1"
    assert env["LMCACHE_TRACK_USAGE"] == "false"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "4.0"
    assert env["LMCACHE_LOCAL_DISK"].endswith("logs-demo/lmcache-disk")
    assert env["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "2.0"
    assert env["LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR"] == "true"
    assert env["LMCACHE_INTERNAL_API_SERVER_ENABLED"] == "true"
    assert env["LMCACHE_INTERNAL_API_SERVER_HOST"] == "127.0.0.1"
    assert env["LMCACHE_INTERNAL_API_SERVER_PORT_START"] == "6999"


def test_kv_capacity_tokens_reads_latest_startup_value(tmp_path):
    manager = _manager(tmp_path)
    (tmp_path / "engine.log").write_text(
        "GPU KV cache size: 775,936 tokens\nGPU KV cache size: 1,556,992 tokens\n",
        encoding="utf-8",
    )

    assert manager.kv_capacity_tokens() == 1_556_992
