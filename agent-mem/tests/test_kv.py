"""缝C KV connector 配置测试（纯函数，无 NPU / 无引擎）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_mem.kv import KVConnectorConfig, render_kv_connector_args
from agent_mem.kv.c8 import (
    C8,
    FLOAT,
    QUANT_DESC_FILENAME,
    annotate_model,
    build_c8_quant_description,
    is_annotated,
)

# ---- KV connector 抽象（vLLM 0.22.1 flat schema）----


def test_render_connector_none_empty():
    assert render_kv_connector_args(None) == []


def test_render_connector_flat_schema_no_kv_connector_flag():
    kcc = KVConnectorConfig(connector="SimpleCPUOffloadConnector")
    args = render_kv_connector_args(kcc)
    assert "--kv-connector" not in args  # vLLM 0.22.1 不认该 flag（被拒）
    j = args.index("--kv-transfer-config")
    cfg = json.loads(args[j + 1])
    assert cfg["kv_connector"] == "SimpleCPUOffloadConnector"
    assert cfg["kv_role"] == "kv_both"
    assert cfg["kv_connector_extra_config"] == {}


def test_render_connector_extra_config_and_raw():
    kcc = KVConnectorConfig(
        connector="SimpleCPUOffloadConnector",
        extra_config={"cpu_bytes_to_use": 4294967296, "lazy_offload": True},
        extra=["--max-num-seqs", "8"],
    )
    args = render_kv_connector_args(kcc)
    cfg = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert cfg["kv_connector_extra_config"]["cpu_bytes_to_use"] == 4294967296
    assert cfg["kv_connector_extra_config"]["lazy_offload"] is True
    assert "--max-num-seqs" in args and "8" in args


def test_connector_rejects_empty_name():
    with pytest.raises(ValueError):
        KVConnectorConfig(connector="")


# ---- 缝A F1 C8 int8 KV annotated 产物（纯函数，无 NPU）----


def _fake_qwen2_model(tmp_path: Path, num_layers: int = 3) -> Path:
    """造一个最小 model 目录（config.json + index.json），用于测 c8 生成。"""
    d = tmp_path / "fake-qwen"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"num_hidden_layers": num_layers}))
    weight_map = {"lm_head.weight": "s.safetensors", "model.embed_tokens.weight": "s.safetensors"}
    for i in range(num_layers):
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weight_map[f"model.layers.{i}.self_attn.{p}.weight"] = "s.safetensors"
        for p in ("gate_proj", "up_proj", "down_proj"):
            weight_map[f"model.layers.{i}.mlp.{p}.weight"] = "s.safetensors"
        for n in ("input_layernorm", "post_attention_layernorm"):
            weight_map[f"model.layers.{i}.{n}.weight"] = "s.safetensors"
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return d


def test_c8_description_marks_all_weights_float_and_kv_c8(tmp_path):
    desc = build_c8_quant_description(_fake_qwen2_model(tmp_path, num_layers=3))
    # 所有 .weight 都是 FLOAT（缺了会 KeyError）
    assert all(v == FLOAT for k, v in desc.items() if k.endswith(".weight"))
    # kv_cache_type 触发 enable_c8_quant
    assert desc["kv_cache_type"] == C8
    # 每层 k/v_proj.kv_cache_scale → 填充 c8_quant_layers
    scale_keys = [k for k, v in desc.items() if v == C8 and k != "kv_cache_type"]
    assert len(scale_keys) == 3 * 2  # 3 层 × (k+v)
    assert "model.layers.0.self_attn.k_proj.kv_cache_scale" in desc


def test_annotate_writes_json_and_status(tmp_path):
    model = _fake_qwen2_model(tmp_path, num_layers=2)
    assert not is_annotated(model)
    out = annotate_model(model)
    assert out.name == QUANT_DESC_FILENAME and out.exists()
    assert is_annotated(model)
    # 重复写须 overwrite
    with pytest.raises(FileExistsError):
        annotate_model(model)
    annotate_model(model, overwrite=True)  # 不报错
    # 删文件恢复 stock
    out.unlink()
    assert not is_annotated(model)
