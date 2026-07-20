"""缝C KV connector 配置测试（纯函数，无 NPU / 无引擎）。"""

from __future__ import annotations

import json

import pytest

from agent_mem.kv import KVConnectorConfig, render_kv_connector_args


# ---- KV connector 抽象 ----


def test_render_connector_none_empty():
    assert render_kv_connector_args(None) == []


def test_render_connector_produces_flag_and_json():
    kcc = KVConnectorConfig(connector="pykvconnector")
    args = render_kv_connector_args(kcc)
    i = args.index("--kv-connector")
    assert args[i + 1] == "pykvconnector"
    j = args.index("--kv-transfer-config")
    transfer = json.loads(args[j + 1])
    assert transfer["format"] == "by_layer"
    assert transfer["connector"]["name"] == "pykvconnector"


def test_render_connector_opts_and_extra():
    kcc = KVConnectorConfig(
        connector="pykvconnector",
        transfer_format="split_pytorch_serialize",
        connector_opts={"host": "127.0.0.1"},
        extra=["--max-num-seqs", "8"],
    )
    args = render_kv_connector_args(kcc)
    transfer = json.loads(args[args.index("--kv-transfer-config") + 1])
    assert transfer["format"] == "split_pytorch_serialize"
    assert transfer["connector"]["host"] == "127.0.0.1"
    assert "--max-num-seqs" in args and "8" in args


def test_connector_rejects_bad_args():
    with pytest.raises(ValueError):
        KVConnectorConfig(connector="")
    with pytest.raises(ValueError):
        KVConnectorConfig(connector="x", transfer_format="bogus")
