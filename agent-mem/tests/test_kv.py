"""缝C KV connector / LMCache 配置测试（纯函数，无 NPU / 无 LMCache 安装）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, LmCacheConfig, MetricsConfig
from agent_mem.kv import (
    LMCACHE_TEMPLATE,
    KVConnectorConfig,
    lmcache_env,
    lmcache_serve_flag,
    render_kv_connector_args,
    resolve_lmcache_config,
)
from agent_mem.kv.lmcache import _packaged_template_path
from agent_mem.server.vllm_server import build_serve_args, engine_env

# ---- LMCache 解析 ----


def test_resolve_alias_uses_packaged_template():
    p = resolve_lmcache_config(None)
    assert p == _packaged_template_path()
    assert p.name == LMCACHE_TEMPLATE
    assert p.exists()  # 模板确实打包了
    for alias in ("local", "default", "builtin", ""):
        assert resolve_lmcache_config(alias) == p


def test_resolve_custom_path_passthrough():
    p = resolve_lmcache_config("/some/dir/my.yaml")
    assert p == Path("/some/dir/my.yaml")


def test_serve_flag_and_env_disabled():
    lm = LmCacheConfig()  # enabled=False
    assert lmcache_serve_flag(lm) == []


def _cfg(lmcache: LmCacheConfig) -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm", model="Qwen2.5-7B-Instruct", lmcache=lmcache),
        benchmark=BenchmarkConfig(),
        metrics=MetricsConfig(),
        config_name="x",
    )


def test_env_enabled_points_to_template():
    cfg = _cfg(LmCacheConfig(enabled=True, config_file=None))  # 用打包模板
    env = lmcache_env(cfg)
    assert "LMCACHE_CONFIG_FILE" in env
    assert Path(env["LMCACHE_CONFIG_FILE"]).name == LMCACHE_TEMPLATE


def test_env_disabled_empty():
    cfg = _cfg(LmCacheConfig(enabled=False))
    assert lmcache_env(cfg) == {}


def test_build_serve_args_emits_enable_lmcache():
    cfg = _cfg(LmCacheConfig(enabled=True))
    args = build_serve_args(cfg, model_path="models/Qwen3-0.6B")
    assert "--enable-lmcache" in args


def test_build_serve_args_no_lmcache_when_disabled():
    cfg = _cfg(LmCacheConfig(enabled=False))
    args = build_serve_args(cfg, model_path="models/Qwen3-0.6B")
    assert "--enable-lmcache" not in args


def test_engine_env_merges_lmcache():
    cfg = _cfg(LmCacheConfig(enabled=True, config_file="/abs/x.yaml"))
    env = engine_env(cfg)
    assert env["LMCACHE_CONFIG_FILE"] == "/abs/x.yaml"


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
        connector="lmcache_connector",
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
