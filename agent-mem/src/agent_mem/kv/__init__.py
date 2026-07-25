"""缝C · KV Cache 量化与分层存储（F1 int8 量化 / F4 LMCache 分层）。

- **F1 int8 KV 量化（C8）**：缝A——真 int8 KV = **C8**，不走 no-op 的
  ``--kv-cache-dtype int8``，改走 ``--quantization ascend`` + 模型目录的
  ``quant_model_description.json``（见 :mod:`agent_mem.kv.c8`）。910B 无 FP8 单元，
  FP8 作废，C8 是 Ascend 上 KV 量化的唯一支持格式。
- **F4 LMCache 分层**：缝C（``--enable-lmcache`` + ``LMCACHE_CONFIG_FILE``），见
  :mod:`agent_mem.kv.lmcache`。
- **F5/F6 的搬运机制**：借 V1 ``SimpleCPUOffloadConnector``，见
  :mod:`agent_mem.kv.connector`（策略在 :mod:`agent_mem.scheduler`）。
"""

from __future__ import annotations

from agent_mem.kv.connector import KVConnectorConfig, render_kv_connector_args
from agent_mem.kv.lmcache import (
    LMCACHE_TEMPLATE,
    lmcache_env,
    lmcache_serve_flag,
    resolve_lmcache_config,
)
# 注：c8（可 ``python -m agent_mem.kv.c8`` 运行）不在此 eager import——与 server/ 不
# eager-import 可运行的 vllm_server 一致，避免 -m 双重导入告警。直接 ``from
# agent_mem.kv.c8 import annotate_model`` 取用。

__all__ = [
    "KVConnectorConfig",
    "render_kv_connector_args",
    "LMCACHE_TEMPLATE",
    "lmcache_serve_flag",
    "lmcache_env",
    "resolve_lmcache_config",
]
