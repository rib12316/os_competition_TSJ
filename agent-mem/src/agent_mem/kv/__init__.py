"""缝C · KV Cache 量化与分层存储（F1 int8 量化 / F4 LMCache 分层）。

- **F1 int8 KV 量化**：缝A（``--kv-cache-dtype int8`` flag，见
  ``configs/f1-int8.yaml``）—— 910B 无 FP8 单元，FP8 作废，int8 是 Ascend 上 KV
  量化的唯一支持格式（待真机验证 0.22.1rc1 是否恢复支持）。
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

__all__ = [
    "KVConnectorConfig",
    "render_kv_connector_args",
    "LMCACHE_TEMPLATE",
    "lmcache_serve_flag",
    "lmcache_env",
    "resolve_lmcache_config",
]
