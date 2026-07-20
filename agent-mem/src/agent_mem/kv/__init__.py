"""缝C · KV Cache 量化与分层存储。

- **F1 int8 KV 量化**：缝A（``--kv-cache-dtype int8`` flag，见
  ``configs/f1-int8.yaml``）—— 910B 无 FP8 单元，FP8 作废，int8 是 Ascend 上 KV
  量化的唯一支持格式（待真机验证 0.22.1rc1 是否恢复支持）。
- **F4 LMCache Ascend 分层**：缝C（``--kv-transfer-config`` 激活
  ``LMCacheAscendConnector``）。vllm-ascend 0.22.1rc1 已内置 connector（factory
  注册），只需 yaml 开关 + NPU 上安装 ``lmcache_ascend`` 包。见
  :mod:`agent_mem.kv.lmcache_check` 和 ``docs/F4-lmcache-ascend.md``。
- **F5/F6 的搬运机制**：借 V1 ``SimpleCPUOffloadConnector``，见
  :mod:`agent_mem.kv.connector`（策略在 :mod:`agent_mem.scheduler`）。
"""

from __future__ import annotations

from agent_mem.kv.connector import KVConnectorConfig, render_kv_connector_args
from agent_mem.kv.lmcache_check import check_lmcache_ascend, is_lmcache_ascend_available

__all__ = [
    "KVConnectorConfig",
    "render_kv_connector_args",
    "check_lmcache_ascend",
    "is_lmcache_ascend_available",
]