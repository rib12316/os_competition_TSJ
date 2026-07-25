"""缝C · KV Cache 量化与分层存储。

- **F1 int8 KV 量化（C8）**：缝A——真 int8 KV = **C8**，不走 no-op 的
  ``--kv-cache-dtype int8``，改走 ``--quantization ascend`` + 模型目录的
  ``quant_model_description.json``（见 :mod:`agent_mem.kv.c8`）。910B 无 FP8 单元，
  FP8 作废，C8 是 Ascend 上 KV 量化的唯一支持格式。
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
# 注：c8（可 ``python -m agent_mem.kv.c8`` 运行）不在此 eager import——与 server/ 不
# eager-import 可运行的 vllm_server 一致，避免 -m 双重导入告警。直接 ``from
# agent_mem.kv.c8 import annotate_model`` 取用。

__all__ = [
    "KVConnectorConfig",
    "render_kv_connector_args",
    "check_lmcache_ascend",
    "is_lmcache_ascend_available",
]