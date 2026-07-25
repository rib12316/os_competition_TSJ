"""缝C · 通用 V1 KV connector 抽象（``--kv-transfer-config``）。

LMCache（F4）走 ``--enable-lmcache`` 专用 flag（见 :mod:`agent_mem.kv.lmcache`）；本模块覆盖
**其它** V1 KV connector——经 vLLM 0.22.1 的 ``--kv-transfer-config``（**flat schema**）启用：

- ``SimpleCPUOffloadConnector`` —— **F5/F6 借用的无损 KV offload 机制**（在 Ascend 上注册时被
  vllm-ascend 自动替换成 ``AscendSimpleCPUOffloadConnector``，NPU 原生 ``aclrtMemcpyBatchAsync``，
  支持 ``lazy_offload``；idle eviction / checkpoint 的 NPU↔CPU KV 搬运，策略见
  :mod:`agent_mem.scheduler.strategies`）。真机验证可用（2026-07-25，0.22.1rc1）。

把一个 :class:`KVConnectorConfig` 渲染成 vLLM CLI 参数（纯函数，可单测）。

.. note::
   旧版用 ``--kv-connector <name>`` + 嵌套 ``{"format":..,"connector":{..}}`` —— vLLM 0.22.1
   **不再认 ``--kv-connector``**（被拒），且 schema 改 flat。本模块已修正。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class KVConnectorConfig:
    """一个 V1 KV connector 的声明（vLLM 0.22.1 flat schema）。

    - ``connector``：vLLM connector 名（``SimpleCPUOffloadConnector`` / ``lmcache_connector`` …）。
    - ``kv_role``：``kv_both``（单机 offload）/ kv_producer / kv_consumer。
    - ``extra_config``：进 ``kv_connector_extra_config`` 的字段（如
      ``{"cpu_bytes_to_use": 4294967296, "lazy_offload": True}``）。
    - ``extra``：直接透传的原始 CLI flag（escape hatch，不经结构化）。
    """

    connector: str
    kv_role: str = "kv_both"
    extra_config: dict = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.connector:
            raise ValueError("KVConnectorConfig.connector 不能为空")


def render_kv_connector_args(kcc: KVConnectorConfig | None) -> list[str]:
    """把 :class:`KVConnectorConfig` 渲染成 vLLM CLI 参数列表。

    产出形如（vLLM 0.22.1 flat schema，**不**含被拒的 ``--kv-connector``）::

        --kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_extra_config":{...}}'

    ``None`` → 空列表（不启用任何 connector）。``extra`` 原样追加在后。
    """
    if kcc is None:
        return []
    cfg = {
        "kv_connector": kcc.connector,
        "kv_role": kcc.kv_role,
        "kv_connector_extra_config": dict(kcc.extra_config),
    }
    args = ["--kv-transfer-config", json.dumps(cfg)]
    if kcc.extra:
        args.extend(kcc.extra)
    return args
