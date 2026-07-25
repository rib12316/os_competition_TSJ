"""缝C · 通用 V1 KV connector 抽象（``--kv-connector`` + ``--kv-transfer-config``）。

F4 LMCache Ascend 走 ``--kv-transfer-config`` 激活 ``LMCacheAscendConnector``
（见 ``docs/F4-lmcache-ascend.md``）。本模块覆盖其它 V1 KV connector：

- ``pykvconnector``（``SimpleCPUOffloadConnector``）—— **F5/F6 借用的机制**
  （idle eviction / checkpoint 的 NPU↔CPU KV 搬运，策略见
  :mod:`agent_mem.scheduler.strategies`）；
- ``MultiConnector`` / ``P2pNccl`` 等。

把一个 :class:`KVConnectorConfig` 渲染成 vLLM CLI 参数（纯函数，可单测）。真机时
``extra_args`` 自动带上这些 flag。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# V1 connector 的 transfer 格式（vLLM 约定）
_TRANSFER_FORMATS = ("by_layer", "split_pytorch_serialize")


@dataclass
class KVConnectorConfig:
    """一个 V1 KV connector 的声明。

    - ``connector``：vLLM connector 名（``pykvconnector`` / ``lmcache_connector`` …）。
    - ``transfer_format``：KV 搬运格式（默认 ``by_layer``，按层搬，适配 offload）。
    - ``connector_opts``：进 ``kv_transfer_config.connector`` 的额外字段（自由 dict）。
    - ``extra``：直接透传的原始 CLI flag（escape hatch，不经结构化）。
    """

    connector: str
    transfer_format: str = "by_layer"
    connector_opts: dict = field(default_factory=dict)
    extra: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.connector:
            raise ValueError("KVConnectorConfig.connector 不能为空")
        if self.transfer_format not in _TRANSFER_FORMATS:
            raise ValueError(
                f"transfer_format={self.transfer_format!r} 不在 {_TRANSFER_FORMATS}"
            )


def render_kv_connector_args(kcc: KVConnectorConfig | None) -> list[str]:
    """把 :class:`KVConnectorConfig` 渲染成 vLLM CLI 参数列表。

    产出形如::

        --kv-connector pykvconnector
        --kv-transfer-config '{"format":"by_layer","connector":{...}}'

    ``None`` → 空列表（不启用任何 connector）。``extra`` 原样追加在后。
    """
    if kcc is None:
        return []
    transfer = {
        "format": kcc.transfer_format,
        "connector": {"name": kcc.connector, **kcc.connector_opts},
    }
    args = [
        "--kv-connector", kcc.connector,
        "--kv-transfer-config", json.dumps(transfer),
    ]
    if kcc.extra:
        args.extend(kcc.extra)
    return args
