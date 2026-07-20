"""TTFT（首 token 时间）测量 —— B 组骨架。

真实 TTFT 需要**流式**响应：从发出请求到收到第一个 token chunk 的墙钟时间。
vLLM 的流式端点是 ``POST /v1/chat/completions`` 带 ``stream=true``（SSE）。

本模块提供可单测的骨架：
- :func:`ttft_from_stream` —— 消费任意流迭代器，测到首 chunk 的秒数（clock 可注入）
- :class:`TTFTMeasurer` —— 累计多次请求的 TTFT，给 p50/p95/mean

⚠️ 真实联调（接 vLLM 流式 + P2 agent）留待引擎就绪后；当前用 fake 流验证机制。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent_mem.bench.stats import p50, p95


def ttft_from_stream(stream_iter, *, clock=time.monotonic) -> float:
    """消费 ``stream_iter`` 直到首个 chunk，返回自开始到首 chunk 的秒数。

    真实用法（httpx 流式）::

        with httpx.stream("POST", url, json=payload) as r:
            ttft = ttft_from_stream(r.iter_lines())
    """
    t0 = clock()
    for _ in stream_iter:
        return clock() - t0
    return clock() - t0  # 流里没有任何 chunk


@dataclass
class TTFTMeasurer:
    """累计多次请求的 TTFT（毫秒），提供 p50/p95/mean。"""

    _samples_ms: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        """记录一次 TTFT（秒）。"""
        self._samples_ms.append(seconds * 1000.0)

    def record_ms(self, ms: float) -> None:
        self._samples_ms.append(float(ms))

    @property
    def count(self) -> int:
        return len(self._samples_ms)

    @property
    def ttft_p50_ms(self) -> float:
        return p50(self._samples_ms)

    @property
    def ttft_p95_ms(self) -> float:
        return p95(self._samples_ms)

    @property
    def ttft_mean_ms(self) -> float:
        if not self._samples_ms:
            return 0.0
        return sum(self._samples_ms) / len(self._samples_ms)

    def stats(self) -> dict[str, float]:
        return {
            "count": float(self.count),
            "ttft_p50_ms": self.ttft_p50_ms,
            "ttft_p95_ms": self.ttft_p95_ms,
            "ttft_mean_ms": self.ttft_mean_ms,
        }
