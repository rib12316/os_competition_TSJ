"""vLLM ``/metrics`` 抓取 + KV cache 命中率计算。

vLLM V1 注册的 prefix cache 指标（见 ``third_party/vllm/vllm/v1/metrics/loggers.py``）：
- counter ``vllm:prefix_cache_hits``  → 暴露为 ``vllm:prefix_cache_hits_total``
- counter ``vllm:prefix_cache_queries`` → 暴露为 ``vllm:prefix_cache_queries_total``

KV 命中率 = hits / queries（queries=0 时记 0.0）。
day-1 可对着 stub server 的 ``/metrics`` 联调；真值得 P1 真引擎跑起来后才有。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

# 抓取的指标基名（暴露时 counter 自动加 _total 后缀）
HITS_BASE = "vllm:prefix_cache_hits"
QUERIES_BASE = "vllm:prefix_cache_queries"


@dataclass(frozen=True)
class PromSample:
    name: str
    value: float


def parse_prometheus(text: str) -> list[PromSample]:
    """解析 Prometheus exposition 文本为样本列表（忽略注释行，剥离 labels）。"""
    out: list[PromSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)", line)
        if m:
            try:
                out.append(PromSample(m.group(1), float(m.group(2))))
            except ValueError:
                continue
    return out


def metric_sum(text: str, base: str) -> float:
    """对指标 ``base``（及其 ``_total`` 变体）跨所有 label 求和。"""
    names = {base, f"{base}_total"}
    return sum(s.value for s in parse_prometheus(text) if s.name in names)


def kv_cache_hit_rate(text: str) -> float:
    """KV cache 命中率 = hits / queries；queries=0 返回 0.0。"""
    queries = metric_sum(text, QUERIES_BASE)
    if queries <= 0:
        return 0.0
    hits = metric_sum(text, HITS_BASE)
    return hits / queries


def _metrics_url(base_url: str) -> str:
    """从引擎 base_url（可能含 ``/v1``）推出 ``/metrics`` 端点 URL。"""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return f"{url}/metrics"


def scrape(base_url: str, *, timeout: float = 10.0) -> str:
    """GET 引擎 ``/metrics``，返回 Prometheus exposition 原始文本。"""
    url = _metrics_url(base_url)
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def summarize(text: str) -> dict[str, float | str]:
    """从 exposition 文本算出关键摘要（KV 命中率 + 原始计数）。"""
    return {
        "kv_cache_hit_rate": kv_cache_hit_rate(text),
        "prefix_cache_hits": metric_sum(text, HITS_BASE),
        "prefix_cache_queries": metric_sum(text, QUERIES_BASE),
    }


def dump(path: str | Path, text: str) -> Path:
    """落 ``vllm_metrics.json``：摘要 + 原始 exposition 文本（可复现/审计）。"""
    p = Path(path)
    payload = {**summarize(text), "raw_exposition": text}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return p
