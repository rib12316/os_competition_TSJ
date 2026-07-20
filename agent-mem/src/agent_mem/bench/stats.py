"""统计工具：百分位 / 中位数。

供 runner 计算 e2e 延迟 p50/p95、对照聚合取中位数。纯函数、无依赖、易单测。
"""

from __future__ import annotations


def percentile(values: list[float], p: float) -> float:
    """线性插值百分位（``p`` ∈ [0, 100]）。空列表返回 0.0。

    用线性插值（与 numpy 默认 ``linear`` 一致），便于和业界 benchmark 对齐。
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"p 必须 ∈ [0,100]，得到 {p}")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def p50(values: list[float]) -> float:
    """中位数（p50）。"""
    return percentile(values, 50)


def p95(values: list[float]) -> float:
    return percentile(values, 95)


def median(values: list[float]) -> float:
    """中位数（与 :func:`p50` 一致，语义化别名）。"""
    return p50(values)
