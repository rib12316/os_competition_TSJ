"""统计工具测试。"""

from __future__ import annotations

import pytest

from agent_mem.bench.stats import median, p50, p95, percentile


def test_percentile_basic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0
    assert percentile(vals, 50) == 3.0


def test_percentile_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_single():
    assert percentile([42.0], 95) == 42.0


def test_percentile_invalid_p():
    with pytest.raises(ValueError):
        percentile([1.0], 150)


def test_p50_and_median_alias():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert p50(vals) == 30.0
    assert median(vals) == 30.0


def test_p95_interpolation():
    # 100 个点，p95 应在第 94.x 附近（线性插值）
    vals = [float(i) for i in range(100)]
    assert 93.0 <= p95(vals) <= 95.0
