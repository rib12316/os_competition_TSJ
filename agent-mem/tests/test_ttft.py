"""TTFT 测量骨架测试（fake 流 + 注入 clock）。"""

from __future__ import annotations

import pytest

from agent_mem.bench.ttft import TTFTMeasurer, ttft_from_stream


def test_ttft_from_stream_measures_to_first_chunk():
    clock_vals = iter([10.0, 10.5])  # start, first chunk

    def clock() -> float:
        return next(clock_vals)

    stream = iter(["chunk1", "chunk2"])
    assert ttft_from_stream(stream, clock=clock) == 0.5


def test_ttft_from_stream_empty_returns_elapsed():
    clock_vals = iter([10.0, 10.2])

    def clock() -> float:
        return next(clock_vals)

    assert ttft_from_stream(iter([]), clock=clock) == pytest.approx(0.2)


def test_ttft_measurer_stats():
    m = TTFTMeasurer()
    for ms in (100, 120, 200, 90, 150):
        m.record_ms(ms)
    stats = m.stats()
    assert stats["count"] == 5
    assert stats["ttft_mean_ms"] == pytest.approx(132.0)
    assert stats["ttft_p50_ms"] == pytest.approx(120.0)


def test_ttft_measurer_record_seconds():
    m = TTFTMeasurer()
    m.record(0.1)  # 100ms
    assert m._samples_ms == [100.0]


def test_ttft_measurer_empty():
    m = TTFTMeasurer()
    assert m.stats()["count"] == 0
    assert m.ttft_p50_ms == 0.0
