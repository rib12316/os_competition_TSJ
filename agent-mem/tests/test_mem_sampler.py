"""显存采样器测试（FakeBackend + npu-smi 解析；真实后端待设备验证）。"""

from __future__ import annotations

import time

import pytest

from agent_mem.bench.mem_sampler import (
    FakeBackend,
    MemSampler,
    MemSeries,
    NpuSmiBackend,
    make_backend,
    parse_npu_smi_mem,
)

# ---- npu-smi 解析 ----


def test_parse_npu_smi_mem_hbm_usage():
    out = "HBM-Usage : 1234 / 32768 (MB)\n..."
    assert parse_npu_smi_mem(out) == 1234


def test_parse_npu_smi_mem_bus_id_row():
    # 真实 npu-smi 26.x：chip 行含 Bus-Id + 两个 "x / y"（Memory-Usage 0/0、HBM-Usage 3410/65536）
    out = (
        "| 0     910B2C               | OK            | 82.3         31                      |\n"
        "| 0                         | 0000:66:00.0  | 0            0    / 0                3410 / 65536            |\n"
    )
    assert parse_npu_smi_mem(out) == 3410  # 取 HBM-Usage（最后一个），不是 Memory-Usage 的 0


def test_parse_npu_smi_mem_no_match():
    assert parse_npu_smi_mem("nothing useful here") is None


# ---- FakeBackend ----


def test_fake_backend_sequence_then_repeats_last():
    b = FakeBackend([100, 200, 300])
    assert [b.used_mb(), b.used_mb(), b.used_mb(), b.used_mb()] == [100, 200, 300, 300]


def test_fake_backend_empty_raises():
    with pytest.raises(RuntimeError):
        FakeBackend([]).used_mb()


def test_mem_series_peak_empty():
    assert MemSeries().peak_mb == 0


# ---- MemSampler ----


def test_sampler_collects_series_and_writes_csv(tmp_path):
    backend = FakeBackend([100, 500, 250, 800, 300])
    csv_path = tmp_path / "mem_timeseries.csv"
    sampler = MemSampler(backend, interval=0.02, out_csv=csv_path)
    sampler.start()
    time.sleep(0.12)
    series = sampler.stop()

    assert len(series.samples) >= 1
    assert series.peak_mb == 800  # max of collected
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "timestamp,used_mb"
    assert len(lines) == len(series.samples) + 1


def test_sampler_context_manager(tmp_path):
    backend = FakeBackend([10, 20, 30])
    with MemSampler(backend, interval=0.02) as s:
        time.sleep(0.08)
    assert s._series.samples  # 采样到了
    assert s._series.peak_mb in (10, 20, 30)


def test_sampler_resilient_to_backend_errors(tmp_path):
    class _Boom:
        def used_mb(self):
            raise RuntimeError("device off")

    sampler = MemSampler(_Boom(), interval=0.02, out_csv=tmp_path / "x.csv")
    sampler.start()
    time.sleep(0.06)
    series = sampler.stop()
    assert series.samples == []  # 全失败，无样本
    assert sampler._errors > 0


# ---- make_backend ----


def test_make_backend_unknown_raises():
    with pytest.raises(ValueError):
        make_backend("tpu")


def test_make_backend_cuda_returns_instance():
    # 不调用 used_mb（无设备），仅确认构造不抛
    assert make_backend("cuda").__class__.__name__ == "TorchCudaBackend"


def test_npu_smi_backend_class_exists():
    # 无设备时调用会抛，这里只确认类可实例化
    assert NpuSmiBackend().__class__.__name__ == "NpuSmiBackend"
