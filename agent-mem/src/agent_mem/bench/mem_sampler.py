"""显存采样器：按时间采样写 ``mem_timeseries.csv``，峰值从时序取 max。

后端抽象（:class:`MemBackend`）便于适配 GPU / NPU：
- :class:`TorchCudaBackend` —— GPU，走 ``torch.cuda.mem_get_info``
- :class:`TorchNpuBackend` —— 昇仑 NPU，走 ``torch.npu.mem_get_info``（torch_npu 注入）
- :class:`NpuSmiBackend` —— 子进程 ``npu-smi`` 兜底
- :class:`FakeBackend` —— 测试用，注入固定序列

⚠️ 真实后端（cuda/npu）需设备启动后才能取到真值——**NPU 默认停着**，真机 benchmark
前先暂停等用户启动设备。当前无设备时 ``used_mb()`` 会抛异常，:class:`MemSampler`
会跳过该次采样并继续（不会因单次读失败终止采样）。
"""

from __future__ import annotations

import csv
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class MemSample:
    timestamp: float  # 自采样开始的相对秒数
    used_mb: int


@dataclass
class MemSeries:
    samples: list[MemSample] = field(default_factory=list)

    @property
    def peak_mb(self) -> int:
        return max((s.used_mb for s in self.samples), default=0)


class MemBackend(Protocol):
    """显存读取后端：返回当前已用显存（MB）。设备不可用时应抛异常。"""

    def used_mb(self) -> int: ...


class FakeBackend:
    """测试用：按注入序列依次返回；序列耗尽后重复最后一个值。"""

    def __init__(self, values: list[int]):
        self._values = list(values)
        self._idx = 0

    def used_mb(self) -> int:
        if not self._values:
            raise RuntimeError("FakeBackend 未注入任何值")
        if self._idx < len(self._values):
            v = self._values[self._idx]
            self._idx += 1
            return v
        return self._values[-1]


class TorchCudaBackend:
    """GPU：``torch.cuda.mem_get_info()`` → (free, total) 字节。"""

    def used_mb(self) -> int:
        import torch

        free, total = torch.cuda.mem_get_info()
        return int((total - free) // (1024 * 1024))


class TorchNpuBackend:
    """昇仑 NPU：``torch.npu.mem_get_info()``（由 torch_npu 注入）。"""

    def used_mb(self) -> int:
        import torch  # torch_npu 导入后会 patch torch.npu
        import torch_npu  # noqa: F401  触发 patch

        free, total = torch.npu.mem_get_info()
        return int((total - free) // (1024 * 1024))


def parse_npu_smi_mem(output: str) -> int | None:
    """从 ``npu-smi info`` 文本解析已用 HBM（MB），失败返回 ``None``。

    兼容两种常见格式：
    1. ``HBM-Usage : 1234 / 32768 (MB)``（``npu-smi info -t board``）
    2. 表格行末尾的 ``used / total``（``npu-smi info`` 的 Memory-Usage 列）

    ⚠️ 实际格式随 CANN 版本变化，需设备启动后用真实输出复核。
    """
    m = re.search(r"HBM[-_\s]?[Uu]sage\s*:?\s*(\d+)\s*/\s*(\d+)", output)
    if m:
        return int(m.group(1))
    # 兜底：含 Bus-Id 的 chip 行里有多个 "used / total"（Memory-Usage 在前、HBM-Usage 在后），
    # 取最后一个（HBM-Usage = NPU 显存）。真实 npu-smi 26.x 格式实测。
    for line in output.splitlines():
        if re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", line):
            pairs = re.findall(r"(\d+)\s*/\s*(\d+)", line)
            if pairs:
                return int(pairs[-1][0])  # HBM-Usage 的 used
    return None


class NpuSmiBackend:
    """NPU 兜底：子进程 ``npu-smi info`` 解析。"""

    def used_mb(self) -> int:
        out = subprocess.check_output(["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL)
        used = parse_npu_smi_mem(out)
        if used is None:
            raise RuntimeError("无法从 npu-smi 输出解析显存")
        return used


def make_backend(device: str) -> MemBackend:
    """按 device 名构造后端。``cuda`` → TorchCudaBackend；``npu`` → 优先 torch_npu，回退 npu-smi。"""
    if device == "cuda":
        return TorchCudaBackend()
    if device == "npu":
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return TorchNpuBackend()
        except Exception:
            pass
        return NpuSmiBackend()
    raise ValueError(f"未知 device={device!r}，可选：cuda | npu")


class MemSampler:
    """后台线程定时采样显存。用 ``with`` 或显式 start/stop。

    单次读失败（设备未就绪等）会跳过该次采样，不终止采样循环。
    """

    def __init__(
        self,
        backend: MemBackend,
        *,
        interval: float = 0.5,
        out_csv: str | Path | None = None,
    ):
        self.backend = backend
        self.interval = interval
        self.out_csv = Path(out_csv) if out_csv else None
        self._series = MemSeries()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._errors = 0

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._sample_once()  # 同步先采一个，保证至少 1 个样本（即使后续 run 瞬时完成）
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _sample_once(self) -> None:
        try:
            mb = int(self.backend.used_mb())
            self._series.samples.append(MemSample(time.monotonic() - self._t0, mb))
        except Exception:
            self._errors += 1

    def _loop(self) -> None:
        while not self._stop.is_set():
            # 先等再采，避免与 start() 的首次采样重叠
            if self._stop.wait(self.interval):
                break
            self._sample_once()

    def stop(self) -> MemSeries:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1)
        if self.out_csv is not None:
            self._write_csv()
        return self._series

    def _write_csv(self) -> None:
        assert self.out_csv is not None
        with self.out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "used_mb"])
            for s in self._series.samples:
                w.writerow([f"{s.timestamp:.3f}", s.used_mb])

    def __enter__(self) -> MemSampler:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
