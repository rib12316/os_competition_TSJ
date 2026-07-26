"""bench_runner 测试（无 NPU / 无引擎 / 无 gradio）。"""

from __future__ import annotations

import threading
import time

from agent_mem.demo.bench_runner import BenchHandle, run_bench_async


def test_bench_handle_snapshot_thread_safe():
    h = BenchHandle()
    assert h.snapshot()["status"] == "idle"

    def worker():
        for i in range(200):
            h.update(completed_runs=i, status="running")
            h.snapshot()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    snap = h.snapshot()
    assert snap["status"] == "running"
    assert snap["completed_runs"] == 199  # 最后一次写


def test_run_bench_async_marks_queued_then_error_on_bad_preset(tmp_path):
    """坏 preset 路径 → load_config 在 worker 里立刻失败 → status=error（快、无需 NPU）。"""
    h = BenchHandle()
    run_bench_async(
        h,
        preset_path=str(tmp_path / "does-not-exist.yaml"),
        engine_url="http://127.0.0.1:9/v1",
        run_root=str(tmp_path / "logs"),
        runs=1,
    )
    # 立即（同步部分）置 queued
    assert h.snapshot()["status"] == "queued"
    # 轮询直到终态（load_config 失败是毫秒级）
    for _ in range(50):
        if h.snapshot()["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    snap = h.snapshot()
    assert snap["status"] == "error"
    assert snap["error"]


def test_run_bench_async_is_non_blocking(tmp_path):
    """run_bench_async 必须立即返回（不阻塞调用线程）。"""
    h = BenchHandle()
    t0 = time.monotonic()
    run_bench_async(
        h,
        preset_path=str(tmp_path / "nope.yaml"),
        engine_url="http://127.0.0.1:9/v1",
        run_root=str(tmp_path / "logs"),
    )
    # 应在 1s 内返回（worker 跑在 daemon 线程）
    assert time.monotonic() - t0 < 1.0
    assert h.snapshot()["status"] in ("queued", "error", "done")
