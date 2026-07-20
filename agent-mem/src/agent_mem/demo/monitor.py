"""实时监控 + 历史 before/after 数据层（**纯 Python，无 GUI 依赖，可 pytest**）。

供 ``demo/chat_app.py`` 右列监控面板调用：

- :class:`LiveMonitor` —— 后台 daemon 线程，定时采 NPU HBM（复用
  :class:`agent_mem.bench.mem_sampler.NpuSmiBackend`，subprocess ``npu-smi``，
  headless 可跑、无需 torch）+ 抓 vLLM ``/metrics``（复用
  :func:`agent_mem.bench.vllm_metrics.scrape` / :func:`kv_cache_hit_rate`），
  维护一个滚动 buffer（最近 ``window_s`` 秒）。
- :func:`engine_status` —— 探活 ``<root>/health`` → ``"online"`` / ``"offline"``。
- :func:`load_history` —— 扫历史 run 的 ``metrics.json`` + ``mem_timeseries.csv``，
  按 ``config`` 分组取中位数，供 before/after 对比。

设计：引擎离线时采显存仍可（NPU 占用），``/metrics`` 失败则 KV 等指标记 ``None``，
单次采样失败不终止监控循环（与 :class:`MemSampler` 同策略）。
"""

from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import httpx

from agent_mem.bench import vllm_metrics
from agent_mem.bench.mem_sampler import MemBackend, make_backend
from agent_mem.metrics import RunMetrics


def _root_url(base_url: str) -> str:
    """从引擎 base_url（可能含 ``/v1``）推出根 URL（去 ``/v1`` 尾）。"""
    url = base_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def engine_status(base_url: str, *, timeout: float = 3.0) -> str:
    """GET ``<root>/health``：200 → ``"online"``，否则/连不上 → ``"offline"``。"""
    health = f"{_root_url(base_url)}/health"
    try:
        r = httpx.get(health, timeout=timeout)
        return "online" if r.status_code == 200 else "offline"
    except Exception:  # noqa: BLE001 — 引擎没起属正常离线态
        return "offline"


@dataclass(frozen=True)
class Sample:
    """单次监控快照（``t`` 为自监控开始的相对秒数）。

    各 ``*_sum``/``*_count``/计数器都是 **累积值**（直方图 sum/count、counter），
    供 :func:`compute_window_series` 算窗口内增量；设备/引擎抓不到的字段为 ``None``。
    """

    t: float
    mem_mb: int | None  # NPU HBM 已用 MB（gauge，瞬时）
    kv_hits: float | None  # prefix_cache_hits（counter）
    kv_queries: float | None  # prefix_cache_queries（counter）
    ttft_sum: float | None  # time_to_first_token_seconds _sum（秒）
    ttft_count: float | None
    e2e_sum: float | None  # e2e_request_latency_seconds _sum（秒）
    e2e_count: float | None
    inter_tok_sum: float | None  # request_time_per_output_token_seconds _sum（秒）
    inter_tok_count: float | None
    gen_tokens: float | None  # generation_tokens_total（counter）
    running: int | None  # num_requests_running（gauge）
    waiting: int | None  # num_requests_waiting（gauge）


# vLLM /metrics 指标名（直方图暴露 _sum/_count；counter 暴露 _total/本名；gauge 本名）
M_KV_HITS = "vllm:prefix_cache_hits"
M_KV_QUERIES = "vllm:prefix_cache_queries"
M_TTFT_SUM = "vllm:time_to_first_token_seconds_sum"
M_TTFT_COUNT = "vllm:time_to_first_token_seconds_count"
M_E2E_SUM = "vllm:e2e_request_latency_seconds_sum"
M_E2E_COUNT = "vllm:e2e_request_latency_seconds_count"
M_INTER_SUM = "vllm:request_time_per_output_token_seconds_sum"
M_INTER_COUNT = "vllm:request_time_per_output_token_seconds_count"
M_GEN_TOKENS = "vllm:generation_tokens_total"
M_RUNNING = "vllm:num_requests_running"
M_WAITING = "vllm:num_requests_waiting"


def _metrics_map(text: str) -> dict[str, float]:
    """Prometheus exposition 文本 → {metric_name: 跨 label 求和值}（一次解析）。"""
    agg: dict[str, float] = {}
    for s in vllm_metrics.parse_prometheus(text):
        agg[s.name] = agg.get(s.name, 0.0) + s.value
    return agg


def scrape_snapshot(base_url: str | None, *, timeout: float = 3.0) -> dict[str, float] | None:
    """抓一次 ``/metrics`` → 指标 map；引擎不可达返回 ``None``。"""
    if not base_url:
        return None
    try:
        text = vllm_metrics.scrape(base_url, timeout=timeout)
    except Exception:  # noqa: BLE001 — 引擎离线
        return None
    return _metrics_map(text)


@dataclass(frozen=True)
class WindowSeries:
    """``compute_window_series`` 的产物：各指标的等长时间序列（前 ``window_s`` 内为 None）。"""

    t: list[float]
    mem: list[float | None]
    kv_rate: list[float | None]  # 0~1
    throughput: list[float | None]  # tok/s
    ttft: list[float | None]  # ms
    e2e: list[float | None]  # ms
    inter_tok: list[float | None]  # ms
    running: list[float | None]
    waiting: list[float | None]


def _delta(a: Sample, b: Sample, attr: str) -> float | None:
    """``a``（后）减 ``b``（前）的某累积字段；任一 None → None。"""
    va, vb = getattr(a, attr), getattr(b, attr)
    if va is None or vb is None:
        return None
    return va - vb


def compute_window_series(samples: list[Sample], window_s: float = 10.0) -> WindowSeries:
    """对每个样本算「过去 ``window_s`` 秒」窗口内的均值/速率。

    - KV 命中率 = Δhits / Δqueries
    - TTFT / e2e / inter-token = Δsum / Δcount（秒→毫秒）
    - 吞吐 = Δgen_tokens / Δt
    - mem / running / waiting 取瞬时值
    窗口起点之前（开头 ``window_s`` 秒）的点对应值为 ``None``。
    """
    import bisect

    ts = [s.t for s in samples]
    out = WindowSeries(
        t=[], mem=[], kv_rate=[], throughput=[], ttft=[], e2e=[], inter_tok=[], running=[], waiting=[]
    )
    for i, s in enumerate(samples):
        out.t.append(s.t)
        out.mem.append(s.mem_mb)
        out.running.append(s.running)
        out.waiting.append(s.waiting)
        j = bisect.bisect_right(ts, s.t - window_s) - 1  # 窗口起点
        if j < 0:
            j = 0  # 窗口未满：用最早可用样本（部分窗口，demo 更早出值）
        if j >= i:
            # 无前置样本可配对（如首个样本）→ 无法算增量
            out.kv_rate.append(None)
            out.throughput.append(None)
            out.ttft.append(None)
            out.e2e.append(None)
            out.inter_tok.append(None)
            continue
        s0 = samples[j]
        dt = s.t - s0.t

        def ratio(sum_attr: str, count_attr: str, *, to_ms: bool = False) -> float | None:
            dc = _delta(s, s0, count_attr)
            if dc is None or dc <= 0:
                return None
            ds = _delta(s, s0, sum_attr) or 0.0
            v = ds / dc
            return v * 1000.0 if to_ms else v

        dq = _delta(s, s0, "kv_queries")
        out.kv_rate.append(None if (dq is None or dq <= 0) else (_delta(s, s0, "kv_hits") or 0.0) / dq)
        dg = _delta(s, s0, "gen_tokens")
        out.throughput.append(None if (dg is None or dt <= 0) else dg / dt)
        out.ttft.append(ratio("ttft_sum", "ttft_count", to_ms=True))
        out.e2e.append(ratio("e2e_sum", "e2e_count", to_ms=True))
        out.inter_tok.append(ratio("inter_tok_sum", "inter_tok_count", to_ms=True))
    return out


class LiveMonitor:
    """后台线程定时采显存 + vLLM 指标，维护滚动 buffer。

    用 ``monitor.start()`` 启动、``monitor.snapshot()`` 读最近 buffer、
    ``monitor.stop()`` 停。daemon 线程随主进程退出。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        interval: float = 0.5,
        window_s: float = 300.0,
        device: str = "npu",
        backend: MemBackend | None = None,
    ):
        self.base_url = base_url
        self.interval = max(0.1, float(interval))
        maxlen = max(16, int(window_s / self.interval) + 1)
        self._buf: deque[Sample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        # 显存后端：可注入（测试用 FakeBackend）；否则 npu→npu-smi、cuda→torch.cuda，失败降级 None。
        if backend is not None:
            self._backend = backend
        else:
            try:
                self._backend = make_backend(device)
            except Exception:  # noqa: BLE001
                self._backend = None

    def start(self) -> LiveMonitor:
        self._t0 = time.monotonic()
        self._stop.clear()
        s = self._sample_once()  # 同步先采一个并入 buffer，保证 latest() 启动立即可用
        with self._lock:
            self._buf.append(s)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="agent-mem-monitor")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1)

    def _sample_once(self) -> Sample:
        t = time.monotonic() - self._t0
        mem: int | None = None
        if self._backend is not None:
            try:
                mem = int(self._backend.used_mb())
            except Exception:  # noqa: BLE001 — 设备未就绪等，跳过本次显存
                mem = None
        m = scrape_snapshot(self.base_url)  # None → 所有 /metrics 字段为 None

        def pick(name: str) -> float | None:
            return None if m is None else m.get(name)

        return Sample(
            t=t,
            mem_mb=mem,
            kv_hits=pick(M_KV_HITS),
            kv_queries=pick(M_KV_QUERIES),
            ttft_sum=pick(M_TTFT_SUM),
            ttft_count=pick(M_TTFT_COUNT),
            e2e_sum=pick(M_E2E_SUM),
            e2e_count=pick(M_E2E_COUNT),
            inter_tok_sum=pick(M_INTER_SUM),
            inter_tok_count=pick(M_INTER_COUNT),
            gen_tokens=pick(M_GEN_TOKENS),
            running=None if m is None else m.get(M_RUNNING),
            waiting=None if m is None else m.get(M_WAITING),
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            # 先等再采，避免与 start() 的首次采样在 t≈0 重叠（与 MemSampler._loop 一致）
            if self._stop.wait(self.interval):
                break
            s = self._sample_once()
            with self._lock:
                self._buf.append(s)

    def snapshot(self) -> list[Sample]:
        """返回最近 buffer 的拷贝（线程安全）。"""
        with self._lock:
            return list(self._buf)

    def latest(self) -> Sample | None:
        """最近一次快照（buffer 空时 None）。"""
        with self._lock:
            return self._buf[-1] if self._buf else None


# ---- 历史 before/after ----


@dataclass(frozen=True)
class HistoryConfig:
    """一个 config（baseline / prefix-cache / optimized / ...）的中位数指标 + 代表性显存曲线。"""

    config: str
    n_runs: int
    mem_peak_mb: float
    e2e_latency_p50_ms: float
    e2e_latency_p95_ms: float
    qps: float
    kv_cache_hit_rate: float
    ttft_ms: float
    mem_curve: list[tuple[float, int]]  # (timestamp_s, used_mb)，取该 config 首条 run


def _read_mem_curve(csv_path: Path) -> list[tuple[float, int]]:
    """读 ``mem_timeseries.csv`` → [(timestamp, used_mb), ...]，坏行跳过。"""
    pts: list[tuple[float, int]] = []
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pts.append((float(row["timestamp"]), int(float(row["used_mb"]))))
            except (KeyError, ValueError, TypeError):
                continue
    return pts


def _runmetrics_from_json(data: dict) -> RunMetrics:
    """从 metrics.json dict 构造 RunMetrics（忽略未知字段，缺失项走 dataclass 默认）。"""
    fields = RunMetrics.__dataclass_fields__
    return RunMetrics(**{k: v for k, v in data.items() if k in fields})


def load_history(logs_dir: str | Path = "logs/mvp-newframework") -> list[HistoryConfig]:
    """扫 ``logs_dir`` 下 run 子目录的 ``metrics.json``，按 ``config`` 分组取中位数。

    缺 ``metrics.json`` 的目录跳过；按 config 名升序返回。目录不存在 → 空列表。
    """
    root = Path(logs_dir)
    if not root.is_dir():
        return []
    groups: dict[str, list[RunMetrics]] = {}
    curves: dict[str, list[tuple[float, int]]] = {}
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mj = run_dir / "metrics.json"
        if not mj.is_file():
            continue
        try:
            m = _runmetrics_from_json(json.loads(mj.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        groups.setdefault(m.config, []).append(m)
        # 取该 config 首条 run 的显存曲线作代表（覆盖即得，无需聚合多 run）
        if m.config not in curves:
            curve_csv = run_dir / "mem_timeseries.csv"
            if curve_csv.is_file():
                curves[m.config] = _read_mem_curve(curve_csv)

    out: list[HistoryConfig] = []
    for cfg, runs in groups.items():

        def med(field: str) -> float:
            return float(statistics.median(getattr(r, field) for r in runs))

        out.append(
            HistoryConfig(
                config=cfg,
                n_runs=len(runs),
                mem_peak_mb=med("mem_peak_mb"),
                e2e_latency_p50_ms=med("e2e_latency_p50_ms"),
                e2e_latency_p95_ms=med("e2e_latency_p95_ms"),
                qps=med("qps"),
                kv_cache_hit_rate=med("kv_cache_hit_rate"),
                ttft_ms=med("ttft_ms"),
                mem_curve=curves.get(cfg, []),
            )
        )
    out.sort(key=lambda h: h.config)
    return out
