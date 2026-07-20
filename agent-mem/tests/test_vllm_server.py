"""vLLM serve 封装测试（build_serve_args 纯函数，无需设备）。"""

from __future__ import annotations

from agent_mem.config import AppConfig, BenchmarkConfig, EngineConfig, MetricsConfig
from agent_mem.server.vllm_server import _root_url, build_serve_args


def _cfg(extra_args: list[str], model: str = "Qwen2.5-7B-Instruct") -> AppConfig:
    return AppConfig(
        engine=EngineConfig(backend="vllm", model=model, extra_args=extra_args),
        benchmark=BenchmarkConfig(),
        metrics=MetricsConfig(),
        config_name="x",
    )


def test_args_common_fields():
    args = build_serve_args(_cfg([]), model_path="models/Qwen3-0.6B", port=8000)
    assert args[0] == "--model" and args[1] == "models/Qwen3-0.6B"
    assert "--host" in args and "--port" in args
    assert "--served-model-name" in args
    # served-name 默认 = engine.model
    i = args.index("--served-model-name")
    assert args[i + 1] == "Qwen2.5-7B-Instruct"


def test_args_baseline_disables_prefix_cache():
    args = build_serve_args(
        _cfg(["--no-enable-prefix-caching"]), model_path="models/Qwen3-0.6B"
    )
    assert "--no-enable-prefix-caching" in args
    assert "--enable-prefix-caching" not in args


def test_args_optimized_fp8_shlex_split():
    # config 里 "--kv-cache-dtype fp8" 是一个带空格的字符串，应被拆成两个参数
    args = build_serve_args(
        _cfg(["--kv-cache-dtype fp8"]), model_path="models/Qwen3-0.6B"
    )
    i = args.index("--kv-cache-dtype")
    assert args[i + 1] == "fp8"


def test_args_device_and_tool_parser_and_served_override():
    args = build_serve_args(
        _cfg([]),
        model_path="models/Qwen3-0.6B",
        served_name="Qwen3-0.6B",
        device="cpu",  # 仅 cpu/cuda/tpu/xpu 显式传 --device
        tool_call_parser="hermes",
    )
    i = args.index("--served-model-name")
    assert args[i + 1] == "Qwen3-0.6B"
    assert "--device" in args and args[args.index("--device") + 1] == "cpu"
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "hermes"


def test_args_npu_does_not_emit_device():
    # NPU 由 vllm-ascend 自动识别——不传 --device（api_server 不认 npu/auto）
    for d in ("npu", "auto", None):
        args = build_serve_args(_cfg([]), model_path="models/Qwen3-0.6B", device=d)
        assert "--device" not in args


def test_root_url_strips_v1():
    assert _root_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000"
    assert _root_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
