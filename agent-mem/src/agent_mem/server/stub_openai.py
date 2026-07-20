"""假 OpenAI 兼容 server（stdlib ``http.server`` 实现）。

用途：在 P1 的真实 vLLM 引擎就绪前，给 P2 的 agent loop 提供一个可联调的
OpenAI 兼容端点，让多轮 tool-calling loop 能跑通。骨架阶段非流式 only。

端点：
- ``POST /v1/chat/completions`` —— 请求带 ``tools`` 时返回一个 tool_call 响应
  （取请求里第一个 tool 的名字、空参数 ``{}``），否则返回普通 text 响应。
- ``GET  /v1/models``            —— 返回模型列表。
- ``GET  /metrics``              —— 返回固定 0 的 Prometheus 文本（供 P3 抓取器联调）。

返回结构对齐 openai SDK 2.x ``ChatCompletion`` schema（含 ``tool_calls``）。
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "Qwen2.5-7B-Instruct"

# 固定假返回用到的 id（确定性，便于测试）
_COMPLETION_ID = "chatcmpl-stub-0001"
_TOOL_CALL_ID = "call_stub_0"
_CREATED = 1700000000

# /metrics 端点返回的 Prometheus 文本（KV 命中率=0，P3 抓取器联调用）
# 指标名对齐 vLLM V1 真实注册名（vllm/v1/metrics/loggers.py）：
#   counter "vllm:prefix_cache_hits" / "vllm:prefix_cache_queries"
# Prometheus 暴露时 counter 自动加 _total 后缀。
_METRICS_TEXT = (
    "# HELP vllm:prefix_cache_hits_total Prefix cache hits.\n"
    "# TYPE vllm:prefix_cache_hits_total counter\n"
    "vllm:prefix_cache_hits_total 0\n"
    "# HELP vllm:prefix_cache_queries_total Prefix cache queries.\n"
    "# TYPE vllm:prefix_cache_queries_total counter\n"
    "vllm:prefix_cache_queries_total 0\n"
)


# ---- 响应构造 ----


def _usage(prompt_tokens: int = 128, completion_tokens: int = 16) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def tool_call_response(
    model: str,
    *,
    tool_name: str = "calculate",
    tool_args: dict[str, Any] | None = None,
    prompt_tokens: int = 128,
    completion_tokens: int = 16,
) -> dict[str, Any]:
    """构造一个合法的 tool_call chat.completion dict。"""
    return {
        "id": _COMPLETION_ID,
        "object": "chat.completion",
        "created": _CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": _TOOL_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args or {}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def text_response(
    model: str,
    *,
    content: str = "ok",
    prompt_tokens: int = 128,
    completion_tokens: int = 16,
) -> dict[str, Any]:
    """构造一个普通 text chat.completion（finish_reason=stop）。"""
    return {
        "id": _COMPLETION_ID,
        "object": "chat.completion",
        "created": _CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": _usage(prompt_tokens, completion_tokens),
    }


def models_list(models: list[str]) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": _CREATED, "owned_by": "stub"}
            for m in models
        ],
    }


# ---- HTTP handler ----


def _make_handler(model: str, *, verbose: bool) -> type[BaseHTTPRequestHandler]:
    """构造一个绑定了 model 配置的 handler 子类。"""

    class _StubHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                obj = json.loads(raw)
                return obj if isinstance(obj, dict) else {}
            except (ValueError, TypeError):
                return {}

        def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler 约定)
            path = urlparse(self.path).path
            if path == "/v1/chat/completions":
                self._handle_chat()
            else:
                self._send_json(404, {"error": {"message": f"unknown path {path}"}})

        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/v1/models":
                self._send_json(200, models_list([model]))
            elif path == "/metrics":
                body = _METRICS_TEXT.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json(404, {"error": {"message": f"unknown path {path}"}})

        def _handle_chat(self) -> None:
            body = self._read_body()
            tools = body.get("tools")
            req_model = body.get("model") or model
            if tools:  # 请求带 tools → 返回第一个 tool 的 tool_call
                first = tools[0] if isinstance(tools, list) else {}
                fn = first.get("function", {}) if isinstance(first, dict) else {}
                name = fn.get("name", "calculate") if isinstance(fn, dict) else "calculate"
                payload = tool_call_response(req_model, tool_name=name, tool_args={})
            else:
                payload = text_response(req_model)
            self._send_json(200, payload)

        def log_message(self, fmt, *args):  # noqa: A003 — 抑制 stderr 日志
            if verbose:
                super().log_message(fmt, *args)

    return _StubHandler


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> ThreadingHTTPServer:
    """构造并返回 server（**不阻塞**）。调用方负责 ``serve_forever()``。"""
    handler = _make_handler(model, verbose=verbose)
    return ThreadingHTTPServer((host, port), handler)


def start_background(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    model: str = DEFAULT_MODEL,
) -> tuple[ThreadingHTTPServer, str]:
    """在 daemon 线程里起 server，返回 ``(server, base_url)``。pytest fixture 用。"""
    import threading

    server = serve(host, port, model=model)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    actual_port = server.server_address[1]  # port=0 时回读 OS 分配的端口
    base_url = f"http://{host}:{actual_port}/v1"
    return server, base_url


def main() -> None:
    """CLI 入口：``python -m agent_mem.server.stub_openai [--port N] [--model M]``。"""
    import argparse

    parser = argparse.ArgumentParser(description="stub OpenAI server (非流式)")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-v", "--verbose", action="store_true", help="打印访问日志")
    args = parser.parse_args()

    server = serve(args.host, args.port, model=args.model, verbose=args.verbose)
    print(
        f"[stub-openai] listening on http://{args.host}:{args.port}/v1 "
        f"(model={args.model}) — Ctrl-C 退出",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stub-openai] shutting down", flush=True)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
