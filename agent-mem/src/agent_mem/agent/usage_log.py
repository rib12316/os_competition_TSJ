"""模型输入 token 的公共 JSONL 记录器，与任何 middleware/优化方法无关。"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from agent_mem import token_counting
from agent_mem.middleware import MiddlewareContext

_LOG_LOCK = threading.Lock()


def prompt_meter_enabled() -> bool:
    """Return whether strict prompt metering is enabled for this process."""
    return bool(os.environ.get("PROMPT_TOKEN_LOG"))


# Keep these wrappers local because tests and diagnostic callers patch them.
def _resolve_tokenizer_path(model: str) -> str:
    return token_counting.resolve_tokenizer_path(model)


def _get_tokenizer(model: str) -> Any:
    return token_counting.get_tokenizer(model)


def prepare_prompt_meter(model: str) -> None:
    """在 benchmark 计时前预热 tokenizer；未开启日志时不加载。"""
    if prompt_meter_enabled():
        _get_tokenizer(model)


def _count_chat_tokens(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> int:
    return token_counting.count_chat_tokens(
        tokenizer, messages, tools, chat_template_kwargs
    )


def measure_prompt_pair(
    *,
    model: str,
    original_messages: list[dict],
    transformed_messages: list[dict],
    original_tools: list[dict] | None,
    transformed_tools: list[dict] | None,
    extra_body: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """同一 tokenizer/chat template 下配对计算变换前后的完整 prompt。"""
    if not force and not prompt_meter_enabled():
        return {}
    t0 = time.monotonic()
    try:
        tokenizer = _get_tokenizer(model)
        template_kwargs = (extra_body or {}).get("chat_template_kwargs")
        original = _count_chat_tokens(
            tokenizer, original_messages, original_tools, template_kwargs
        )
        transformed = _count_chat_tokens(
            tokenizer, transformed_messages, transformed_tools, template_kwargs
        )
        return {
            "original_prompt_tokens": original,
            "transformed_prompt_tokens": transformed,
            "saved_tokens": original - transformed,
            "saved_percent": round((original - transformed) / max(1, original) * 100, 4),
            "meter_ms": round((time.monotonic() - t0) * 1000, 3),
            "tokenizer_source": _resolve_tokenizer_path(model),
        }
    except Exception as exc:  # noqa: BLE001 - 计量失败不能影响 agent
        return {
            "original_prompt_tokens": None,
            "transformed_prompt_tokens": None,
            "saved_tokens": None,
            "saved_percent": None,
            "meter_ms": round((time.monotonic() - t0) * 1000, 3),
            "meter_error": repr(exc),
        }


def log_prompt_tokens(
    ctx: MiddlewareContext,
    prompt_tokens: int | None,
    measurement: dict[str, Any] | None = None,
) -> None:
    """记录服务端 ``usage.prompt_tokens``；未配置日志路径时零开销返回。"""
    path = os.environ.get("PROMPT_TOKEN_LOG", "")
    if not path:
        return
    payload = {
        "session_id": ctx.session_id,
        "step": ctx.step,
        "prompt_tokens": prompt_tokens,
        "source": (
            "response.usage.prompt_tokens" if prompt_tokens is not None else "unavailable"
        ),
        "ts": time.time(),
    }
    payload.update(measurement or {})
    transformed = payload.get("transformed_prompt_tokens")
    payload["tokenizer_drift"] = (
        prompt_tokens - transformed
        if prompt_tokens is not None and isinstance(transformed, int)
        else None
    )
    try:
        with _LOG_LOCK:
            with open(path, "a") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
