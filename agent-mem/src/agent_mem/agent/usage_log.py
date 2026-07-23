"""模型输入 token 的公共 JSONL 记录器，与任何 middleware/优化方法无关。"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from agent_mem.middleware import MiddlewareContext

_LOG_LOCK = threading.Lock()
_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER_CACHE: dict[str, Any] = {}


def _resolve_tokenizer_path(model: str) -> str:
    override = os.environ.get("PROMPT_TOKENIZER_PATH", "")
    if override:
        return override
    candidates = [
        Path("/data/os_competition_TSJ/models") / model,
        Path.cwd() / "models" / model,
        Path(model),
    ]
    return str(next((path for path in candidates if path.exists()), Path(model)))


def _get_tokenizer(model: str) -> Any:
    path = _resolve_tokenizer_path(model)
    if path not in _TOKENIZER_CACHE:
        with _TOKENIZER_LOCK:
            if path not in _TOKENIZER_CACHE:
                from transformers import AutoTokenizer

                _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(
                    path, local_files_only=Path(path).exists()
                )
    return _TOKENIZER_CACHE[path]


def prepare_prompt_meter(model: str) -> None:
    """在 benchmark 计时前预热 tokenizer；未开启日志时不加载。"""
    if os.environ.get("PROMPT_TOKEN_LOG"):
        _get_tokenizer(model)


def _count_chat_tokens(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def measure_prompt_pair(
    *,
    model: str,
    original_messages: list[dict],
    transformed_messages: list[dict],
    tools: list[dict] | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """同一 tokenizer/chat template 下配对计算变换前后的完整 prompt。"""
    if not os.environ.get("PROMPT_TOKEN_LOG"):
        return {}
    t0 = time.monotonic()
    try:
        tokenizer = _get_tokenizer(model)
        template_kwargs = (extra_body or {}).get("chat_template_kwargs")
        original = _count_chat_tokens(
            tokenizer, original_messages, tools, template_kwargs
        )
        transformed = _count_chat_tokens(
            tokenizer, transformed_messages, tools, template_kwargs
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
