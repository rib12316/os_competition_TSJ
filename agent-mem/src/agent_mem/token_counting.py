"""Shared tokenizer loading and token counting helpers.

The module stays independent from the agent and middleware packages so prompt
metering and trigger policies can share one tokenizer cache without circular
imports.  ``transformers`` remains a lazy dependency.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

_TOKENIZER_LOCK = threading.Lock()
_TOKENIZER_CACHE: dict[str, Any] = {}


def resolve_tokenizer_path(model: str) -> str:
    """Resolve a served model name to the tokenizer used by the local engine."""
    override = os.environ.get("PROMPT_TOKENIZER_PATH", "")
    if override:
        return override
    candidates = [
        Path("/data/os_competition_TSJ/models") / model,
        Path.cwd() / "models" / model,
        Path(model),
    ]
    return str(next((path for path in candidates if path.exists()), Path(model)))


def get_tokenizer(model: str) -> Any:
    """Load and cache one Hugging Face tokenizer per resolved model path."""
    path = resolve_tokenizer_path(model)
    if path not in _TOKENIZER_CACHE:
        with _TOKENIZER_LOCK:
            if path not in _TOKENIZER_CACHE:
                from transformers import AutoTokenizer

                _TOKENIZER_CACHE[path] = AutoTokenizer.from_pretrained(
                    path, local_files_only=Path(path).exists()
                )
    return _TOKENIZER_CACHE[path]


def _encoded_length(encoded: Any) -> int:
    token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    """Count plain-text tokens without model special tokens."""
    if not text:
        return 0
    return _encoded_length(tokenizer.encode(text, add_special_tokens=False))


def count_text_chunks(tokenizer: Any, chunks: list[str]) -> int:
    """Count chunks using the same double-newline boundary as F2 serialization."""
    return count_text_tokens(tokenizer, "\n\n".join(chunk for chunk in chunks if chunk))


def count_chat_tokens(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> int:
    """Count a complete chat request, including tools and generation prompt."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        tokenize=True,
        add_generation_prompt=True,
        **(chat_template_kwargs or {}),
    )
    return _encoded_length(encoded)


def estimate_text_tokens(chunks: list[str]) -> int:
    """Language-aware fallback for callers without a configured model tokenizer.

    CJK code points are counted individually while other text is estimated from
    UTF-8 bytes.  Production F2 configs inject the engine tokenizer and do not
    use this fallback; it keeps lightweight middleware use free of transformers.
    """
    text = "\n\n".join(chunk for chunk in chunks if chunk)
    if not text:
        return 0
    cjk = 0
    other: list[str] = []
    for char in text:
        codepoint = ord(char)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            cjk += 1
        else:
            other.append(char)
    return cjk + math.ceil(len("".join(other).encode("utf-8")) / 4)
