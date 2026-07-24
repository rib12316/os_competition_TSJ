#!/usr/bin/env python3
"""Reproducible long-context exercise for the F2 tool-aware LLMLingua-2 path."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from agent_mem.middleware.base import MiddlewareContext
from agent_mem.middleware.compress import CompressMiddleware, _tool_aware_segments

DEFAULT_TOKENIZER = "/data/os_competition_TSJ/models/Qwen2.5-7B-Instruct"
DEFAULT_WORKER = "/data/os_competition_TSJ/.venv-compress/bin/python"
DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def _middleware(*, trigger_tokens: int) -> CompressMiddleware:
    return CompressMiddleware(
        method="llmlingua2",
        rate=0.65,
        trigger_tokens=trigger_tokens,
        recompress_delta_tokens=4000,
        keep_hot=6,
        tool_aware=True,
        assistant_rate=0.75,
        tool_result_rate=0.60,
        hot_tool_trigger_tokens=1000,
        optimize_static_prompt=False,
        device="cpu",
        model_name=DEFAULT_MODEL,
        backend="subprocess",
        worker_venv=DEFAULT_WORKER,
        worker_pool_size=1,
        worker_threads=32,
    )


def _split_cold_hot(messages: list[dict[str, Any]], keep_hot: int) -> tuple[list[dict], list[dict]]:
    rest = [message for message in messages if message.get("role") != "system"]
    split = len(rest) - keep_hot
    while split > 0 and rest[split].get("role") == "tool":
        split -= 1
    return rest[:split], rest[split:]


def _build_messages(rounds: int, detail_repeats: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": "Follow retail policy and preserve exact order, payment, and confirmation facts.",
    }]
    detail = (
        "The customer compared fit, material, color, delivery timing, return eligibility, "
        "and compatibility before deciding which option best matched the original request. "
    )
    for index in range(rounds):
        order_id = f"#W{index:07d}"
        call_id = f"call-{index:03d}"
        arguments = json.dumps({"order_id": order_id}, separators=(",", ":"))
        result = {
            "order_id": order_id,
            "status": "pending" if index % 2 == 0 else "delivered",
            "total": round(42.50 + index, 2),
            "payment_method": f"card-{index % 4}",
            "shipping_address": f"{100 + index} Example Street, Test City",
            "items": [{
                "product_id": f"P-{index:04d}",
                "quantity": 1 + index % 3,
                "price": round(20.25 + index / 2, 2),
                "description": detail * detail_repeats,
                "notes": (
                    "The catalog entry includes optional accessories and several descriptive "
                    "phrases that are useful for browsing but are not transaction identifiers. "
                ) * max(1, detail_repeats // 3),
            }],
        }
        messages.extend([
            {
                "role": "user",
                "content": (
                    f"Review {order_id}. Do not change it unless I explicitly confirm; "
                    f"remember constraint-{index:03d}."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I will inspect the order first and keep the requested confirmation boundary "
                    "while checking the transaction details."
                ),
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_order_details", "arguments": arguments},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "get_order_details",
                "content": json.dumps(result, separators=(",", ":")),
            },
            {
                "role": "assistant",
                "content": (
                    f"I reviewed {order_id}. Its current state and payment facts must remain exact. "
                    "The remaining catalog prose is background information for the next decision."
                ),
            },
        ])
    return messages


def _body_tokens(
    middleware: CompressMiddleware, messages: list[dict[str, Any]], tokenizer: Any
) -> int:
    cold, _ = _split_cold_hot(messages, middleware.keep_hot)
    bodies = middleware._compressible_history_texts(cold)
    return sum(len(tokenizer.encode(body, add_special_tokens=False)) for body in bodies)


def _messages_for_target(
    target_tokens: int, middleware: CompressMiddleware, tokenizer: Any
) -> list[dict[str, Any]]:
    rounds = max(18, math.ceil(target_tokens / 320))
    best: tuple[int, list[dict[str, Any]]] | None = None
    for repeats in range(1, 80):
        messages = _build_messages(rounds, repeats)
        tokens = _body_tokens(middleware, messages, tokenizer)
        candidate = (abs(tokens - target_tokens), messages)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if tokens >= target_tokens:
            break
    assert best is not None
    return best[1]


def _plain_narrative(target_tokens: int, tokenizer: Any) -> str:
    paragraph = (
        "The product specialist reviewed fabric, fit, color, packaging, delivery timing, "
        "compatibility, return eligibility, and alternative variants. The explanation repeats "
        "background catalog language so a compression model can remove redundant prose while "
        "retaining the important comparison and decision context. "
    )
    text = ""
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens:
        text += paragraph
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _message_tokens(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _serialized_cold_tokens(
    tokenizer: Any, middleware: CompressMiddleware, messages: list[dict[str, Any]]
) -> tuple[int, str]:
    cold, _ = _split_cold_hot(messages, middleware.keep_hot)
    segments = _tool_aware_segments(
        cold,
        assistant_rate=middleware.assistant_rate,
        tool_rate=middleware.tool_result_rate,
    )
    text = "\n\n".join(segment.original_text() for segment in segments)
    return len(tokenizer.encode(text, add_special_tokens=False)), text


def _pending_event(ctx: MiddlewareContext) -> dict[str, Any]:
    return dict(ctx.scratch.get("compress:pending_event", {}))


def _audit(
    compressed: str, messages: list[dict[str, Any]], middleware: CompressMiddleware
) -> dict[str, Any]:
    cold, hot = _split_cold_hot(messages, middleware.keep_hot)
    call_ids: list[str] = []
    order_ids: list[str] = []
    arguments: list[str] = []
    user_constraints: list[str] = []
    for message in cold:
        if message.get("role") == "user":
            text = str(message.get("content") or "")
            user_constraints.append(text)
            marker = next((word.rstrip(".") for word in text.split() if word.startswith("#W")), "")
            if marker:
                order_ids.append(marker)
        for call in message.get("tool_calls") or []:
            call_ids.append(str(call.get("id") or ""))
            arguments.append(str(call.get("function", {}).get("arguments") or ""))
    hot_call_ids = {
        str(call.get("id") or "")
        for message in hot
        for call in message.get("tool_calls") or []
    }
    orphan_hot_tools = [
        str(message.get("tool_call_id") or "")
        for message in hot
        if message.get("role") == "tool"
        and str(message.get("tool_call_id") or "") not in hot_call_ids
    ]
    return {
        "all_user_messages_preserved": all(text in compressed for text in user_constraints),
        "all_call_ids_preserved": all(value in compressed for value in call_ids),
        "all_arguments_preserved": all(value in compressed for value in arguments),
        "all_order_ids_preserved": all(value in compressed for value in order_ids),
        "status_pending_present": '"status":"pending"' in compressed,
        "amount_fields_present": '"total":' in compressed and '"price":' in compressed,
        "address_fields_present": '"shipping_address":' in compressed,
        "hot_tool_protocol_valid": not orphan_hot_tools,
        "cold_call_count": len(call_ids),
    }


def _run_case(
    *,
    name: str,
    middleware: CompressMiddleware,
    messages: list[dict[str, Any]],
    tokenizer: Any,
    shared_compressor: Any,
) -> dict[str, Any]:
    middleware._compressor = shared_compressor
    ctx = MiddlewareContext(name)
    ctx.bump_step()
    canonical_tokens = _message_tokens(tokenizer, messages)
    body_tokens = _body_tokens(middleware, messages, tokenizer)
    serialized_tokens, _ = _serialized_cold_tokens(tokenizer, middleware, messages)

    started = time.monotonic()
    transformed = middleware.transform_messages(messages, ctx)
    first_ms = (time.monotonic() - started) * 1000
    event = _pending_event(ctx)
    compressed = str(ctx.scratch.get("compress", {}).get("compressed") or "")
    transformed_tokens = _message_tokens(tokenizer, transformed)
    compressed_tokens = (
        len(tokenizer.encode(compressed, add_special_tokens=False)) if compressed else 0
    )

    started = time.monotonic()
    reused = middleware.transform_messages(messages, ctx)
    reuse_ms = (time.monotonic() - started) * 1000
    reuse_event = _pending_event(ctx)

    return {
        "name": name,
        "message_count": len(messages),
        "canonical_full_prompt_tokens": canonical_tokens,
        "transformed_full_prompt_tokens": transformed_tokens,
        "full_prompt_saved_tokens": canonical_tokens - transformed_tokens,
        "full_prompt_saved_percent": round(
            (canonical_tokens - transformed_tokens) * 100 / canonical_tokens, 2
        ),
        "qwen_compressible_body_tokens": body_tokens,
        "qwen_serialized_cold_tokens": serialized_tokens,
        "qwen_compressed_cold_tokens": compressed_tokens,
        "qwen_cold_saved_percent": round(
            (serialized_tokens - compressed_tokens) * 100 / serialized_tokens, 2
        ) if compressed else 0.0,
        "middleware_event": event,
        "first_transform_ms": round(first_ms, 1),
        "reuse_action": reuse_event.get("action"),
        "reuse_transform_ms": round(reuse_ms, 1),
        "reuse_output_identical": transformed == reused,
        "audit": _audit(compressed, messages, middleware) if compressed else None,
        "compressed_preview": compressed[:1200],
    }


def _run_raw_case(compressor: Any, tokenizer: Any) -> dict[str, Any]:
    narrative = _plain_narrative(5000, tokenizer)
    original_tokens = len(tokenizer.encode(narrative, add_special_tokens=False))
    started = time.monotonic()
    result = compressor.compress_prompt(
        [narrative],
        rate=0.60,
        force_tokens=["\n", "?", "."],
        force_reserve_digit=True,
        use_context_level_filter=False,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    values = result.get("compressed_prompt_list") or [narrative]
    compressed = values[0] if values and isinstance(values[0], str) else narrative
    compressed_tokens = len(tokenizer.encode(compressed, add_special_tokens=False))
    return {
        "name": "raw_five_k_tool_narrative_rate_060",
        "qwen_original_tokens": original_tokens,
        "qwen_compressed_tokens": compressed_tokens,
        "saved_tokens": original_tokens - compressed_tokens,
        "saved_percent": round(
            (original_tokens - compressed_tokens) * 100 / original_tokens, 2
        ),
        "compress_ms": round(elapsed_ms, 1),
        "compressed_preview": compressed[:1200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True, local_files_only=True
    )
    sizing_middleware = _middleware(trigger_tokens=8000)
    messages_5k = _messages_for_target(5000, sizing_middleware, tokenizer)
    messages_9k = _messages_for_target(9000, sizing_middleware, tokenizer)

    gate_ctx = MiddlewareContext("current-gate-5k")
    gate_ctx.bump_step()
    gate_started = time.monotonic()
    sizing_middleware.transform_messages(messages_5k, gate_ctx)
    gate_ms = (time.monotonic() - gate_started) * 1000
    gate_event = _pending_event(gate_ctx)

    forced = _middleware(trigger_tokens=1)
    compressor = forced._get_compressor()
    try:
        warmup_started = time.monotonic()
        compressor.compress_prompt(
            ["Warm up the LLMLingua-2 worker before timed long-context inference."],
            rate=0.60,
            force_tokens=["\n", "?", "."],
            force_reserve_digit=True,
            use_context_level_filter=False,
        )
        warmup_ms = (time.monotonic() - warmup_started) * 1000
        result = {
            "config": {
                "method": "llmlingua2",
                "assistant_rate": 0.75,
                "tool_result_rate": 0.60,
                "production_trigger_tokens": 8000,
                "keep_hot": 6,
                "worker_threads": 32,
                "tokenizer": args.tokenizer,
            },
            "worker_startup_and_warmup_ms": round(warmup_ms, 1),
            "five_k_production_gate": {
                "action": gate_event.get("action"),
                "reason": gate_event.get("reason"),
                "compressible_cold_tokens_est": gate_event.get("compressible_cold_tokens"),
                "transform_ms": round(gate_ms, 1),
            },
            "raw_llmlingua2_case": _run_raw_case(compressor, tokenizer),
            "middleware_cases": [
                _run_case(
                    name="five_k_forced",
                    middleware=forced,
                    messages=messages_5k,
                    tokenizer=tokenizer,
                    shared_compressor=compressor,
                ),
                _run_case(
                    name="nine_k_production_threshold",
                    middleware=_middleware(trigger_tokens=8000),
                    messages=messages_9k,
                    tokenizer=tokenizer,
                    shared_compressor=compressor,
                ),
            ],
        }
    finally:
        compressor.close()

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
