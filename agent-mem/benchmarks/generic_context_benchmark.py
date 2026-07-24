#!/usr/bin/env python3
"""Measure generic compiled-policy + tool-aware LLMLingua on a long agent trace."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from agent_mem.middleware.base import MiddlewareContext
from agent_mem.middleware.compress import CompressMiddleware

DEFAULT_SOURCE = Path(__file__).with_name("generic_policy_source.md")
DEFAULT_ARTIFACT = Path(__file__).parents[1] / "configs/policies/knowledge_incident.json"
DEFAULT_TOKENIZER = "/data/os_competition_TSJ/models/Qwen2.5-7B-Instruct"
DEFAULT_WORKER = "/data/os_competition_TSJ/.venv-compress/bin/python"
DEFAULT_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


def count_prompt(tokenizer: Any, messages: list[dict], tools: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=True
    )
    token_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def build_tools() -> list[dict]:
    repeated = "Provide the incident identifier exactly as returned by the service."
    tools: list[dict] = []
    for name in (
        "search_knowledge", "get_incident", "list_related_incidents", "get_runbook",
        "get_member", "add_incident_note", "change_incident_severity", "close_incident",
    ):
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": (
                    "Use this tool only for the authenticated organization and explain the "
                    "result before any state-changing action."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "incident_id": {"type": "string", "description": repeated},
                        "query": {"type": "string", "description": "Search query or note text."},
                        "severity": {"type": "string", "enum": ["P0", "P1", "P2"]},
                    },
                    "required": ["incident_id"],
                },
            },
        })
    return tools


def build_trace(source: str, target_prompt_tokens: int, tokenizer: Any) -> list[dict]:
    narrative = (
        "The incident timeline includes a customer-visible symptom, affected component, "
        "observed evidence, attempted mitigation, dependency status, and a suggested next "
        "diagnostic step. This historical detail is useful for investigation but must not be "
        "confused with a confirmed state change. "
    )
    rounds = 45
    repeats = 1
    while repeats < 20:
        messages: list[dict] = [{"role": "system", "content": source}]
        for index in range(rounds):
            incident_id = f"INC-{index:05d}"
            call_id = f"search-{index:04d}"
            detail = narrative * repeats
            result = {
                "incident_id": incident_id,
                "severity": ["P0", "P1", "P2"][index % 3],
                "status": "open" if index % 4 else "monitoring",
                "owner": f"team-{index % 7}",
                "created_at": f"2026-07-{(index % 27) + 1:02d}T10:00:00Z",
                "summary": detail,
                "related_documents": [
                    {"title": f"Runbook chapter {index % 9}", "body": detail},
                    {"title": f"Postmortem note {index % 11}", "body": detail},
                ],
            }
            messages.extend([
                {
                    "role": "user",
                    "content": (
                        f"Investigate {incident_id} for the authenticated organization. "
                        "Do not modify it without explicit confirmation."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "I will search the evidence before making any recommendation.",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "search_knowledge",
                            "arguments": json.dumps({"incident_id": incident_id}),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "search_knowledge",
                    "content": json.dumps(result, separators=(",", ":")),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"The evidence for {incident_id} is recorded. The next step must use the "
                        "returned status and preserve the confirmation boundary."
                    ),
                },
            ])
        if count_prompt(tokenizer, messages, build_tools()) >= target_prompt_tokens:
            return messages
        repeats += 1
    return messages


def make_middleware(
    *,
    artifact_path: Path,
    trigger_tokens: int,
    optimize_static_prompt: bool,
) -> CompressMiddleware:
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
        optimize_static_prompt=optimize_static_prompt,
        system_prompt_mode="compiled" if optimize_static_prompt else "none",
        policy_artifact_path=str(artifact_path) if optimize_static_prompt else "",
        device="cpu",
        model_name=DEFAULT_MODEL,
        backend="subprocess",
        worker_venv=DEFAULT_WORKER,
        worker_pool_size=1,
        worker_threads=32,
    )


def audit_transformed_history(
    original: list[dict], transformed: list[dict]
) -> dict[str, Any]:
    searchable: list[str] = []
    for message in transformed:
        searchable.append(str(message.get("content") or ""))
        searchable.append(str(message.get("tool_call_id") or ""))
        for call in message.get("tool_calls") or []:
            searchable.append(str(call.get("id") or ""))
            searchable.append(str(call.get("function", {}).get("arguments") or ""))
    memory = "\n".join(searchable)
    incident_ids: list[str] = []
    call_ids: list[str] = []
    arguments: list[str] = []
    critical_values: list[str] = []
    for message in original[:-7]:
        for call in message.get("tool_calls") or []:
            call_ids.append(str(call.get("id") or ""))
            arguments.append(str(call.get("function", {}).get("arguments") or ""))
        if message.get("role") == "tool":
            content = str(message.get("content") or "")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            incident_ids.append(str(payload.get("incident_id") or ""))
            critical_values.extend([
                str(payload.get("severity") or ""),
                str(payload.get("status") or ""),
                str(payload.get("owner") or ""),
                str(payload.get("created_at") or ""),
            ])
    return {
        "cold_incident_ids": len(incident_ids),
        "all_incident_ids_preserved": all(value in memory for value in incident_ids),
        "all_call_ids_preserved": all(value in memory for value in call_ids),
        "all_arguments_preserved": all(value in memory for value in arguments),
        "all_severity_status_owner_time_preserved": all(
            value in memory for value in critical_values
        ),
        "hot_tail_unchanged": transformed[-7:] == original[-7:],
    }


def run_transform(
    middleware: CompressMiddleware,
    messages: list[dict],
    tools: list[dict],
    tokenizer: Any,
    session_id: str,
    ctx: MiddlewareContext | None = None,
) -> dict[str, Any]:
    if ctx is None:
        ctx = MiddlewareContext(session_id)
    ctx.bump_step()
    started = time.monotonic()
    output_messages, output_tools = middleware.transform_request(
        copy.deepcopy(messages), copy.deepcopy(tools), ctx
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    event = dict(ctx.scratch.get("compress:pending_event") or {})
    static = dict(ctx.scratch.get("compress:static_metrics") or {})
    original_tokens = count_prompt(tokenizer, messages, tools)
    transformed_tokens = count_prompt(tokenizer, output_messages, output_tools)
    return {
        "original_tokens": original_tokens,
        "transformed_tokens": transformed_tokens,
        "saved_tokens": original_tokens - transformed_tokens,
        "saved_percent": round((original_tokens - transformed_tokens) * 100 / original_tokens, 2),
        "message_count": len(messages),
        "tool_count": len(tools),
        "elapsed_ms": round(elapsed_ms, 1),
        "event": event,
        "static": static,
        "audit": audit_transformed_history(messages, output_messages),
        "output_messages": output_messages,
        "output_tools": output_tools,
        "context": ctx,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--target-tokens", type=int, default=24000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    tools = build_tools()
    messages = build_trace(source, args.target_tokens, tokenizer)
    canonical_tokens = count_prompt(tokenizer, messages, tools)

    static_mw = make_middleware(
        artifact_path=args.artifact, trigger_tokens=10**9, optimize_static_prompt=True
    )
    static = run_transform(static_mw, messages, tools, tokenizer, "generic-static")

    full_mw = make_middleware(
        artifact_path=args.artifact, trigger_tokens=8000, optimize_static_prompt=True
    )
    compressor = full_mw._get_compressor()
    warm_started = time.monotonic()
    compressor.compress_prompt(
        ["Warm up policy benchmark worker."],
        rate=0.60,
        force_tokens=["\n", "?", "."],
        force_reserve_digit=True,
        use_context_level_filter=False,
    )
    warmup_ms = (time.monotonic() - warm_started) * 1000
    try:
        full = run_transform(full_mw, messages, tools, tokenizer, "generic-full")
        reused = run_transform(
            full_mw,
            messages,
            tools,
            tokenizer,
            "generic-full",
            ctx=full["context"],
        )
    finally:
        compressor.close()

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()
                    if key not in {"output_messages", "output_tools", "context"}}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    result = {
        "benchmark": "generic-knowledge-incident-long-context",
        "target_tokens": args.target_tokens,
        "canonical_tokens": canonical_tokens,
        "message_count": len(messages),
        "tool_count": len(tools),
        "worker_warmup_ms": round(warmup_ms, 1),
        "static_only": clean(static),
        "static_plus_dynamic": clean(full),
        "reuse": clean(reused),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
