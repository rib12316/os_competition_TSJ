#!/usr/bin/env python
"""Small real-data F3 quality probe using LongBench 2WikiMultihopQA."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

from agent_mem.agent.react import run_react
from agent_mem.middleware import MiddlewareStack
from agent_mem.middleware.artifact_store import SQLiteArtifactStore
from agent_mem.middleware.compress import CompressMiddleware
from agent_mem.middleware.lazyload import FETCH_TOOL_NAME, LazyLoadMiddleware
from agent_mem.token_counting import count_text_tokens, get_tokenizer

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_documents",
        "description": "Return all candidate documents for the current question as JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The original question."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class _RecordingCompletions:
    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.prompt_tokens: list[int] = []
        self.wall_ms: list[float] = []
        self.tool_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        started = time.monotonic()
        response = self._delegate.create(**kwargs)
        self.wall_ms.append((time.monotonic() - started) * 1000)
        prompt_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        if isinstance(prompt_tokens, int):
            self.prompt_tokens.append(prompt_tokens)
        message = response.choices[0].message
        for call in getattr(message, "tool_calls", None) or []:
            raw_arguments = call.function.arguments or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError):
                arguments = raw_arguments
            self.tool_calls.append({"name": call.function.name, "arguments": arguments})
        return response


class _RecordingClient:
    def __init__(self, delegate: Any):
        self.chat = type("RecordingChat", (), {})()
        self.chat.completions = _RecordingCompletions(delegate.chat.completions)


def _load_examples(path: Path, *, start: int, limit: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read("data/2wikimqa.jsonl").splitlines()
    return [json.loads(line) for line in lines[start : start + limit]]


def _context_payload(context: str) -> tuple[str, int]:
    parts = re.split(r"(?m)^Passage \d+:\n", context)[1:]
    documents = []
    for part in parts:
        title, _, body = part.partition("\n")
        documents.append({"title": title.strip(), "text": body.strip()})
    return (
        json.dumps({"count": len(documents), "documents": documents}, separators=(",", ":")),
        len(documents),
    )


def _stack(variant: str, *, store_path: Path, tokenizer_model: str) -> MiddlewareStack:
    if variant == "baseline":
        return MiddlewareStack()
    lazy = LazyLoadMiddleware(
        artifact_store=SQLiteArtifactStore(str(store_path)),
        tokenizer_model=tokenizer_model,
        externalize_trigger_tokens=4000,
        max_reference_tokens=512,
        fetch_max_tokens=768,
    )
    middlewares: list[Any] = [lazy]
    if variant == "f2_f3":
        middlewares.append(CompressMiddleware(
            method="llmlingua2",
            trigger_tokens=8000,
            recompress_delta_tokens=4000,
            keep_hot=6,
            tool_aware=True,
            assistant_rate=0.75,
            tool_result_rate=0.60,
            hot_tool_trigger_tokens=1000,
            optimize_static_prompt=False,
            tokenizer_model=tokenizer_model,
        ))
    return MiddlewareStack(middlewares)


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _answer_correct(text: str, answers: list[str]) -> bool:
    normalized = _normalize(text)
    return any(
        normalized == _normalize(answer) or _normalize(answer) in normalized
        for answer in answers
    )


def _run_example(
    *,
    client: Any,
    model: str,
    tokenizer_model: str,
    tokenizer: Any,
    example: dict[str, Any],
    example_index: int,
    variant: str,
    store_path: Path,
    max_steps: int,
) -> dict[str, Any]:
    payload, document_count = _context_payload(str(example["context"]))
    payload_tokens = count_text_tokens(tokenizer, payload)
    stack = _stack(variant, store_path=store_path, tokenizer_model=tokenizer_model)
    recorder = _RecordingClient(client)
    business_calls = 0

    def execute_tool(name: str, args: dict[str, Any]) -> str:
        nonlocal business_calls
        if name != "retrieve_documents":
            raise ValueError(f"unexpected business tool: {name}")
        business_calls += 1
        return payload

    result = run_react(
        recorder,
        model,
        [
            {
                "role": "system",
                "content": (
                    "Answer this two-hop question using retrieve_documents. Use evidence from two "
                    "relevant documents: first read the title that most directly matches the "
                    "subject explicitly named in the question, then follow the discovered person, "
                    "place, work, or relation to a second title. Do not return the intermediate "
                    "entity as the answer. When documents are externalized, search the /documents "
                    "array with fetch_tool_result and match_field=title. Copy titles from the "
                    "reference summary; never invent one. Use match_mode=iexact or icontains for "
                    "capitalization and parenthetical differences. Return only the short final "
                    "answer; do not guess."
                ),
            },
            {"role": "user", "content": str(example["input"])},
        ],
        [RETRIEVE_TOOL],
        execute_tool,
        max_steps=max_steps,
        temperature=0.0,
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        middlewares=stack,
        session_id=f"longbench-{variant}-{example_index}-{time.time_ns()}",
    )
    calls = recorder.chat.completions.tool_calls
    fetch_calls = sum(call["name"] == FETCH_TOOL_NAME for call in calls)
    success = _answer_correct(result.final_text, list(example["answers"]))
    row = {
        "example_index": example_index,
        "example_id": example.get("_id"),
        "variant": variant,
        "question": example["input"],
        "gold_answers": example["answers"],
        "answer": result.final_text,
        "success": success,
        "failure_reason": None if success else (
            "max_steps" if result.truncated else
            "fetch_not_called" if variant != "baseline" and fetch_calls == 0 else
            "wrong_answer"
        ),
        "document_count": document_count,
        "payload_tokens": payload_tokens,
        "model_steps": result.n_steps,
        "business_calls": business_calls,
        "fetch_calls": fetch_calls,
        "tool_calls": calls,
        "prompt_tokens_by_step": recorder.chat.completions.prompt_tokens,
        "cumulative_prompt_tokens": sum(recorder.chat.completions.prompt_tokens),
        "model_wall_ms_by_step": [
            round(value, 3) for value in recorder.chat.completions.wall_ms
        ],
        "total_wall_ms": round(sum(recorder.chat.completions.wall_ms), 3),
    }
    for middleware in stack.middlewares:
        close = getattr(middleware, "close", None)
        if close is not None:
            close()
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        out[variant] = {
            "passed": sum(bool(row["success"]) for row in selected),
            "total": len(selected),
            "accuracy": round(
                sum(bool(row["success"]) for row in selected) / max(1, len(selected)), 4
            ),
            "fetch_calls": sum(int(row["fetch_calls"]) for row in selected),
            "cumulative_prompt_tokens": sum(
                int(row["cumulative_prompt_tokens"]) for row in selected
            ),
            "median_wall_ms": round(
                statistics.median(float(row["total_wall_ms"]) for row in selected), 3
            ),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import OpenAI

    tokenizer = get_tokenizer(args.tokenizer_model)
    examples = _load_examples(args.data_zip, start=args.start, limit=args.limit)
    client = OpenAI(base_url=args.engine_url, api_key="stub", timeout=args.timeout)
    rows = []
    with tempfile.TemporaryDirectory(prefix="f3-longbench-") as tmpdir:
        for variant in args.variants:
            for offset, example in enumerate(examples):
                example_index = args.start + offset
                rows.append(_run_example(
                    client=client,
                    model=args.served_model,
                    tokenizer_model=args.tokenizer_model,
                    tokenizer=tokenizer,
                    example=example,
                    example_index=example_index,
                    variant=variant,
                    store_path=Path(tmpdir) / f"{variant}-{example_index}.sqlite3",
                    max_steps=args.max_steps,
                ))
    return {
        "benchmark": "f3-longbench-2wikimqa",
        "source": "THUDM/LongBench data/2wikimqa.jsonl",
        "model": args.served_model,
        "config": {
            "start": args.start,
            "limit": args.limit,
            "variants": args.variants,
            "max_steps": args.max_steps,
            "temperature": 0.0,
        },
        "summary": _summary(rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--engine-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "f3", "f2_f3"],
        default=["baseline", "f3", "f2_f3"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
