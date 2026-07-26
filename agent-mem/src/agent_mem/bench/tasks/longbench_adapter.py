"""LongBench suite adapter — 长上下文 QA（2WikiMultihopQA）经 F2/F3 middleware。

从 ``benchmarks/f3_longbench_quality_benchmark.py`` 移植：把"单 JSONL example = 一个任务"
接入统一 harness 的 ``list_tasks``/``run_task`` 契约（与 tau_bench_adapter 对偶）。
context 经 ``retrieve_documents`` 业务工具注入 → :class:`LazyLoadMiddleware` 外化、
:class:`CompressMiddleware` 压缩；judge 用 exact/substring（不同于 tau 的 env reward gate）。

**惰性 import**：openai / run_react 的重依赖只在 :func:`run_task` 真跑时加载——
import 本模块是廉价的（与 tau_bench_adapter 纪律一致），不污染 pytest 收集。
"""

from __future__ import annotations

import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.bench.tasks.types import TaskInfo
from agent_mem.config import AppConfig
from agent_mem.middleware import MiddlewareStack

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

TWO_WIKI_MQA_SYSTEM_PROMPT = (
    "Answer this two-hop question using retrieve_documents. Use evidence from two "
    "relevant documents: first read the title that most directly matches the "
    "subject explicitly named in the question, then follow the discovered person, "
    "place, work, or relation to a second title. Do not return the intermediate "
    "entity as the answer. When documents are externalized, search the /documents "
    "array with fetch_tool_result and match_field=title. Copy titles from the "
    "reference summary; never invent one. Use match_mode=iexact or icontains for "
    "capitalization and parenthetical differences. Return only the short final "
    "answer; do not guess."
)

_DATASET_PATH_IN_ZIP = "data/2wikimqa.jsonl"
_DEFAULT_DOMAIN = "2wikimqa"


# ---- LLM 调用计量垫片（捕获 prompt_tokens / wall_ms / tool_calls）----


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


# ---- 数据 / judge 纯函数（从 f3 移植）----


def _load_examples(path: Path, *, start: int, limit: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        lines = archive.read(_DATASET_PATH_IN_ZIP).splitlines()
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


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.casefold()))


def _answer_correct(text: str, answers: list[str]) -> bool:
    normalized = _normalize(text)
    return any(
        normalized == _normalize(answer) or _normalize(answer) in normalized
        for answer in answers
    )


# ---- suite adapter 契约：list_tasks / run_task ----


def list_tasks(cfg: AppConfig) -> list[TaskInfo]:
    """枚举 LongBench 任务：每条 JSONL = 一个 TaskInfo（payload=整条 example）。"""
    data_zip = cfg.benchmark.data_zip
    if not data_zip:
        raise ValueError(
            "longbench 需要 benchmark.data_zip（THUDM/LongBench zip 路径），"
            "或 CLI --data-zip"
        )
    opts = cfg.benchmark.options or {}
    start = int(opts.get("start", 0))
    limit = int(opts.get("limit", 3))
    examples = _load_examples(Path(data_zip), start=start, limit=limit)
    return [
        TaskInfo(
            task_id=start + i, suite="longbench",
            domain=_DEFAULT_DOMAIN, split="test", payload=ex,
        )
        for i, ex in enumerate(examples)
    ]


def run_task(
    task: TaskInfo,
    *,
    engine_url: str,
    model: str,
    api_key: str = "stub",
    client: Any = None,
    middlewares: list | None = None,
    max_steps: int = 8,
    max_tokens: int = 256,
    system_prompt: str | None = None,
) -> TaskRunResult:
    """跑一个 LongBench 任务：context 经工具注入 → middleware → judge。

    ``client`` 给定时直接用（测试注入 fake）；否则按 ``engine_url`` 建 OpenAI。
    ``middlewares`` 由调用方（runner 经 ``middlewares_from_config``）构建——
    本 adapter **不**自带硬编码 worker_venv。
    """
    from agent_mem.agent.react import run_react

    example = task.payload or {}
    payload, _doc_count = _context_payload(str(example.get("context", "")))
    stack = MiddlewareStack(list(middlewares)) if middlewares else MiddlewareStack()
    if client is None:
        from openai import OpenAI

        client = OpenAI(base_url=engine_url, api_key=api_key)
    recorder = _RecordingClient(client)

    def execute_tool(name: str, args: dict[str, Any]) -> str:
        if name != "retrieve_documents":
            raise ValueError(f"unexpected business tool: {name}")
        return payload

    started = time.monotonic()
    try:
        result = run_react(
            recorder,
            model,
            [
                {"role": "system", "content": system_prompt or TWO_WIKI_MQA_SYSTEM_PROMPT},
                {"role": "user", "content": str(example.get("input", ""))},
            ],
            [RETRIEVE_TOOL],
            execute_tool,
            max_steps=max_steps,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            middlewares=stack,
            session_id=f"longbench-{task.task_id}-{time.time_ns()}",
        )
    except Exception as exc:  # noqa: BLE001 — 单任务异常返回失败，不杀 run
        for mw in stack.middlewares:
            close = getattr(mw, "close", None)
            if close is not None:
                close()
        return TaskRunResult(
            task_id=task.task_id, reward=0.0, success=False,
            latency_ms=(time.monotonic() - started) * 1000, n_steps=0, error=repr(exc),
        )

    success = _answer_correct(result.final_text, list(example.get("answers", [])))
    for mw in stack.middlewares:
        close = getattr(mw, "close", None)
        if close is not None:
            close()
    return TaskRunResult(
        task_id=task.task_id,
        reward=1.0 if success else 0.0,
        success=success,
        latency_ms=sum(recorder.chat.completions.wall_ms),
        n_steps=result.n_steps,
        error=None,
    )
