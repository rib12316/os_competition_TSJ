"""通用 ReAct agent CLI（赛题验证方案第一条：CLI 跑工具调用）。

用法::

    python -m agent_mem.cli "帮我查今天的天气并写到文件" \\
        --engine-url http://localhost:8000/v1 --model Qwen3-0.6B

dev 阶段 engine-url 指向 P3 的 stub server 也能跑通（证明 agent loop）。
多轮 trace 打到 stderr，最终回答打到 stdout。每个 session 带 session_id（为 F5/F6 预留）。
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid

from openai import OpenAI

from agent_mem.agent import tools
from agent_mem.agent.react import run_react
from agent_mem.middleware import build_middlewares

DEFAULT_SYSTEM = (
    "你是一个能调用工具的助手。需要时调用工具获取信息或计算，拿到结果后用中文回答。"
)


def _print_trace(messages: list[dict], stream) -> None:
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                names = ", ".join(tc["function"]["name"] for tc in tcs)
                print(f"[trace] assistant → call tool: {names}", file=stream)
            else:
                print(f"[trace] assistant → {m.get('content') or ''}", file=stream)
        elif role == "tool":
            print(f"[trace] tool result → {m.get('content', '')[:120]}", file=stream)
        elif role == "user":
            print(f"[trace] user → {m.get('content', '')[:120]}", file=stream)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="agent-mem ReAct agent CLI")
    p.add_argument("prompt", help="用户指令")
    p.add_argument(
        "--engine-url",
        default=os.environ.get("AGENT_MEM_ENGINE_URL", "http://localhost:8000/v1"),
        help="引擎 OpenAI base_url（dev 指向 stub）",
    )
    p.add_argument("--model", default=os.environ.get("AGENT_MEM_MODEL", "Qwen3-0.6B"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "stub"))
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument(
        "--middleware",
        action="append",
        default=None,
        metavar="NAME",
        help="激活缝D 中间件（注册表里的名字，如 noop；可重复）。默认无。",
    )
    args = p.parse_args(argv)

    session_id = uuid.uuid4().hex[:12]
    stack = build_middlewares(args.middleware)
    print(f"[cli] session_id={session_id} engine={args.engine_url} model={args.model} "
          f"middleware={stack.names}", file=sys.stderr)

    client = OpenAI(base_url=args.engine_url, api_key=args.api_key)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    res = run_react(
        client, args.model, messages, tools.TOOLS, tools.execute_tool,
        max_steps=args.max_steps,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        middlewares=stack,
        session_id=session_id,
    )
    _print_trace(res.messages, sys.stderr)
    print(res.final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
