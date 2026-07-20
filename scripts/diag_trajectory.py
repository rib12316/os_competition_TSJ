"""诊断脚本：dump 一个 τ-bench 任务的 ground-truth vs agent 实际轨迹。

用法: python scripts/diag_trajectory.py [--task 0] [--max-steps 10] [--engine-url http://127.0.0.1:8000/v1]
只读诊断（跑 agent + 打印），不改任何东西。
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--engine-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--domain", default="retail")
    # user-sim 扩充（默认本地；给 --user-api-base 切外部）
    p.add_argument("--user-model", default=None)
    p.add_argument("--user-provider", default="openai")
    p.add_argument("--user-api-base", default=None)
    p.add_argument("--user-api-key", default=None)
    args = p.parse_args()

    from agent_mem.bench.tasks.tau_bench_adapter import _resolve_user_sim

    um, up, env_overrides = _resolve_user_sim(
        engine_url=args.engine_url, model=args.model, user_model=args.user_model,
        user_provider=args.user_provider, user_api_base=args.user_api_base,
        user_api_key=args.user_api_key, api_key="stub",
    )
    os.environ.update(env_overrides)  # 仅供 litellm user-sim

    from openai import OpenAI
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy

    from agent_mem.agent.tau_bench_agent import TauBenchAgent

    env = get_env(
        args.domain, UserStrategy.LLM, um, "test",
        user_provider=up, task_index=args.task,
    )
    t = env.tasks[args.task]
    print(f"=== GROUND TRUTH (task {args.task}) ===")
    print("instruction:", (t.instruction or "")[:300])
    print("expected actions:")
    for a in t.actions:
        print("  -", a.name, json.dumps(a.kwargs)[:200])
    print("required outputs:", t.outputs)
    print()

    agent = TauBenchAgent(OpenAI(base_url=args.engine_url, api_key="stub"), args.model)
    out = agent.solve(env, task_index=args.task, max_num_steps=args.max_steps)
    print(f"=== AGENT TRAJECTORY (reward={out.reward}, n_steps={out.n_steps}) ===")
    for i, m in enumerate(out.messages):
        r = m.get("role")
        if r == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    print(f'[{i}] CALL {fn.get("name")}({(fn.get("arguments") or "")[:160]})')
            else:
                print(f'[{i}] RESPOND: {(m.get("content") or "")[:200]}')
        elif r == "tool":
            print(f'[{i}]   -> obs: {(m.get("content") or "")[:160]}')
        elif r == "user":
            print(f'[{i}] USER: {(m.get("content") or "")[:160]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
