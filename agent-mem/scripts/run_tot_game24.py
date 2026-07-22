#!/usr/bin/env python
"""Princeton ToT Game of 24 —— 适配 vllm 引擎。

用法（需先起引擎）:
    OPENAI_API_KEY=stub OPENAI_API_BASE=http://127.0.0.1:8000/v1 \
    .venv/bin/python scripts/run_tot_game24.py --puzzles 5 --model Qwen2.5-7B-Instruct
"""
import argparse
import os
import time

# 指向本地 vllm 引擎
os.environ.setdefault("OPENAI_API_KEY", "stub")
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")


def main():
    p = argparse.ArgumentParser(description="Princeton ToT Game of 24 on vllm")
    p.add_argument("--puzzles", type=int, default=3, help="跑几道题")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--n-generate", type=int, default=3, help="每步候选数 (k)")
    p.add_argument("--n-evaluate", type=int, default=3, help="评估采样数")
    p.add_argument("--n-select", type=int, default=2, help="beam 宽度 (b)")
    args = p.parse_args()

    from tot.tasks.game24 import Game24Task
    from tot.methods.bfs import solve

    task = Game24Task()
    n_puzzles = min(args.puzzles, len(task))

    # argparse.Namespace — bfs.solve 用这个传参
    bfs_args = argparse.Namespace(
        backend=args.model,
        temperature=args.temperature,
        method_generate="propose",
        method_evaluate="value",
        method_select="greedy",
        n_generate_sample=args.n_generate,
        n_evaluate_sample=args.n_evaluate,
        n_select_sample=args.n_select,
        prompt_sample="standard",
    )

    print(f"ToT Game of 24: {n_puzzles} puzzles, "
          f"model={args.model}, k={args.n_generate}, b={args.n_select}")
    print(f"Engine: {os.environ['OPENAI_API_BASE']}")
    print("=" * 60)

    solved = 0
    total_calls = 0
    t0 = time.monotonic()

    for i in range(n_puzzles):
        puzzle = task.data[i]
        print(f"\n[Puzzle {i}] {puzzle}")
        try:
            ys, info = solve(bfs_args, task, i, to_print=False)
            # 检查是否 solve 了一题
            for y in ys:
                if '24' in y or task.test_output(i, y).get('r', 0) == 1:
                    solved += 1
                    print(f"  -> SOLVED: {y.strip()[:120]}")
                    break
            else:
                print(f"  -> 未解出, best candidates: {[y[:60] for y in ys[:3]]}")
            n_steps = len(info.get("steps", []))
            total_calls += n_steps * (args.n_generate + args.n_evaluate)
        except Exception as e:
            print(f"  -> ERROR: {e}")

    wall = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"Results: {solved}/{n_puzzles} solved in {wall:.0f}s")
    print(f"Estimated LLM calls: ~{total_calls}")


if __name__ == "__main__":
    main()
