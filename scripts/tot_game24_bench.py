"""多路径决策 Baseline — Princeton ToT + Game of 24 + 本地 vllm 引擎

不改 third_party 代码。通过环境变量指向我们的引擎。
"""
import os
import sys
import json
import time
from argparse import Namespace

# 指向我们的引擎
os.environ["OPENAI_API_KEY"] = "stub"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"

# import Princeton ToT（不改）
sys.path.insert(0, "third_party/tree-of-thought-llm/tree-of-thought-llm/src")
from tot.tasks.game24 import Game24Task
from tot.methods.bfs import solve
from tot.models import gpt_usage

# ---- 实验配置 ----
args = Namespace(
    backend="Qwen2.5-7B-Instruct",
    temperature=0.7,
    method_generate="propose",
    method_evaluate="value",
    method_select="greedy",
    n_generate_sample=1,
    n_evaluate_sample=1,
    n_select_sample=1,
    prompt_sample="cot",
)

task = Game24Task()
total = min(len(task), 20)
n_correct = 0
results = []

print(f"ToT Game24 Baseline: {total} problems, backend={args.backend}")
print()

t0_total = time.monotonic()
for i in range(total):
    x = task.get_input(i)
    t0 = time.monotonic()
    ys, info = solve(args, task, i, to_print=False)
    wall = time.monotonic() - t0

    # 检查是否正确
    test_results = [task.test_output(i, y) for y in ys]
    solved = any(r["r"] == 1 for r in test_results)

    if solved:
        n_correct += 1

    print(f"  [{i+1}/{total}] {x}: {'✅' if solved else '❌'} ({wall:.1f}s)")

    results.append({
        "idx": i,
        "problem": x,
        "solved": solved,
        "wall_s": round(wall, 1),
        "candidates": len(ys),
    })

total_wall = time.monotonic() - t0_total

# 汇总
usage = gpt_usage(args.backend)
print(f"\n=== SUMMARY ===")
print(f"Accuracy: {n_correct}/{total} ({n_correct/total*100:.1f}%)")
print(f"Total wall: {total_wall:.0f}s ({total_wall/total:.1f}s/problem)")
print(f"LLM calls: prompt_tokens={usage['prompt_tokens']} completion_tokens={usage['completion_tokens']}")

# 写结果
os.makedirs("logs-tot", exist_ok=True)
out = {
    "benchmark": "game24",
    "backend": args.backend,
    "strategy": f"{args.method_generate}+{args.method_evaluate}+{args.method_select}",
    "temperature": args.temperature,
    "results": results,
    "summary": {
        "accuracy": f"{n_correct}/{total}",
        "total_wall_s": round(total_wall, 0),
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
    },
}
with open("logs-tot/tot_baseline_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nResults saved to logs-tot/tot_baseline_results.json")
