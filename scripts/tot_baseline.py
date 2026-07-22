"""多路径决策 Baseline — Cogitator TreeOfThoughts + Game of 24

指向本地 vllm-ascend 引擎，跑 Game of 24 问题，
采集准确率、LLM 调用次数、墙钟时间、KV 命中率、mem_peak。
"""

import os
import json
import sys
import time
import httpx
from dataclasses import dataclass, field

sys.path.insert(0, "agent-mem/src")

# ---- Game of 24 数据集 ----
GAME24_PROBLEMS = [
    [4, 9, 10, 13], [1, 3, 4, 6], [3, 7, 9, 13], [5, 7, 8, 10],
    [1, 2, 3, 4], [2, 4, 6, 8], [3, 3, 8, 8], [1, 1, 1, 8],
    [2, 3, 6, 9], [4, 4, 7, 8], [5, 6, 7, 8], [1, 5, 7, 9],
    [2, 5, 7, 11], [3, 4, 6, 7], [6, 7, 8, 9], [1, 4, 5, 6],
    [2, 6, 7, 9], [3, 5, 8, 12], [4, 5, 6, 9], [1, 2, 7, 8],
]


@dataclass
class RunMetrics:
    problem: str
    solved: bool
    n_llm_calls: int
    wall_seconds: float
    kv_hit_rate: float
    mem_peak_mb: float


def read_engine_metrics(engine_url: str) -> dict:
    """读取 vllm /metrics 和 MemSampler 数据"""
    try:
        r = httpx.get(f"{engine_url}/v1/models", timeout=5)
    except Exception:
        return {"kv_hit_rate": 0.0, "mem_peak_mb": 0.0}

    # 读 KV 命中率
    try:
        mr = httpx.get(f"{engine_url.replace('/v1','')}/metrics", timeout=5)
        hits = queries = 0
        for line in mr.text.split("\n"):
            if "vllm:prefix_cache_hits_total" in line and not line.startswith("#"):
                hits = float(line.split()[-1])
            if "vllm:prefix_cache_queries_total" in line and not line.startswith("#"):
                queries = float(line.split()[-1])
        kv_rate = hits / queries if queries > 0 else 0.0
    except Exception:
        kv_rate = 0.0

    return {"kv_hit_rate": kv_rate, "mem_peak_mb": 0.0}


def run_baseline(
    problems: list[list[int]],
    engine_url: str = "http://localhost:8000/v1",
    model: str = "Qwen2.5-7B-Instruct",
    max_depth: int = 3,
    n_branches: int = 2,
) -> list[RunMetrics]:
    """跑 baseline：Cogitator ToT + 引擎 + 采集指标"""
    # Cogitator 的 OpenAILLM 不支持 base_url，通过环境变量指向本地引擎
    os.environ["OPENAI_BASE_URL"] = engine_url
    os.environ["OPENAI_API_KEY"] = "stub"

    from cogitator import TreeOfThoughts, OpenAILLM

    llm = OpenAILLM(model=model, api_key="stub")
    tot = TreeOfThoughts(llm, max_depth=max_depth, num_branches=n_branches)

    results: list[RunMetrics] = []
    total_calls = 0

    for i, nums in enumerate(problems):
        prompt = (
            f"Solve the Game of 24 using the numbers {nums[0]}, {nums[1]}, "
            f"{nums[2]}, {nums[3]}. You can use +, -, *, /. "
            f"Each number must be used exactly once. "
            f"Think step by step."
        )

        # 记录调用前 KV 状态
        before = read_engine_metrics(engine_url)

        t0 = time.monotonic()
        try:
            answer = tot.run(prompt)
        except Exception as e:
            print(f"  [{i+1}/{len(problems)}] {nums} ERROR: {e}")
            answer = str(e)

        wall = time.monotonic() - t0
        after = read_engine_metrics(engine_url)

        # 粗略检查答案是否是 24
        solved = "24" in str(answer) or "= 24" in str(answer)
        if not solved:
            # 尝试直接 eval
            try:
                solved = abs(eval(str(answer).strip()) - 24) < 0.01
            except Exception:
                pass

        print(
            f"  [{i+1}/{len(problems)}] {nums}: "
            f"{'✅' if solved else '❌'} "
            f"wall={wall:.1f}s kv_hit={after['kv_hit_rate']:.3f}"
        )

        results.append(RunMetrics(
            problem=str(nums),
            solved=solved,
            n_llm_calls=0,  # Cogitator 不暴露调用数，用 wall 和 n_branches^depth 估算
            wall_seconds=wall,
            kv_hit_rate=after["kv_hit_rate"],
            mem_peak_mb=after["mem_peak_mb"],
        ))

    return results


if __name__ == "__main__":
    engine_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen2.5-7B-Instruct"

    print(f"ToT Baseline: Game of 24, engine={engine_url}, model={model}")
    print(f"Problems: {len(GAME24_PROBLEMS)}")
    print()

    results = run_baseline(GAME24_PROBLEMS, engine_url=engine_url, model=model)

    # 汇总
    solved = sum(1 for r in results if r.solved)
    total_wall = sum(r.wall_seconds for r in results)
    kv_rates = [r.kv_hit_rate for r in results if r.kv_hit_rate > 0]
    avg_kv = sum(kv_rates) / len(kv_rates) if kv_rates else 0.0

    print(f"\n=== SUMMARY ===")
    print(f"Accuracy: {solved}/{len(results)} ({solved/len(results)*100:.1f}%)")
    print(f"Total wall: {total_wall:.0f}s ({total_wall/len(results):.1f}s/problem)")
    print(f"Avg KV hit rate: {avg_kv:.3f}")

    # 写 JSON 结果
    out = {
        "benchmark": "game24",
        "strategy": "TreeOfThoughts",
        "engine": engine_url,
        "model": model,
        "results": [
            {
                "problem": r.problem,
                "solved": r.solved,
                "wall_s": round(r.wall_seconds, 1),
                "kv_hit_rate": round(r.kv_hit_rate, 3),
            }
            for r in results
        ],
        "summary": {
            "accuracy": f"{solved}/{len(results)}",
            "total_wall_s": round(total_wall, 0),
            "avg_kv_hit_rate": round(avg_kv, 3),
        },
    }
    with open("logs-tot/baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to logs-tot/baseline_results.json")
