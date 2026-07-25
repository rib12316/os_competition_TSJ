"""Baseline 实验 — 全指标采集
================================
中分支 (C2, n=4) × HotpotQA validation 全集 (7405)
支持断点续跑，单任务异常隔离。

采集指标：
  系统:   峰值/平均 HBM, KV cache 占比, TTFT, TPOT, 吞吐
  任务:   成功率, 最终答案正确性
  MCTS:   平均分支数, 平均搜索深度, 总请求数, token 消耗

用法:
  python scripts/run_baseline.py                     # 默认：validation 全集, C2
  python scripts/run_baseline.py --num-tasks 100     # 只跑 100 条
  python scripts/run_baseline.py --resume            # 断点续跑
  python scripts/run_baseline.py --group C3          # 高分支
"""
import os, sys, time, json, math, re, threading, subprocess, urllib.request, argparse
from collections import defaultdict

os.environ["OPENAI_API_KEY"] = "stub"
import openai
openai.api_key = "stub"
openai.api_base = "http://localhost:8000/v1"

sys.path.insert(0, "/tmp/multi-path/scripts")
from offline_hotpot_env import OfflineHotpotEnv
from lats_hotpot_offline import (
    MCTSNode, mcts_search, _extract_best_answer,
    MCTS_ITERS, ROLLOUT_DEPTH, GROUPS,
)

MODEL = "Qwen2.5-7B-Instruct"
HBM_TOTAL_MB = 65536  # 910B2C HBM 总量

# ═══════════════════════════════════════════════════════════════════════
# 默认配置（可通过 CLI 覆盖）
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "group": "C2",
    "desc": "中分支 baseline",
    "n_branches": 4,
    "max_iters": 8,
    "num_tasks": 0,             # 0 = 全集
    "dataset_split": "train",   # HotpotQA distractor 只有 train 和 validation
    "output_dir": "/tmp/multi-path/logs-lats",
    "checkpoint_file": "baseline_checkpoint.json",
}

# ═══════════════════════════════════════════════════════════════════════
# vLLM 指标采集
# ═══════════════════════════════════════════════════════════════════════

def fetch_vllm_metrics():
    """抓取 vLLM Prometheus 指标（同名多标签的 counter 会累加）"""
    try:
        t = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5).read().decode()
    except Exception:
        return None

    m = {}
    for line in t.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        if "_created" in line or "_bucket" in line or "le=" in line:
            continue
        try:
            name_end = line.index("{") if "{" in line else line.index(" ")
        except ValueError:
            continue
        name = line[:name_end]
        val = float(line.split()[-1])
        if name in m:
            m[name] += val
        else:
            m[name] = val
    return m


def compute_vllm_delta(after, before):
    """计算前后 metrics 差值"""
    if not after or not before:
        return {}
    counters = [
        "vllm:request_success_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
    ]
    d = {}
    for k in counters:
        d[k] = after.get(k, 0) - before.get(k, 0)

    for metric in ["vllm:time_to_first_token_seconds",
                   "vllm:request_time_per_output_token_seconds",
                   "vllm:e2e_request_latency_seconds"]:
        s_after = after.get(f"{metric}_sum", 0)
        s_before = before.get(f"{metric}_sum", 0)
        c_after = after.get(f"{metric}_count", 0)
        c_before = before.get(f"{metric}_count", 0)
        d_sum = s_after - s_before
        d_cnt = c_after - c_before
        d[f"{metric}_avg_ms"] = (d_sum / max(d_cnt, 1)) * 1000
        d[f"{metric}_cnt"] = d_cnt

    hits = d.get("vllm:prefix_cache_hits_total", 0)
    queries = d.get("vllm:prefix_cache_queries_total", 0)
    d["kv_cache_hit_rate"] = hits / max(queries, 1)
    d["kv_cache_usage_perc"] = after.get("vllm:kv_cache_usage_perc", 0)
    return d


# ═══════════════════════════════════════════════════════════════════════
# 显存采样
# ═══════════════════════════════════════════════════════════════════════

class MemSampler:
    """持续采样 HBM 和 KV cache 使用率"""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.summary()

    def _loop(self):
        while not self._stop.is_set():
            ts = time.time()
            hbm_mb, hbm_pct = self._sample_hbm()
            kv_pct = self._sample_kv_cache()
            self.samples.append((ts, hbm_mb, hbm_pct, kv_pct))
            time.sleep(self.interval)

    def _sample_hbm(self):
        try:
            out = subprocess.check_output(
                ["npu-smi", "info", "-t", "usages", "-i", "10", "-c", "0"],
                text=True, stderr=subprocess.DEVNULL)
            for l in out.split("\n"):
                if "HBM Usage Rate" in l:
                    pct = float(l.split(":")[-1].strip().replace("%", ""))
                    return (pct / 100 * HBM_TOTAL_MB, pct)
        except Exception:
            pass
        return (0, 0)

    def _sample_kv_cache(self):
        try:
            m = fetch_vllm_metrics()
            if m:
                return m.get("vllm:kv_cache_usage_perc", 0) * 100
        except Exception:
            pass
        return 0

    def summary(self):
        if not self.samples:
            return {"hbm_peak_mb": 0, "hbm_avg_mb": 0, "kv_cache_avg_pct": 0, "kv_cache_peak_pct": 0}
        hbm_vals = [s[1] for s in self.samples]
        kv_vals = [s[3] for s in self.samples]
        return {
            "hbm_peak_mb": max(hbm_vals),
            "hbm_avg_mb": sum(hbm_vals) / len(hbm_vals),
            "hbm_min_mb": min(hbm_vals),
            "kv_cache_avg_pct": sum(kv_vals) / len(kv_vals) if kv_vals else 0,
            "kv_cache_peak_pct": max(kv_vals) if kv_vals else 0,
            "num_samples": len(self.samples),
        }


# ═══════════════════════════════════════════════════════════════════════
# MCTS 树统计
# ═══════════════════════════════════════════════════════════════════════

def analyze_mcts_tree(root):
    """从 MCTS 树提取详细统计"""
    all_nodes = []
    def _walk(n):
        all_nodes.append(n)
        for c in n.children:
            _walk(c)
    _walk(root)

    if not all_nodes:
        return {}

    total_nodes = len(all_nodes)
    max_depth = max(n.depth for n in all_nodes)
    terminal_nodes = [n for n in all_nodes if n.is_terminal]
    solved = [n for n in terminal_nodes if n.reward == 1]
    finish_nodes = [n for n in all_nodes if n.action.startswith("Finish[")]

    expanded = [n for n in all_nodes if n.children and not n.is_terminal]
    branches_per_expansion = [len(n.children) for n in expanded]

    best_ans, best_reward, best_depth = _extract_best_answer(root)

    return {
        "total_nodes": total_nodes,
        "max_depth": max_depth,
        "num_terminal": len(terminal_nodes),
        "num_solved": len(solved),
        "num_finishes": len(finish_nodes),
        "num_expansions": len(expanded),
        "avg_branches": sum(branches_per_expansion) / max(len(expanded), 1),
        "branches_dist": branches_per_expansion,
        "best_reward": best_reward,
        "best_depth": best_depth,
        "all_finishes": [
            {"answer": n.action[7:-1] if n.action.endswith("]") else n.action[7:],
             "reward": n.reward, "depth": n.depth}
            for n in finish_nodes
        ],
    }


# ═══════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════

def run_baseline(cfg_override=None):
    cfg = dict(DEFAULT_CFG)
    if cfg_override:
        cfg.update(cfg_override)

    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, cfg["checkpoint_file"])

    # ── 加载数据集 ──
    print(f"Loading HotpotQA ({cfg['dataset_split']})...")
    env = OfflineHotpotEnv(split=cfg["dataset_split"],
                           max_examples=cfg["num_tasks"] if cfg["num_tasks"] > 0 else 100000)
    n_total = min(len(env.data), cfg["num_tasks"] if cfg["num_tasks"] > 0 else len(env.data))

    # ── 断点恢复 ──
    completed_ids = set()
    per_task_results = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        per_task_results = ckpt.get("results", [])
        completed_ids = {r["task_id"] for r in per_task_results}
        print(f"Resuming from checkpoint: {len(completed_ids)}/{n_total} already done")

    pending_ids = [i for i in range(n_total) if i not in completed_ids]

    print(f"Config: {cfg['desc']} | branches={cfg['n_branches']} | iters={cfg['max_iters']}")
    print(f"Dataset: {cfg['dataset_split']} | Total: {n_total} | Pending: {len(pending_ids)}")
    print(f"{'='*80}")

    if not pending_ids:
        print("All tasks completed!")
        return _finalize(per_task_results, cfg, output_dir)

    # ── 开始采样 ──
    sampler = MemSampler(interval=2.0)  # 2s 间隔，减少开销
    sampler.start()
    before_metrics = fetch_vllm_metrics()

    # ── 逐任务运行 ──
    t_start = time.monotonic()
    solved_count = sum(1 for r in per_task_results if r.get("success"))

    for i, idx in enumerate(pending_ids):
        try:
            obs = env.reset(idx=idx)
        except Exception as e:
            print(f"  [SKIP] Task {idx}: env reset failed: {e}")
            continue

        question = env._question
        ground_truth = env._answer
        t_task = time.monotonic()

        try:
            node = mcts_search(
                env, n_branches=cfg["n_branches"],
                max_iters=cfg["max_iters"], verbose=False
            )
        except Exception as e:
            # 单任务异常隔离：记录失败但不中断
            wall_task = time.monotonic() - t_task
            task_result = {
                "task_id": idx, "question": question, "ground_truth": ground_truth,
                "final_answer": "", "success": False, "best_reward": 0,
                "best_depth": 0, "wall_time_s": round(wall_task, 2),
                "tree": {}, "error": str(e)[:200],
            }
            per_task_results.append(task_result)
            _save_checkpoint(checkpoint_path, cfg, per_task_results)
            print(f"  [ERR] Task {idx}: {str(e)[:100]}")
            continue

        wall_task = time.monotonic() - t_task

        # 找到 root
        root = node
        while root.parent:
            root = root.parent

        # 提取最佳答案
        best_ans, best_reward, best_depth = _extract_best_answer(root)
        success = best_reward == 1

        # 树统计
        tree_stats = analyze_mcts_tree(root)

        if success:
            solved_count += 1

        task_result = {
            "task_id": idx,
            "question": question,
            "ground_truth": ground_truth,
            "final_answer": best_ans[:200] if best_ans else "",
            "success": success,
            "best_reward": best_reward,
            "best_depth": best_depth,
            "wall_time_s": round(wall_task, 2),
            "tree": tree_stats,
        }
        per_task_results.append(task_result)

        # 每个任务都保存 checkpoint（关键：确保断点续跑的可靠性）
        _save_checkpoint(checkpoint_path, cfg, per_task_results)

        # 进度打印
        done = len(completed_ids) + i + 1
        elapsed = time.monotonic() - t_start
        avg_wall = elapsed / (i + 1)
        eta = avg_wall * (len(pending_ids) - i - 1)
        print(f"[{done:5d}/{n_total}] "
              f"solved={solved_count}/{done} ({solved_count/done*100:.1f}%) "
              f"avg={avg_wall:.1f}s/task ETA={eta/60:.0f}min "
              f"last={wall_task:.1f}s {'✓' if success else '✗'} "
              f"{best_ans[:50] if best_ans else '?'}")

    # ── 停止采样 ──
    wall_total = time.monotonic() - t_start
    after_metrics = fetch_vllm_metrics()
    mem_summary = sampler.stop()

    return _finalize(per_task_results, cfg, output_dir, wall_total,
                     before_metrics, after_metrics, mem_summary)


def _save_checkpoint(path, cfg, results):
    """保存断点文件（原子写入）"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _finalize(per_task_results, cfg, output_dir, wall_total=0,
              before_metrics=None, after_metrics=None, mem_summary=None):
    """汇总并保存最终结果"""
    n_tasks = len(per_task_results)
    solved_count = sum(1 for r in per_task_results if r.get("success"))

    # ── 汇总指标 ──
    total_nodes_all = sum(r.get("tree", {}).get("total_nodes", 0) for r in per_task_results)
    total_expansions = sum(r.get("tree", {}).get("num_expansions", 0) for r in per_task_results)
    all_branches = []
    for r in per_task_results:
        all_branches.extend(r.get("tree", {}).get("branches_dist", []))

    wall_tasks = [r["wall_time_s"] for r in per_task_results]
    depths = [r.get("best_depth", 0) for r in per_task_results if r.get("best_depth", 0) > 0]

    solved_tasks = [r for r in per_task_results if r.get("success")]
    failed_tasks = [r for r in per_task_results if not r.get("success")]

    summary = {
        "config": cfg,
        "n_tasks": n_tasks,
        "solved": solved_count,
        "success_rate": solved_count / n_tasks if n_tasks else 0,
        "wall_total_s": round(wall_total, 1),
        "wall_total_h": round(wall_total / 3600, 2),
        "throughput_tasks_per_min": round(n_tasks / wall_total * 60, 2) if wall_total > 0 else 0,

        # ── MCTS 树 ──
        "avg_nodes_per_task": round(total_nodes_all / n_tasks, 1) if n_tasks else 0,
        "avg_branches_per_expansion": round(sum(all_branches) / max(len(all_branches), 1), 2),
        "median_branches": round(sorted(all_branches)[len(all_branches)//2], 1) if all_branches else 0,
        "avg_max_depth": round(sum(r.get("tree", {}).get("max_depth", 0) for r in per_task_results) / n_tasks, 1) if n_tasks else 0,
        "avg_best_depth": round(sum(depths) / len(depths), 1) if depths else 0,
        "avg_expansions_per_task": round(total_expansions / n_tasks, 1) if n_tasks else 0,

        # ── 耗时分布 ──
        "wall_time_avg_s": round(sum(wall_tasks) / len(wall_tasks), 2) if wall_tasks else 0,
        "wall_time_median_s": round(sorted(wall_tasks)[len(wall_tasks)//2], 2) if wall_tasks else 0,
        "wall_time_max_s": round(max(wall_tasks), 2) if wall_tasks else 0,
        "wall_time_min_s": round(min(wall_tasks), 2) if wall_tasks else 0,
    }

    # ── 系统指标 ──
    if mem_summary:
        summary.update({
            "hbm_peak_mb": round(mem_summary["hbm_peak_mb"], 0),
            "hbm_avg_mb": round(mem_summary["hbm_avg_mb"], 0),
            "hbm_min_mb": round(mem_summary.get("hbm_min_mb", 0), 0),
            "kv_cache_avg_pct": round(mem_summary["kv_cache_avg_pct"], 1),
            "kv_cache_peak_pct": round(mem_summary["kv_cache_peak_pct"], 1),
        })

    if before_metrics and after_metrics:
        vllm_delta = compute_vllm_delta(after_metrics, before_metrics)
        summary.update({
            "total_requests": int(vllm_delta.get("vllm:request_success_total", 0)),
            "total_prompt_tokens": int(vllm_delta.get("vllm:prompt_tokens_total", 0)),
            "total_gen_tokens": int(vllm_delta.get("vllm:generation_tokens_total", 0)),
            "total_tokens": int(vllm_delta.get("vllm:prompt_tokens_total", 0) +
                              vllm_delta.get("vllm:generation_tokens_total", 0)),
            "avg_ttft_ms": round(vllm_delta.get("vllm:time_to_first_token_seconds_avg_ms", 0), 1),
            "avg_tpot_ms": round(vllm_delta.get("vllm:request_time_per_output_token_seconds_avg_ms", 0), 1),
            "avg_e2e_ms": round(vllm_delta.get("vllm:e2e_request_latency_seconds_avg_ms", 0), 1),
            "kv_cache_hit_rate": round(vllm_delta.get("kv_cache_hit_rate", 0), 4),
            "tokens_per_second": round(
                vllm_delta.get("vllm:generation_tokens_total", 0) / max(wall_total, 1), 1
            ),
        })

    # ── 失败分析 ──
    if failed_tasks:
        # 失败原因分类
        no_finish = sum(1 for r in failed_tasks if r.get("tree", {}).get("num_finishes", 0) == 0)
        wrong_answer = sum(1 for r in failed_tasks if r.get("tree", {}).get("num_finishes", 0) > 0)
        summary["failure_analysis"] = {
            "no_finish_attempted": no_finish,
            "finish_but_wrong": wrong_answer,
        }

    # ── 保存 ──
    output = {
        "config": cfg,
        "summary": summary,
        "per_task": per_task_results,
    }

    result_path = os.path.join(output_dir, "baseline_full.json")
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 也保存一份 summary only（方便快速查看）
    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── 打印报告 ──
    _print_report(summary, result_path)

    return output


def _print_report(s, path):
    print(f"\n{'='*80}")
    print(f"  BASELINE RESULTS — {s.get('config', {}).get('desc', '?')}")
    print(f"{'='*80}")
    print(f"  Tasks:            {s['n_tasks']}")
    print(f"  Success rate:     {s['success_rate']:.1%} ({s['solved']}/{s['n_tasks']})")
    print(f"  Wall time:        {s.get('wall_total_s', 0):.0f}s ({s.get('wall_total_h', 0):.1f}h)")
    print(f"  Throughput:       {s.get('throughput_tasks_per_min', 0):.1f} tasks/min")
    print()
    print(f"  ── MCTS Tree ──")
    print(f"  Avg nodes/task:   {s.get('avg_nodes_per_task', 0):.0f}")
    print(f"  Avg branches:     {s.get('avg_branches_per_expansion', 0):.1f}")
    print(f"  Avg max depth:    {s.get('avg_max_depth', 0):.1f}")
    print(f"  Avg best depth:   {s.get('avg_best_depth', 0):.1f}")
    print(f"  Avg expansions:   {s.get('avg_expansions_per_task', 0):.1f}")
    print()
    print(f"  ── Wall Time ──")
    print(f"  Avg: {s.get('wall_time_avg_s', 0):.1f}s  Median: {s.get('wall_time_median_s', 0):.1f}s  "
          f"Min: {s.get('wall_time_min_s', 0):.1f}s  Max: {s.get('wall_time_max_s', 0):.1f}s")
    print()
    print(f"  ── Memory ──")
    print(f"  HBM peak:         {s.get('hbm_peak_mb', 0):.0f} MB  "
          f"avg: {s.get('hbm_avg_mb', 0):.0f} MB  min: {s.get('hbm_min_mb', 0):.0f} MB")
    print(f"  KV cache avg:     {s.get('kv_cache_avg_pct', 0):.1f}%  "
          f"peak: {s.get('kv_cache_peak_pct', 0):.1f}%")
    print()
    print(f"  ── LLM Performance ──")
    print(f"  Total requests:   {s.get('total_requests', 0)}")
    print(f"  Total tokens:     {s.get('total_tokens', 0)} "
          f"(P:{s.get('total_prompt_tokens', 0)} G:{s.get('total_gen_tokens', 0)})")
    print(f"  Tokens/sec:       {s.get('tokens_per_second', 0):.1f}")
    print(f"  Avg TTFT:         {s.get('avg_ttft_ms', 0):.0f} ms")
    print(f"  Avg TPOT:         {s.get('avg_tpot_ms', 0):.1f} ms")
    print(f"  Avg E2E:          {s.get('avg_e2e_ms', 0):.0f} ms")
    print(f"  KV hit rate:      {s.get('kv_cache_hit_rate', 0):.2%}")
    if "failure_analysis" in s:
        fa = s["failure_analysis"]
        print(f"\n  ── Failure Analysis ──")
        print(f"  No Finish:        {fa['no_finish_attempted']}")
        print(f"  Finish but wrong: {fa['finish_but_wrong']}")
    print(f"\n  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="LATS MCTS Baseline Experiment")
    ap.add_argument("--num-tasks", type=int, default=0,
                    help="Number of tasks (0=all in split)")
    ap.add_argument("--group", type=str, default="C2",
                    help="Experiment group: A, B, C1, C2, C3")
    ap.add_argument("--split", type=str, default="train",
                    help="Dataset split: train, validation")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from checkpoint")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore checkpoint, start fresh")
    args = ap.parse_args()

    # 获取 group 配置
    group_cfg = GROUPS.get(args.group, GROUPS["C2"])

    cfg = {
        **DEFAULT_CFG,
        "group": args.group,
        "desc": group_cfg["desc"],
        "n_branches": group_cfg["n"],
        "num_tasks": args.num_tasks,
        "dataset_split": args.split,
    }

    output_dir = cfg["output_dir"]
    checkpoint_path = os.path.join(output_dir, cfg["checkpoint_file"])

    if args.fresh and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Removed existing checkpoint, starting fresh.")

    run_baseline(cfg)


if __name__ == "__main__":
    main()
