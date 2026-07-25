"""LATS MCTS + 离线 HotpotQA 多路径决策实验"""
import os, sys, time, json, math, subprocess, urllib.request, threading
from argparse import Namespace
import random

os.environ["OPENAI_API_KEY"] = "stub"
import openai
openai.api_key = "stub"
openai.api_base = "http://localhost:8000/v1"

REPO = "/data/os_competition_TSJ"
sys.path.insert(0, f"{REPO}/third_party/LanguageAgentTreeSearch/programming")
sys.path.insert(0, "/tmp/multi-path/scripts")
from offline_hotpot_env import OfflineHotpotEnv

MODEL = "Qwen2.5-7B-Instruct"
N_TASKS, MCTS_ITERS = 10, 5
GROUPS = {
    "A":  {"prefix_cache": False, "n": 2, "desc": "无共享基线"},
    "B":  {"prefix_cache": True,  "n": 1, "desc": "单分支基准"},
    "C1": {"prefix_cache": True,  "n": 2, "desc": "低分支"},
    "C2": {"prefix_cache": True,  "n": 4, "desc": "中分支"},
    "C3": {"prefix_cache": True,  "n": 8, "desc": "高分支"},
}


# ---- MCTS 核心（简版，基于 LATS 思想）----
class MCTSNode:
    def __init__(self, state="", parent=None, depth=0):
        self.state = state      # 当前文本（搜索历史）
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0
        self.depth = depth
        self.is_terminal = False

    def uct(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)


def gpt(prompt, n=1, temperature=0.7, max_tokens=256):
    """调用 vllm 引擎"""
    try:
        r = openai.ChatCompletion.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, n=n,
        )
        if n == 1:
            return [r.choices[0].message.content]
        return [c.message.content for c in r.choices]
    except Exception as e:
        return [f"ERROR: {e}"]


def propose_actions(trajectory, n_branches):
    """生成 N 个候选 action"""
    prompt = (
        f"Based on the following search history, propose {n_branches} different "
        f"next actions to find the answer. Each action should be one of:\n"
        f"  Search[query] - search documents\n"
        f"  Lookup[keyword] - find specific sentences\n"
        f"  Finish[answer] - submit final answer\n\n"
        f"History:\n{trajectory}\n\n"
        f"Propose exactly {n_branches} different actions, one per line:"
    )
    responses = gpt(prompt, n=1, temperature=0.7, max_tokens=200)
    actions = [a.strip() for a in responses[0].split("\n") if a.strip() and "[" in a]
    return actions[:n_branches]


def evaluate_action(question, trajectory, action):
    """评估 action 的质量（1-10）"""
    prompt = (
        f"Question: {question}\n\n"
        f"Search history:\n{trajectory}\n\n"
        f"Next proposed action: {action}\n\n"
        f"Rate this action's usefulness for answering the question on a scale of 1-10. "
        f"Respond with just the number."
    )
    responses = gpt(prompt, n=1, temperature=0.3, max_tokens=5)
    try:
        nums = [int(s) for s in responses[0].split() if s.isdigit()]
        return nums[0] / 10.0 if nums else 0.5
    except Exception:
        return 0.5


def mcts_search(env, n_branches=2, max_iters=5):
    """MCTS 搜索：选择→扩展→评估→回传"""
    root_traj = f"Question: {env._question}\n"
    root = MCTSNode(state=root_traj)

    for iteration in range(max_iters):
        # Selection: 选 UCT 最大的叶子
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.uct())

        if node.is_terminal:
            break

        # Expansion: 生成候选 action
        actions = propose_actions(node.state, n_branches)

        for action in actions:
            if "Finish" in action:
                child = MCTSNode(state=node.state + f"Action: {action}\n", parent=node, depth=node.depth + 1)
                # 执行获得 reward
                obs, reward, done, info = env.step(action)
                child.state += f"Observation: {obs}\n"
                child.reward = reward
                child.is_terminal = done
                child.value = reward
                child.visits = 1
                node.children.append(child)
                if reward == 1:
                    return child  # 找到了
            else:
                child = MCTSNode(state=node.state + f"Action: {action}\n", parent=node, depth=node.depth + 1)
                value = evaluate_action(env._question, node.state, action)
                obs, _, _, _ = env.step(action)
                child.state += f"Observation: {obs}\n"
                child.value = value
                child.visits = 1
                node.children.append(child)

        # Backpropagation
        for child in node.children:
            cur = child.parent
            while cur:
                cur.visits += 1
                cur.value += child.value
                cur = cur.parent

        # 检查是否有 terminal 且 reward=1
        for child in node.children:
            if child.is_terminal and child.reward == 1:
                return child

    # 没找到完全正确的，返回最佳
    all_nodes = [root]
    def collect(n):
        for c in n.children:
            all_nodes.append(c)
            collect(c)
    collect(root)
    best = max(all_nodes, key=lambda n: n.reward * 10 + n.value)
    return best


# ---- 指标采集 ----
def collect_metrics():
    """完整指标采集"""
    t = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5).read().decode()
    hits = queries = reqs = gen_tok = prom_tok = 0.0
    ttft_sum = ttft_cnt = tpot_sum = tpot_cnt = e2e_sum = e2e_cnt = 0.0

    for line in t.split("\n"):
        if line.startswith("vllm:prefix_cache_hits_total"):
            hits = float(line.split()[-1])
        if line.startswith("vllm:prefix_cache_queries_total"):
            queries = float(line.split()[-1])
        if line.startswith("vllm:request_success_total"):
            reqs += float(line.split()[-1])
        if line.startswith("vllm:generation_tokens_total"):
            gen_tok = float(line.split()[-1])
        if line.startswith("vllm:prompt_tokens_total"):
            prom_tok = float(line.split()[-1])
        if line.startswith("vllm:time_to_first_token_seconds_sum"):
            ttft_sum = float(line.split()[-1])
        if line.startswith("vllm:time_to_first_token_seconds_count"):
            ttft_cnt = float(line.split()[-1])
        if line.startswith("vllm:request_time_per_output_token_seconds_sum"):
            tpot_sum = float(line.split()[-1])
        if line.startswith("vllm:request_time_per_output_token_seconds_count"):
            tpot_cnt = float(line.split()[-1])
        if line.startswith("vllm:e2e_request_latency_seconds_sum"):
            e2e_sum = float(line.split()[-1])
        if line.startswith("vllm:e2e_request_latency_seconds_count"):
            e2e_cnt = float(line.split()[-1])

    return {
        "kv_hit": hits / max(queries, 1),
        "reqs": reqs,
        "gen_tokens": gen_tok,
        "prompt_tokens": prom_tok,
        "ttft_ms": (ttft_sum / max(ttft_cnt, 1)) * 1000,
        "tpot_ms": (tpot_sum / max(tpot_cnt, 1)) * 1000,
        "e2e_ms": (e2e_sum / max(e2e_cnt, 1)) * 1000,
    }


class MemSampler:
    def __init__(self):
        self.peak = 0.0; self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
    def start(self): self._t.start()
    def stop(self):
        self._stop.set(); self._t.join(timeout=5)
        return self.peak
    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(["npu-smi","info","-t","usages","-i","10","-c","0"], text=True, stderr=subprocess.DEVNULL)
                for l in out.split("\n"):
                    if "HBM Usage Rate" in l:
                        self.peak = max(self.peak, float(l.split(":")[-1].strip().replace("%","")) * 655.36)
            except: pass
            time.sleep(1)


# ---- 主实验 ----
def main():
    env = OfflineHotpotEnv(max_examples=N_TASKS)
    results = []

    for name, cfg in GROUPS.items():
        print(f"\n=== {name}: {cfg['desc']} (pcache={cfg['prefix_cache']}, n={cfg['n']}) ===")

        # 重置后跑所有问题
        sampler = MemSampler()
        sampler.start()
        before = collect_metrics()

        t0 = time.monotonic()
        solved = 0
        for idx in range(N_TASKS):
            obs = env.reset(idx=idx)
            node = mcts_search(env, n_branches=cfg["n"], max_iters=MCTS_ITERS)
            if node and node.reward == 1:
                solved += 1
            sys.stdout.write(f"\r  [{idx+1}/{N_TASKS}] solved={solved}")

        wall = time.monotonic() - t0
        after = collect_metrics()
        mem = sampler.stop()

        after_m = after
        r = {
            "group": name, "desc": cfg["desc"],
            "pc": cfg["prefix_cache"], "n": cfg["n"],
            "wall": wall, "success": solved / N_TASKS,
            "kv_before": before["kv_hit"], "kv_after": after_m["kv_hit"],
            "mem_peak": mem, "reqs": after_m["reqs"] - before["reqs"],
            "ttft_ms": after_m["ttft_ms"], "tpot_ms": after_m["tpot_ms"],
            "e2e_ms": after_m["e2e_ms"],
            "gen_tokens": after_m["gen_tokens"], "prompt_tokens": after_m["prompt_tokens"],
        }
        results.append(r)
        print(f"\n  wall={wall:.0f}s success={r['success']:.0%} kv={after_m['kv_hit']:.3f} "
              f"mem={mem:.0f}MB ttft={after_m['ttft_ms']:.0f}ms tpot={after_m['tpot_ms']:.1f}ms "
              f"e2e={after_m['e2e_ms']:.0f}ms reqs={r['reqs']:.0f}")

    print(f"\n{'='*130}")
    print(f"{'Grp':<5} {'pc':>3} {'n':>3} {'wall':>7} {'succ':>6} {'kv_bef':>8} {'kv_aft':>8} {'mem':>8} {'ttft':>7} {'tpot':>7} {'e2e':>7} {'reqs':>6}")
    for r in results:
        print(f"{r['group']:<5} {str(r['pc']):>3} {r['n']:>3} {r['wall']:>7.0f}s {r['success']:>5.0%} "
              f"{r['kv_before']:>8.3f} {r['kv_after']:>8.3f} {r['mem_peak']:>8.0f} "
              f"{r['ttft_ms']:>7.0f} {r['tpot_ms']:>7.1f} {r['e2e_ms']:>7.0f} {r['reqs']:>6.0f}")

    with open("/tmp/multi-path/logs-lats/hotpot_offline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: logs-lats/hotpot_offline.json")


if __name__ == "__main__":
    main()
