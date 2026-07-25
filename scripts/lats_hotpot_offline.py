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


import re


def gpt(prompt, n=1, temperature=0.7, max_tokens=256):
    try:
        r = openai.ChatCompletion.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, n=n,
        )
        return [c.message.content for c in r.choices]
    except Exception as e:
        return [f"ERROR: {e}"]


def _parse_actions(text, n_branches):
    """从 LLM 返回文本中解析 action，支持多种格式"""
    actions = []
    # 格式 1: Search[xxx] / Lookup[xxx] / Finish[xxx]
    for pattern in [r'Search\[([^\]]+)\]', r'Lookup\[([^\]]+)\]', r'Finish\[([^\]]+)\]']:
        for m in re.finditer(pattern, text):
            if pattern.startswith('S'):
                actions.append(f"Search[{m.group(1)}]")
            elif pattern.startswith('L'):
                actions.append(f"Lookup[{m.group(1)}]")
            else:
                actions.append(f"Finish[{m.group(1)}]")

    # 格式 2: 每行一个 action 名 + 参数
    if not actions:
        for line in text.split("\n"):
            line = line.strip().rstrip(".,;")
            if line and len(line) < 200:
                if any(kw in line.lower() for kw in ["search", "查找", "搜索"]):
                    q = line.split(":", 1)[-1].strip() if ":" in line else line
                    actions.append(f"Search[{q}]")
                elif any(kw in line.lower() for kw in ["lookup", "查找句子", "定位"]):
                    q = line.split(":", 1)[-1].strip() if ":" in line else line
                    actions.append(f"Lookup[{q}]")
                elif any(kw in line.lower() for kw in ["finish", "answer", "答案", "最终"]):
                    q = line.split(":", 1)[-1].strip() if ":" in line else line
                    actions.append(f"Finish[{q}]")
    return actions[:n_branches]


def propose_actions(trajectory, question, n_branches):
    prompt = (
        f"You are answering a multi-hop question using search.\n"
        f"Question: {question}\n\n"
        f"Previous steps:\n{trajectory}\n"
        f"Propose {n_branches} different next actions. Use EXACTLY this format for each:\n"
        f"Search: <query>  (to search all documents)\n"
        f"Lookup: <keyword> (to find specific sentences)\n"
        f"Finish: <answer>  (to submit final answer)\n"
        f"Respond with one action per line."
    )
    text = gpt(prompt, temperature=0.7, max_tokens=300)[0]
    actions = _parse_actions(text, n_branches)

    # Fallback: 从问题提取关键词搜索
    if not actions:
        words = [w for w in question.split() if len(w) > 3][:3]
        if words:
            actions = [f"Search[{w}]" for w in words[:n_branches]]
        else:
            actions = [f"Search[{question[:50]}]"]
    return actions


def evaluate_action(question, trajectory, action):
    prompt = (
        f"Question: {question}\nHistory:\n{trajectory}\n"
        f"Proposed action: {action}\n"
        f"Rate usefulness (1-10, number only):"
    )
    text = gpt(prompt, temperature=0.3, max_tokens=10)[0]
    nums = [int(s) for s in re.findall(r'\d+', text)]
    return nums[0] / 10.0 if nums else 0.5


def mcts_search(env, n_branches=2, max_iters=5):
    root_traj = f"Question: {env._question}\n"
    root = MCTSNode(state=root_traj)
    best_overall = root

    for iteration in range(max_iters):
        # Selection: 选 UCT 最大的叶子节点
        node = root
        while node.children and not node.is_terminal:
            unvisited = [c for c in node.children if c.visits == 0]
            if unvisited:
                node = unvisited[0]
            else:
                node = max(node.children, key=lambda c: c.uct())

        if node.is_terminal:
            if node.reward == 1:
                return node
            continue

        # Expansion
        actions = propose_actions(node.state, env._question, n_branches)

        for action in actions:
            child = MCTSNode(state=node.state + f"Action: {action}\n", parent=node, depth=node.depth + 1)
            obs, reward, done, info = env.step(action)
            child.state += f"Observation: {obs}\n"
            child.reward = reward
            child.is_terminal = done

            if not done and "Finish" not in action:
                value = evaluate_action(env._question, node.state, action)
            else:
                value = reward

            child.value = value
            child.visits = 1
            node.children.append(child)

            if child.reward == 1:
                return child
            if child.reward > best_overall.reward:
                best_overall = child

        # Backpropagation
        for child in node.children:
            cur = child.parent
            while cur:
                cur.visits += 1
                cur.value += child.value
                cur = cur.parent

    # 收集所有节点，返回最好的
    all_nodes = [root]
    def collect(n):
        for c in n.children:
            all_nodes.append(c)
            collect(c)
    collect(root)
    best = max(all_nodes, key=lambda n: n.reward * 10 + n.value)
    return best if best.reward > best_overall.reward else best_overall


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
