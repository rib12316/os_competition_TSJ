"""LATS MCTS + 离线 HotpotQA 多路径决策实验
v2 改进：
- _parse_actions: Finish 永远不被截断
- MCTS rollout: 替代单点 LLM 评估，模拟到终止
- 树过程打印
"""
import os, sys, time, json, math, subprocess, urllib.request, threading, re
from argparse import Namespace

os.environ["OPENAI_API_KEY"] = "stub"
import openai
openai.api_key = "stub"
openai.api_base = "http://localhost:8000/v1"

REPO = "/data/os_competition_TSJ"
sys.path.insert(0, "/tmp/multi-path/scripts")
from offline_hotpot_env import OfflineHotpotEnv

MODEL = "Qwen2.5-7B-Instruct"
N_TASKS, MCTS_ITERS = 10, 8           # v2: 增加到 8 轮
ROLLOUT_DEPTH = 2                      # rollout 最多额外 2 步
ROLLOUT_BRANCHES = 2                   # rollout 分支数

GROUPS = {
    "A":  {"prefix_cache": False, "n": 2, "desc": "无共享基线"},
    "B":  {"prefix_cache": True,  "n": 1, "desc": "单分支基准"},
    "C1": {"prefix_cache": True,  "n": 2, "desc": "低分支"},
    "C2": {"prefix_cache": True,  "n": 4, "desc": "中分支"},
    "C3": {"prefix_cache": True,  "n": 8, "desc": "高分支"},
}


# ═══════════════════════════════════════════════════════════════════════
# MCTS 核心（v2：Finish 不截断 + rollout）
# ═══════════════════════════════════════════════════════════════════════

class MCTSNode:
    __slots__ = ("state", "parent", "children", "visits", "value",
                 "reward", "is_terminal", "depth", "action", "obs",
                 "eval_score", "rollout_value", "id")
    _next_id = 0

    def __init__(self, state="", parent=None, depth=0, action="", obs=""):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.reward = 0
        self.is_terminal = False
        self.depth = depth
        self.action = action
        self.obs = obs
        self.eval_score = 0.0       # 保留兼容
        self.rollout_value = 0.0    # rollout 平均 reward
        self.id = MCTSNode._next_id
        MCTSNode._next_id += 1

    def uct(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    @property
    def avg_value(self):
        return self.value / self.visits if self.visits > 0 else 0.0


def gpt(prompt, n=1, temperature=0.7, max_tokens=300):
    try:
        r = openai.ChatCompletion.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=max_tokens, n=n,
        )
        return [c.message.content for c in r.choices]
    except Exception as e:
        return [f"ERROR: {e}"]


def _clean_answer(text):
    """清理 Finish 中的答案文本：去掉尾部说明性文字"""
    # 去掉常见的尾部说明（括号内或破折号后）
    for sep in [" (to ", " (e.g.", " (for ", " - to ", " - based "]:
        if sep in text:
            text = text[:text.index(sep)]
    # 去掉末尾多余标点
    text = text.strip().rstrip(".,;: ")
    return text


def _parse_actions(text, n_branches):
    """v3: 从 LLM 返回文本中解析 action。

    - Finish 永远不被截断
    - Finish 答案自动清理尾部说明文字
    """
    explore_actions = []   # Search, Lookup
    terminal_actions = []  # Finish

    # 格式 1: Search[xxx] / Lookup[xxx] / Finish[xxx]
    for pattern, tag in [(r'Search\[([^\]]+)\]', 'Search'),
                         (r'Lookup\[([^\]]+)\]', 'Lookup'),
                         (r'Finish\[([^\]]+)\]', 'Finish')]:
        for m in re.finditer(pattern, text):
            arg = m.group(1)
            if tag == 'Finish':
                arg = _clean_answer(arg)
                terminal_actions.append(f"Finish[{arg}]")
            else:
                explore_actions.append(f"{tag}[{arg}]")

    # 格式 2: 每行一个 action 名 + 参数（fallback）
    if not explore_actions and not terminal_actions:
        for line in text.split("\n"):
            line = line.strip().rstrip(".,;")
            if not line or len(line) >= 200:
                continue
            if any(kw in line.lower() for kw in ["search", "查找", "搜索"]):
                q = line.split(":", 1)[-1].strip() if ":" in line else line
                explore_actions.append(f"Search[{q}]")
            elif any(kw in line.lower() for kw in ["lookup", "查找句子", "定位"]):
                q = line.split(":", 1)[-1].strip() if ":" in line else line
                explore_actions.append(f"Lookup[{q}]")
            elif any(kw in line.lower() for kw in ["finish", "answer", "答案", "最终"]):
                q = line.split(":", 1)[-1].strip() if ":" in line else line
                q = _clean_answer(q)
                terminal_actions.append(f"Finish[{q}]")

    # 组装：explore 取前 n_branches 个，terminal 始终保留（最多 1 个）
    result = explore_actions[:n_branches] + terminal_actions[:1]
    return result


def propose_actions(trajectory, question, n_branches):
    prompt = (
        f"You are answering a multi-hop question using search.\n"
        f"Question: {question}\n\n"
        f"Previous steps:\n{trajectory}\n"
        f"Propose {n_branches} different next actions. Use EXACTLY this format for each:\n"
        f"Search: <query>  (to search all documents)\n"
        f"Lookup: <keyword> (to find specific sentences)\n"
        f"Finish: <answer>  (to submit final answer)\n"
        f"Respond with one action per line.\n"
        f"If you have gathered enough evidence to answer, include a Finish action."
    )
    text = gpt(prompt, temperature=0.5, max_tokens=300)[0]
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
    """单点评估（保留用于 rollout 内部）"""
    prompt = (
        f"Question: {question}\nHistory:\n{trajectory}\n"
        f"Proposed action: {action}\n"
        f"Rate usefulness (1-10, number only):"
    )
    text = gpt(prompt, temperature=0.1, max_tokens=10)[0]
    nums = [int(s) for s in re.findall(r'\d+', text)]
    return nums[0] / 10.0 if nums else 0.5


def rollout(env, state, question, depth=ROLLOUT_DEPTH, n_branches=ROLLOUT_BRANCHES):
    """v3: 确定性 rollout — 温度=0，始终选第一个 explore action。

    从当前 state 模拟 depth 步：
    - 每步用 temperature=0 propose（确定性）
    - 如果模型给 Finish → 执行并返回 reward
    - 否则选第一个 explore action 继续
    - 最后没 Finish → 返回 0（保守估计，不靠 LLM 打分）
    """
    saved_idx = env._idx
    saved_steps = env._steps
    saved_done = env._done

    current_state = state
    last_explore = None

    for step in range(depth):
        # v3: rollout 内用 temperature=0（确定性）
        prompt = (
            f"You are answering a multi-hop question using search.\n"
            f"Question: {question}\n\n"
            f"Previous steps:\n{current_state}\n"
            f"Propose {n_branches} different next actions. Use EXACTLY this format for each:\n"
            f"Search: <query>\nLookup: <keyword>\nFinish: <answer>\n"
            f"Respond with one action per line.\n"
            f"If you have enough evidence to answer, include a Finish action."
        )
        text = gpt(prompt, temperature=0, max_tokens=200)[0]
        actions = _parse_actions(text, n_branches)

        # 遇到 Finish → 直接返回 reward
        finish_actions = [a for a in actions if a.startswith("Finish[")]
        if finish_actions:
            obs, reward, done, info = env.step(finish_actions[0])
            env._done = saved_done
            env._steps = saved_steps
            env._idx = saved_idx
            return float(reward)

        # 选第一个 explore action（确定性）
        explore_actions = [a for a in actions if not a.startswith("Finish[")]
        if not explore_actions:
            break

        chosen = explore_actions[0]  # v3: 确定性选择
        last_explore = chosen
        obs, reward, done, info = env.step(chosen)
        current_state += f"Action: {chosen}\nObservation: {obs}\n"

        if done:
            env._done = saved_done
            env._steps = saved_steps
            env._idx = saved_idx
            return float(reward)

    env._done = saved_done
    env._steps = saved_steps
    env._idx = saved_idx
    # v3: 没到达终止 → 返回 0（保守），不再靠 LLM 打分
    return 0.0


def _extract_best_answer(root):
    """v3: 扫描整棵树中所有 Finish 节点，返回最佳答案及其信息。

    优先级：reward=1 > reward=0（选 rollout_value 最高的那个）
    返回 (answer_string, reward, depth)
    """
    all_finishes = []

    def _collect(n):
        if n.action.startswith("Finish["):
            ans = n.action[len("Finish["):-1] if n.action.endswith("]") else n.action[7:]
            all_finishes.append((ans, n.reward, n.depth, n.rollout_value))
        for c in n.children:
            _collect(c)

    _collect(root)

    if not all_finishes:
        return ("", 0, 0)

    # 排序：reward 降序 → rollout_value 降序
    all_finishes.sort(key=lambda x: (x[1], x[3]), reverse=True)
    best = all_finishes[0]
    return (best[0], best[1], best[2])


def _extract_answer(state):
    """从 state 文本中提取最后一个 Finish 的答案（兼容旧接口）"""
    matches = re.findall(r'Finish\[([^\]]+)\]', state)
    return matches[-1] if matches else ""


# ── 树打印 ──
def print_tree_compact(root, max_depth=5):
    """紧凑版树打印"""
    GREEN = "\033[32m"; RED = "\033[31m"; CYAN = "\033[36m"
    BOLD = "\033[1m"; YELLOW = "\033[33m"; RESET = "\033[0m"

    def _walk(n, prefix="", is_last=True, depth=0):
        if depth > max_depth:
            return
        if depth == 0:
            line = f"{BOLD}ROOT{RESET} v={n.visits} avg={n.avg_value:.2f}"
        else:
            connector = "└─" if is_last else "├─"
            action = n.action[:45]
            if n.reward == 1:
                mark = f"{GREEN}✓{RESET}"
            elif n.is_terminal:
                mark = f"{RED}✗{RESET}"
            else:
                mark = ""
            rv = n.rollout_value
            rv_str = f" roll={rv:.2f}" if rv > 0 else ""
            line = (f"{connector} {CYAN}N{n.id}{RESET} {action} {mark} "
                    f"v={n.visits} avg={n.avg_value:.2f}{rv_str}")
        print(f"{prefix}{line}")

        child_prefix = prefix + ("   " if is_last else "│  ")
        for i, child in enumerate(n.children):
            _walk(child, child_prefix, i == len(n.children) - 1, depth + 1)

    _walk(root)


# ── MCTS 搜索（v2：rollout 评估） ──
def mcts_search(env, n_branches=2, max_iters=8, verbose=False):
    MCTSNode._next_id = 0
    root_traj = f"Question: {env._question}\n"
    root = MCTSNode(state=root_traj)
    best_overall = root

    if verbose:
        BOLD = "\033[1m"; YELLOW = "\033[33m"; CYAN = "\033[36m"
        GREEN = "\033[32m"; RED = "\033[31m"; BLUE = "\033[34m"; RESET = "\033[0m"
        print(f"{BLUE}{'═'*70}{RESET}")

    for iteration in range(max_iters):
        # ── Selection ──
        # v3: 跳过 is_terminal 节点（它们无法再扩展）
        node = root
        while node.children and not node.is_terminal:
            # 优先选未访问的；如果所有子节点都访问过，选 UCT 最大的
            expandable = [c for c in node.children if not c.is_terminal]
            if not expandable:
                break  # 所有子节点都终止了，回退到父节点继续
            unvisited = [c for c in expandable if c.visits == 0]
            if unvisited:
                node = unvisited[0]
            else:
                node = max(expandable, key=lambda c: c.uct())

        if node.is_terminal:
            if node.reward == 1:
                return node
            continue  # 已终止但 reward=0，跳过

        # ── Expansion ──
        actions = propose_actions(node.state, env._question, n_branches)

        if verbose:
            print(f"{YELLOW}[iter {iteration+1}/{max_iters}]{RESET} "
                  f"N{node.id}(d={node.depth}) → {len(actions)} actions", end="")

        for action in actions:
            child = MCTSNode(
                state=node.state + f"Action: {action}\n",
                parent=node, depth=node.depth + 1, action=action,
            )
            obs, reward, done, info = env.step(action)
            child.state += f"Observation: {obs}\n"
            child.reward = reward
            child.is_terminal = done
            child.obs = obs

            # v2: rollout 评估替代单点 LLM 评估
            if not done:
                child.rollout_value = rollout(env, child.state, env._question)
                child.eval_score = child.rollout_value  # 兼容
            else:
                child.rollout_value = reward
                child.eval_score = reward

            child.value = child.rollout_value
            child.visits = 1
            node.children.append(child)

            if verbose:
                tag = "F" if "Finish" in action else ("S" if "Search" in action else "L")
                r_str = f" {GREEN}R=1!{RESET}" if reward == 1 else ""
                t_str = f" {RED}END{RESET}" if done else ""
                print(f"\n  {CYAN}N{child.id}{RESET} [{tag}] {action[:40]:40s} "
                      f"roll={child.rollout_value:.2f}{r_str}{t_str}", end="")

            if child.reward == 1:
                if verbose:
                    print(f"\n  {GREEN}{BOLD}★★★ SOLVED! ★★★{RESET}")
                return child
            if child.reward > best_overall.reward:
                best_overall = child

        if verbose:
            print()

        # ── Backpropagation ──
        for child in node.children:
            cur = child.parent
            while cur:
                cur.visits += 1
                cur.value += child.rollout_value
                cur = cur.parent

    if verbose:
        print(f"\n{BLUE}{'─'*70}{RESET}")
        print(f"Tree:")
        print_tree_compact(root)

    # 收集所有节点，返回最好的
    all_nodes = [root]
    def collect(n):
        for c in n.children:
            all_nodes.append(c)
            collect(c)
    collect(root)

    # v2: 优先 reward>0，其次 rollout_value 高
    best = max(all_nodes, key=lambda n: n.reward * 10 + n.rollout_value)
    return best if best.reward >= best_overall.reward else best_overall


# ═══════════════════════════════════════════════════════════════════════
# 指标采集（不变）
# ═══════════════════════════════════════════════════════════════════════

def collect_metrics():
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
                out = subprocess.check_output(
                    ["npu-smi","info","-t","usages","-i","10","-c","0"],
                    text=True, stderr=subprocess.DEVNULL)
                for l in out.split("\n"):
                    if "HBM Usage Rate" in l:
                        self.peak = max(self.peak, float(
                            l.split(":")[-1].strip().replace("%","")) * 655.36)
            except: pass
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", type=int, default=-1,
                    help="Run single case in verbose mode (index)")
    ap.add_argument("--group", type=str, default=None,
                    help="Only run this group (A/B/C1/C2/C3)")
    args = ap.parse_args()

    # ── 单例调试模式 ──
    if args.debug >= 0:
        env = OfflineHotpotEnv(max_examples=100)
        idx = args.debug
        cfg = GROUPS.get(args.group, GROUPS["C1"]) if args.group else {"n": 2, "prefix_cache": True}
        obs = env.reset(idx=idx)
        print(f"Question [{idx}]: {env._question}")
        print(f"GT: {env._answer}")
        print(f"Docs: {', '.join(d['title'] for d in env._context)}")
        print()

        t0 = time.monotonic()
        node = mcts_search(env, n_branches=cfg["n"], max_iters=MCTS_ITERS, verbose=True)
        wall = time.monotonic() - t0

        # 找到 root 扫描全树
        root = node
        while root.parent:
            root = root.parent
        best_ans, best_reward, best_depth = _extract_best_answer(root)

        # 同时从最佳节点提取
        node_ans = _extract_answer(node.state)
        correct = env._is_correct(best_ans) if best_ans else False

        print(f"\nWall={wall:.1f}s  BestAnswer={best_ans[:80]}  GT={env._answer}")
        print(f"Best reward={best_reward}  depth={best_depth}  correct(lenient)={correct}")
        print(f"Returned node: N{node.id} reward={node.reward} depth={node.depth}")
        if node_ans and node_ans != best_ans:
            print(f"Node answer: {node_ans[:80]}")

        # 列出所有 Finish 尝试
        all_finishes = []
        def _collect(n):
            if n.action.startswith("Finish["):
                ans = n.action[len("Finish["):-1] if n.action.endswith("]") else n.action[7:]
                all_finishes.append((n.id, ans[:60], n.reward, n.rollout_value))
            for c in n.children:
                _collect(c)
        _collect(root)
        if all_finishes:
            print(f"\nAll Finish attempts ({len(all_finishes)}):")
            for nid, ans, r, rv in all_finishes:
                mark = "✓" if r == 1 else "✗"
                print(f"  N{nid}: [{mark}] {ans}")
        return

    # ── 正常实验模式 ──
    env = OfflineHotpotEnv(max_examples=N_TASKS)
    results = []

    for name, cfg in GROUPS.items():
        if args.group and name != args.group:
            continue
        print(f"\n=== {name}: {cfg['desc']} (pcache={cfg['prefix_cache']}, n={cfg['n']}) ===")

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

        r = {
            "group": name, "desc": cfg["desc"],
            "pc": cfg["prefix_cache"], "n": cfg["n"],
            "wall": wall, "success": solved / N_TASKS,
            "kv_before": before["kv_hit"], "kv_after": after["kv_hit"],
            "mem_peak": mem, "reqs": after["reqs"] - before["reqs"],
            "ttft_ms": after["ttft_ms"], "tpot_ms": after["tpot_ms"],
            "e2e_ms": after["e2e_ms"],
            "gen_tokens": after["gen_tokens"], "prompt_tokens": after["prompt_tokens"],
        }
        results.append(r)
        print(f"\n  wall={wall:.0f}s success={r['success']:.0%} kv={after['kv_hit']:.3f} "
              f"mem={mem:.0f}MB ttft={after['ttft_ms']:.0f}ms tpot={after['tpot_ms']:.1f}ms "
              f"e2e={after['e2e_ms']:.0f}ms reqs={r['reqs']:.0f}")

    print(f"\n{'='*130}")
    print(f"{'Grp':<5} {'pc':>3} {'n':>3} {'wall':>7} {'succ':>6} {'kv_bef':>8} {'kv_aft':>8} "
          f"{'mem':>8} {'ttft':>7} {'tpot':>7} {'e2e':>7} {'reqs':>6}")
    for r in results:
        print(f"{r['group']:<5} {str(r['pc']):>3} {r['n']:>3} {r['wall']:>7.0f}s {r['success']:>5.0%} "
              f"{r['kv_before']:>8.3f} {r['kv_after']:>8.3f} {r['mem_peak']:>8.0f} "
              f"{r['ttft_ms']:>7.0f} {r['tpot_ms']:>7.1f} {r['e2e_ms']:>7.0f} {r['reqs']:>6.0f}")

    with open("/tmp/multi-path/logs-lats/hotpot_offline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: logs-lats/hotpot_offline.json")


if __name__ == "__main__":
    main()
