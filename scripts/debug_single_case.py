"""单例 MCTS 调试 — 复用 lats_hotpot_offline 的 MCTS 核心，增强打印"""
import os, sys, time, re

os.environ["OPENAI_API_KEY"] = "stub"

REPO = "/data/os_competition_TSJ"
sys.path.insert(0, f"{REPO}/third_party/LanguageAgentTreeSearch/programming")
sys.path.insert(0, "/tmp/multi-path/scripts")

from offline_hotpot_env import OfflineHotpotEnv
from lats_hotpot_offline import (
    MCTSNode, gpt, _parse_actions, _clean_answer,
    propose_actions, evaluate_action, rollout,
    _extract_best_answer, _extract_answer,
    print_tree_compact,
    MCTS_ITERS, ROLLOUT_DEPTH,
)

MODEL = "Qwen2.5-7B-Instruct"

# ── 颜色 ──
BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
BLUE = "\033[34m"; CYAN = "\033[36m"; MAGENTA = "\033[35m"
RESET = "\033[0m"

def c(s, color):
    return f"{color}{s}{RESET}"


# ═══════════════════════════════════════════════════════════════════════
# 带详细日志的 MCTS 搜索（包装 lats_hotpot_offline.mcts_search）
# ═══════════════════════════════════════════════════════════════════════

def mcts_search_verbose(env, n_branches=2, max_iters=8):
    """与 lats_hotpot_offline.mcts_search 相同的算法，但打印所有细节"""
    MCTSNode._next_id = 0
    root_traj = f"Question: {env._question}\n"
    root = MCTSNode(state=root_traj)
    best_overall = root

    print(c(f"\n{'═'*70}", BLUE))
    print(c(f"  MCTS SEARCH  |  branches={n_branches}  iters={max_iters}  rollout_depth={ROLLOUT_DEPTH}", BOLD))
    print(c(f"{'═'*70}\n", BLUE))

    for iteration in range(max_iters):
        # ── Selection ──
        node = root
        path = [node]
        while node.children and not node.is_terminal:
            expandable = [c for c in node.children if not c.is_terminal]
            if not expandable:
                break
            unvisited = [c for c in expandable if c.visits == 0]
            if unvisited:
                node = unvisited[0]
            else:
                # 显示 UCT 排序
                ranked = sorted(expandable, key=lambda c: c.uct(), reverse=True)
                node = ranked[0]
            path.append(node)

        print(c(f"── Iteration {iteration+1}/{max_iters} ──", YELLOW + BOLD))
        print(f"  Path: {' → '.join(f'N{n.id}(d={n.depth})' for n in path)}")

        if node.is_terminal:
            print(f"  {c('Terminal node, skip', RED)}")
            if node.reward == 1:
                return root  # 返回 root 让调用方遍历全树
            continue

        # ── Expansion ──
        print(f"  {c(f'Expanding N{node.id} (depth={node.depth})', CYAN)}")
        state_preview = node.state.replace('\n', '↵')[-200:]
        print(f"  {c('State tail:', DIM)} ...{state_preview[-150:]}")

        t0 = time.time()
        actions = propose_actions(node.state, env._question, n_branches)
        dt = time.time() - t0
        print(f"  Propose → {len(actions)} actions ({dt:.1f}s):")

        for i, action in enumerate(actions):
            color = GREEN if "Finish" in action else (BLUE if "Search" in action else MAGENTA)
            tag = "F" if "Finish" in action else ("S" if "Search" in action else "L")
            print(f"    [{i+1}] {c(f'[{tag}] {action}', color)}")

        if not actions:
            print(f"  {c('WARNING: No actions!', RED)}")
            continue

        # 执行每个 action + rollout
        for action in actions:
            child = MCTSNode(
                state=node.state + f"Action: {action}\n",
                parent=node, depth=node.depth + 1, action=action,
            )

            # 环境 step
            obs, reward, done, info = env.step(action)
            child.state += f"Observation: {obs}\n"
            child.reward = reward
            child.is_terminal = done
            child.obs = obs

            # rollout 评估
            if not done:
                child.rollout_value = rollout(env, child.state, env._question)
                child.eval_score = child.rollout_value
            else:
                child.rollout_value = reward
                child.eval_score = reward

            child.value = child.rollout_value
            child.visits = 1
            node.children.append(child)

            # 打印子节点
            tag = "F" if "Finish" in action else ("S" if "Search" in action else "L")
            act_short = action[:45]
            r_str = c(" R=1!", GREEN + BOLD) if reward == 1 else ""
            t_str = c(" END", RED) if done else ""
            print(f"      → N{child.id} [{tag}] {act_short:45s} roll={child.rollout_value:.2f}{r_str}{t_str}")

            if reward == 1:
                print(f"\n  {c('★★★ SOLVED! ★★★', GREEN + BOLD)}")
                return root  # 返回 root 让调用方遍历全树
            if reward > best_overall.reward:
                best_overall = child

        # ── Backpropagation ──
        for child in node.children:
            cur = child.parent
            while cur:
                cur.visits += 1
                cur.value += child.rollout_value
                cur = cur.parent

        # 打印当前树
        print(f"\n  {c('Tree:', DIM)}")
        print_tree_compact(root)

    print(c(f"\n{'─'*70}", BLUE))
    print(c(f"  SEARCH COMPLETE (max iters reached)", BOLD))
    print(c(f"{'─'*70}\n", BLUE))
    return root


# ═══════════════════════════════════════════════════════════════════════
# 树详情打印（递归）
# ═══════════════════════════════════════════════════════════════════════

def print_tree_detail(node, indent=0, max_depth=6):
    """递归打印树详情（含 observation 片段）"""
    if indent > max_depth * 3:
        return
    prefix = "  " * indent
    if indent == 0:
        print(f"{prefix}{c('● ROOT', BOLD)}  v={node.visits}  val={node.value:.2f}  avg={node.avg_value:.2f}")
    else:
        action_short = node.action[:55]
        term = c(" [TERM]", RED) if node.is_terminal else ""
        r = c(f" R={node.reward}", GREEN) if node.reward > 0 else ""
        print(f"{prefix}├─ {c(f'N{node.id}', CYAN)} d={node.depth} {action_short}{term}{r}")
        print(f"{prefix}│  v={node.visits} val={node.value:.2f} avg={node.avg_value:.2f} "
              f"roll={node.rollout_value:.2f} uct={node.uct():.3f}")
        if node.obs:
            obs_short = node.obs[:100].replace('\n', '↵')
            print(f"{prefix}│  {c('obs:', DIM)} {obs_short}")

    for child in node.children:
        print_tree_detail(child, indent + 1, max_depth)


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_branches = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_iters = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    print(f"\n{'█'*70}")
    print(f"  LATS MCTS Debug  |  idx={idx}  branches={n_branches}  iters={max_iters}")
    print(f"{'█'*70}")

    print(f"\n  Loading HotpotQA...")
    env = OfflineHotpotEnv(max_examples=100)
    obs = env.reset(idx=idx)

    # 问题信息
    print(f"\n{'─'*70}")
    print(f"  QUESTION #{idx}")
    print(f"  {c(env._question, CYAN)}")
    print(f"  {c(f'GT: {env._answer}', GREEN)}")
    print(f"\n  Documents ({len(env._context)}):")
    for i, doc in enumerate(env._context):
        print(f"    [{i}] {doc['title']} ({len(doc['sentences'])} sentences)")
    print(f"{'─'*70}\n")

    # 运行 MCTS
    t0 = time.time()
    root = mcts_search_verbose(env, n_branches=n_branches, max_iters=max_iters)
    wall = time.time() - t0

    # 分析结果
    best_ans, best_reward, best_depth = _extract_best_answer(root)

    print(f"\n{'█'*70}")
    print(f"  RESULTS")
    print(f"{'█'*70}")
    print(f"  Wall time:        {wall:.1f}s")
    print(f"  Best answer:      {c(best_ans[:80] if best_ans else '(none)', CYAN)}")
    print(f"  Ground truth:     {c(env._answer, GREEN)}")
    print(f"  Best reward:      {best_reward}")
    print(f"  Best depth:       {best_depth}")
    print(f"  Correct:          {c('YES ✓', GREEN) if best_reward == 1 else c('NO ✗', RED)}")

    # 所有 Finish 尝试（按 reward 排序）
    all_finishes = []
    def _collect(n):
        if n.action.startswith("Finish["):
            ans = n.action[len("Finish["):-1] if n.action.endswith("]") else n.action[7:]
            all_finishes.append((n.id, n.depth, ans[:80], n.reward, n.rollout_value))
        for c in n.children:
            _collect(c)
    _collect(root)
    all_finishes.sort(key=lambda x: (x[3], x[4]), reverse=True)

    if all_finishes:
        print(f"\n  {c('All Finish attempts:', BOLD)} ({len(all_finishes)} total)")
        for nid, depth, ans, r, rv in all_finishes:
            mark = c("✓", GREEN) if r == 1 else c("✗", RED)
            print(f"    N{nid} d={depth} [{mark}]: {ans}")
    else:
        print(f"\n  {c('WARNING: No Finish actions were attempted!', YELLOW)}")

    # 树统计
    all_nodes = []
    def _collect_all(n):
        all_nodes.append(n)
        for c in n.children:
            _collect_all(c)
    _collect_all(root)

    terms = [n for n in all_nodes if n.is_terminal]
    searches = [n for n in all_nodes if "Search" in n.action]
    lookups = [n for n in all_nodes if "Lookup" in n.action]

    print(f"\n  {c('Tree stats:', BOLD)}")
    print(f"  Nodes: {len(all_nodes)}  (Search:{len(searches)} Lookup:{len(lookups)} "
          f"Finish:{len(all_finishes)} Term:{len(terms)})")
    print(f"  Max depth: {max(n.depth for n in all_nodes)}")
    print(f"  Solved terminals: {sum(1 for n in terms if n.reward == 1)}")

    # 详细树
    print(f"\n{'─'*70}")
    print(f"  DETAILED TREE")
    print(f"{'─'*70}")
    print_tree_detail(root)

    print()


if __name__ == "__main__":
    main()
