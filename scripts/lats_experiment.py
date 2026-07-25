"""LATS+MCTS 多路径 KV 共享实验 — 完整指标采集版"""
import os, sys, time, json, subprocess, threading, re
from collections import defaultdict

REPO = "/data/os_competition_TSJ"
PY_LATS = f"{REPO}/.venv-lats/bin/python"
LATS_DIR = f"{REPO}/third_party/LanguageAgentTreeSearch/programming"
MODEL = "Qwen2.5-7B-Instruct"
N_TASKS, MAX_ITERS = 10, 3

GROUPS = {
    "A":  {"prefix_cache": False, "n": 2, "desc": "无共享基线"},
    "B":  {"prefix_cache": True,  "n": 1, "desc": "单分支基准"},
    "C1": {"prefix_cache": True,  "n": 2, "desc": "低分支"},
    "C2": {"prefix_cache": True,  "n": 4, "desc": "中分支"},
    "C3": {"prefix_cache": True,  "n": 8, "desc": "高分支"},
}


# ---- metrics 读取 ----
def curl_metrics():
    import urllib.request
    try:
        r = urllib.request.urlopen("http://localhost:8000/metrics", timeout=5)
        return r.read().decode()
    except Exception:
        return ""


def parse_counter(text, name):
    """Counter 类型：直接读值"""
    for line in text.split("\n"):
        if line.startswith(name + "{"):
            return float(line.split()[-1])
    return 0.0


def parse_histogram(text, name):
    """Histogram 类型：_sum/_count → 平均值"""
    total = count = 0.0
    for line in text.split("\n"):
        if line.startswith(name + "_sum{"):
            total = float(line.split()[-1])
        if line.startswith(name + "_count{"):
            count = float(line.split()[-1])
    return total / count if count > 0 else 0.0


def collect_metrics():
    """采集所有可用指标"""
    t = curl_metrics()
    return {
        # KV 缓存复用
        "kv_hit_rate": (
            parse_counter(t, "vllm:prefix_cache_hits_total") /
            max(parse_counter(t, "vllm:prefix_cache_queries_total"), 1)
        ),
        "prompt_tokens_cached": parse_counter(t, "vllm:prompt_tokens_cached_total"),
        "prompt_tokens_total": parse_counter(t, "vllm:prompt_tokens_total"),
        "kv_cache_usage_perc": parse_counter(t, "vllm:kv_cache_usage_perc"),
        # 推理效率
        "ttft_mean_s": parse_histogram(t, "vllm:time_to_first_token_seconds"),
        "tpot_mean_s": parse_histogram(t, "vllm:request_time_per_output_token_seconds"),
        "e2e_latency_mean_s": parse_histogram(t, "vllm:e2e_request_latency_seconds"),
        "request_count": parse_counter(t, "vllm:request_success_total"),
        "generation_tokens": parse_counter(t, "vllm:generation_tokens_total"),
    }


# ---- mem_peak 后台采样 ----
class MemSampler:
    def __init__(self):
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=5)
        return self.peak_mb

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["npu-smi", "info", "-t", "usages", "-i", "10", "-c", "0"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                for line in out.split("\n"):
                    if "HBM Usage Rate" in line:
                        pct = float(line.split(":")[-1].strip().replace("%", ""))
                        self.peak_mb = max(self.peak_mb, pct * 655.36)
            except Exception:
                pass
            time.sleep(1)


# ---- 引擎管理 ----
def start_engine(prefix_cache=True):
    cmd = [
        f"{REPO}/.venv/bin/python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", f"{REPO}/models/Qwen2.5-7B-Instruct",
        "--port", "8000", "--host", "0.0.0.0",
        "--served-model-name", MODEL, "--max-model-len", "32768",
    ]
    if not prefix_cache:
        cmd.append("--no-enable-prefix-caching")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import urllib.request
    for i in range(120):
        time.sleep(5)
        try:
            urllib.request.urlopen("http://localhost:8000/health", timeout=3)
            return proc
        except Exception:
            if proc.poll() is not None: raise RuntimeError("engine crashed")
    raise TimeoutError("引擎超时")


def stop_engine(proc):
    proc.terminate()
    try: proc.wait(timeout=15)
    except subprocess.TimeoutExpired: proc.kill()
    time.sleep(10)


# ---- LATS 运行 ----
def run_lats(group, n_branches):
    script = f"""
import os, sys, time
os.environ['OPENAI_API_KEY'] = 'stub'
import openai; openai.api_key = 'stub'; openai.api_base = 'http://localhost:8000/v1'
sys.path.insert(0, '{LATS_DIR}')
sys.path.insert(0, '{LATS_DIR}/human-eval')
from generators import model as gm, factory as gf
_o = gf.model_factory
def _p(n):
    from generators.model import GPTChat
    try: return _o(n)
    except ValueError: return GPTChat(n)
gf.model_factory = _p; gm.model_factory = _p
import mcts; mcts.model_factory = _p
from utils import read_jsonl
data = read_jsonl('{LATS_DIR}/human-eval/data/HumanEval.jsonl')[:{N_TASKS}]
t0 = time.monotonic()
mcts.run_mcts(dataset=data, model_name='{MODEL}', language='py',
              max_iters={MAX_ITERS}, pass_at_k=1,
              log_path='/tmp/lats_{group}_log.jsonl', verbose=False, n={n_branches})
print(f'LATS_WALL={{time.monotonic()-t0:.0f}}')
"""
    sp = f"/tmp/lats_{group}.py"
    with open(sp, "w") as f:
        f.write(script)
    r = subprocess.run([PY_LATS, sp], capture_output=True, text=True, timeout=7200)
    wall = 0
    for line in (r.stdout + r.stderr).split("\n"):
        if "LATS_WALL=" in line:
            wall = float(line.split("=")[1])
    return wall, r.returncode


def lats_success_rate(group):
    """从 LATS log 文件读成功率"""
    log_file = f"/tmp/lats_{group}_log.jsonl"
    if not os.path.exists(log_file):
        return 0
    try:
        with open(log_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        solved = sum(1 for l in lines if l.get("solution"))
        return solved / len(lines) if lines else 0
    except Exception:
        return 0


# ---- 主流程 ----
def main():
    results = []
    for name, cfg in GROUPS.items():
        print(f"\n{'='*50}")
        print(f"Group {name}: {cfg['desc']} (pcache={cfg['prefix_cache']}, n={cfg['n']})")

        # 起引擎 + mem sampler
        print("  starting engine...")
        proc = start_engine(prefix_cache=cfg["prefix_cache"])
        sampler = MemSampler()
        sampler.start()

        # before 指标
        before = collect_metrics()
        r_before = before["request_count"]

        # 跑 LATS
        print("  running LATS...")
        wall, rc = run_lats(name, cfg["n"])

        # after 指标
        after = collect_metrics()
        r_after = after["request_count"]
        mem_peak = sampler.stop()
        success = lats_success_rate(name)

        # 差量指标
        r = {
            "group": name, "desc": cfg["desc"],
            "prefix_cache": cfg["prefix_cache"], "n_branches": cfg["n"],
            "wall_s": wall,
            # KV 缓存
            "kv_hit_before": before["kv_hit_rate"],
            "kv_hit_after": after["kv_hit_rate"],
            "prompt_cached": after["prompt_tokens_cached"],
            "kv_cache_usage": after["kv_cache_usage_perc"],
            # 显存
            "mem_peak_mb": mem_peak,
            # 推理效率
            "ttft_ms": after["ttft_mean_s"] * 1000,
            "tpot_ms": after["tpot_mean_s"] * 1000,
            "e2e_latency_s": after["e2e_latency_mean_s"],
            "requests": r_after - r_before,
            "gen_tokens": after["generation_tokens"],
            # 任务
            "success_rate": success,
        }
        results.append(r)
        stop_engine(proc)

        # 打印
        print(f"  wall={wall:.0f}s success={success:.0%} kv_hit={after['kv_hit_rate']:.3f}")
        print(f"  mem_peak={mem_peak:.0f}MB ttft={r['ttft_ms']:.1f}ms tpot={r['tpot_ms']:.1f}ms reqs={r['requests']:.0f}")

    # 汇总表
    cols = ["group", "prefix_cache", "n_branches", "wall_s", "success_rate",
            "kv_hit_before", "kv_hit_after", "mem_peak_mb", "ttft_ms", "tpot_ms", "requests"]
    print(f"\n{'='*120}")
    header = f"{'Grp':<5} {'PC':>3} {'n':>3} {'wall':>7} {'succ':>6} {'kv_bef':>8} {'kv_aft':>8} {'mem':>8} {'ttft':>8} {'tpot':>8} {'reqs':>6}"
    print(header)
    print("-" * 120)
    for r in results:
        print(f"{r['group']:<5} {str(r['prefix_cache']):>3} {r['n_branches']:>3} "
              f"{r['wall_s']:>7.0f}s {r['success_rate']:>5.0%} "
              f"{r['kv_hit_before']:>8.3f} {r['kv_hit_after']:>8.3f} "
              f"{r['mem_peak_mb']:>8.0f} {r['ttft_ms']:>8.1f} {r['tpot_ms']:>8.1f} {r['requests']:>6.0f}")

    os.makedirs(f"{REPO}/logs-lats", exist_ok=True)
    with open(f"{REPO}/logs-lats/summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {REPO}/logs-lats/summary.json")


if __name__ == "__main__":
    main()
