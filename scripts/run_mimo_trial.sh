#!/usr/bin/env bash
# mimo user-sim 真样本试验：baseline + prefix-cache 两档，各 10 任务，再 compare。
# 用法（key 经环境变量传，不写进本文件）：
#   MIMO_KEY='tp-...' nohup bash scripts/run_mimo_trial.sh > logs-mimo/run.log 2>&1 &
#
# fire-and-forget：自动起/停两档引擎（PID/进程组管理，不用 pkill pattern），全程带时间戳日志。
# 引擎：Qwen2.5-7B-Instruct（本地 NPU，agent 侧，被测）。user-sim：mimo-v2.5-pro（外部 API）。

set -u
REPO=/data/os_competition_TSJ
PY=$REPO/.venv/bin/python
LOGROOT=$REPO/logs-mimo
USER_MODEL=mimo-v2.5-pro
USER_API_BASE=https://token-plan-cn.xiaomimimo.com/v1
mkdir -p "$LOGROOT"

ts() { date +"%H:%M:%S"; }
echo "[$(ts)] === mimo trial start (model=Qwen2.5-7B-Instruct, user-sim=$USER_MODEL, 10 tasks, max-steps 25) ==="

if [ -z "${MIMO_KEY:-}" ]; then
  echo "[$(ts)] ERROR: MIMO_KEY 未设置。用法: MIMO_KEY='tp-...' nohup bash scripts/run_mimo_trial.sh > logs-mimo/run.log 2>&1 &"
  exit 2
fi

ENGINE_PGID=""

kill_all_bench() {
  # 可靠清理：杀掉所有 vllm 引擎 + EngineCore + benchmark runner 残留。
  # [v]/[E]/[b] 正则技巧：避免 pkill 匹配到这条命令自身（本命令行是括号形式，正则匹配无括号形式）。
  pkill -9 -f "[v]llm.entrypoints" 2>/dev/null || true
  pkill -9 -f "[E]ngineCore" 2>/dev/null || true
  pkill -9 -f "[b]enchmarks/runner.py" 2>/dev/null || true
  sleep 6
}

start_engine() {
  # $1 = 引擎日志名；$2 = 可选额外 vllm 参数（如 --no-enable-prefix-caching）
  local logname=$1; local extra=${2:-}
  setsid $PY -m vllm.entrypoints.openai.api_server \
    --model "$REPO/models/Qwen2.5-7B-Instruct" \
    --port 8000 --host 0.0.0.0 --served-model-name Qwen2.5-7B-Instruct \
    --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 \
    $extra > "$LOGROOT/$logname" 2>&1 &
  ENGINE_PGID=$!
  echo "[$(ts)] [engine] starting ($logname, extra='$extra', pgid=$ENGINE_PGID)"
  for i in $(seq 1 120); do
    if $PY -c "import httpx; exit(0 if httpx.get('http://127.0.0.1:8000/health',timeout=3).status_code==200 else 1)" 2>/dev/null; then
      echo "[$(ts)] [engine] ready (~$((i*5))s)"; return 0
    fi
    if grep -qE "terminate called|RuntimeError: Engine core" "$LOGROOT/$logname" 2>/dev/null; then
      echo "[$(ts)] [engine] CRASHED — 见 $LOGROOT/$logname"; return 1
    fi
    sleep 5
  done
  echo "[$(ts)] [engine] TIMEOUT 未就绪"; return 1
}

stop_engine() {
  if [ -n "$ENGINE_PGID" ]; then
    kill -TERM "-$ENGINE_PGID" 2>/dev/null || true
    sleep 5
  fi
  kill_all_bench  # 兜底：清掉任何残留引擎/runner
}

# 启动前：杀掉所有残留引擎/runner + 清空旧 run 目录（干净对照）
kill_all_bench
rm -rf "$LOGROOT"/*_vllm_*_run* "$LOGROOT"/_summaries 2>/dev/null || true
echo "[$(ts)] [init] 残留已清理，logs-mimo 重置"

run_tier() {
  # $1 = tier 名（baseline|prefix_cache）；$2 = 引擎额外参数
  local tier=$1; local extra=${2:-}
  echo "[$(ts)] ===== TIER: $tier ====="
  stop_engine
  if ! start_engine "engine-$tier.log" "$extra"; then
    echo "[$(ts)] [run] $tier 引擎失败，跳过"; stop_engine; return
  fi
  $PY "$REPO/agent-mem/benchmarks/runner.py" \
    --config "$REPO/agent-mem/configs/$tier.yaml" \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root "$LOGROOT" \
    --user-model "$USER_MODEL" --user-api-base "$USER_API_BASE" \
    --user-api-key "$MIMO_KEY"
  echo "[$(ts)] [run] $tier 完成"
  stop_engine
}

# 两档：baseline 关 prefix cache，prefix_cache 用 V1 默认
run_tier baseline "--no-enable-prefix-caching"
run_tier prefix_cache ""

echo "[$(ts)] ===== compare ====="
$PY "$REPO/agent-mem/benchmarks/runner.py" --compare --study mimo-trial --log-root "$LOGROOT"
echo "[$(ts)] === DONE — 对照报告: $LOGROOT/_summaries/*_mimo-trial_comparison.md ==="
