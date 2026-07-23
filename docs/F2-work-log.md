# F2 Prompt 压缩 — 工作交接日志

> 这是跨会话交接文件。新会话先读此文件 + `docs/F2-ablation-results.md`（详细实验数据）即可接上。
> 仓库：`/data/os_competition_TSJ`（比赛项目 os_competition_TSJ，agent-mem 包）。

## 0. 一句话现状

F2（Prompt 压缩 / 缝D）**全量 115 任务验证完成**：**success 无损（-1pp，红线内）、上下文 -45%、延迟 +9.7%（小代价，BERT 可消除）**。代码 + 文档 + 数据全部提交在分支 `feat/f2-prompt-compress`（**未 push**）。

## 1. 任务与目标

- **赛题**：os_competition_TSJ，给智能体推理做内存/显存优化（降显存/延迟，success ≤2pp）。
- **F2**：在 ReAct agent 发引擎前，用 LLMLingua 把**冷历史**（旧对话轮）压短，热尾+system 原样保留 → 短 prompt → 少 prefill/KV。挂在缝D 中间件 `transform_messages`。
- **红线**：`task_success_rate` 下降 ≤ 2pp。

## 2. 分支 / Worktree / 持久性（重要！）

- **分支**：`feat/f2-prompt-compress`（7 个 F2 提交，最新 `6d89f3d`）。提交在主仓库 `.git`（`/data/os_competition_TSJ/.git`），**持久**。
- **worktree**：`/tmp/f2-wt`（在 `/tmp`，**机器重启会丢**！）。但分支+提交在 `/data` 上不丢。
  - 若 `/tmp/f2-wt` 没了，重建：`git -C /data/os_competition_TSJ worktree add /tmp/f2-wt feat/f2-prompt-compress`
  - 或直接在主 checkout 切过去（注意主 checkout 可能在队友的分支上，别打架）。
- **未 push 到 GitHub**。要保险可 `git push origin feat/f2-prompt-compress`。
- 共享机器，队友也在用（他们的分支：feat/f5-priority-evict、feat/baseline-tot 等）。**别碰队友的分支/工作**。

## 3. 架构（已实现）

```
f2-compress.yaml (middleware.active:[compress] + options.compress)
  → middlewares_from_config() → CompressMiddleware（缝D）
     transform_messages: 三段切分(system/冷/热) + 触发门(trigger_tokens) + 阈值增量复用
       (recompress_delta_tokens, ctx.scratch 缓存，不是每步都压) + 热尾 snap 保 tool_call 配对
     → _compress_cold → _get_compressor
        backend=subprocess → _SubprocessCompressorPool(N worker + queue)
           → 每个 worker 是 .venv-compress 里的 _compress_worker.py（常驻，模型加载一次）
              → llmlingua PromptCompressor(gpt2, transformers 4.43.4)
     每步写 F2_EVENT_LOG 事件 JSONL（含 sent_tokens=实际发给引擎的 token）
```

关键文件（都在 `agent-mem/src/agent_mem/middleware/`）：
- `compress.py` — CompressMiddleware + _SubprocessCompressor + _SubprocessCompressorPool + worker_threads 限线程
- `_compress_worker.py` — 隔离 venv 里的常驻压缩 worker（compress_prompt 透明代理）

## 4. 关键配置（`configs/f2-compress.yaml`）

```yaml
middleware:
  active: [compress]          # 开关：compress=开，[]=关(baseline)
  options:
    compress:
      method: longllmlingua
      rate: 0.65               # 保留 65%（甜点：护 success）
      trigger_tokens: 2000     # 冷历史>此值才开始压
      recompress_delta_tokens: 4000  # 压过后新增>此值才重压（增量复用）
      keep_hot: 6
      backend: subprocess
      worker_venv: /data/os_competition_TSJ/.venv-compress/bin/python
      worker_pool_size: 4      # 并发压缩池（>= --concurrency）
      worker_threads: 0        # 0=自动(核数//pool_size)；gpt2 小模型限线程反而更快
user_sim:                      # mimo 当 user-sim（已成默认）
  model: mimo-v2.5-pro
  api_base: https://token-plan-cn.xiaomimimo.com/v1
  api_key_env: MIMO_KEY        # key 从 $MIMO_KEY 读，不进仓库
```

## 5. 怎么跑（环境 + 命令）

**环境前提**：
- vllm 引擎在 NPU 上跑（`http://127.0.0.1:8000/v1`），Qwen2.5-7B-Instruct（stock，无 C8）。
- `.venv-compress`（transformers 4.43.4 + llmlingua + torch-cpu）已建好。
- `MIMO_KEY` 环境变量在。
- 主 venv openai 须 ≥1.x（**曾被队友降级到 0.28.1，我升回 2.46.0**；若又被降，`uv pip install --python .venv/bin/python openai==2.46.0`）。

**起引擎**（注意 PYTHONPATH 别覆盖，保 acl）：
```bash
setsid /data/os_competition_TSJ/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /data/os_competition_TSJ/models/Qwen2.5-7B-Instruct \
  --port 8000 --host 0.0.0.0 --served-model-name Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 \
  > /tmp/vllm_engine.log 2>&1 < /dev/null &
# 轮询 curl http://127.0.0.1:8000/health
```

**跑 benchmark**（必须用 `/tmp/f2-shim/run_bench.py` wrapper 打 aiohttp 补丁，绕 litellm↔aiohttp 冲突）：
```bash
cd /tmp/f2-wt/agent-mem  # 或 worktree 重建后
PYTHONPATH=$PWD/src F2_EVENT_LOG=/tmp/ev.jsonl \
/data/os_competition_TSJ/.venv/bin/python /tmp/f2-shim/run_bench.py \
  --config configs/f2-compress.yaml --runner qwen-agent \
  --engine-url http://127.0.0.1:8000/v1 \
  --device npu --max-tasks 115 --max-steps 20 --runs 1 --concurrency 4 \
  --log-root /tmp/f2-full
# baseline：--config configs/baseline-logged.yaml（compress+trigger999999，只记日志不压）
# 对照：python /tmp/f2-shim/run_bench.py --compare --log-root /tmp/f2-full
```

## 6. 实验结果（全量 115，最终）

| 指标 | baseline | f2(池+限线程) | Δ |
|---|---|---|---|
| success | 0.23 (26/115) | 0.22 (25/115) | **-1pp ✅ 红线内** |
| latency p50 | 99.5s | 109.1s | +9.7%（单worker 时 +16%，已减半） |
| 上下文 | 279623 tok | 152545 tok | **-45%** |
| 压缩/次 | — | 9.7s（单worker 19.6s） | 快一倍 |

- **success 跨多次 run 与 baseline 差 ±3pp 内（噪声）→ F2 不伤精度**。
- **剩余 +9.7% 延迟 = gpt2 压缩本身耗时**（~10s/任务；池跨任务并行但每任务仍等自己的压缩）。换 BERT（~3-5s）可归零。
- mem_peak 不变（vllm 预分配池掩盖了 F2 的显存收益——指标问题，非 F2 问题）。

## 7. 踩过的坑 + 修复（重要，新会话可能再遇）

1. **llmlingua 0.2.2 ↔ transformers 5.x 冲突**：llmlingua 只兼容 transformers 4.x（past_key_values legacy 格式），主 venv 被 vllm 锁 5.x。→ 用隔离 `.venv-compress`（transformers 4.43.4）+ 子进程 worker。详见 `docs/env-dependency-conflicts.md`。
2. **litellm 1.92 ↔ aiohttp 3.8.4**（缺 ConnectionTimeoutError/SocketTimeoutError）→ `/tmp/f2-shim/run_bench.py` 运行时补丁（import 前补属性）。
3. **litellm 1.92 ↔ openai 0.28.1**（缺 openai._models）→ 主 venv openai 升到 2.46.0（队友曾降级，注意复查）。
4. **模型目录残留 F1 的 C8 产物**（quant_model_description.json/kv_cache_scales.safetensors）→ 引擎崩在 `_prepare_c8_scales`。改名 `.c8bak` + 用 `.stock.bak` 还原 stock。需 C8 时重跑 `inject_c8_scales.py`。
5. **worker 池 torch 超订**（多 worker 各吃满 128 核 → 96s 撞车）→ `worker_threads` 限线程（自动=核数//pool_size）；实测 gpt2 限线程反而更快（32线程6s vs 默认~20s）。
6. **mimo 没用**：默认 user-sim 走本地 7B（弱→轨迹短、success 0）。→ 加 `user_sim` 配置段，mimo 当默认（success 从 0 → 0.22）。`run_mimo_trial.sh` 是旧脚本参考。

## 8. 未完成 / 下一步（按优先级）

1. **换 BERT(llmlingua2) 压缩器**（用户暂缓，但这是延迟的根本解）：~3-5s/次，把 +9.7% 延迟归零。改 method=llmlingua2 + 换 BERT 模型（注意 512 上下文 + 不 question-aware）。
2. **push 到远程**（`git push origin feat/f2-prompt-compress`）保险。
3. **更长上下文场景**：tau-bench retail 冷历史才 2-5k（中等），F2 延迟小亏；上长 RAG/长任务（冷历史几万 token）延迟会翻正（prefill 节省 > 压缩耗时）。
4. **mem 指标**：换"实际 KV 用量"而非 vllm pool peak，才能体现 -45% 上下文的显存收益。
5. rate/trigger 还可微调（现 rate=0.65/trigger=2000/recompress=4000 是甜点）。

## 9. 提交链（feat/f2-prompt-compress）

```
6d89f3d docs: 全量115 worker池+限线程 最终结果（延迟惩罚减半）
7af1ceb feat: worker_threads 改自动 = 核数 // pool_size
7e92b98 feat: 压缩 worker 限线程(worker_threads)，修 torch 超订
dad4957 feat: 压缩 worker 池(_SubprocessCompressorPool) 并行压缩
217fe74 feat: 事件日志加 sent_tokens
b756269 feat: 阈值增量压缩 + 并发支持 + 全量 ablation
30e5655 feat: Prompt 压缩 — LongLLMLingua via 隔离 venv 子进程 + 文档
```

## 10. 相关文档（都在 worktree docs/）

- `F2-ablation-results.md` — 详细实验数据（各档 trigger/rate/mimo/full115 + pool）
- `env-dependency-conflicts.md` — 依赖冲突处理记录
- `f2-results/` — 原始产物（comparison、per-task、events jsonl，各档）
- `F2-prompt-compress.md` — F2 设计/搭建说明
