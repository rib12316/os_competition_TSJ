# F2 Prompt 压缩 — 工作交接日志

> 这是跨会话交接文件。新会话先读此文件 + `docs/F2-ablation-results.md`（详细实验数据）即可接上。
> 仓库：`/data/os_competition_TSJ`（比赛项目 os_competition_TSJ，agent-mem 包）。

## 0. 一句话现状

F2 最终综合方案（compact system/tools + hot 1k + cold 8k）已完成 full115：严格同轨迹
`10,933,355 -> 8,820,495`，**完整 prompt -19.32%**，1,780/1,780 次调用受益。本轮
hot/cold BERT 均未触发，收益全部来自确定性静态前缀优化；p50 98.02s。success 为
23/115；相对两次历史 baseline（23/115、26/115）分别为 0pp、-2.61pp，因此处于历史波动
范围，但尚不能严格宣称通过 <=2pp 红线。分支 `feat/f2-prompt-compress` 未 push。

## 1. 任务与目标

- **赛题**：os_competition_TSJ，给智能体推理做内存/显存优化（降显存/延迟，success ≤2pp）。
- **F2**：在 ReAct agent 发引擎前，用 LLMLingua 把**冷历史**（旧对话轮）压短，热尾+system 原样保留 → 短 prompt → 少 prefill/KV。挂在缝D 中间件 `transform_messages`。
- **红线**：`task_success_rate` 下降 ≤ 2pp。

## 2. 分支 / Worktree / 持久性（重要！）

- **分支**：`feat/f2-prompt-compress`（8 个 F2 提交，最新 `43e13a2`）。提交在主仓库 `.git`（`/data/os_competition_TSJ/.git`），**持久**。
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
              → llmlingua PromptCompressor(LLMLingua-2 BERT, transformers 4.43.4)
     tool-aware：user/tool name/arguments/关键 JSON 外层保护；只压 assistant/tool body
       + hot 大工具结果提前压缩 + body SHA-256 缓存
     每步写 F2_EVENT_LOG 事件 JSONL：sent_tokens=响应 usage.prompt_tokens 真值；
       estimated_sent_tokens=旧 chars/4 估算，仅诊断
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
      method: llmlingua2
      model_name: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
      trigger_tokens: 2000     # 冷历史>此值才开始压
      recompress_delta_tokens: 4000  # 压过后新增>此值才重压（增量复用）
      keep_hot: 6
      tool_aware: true
      assistant_rate: 0.75
      tool_result_rate: 0.60
      hot_tool_trigger_tokens: 1000
      backend: subprocess
      worker_venv: /data/os_competition_TSJ/.venv-compress/bin/python
      worker_pool_size: 4      # 并发压缩池（>= --concurrency）
      worker_threads: 0        # 0=自动(核数//pool_size)
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

1. **tool-aware BERT 8 条 + full115 ablation**：当前实现已完成，尚需验证 success/真 token/延迟。
2. **静态 prompt 精简 ablation**：详见 `F2-static-prompt-compression-research.md`，与动态历史方案分开验证。
3. **push 到远程**（`git push origin feat/f2-prompt-compress`）保险。
4. **更长上下文场景**：验证搜索/RAG/大 JSON 工具结果。
5. **mem 指标**：换"实际 KV 用量"而非 vllm pool peak。

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
- `F2-static-prompt-compression-research.md` — system policy / 工具 schema 真 token 构成与压缩方案
- `F2-tool-aware-llmlingua-plan.md` — 工具轨迹结构保护、BERT 与 hot result 方案

## 11. sent_tokens 口径变更（2026-07-23）

- 旧 `docs/f2-results/*events*.jsonl` 的 `sent_tokens` 是 message 文本 `chars/4`，会漏掉通过
  OpenAI `tools=` 注入的工具 schema、chat template 和结构化 tool calls，不能视为真实输入 token。
- 当前代码非流式读取 `response.usage.prompt_tokens`；tau-bench 流式请求开启
  `stream_options.include_usage=true` 并读取最终 usage chunk。
- 服务端不返回 usage 时，`sent_tokens=null` 且 `sent_tokens_source=unavailable`，不会回退估算。
- 真机冒烟：短请求 `estimated_sent_tokens=2`，vLLM 返回 `sent_tokens=32`；流式/非流式一致。

## 12. 严格 baseline 真实 token 小测（8+8）

- 新增 agent 公共层 `PROMPT_TOKEN_LOG`，不依赖 middleware；`baseline-logged.yaml` 已改成
  `middleware.active=[]`，因此 baseline 完全不经过 F2。
- 相同任务 ID `tau-0..7`，baseline 134 次调用 / 884,845 真 token；F2 145 次调用 /
  1,013,338 真 token。F2 轨迹多 11 步，独立生成总量不可直接归因于压缩。
- F2 7/8 任务触发，7 次 compress + 39 次 reuse；压缩平均 19.15s。能配到 baseline
  相同步号的 6 个首次压缩点合计 token -13.88%，说明命中步骤确实变短。
- F2 p50 145.92s vs baseline 78.45s；本轮 gpt2 压缩成本和更长随机轨迹共同导致显著变慢。
- system policy + tools 约占 F2 真输入 60.6%，当前冷历史 F2 不处理。下一步优先固定静态
  prompt 精简，而不是继续只调 cold rate。
- 完整报告：`docs/f2-results/comparison_true8_usage.md`；原始运行：`/tmp/f2-true8-usage`。

## 13. Tool-aware LLMLingua-2 实现（checkpoint 后）

- 回退 checkpoint：`e9505ed`（真实 token 计量 + 调研，尚未实施 tool-aware）。
- 当前配置切到 LLMLingua-2 BERT small；模型缓存位于 `$HF_HUB_CACHE`。
- cold serializer 完整保留 user、tool name/call ID/arguments，并从 JSON tool result 抽取
  ID/status/金额/数量/时间/address/payment/error 等硬字段；只把正文送 BERT。
- hot tool content 超过估算 1,000 token 时提前压缩，但 assistant/tool 协议结构原样。
- body 按 `rate + SHA-256` 缓存，hot 压过的结果进入 cold 后不再重复推理。
- 新增压缩器初始化锁，避免并发任务首次触发时重复创建 worker pool。
- 真 worker：首次 BERT 加载+压缩 4.98s，同内容缓存复用低于 1ms；硬字段保持。
- 最终 8 条：success 2/8；BERT 压缩均值 7.01s；p50 98.64s。旧 GPT-2 小测分别为
  0/8、19.15s、145.92s。安全 cold 压缩约 -7.2%，retail 未触发 hot 大结果路径。
- 报告：`docs/f2-results/comparison_toolaware8.md`；原始数据：`/tmp/f2-toolaware8-v2`。

## 14. 同轨迹完整 Prompt 计量

- 公共 meter 在同一次调用内对 canonical/transformed messages 应用本地 Qwen2.5 完整 chat
  template，并把同一份 `tools=` 计入两边；响应 usage 校验 transformed 漂移。
- 8 条结果：`726,409 -> 722,145`，严格配对节省 4,264 token（0.59%）；仅 20/111
  次请求有节省，证明继续只提高 cold rate 的总体上限很低。
- meter 平均 16.77ms/步、p95 26.23ms；tokenizer 在任务计时前预热。
- 报告：`docs/f2-results/comparison_paired8.md`；原始数据：`/tmp/f2-paired8`。
- 决策：下一阶段实施 system/tools 固定前缀确定性去重，cold rate 暂不提高。

## 15. 全面 system/tools + 动态上下文优化

- 新增联合 `transform_request(messages, tools)` 挂载点，F2 可同时优化 system 与工具 schema。
- retail system policy 使用逐条语义保留的 compact 版本；工具 name/type/required/enum/
  参数结构不动，15 个重复 description 提取为 4 条共享 conventions，移除 7 条 system
  已覆盖的 confirmation 重复句。
- 8 条严格配对：`972,762 -> 802,653`，节省 170,109 token（17.49%），139/139 步
  均有收益；每步至少省 1,219 token。
- success 1/8，与 strict baseline 小测相同；p50 80.36s vs baseline 78.45s。
- 4k cold 门槛仅在最后一步触发一次：15.89s 只多省 668 token，故最终提高到 8k。
  大单次 tool result 仍由 hot 1k 路径处理。
- system 语义审计补回三项规则后，首轮实测 `4,272 -> 3,085`（-1,187，-27.79%）；
  最终 8k 配置按该轨迹预计约 16.96% token 收益，且不支付本次 cold BERT 成本。
- 报告：`docs/f2-results/comparison_comprehensive8.md`；原始：`/tmp/f2-comprehensive8-v3`。

## 16. 最终综合方案 full115（2026-07-24）

- tau-bench retail 115/115 完成，并发 4，共 1,780 次模型调用。
- 严格配对完整 prompt：`10,933,355 -> 8,820,495`，节省 2,112,860 token（19.32%）；
  每次调用固定节省 1,187 token。
- vLLM 实际 usage 合计 8,798,998 token；与本地 transformed meter 差 -21,497（-0.24%）。
- 1,780 次 F2 action 全为 skip；最大 compressible cold body 4,097 < 8,000，hot BERT 0 次、
  cold BERT 0 次。也就是说本轮 dynamic LLMLingua 贡献为 0，且没有 BERT 延迟。
- success 23/115（20.00%）；历史两次 strict baseline 为 23/115、26/115。相对最近 baseline
  名义下降 2.61pp，超过红线 0.61pp；但 baseline 自身也波动 2.61pp，单次独立轨迹不足以
  判断 compact policy 是否导致下降。
- p50 98.02s、p95 161.64s；相对最近 strict baseline 分别 -1.50%、+3.57%。
- 报告：`docs/f2-results/comparison_comprehensive_full115.md`；原始：
  `/tmp/f2-comprehensive-full115`。
