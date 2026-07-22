# F2 · Prompt / 上下文压缩（LongLingua）

> 状态：**Phase 1 完成**（中间件 + 隔离 venv 子进程后端 + 单测 + 端到端冒烟通过）。
> 待办：真机三档 ablation（`baseline` / `prefix_cache` / `f2-compress`），盯 success_rate ≤2pp 红线。
> 分支：`feat/f2-prompt-compress`

## 1. 做什么

在 ReAct agent 发请求给引擎**之前**，把**冷历史**（旧对话轮、旧工具结果）压短，
system prompt 和热尾（最新几条）原样保留。更短的 prompt → 更少的 prefill 与 KV 显存。
挂在缝D 中间件 `transform_messages`，纯 agent 层、与硬件无关。

## 2. 为什么选 LongLingua（而非原版 LLMLingua）

agent 会话是**长上下文**场景，核心失败模式是 **lost-in-the-middle**（早期重要信息被埋在中间）。
LongLingua 比 LLMLingua 多四件事，正中此痛点：

- **question-aware**（`condition_in_question="after"`）：按"和当前问题的相关性"决定留哪些 token，
  更可能保住工具结果里的关键事实（订单号、价格、电池）→ 护住成功率红线。
- **分段动态压缩率**（`dynamic_context_compression_ratio`）：冗余段压狠、关键段压轻。
- **邻近去重**（`condition_compare`，0.2.2 下默认关，ablation 再试）。
- **重排**（`reorder_context`，agent 轨迹默认 `"original"` 保时序）。

**方案 A**：单一方法（longllmlingua）+ `trigger_tokens` **压/不压门**——冷历史太短就放行
（短上下文无 lost-in-the-middle，压了白费延迟）。不是"两种方法分场景"。

## 3. 关键障碍：llmlingua ↔ transformers 5.x 冲突（已解决）

- `llmlingua==0.2.2`（PyPI 最新，2024 后停更）只兼容 **transformers 4.x**（依赖 legacy 格式的
  `past_key_values`）。其 PPL 路径有 5 处 `for k,v in past_key_values`，在 5.x 的 `Cache` 对象上全崩。
- 项目主 venv 的 **transformers 5.14.1** 被 `vllm 0.22.1` / `vllm_ascend` / `lmcache` 锁定，**不能降级**。
- 没有适配 5.x 的新版 llmlingua。

**解法：隔离压缩 venv + 子进程**。把真压缩放进一个独立的 `.venv-compress`
（transformers 4.43.4 + llmlingua + torch-cpu），`CompressMiddleware` 通过常驻子进程调它。
主 venv 的 transformers 5.x 一点不动。模型只加载一次、跨多次压缩复用。

## 4. 架构

```
f2-compress.yaml (options.compress: {method, rate, trigger_tokens, backend, worker_venv, ...})
        │ build_middlewares(["compress"], options)
        ▼
CompressMiddleware(BaseMiddleware)            ← middleware/compress.py
   ├─ transform_messages(): 三段切分 → 触发门 → 压缩冷历史 → 重建
   ├─ _compress_cold(): 调 compress_prompt(...)（与后端无关的统一调用）
   └─ _get_compressor(): backend=subprocess → _SubprocessCompressor
                              │ stdin/stdout JSON
                              ▼
                  _compress_worker.py（在 .venv-compress 里跑）   ← middleware/_compress_worker.py
                              │ 透明代理 PromptCompressor.compress_prompt
                              ▼
                  llmlingua.PromptCompressor(gpt2) → 压缩
```

`_compress_worker.py` 是 `compress_prompt` 的**透明 RPC 代理**（原样转发 args/kw），
因此 `_compress_cold` 不区分后端、无分支重复。

### transform_messages 三段切分（tool_call 配对安全）

- `system` 消息：原样保留（不压）。
- `question` = 最近一条带内容的 `user` 消息（LongLingua 相关性锚点；无则退化用最近非空消息）。
- **冷历史**：中间消息，整段压成**一条**文本消息——冷的 assistant `tool_calls` 与 tool 结果
  一起进文本，互不残留引用。
- **热尾**：最后 `keep_hot` 条，原样保留；边界 **snap**——若热尾从 `role=tool` 起，
  自动把 caller assistant 拉进热尾，保证 `tool_call→tool` 组完整，**绝不留孤立 tool**（→ 不会 400）。

## 5. 配置（`agent-mem/configs/f2-compress.yaml`）

```yaml
middleware:
  active: [compress]
  options:
    compress:
      method: longllmlingua          # llmlingua | longllingua | llmlingua2（ablation 可切）
      rate: 0.4                      # 保留比例（0.4≈2.5× 压缩）
      trigger_tokens: 4000           # 冷历史 < 此值不压（压/不压门）
      keep_hot: 6                    # 热尾条数（自动 snap 到完整 tool 组）
      device: cpu                    # 压缩器小模型放 CPU，NPU 让给主 LLM
      model_name: gpt2               # 小模型(开发用)；生产可换 Llama-2-7b-hf 提质量
      backend: subprocess            # subprocess(默认) | inprocess(仅隔离 venv 自身跑时)
      worker_venv: /data/os_competition_TSJ/.venv-compress/bin/python
      worker_script: ""              # 空=用随包的 agent_mem/middleware/_compress_worker.py
      # LongLLMLingua 专属（枚举对齐 llmlingua 0.2.2）
      condition_in_question: after   # none | before | after
      dynamic_context_compression_ratio: 0.3
      condition_compare: false        # 0.2.2 下 True 会改 condition 值致分发失配
      reorder_context: original       # original | two_stage
```

> ⚠️ `worker_venv` 是机器本地路径，部署到别处需改。`model_name=null` 会下 13GB 的
> Llama-2-7b-hf，开发用 `gpt2`（~500MB）。

## 6. 搭建隔离压缩 venv（一次性，机器本地）

```bash
uv venv /data/os_competition_TSJ/.venv-compress
uv pip install --python /data/os_competition_TSJ/.venv-compress/bin/python \
  "transformers==4.43.4" llmlingua \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cpu
# 结果：transformers 4.43.4 + torch 2.x cpu + llmlingua 0.2.2
```

`.venv-compress/` 与 `huggingface_home/` 都是机器本地，**不应进 git**（建议加进 `.gitignore`）。

## 7. 运行 / 验证

```bash
# 单测（fake compressor，不需真模型；在主 venv 跑）
PYTHONPATH=agent-mem/src .venv/bin/python -m pytest agent-mem/tests/test_middleware.py -q

# 端到端冒烟（真压缩 + stub server + openai SDK；worker 跑在 .venv-compress）
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/data/os_competition_TSJ/huggingface_home \
PYTHONPATH=agent-mem/src .venv/bin/python <smoke_script>.py
```

冒烟实测（gpt2/cpu，rate=0.3，retail 长历史场景）：**chars 缩短 87%**（3125→408），
question-aware 可见（battery 相关信息被保留，价格/颜色被压掉），run_react 经 openai SDK 无 400。

## 8. 后续 / 注意

- **latency 预算**：cpu gpt2 单次压缩约 6–12s。真机 ablation 须确认"省下的 prefill" > "压缩耗时"，
  长会话才净赚（这正是 F2 卖点场景）。可优化：增量压缩、`trigger_tokens` 调高只压真长历史。
- **rate 调参**：演示用 0.3（激进、lossy）只为显眼；实际在 0.4–0.6 间 ablation，
  权衡压缩率 vs success_rate ≤2pp。
- **生产模型**：gpt2 压缩质量弱，正式评测可换 `NousResearch/Llama-2-7b-hf`（13GB，质量更好但更慢）。
- **`condition_compare=True` + `reorder_context=two_stage`** 作为 ablation 单独试，可能进一步提升质量。
