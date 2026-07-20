# F1 — int8 KV Cache 量化

> 主责：P1 | 缝：A | 一行 flag，无额外代码

## 原理

vllm V1 内置 `int8_per_token_head` —— 每 token 每 head 动态算 scale，将 KV cache 从 float16/int8 → int8 存储，理论显存减半。纯在线量化，无需校准。

## 代码

| 文件 | 内容 |
|---|---|
| `configs/f1-int8.yaml` | `--kv-cache-dtype int8_per_token_head` |
| `configs/optimized.yaml` | 同上（三档最高档） |
| `vllm_server.py` | `extra_args` 经 `shlex.split` 传入 API server |

## NPU 探针

```bash
# 起引擎，看是否报错或不认该 flag
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen2.5-7B-Instruct \
    --port 8000 \
    --kv-cache-dtype int8_per_token_head \
    --max-model-len 4096
```

**分支**：
- ✅ 引擎正常启动 → int8 KV 生效，跑 benchmark 对比 mem_peak
- ❌ `invalid choice` / `unrecognized` → Ascend 后端不支持，走降级方案
- ❌ 启动成功但 mem_peak 不降 → flag 被静默忽略（no-op），走降级方案

## 降级方案

如果 `int8_per_token_head` 在 Ascend 上不生效：

1. **`--gpu-memory-utilization 0.7`** — 降低 KV cache pool，迫使引擎更激进回收，间接降低 mem_peak
2. **`--max-num-seqs 4`** — 限制并发数，控制 KV 总量
3. 叙事改为"调度参数调优降低 Agent 推理显存占用"

## Benchmark

```bash
# baseline
.venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/baseline.yaml \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root logs-mimo \
    --user-model mimo-v2.5-pro --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"

# F1
.venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/f1-int8.yaml \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root logs-mimo \
    --user-model mimo-v2.5-pro --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"
```
