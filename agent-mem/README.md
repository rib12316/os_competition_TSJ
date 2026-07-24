# agent-mem

赛题 14 的实现主体 —— 面向智能体推理的内存管理优化系统。

架构（详见 `docs/技术设计方案.md`）：MVP 基础地基 + 5 个 agent 内存特征 ↔ 5 个优化机制的映射。

## 包结构（与优化模块对应）

| 子包 | 对应模块 | 职责 |
|---|---|---|
| `agent_mem/server/` | M5 / MVP | vLLM / SGLang / vLLM-Ascend 启动封装 |
| `agent_mem/agent/` | MVP | Qwen-Agent 包装层（ReAct + 多轮工具调用） |
| `agent_mem/middleware/` | M2 / M6 | Prompt 压缩 / 工具数据 lazy-load |
| `agent_mem/scheduler/` | M3 / M10 | session-aware 调度 / KV checkpoint 恢复 |
| `agent_mem/kv/` | M1 / M4 / M9 | KV 量化 / LMCache 分层 / 分布式 KV 池 |

F3 tool-result lazy-load is implemented in `middleware/lazyload.py`: large tool
results are stored outside model history and retrieved through the bounded local
`fetch_tool_result` tool. JSON arrays support bounded exact/case-insensitive/contains
field selection, stable match pointers, and token-capped previews. The inference
backend remains vLLM/vLLM-Ascend.

## 开发

```bash
# 从仓库根目录（环境在 ./.venv）
uv pip install -e agent-mem[dev]
pytest -q
ruff check src tests
```

Benchmark：`python -m agent_mem.benchmarks.runner --config configs/optimized.yaml`（三档对照见 `configs/`）。

F3 deterministic comparison:

```bash
PYTHONPATH=src python benchmarks/f3_long_tool_benchmark.py \
  --model Qwen2.5-7B-Instruct --results 3 --result-tokens 5000

PYTHONPATH=src python benchmarks/f3_retrieval_quality_benchmark.py \
  --records 360 --variants baseline f3 f2_f3
```
