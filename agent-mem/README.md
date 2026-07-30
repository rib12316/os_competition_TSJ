# agent-mem

`agent-mem` 是项目的 Python 实现主体，提供引擎配置、Agent loop、上下文中间件、KV 管理、会话调度、Benchmark 和 Gradio 演示界面。完整作品简介、Ascend 环境要求和运行步骤见仓库根目录 [`README.md`](../README.md)。

## 包结构

| 路径 | 职责 |
|---|---|
| `src/agent_mem/server/` | vLLM/vLLM-Ascend 配置翻译与生命周期管理 |
| `src/agent_mem/agent/` | ReAct、多轮工具调用与 tau-bench Agent |
| `src/agent_mem/middleware/` | F2 Prompt 压缩、F3 工具数据外置 |
| `src/agent_mem/kv/` | F1 C8、F4 LMCache/KV connector |
| `src/agent_mem/scheduler/` | F5 session 状态、准入控制和并发驱动 |
| `src/agent_mem/bench/` | Runner、任务适配、指标与运行目录 |
| `src/agent_mem/demo/` | Gradio 界面、引擎控制和实时监控 |
| `configs/` | baseline、专项功能和组合 preset |
| `tests/` | 不依赖在线模型的单元与构造测试 |

## 开发验证

从仓库根目录安装，再进入本目录运行检查：

```bash
python -m pip install -e "agent-mem[dev,demo]" qwen-agent
cd agent-mem
python -m pytest tests -q
python -m ruff check src tests
python benchmarks/runner.py --config configs/prefix_cache.yaml --runner dry-run --runs 1
```

真实 tau-bench、LongBench、C8 和 LMCache 运行需要额外数据、模型和 Ascend 依赖。相关配置与限制见根 README 和 `../dev-guide/`，不要把本地模型、缓存、密钥或运行日志加入版本库。
