# F5 — 并发场景下的动态资源回收（Priority 调度）

> 主责：P1 | 缝：E | 方案：利用 vllm 内置 priority 调度，不改源码

## 机制

vllm V1 scheduler 支持 `SchedulingPolicy.PRIORITY`：HBM 满时优先踢
低优先级（priority 数值大）的请求。我们利用这个内置机制实现简单的
动态资源回收——活跃 session 设 priority=0（受保护），idle session 设
priority=100（可被踢）。

并发 benchmark 下 HBM 压力增加 → scheduler 自动踢低优请求释放 block →
活跃 session 不受影响。

## 实现

3 个改动，均不改 vllm 源码：

| 文件 | 改动 |
|---|---|
| `vllm_server.py` | 引擎启动加 `--scheduling-policy priority` |
| `tau_bench_agent.py` | `__init__` 加 `priority` 参数，透传到 `extra_body` |
| `qwen_agent.py` + adapter | priority 从 runner → run_task → agent 透传 |

## 验证方案

1. 起引擎（priority 调度已内置在 build_serve_args）
2. 跑并发 benchmark（n_sessions=4），活跃 session priority=0，idle session priority=100
3. 降低 `--gpu-memory-utilization 0.3` 制造 HBM 压力
4. 观察：低优请求被 preempt，高优请求不受影响
5. 记录 mem_peak + 每个 session 的 success rate

## 引擎启动

```bash
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen2.5-7B-Instruct --port 8000 \
    --scheduling-policy priority \
    --gpu-memory-utilization 0.3 --max-model-len 32768
```

## 并发验证

```python
# 活跃 session：priority=0（受保护）
runner_active = QwenAgentRunner(engine_url=..., priority=0)
# idle session：priority=100（可被驱逐）
runner_idle = QwenAgentRunner(engine_url=..., priority=100)
```
