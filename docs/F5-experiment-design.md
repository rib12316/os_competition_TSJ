# F5 实验设计 — 并发场景动态资源回收与重分配

## 实验目标

在并发 τ-bench 场景下，验证 AdmissionController + idle 驱逐相比 vllm
原生的随机踢人，在吞吐、延迟、内存、成功率上的优势。

## 实验组

三组对照，每组起一次引擎、跑一次 benchmark：

| 组 | 引擎配置 | 驱逐策略 | 并发度 |
|---|---|---|---|
| **A：vllm 原生** | FCFS（默认） | pool 满随机踢 | 固定 6 |
| **B：vllm priority** | `--scheduling-policy priority` | 按固定 priority 踢 | 固定 6 |
| **C：我们的** | `--scheduling-policy priority` + `AdmissionController` | 按 idle 时间驱逐 | 动态 2-6 |

## 采集指标

| 指标 | 来源 | 怎么比 |
|---|---|---|
| 总完成时间 | 墙钟 start → 最后一个任务 done | C < B < A |
| QPS | N_tasks / 总时间 | C > B > A |
| P50/P95 延迟 | 每个 task 的 latency_ms | C 更稳 |
| 并发度时序 | 记录每分钟并发数 | C 自适应 |
| mem_peak MB | MemSampler 峰值 | C ≤ 阈值 |
| HBM 利用率曲线 | 每分钟采样一次 | C 在 70-85% 震荡 |
| 成功率 | task_success_rate | C 活跃 session 不掉 |
| 驱逐统计 | 被踢 session 的 idle 时间 | C 驱逐的是真的 idle |

## 固定参数

| 参数 | 值 | 原因 |
|---|---|---|
| 模型 | Qwen2.5-7B-Instruct | 统一 |
| engine | vllm-ascend | NPU |
| domain | τ-bench retail | 统一 |
| gpu-memory-utilization | 0.3 | 制造 HBM 压力 |
| max-model-len | 32768 | 允许长序列 |
| total tasks | 10 | 足够观察并发行为 |
| 外部 user-sim | mimo-v2.5-pro | 维持成功率 |

## 实验步骤（NPU 开时）

```bash
# === A 组：vllm 原生 FCFS ===
# 起引擎（默认 FCFS）
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model models/Qwen2.5-7B-Instruct --port 8000 \
    --gpu-memory-utilization 0.3 --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser hermes

# 跑并发 benchmark（固定 6 并发，FCFS）
make -f scripts/f5_experiment.mk run-group-a

# === B 组：vllm priority ===
# 起引擎（priority 调度）
.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model ... --scheduling-policy priority ...（同上）

make -f scripts/f5_experiment.mk run-group-b

# === C 组：我们的 AdmissionController ===
# 起引擎（priority 调度）
# 跑带 AdmissionController 的并发 benchmark
make -f scripts/f5_experiment.mk run-group-c
```

## 预期结果

C 组在 mem_peak 控制在阈值内的前提下，成功率 ≥ A/B 组，P95 延迟方差更小，
驱逐命中 idle session 比例 > 90%。
