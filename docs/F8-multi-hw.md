# F8 — 多硬件对照

> 主责：P1 | 缝：B | 代码零改动，纯 config 切换

## 原理

同一 agent + 同一 yaml，只改 `engine.backend` 字段，vllm(CPU) vs vllm-ascend(NPU) 各跑一遍对照。命中赛题"适配不同框架/硬件"20 分项。

## 代码

| 文件 | 内容 |
|---|---|
| `configs/f8-multi-hw.yaml` | `backend: vllm-ascend`（NPU 档，对照档改 `vllm`） |
| `config.py` | backend 白名单：`vllm`, `vllm-ascend` |
| `benchmarks/runner.py` | `--engine` 参数可覆盖 config 中的 backend |

无额外代码——backend 切换是 config 层的一行字段。

## 对照方案

两档跑同一 workload：

```bash
# NPU 档（f8-multi-hw.yaml 默认 backend: vllm-ascend）
.venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/f8-multi-hw.yaml \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root logs-f8 \
    --user-model mimo-v2.5-pro --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"

# CPU 档（同 config，--engine 覆盖 backend 为 vllm）
.venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/f8-multi-hw.yaml --engine vllm \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device cpu --log-root logs-f8 \
    --user-model mimo-v2.5-pro --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"
```

## 预期对比

| 指标 | vllm (CPU) | vllm-ascend (NPU) | 差异 |
|---|---|---|---|
| P50 延迟 | 更高（CPU 推理） | 更低（NPU 加速） | NPU 预期快数倍 |
| mem_peak | 系统 RAM | NPU HBM | 不同内存类型 |
| 成功率 | 相同 workload | 相同 workload | 应一致 |

## 状态

config/yaml 已就绪，待 NPU 空闲时跑两档 benchmark 出对照报告。
