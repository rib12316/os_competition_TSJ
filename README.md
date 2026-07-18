# os_competition_TSJ

> 赛题 14：**面向智能体的内存管理系统设计与实现**。
> 基于 vLLM / vLLM-Ascend 等开源推理框架扩展，面向 LLM agent 推理流程做 KV Cache / 上下文 / 显存分配的系统性优化，在保证任务成功率不下降的前提下降低显存占用、提升推理效率，并在国产 OS（openEuler）+ 国产 NPU（Ascend）上部署。

本仓库是**开发基础设施 + 实现主体**。赛题原始材料见 [`docs/`](docs/)，技术总方案见 [`docs/技术设计方案.md`](docs/技术设计方案.md)。

## 目录结构

| 目录 | 说明 | 入 git |
|---|---|---|
| [`agent-mem/`](agent-mem/) | ★ 赛题实现主体（Python 包 `agent_mem`） | ✅ |
| [`docs/`](docs/) | 赛题与要求、技术设计方案等交付/参考文档 | ✅ |
| [`third_party/`](third_party/) | 第三方代码库克隆（vllm / vllm-ascend / Qwen-Agent / tau-bench）；仅 [`VERSIONS.md`](third_party/VERSIONS.md) 入库 | ❌（克隆体） |
| [`logs/`](logs/) | 实验日志；命名规范见 [`dev-guide/log-naming-convention.md`](dev-guide/log-naming-convention.md) | ❌（原始日志） |
| [`dev-guide/`](dev-guide/) | 内部开发指南：日志规范、环境记录、安装状态、路线图 | ✅ |

## 快速上手

```bash
# 所有 uv 命令均在仓库根目录执行（环境固定在 ./.venv）
source .venv/bin/activate           # 或用 .venv/bin/python
python -c "import agent_mem; print(agent_mem.__file__)"
```

> 完整环境搭建步骤见 [`dev-guide/environment-setup.md`](dev-guide/environment-setup.md)，
> 当前安装状态与已知问题见 [`dev-guide/install-status.md`](dev-guide/install-status.md)。

## 关键约束

- **国产化**：openEuler 容器化（必做）+ vLLM-Ascend NPU（建议）。
- **公平对照**：同硬件 / 同模型 / 同 prompt / 同 seed，before-after 各跑 3 次取中位数。
- **成功率红线**：优化不得使任务成功率下降超过 2 个百分点。
