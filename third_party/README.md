# third_party — 第三方代码库

本目录存放**克隆的开源代码库**，供阅读、扩展与对照。克隆体**不入 git**（见根 `.gitignore`），
精确版本（repo / tag / commit）记录在 [`VERSIONS.md`](VERSIONS.md)。

> 所有命令在仓库根目录执行；隔离环境固定在 `../.venv`。

## 当前克隆（MVP 核心）

| 目录 | 上游 | 用途 | 许可 |
|---|---|---|---|
| `vllm/` | https://github.com/vllm-project/vllm | 推理后端 V1（默认开 prefix cache） | Apache-2.0 |
| `vllm-ascend/` | https://github.com/vllm-project/vllm-ascend | 国产 NPU 适配插件 | Apache-2.0 |
| `qwen-agent/` | https://github.com/QwenLM/Qwen-Agent | Agent 框架（ReAct + 工具调用） | Apache-2.0 |
| `tau-bench/` | https://github.com/sierra-research/tau-bench | 多轮工具调用 Benchmark（retail/airline） | MIT |

## 重新克隆

```bash
# 见 dev-guide/environment-setup.md 的「第三方克隆」一节，或按 VERSIONS.md 中的 tag/commit
git clone --depth 1 --branch v0.22.1 https://github.com/vllm-project/vllm third_party/vllm
git clone --depth 1 --branch v0.22.1rc1 https://github.com/vllm-project/vllm-ascend.git third_party/vllm-ascend
git clone --depth 1 https://github.com/QwenLM/Qwen-Agent third_party/qwen-agent
git clone --depth 1 https://github.com/sierra-research/tau-bench third_party/tau-bench
```

## 后续按需补充（对应技术方案各模块）

- SGLang（M5 分支共享）、LMCache（M4 分层）、LLMLingua（M2 压缩）、AgentBench、Mooncake（M9 多机）
