# 第三方代码库版本清单

> 克隆体不入 git；本文件记录精确版本以保证可复现。克隆时间见各条目。
> 克隆方式：均 `--depth 1`（vllm-ascend 含子模块 `csrc/third_party/catlass`）。

| 目录 | 上游 | tag / branch | commit hash | 克隆时间 (UTC) |
|---|---|---|---|---|
| `vllm/` | vllm-project/vllm | `v0.22.1` | `0decac0d96c42b49572498019f0a0e3600f50398` | 2026-07-18 02:05 |
| `vllm-ascend/` | vllm-project/vllm-ascend | `v0.22.1rc1` | `5f6faa0cb8830f667266f3b8121cd1383606f2a1` | 2026-07-18 02:05 |
| `qwen-agent/` | QwenLM/Qwen-Agent | `main` | `31a4d36d123688581a9e9744427272b33ce940e0` | 2026-07-18 02:05 |
| `tau-bench/` | **sierra-research**/tau-bench | `main` | `59a200c6d575d595120f1cb70fea53cef0632f6b` | 2026-07-18 02:05 |

> 注：技术方案中 τ-bench 上游误写为 `sierra-org/tau-bench`（已 404），正确为 `sierra-research/tau-bench`。

## 关键依赖版本

| 软件 | 要求版本 | 本机实际 | 状态 |
|---|---|---|---|
| CANN | 9.0.1 | 8.5.1 | ⚠️ 运行时需升级（见 install-status.md） |
| torch | 2.10.0 | **2.10.0+cpu** | ✅ CPU 版（无 nvidia 包） |
| torch-npu | 2.10.0.post2 | **2.10.0.post2** | ✅（运行需 NPU） |
| vllm | 0.22.1 | **0.22.1+empty**（editable） | ✅ 可导入 |
| qwen-agent | — | 0.0.34 | ✅ |
| tau-bench | — | 0.1.0（editable） | ✅ |
| agent-mem | — | 0.0.1（editable） | ✅ |
| **vllm-ascend** | 0.22.1rc1 | **0.22.1rc1** | ✅ ascend 插件已注册（NPU 启动后装） |
| **triton-ascend** | 3.2.1 | **3.2.1** | ✅ triton 后端 `['ascend']` |
| triton | 3.2.0 | 3.2.0 | ✅（triton-ascend 补丁版 libtriton） |
| nvidia-* CUDA 包 | — | **0 个** | ✅ CUDA-free 达成 |
