# F5 Phase 2 调研与实证结论（无损 KV offload）

> 分支：`feat/f5-priority-evict` ｜ 日期：2026-07-25 ｜ 结论：**当前栈不可行**（gated on 更新版 vllm-ascend）

## 目标
把 F5 的资源回收从「有损 priority 抢占」升级为「无损 KV offload」（冷 KV→CPU，命中时回读），
并进一步做「按需、逐 session 定向 offload」。

## 调研（3 路并行：vllm/vllm-ascend 源码 + 外部 SOTA + spike 可行性）

### 1. 按需、逐请求、无损 offload —— 被 vLLM 架构挡住
- V1 connector API 是**纯回调**，只在 scheduler 前向循环内被调（`get_num_new_matched_tokens` /
  `request_finished` / `build_connector_meta` / `save_kv_layer` / `get_finished`）。
  **无外部入口、无中途 hook、无 per-request 注入通道。**
- 外部应用层无法 mid-step 触发"offload 这个请求的 KV"——上游也还没一等公民支持
  （RFC #33689 OffloadPolicy、#22605 分离进程，**均未落地**）。
- 唯一**架构支持**的外部驱动路径 = priority 抢占 → `handle_preemptions`，**有损**（KV 丢弃）。
- **身份/IPC 都已解决**（`extra_body.vllm_xargs.session_id`→`request.kv_transfer_params`；
  unix socket IPC；`--kv-transfer-config` + `kv_connector_module_path` 注册自定义 connector）——
  唯一缺的是"外部触发无损 offload"这一环，必须 fork vLLM core（加 offload_request + 触发通道）。

### 2. 外部 SOTA —— 对单 NPU 都不适用
- **Llumnix**（OSDI'24）：跨实例活迁移（Gloo），**无单实例 HBM↔CPU 路径**；vLLM v0.6.3 旧。
- **Mooncake**（FAST'25）：全局 KV 池 + RDMA，跨节点价值，单节点多余。
- **AttentionStore**（ATC'24）：分层 KV 存取，设计最接近，但**无开源代码**。
- **vAttention**（MSR）：CUDA 虚拟内存，**CANN 无对应 API，Ascend 跑不了**。
- vLLM 上游已有 **OffloadingConnector**（自动 LRU 无损 offload），vllm-ascend 有 **NPUOffloadingSpec**。

## 实证（Option B：开 NPUOffloadingSpec 自动无损 offload 层）
**在本机 vllm-ascend 0.22.1rc1 + vllm 0.22.1 上实测 NPUOffloadingSpec —— 失败（版本 drift）：**

1. `--kv-connector` **被拒**（本版本不存在）；正确入口是 `--kv-transfer-config`
   （`OffloadingConnector` + `kv_connector_extra_config.spec_name=NPUOffloadingSpec,
   spec_module_path=vllm_ascend.kv_offload.npu, num_cpu_blocks=N`）。
2. `--kv-offloading-backend native` 默认走 CUDA 的 `CPUOffloadingSpec` → Ascend 上
   `register_kv_caches` 报错。
3. NPUOffloadingSpec 的 `npu.py`/`cpu_npu.py` **import 了不存在的模块**
   （`vllm.v1.kv_offload.abstract` / `mediums` / `spec`——本版 vllm 已重命名为 `base` / `cpu.common`）。
   加 import shim 后能 import，但 `register_kv_caches` 仍 **`AssertionError`**
   （`vllm/v1/executor/abstract.py:123`）——**更深的行为级版本不兼容**（#5948-class bug）。

> 结论：文档所述"shipping、NPU-tested"的 NPUOffloadingSpec 是**更新版本对**的特性；
> 本仓 pinned 的 0.22.1rc1/0.22.1 这对**不可用**，需 fork vllm-ascend offloading 路径才能修
> （中高风险，agent 调研早有预警）。

## 最终结论
- **Phase 2 无损 offload 在当前栈不可行**：按需定向需 fork vLLM core；自动无损（NPUOffloadingSpec）
  在本版本对坏掉（版本 drift + 行为 assertion）。
- **F5 Phase 1（有损 KV-pool 准入控制）是当前栈的交付**：消除抢占（3→0）、KV 命中率 0.93 vs 0.46、
  eviction-hit-idle 100%——已足够命中赛题"动态资源回收"。
- **解锁条件**：升级 vllm-ascend 到 NPUOffloadingSpec 可用的版本对（追踪 issue #3241/#5948），
  或 fork vllm-ascend 修 offloading 兼容（中高风险）。

## 产物
- `configs/f5-evict-dynamic-offload.yaml`：F5 + NPUOffloadingSpec 的 D 组配置（**当前栈 blocked**，
  升级 vllm-ascend 后可直接复用）。
- 实验脚本 `scripts/f5_experiment.py`（`--only` 跑单组）可复用跑 D。
- third_party / venv 的临时 import shim 已**全部回退**，栈恢复 pristine。
