# F5 Phase 2 调研与实证结论（无损 KV offload）

> 分支：`feat/f5-priority-evict` ｜ 日期：2026-07-25 ｜ **结论：已解锁** —— 用 shipping 的 SimpleCPUOffloadConnector（→ Ascend 变体），无需升级/fork。

## 目标
把 F5 的资源回收从「有损 priority 抢占」升级为「无损 KV offload」（冷 KV→CPU，命中时回读），
叠加在 F5 的 KV-pool 准入控制之上。

## 调研（3 路并行）+ 关键纠正

### 1. 按需、逐请求、无损 offload —— 仍被 vLLM 架构挡住
V1 connector 是纯回调、只在 scheduler 前向循环内被调；外部应用层无法 mid-step 触发"offload
这个请求的 KV"。身份/IPC 都已解决（`extra_body.vllm_xargs.session_id`；unix socket），唯一缺
"外部触发无损 offload"，需 fork vLLM core（上游 RFC #33689/#22605 未落地）。**这条（on-demand
定向）仍属未来。**

### 2. 外部 SOTA —— 对单 NPU 都不适用
Llumnix（跨实例）/Mooncake（跨节点）/AttentionStore（无代码）/vAttention（CUDA 锁）均不可用于单 NPU。

### 3. ⚠️ 关键纠正：NPUOffloadingSpec 是废弃路径，不是"待升级/待 fork"
- `NPUOffloadingSpec`/`OffloadingConnector` **上游已废弃**——vllm-ascend 自己的 e2e 测试标
  `@pytest.mark.skip(reason="cpu offload connector is deprecated.")`；0.23.0rc1 列入 "Ready to Deprecate"。
- 本仓 0.22.1rc1 上它坏在两处：① import drift（`abstract`/`mediums`/`spec`→`base`/`cpu.common`）；
  ② `register_kv_caches` 的 `assert`（vllm 0.22.1 用 `CanonicalKVCaches`，vllm-ascend 还按老
  `dict[str,Tensor]` 写——深层架构 drift，#5948-class）。
- → **升级（只到 0.23.0rc1，CANN 9.0.0→9.0.1）不解决**（路径废弃）；**fork 无意义**（修废弃路径）。
  本文件早先的"blocked、需 fork/升级"结论是基于这条废弃路径的误判，特此纠正。

### 4. ✅ 真正可用：SimpleCPUOffloadConnector（注册时自动→ AscendSimpleCPUOffloadConnector）
- **本仓 0.22.1rc1 自带 + CI 实测非 skip**（`tests/e2e/.../test_simple_cpu_offload.py`），
  NPU 原生（`aclrtMemcpyBatchAsync` + `torch.npu` streams），**正确处理 Ascend 的 K/V 分离 + 2MiB
  对齐**（正是 NPUOffloadingSpec 栽掉的点），支持 `lazy_offload`。
- 注册时 vllm-ascend `__init__.py` 自动把上游 `SimpleCPUOffloadConnector` 的 CUDA worker 换成 NPU worker。
- 入口：`--kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"cpu_bytes_to_use":N,"lazy_offload":true}}'`（**非** `--kv-connector`，该 flag 本版被拒）。

## 实证：真机起 + D 组实验
- **真机起引擎** ✅（2026-07-25）：`SimpleCPUOffloadNPUWorker: 56 unique NPU KV tensors,
  allocating 585 CPU blocks (4.00 GB)`；`AscendSimpleCPUOffloadConnector: swapped CUDA worker
  for NPU worker`；`mode=lazy`。无需升级/fork。
- **D 组实验**（F5 priority/准入 + 该 offload 层，3 run 中位数，对照 C=f5-evict-dynamic）：

| 指标 | C（F5 only） | D（F5 + 无损 offload） |
|---|---|---|
| e2e p50 | 90 s | **47 s（1.9× 更快）** |
| e2e p95 | 155 s | 126 s（−19%） |
| QPS | 0.040 | 0.046（+16%） |
| mem_peak | 17401 | 17526（~同） |
| KV 命中率（GPU prefix） | 0.93 | 0.71（指标只数 GPU 命中、不数 CPU 回读，故低估） |

**机制**：offload 把冷 KV 搬 CPU → GPU KV 利用率更低 → **KV-pool 准入放开更高并发**（C 把
running 压到 ~3，D 能更高）→ p50 快 1.9×、QPS +16%。GPU prefix-hit 看着掉是因为搬走的块在 CPU、
不在 GPU cache（该指标不数 CPU 回读）——**延迟变快证明确实净收益**（CPU 回读 << 重算）。

## 最终结论
- **Phase 2 无损 offload 已在当前栈解锁**：用 SimpleCPUOffloadConnector（→ Ascend 变体），
  叠加在 F5 准入控制之上，**p50 再快 1.9×、QPS +16%**。
- 之前的"blocked/需 fork/升级"是基于废弃的 NPUOffloadingSpec 路径的误判，已纠正。
- 仍属未来：on-demand 逐 session 定向 offload（gated on vLLM core fork / 上游 RFC）。

## 产物
- `agent_mem/src/agent_mem/kv/connector.py`：修正 3 个 bug（connector 名 pykvconnector→
  SimpleCPUOffloadConnector；去掉被拒的 `--kv-connector` flag；JSON 改 vLLM 0.22.1 flat schema）。
- `configs/f5-evict-dynamic-offload.yaml`：D 组配置（F5 + kv_offload: SimpleCPUOffloadConnector）。
- `scripts/f5_experiment.py`：加 D 组（`--only D`）。
- 对照报告 `logs/_summaries/20260725_f5-C-vs-D_comparison.md`。
