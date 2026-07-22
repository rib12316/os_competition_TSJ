# F1 C8 int8 KV 量化 · Ascend 调研结论（迁移 CUDA 前存档）

> 调研日期：2026-07-22 ｜ 环境：vllm 0.22.1 + vllm-ascend 0.22.1rc1 + 910B(A3) + CANN 9.0.0
> 结论：**Ascend 上 KV 量化此路不通（fp8 硬件缺、int8 有 graph 死锁 bug）→ 决定迁移 CUDA vLLM。**
> 工具存档：`scripts/probe_f1_c8.py`（探针）、`scripts/calibrate_f1_c8.py`（自校准）。

## 一句话结论

裸 `--kv-cache-dtype int8` 是 **no-op**；真 int8 走 **C8**，显存↓已铁证（int8 KV ≈ bf16 2× 容量），但 C8 decode 的 **FULL ACL-graph capture 死锁** → 退化 eager ~0.1 tok/s 不可用。fp8 KV 在 910B **硬件级不可用**（A5-only）。**这两个坑都是 Ascend/vllm-ascend 专属，迁移 CUDA 后不复存在。**

## 逐条结论

### fp8 KV — 910B 硬件级 NO
910B（A3 类）无 fp8 单元，vllm-ascend 把 fp8 KV 硬门控到 A5（`modelslim_config.py:647`、`models/layer/attention/layer.py:178`）。换 A5 卡才行。

### int8 KV（C8）— 显存可行、速度被 graph bug 卡死
- **激活机制**：不走 `--kv-cache-dtype`，走 `--quantization ascend` + 模型目录 `quant_model_description.json`（`kv_cache_type:"C8"`），触发 `AscendC8KVCacheAttentionMethod`（`kv_c8.py:108`）。
- **必须 per-channel scale**：默认 `ones(1)` 会在 `_prepare_c8_scales` 的 `.view(num_kv_heads,1,head_size)` 崩（Qwen2.5-7B 需 (512,)、Qwen3-0.6B 需 (1024,)，后者因显式 `head_dim=128`）。
- **Qwen2 不在 C8 scale 加载 patch 里**：`patch_gqa_c8.py` 只 patch Qwen3/Glm4Moe/MiniMaxM2，不覆盖 Qwen2（Qwen2.5 是 Qwen2）→ 需扩展 vendored patch（本会话做过、已还原 stock）。
- **显存↓铁证**：int8 KV cache 在 41.39 GiB 可用空间装 1.55M tokens（≈ bf16 2× 容量）。
- **速度阻塞（根因）**：C8 decode 用 `npu_fused_infer_attention_score` + int8 分页 KV + per-channel antiquant scale（NZ BNSD 布局）——这条 attention 路径在 **FULL graph capture 大 decode batch 时确定性死锁**（Qwen2 batch26、Qwen3 batch19，NPU 101W 空转）。PIECEWISE capture 每轮 100% 通过、只有 FULL 挂。
- **已试且失败的修法**：换 Qwen3（同挂）、真 scale 自校准（同挂、同 batch）、小 max-num-seqs、`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`（只解 profiling、不解 FULL）。**根因是 C8 graph 路径 bug，与模型/scale 无关。**
- **未穷尽的一路**：强制 `cudagraph_mode=PIECEWISE`（`compilation_config.cudagraph_mode`）跳过挂死的 FULL——本会话未及测试。

### KVarN（华为 CSL）— 非 Ascend、且同款 graph 风险
vLLM v0.23 fork，CUDA/Triton+FlashAttention+CUDA-graph 专用（NVIDIA sm_120），**仓库零 Ascend 支持**（issue #23「support NPU」零回复；CI 是烟雾弹）。其 graph 安全只对 CUDA-graph 成立，**不迁移到 ACL-graph**——结构上和 C8 同一类陷阱。triton_ascend 本地 `import` 还报 libtorch_npu.so 链接错。

## 迁移 CUDA vLLM 的影响（为何此路 reopening）

KV 量化的所有 graph 陷阱都是 **Ascend ACL-graph 专属**。在 CUDA vLLM 上：
- **fp8 KV**（`--kv-cache-dtype fp8` / `fp8_e4m3` / `fp8_e5m2`）**原生可用**（CUDA 有 fp8 单元 + FlashInfer/Triton 后端）。
- **int8 KV**（`int8_per_token_head`）原生可用（per-token 动态、免校准）。
- **KVarN**（`kvarn_k4v2_g128`，4-bit K/2-bit V）原生可用（华为出的，正是为 CUDA 设计）。
- 这些量化 kernel 都能在 **CUDA-graph** replay 里存活 → 不存在 Ascend 那个死锁。

⇒ **迁移 CUDA 后，"降 KV 显存且不掉速"重新完全可行**——fp8/int8_per_token_head/KVarN 任选，且都是 vLLM 原生 flag，无需 vendored 改动。

## 仍 graph 安全的 Ascend 备选（若部分留 Ascend）
APC 前缀去重、native KV CPU offload（`--kv-offloading-size`）、LMCache-Ascend storage-layer 量化（serde，attention 前反量化回 fp16）、UCM/Mooncake 跨 session 复用——均不碰 attention kernel。
