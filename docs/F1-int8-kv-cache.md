# F1 — int8 KV Cache 量化

> 主责：P1 | 缝：A | 状态：❌ 暂停（Ascend 上三条路均不通）

## 探索记录

2026-07-20 在 vllm-ascend 0.22.1rc1 + Ascend 910B2C + CANN 9.0.0 上依次尝试：

### 路径 1：`--kv-cache-dtype int8`

**结论：no-op。** vllm V1 `kv_cache_dtype` 枚举里没有 `int8`——有 `int8_per_token_head`、`fp8_per_token_head`、`fp8_e4m3` 等，不存在裸 `int8`。传了不报错也不生效。

### 路径 2：`--kv-cache-dtype int8_per_token_head`

**结论：Ascend 后端崩溃。** vllm 内置的 per-token-head 量化是 CUDA 路径。vllm-ascend 虽接受 flag，但在 `_reshape_kv_cache_tensors` 中按 float16 维度分配 buffer，int8（1 字节）元素数是 float16（2 字节）的两倍，reshape 崩溃：

```
RuntimeError: shape '[256, 128, 4, 128]' is invalid for input of size 17301504
```

vllm-ascend 的 KV cache buffer 管理写死了 float16，接不住 int8。

### 路径 3：C8（Ascend 原生 int8 KV）

**结论：需要完整校准文件，不能一行 flag 搞定。** C8 是 vllm-ascend 自己写的 int8 KV（`AscendC8KVCacheAttentionMethod`），通过 `--quantization ascend` 激活。但它走 `AscendModelSlimConfig` 框架，会量化**所有权重 + KV cache**——只写 `{"kv_cache_type": "C8"}` 不够，遇到 `model.embed_tokens.weight` 等没有在 config 里声明的层就报 `KeyError` 崩溃。

完整配置（`quant_model_description.json`）需要华为 `msmodelslim` 校准工具逐层生成——在非校准模型上无法使用。

## 结论

Ascend 910B2C（无 FP8 单元）上做 KV 量化的三条路都不可行：

| 路径 | 失败原因 |
|---|---|
| `--kv-cache-dtype int8` | vllm 枚举中不存在，no-op |
| `--kv-cache-dtype int8_per_token_head` | CUDA 路径，Ascend 后端 buffer 不兼容 |
| C8 + `--quantization ascend` | 需 per-layer 校准文件，非校准模型不可用 |

## 代码（保留 config 供参考，flag 不生效）

`configs/f1-int8.yaml` 已更新为 `int8_per_token_head`（最新尝试值）。该 yaml 仍然存在的意义：如果 vllm-ascend 后续版本修复了 buffer reshape 兼容性，可以直接启用。

## 备选方向

如果后续需要 KV 量化叙事：

1. 等 vllm-ascend 升级 → 支持 `int8_per_token_head` 的 buffer reshape
2. 跑 `msmodelslim` 校准 → 生成完整 `quant_model_description.json` → C8
3. 改为调度参数调优（`gpu-memory-utilization` + `max-num-seqs`），实现同等的"显存↓"叙事
