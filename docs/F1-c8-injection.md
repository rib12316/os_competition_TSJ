# F1 C8 库注入 — 模块说明

> **缝A · 精度层** ｜ 调研日期：2026-07-22 ｜ 分支：`feat/f1-int8-kv-c8`
> 关联：[`F1-c8-upstream-survey.md`](F1-c8-upstream-survey.md)（上游 issue/PR 实测）、`F1-int8-kv-cache.md`（代码层调研）、`configs/f1-int8.yaml`、`agent_mem/kv/c8.py`
> 装配版本：`vllm==0.22.1` + `vllm_ascend==0.22.1rc1`

---

## 0. 这是什么

把 F1 的 int8 KV 量化（Ascend 上叫 **C8**）做成项目一等公民库模块
`agent_mem/kv/c8.py`：纯函数 + CLI，负责"给模型目录注入 quant_description / 占位
scale / 还原 stock"，触发 vllm-ascend 原生 C8 路径。**纯函数、单测全过、无需 NPU**。

> 与旧 stash 的关系：`stash@{0}`（feat/f7-branch-kv）已写好 configs + 测试契约但缺
> `c8.py`；本分支把缺的模块补齐并应用那份契约。

---

## 1. 激活机制（与上游 PR [#7474](https://github.com/vllm-project/vllm-ascend/pull/7474) 一致）

Ascend 真 int8 KV = **C8**，**不**靠 `--kv-cache-dtype int8`（0.22.1rc1 已证 no-op：
`attention_v1.py:1309` 硬断言 `scales==1.0`，全仓无 int8→C8 路由）。正确激活靠：

1. 模型目录写 `quant_model_description.json`（`"kv_cache_type":"C8"` + 每层
   `k/v_proj.kv_cache_scale` 标记）→ 触发 `AscendC8KVCacheAttentionMethod`
   （`vllm_ascend/quantization/methods/kv_c8.py`）。
2. 引擎加 `--quantization ascend`（见 `configs/f1-int8.yaml` / `optimized.yaml`）。

C8 是**静态 per-channel** 量化（antiquant `perchannel` 模式），scale 从权重加载，
**非**运行时动态标定。

---

## 2. 用法

### Python API（`agent_mem.kv.c8`）

| 函数 | 作用 | 需 torch? |
|---|---|---|
| `build_c8_quant_description(model_dir)` | 构造 quant_description dict（不落盘） | 否 |
| `is_annotated(model_dir)` | 是否已注入 quant_description | 否 |
| `annotate_model(model_dir, overwrite=False)` | 写 `quant_model_description.json`（已存在且不 overwrite → `FileExistsError`） | 否 |
| `inject_placeholder_scales(model_dir, value=0.05)` | 注入占位 per-channel scale（探针用，精度垃圾） | 是 |
| `restore_model(model_dir)` | 还原 stock（删 json + scales + 还原备份） | 否 |

### CLI

```bash
# 跑前必做：标注模型（触发 C8，无需 scale）
PYTHONPATH=agent-mem/src:$PYTHONPATH python -m agent_mem.kv.c8 annotate models/Qwen2.5-7B-Instruct

# 注入占位 scale（探针路径验证用；真精度 scale 走自校准，见下）
python -m agent_mem.kv.c8 inject-scales models/Qwen2.5-7B-Instruct --value 0.05

# 还原 stock
python -m agent_mem.kv.c8 restore models/Qwen2.5-7B-Instruct
```

### 起引擎（真机，缝A）

`configs/f1-int8.yaml` 已配 `--quantization ascend`，经 `build_serve_args` 的
`extra_args` 透传（无需改 `vllm_server.py`）。annotate 完模型后直接起：

```bash
python -m agent_mem.server.vllm_server --config agent-mem/configs/f1-int8.yaml \
    --model-path models/Qwen2.5-7B-Instruct
```

---

## 3. ⚠️ 真机 Caveat（910B / A3）

来自 `feat/f1-c8-ascend-investigation` 的真机结论文档（2026-07-22，6 轮探针）：

- **显存↓ 已铁证**：int8 KV cache 在 41.39 GiB 可用空间装 1.55M tokens（≈ bf16 2× 容量）。
- **速度被 graph 死锁卡死**：C8 decode 用 `npu_fused_infer_attention_score` + int8
  分页 KV + per-channel antiquant（NZ BNSD 布局），这条 attention 路径在 **FULL
  ACL-graph capture 大 decode batch 时确定性死锁**（Qwen2 batch26、Qwen3 batch19，
  NPU 101W 空转）→ 退化 eager ~0.1 tok/s 不可用。
- PIECEWISE capture 每轮 100% 通过，**只有 FULL 挂**。
- 已试且失败的修法：换 Qwen3、真 scale 自校准、小 `max-num-seqs`、
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`（只解 profiling，不解 FULL）。
- **根因是 vllm-ascend 的 C8 graph 路径 bug，与模型/scale 无关**。

### 2026-07-23 真机端到端 debug 结果（Qwen3-0.6B，910B2C）

**前置大坑先解**：vllm-ascend 引擎一启动就崩，根因是 **numpy 2.4.6 ABI 砸了 serving**
（transformers→sklearn→pandas 崩，`numpy.dtype size changed`）——不是 C8 的问题。
fix = pin `numpy==1.26.4`（已应用）。详见 memory `numpy-abi-broke-npu-serving`。

**① ✅ FULL_DECODE_ONLY 解开了 graph 死锁**（investigation "此路不通" 只测了 FULL）：
```
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
# CUDAGraphMode.FULL_DECODE_ONLY = (FULL, NONE)：prefill 走 FULL graph，decode 走 eager
```
引擎正常起（init ~6.6s）、C8 激活（`[kv_c8.py:122] setting kv_cache_torch_dtype to torch.int8`）、
**不死锁**、正常生成 ~200–260 tok/s。死锁根因确认是 **decode 走 FULL graph capture**；
decode 改 eager（NONE）即绕开。enum 另有 `FULL_AND_PIECEWISE`（decode=PIECEWISE）未测，可能更快。

**② ✅ C8 输出垃圾的根因 = 校准采错张量（已修，非上游 bug）**

初次用"自校准 MinMax scale"跑 C8，输出是垃圾（`coholiccoholic...`），bf16 baseline 同 prompt
连贯。**一度误判为 vllm-ascend 0.22.1rc1 的 C8 正确性 bug——此判断错误，现更正。**

`scripts/verify_c8_rope.py` 实测定因：校准脚本 hook 的是 **pre-RoPE 的 `k_proj`**，但 KV cache 存的是
**post-RoPE K**（vLLM V1 在 rotary 之后才把 K 交给 C8 backend 量化）。RoPE + `k_norm` 把早期层 K 的
逐通道幅值放大 **最多 ~435×**（L0），pre-RoPE scale 对早期层小几个数量级 → 量化饱和截断 → 反量化错位 → 垃圾。

| Layer | post/pre maxabs | pre-scale roundtrip err | pre-scale 饱和 | **post-scale err** |
|---|---|---|---|---|
| L0 | 435× | 0.845 | 68% | **0.004** |
| L7 | 29× | 0.477 | 26% | 0.005 |
| L14 | 11× | 0.086 | 4.8% | 0.006 |
| L27 | 1.2× | 0.036 | 0.02% | 0.005 |

⇒ **per-channel MinMax scheme 本身完全正确**（post-RoPE scale 的 quant→dequant roundtrip 误差 ~0.5%）。
bug 100% 在校准采的张量，是我们能修的，**不是上游 bug**。

**修法**（`scripts/calibrate_c8.py`）：K 改 hook **post-RoPE**（monkeypatch
`transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb` 抓旋转后 K 的逐通道 maxabs）；
V 无 RoPE，`v_proj` 直采不变。scale = maxabs/127，注入 model.safetensors。

**③ ✅ 修正后 C8 完全可用**（FULL_DECODE_ONLY + post-RoPE scale）：
- 输出连贯正确：`7×8 → "...7*8 is 56..."`、`primary colors → "...red, blue, and yellow..."`。
- ~200 tok/s、不死锁、KV cache 是 int8。
- KV 容量：54.85 GiB int8 池装 **1,026,944 tokens**（≈ bf16 2×）。

### 结论（已更正）

- **C8 在 vllm-ascend 0.22.1rc1 上完全可用**：`FULL_DECODE_ONLY`（解 graph 死锁）+ **post-RoPE 逐通道
  MinMax 校准**（解精度）。**F1 解锁，无需升级版本、无需 ModelSlim。**
- investigation 的 "此路不通" 两处都错：① 只测 FULL graph（死锁）→ `FULL_DECODE_ONLY` 可绕开；
  ② 校准 hook pre-RoPE（垃圾）→ post-RoPE 校准正确。
- ⚠️ **本结论 Qwen3-specific**：校准脚本 monkeypatch 的是 qwen3 的 `apply_rotary_pos_emb`；换模型族要改 hook 点。换模型/版本前用 `verify_c8_rope.py` 复验 post/pre 比值。
- 调试后模型已还原 stock（无残留）。

---

## 4. 真精度 scale（post-RoPE 自校准，已含，需 NPU）

占位 `scale=0.05` 精度垃圾，仅验证路径。真 per-channel MinMax scale 由自校准产出——**必须 hook
post-RoPE K**（见 §3 ②的坑）：

- `scripts/calibrate_c8.py`：跑模型 + monkeypatch `apply_rotary_pos_emb` 抓 post-RoPE K 逐通道 maxabs
  + `v_proj` 抓 V（无 RoPE）→ `/127` → 注入 model.safetensors。**这是让 C8 输出正确的关键脚本。**
- `scripts/verify_c8_rope.py`：诊断工具——对比 pre/post-RoPE 逐通道 maxabs + roundtrip 误差，确认
  校准采的张量是否对得上 cache。换模型/版本前先跑它。

⚠️ `calibrate_c8.py` 的 monkeypatch 针对 **Qwen3**（`transformers.models.qwen3.modeling_qwen3`）；
换模型族要改 hook 点（其它模型同理：cache 的是 post-RoPE K，校准必须采 post-RoPE）。

---

## 5. 测试

```bash
PYTHONPATH=agent-mem/src:$PYTHONPATH python -m pytest agent-mem/tests/test_kv.py -q
```

`test_c8_description_marks_all_weights_float_and_kv_c8` 与 `test_annotate_writes_json_and_status`
是 `c8.py` 的验收契约（纯函数，fake model 仅需 `config.json` + `model.safetensors.index.json`）。

---

## 6. 避雷（与上游 survey 一致）

C8 别叠：Mooncake PD 分离（bf16↔int8 mismatch）、MTP / EAGLE3 投机采样（crash）、
Ascend 950（不支持 sparse C8）。详见 [`F1-c8-upstream-survey.md`](F1-c8-upstream-survey.md) §4。
