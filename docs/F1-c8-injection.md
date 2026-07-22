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

### 唯一未穷尽的一路（待 NPU 开启后验证）

强制 PIECEWISE capture 跳过挂死的 FULL：

```python
# 通过 VLLM 编译配置（或 extra_args 等价 CLI）
compilation_config = {"cudagraph_mode": "PIECEWISE"}  # 跳过 FULL graph capture
```

本分支**未验证**（NPU 停着，由用户控制）。验证路径：annotate + inject-scales 后起引擎，
强制 PIECEWISE，看是否仍死锁。详见 memory `npu-start-stop-user-controlled`——动 NPU 前暂停等用户。

---

## 4. 真精度 scale（自校准，本分支不含）

占位 `scale=0.05` 精度垃圾，仅验证路径。真 per-channel MinMax scale 由自校准产出，
实现见 `feat/f1-c8-ascend-investigation` 的 `scripts/calibrate_f1_c8.py`（跑模型 + k/v_proj
forward hook 收集 maxabs → `/127` → 写 safetensors + 更新 index）。移植进库属"可运行探针"
路径（需 NPU），不在本次"库+配置+测试"范围内。

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
