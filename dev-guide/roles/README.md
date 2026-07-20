# 三人协作模型（Model C）— 缝与功能驱动版

> 本文件定义三人并行开发的**协作契约**。三份角色清单见同目录：
> [`p1-engine-backend.md`](p1-engine-backend.md) · [`p2-agent-app.md`](p2-agent-app.md) · [`p3-measure-quality.md`](p3-measure-quality.md)
>
> 总方案：[`docs/技术设计方案.md`](../../docs/技术设计方案.md)（v2 Ascend 重构版）；路线图：[`../roadmap.md`](../roadmap.md)。

---

## 本版相对旧版的核心变化

旧版按 `M1–M11` 切模块，**任务之间互相绑死**（M3+M10 合并叙事、M7 依赖 M5、M5 跨三人）。v2 重构后：

1. **9 个独立功能 F1–F9** 取代耦合的 M 清单，每个落在 **7 条稳定缝**之一（详见总方案 [§7 条稳定缝](../../docs/技术设计方案.md#7-条稳定缝--9-个独立功能)）。
2. **每个功能单主责**，可独立开发/独立测/独立交付——三人现在是**真并行**，不再是"一个功能绑三个角色"。
3. 据硬件现实作废 SGLang 双引擎（原 M5）、FP8 KV（原 M1 方案）；新增 F8 多硬件对照顶替"适配不同框架/硬件"。

### 旧 M → 新 F 映射

| 旧 | 新 | 说明 |
|---|---|---|
| M1 | F1 | FP8 → **int8**（910B 无 FP8） |
| M2 | F2 | Prompt 压缩 |
| M3 | F5 | Session idle eviction |
| M4 | F4 | LMCache 分层 |
| M5 | ❌ | 作废（SGLang 不支持 Ascend） |
| M6 | F3 | 工具数据 lazy-load |
| M7 | F7 | 降级为"分支共享测量版" |
| M8 | F9 | openEuler/Ascend 部署 |
| M9 | — | 低优，不入 F（单机 Mooncake 性价比低） |
| M10 | F6 | KV checkpoint/恢复（**与 F5 彻底拆开**） |
| M11 | — | 低优，文档为主 |
| — | F8 | 新增：多硬件对照（顶替 M5 的 20 分项） |

---

## 为什么仍是 Model C（2 特性开发 + 1 测量中枢）

评分里 **70%**（应用效果 40% + 代码规范 20% + 文档 10%）取决于「能测、测得准、可复现、文档全」——这是**横切性质量工作**，不是某人的功能模块。留一个人（P3）守测量骨干 + 质量门，最契合评分指挥棒。

v2 的改进：**功能独立后，Model C 终于能真并行了**——不再有"一个功能绑三个角色"的死结。

---

## 三个角色一句话定位

| 角色 | 主线 | MVP 解锁物（已完成） | 负责的功能 |
|---|---|---|---|
| **P1 引擎/后端** | 推理引擎启停 + KV/分层/部署机制 | vllm-ascend serve + prefix-cache 开关 + `/metrics` | **F1 / F4 / F7 / F8 / F9** |
| **P2 Agent/应用** | Agent 编排 + 上下文中间件 | 自写 ReAct + 多轮工具 + CLI | **F2 / F3**（+ 补缝D 接口） |
| **P3 测量/质量中枢** | benchmark 骨干 + 质量门 | 契约 schema + stub server + harness（已交付） | **F5 / F6** + 全模块 before/after + 文档 |

---

## 7 缝 × 9 功能 × 3 角色 矩阵

> 每个功能**只有一个主责**；P3 为所有功能出 before/after（横切，不算共同主责）。

| 缝 | 功能 | 主责 | 最简实现 | Ascend 可行 |
|---|---|---|---|---|
| **A** 引擎 flag | **F1** int8 KV | P1 | `--kv-cache-dtype int8` | ⚠️ 待真机验证 |
| **B** 引擎后端 | **F8** 多硬件对照 | P1 | `backend: vllm` vs `vllm-ascend` | ✅ |
| **C** KV connector | **F4** LMCache Ascend 分层 | P1 | `--kv-transfer-config` + LMCacheAscendConnector | ✅ |
| **D** 上下文中间件 ⭐ | **F2** Prompt 压缩 | P2 | LLMLingua 包一层 | ✅ |
| **D** 上下文中间件 ⭐ | **F3** 工具 lazy-load | P2 | 工具结果存 SQLite + 引用 ID | ✅ |
| **E** Session 策略 | **F5** idle eviction | P3 | 原生 `simple_kv_offload` + idle 策略 | ✅ |
| **E** Session 策略 | **F6** checkpoint/恢复 | P3 | `save_kv_layer` 落盘 + 重启加载 | ✅ |
| **F** 研究/demo | **F7** 分支共享测量 | P1 | 自带 ToT driver 测 APC 隐式 CoW | ✅（测量版） |
| **G** 部署 | **F9** openEuler/Ascend | P1 | CANN openeuler 镜像 + 脚本 | ✅ 已基本就绪 |

> ⭐ 缝D 接口当前**不存在**，P2 需先补 `Middleware.transform_messages()` / `.intercept_tool_result()` 接口，F2/F3 才能挂上去。

---

## 接口契约表（并行的命脉）

| 契约 | 产出方 | 消费方 | 落点 |
|---|---|---|---|
| 引擎 OpenAI 兼容 API（`/v1/chat/completions` + tool_call） | P1 | P2、P3 | vllm-ascend 默认；P1 保证 tool calling 可用 |
| 引擎 `/metrics` Prometheus 端点 | P1 | P3（抓 KV 命中率） | `vllm:gpu_prefix_cache_hits_total` |
| `configs/*.yaml` schema（engine/benchmark/metrics + 每功能一档） | P3 | P1、P2 | `agent-mem/configs/` |
| Agent CLI / 任务入口（prompt → trace） | P2 | P3 | `python -m agent_mem.cli ...` |
| 缝D 中间件接口 | P2 | F2/F3 各自 | `agent/middleware/` |
| `metrics.json` schema（6 指标） | P3 | 全员 | 见 [`../log-naming-convention.md`](../log-naming-convention.md) |
| stub OpenAI server | P3 | P2（无 NPU 联调） | `server/stub_openai.py`（已交付） |

---

## Git 并行约定（v2 真并行）

- 每个功能一条短命 feature 分支：`feat/fX-<short>`（如 `feat/f8-multi-hw`、`feat/f2-compress`）。
- **功能独立 = 分支独立**：F1–F9 之间无前置依赖（F5/F6 借用缝C connector 作机制库，但策略各自实现，不算阻塞），可任意并行。
- **建议落地顺序**（ROI，不被前一步卡）：
  ```
  F8 ＞ 补缝D接口 ＞ F2 ＞ F3 ＞ F4 ＞ F1(验) ＞ F5 ＞ F6 ＞ F7 ＞ F9
  ```
- 所有 PR 由 **P3 审查**（P3 是质量门）；P1/P2 互审可加速。
- commit 规范：每个功能独立分支/独立提交，**附 before/after 数字**。

---

## 共同红线

- **成功率红线**：任何优化不得使 τ-bench 任务成功率下降 > 2pp（赛题硬指标）。
- **公平对照**：同硬件/同模型/同 prompt/同 seed，before/after 各 **3 次取中位数**。
- **日志规范**：所有 run 按 [`../log-naming-convention.md`](../log-naming-convention.md) 落盘到 `../logs/`；`config` 字段用功能名（`f1-int8` / `f2-compress` / `f4-lmcache` / `f5-evict` …）。
- **复现性**：每个 run 目录含 `config.yaml` + `git_commit.txt` + `env.txt`。
- **NPU 启停由用户控制**：NPU 默认停着。任何真机步骤（F1 验证、F4/F5/F6 联调、F9 容器内 benchmark）**先暂停等用户启动设备**。

---

## 每个功能的交付标准（叙事模板，所有人通用）

1. **现象**：agent 场景观察到的具体内存问题（带数据）
2. **根因**：为什么（agent 特征）
3. **机制**：优化方案（技术原理）
4. **验证**：before/after 数字（显存↓、延迟↓、成功率不掉）
5. **权衡**：代价（如压缩的成功率损失）

> 详见总方案 [技术叙事模板](../../docs/技术设计方案.md#技术叙事模板每个功能都要套用)。
