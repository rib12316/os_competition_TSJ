# 开发路线图（v2 Ascend 重构版）

> 总方案：[`docs/技术设计方案.md`](../docs/技术设计方案.md)（v2：9 个独立功能 F1–F9，落在 7 条稳定缝上）。
> 三人协作：[`roles/README.md`](roles/README.md)。
> 评分指挥棒：功能完整性 30% / 应用效果 40%（**必须 before/after 数字**）/ 代码规范 20% / 文档 10%。

## v1 → v2 的关键变化

- 旧 `M1–M11` 耦合清单 → **9 个独立功能 F1–F9**（旧→新映射见总方案 §0）。
- 作废：SGLang 双引擎（不支持 Ascend）、FP8 KV（910B 无 FP8 单元 → 改 **int8**）。
- 新增 **F8 多硬件对照**顶替「适配不同框架/硬件」20 分项。

## MVP 基座（已实现，后续功能的落脚点）

- [x] 测量骨干（`bench/`：Runner Protocol + 6 指标 + 三档对照 + 门槛判定）
- [x] Agent 层（自写 ReAct + τ-bench 直驱 + CLI）
- [x] 引擎封装（`server/vllm_server.py` + stub server）
- [x] 配置/复现（`config.py` + `repro.py`）
- [x] **缝脚手架已补齐**：缝D 中间件接口、缝E session 策略、缝C KV connector 配置
      （均含注册表/工厂/独立 yaml/单测，机制留待真机）

> 现在是「能测、且有挂载点」的测试机。每个 F 就是把对应缝上的占位填成真模块。

## 优先级（按 ROI，每步不被前一步卡）

```
F8 ＞ 补缝D接口(已done) ＞ F2 ＞ F3 ＞ F4 ＞ F1(验) ＞ F5 ＞ F6 ＞ F7 ＞ F9
```

| 功能 | 缝 | 主责 | 最简实现 | Ascend 可行 |
|---|---|---|---|---|
| **F8** 多硬件对照 | B | P1 | `backend: vllm` vs `vllm-ascend` | ✅ 两套引擎已装 |
| **F2** Prompt 压缩 | D | P2 | LLMLingua 包一层 `transform_messages` | ✅ 纯 agent 层 |
| **F3** 工具 lazy-load | D | P2 | 工具结果存 store + 引用 id（`intercept_tool_result`） | ✅ 纯 agent 层 |
| **F4** LMCache 分层 | C | P1 | `--kv-transfer-config` + LMCacheAscendConnector | ✅ config 就绪 |
| **F1** int8 KV | A | P1 | `--kv-cache-dtype int8` | ⚠️ 待真机验证 |
| **F5** idle eviction | E | P3 | `simple_kv_offload` + idle 策略 | ✅ 机制现成 |
| **F6** checkpoint/恢复 | E | P3 | `save_kv_layer` 落盘 + 重启加载 | ✅ 与 F5 共机制 |
| **F7** 分支共享（测量版） | F | P1 | 自带 ToT driver 测 APC 隐式 CoW | ✅ 测量版 |
| **F9** openEuler/Ascend 部署 | G | P1 | CANN openeuler 镜像 + 脚本 | ✅ Ascend 已就绪 |

## 阶段 1 — MVP（必做地基，约 65–70%）

- [x] vLLM V1 启动封装（默认开 prefix cache）
- [x] 自写 ReAct + 多轮工具调用，直连 OpenAI 兼容 API
- [x] Benchmark harness：τ-bench retail/airline，采集 6 指标
- [x] 三档对照：baseline / prefix_cache / optimized
- [x] Docker compose 一键启动

## 阶段 2 — 高 ROI 功能（累计约 78–90%）

- [ ] **F8 多硬件对照**（1 天，命中「适配不同框架/硬件」20 分）
- [ ] **F2 Prompt 压缩**（2–3 天，LLMLingua + 成功率 ablation）
- [ ] **F3 工具 lazy-load**（3–4 天，引用语法 + few-shot 引导 fetch）
- [x] **F4 LMCache 分层**（config/yaml/args 翻译 ✅；NPU 开时装 lmcache_ascend + 真机验证）
- [ ] **F1 int8 KV**（验证 0.5 天 + benchmark 1 天，⚠️ 尽早上 NPU 探针）

## 阶段 3 — 深化（按时间余量）

- [ ] **F5 session idle eviction**（2–3 天，复用 simple_kv_offload）
- [ ] **F6 KV checkpoint/恢复**（2–4 天，跨重启持久化）
- [ ] **F7 分支共享测量版**（1 周+，自带 ToT driver）
- [ ] **F9 openEuler/Ascend 部署**（1 天，国产 OS 硬性要求，**不要漏做**）

## 真机前提（NPU 启停由用户控制）

NPU 默认停着。任何真机步骤（F1 验证、F4/F5/F6 联调、F9 容器内 benchmark）**先暂停等
用户启动设备**。纯 agent 层（F2/F3）与 harness 可在无 NPU 下开发（对 stub server）。

## 每个功能的交付标准（叙事模板）

1. **现象**：agent 场景观察到的具体内存问题（带数据）
2. **根因**：为什么（agent 特征）
3. **机制**：优化方案（技术原理）
4. **验证**：before/after 数字（显存↓、延迟↓、成功率）
5. **权衡**：代价（如压缩的成功率损失）

> 日志按 [`log-naming-convention.md`](log-naming-convention.md) 落盘，before/after 各 3 次取中位数；
> `config` 字段用功能名（`f1-int8` / `f2-compress` / `f4-lmcache` / `f5-evict` …）。
