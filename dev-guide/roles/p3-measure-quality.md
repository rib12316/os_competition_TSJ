# P3 — 测量 / 质量中枢

> 角色主线：**benchmark 骨干 + 质量门**。你既产出「证明一切有效的数字」（应用效果 40%），又守代码质量（代码规范 20%）+ 文档（10%）。你的 day 1 契约 + 桩，是 P1/P2 能真并行的前提——**现已交付**。
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)（v2）；日志规范 [`../log-naming-convention.md`](../log-naming-convention.md)。

## 你的协作边界

- **产出接口**：① `configs/*.yaml` schema；② stub OpenAI server；③ τ-bench 任务接口；④ `metrics.json` schema。
- **消费接口**：① P1 的 `/metrics` 端点（KV 命中率）；② P2 的 agent CLI / trace。
- **质量门职责**：审查 P1/P2 所有 PR；CI / pytest / ruff；复现性检查；成功率红线监控。
- **审查**：你审 P1/P2；你的 PR 由 P1/P2 互审。

## 你主责的功能（2 个，落在缝E · Session 生命周期）

> v2 把旧 M3+M10 的"合并叙事"**彻底拆开**：F5 = 运行时 idle 淘汰；F6 = 跨重启持久化。两者共机制（offload connector），但功能独立、各自交付。

### F5 — Session idle eviction

- **做什么**：session idle N 秒 → 把 GPU KV 块 offload 到 CPU，session 重新激活时恢复。命中"长生命周期动态资源回收"。
- **落点**：`scheduler/evict.py`（一个策略模块）+ `configs/f5-evict.yaml`。
- **机制（现成，你只补策略）**：vllm-ascend 原生 `simple_kv_offload/`（"NPU adaptation of vLLM's simple CPU KV-cache offloading"）+ 上游 `SimpleCPUOffloadConnector`。
- **独立性**：F5 的策略可对 stub offload 单独开发/测；只把"KV 怎么搬到 CPU"委托给现成 connector，不与 F6 共代码。
- **工作量**：2–3 天。

### F6 — KV checkpoint / 恢复

- **做什么**：长任务中断后恢复——进程退出前 `save_kv_layer` 落盘；重启冷启动按 session_id 反向加载。命中"长生命周期"差异化叙事。
- **落点**：`scheduler/checkpoint.py`（一个策略模块）+ `configs/f6-checkpoint.yaml`。
- **机制（现成）**：同 F5 的 offload（`SimpleCPUOffloadConnector.save_kv_layer` 已存在）+ SharedStorageConnector 思路。
- **独立性**：与 F5 共机制库，但触发逻辑/落盘格式/恢复流程各自独立，**不再合并叙事**。
- **工作量**：2–4 天。

## 横切主线（贯穿所有功能，是你的持续职责）

### 全模块 before/after 数字

- F1/F2/F3/F4/F5/F6/F7/F8/F9 每个功能 PR，都要你出对照表（显存↓、延迟↓、成功率），附结论 + 门槛判定。
- 门槛：显存峰值↓ ≥ 30%、延迟↓ ≥ 20%、成功率差 ≤ 2pp（不达标标红）。
- 落 `logs/_summaries/<date>_<fX>_comparison.md`。

### 多模型 / 多硬件对照

- 多模型：Qwen + MiniCPM 各跑全套，证优化方法通用。
- 多硬件（配合 F8）：vllm(CPU) vs vllm-ascend(NPU)。

### 文档（10% 评分，你的活）

- 技术报告：架构图 + 特征↔机制↔功能映射表 + 优化前后对比表 + 复现步骤。
- mkdocs 站点：architecture / deployment / benchmark-report / optimization-deep-dive。
- 部署指南：openEuler / Ascend 步骤。

## 已完成的测量骨干（你的产出，已 merge）

- [x] config schema（`config.py`：dataclass + 白名单校验）
- [x] stub OpenAI server（`server/stub_openai.py`，功能完整）
- [x] harness 骨架 + Runner Protocol（`bench/runner.py`，可插拔）
- [x] 6 指标采集（`metrics.py`，全实现）
- [x] 三档对照 + 门槛判定（`bench/compare.py`，全实现 + 测试覆盖）
- [x] run 目录复现脚手架（`bench/run_dir.py` + `repro.py`）
- [x] 显存时序采样（`bench/mem_sampler.py`，NPU 后端已实现）

## 质量门（持续）

- [ ] 实时 code review：P1/P2 每个 PR 过——查①接口契约一致 ②可复现性 ③成功率风险 ④ruff/测试。
- [ ] CI：GitHub Actions 跑 `pytest --cov`（≥ 60%）+ `ruff check`，全绿才许 merge。
- [ ] 复现性检查：抽检 run 目录 `config.yaml`/`git_commit.txt`/`env.txt` 齐全且可重放。
- [ ] 成功率红线：每个功能 PR 必须附成功率数字，下降 > 2pp 拒收。
- [ ] NPU/GPU 约束：真机 benchmark 前确认设备已启动——**NPU 默认停着，暂停等用户**。

## 交付标准（每个功能完成时自检）

- [ ] 数字可复现：同 seed、3 次中位数，run 目录齐全。
- [ ] 对照报告含结论 + 门槛判定（达标/标红）。
- [ ] CI 全绿，覆盖率 ≥ 60%。
- [ ] 你审过的 PR 都附了成功率数字。
- [ ] `ruff check` 无错。
