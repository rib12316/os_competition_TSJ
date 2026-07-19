# P3 — 测量 / 质量中枢

> 角色主线：**benchmark 骨干 + 质量门**。你是 Model C 的关键——既产出「证明一切有效的数字」（应用效果 40%），又守代码质量（代码规范 20%）。**你 day 1 的契约 + 桩，是 P1/P2 能真并行的前提。**
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)；日志规范 [`../log-naming-convention.md`](../log-naming-convention.md)。

## 你的协作边界

- **产出接口（给别人用，最先交付）**：① `configs/*.yaml` schema；② stub OpenAI server；③ τ-bench 任务接口；④ `metrics.json` schema。
- **消费接口（别人给你）**：① P1 的 `/metrics` 端点（KV 命中率）；② P2 的 agent CLI / trace。
- **质量门职责**：审查 P1/P2 所有 PR；debug 反馈；CI / pytest / ruff；复现性检查；成功率红线监控。
- **审查**：你审 P1/P2；你的 PR 由 P1/P2 互审（你也是开发，不只是审查）。

---

## 🔓 阶段 0 — 契约先行（day 1，必须最先 merge，解阻塞）

> 这是整个并行的命脉。这一节没做完，P1/P2 没法真正并行。落点：[`agent-mem/configs/`](../../agent-mem/configs/)、[`benchmarks/`](../../agent-mem/benchmarks/)。

- [ ] **钉死 config schema**：把 [`configs/*.yaml`](../../agent-mem/configs/) 的 `engine / benchmark / metrics` 三段结构定稿（已有 baseline/prefix_cache/optimized 雏形，补齐字段注释）。P1 按此启动引擎，P2 按此接任务。
- [ ] **钉死 `metrics.json` schema**：6 大指标字段对齐 [`log-naming-convention.md`](../log-naming-convention.md) 的 schema（`e2e_latency_p50/p95`、`qps`、`mem_peak_mb`、`kv_cache_hit_rate`、`task_success_rate`、`ttft_ms`、`seed`、`started_at`…）。
- [ ] **stub OpenAI server**：实现一个 mock `/v1/chat/completions`（含 tool_call 假返回），让 P2 在 P1 引擎没好之前就能跑通 agent loop。
- [ ] **harness 骨架**：把 [`benchmarks/runner.py`](../../agent-mem/benchmarks/runner.py)（当前是 `NotImplementedError` 占位）替换成可跑的骨架——能解析 config、调 agent、落 run 目录（即使指标先填 0）。
- [ ] **run 目录脚手架**：按日志规范自动建 `logs/<ts>_<engine>_<model>_<config>_run<N>/`，写 `config.yaml` + `git_commit.txt` + `env.txt`（复现性从 day 1 就有）。
- [ ] **PR + merge 到 main**：这一批**最先合并**，作为 P1/P2 的集成基线。

---

## MVP 任务清单（必做地基）

> 命中：应用效果（before/after 数字）+ 代码规范（CI/测试/复现）。落点：[`benchmarks/`](../../agent-mem/benchmarks/)、[`tests/`](../../agent-mem/tests/)。

### benchmark harness 实现

- [ ] **τ-bench 任务接入**：接 τ-bench retail/airline 任务集到 [`benchmarks/tasks/`](../../agent-mem/benchmarks/tasks/)（多轮工具调用 + 策略合规）。
- [ ] **6 大指标采集**（赛题必采）：
  1. 单 agent end-to-end 延迟（p50 + p95）
  2. 多 agent 并发吞吐（QPS）
  3. GPU/NPU 显存峰值
  4. KV cache 命中率（抓 P1 的 `/metrics`：`vllm:gpu_prefix_cache_hits_total`）
  5. 任务成功率（τ-bench task success rate）
  6. TTFT（首 token 时间）
- [ ] **显存时间序列**：`mem_timeseries.csv`（表头 `timestamp,used_mb`），按时间采样（峰值要从时序算，不能只看瞬时）。
- [ ] **vLLM `/metrics` 抓取**：落 `vllm_metrics.json`，提取 KV 命中率等。
- [ ] **三档对照自动化**：baseline（`--no-enable-prefix-caching`）/ prefix_cache / optimized 三档，各跑 3 次取中位数，自动产出 [`logs/_summaries/<date>_<study>_comparison.md`](../../logs/_summaries/)。
- [ ] **对照结论门槛**：要求显存峰值降幅 ≥ 30%、延迟降幅 ≥ 20%、成功率差距 ≤ 2 个百分点（不达标要在报告里标红）。

### 质量门（持续，贯穿全程）

- [ ] **实时 code review**：P1/P2 每个 PR 你都过——重点查①接口契约一致性 ②可复现性 ③成功率风险 ④`ruff`/测试。
- [ ] **debug 反馈**：P1/P2 卡住时优先帮 debug（尤其引擎/agent 联调、tool_call 解析）。
- [ ] **CI**：GitHub Actions 跑 `pytest --cov`（覆盖率 ≥ 60%）+ `ruff check`，全绿才许 merge。
- [ ] **pytest 套件**：覆盖 config 解析、metrics 采集、run 目录生成、对照聚合等核心逻辑（[`tests/`](../../agent-mem/tests/)）。
- [ ] **复现性检查**：抽检 run 目录的 `config.yaml`/`git_commit.txt`/`env.txt` 齐全且可重放。
- [ ] **成功率红线监控**：每个优化模块的 PR 必须附成功率数字，下降 > 2pp 拒收。
- [ ] **NPU/GPU 约束**：真机 benchmark 前确认设备已启动——**NPU 默认停着，真跑前暂停等用户启动**。

### MVP 集成时序（你的位置）

1. **day 1**：merge 契约 + stub + harness 骨架（阶段 0）。
2. P1/P2 并行开发（你审查他们的 PR）。
3. 收尾：接 τ-bench + 6 指标 + 三档对照，跑出第一份 `mvp-three-tier_comparison.md`。
4. 配合 P1 完善 `docker compose` 一键启动。

---

## 扩展任务（M1–M11 + 横切）

### 你主责的模块

- [ ] **M3 session-aware 调度（2–3 天）**：落点 [`scheduler/`](../../agent-mem/src/agent_mem/scheduler/)。复用 vLLM V1 的 SharedStorageConnector（KV 存盘 + 按需加载），在其上加 session 生命周期策略（idle N 秒 evict GPU KV、reactivate 从 CPU/disk 恢复）。**协作**：P1 接 connector；P2 透传 session 信号。命中「长生命周期动态资源回收」。
- [ ] **M10 session KV checkpoint / 恢复（2–4 天）**：与 M3 合并叙事为「长生命周期 Session 管理总章」——M3 管运行中 session，M10 管跨进程/跨重启持久化。复用 SharedStorageConnector。

### 横切（每个模块都要做，是你的持续主线）

- [ ] **全模块 before/after 数字**：M1/M2/M4/M5/M6… 每个优化点都要你出对照表（显存↓、延迟↓、成功率）。
- [ ] **多模型对照**：Qwen + MiniCPM 各跑全套 benchmark，证明优化方法通用（命中「适配不同模型规模」）。
- [ ] **多引擎对照（配合 M5）**：vLLM(block 级) vs SGLang(token 级) 在分支工作流下的差异。
- [ ] **国产化容器内 benchmark（配合 M8）**：在 openEuler 容器中跑通完整 benchmark；有 Ascend 则验 NPU。

### 文档（10% 评分，你的活）

- [ ] **技术报告**：架构图 + 5 特征↔5 机制映射表 + 优化前后对比表 + 复现步骤。
- [ ] **mkdocs 站点**：`docs/`（architecture / deployment / benchmark-report / optimization-deep-dive）。
- [ ] **部署指南**：openEuler / Ascend 部署步骤。

---

## 交付标准（每个任务完成时自检）

- [ ] 数字可复现：同 seed、3 次中位数，run 目录齐全。
- [ ] 对照报告含结论 + 门槛判定（达标/标红）。
- [ ] CI 全绿，覆盖率 ≥ 60%。
- [ ] 你审过的 PR 都附了成功率数字。
- [ ] `ruff check` 无错。
