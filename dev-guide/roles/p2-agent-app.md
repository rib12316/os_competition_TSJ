# P2 — Agent / 应用开发

> 角色主线：**Agent 编排 + 上下文中间件**。你把 Qwen-Agent 包成可驱动的 ReAct agent，对接 P1 的引擎和 P3 的 benchmark。
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)。

## 你的协作边界

- **产出接口（给别人用）**：Agent CLI / 任务入口（输入 prompt → 输出多轮 trace），P3 用它驱动 benchmark。
- **消费接口（别人给你）**：① P1 的引擎 OpenAI 兼容 API（含 tool_call）；② P3 的 `configs/*.yaml` + τ-bench 任务接口。
- **关键解锁（不被卡）**：**P1 引擎没好之前，你对着 P3 的 stub OpenAI server 开发**——所以你从 day 1 就能并行，不用等 P1。
- **审查**：你的 PR 由 **P3 审查**。

---

## MVP 任务清单（必做地基）

> 命中：功能完整性（跑得通的 agent 系统）。落点：[`agent-mem/src/agent_mem/agent/`](../../agent-mem/src/agent_mem/agent/)。

- [ ] **Qwen-Agent 包装层**：用 `Assistant` 类直连 vLLM OpenAI API（base_url 指向 P1 引擎 / dev 阶段指向 P3 的 stub）。落点 `src/agent_mem/agent/assistant.py`。
- [ ] **ReAct + 多轮工具调用 loop**：实现多步决策（plan → tool_call → observe → reflect），不是单轮 chat。
- [ ] **工具集**：至少 `search` + `python` 两个工具；并对接 τ-bench retail/airline 需要的 tool schema（订单查询 / 航班 / 退换货策略合规等）。
- [ ] **CLI 入口**：`python -m agent_mem.cli "帮我查今天的天气并写到文件"`，能跑通工具调用并返回（赛题验证方案第一条）。
- [ ] **trace 输出**：每次 run 产出可解析的多轮 trace（写到 `agent.log`），供 P3 采集成功率/延迟。
- [ ] **session_id 透传**：每个 agent session 带 session_id（为后续 M3/M10 预留，别到时再改接口）。
- [ ] **多模型验证**：同一 agent loop 在 Qwen + MiniCPM 各跑通（dev 用 Qwen3-0.6B，真实 benchmark 切 Qwen2.5-7B / MiniCPM3——模型由 P1 落地，你只保证 agent 层模型无关）。
- [ ] **tool_call 解析鲁棒性**：vLLM 的 tool calling 是单次 chat completion，多轮语义靠你这一层编排（赛题已知约束：vLLM Responses API 仍在开发中，[Issue #33089](https://github.com/vllm-project/vllm/issues/33089)）。

### MVP 集成时序（你的位置）

1. P3 先 merge 契约 + stub（day 1）→ 你拿到 stub server + τ-bench 任务接口。
2. 你开 `feat/p2-agent`，**对着 stub** 写 Qwen-Agent + tools + CLI（不被 P1 阻塞）。
3. P1 merge 真实引擎后，你把 base_url 从 stub 切到真实引擎联调 → PR → **P3 审查** → merge。

---

## 扩展任务（M1–M11，按 ROI 排序）

### 高 ROI（优先做）

- [ ] **M2 Prompt / 上下文压缩（2–3 天）**：落点 [`middleware/`](../../agent-mem/src/agent_mem/middleware/)。
  - 工具定义 schema 多轮重复 → 只发一次后续用引用。
  - 系统提示用 LLMLingua 压 2–4×。
  - 冷历史轮次自动摘要替换。
  - ⚠️ **红线**：压过头会伤成功率，必须配 ablation，**成功率下降 ≤ 2 个百分点**，由 P3 验。
- [ ] **M5 协同（SGLang 双引擎）**：同 agent loop 在 P1 的 SGLang 后端也跑通（OpenAI 兼容，agent 代码基本不动），配合 P3 出双引擎对比。
- [ ] **M6 工具调用数据 lazy-load（3–4 天）**：落点 [`middleware/`](../../agent-mem/src/agent_mem/middleware/)。
  - 拦截工具返回值（长 HTML / JSON / 图），存 Redis/SQLite，context 只放摘要 + 引用 ID（如 `<doc id="d_42" summary="...">`）。
  - Agent 需要详情时再 lazy load。
  - 挑战：模型可能不主动用 lazy load 接口，需 few-shot 引导。

### 中 ROI（按时间余量）

- [ ] **协同 M3/M10**：配合 P3 的 session 机制——确保 agent 层把 session 生命周期信号（idle / reactivate / checkpoint）正确透传到引擎层。

---

## 交付标准（每个任务完成时自检）

- [ ] 对 stub 和真实引擎都能跑（base_url 可切，不写死）。
- [ ] 多轮 trace 可解析（P3 能从中算成功率/延迟）。
- [ ] session_id 全链路透传。
- [ ] 成功率下降 ≤ 2 个百分点（压缩类优化尤其要验）。
- [ ] before/after 数字交 P3（同 seed、3 次中位数）。
- [ ] `ruff check` + `pytest`（P3 的 CI 会卡）。
