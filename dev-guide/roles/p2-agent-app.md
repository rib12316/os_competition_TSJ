# P2 — Agent / 应用开发

> 角色主线：**Agent 编排 + 上下文中间件**。你维护自写 ReAct agent，对接 P1 的引擎和 P3 的 benchmark；并把缝D 中间件接口搭起来，让 F2/F3 挂上去。
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)（v2）。

## 你的协作边界

- **产出接口**：① Agent CLI / 任务入口（prompt → trace），P3 用它驱动 benchmark；② **缝D 中间件接口**（F2/F3 都挂这）。
- **消费接口**：① P1 引擎 OpenAI 兼容 API（含 tool_call）；② P3 的 `configs/*.yaml` + τ-bench 任务接口 + stub server（无 NPU 时联调）。
- **关键解锁（不被卡）**：P1 引擎没好之前，对着 P3 的 stub OpenAI server 开发——day 1 即可并行。
- **审查**：你的 PR 由 **P3 审查**。

## 第一步：补缝D 中间件接口（解锁 F2/F3）

> 当前 `react.py` / `tau_bench_agent.py` 直接把 messages 喂给 client，**没有中间件挂载点**。这是 F2/F3 的前置动作，最小代码改动。

- 在 agent 层加一个 `Middleware` 接口：
  - `transform_messages(messages) -> messages`：发引擎前变换（F2 压缩、F3 注入引用）
  - `intercept_tool_result(name, result) -> result`：拦工具返回值（F3 存外部 store）
- 落点：`agent/middleware/__init__.py`（当前空壳）。
- 接口钉死后，F2/F3 各自一个独立类挂上去，互不干涉。

## 你负责的功能（2 个，落在缝D）

### F2 — Prompt / 上下文压缩

- **做什么**：包一层 `transform_messages`，对 system prompt / 冷历史用 LLMLingua 压 2–4×；工具定义 schema 多轮重复 → 只发一次后续用引用。
- **落点**：`middleware/compress.py`（一个 Middleware 子类）+ `configs/f2-compress.yaml`。
- **开源借鉴**：[LLMLingua](https://github.com/microsoft/LLMLingua)（+ LLMLingua-2，task-agnostic 低延迟）；[AutoGen 集成示例](https://microsoft.github.io/autogen/0.2/docs/topics/handling_long_contexts/compressing_text_w_llmligua/)。
- **⚠️ 红线**：压过头伤成功率，必须配 ablation，**成功率下降 ≤ 2pp**，由 P3 验。
- **工作量**：2–3 天 · 难度中。

### F3 — 工具调用数据 lazy-load

- **做什么**：拦截工具返回值（长 HTML/JSON/图）→ 存 SQLite/内存 → context 只放 `<doc id=.. summary=..>`；加 `fetch(id)` 工具让模型按需取详情。
- **落点**：`middleware/lazyload.py`（一个 Middleware 子类，实现 `intercept_tool_result`）+ 一个外部 store + `configs/f3-lazyload.yaml`。
- **开源借鉴**：mem0 / LangChain retriever 的 reference-ID 模式；τ-bench 的订单/航班 JSON 是天然真实数据。
- **挑战**：模型可能不主动用 fetch 接口，需 few-shot 引导。
- **工作量**：3–4 天 · 难度中。

> F2 与 F3 都在缝D，但各自独立类、独立 yaml、独立 toggle，可链式组合。**互不阻塞。**

## 已完成的 MVP 基座（你的产出，已 merge）

- [x] 自写 ReAct loop（`agent/react.py` 的 `run_react` + `stream_chat_with_ttft`，含 TTFT 测量）
- [x] 工具集（`agent/tools.py`：search stub + python 安全执行）
- [x] τ-bench 直驱（`agent/tau_bench_agent.py`，真驱 τ-bench 环境）
- [x] CLI 入口（`python -m agent_mem.cli "..."`）
- [x] 多轮 trace 输出（P3 采集成功率/延迟）
- [x] session_id 全链路透传（为 F5/F6 预留）
- [ ] 模型无关：dev 用 Qwen3-0.6B 跑通，真实 benchmark 切 Qwen2.5-7B / MiniCPM3

> 注意：实现是**自写 ReAct**（走 OpenAI 兼容 client），不依赖 qwen-agent 的 `Assistant` 类。vendored 的 `third_party/qwen-agent` 仅备选。

## 交付标准（每个功能完成时自检）

- [ ] 对 stub 和真实引擎都能跑（base_url 可切，不写死）。
- [ ] 多轮 trace 可解析（P3 能从中算成功率/延迟）。
- [ ] session_id 全链路透传。
- [ ] 成功率下降 ≤ 2pp（压缩类尤其要验）。
- [ ] before/after 数字交 P3（同 seed、3 次中位数）。
- [ ] `ruff check` + `pytest`（P3 的 CI 会卡）。
