# F2/F3 前端适配完整对话交接（2026-07-28）

## 1. 交接范围

本轮只负责赛题中的 F2/F3 及其前端演示，不负责继续修改 F1/F4/F5：

- **F2：Prompt / 上下文压缩**，包括 LLMLingua-2、LongLLMLingua、冷热历史、重压复用和可视化。
- **F3：大型工具结果 lazy-load**，包括 externalize、短引用、`fetch_tool_result` 和可视化。
- **前端：** Gradio 5 dashboard 中的单任务演示、实时 telemetry、LongBench 接入和统一场景路由。

本轮开始时已经阅读：

- `docs/赛题与要求.md`
- `docs/技术设计方案.md`
- `docs/F2-F3-frontend-integration-handoff-20260726.md`
- `docs/F2-F3-method-summary-20260725.md`
- `docs/F2-F3-context-telemetry-interface.md`

必须继续保持的结论边界：Prompt token 下降不能直接写成 vLLM 进程 HBM peak 已下降；F3 可能通过减少
单次 Prompt 避免 context overflow，但额外 fetch 轮次不保证降低单会话延迟。

## 2. 本轮完成的工作

### 2.1 F2 演示参数

τ-bench/上下文演示已经支持以下前端覆盖，只作用于当前任务，不写回正式 YAML：

- 首次压缩阈值：默认 `2000` token；
- 重压新增量：默认 `1000` token；
- 可压正文保留率：默认 `0.40`；
- 压缩方法：`llmlingua2` / `longllmlingua`。

正式生产配置保持：

```text
trigger_tokens=8000
recompress_delta_tokens=4000
hot_tool_trigger_tokens=1000
method=llmlingua2
tool_aware=true
```

方法选择边界：

- `llmlingua2` 是默认安全档，使用 multilingual BERT，支持项目侧 tool-aware 结构保护；
- `longllmlingua` 是 GPT-2 question-aware 实验档，可能更适合长上下文相关性筛选，但慢得多；
- LongLLMLingua 自动设置 `tool_aware=false`、`model_name=gpt2`、`hot_tool_trigger_tokens=0`；
- LongLLMLingua 不能宣称具备现有工具参数、关键 JSON 字段的物理隔离保护。

真实离线 worker 冒烟中，LongLLMLingua 将 241 token 压到 159 token，并保留了问题相关电池信息。

### 2.2 LongBench 单任务前端

新增 `agent_mem.demo.longbench_ui`，把 LongBench 2WikiMQA 接成和 τ-bench 对等的流式单任务流程：

```text
question
  -> retrieve_documents
  -> F3 externalize / passthrough
  -> fetch_tool_result（需要时）
  -> F2 request transform（启用时）
  -> final answer + gold + success
```

支持：

- data zip 路径，默认探测 `/tmp/longbench-data.zip`；
- `task_id`、`max_steps`；
- baseline / F2 / F3 / F2+F3；
- 与 τ-bench 相同的 Prompt/F2/F3 telemetry 面板；
- 真实预测、gold、成功/未命中，不用 token 收益掩盖质量结果。

LongBench stack 以 `configs/unified-longbench.yaml` 为基线，不能复用 τ-bench 的
`retail_compact` system policy。LongBench 保持：

```text
optimize_static_prompt=false
system_prompt_mode=none
F2-only hot_tool_trigger_tokens=1000
F2+F3 order=[lazyload, compress]
```

### 2.3 F2/F3 可读对比面板

原先四个 `gr.JSON` 直接展示 telemetry schema，评委难以理解。现已替换为 HTML 对比面板：

- 左侧 before：格式化真实正文、完整 token、消息数和截断状态；
- 右侧 after：before/after token、节省量、缩减率、method/reference 元信息；
- 差异区：删除为红色、新增/改写为绿色、无底色为保留；
- 用户/工具内容统一 HTML 转义；
- diff 最多处理 7000 展示字符，token 始终来自完整内容；
- 左侧保留折叠 raw telemetry，右侧 after 按后续要求简化，不再重复展示独立正文和 raw telemetry。

当前右侧 F2 after 只保留：

```text
before token / after token / 节省 token / 缩减率
method / preview 状态 / 压缩器正文 token
原文保留与压缩后改写
```

当前右侧 F3 after 使用相同布局，只保留 token、content type、短 result ID 和
“原工具结果 -> 短引用”差异区；独立短引用正文、最近 fetch 展开和 raw telemetry 已移除。

### 2.4 F2 diff 对齐修复

旧 diff 使用包含空格、换行和 UI 行头的全局 `SequenceMatcher`，长文本会因重复空白/高频词发生对齐漂移，
出现类似 `Mechanical -> Next`、`Keyboard -> Steps` 的视觉误导。

当前 F2 使用独立于 F3 的两轨对齐：

1. diff 输入排除 `[01] ASSISTANT` 等 UI 行头；
2. 按实词/标点对齐，空白不参与；
3. 常见停用词和高频重复词不能作为主要锚点；
4. 左轨只显示“原文保留/删除”；
5. 右轨只显示“压缩后保留/新增或改写”，不把无关词拼成替换；
6. before/after 各唯一出现一次的连续 4-token 精确短语作为强锚点；
7. 重压复用时比较完整发送 payload，包括压缩段后追加的未压缩冷消息。

针对下面句子已有回归测试，块移动后仍必须显示为保留：

```text
The order details for #W2378156 show that it was delivered on March 15th.
You received the following items:
```

F3 原工具结果到短引用的 diff 保持原算法，没有随 F2 改动。

### 2.5 统一上下文应用场景

原先页面有独立的“τ-bench 任务”和“LongBench 任务”tab。现已合并成一个
**“上下文优化任务”** tab，只保留一套运行按钮、对话和 telemetry 面板。

自动路由规则：

| 上下文模式 | workload | 说明 |
|---|---|---|
| `F2` | τ-bench | 多轮客服轨迹更适合展示 cold history 压缩 |
| `F3` | LongBench 2WikiMQA | 稳定产生大型 `retrieve_documents` 结果 |
| `F2+F3` | LongBench 2WikiMQA | F3 先外置，F2 再处理剩余上下文 |
| `baseline` | 用户选择 | 可选 F2 对照 τ-bench 或 F3 对照 LongBench |

动态控件行为：

- F2：显示 τ-bench 参数和 F2 参数；
- F3：显示 LongBench 参数，隐藏 τ-bench/F2 参数；
- baseline：显示“baseline 对照场景”选择。
- F2+F3：后端 preset / workload 路由 / 测试仍保留，但当前前端不暴露按钮。

统一 Gradio API 为 `context_task`，契约是 12 输入、7 输出。已经通过 API 真实提交 F3，确认实际进入
LongBench，middleware 为 `lazyload`，不是只做了前端隐藏。

### 2.6 本次交接前的前端瘦身与样式调整

在 `agent-mem/src/agent_mem/demo/chat_app.py` 上继续做了若干 F2/F3 前端小调整，只改展示与控件组织：

1. **上下文优化参数分组。**
   - `上下文模式`、`F2 压缩方法`、`F2 演示正文保留率` 等控件已放入带边框的 `参数调整` 区域。
   - 原 `F2 演示压缩阈值（仅当前前端任务；正式配置仍为 8000）` 与 `F2 演示重压新增量` 合并为一个
     `压缩阈值` 输入，默认 `2000`。
   - Gradio `context_task` 仍保持 12 输入、7 输出；同一个 `压缩阈值` 组件在输入列表中复用两次，分别传给
     `trigger_tokens` 与 `recompress_delta_tokens`，因此两个数始终相同。
   - 该前端演示输入只影响当前任务覆盖；正式 F2/F3 YAML 的 `8000/4000` 安全基线没有修改。

2. **上下文优化 Agent 对话样式。**
   - `上下文优化Agent对话` 外层增加边框。
   - 对话区字体略缩小。
   - 仅对上下文优化 Chatbot 添加气泡颜色：user 为浅绿色，agent/assistant 为白色；自由对话和 F5 对话不受影响。

3. **页面顶部清理。**
   - 顶部只保留 `agent-mem · KV/显存优化对比演示` 标题。
   - 删除了首屏长说明：
     `左：自由对话 / τ-bench 任务 / 📊 并发 benchmark ... 逐步累加成 before/after`。
   - `🏗 架构总览` 暂设为不可见，不再占首屏。
   - `🛠 引擎控制` 已恢复可见且默认展开；保留引擎功能开关、启动/停止按钮、显存上限和 `max_model_len`
     参数。只删除了其解释性说明文字，没有删除控件。

4. **右侧实时监控浮动。**
   - 纯 CSS `position: sticky` 在 Gradio 布局中实测不稳定，已改为滚动触发的 viewport dock。
   - 右侧 Column 使用 `live-monitor-column` 作为定位锚点；监控 Group 保留 `live-monitor-sticky` class。
   - 页面滚过监控区起点后，JavaScript 按右栏实时坐标和宽度增加 `is-docked`，切换为
     `position: fixed; top: 12px`；滚回顶部自动恢复普通布局。
   - 小于 900px 的窄屏始终使用普通流式布局，避免固定面板覆盖正文。
   - Playwright 实测桌面滚动后 Group 顶部坐标稳定为 12px、position 为 fixed；700px 窄屏保持 static。
   - 多段彩色分隔条已移除；恢复 Plot 的“实时监控（窗口=10s）”原生标签，并改为与监控图等宽的正常流标题栏，
     标签下方保留间距，不再遮挡第一排子图；10 秒窗口和刷新逻辑不变。
   - 修复左栏较短时右栏停靠导致页面高度塌缩、滚动位置回弹的问题：停靠期间右侧 Column 保留监控原高度占位。
   - 停靠态禁止 Gradio 内部 flex 块收缩，使右栏产生真实内部滚动范围；监控定时刷新后会恢复用户的右栏滚动位置。

5. **可见性/语法修复。**
   - 修复过一次 `chat_app.py` 中 F5 按钮区域 `with gr.Row()` 下的缩进错误；这是为了恢复整个 demo
     构造和 7860 前端启动，不改变 F2/F3 逻辑。
   - 运行实例 `/config` 已确认：`🛠 引擎控制`、`引擎功能（多选组合）`、`▶ 启动引擎`、`⏹ 停止`、
     `显存上限`、`max_model_len` 均可见。

6. **页面继续精简。**
   - `📊 统一 Benchmark` tab、对应后台 `run_study` 句柄和 `unified_bench` Gradio API 已从 demo 移除。
   - `🧪 高并发·F5` 保留运行参数、开始/停止、结果表和代表性会话；顶部场景背景及“两层交付”说明已移除。

## 3. 提交与回退点

本轮 F2/F3 相关提交按顺序为：

| 提交 | 内容 |
|---|---|
| `2923a06` | 前端增加 F2 重压新增量，演示默认 1000 |
| `7f97b67` | 前端选择 LLMLingua-2 / LongLLMLingua |
| `69cd909` | LongBench 流式单任务与对等 telemetry |
| `ebd0e78` | F2/F3 HTML 内容、token 和 diff 可视化 |
| `ef73422` | F2 两轨词法对齐，避免无关词替换 |
| `f3f17f5` | 精确唯一短语强锚点 |
| `4ed00c1` | 精简 F2/F3 after 面板，移除重复正文 |
| `c3c8141` | 合并 τ-bench/LongBench tab 并自动路由 |

已有 annotated checkpoint：

```text
tag:    f2-f3-frontend-longbench-v1
target: 69cd909
```

如需从初版 LongBench UI 建新恢复分支：

```bash
git switch -c restore-longbench f2-f3-frontend-longbench-v1
```

不要在当前脏 worktree 使用 `git reset --hard` 或 `git checkout --`。

## 4. 当前运行状态

交接检查时间：2026-07-28 UTC。

```text
branch: agent-mem-v1
HEAD:   本交接文档提交（开始新会话时以 git log 为准）
F2/F3 implementation: c3c8141
7860:   HTTP 200
7861:   HTTP 200
8000:   offline / no listener
```

7860 已运行在持久 tmux 会话：

```text
tmux session: agent-mem-demo-7860
```

查看日志/终端：

```bash
tmux attach -t agent-mem-demo-7860
```

从 tmux 脱离但不停止服务：`Ctrl-b`，再按 `d`。

本地 SSH 转发：

```bash
ssh -N -L 7860:127.0.0.1:7860 <user>@<server>
```

浏览器打开 `http://127.0.0.1:7860`。若本地 7860 被占用：

```bash
ssh -N -L 17860:127.0.0.1:7860 <user>@<server>
```

然后打开 `http://127.0.0.1:17860`。

此前出现的 `channel N: open failed: connect failed: Connection refused` 是服务器 7860 没有进程监听；
不是 Gradio 页面逻辑错误。8000 离线时页面仍可打开，但实际 benchmark 会失败，需要先通过引擎控制启动 Qwen。

## 5. 当前验证基线

2026-07-28 在当前工作树运行过两轮全量测试：

```text
pytest: 341 passed, 6 warnings
pytest: 344 passed, 8 warnings
```

最新一次命令使用去代理环境：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY \
    -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    PYTHONPATH=agent-mem/src \
    /data/os_competition_TSJ/.venv/bin/python -m pytest -q agent-mem/tests
```

warnings 主要为 Gradio 6 迁移相关 deprecation，另有 litellm/importlib-resources deprecation。

关键真实 smoke：

1. LongBench task 0，F3：1 次 externalize、1 次 fetch、3 次模型调用；一次运行累计 Prompt
   `18165 -> 3509`（-80.68%）。预测错误，页面如实显示未命中。
2. 统一 `context_task` 选择 F3：实际进入 LongBench，输出包含 `F3`、`lazyload` 和 F3 diff。
3. LongBench task 35，F2：hot tool 压缩 2 次、节省 5634 token，但 cold action 为 `skip`。

历史正式结果仍以 `docs/F2-F3-method-summary-20260725.md` 为准，不用单题 smoke 替代：

- F2 retail full115 完整 Prompt -19.32%，动态 BERT 0 次；
- F2 generic 28.9k trace -27.46%，首次 BERT 36.27 秒；
- F3 2Wiki first100 Prompt -66.03%，总体 31/100 与 baseline 相同；
- baseline 5 次 32k context error，F3 为 0；
- F3 p50 模型请求耗时增加 36.05%，原因是 fetch 增加模型调用。

## 6. 当前脏工作树：必须保留

当前有其他人/并行任务的 F5、引擎、报告和校准改动。它们不属于本轮 F2/F3，不能删除、回退或顺手提交。

主要包括：

```text
agent-mem/src/agent_mem/agent/tau_bench_agent.py
agent-mem/src/agent_mem/bench/tasks/tau_bench_adapter.py
agent-mem/src/agent_mem/demo/chat_app.py
agent-mem/src/agent_mem/demo/engine_control.py
agent-mem/src/agent_mem/demo/monitor.py
agent-mem/src/agent_mem/demo/overview.py
agent-mem/src/agent_mem/demo/tau_bench_ui.py
agent-mem/src/agent_mem/demo/f5_runtime.py
agent-mem/tests/test_demo_* / test_tau_bench_agent.py
scripts/calibrate_c8_qwen2.py
docs/技术报告*、项目说明书* 等未跟踪材料
```

`chat_app.py` 与 `test_demo_app_build.py` 同时包含已提交的 F2/F3 路由和未提交的 F5 工作。后续若要提交
F2/F3 新修改，不能直接 `git add` 整个文件；应检查 diff，并按纯 patch/index hunk 隔离提交。

## 7. 已知边界与后续建议

1. **F2 hot-tool 可视化仍不完整。** 当前主对比面板针对 cold history。某些 LongBench F2 轨迹会显示
   cold `skip`，但实际发生 hot tool compression；telemetry 只有计数/节省量，没有 hot before/after 正文。
   若继续改进，应先扩展 F2 hot-tool telemetry schema，再加对应视图，不能伪造 cold diff。
2. **F2/F3 自动 workload 是演示路由，不是算法限制。** 两个 middleware 都可运行在其他 workload；路由只是选择
   当前最稳定、最容易触发的展示场景。
3. **baseline 必须可选择两类对照。** 不要把 baseline 永久固定为 τ-bench，否则无法做 F3 同 workload 对照。
4. **F2+F3 后端仍保留但当前前端隐藏。** 如后续重新暴露，建议继续路由 LongBench 以稳定触发 F3；
   组合顺序必须保持 `[lazyload, compress]`。
5. **不要恢复右侧重复正文。** 用户明确要求 after 面板只保留指标、元信息和 diff。
6. **F2 相同短语必须保持无底色。** 特别是 `The order details ... following items:` 回归样例。
7. **浏览器截图验收尚缺。** 环境中此前没有 Playwright/Chromium；目前主要通过 Gradio config、API 和单测验证。
8. **执行工具的 workdir 偶尔回退到 `/data`。** 遇到时使用绝对路径或 `git -C /data/os_competition_TSJ`，
   不要误以为仓库消失。
9. **上下文窗口。** Codex 接口未暴露可验证的“是否精确 1M”数值；当前会话支持自动压缩。继续采用小步测试、
   小步提交和 checkpoint，不要依赖一次超长未提交修改。

## 8. 常用检查命令

```bash
# 全仓测试
PYTHONPATH=/data/os_competition_TSJ/agent-mem/src \
  /data/os_competition_TSJ/.venv/bin/python -m pytest -q \
  /data/os_competition_TSJ/agent-mem/tests

# 服务状态
curl -fsS http://127.0.0.1:7860/ -o /dev/null -w '7860=%{http_code}\n'
curl -fsS http://127.0.0.1:8000/health -o /dev/null -w '8000=%{http_code}\n'

# Git 状态（使用 -C，规避 cwd 偶发变化）
git -C /data/os_competition_TSJ status --short --branch

# 7860 tmux
tmux attach -t agent-mem-demo-7860
```

## 9. 下一对话启动 Prompt

将下面整段直接粘贴给新对话：

```text
请先完整阅读：
1. /data/os_competition_TSJ/docs/赛题与要求.md
2. /data/os_competition_TSJ/docs/技术设计方案.md
3. /data/os_competition_TSJ/docs/F2-F3-method-summary-20260725.md
4. /data/os_competition_TSJ/docs/F2-F3-frontend-integration-handoff-20260726.md
5. /data/os_competition_TSJ/docs/F2-F3-frontend-session-handoff-20260728.md

我们后续只关注 F2/F3 前端适配。当前分支 agent-mem-v1，F2/F3 最新提交为 c3c8141；
checkpoint 标签 f2-f3-frontend-longbench-v1 指向 69cd909。不要清理、回退或提交当前工作树中并行的
F5、引擎、校准和技术报告改动，尤其注意 chat_app.py 与 test_demo_app_build.py 同时含有已提交 F2/F3
内容和未提交 F5 内容。

当前页面已经完成：
- 一个“上下文优化任务”tab；
- F2 自动路由 τ-bench；
- F3 自动路由 LongBench；
- baseline 可选 F2/τ-bench 或 F3/LongBench 对照；
- 一套 Prompt/F2/F3 telemetry 面板；
- F2 两轨 diff 与精确短语锚点；
- F2/F3 after 面板只保留 token、元信息和 diff，不再重复正文。
- 前端上下文模式只暴露 baseline / F2 / F3；F2+F3 后端 preset 和路由仍保留，但当前前端不暴露按钮。
- 上下文优化任务已加入 `参数调整` 分组，F2 `压缩阈值` 一个输入同时控制首次压缩阈值和重压新增量。
- 上下文优化 Agent 对话已加外框、缩小字体，并区分 user 绿色气泡和 agent 白色气泡。
- 右侧实时监控已加 sticky 浮动样式，class 挂在内部 group 上。
- 顶部长说明和引擎控制说明文案已删除；引擎控制控件本身必须保留可见。

当前 7860 运行在 tmux session agent-mem-demo-7860，7860/7861 前端在线，8000 引擎当前未监听。
开始时请重新检查 git status、tmux、7860/7861/8000 和最新 pytest，不要只相信交接时状态。

先复述你理解的 F2/F3 当前架构、自动 workload 路由、结论边界和脏工作树风险，然后继续处理我接下来
指出的前端小问题。除非我明确要求，不要扩展到 F1/F4/F5，也不要修改正式 F2/F3 YAML 的 8000/4000
安全基线。
```
