# F2/F3 前端适配交接

- 日期：2026-07-26
- 后续开发基线：`agent-mem-v1`，当前 HEAD `5e205b0`
- F2 checkpoint：`1e64bc8`
- F3 checkpoint：`62a6488`
- F2/F3 总结文档提交：`e827f63`
- 前端：Gradio 5 + Plotly，入口 `python -m agent_mem.demo`

## 1. 下次工作的目标

下一阶段不是再次实现 F2/F3，也不是只在下拉框中显示配置名，而是让前端能够：

1. 选择 baseline、F2、F3、F2+F3，并确认实际构造的 middleware；
2. 运行能触发相应机制的 workload；
3. 实时展示 Prompt 变换、F2 压缩决策、F3 外置/fetch 和任务质量；
4. 在同一页面比较 token、调用轮次、延迟、错误和正确率；
5. 对实验结论使用准确口径，不把局部压缩率、不同轨迹或模型请求耗时误标为端到端收益。

应在整合分支 `agent-mem-v1` 上继续开发。`e827f63` 已是该分支祖先，不需要再从
`feat/f3-tool-data-lazyload` merge 或复制文件。

## 2. 当前仓库和运行状态

### 2.1 Git

```text
主 worktree: /data/os_competition_TSJ
分支:        agent-mem-v1
HEAD:        5e205b0 refactor(demo): 6 tab -> 一屏 dashboard

F2 worktree: /tmp/f2-wt  @ 1e64bc8
F3 worktree: /tmp/f3-wt  @ e827f63
F3 remote:   origin/feat/f3-tool-data-lazyload @ e827f63
```

当前主 worktree 有与本任务无关的未跟踪内容：

```text
.venv-lats/
huggingface_home/docs/
```

不要删除、清理或提交它们。

### 2.2 服务和数据

交接检查时：

- Gradio 已运行在 `http://127.0.0.1:7860`，HTTP 200；
- 进程命令为 `.venv/bin/python -u -m agent_mem.demo ...`；
- `http://127.0.0.1:8000` 的 vLLM 引擎离线；
- LongBench 数据存在于 `/tmp/longbench-data.zip`；
- F2 worker Python 应使用 `/data/os_competition_TSJ/.venv-compress/bin/python`。

服务状态会变化，下次开始时必须重新检查，不能只相信本段记录。不要终止已有前端进程，除非需要
加载新代码；若要重启，先确认它不是其他人正在使用的共享服务。

## 3. 必读资料

优先顺序：

1. `docs/F2-F3-method-summary-20260725.md`：完整方法、用法、实现、归属和结果；
2. 本文件：前端现状、缺口和实施建议；
3. `docs/F3-tool-data-lazyload-report.md`：F3 接口和质量演进；
4. `docs/F2-F3-extended-evaluation-20260725.md`：2Wiki 100 条和 tau-bench 5 条；
5. `docs/F2-F3-implementation-attribution.md`：自研/直接使用/设计借鉴边界。

不要重新从旧的“F2 上下文下降 62%”开始推导。最终 retail full115 的严格完整 Prompt 收益是
19.32%，且该轮动态 BERT 触发 0 次；F3 2Wiki first100 的累计 Prompt 收益是 66.03%。

## 4. 前端当前结构

### 4.1 页面入口

`agent-mem/src/agent_mem/demo/chat_app.py` 当前是一屏 dashboard：

- 顶部折叠的架构总览；
- 常驻引擎控制；
- 左侧 `跑 Benchmark` / `对话演示` 两个 tab；
- 右侧实时 vLLM/NPU 监控和历史结果对比。

页面使用闭包持有 `EngineHandle` 和 `BenchHandle`。后台 benchmark 在线程中运行，Gradio Timer
轮询状态。不要把带 `Lock` 或活动线程的 handle 改回 `gr.State`，Gradio deepcopy 会断开状态。

### 4.2 已经接通的 F2/F3 基础

以下内容已经实现，不要重复造轮子：

- `_MEANINGFUL_PRESETS` 包含 `f2-compress`、`f3-lazyload`、`f2-f3-combined` 和
  `unified-longbench`；
- preset preview 能显示 `cfg.middleware.active`；
- `demo/bench_runner.py::_build_runner()` 调用 `middlewares_from_config(cfg)`；
- `QwenAgentRunner` 通过 suite registry 同时支持 `tau-bench` 和 `longbench`；
- `longbench_adapter.py` 会把 2Wiki context 变成 `retrieve_documents` 工具结果，再走
  F2/F3 middleware；
- F3 内部 `fetch_tool_result` 会绕过业务工具环境；
- `unified-longbench.yaml` 已声明 `[lazyload, compress]`。

因此“配置开关到 middleware 构造”的后端路径已经存在。前端专项工作的重点是 workload 参数、
telemetry、可视化和用户工作流。

### 4.3 相关代码

```text
agent-mem/src/agent_mem/demo/chat_app.py          页面、callback、Timer
agent-mem/src/agent_mem/demo/bench_runner.py      后台 benchmark 桥
agent-mem/src/agent_mem/demo/monitor.py           vLLM/NPU 通用监控
agent-mem/src/agent_mem/demo/overview.py          静态功能说明/KPI
agent-mem/src/agent_mem/demo/engine_control.py    引擎起停
agent-mem/src/agent_mem/demo/tau_bench_ui.py      旧流式 tau UI，目前未接入主 dashboard

agent-mem/src/agent_mem/middleware/compress.py    F2 事件来源
agent-mem/src/agent_mem/middleware/lazyload.py    F3 事件来源
agent-mem/src/agent_mem/agent/usage_log.py        paired Prompt 日志
agent-mem/src/agent_mem/bench/tasks/longbench_adapter.py
agent-mem/src/agent_mem/bench/runners/qwen_agent.py
```

## 5. 已确认的前端缺口

### 5.1 `unified-longbench` 无法直接从页面运行

`configs/unified-longbench.yaml` 的 `benchmark.data_zip` 为空，CLI 可以用 `--data-zip` 覆盖，
但当前 Gradio 页面没有数据路径、start 或 limit 输入，也没有给 `BenchHandle` 传 override。
直接点运行会在 `longbench_adapter.list_tasks()` 报错。

前端应增加数据路径输入，默认可探测 `/tmp/longbench-data.zip`，并支持 start/limit。不要为了 demo
把机器本地 `/tmp` 路径提交进通用 YAML。

### 5.2 页面看不到 F2/F3 是否真正触发

当前右侧只显示 HBM、KV hit、吞吐、TTFT、e2e、running/waiting。F2/F3 的核心信息在三个
JSONL 中，页面完全未读取：

```text
F2_EVENT_LOG
F3_EVENT_LOG
PROMPT_TOKEN_LOG
```

因此用户只能看到 preset 名，无法区分：

- F2 是 `skip`、`compress` 还是 `reuse`；
- token 节省来自 static system/tools 还是动态 BERT；
- F3 观察了多少工具结果、外置了多少、为什么 passthrough；
- reference 从多少 token 降到多少；
- 模型是否调用 fetch、fetch 是否截断；
- paired canonical/transformed Prompt 到底减少多少。

### 5.3 自由对话绕过我们的 F2/F3 Agent loop

`chat_app.py::_build_assistant()` 直接构造 Qwen-Agent `Assistant`。该路径没有
`MiddlewareStack`，所以选择 F2/F3 preset 不会影响自由对话。页面目前也没有明确说明这一点。

可选做法：

- 推荐：新增独立的“上下文优化演示”工作流，复用统一 runner/LongBench，而不是强行把 F3 塞进
  没有大型业务工具的自由聊天；
- 若要让自由聊天演示 F2，需改用项目 `run_react` 或为 Qwen-Agent 请求增加等价 middleware
  接点；
- F3 只有产生大型工具结果才有意义，普通聊天无法证明 F3。

### 5.4 retail tau-bench 不是 F3 演示场

已有 5-task MIMO 探针中，27 个工具结果最大只有 1,416 Qwen token，F3 外置 0 次。页面若用
`f3-lazyload.yaml` 跑标准 retail，正确表现很可能是 no-op，不应把它展示成 F3 失效。

F3 主演示应使用：

- `unified-longbench` / 2Wiki 工具化 workflow；或
- 360-record controlled JSON locator；或
- 其他真实单次结果 `>=4,000` token 的工具 workload。

retail 可以保留为“阈值以下安全 passthrough”演示，但必须明确标注。

### 5.5 baseline/F2/F3/F2+F3 缺少同 workload 四档入口

现有 longbench preset 只有组合档。为了让评委在同一数据和同一引擎上对比，建议提供四个模式：

```text
baseline  -> []
F2        -> [compress]
F3        -> [lazyload]
F2+F3     -> [lazyload, compress]
```

优先保持 config-driven 设计。可以新增四份精简 preset，或者在 UI 生成临时配置副本并显式显示
override；不要静默修改共享 config 对象或 tracked YAML。

F2-only LongBench 必须包含 `hot_tool_trigger_tokens: 1000`，否则短 Agent 轨迹中的大型结果可能一直
留在 hot tail，8k cold gate 不会触发。组合档必须保持 `[lazyload, compress]` 顺序。

### 5.6 引擎选择和 middleware 选择耦合过紧

F2/F3 是 Agent 层功能，切换 baseline/F2/F3 通常不需要重启相同的 Qwen/vLLM-Ascend 引擎。
当前 UI 用同一个 preset 同时控制引擎和 benchmark，容易让用户每次切 middleware 都重载权重。

建议把控件分成：

- 引擎 preset/状态：模型、C8、KV connector、调度等真正影响 server 的项；
- workload：tau-bench / 2Wiki、数据和任务范围；
- 上下文模式：baseline / F2 / F3 / F2+F3。

### 5.7 页面文案有需要校准的地方

`overview.py` 当前存在过度简化：

- F2 写“热尾+system 原样”，但最终 F2 会确定性 compact retail system/tools；
- F3 写“纯自研”，准确表述应是“借鉴 DeerFlow 设计、无代码依赖/复制，存储/检索/集成为独立实现”；
- F3 KPI 写“成功率下降 0”，更严谨的是“2Wiki first100 总体 31/100，与 baseline 持平；95 个
  可比任务 30/95 vs 31/95，`p=1.0`”；
- F2 success 不能笼统宣称已严格满足 `<=2pp`；最终 full115 是 23/115，历史 baseline 为
  23/115 和 26/115；
- 2Wiki 的 p50 是“每题所有模型 HTTP 请求耗时之和的中位数”，不是完整端到端耗时。

## 6. 推荐的前端体验

在现有左侧 tabs 中新增一个真正可用的 `F2/F3 上下文优化` tab，保持右侧通用监控常驻。

### 6.1 控制区

- workload 菜单：`2Wiki 多跳检索`、`retail passthrough`，可选 controlled JSON；
- 四档 segmented control：baseline / F2 / F3 / F2+F3；
- LongBench zip 路径；
- start、limit、重复次数；
- 运行/停止或至少“已有任务运行时禁用再次启动”；
- 当前实际 middleware、阈值和数据范围的只读摘要。

### 6.2 运行摘要

建议首屏展示这些可比较指标：

| 指标 | 来源 | 说明 |
|---|---|---|
| canonical Prompt | `PROMPT_TOKEN_LOG.original_prompt_tokens` | paired 原始请求 |
| transformed Prompt | `PROMPT_TOKEN_LOG.transformed_prompt_tokens` | 实际变换后请求 |
| saved token / percent | paired log | 最可信的 F2/F3 token 收益 |
| success / total | benchmark result | 质量红线 |
| model calls | prompt log 行数或 adapter telemetry | 解释 F3 额外轮次 |
| context errors | task error | F3 溢出规避 |
| model-request p50 | LongBench adapter | 不能标成完整 e2e |

### 6.3 F2 面板

- `compress / reuse / skip` 计数；
- static system compact 次数、tool descriptions replacement；
- hot tool compress 次数与 saved tokens；
- cold/compressible/new token 曲线；
- `compress_ms` p50/p95；
- 最近事件表：session、step、action、reason、before/after、耗时。

必须让 `skip` 成为正常、可解释状态。例如 retail full115 的最终配置动态 BERT 0 次，但 static
Prompt 仍节省 19.32%。

### 6.4 F3 面板

- observed result、passthrough、externalize、fetch 计数；
- original/reference/fetch token；
- saved token/percent；
- content type、result ID 的缩短显示、source/token truncated 状态；
- local store/fetch latency；
- 一条可展开的流程：business result -> reference -> fetch arguments -> bounded result。

不要在页面展示完整外置数据或跨 session artifact；result ID 只显示短前缀即可。

### 6.5 任务级轨迹

允许选择某个 task，展示：

```text
问题
  -> retrieve_documents
  -> 原结果 token
  -> F3 reference 或 F2 compressed content
  -> fetch_tool_result 参数（若有）
  -> 最终答案 / gold / success
```

这比只展示全局柱状图更能说明为什么 Prompt 少了、为什么 F3 可能增加模型轮次。

## 7. 建议的数据层实现

### 7.1 新建纯 Python telemetry 模块

建议新增 `agent_mem/demo/context_telemetry.py`，不要把 JSONL 解析堆进 `chat_app.py`。它应提供：

- 容忍文件不存在、空文件和最后一行尚未写完的增量读取；
- F2/F3/prompt 三类 typed summary；
- 按 `(session_id, step)` 合并事件；
- 聚合与最近事件快照；
- 不依赖 Gradio，便于单测。

### 7.2 每次 run 使用独立日志目录

例如：

```text
logs-demo/context/<timestamp>_<preset>/
  f2-events.jsonl
  f3-events.jsonl
  prompt-tokens.jsonl
```

F2/F3 构造器原生支持 `event_log` 参数，优先通过临时 config options 注入：

```python
cfg.middleware.options["compress"]["event_log"] = f2_path
cfg.middleware.options["lazyload"]["event_log"] = f3_path
```

`PROMPT_TOKEN_LOG` 当前只支持环境变量。Gradio 是单进程，设置它时必须：

- 禁止同时启动第二个 benchmark；
- 保存旧环境值并在 `finally` 恢复；
- 后续如需要真正并行多 run，再把 prompt log path 改成 RunContext/runner 显式参数。

不要让多个前端运行共享一个固定 `/tmp/f2-events.jsonl`，否则 session 和实验会混在一起。

### 7.3 扩展 `BenchHandle`

可增加：

```text
telemetry_dir
suite / variant / data range
current_task / completed_tasks
telemetry_summary
recent_events
```

更新仍必须在锁内，snapshot 返回副本。新增 run 前若状态为 queued/running，应拒绝重复启动。

### 7.4 LongBench overrides

`run_bench_async` 可接受 `data_zip/start/limit/variant`，后台加载 config 后只改当前内存中的 cfg：

```python
cfg.benchmark.data_zip = data_zip
cfg.benchmark.options["start"] = start
cfg.benchmark.options["limit"] = limit
cfg.middleware.active = selected_active
```

要校验 zip 存在且包含 `data/2wikimqa.jsonl`，错误应在提交任务前显示，而不是等后台 traceback。

## 8. Telemetry 字段速查

### 8.1 F2 event

常用字段：

```text
session_id, step, action, reason
cold_tokens, compressible_cold_tokens, hot_tokens, new_tokens
origin_tokens, compressed_tokens, est_compressed_tokens
compress_ms, compress_count, frozen_count
hot_tool_compressed, hot_tool_saved_tokens, hot_compress_ms
system_prompt_compacted, tool_descriptions_replaced
sent_tokens, token_count_source
```

`action` 为 `skip/compress/reuse`。不是每条事件都有全部字段，解析器必须使用可空值。

### 8.2 F3 event

常用字段：

```text
session_id, step, action, tool_name, result_id
content_type, byte_count
original_tokens, reference_tokens, saved_tokens, saved_percent
fetch_tokens, source_truncated, token_truncated
store_ms, elapsed_ms, token_count_source
```

常见 action：`passthrough`、`externalize`、`fetch`、`store_error_passthrough`。

### 8.3 paired Prompt log

```text
session_id, step, prompt_tokens
original_prompt_tokens, transformed_prompt_tokens
saved_tokens, saved_percent, meter_ms
tokenizer_source, tokenizer_drift
```

累计 token 应对各行求和后再计算百分比，不能对逐步 percent 做简单平均。

## 9. 验收标准

### 9.1 无 NPU 单测

- telemetry parser 能处理缺文件、部分 JSONL、不同 event shape；
- baseline/F2/F3/F2+F3 映射正确，组合顺序固定；
- LongBench path/start/limit override 正确且不写回 tracked YAML；
- benchmark 重复启动被拒绝；
- `build_app()` 在无引擎时仍可构造；
- 页面文案使用本文件的准确口径。

### 9.2 本地 smoke

- Gradio 可启动，离线状态不会报错；
- `/tmp/longbench-data.zip` 可从页面运行 1 个 task；
- baseline 无 F2/F3 event；
- F3 至少出现一次 `externalize` 和一次 `fetch`；
- F2/F3+F2 的 Prompt log 有 original/transformed paired 值；
- 完成后页面展示 success、token 和调用轮次；
- 错误、空日志和引擎掉线都有明确状态。

### 9.3 浏览器验收

使用 Playwright 检查桌面和移动 viewport：

- 无文字/图表重叠；
- 长 result ID、路径和错误信息不会撑破布局；
- run 中、完成、失败、无触发四种状态清晰；
- 控件不会因动态 label 改变尺寸；
- 图表非空，Timer 刷新不导致页面跳动。

## 10. 推荐实施顺序

1. 新增 telemetry parser 和单测；
2. 给 `BenchHandle/run_bench_async` 增加独立日志目录和 LongBench override；
3. 增加四档 mode 与数据路径/start/limit 控件；
4. 加 F2/F3 summary 和最近事件表；
5. 加 task trace；
6. 校正文案；
7. 跑无 NPU 测试、1-task 本地 smoke 和 Playwright 截图；
8. 再决定是否让自由聊天接入 F2，不要让这一项阻塞核心演示。

## 11. 运行和验证命令

从仓库根目录：

```bash
PYTHONPATH=agent-mem/src .venv/bin/python -m pytest -q agent-mem/tests
.venv/bin/ruff check agent-mem/src/agent_mem agent-mem/tests agent-mem/benchmarks
```

启动前端：

```bash
PYTHONPATH=agent-mem/src .venv/bin/python -u -m agent_mem.demo \
  --configs-dir agent-mem/configs \
  --model-path models/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 7860
```

若已有 7860 进程，使用其他端口或确认后再重启。前端 URL 只在服务器本机可见时，可用
`ssh -L 7860:localhost:7860 <server>` 转发。

用于对照后端是否正常的 CLI：

```bash
PYTHONPATH=agent-mem/src .venv/bin/python agent-mem/benchmarks/runner.py \
  --config agent-mem/configs/unified-longbench.yaml \
  --runner qwen-agent \
  --engine-url http://127.0.0.1:8000/v1 \
  --data-zip /tmp/longbench-data.zip --start 0 --limit 1 --runs 1
```

## 12. 必须保持的结论边界

- F2 retail full115 完整 Prompt -19.32%，但动态 BERT 0 次；
- F2 generic 28.9k trace -27.46%，首次 BERT 36.27 秒，不是任务正确率结果；
- F3 synthetic Prompt -92.98%，受控 locator 3/3；
- F3 2Wiki first100 Prompt -66.03%，总体 31/100 与 baseline 相同；
- F3 2Wiki p50 模型请求耗时 +36.05%，原因是 240 -> 368 次模型调用；
- baseline 有 5 次 32k context error，F3 为 0；
- F2+F3 为 27/100，4 个答案点差未显著但需要复测；
- vLLM 预分配 KV pool，Prompt/KV token 下降不能直接写成进程 HBM peak 已下降；
- F2 直接使用 LLMLingua-2；F3 借鉴 DeerFlow 设计但代码独立实现。

前端可以展示历史实测数字，但必须同时显示 workload、样本量和指标口径。实时 run 的数据与历史
报告应视觉区分，避免把静态 headline 当成本次运行结果。

## 13. 交接时验证基线

2026-07-26 在 `agent-mem-v1@5e205b0` 运行：

```text
pytest: 297 passed, 17 warnings
ruff:   15 个既有问题
```

warnings 主要来自 Torch/Gradio deprecation。Ruff 问题不是本交接文档引入的，其中一个功能性问题
是 `bench/runners/qwen_agent.py::_run_dynamic()` 最后使用未定义的 `task_ids`；其余主要是 import
顺序、延迟 import、未使用变量和文件末尾换行。F2/F3 的顺序/单并发 LongBench 路径不进入该
dynamic 分支，但后续跑全仓 Ruff 或 F5 动态并发前需要与整合分支一起处理。
