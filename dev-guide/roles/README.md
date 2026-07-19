# 三人协作模型（Model C：2 特性开发 + 1 测量/质量中枢）

> 本文件定义三人并行开发的**协作契约**。三份角色清单见同目录：
> [`p1-engine-backend.md`](p1-engine-backend.md) · [`p2-agent-app.md`](p2-agent-app.md) · [`p3-measure-quality.md`](p3-measure-quality.md)
>
> 总方案：[`docs/技术设计方案.md`](../../docs/技术设计方案.md)；路线图：[`../roadmap.md`](../roadmap.md)。

## 为什么是 Model C（而不是纯模块切分 / 纯审查角色）

评分里 **70%**（应用效果 40% + 代码规范 20% + 文档 10%）取决于「能测、测得准、可复现、文档全」——这些是**横切性质量工作**，不是某个人的功能模块。所以专门留一个人（P3）守测量骨干 + 质量门，是最契合评分指挥棒的结构。

Model C 解决了两个具体痛点：

1. **MVP 假并行**：MVP 三块是严格调用链 `benchmark → agent → vLLM`，纯按文件切会互相卡。Model C 让 P3 **第一天先交付契约 + 桩**，P1/P2 对着同一个契约并行开发，不被互相阻塞。
2. **纯审查角色空转**：小体量 MVP 里专职审查者容易没东西看。Model C 让 P3 把**测量骨干**（harness + 6 指标 + 对照报告 + CI）当主线，审查是顺带的验收——永远有事做，永远是瓶颈的反面。

## 三个角色一句话定位

| 角色 | 主线 | MVP 解锁物 | 后续模块 |
|---|---|---|---|
| **P1 引擎/后端** | 推理引擎启停 + KV/分层机制 | vLLM V1 serve + prefix-cache 开关 + `/metrics` | M1 / M4 / M5 / M8 / M9 / M11 |
| **P2 Agent/应用** | Agent 编排 + 中间件 | Qwen-Agent ReAct + 多轮工具 + CLI | M2 / M6（+ 协同 M5） |
| **P3 测量/质量中枢** | benchmark 骨干 + 质量门 | **契约 schema + stub server + harness 骨架（最先交付）** | M3 / M10 + 全模块 before/after |

## 接口契约表（并行的命脉 —— 谁产出、谁消费）

> **契约先行**是 Model C 能真并行的核心。P3 在 day 1 把下面这些接口钉死并落到 main，P1/P2 此后只对着接口写实现。

| 契约 | 产出方 | 消费方 | 落点 |
|---|---|---|---|
| 引擎 OpenAI 兼容 API（`/v1/chat/completions` + tool_call） | P1 | P2、P3 | vLLM 默认即可，P1 保证 tool calling 可用 |
| 引擎 `/metrics` Prometheus 端点 | P1 | P3（抓 KV 命中率） | 关注 `vllm:gpu_prefix_cache_hits_total` |
| `configs/*.yaml` 配置 schema（engine / benchmark / metrics） | P3 | P1、P2 | [`agent-mem/configs/`](../../agent-mem/configs/) 已有雏形 |
| Agent CLI / 任务入口（输入 prompt → 输出 trace） | P2 | P3（驱动 benchmark） | `python -m agent_mem.cli ...` |
| `metrics.json` schema（6 大指标） | P3 | 全员 | 见 [`../log-naming-convention.md`](../log-naming-convention.md) |
| stub OpenAI server（P1 引擎没好之前给 P2 用） | P3 | P2 | `benchmarks/` 下 mock |

## Git 并行约定

- 每个任务一条短命 feature 分支：`feat/p1-<module>`、`feat/p2-<module>`、`feat/p3-<module>`。
- **MVP 集成顺序**（必须按依赖走，否则互相卡）：
  1. **P3**：`contract + stub server + harness 骨架 + config schema` → 直接 merge 到 main（day 1，解阻塞）
  2. **P1**：`vllm serve + prefix-cache 开关 + /metrics` → PR，**P3 审查**，merge
  3. **P2**：`Qwen-Agent ReAct + tools + CLI` → PR，**P3 审查**，merge（P2 之前对着 stub 写，此时切真实引擎）
  4. **P3**：`τ-bench 接入 + 6 指标采集 + 三档对照` → 收尾集成 benchmark
  5. **P1 + P3**：`docker compose` 一键启动
- 所有 PR 由 **P3 审查**（P3 是质量门）；P1/P2 互审可并行加速。
- commit 规范：每个优化模块独立分支 / 独立提交，**附 before/after 数字**（见 [`../README.md`](../README.md) 工作约定 5）。

## 共同红线（所有人都要守）

- **成功率红线**：任何优化不得使 τ-bench 任务成功率下降 > 2 个百分点（赛题硬指标）。
- **公平对照**：同硬件 / 同模型 / 同 prompt 集 / 同 seed，before/after 各跑 **3 次取中位数**。
- **日志规范**：所有 run 按 [`../log-naming-convention.md`](../log-naming-convention.md) 落盘到 `../logs/`（全 ASCII、可 glob）。
- **复现性**：每个 run 目录必须含 `config.yaml` + `git_commit.txt` + `env.txt`。
- **NPU/GPU 设备**：NPU 默认是停的。任何要真跑 NPU/GPU 的操作（P1 的 M8 Ascend、P3 的真机 benchmark）**先暂停等用户启动设备**，不要假设设备在线。

## M1–M11 模块归属矩阵（扩展阶段怎么分）

> 优先级按 ROI：`M8(openEuler) ＞ M1 ＞ M5 ＞ M4 ＞ M2 ＞ M3 ＞ M6 ＞ M8(Ascend) ＞ M10 ＞ M11 ＞ M7 ＞ M9`

| 模块 | 主责 | 协作 | 说明 |
|---|---|---|---|
| M1 KV FP8 量化 | **P1** | P3 出数字 | 一行 flag，KV 显存砍半 |
| M2 Prompt 压缩 | **P2** | P3 验成功率 | LLMLingua + 工具 schema 去重 |
| M3 session-aware 调度 | **P3** | P1 接 connector | 复用 SharedStorageConnector |
| M4 LMCache 分层 | **P1** | P3 出数字 | GPU↔CPU↔SSD 三级 |
| M5 SGLang 双引擎 | **P1**(server) | P2 跑通 agent loop / P3 对比 benchmark | 命中分支共享 + 适配不同框架 |
| M6 工具数据 lazy-load | **P2** | P3 验 | Redis/SQLite + 引用语法 |
| M7 显式分支 CoW | **P1** | — | 研究性，需构造 tree-search（低优先） |
| M8 国产化 openEuler+Ascend | **P1** | P3 在容器内跑 benchmark | openEuler 必做，Ascend 建议做 |
| M9 Mooncake 分布式 KV 池 | **P1** | — | 仅多机（低优先） |
| M10 session KV checkpoint | **P3** | 与 M3 合并叙事 | 跨重启持久化 |
| M11 Semantic Router | **P1/P3** | 文档为主 | 技术前瞻 |
| 横切：全模块 before/after 测量 + 技术报告 | **P3** | P1/P2 提供实现 | 每个模块都要数字证明 |

## 每个优化模块的交付标准（叙事模板，所有人通用）

1. **现象**：agent 场景观察到的具体内存问题（带数据）
2. **根因**：为什么（agent 特征）
3. **机制**：优化方案（技术原理）
4. **验证**：before/after 数字（显存↓、延迟↓、成功率不掉）
5. **权衡**：代价（如压缩的成功率损失）

> 详见 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md) 第八部分。
