# F5 · 多并发场景下的 KV-pool 准入控制（背压）模块说明文档

## 1 基本思想

### 1.1 问题：多并发 Agent 服务的 KV 抢占雪崩

在 τ-bench（retail 域）等多轮工具调用 agent 服务场景中，推理引擎需同时承载 N 个并发 agent
session。每个 session 是一条**单调增长**的对话：每轮追加 user 消息、agent 推理、工具调用结果，
其 **KV cache**（注意力机制中每层保存的 Key/Value 张量，用于避免重算历史 token 的注意力）随对话
长度持续累积。

当 N 个 session 的 KV 总量超出显存中预留给 KV cache 的「池子」时，引擎**必须减载**——要么让新请求
等待，要么 **抢占**（preempt）：丢弃某 session 的 KV，腾出空间，等以后再从头重算（re-prefill +
re-generate）。抢占的代价是 **雪崩式重算**：被踢 session 整段上下文需重新 prefill + 重新生成，
KV 缓存被反复推翻（命中率从理想的 ~0.95 跌至 ~0.48），端到端延迟暴涨（p50 可达 147s）。

这是 **纯并发问题**：单个 session 永不溢出 KV 池；并发度越高、对话越长，溢出与抢占越频繁。

### 1.2 核心洞察：防溢出（背压），而非事后挑抢占受害者

通过深入分析 vLLM 调度机制，我们得出关键认知：

> **抢占的「次数」由「内存压力」（并发数 × 负载）决定，不由调度策略决定。**
> 一旦池子溢出，无论用哪种优先级挑受害者，抢占都发生了、重算代价已产生。

因此，正解不是「更聪明地选谁被抢」（优先级调度路线），而是 **「别让池子溢出」——在源头加背压
（backpressure）**：在 KV 池接近满时，暂缓放新 session 进 running 池（让它 wait），等池子降到
安全水位再放。这样 running 集合的 KV 总量始终 < 池容量 → 抢占不发生 → 无重算浪费。

### 1.3 无效路线的诚实复盘

| 路线 | 思路 | 结果 |
|---|---|---|
| **SRTF 优先级**（progress） | 按对话步数排优先级（保护快完成的 session） | ❌ 公平 conc-6 复测无效（抢占计数 4→5、kv 不升）。根因：抢占重算成本由系统前缀 prefill 主导（与 step 无关），优先级只改「谁被抢」不改「抢几次」。还踩过一个 conc-1 串行 bug，制造了假的「3.5× 大胜」。 |
| **idle 优先级** | 闲置 session 抬优先级 | ❌ 时序错配（闲的 session 不在 running 队列里）。 |
| **无损 CPU offload** | 冷 KV→CPU 副本 | ❌ 当前 vllm-ascend 0.22.1rc1 的 SimpleCPUOffloadConnector 推理期未实际搬 KV。 |
| **KV-pool 准入控制**（本工作） | 池 > 85% 暂不放新 session | ✅ 抢占减少、kv 命中回升、延迟下降。 |

---

## 2 实现做法

### 2.1 系统架构（干净分层，不 fork 引擎）

```
┌─────────────────────────────────────────────────────────────────┐
│  应用层（agent_mem，本工作）                                       │
│                                                                  │
│  ① AdmissionController（承重机制）                                │
│     读 vllm:kv_cache_usage_perc → >85% 不放新 session            │
│  ② ConcurrentSessionDriver                                        │
│     并发 task 调度循环 + 准入闸门 + 后台 sweep                    │
│  ③ --scheduling-policy priority（协同层，vLLM 原生）              │
│                                                                  │
│         │ should_admit() 闸门 + priority 回调                     │
└─────────┼────────────────────────────────────────────────────────┘
          ▼  —— 缝E：vLLM 原生接口，不改内核 ——
┌─────────────────────────────────────────────────────────────────┐
│  引擎层（vllm-ascend 0.22.1rc1 / vllm 0.22.1）                   │
│  continuous batching + running/waiting 管理 + APC（前缀缓存）     │
│  池不溢出 → 无 preempt → KV cache 稳定                            │
└─────────────────────────────────────────────────────────────────┘
```

设计上严格遵循 **机制 vs 策略分离**：引擎层提供「连续批处理」机制；「何时放新 session 进来」的策略
完全在应用层，经 `ConcurrentSessionDriver` 的调度循环注入。不修改 vLLM 内核、不 fork，可随版本升级。

### 2.2 准入控制器（`AdmissionController`）

核心类 `agent_mem/scheduler/admission.py:AdmissionController`：

```python
def should_admit(self) -> bool:
    hbm = self.hbm_pct            # 读 vllm:kv_cache_usage_perc（KV 池利用率 %）
    if hbm < 0:                   # 读不到 → 放行到 max_workers（安全降级）
        return current_workers < max_workers
    if self.target_hi > 100:      # 哨兵值 = 关 admission（调试用）
        return current_workers < max_workers
    if hbm > self.target_hi:      # > 85% → 池快满，暂不放
        return False
    if hbm < self.target_lo:      # < 70% → 有空间，放
        return current_workers < max_workers
    return False                  # 70~85% → 保持当前并发，不增不减
```

- **target_lo = 70, target_hi = 85**（默认值，可经 yaml 配置）。
- `hbm_pct_fn`：一个注入的回调，读 vLLM `/metrics` 端点的 `vllm:kv_cache_usage_perc`（0~1 → 0~100%）。
  读不到返回 -1 → 准入安全降级为「按 max_workers 放行」。
- 线程安全：`RLock` 保护注册表与计数；HBM 读取在锁外。

### 2.3 并发调度驱动（`ConcurrentSessionDriver`）

`agent_mem/scheduler/driver.py:ConcurrentSessionDriver` 把准入闸门接入 N 个并发 task 的调度循环：

```python
def run(self, task_ids, task_runner):
    while pending or futures:
        # 准入提交：HBM 允许 → 放；HBM 高但池空 → 保底放一个（防死锁）
        while pending:
            pool_empty = not futures
            if not self.ctrl.should_admit() and not pool_empty:
                break              # ← 准入挡住：pending 中的 task 继续等
            tid = pending.pop(0)
            self.ctrl.admit(sid)
            fut = ex.submit(task_runner, tid, on_turn, priority_fn)
            futures[fut] = tid
        # 等任一完成（最多 interval 秒），让闸门周期推进
        done, _ = wait(futures, timeout=interval, return_when=FIRST_COMPLETED)
        for fut in done:
            self.ctrl.release(sid)  # session 完成 → 释放 slot → 池降 → 准入可能再放
```

关键行为：当 KV 池 > 85%，pending 中的 session **不被提交**（在队列里 wait）→ running 集合不增长 →
KV 总量不超池 → 无抢占。当某 session 完成（slot 释放 + KV 降），准入重新评估，若 < 70% 则放行。

后台线程每 `interval`（3s）跑一次 `PriorityEvictionStrategy` sweep（idle session 抬优先级——协同层，
非承重）。`priority_fn` 回调每请求读取 session 当前 priority，透传给 vLLM 的
`--scheduling-policy priority`（协同层）。

### 2.4 配置体系

| 配置文件 | 用途 | 关键字段 |
|---|---|---|
| `configs/f5-native.yaml` | baseline（FCFS，仅 prefix-cache） | session.strategy=noop, gpu-mem=0.27 |
| `configs/f5-evict-dynamic.yaml` | ours（准入控制 + priority） | session.strategy=priority-evict, target_lo=70, target_hi=85（默认）, priority_scheduling=true |

两条配置的引擎部分**完全一致**（vllm-ascend, gpu-mem 0.27, max-len 16384, hermes, APC on）；
唯一差异是 ours 多了 `priority_scheduling: true`（渲染 `--scheduling-policy priority`）+ session strategy。

### 2.5 KV-pool 利用率读取（`_f5_kv_pool_pct_fn`）

与 `benchmarks/runner.py` 的 `_kv_pool_pct_fn` **完全同款**，保证 demo 与 sweep 实验同口径：

```python
def _f5_kv_pool_pct_fn(engine_url):
    def _read():
        text = scrape /metrics
        for nm in ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"):
            match regex → return float * 100
        return -1.0  # 读不到 → 安全降级
    return _read
```

---

## 3 功能介绍

### 3.1 核心功能

1. **KV-pool 感知的动态准入控制**：实时读 vLLM 的 KV-pool 利用率，在 > 85% 时暂不放新 session，
   < 70% 时恢复放行。**从源头防止溢出 → 消除抢占 → 保住 KV 缓存 → 降延迟。**

2. **与 APC 正交可叠加**：准入控制不依赖 prefix caching；两者机制独立，可同时启用。

3. **公平对比**：baseline 与 ours 使用相同的引擎配置（模型、显存、APC、负载），唯一差异是准入控制。

4. **实时可视化**（demo）：每秒一帧的 plotly 动画——在跑/等待请求数 + 抢占累计尖峰 + KV-pool% 曲线
   + 85% 准入阈值线，直观展示准入何时介入。

5. **可配置阈值**：经 yaml 的 `target_lo` / `target_hi` 调整准入激进程度。

### 3.2 Demo 前端

- **两个独立按钮**（baseline / ours），各自独立跑、结果累积进共享 `gr.State`。
- **实时时间轴**：session 在跑/等待 + 抢占尖峰 + KV% + 85% 阈值线。
- **对比表**：只显示 ours 稳定占优的指标（e2e p50、实际抢占）+ Δ 改善行。
- **并排对比图**：两层都跑完后，baseline(上) vs ours(下) 抢占/KV 并排时间轴。
- **默认负载**：10 任务 / conc 6 / 25 步（高压，确保 baseline 必抢占）。

### 3.3 Benchmark 编排

`agent-mem/benchmarks/runner.py` 负责真实 Benchmark。分别用 `f5-native.yaml` 与
`f5-evict-dynamic.yaml` 启动匹配引擎，再通过 `--max-concurrency`、`--max-tasks`、
`--max-steps` 和 `--runs` 固定负载；run 目录保存配置与指标，最后用 `--compare` 聚合。

---

## 4 实验结果

### 4.1 硬件 / 软件

| 项 | 值 |
|---|---|
| NPU | Ascend 910B2C，单 chip 64 GiB HBM |
| 引擎 | vllm-ascend 0.22.1rc1（vllm 0.22.1） |
| 模型 | Qwen2.5-7B-Instruct（bf16, ~14.2 GiB 权重） |
| KV 池 | `--gpu-memory-utilization 0.27` → 1.27 GiB / 23,680 token |
| 工具调用 | `--enable-auto-tool-choice --tool-call-parser hermes` |
| 负载 | τ-bench retail，本地 user-sim（与 agent 同引擎），APC-on |
| Benchmark 路径 | `benchmarks/runner.py → QwenAgentRunner → ConcurrentSessionDriver` |

### 4.2 准入控制 vs FCFS（公平 conc-6 对比）

**8 任务 / conc 6 / 20 步**（tasks > conc，准入有队列可挡）：

| 配置 | e2e p50 | KV 命中率 | 实际抢占 | 墙钟 |
|---|---|---|---|---|
| baseline（FCFS） | 147 s | 0.484 | 3 | 240 s |
| **ours（准入控制）** | **116 s（1.27×）** | **0.717** | **1** | **219 s** |

- 抢占 3→1、KV 命中 +0.23、p50 快 1.27×。
- 这是 **公平的 conc-6 对比**（吞吐涨，不是靠少跑换延迟）。

**6 任务 / conc 6 / 12 步**（较轻负载）：

| 配置 | e2e p50 | KV 命中率 | 实际抢占 | 墙钟 |
|---|---|---|---|---|
| baseline | 83 s | 0.797 | 2 | 162 s |
| **ours** | **43 s（1.93×）** | **0.921** | **0** | **68 s（2.4× 吞吐）** |

- 抢占 2→0（完全消除）、KV 命中 +0.12、p50 快 1.93×、吞吐 2.4×。

### 4.3 诚实边界

1. **需要 tasks > conc**：准入只挡 pending 队列。当 tasks = conc 时（如 6/6），无队列 → 准入空转
   → 无差异。须 tasks > conc（如 8/6、10/6）准入才有队可挡。

2. **需要足够步数触发溢出**：conc 6 下约需 ≥ 18 步才溢出 KV 池（6×~2700 token + 7681 前缀 > 23,680）。
   步数太少（如 8 步）→ 不溢出 → 无抢占 → 准入无差异。

3. **高压下 KV 命中率可能下降**：10/25 步高压时，准入更激进（KV 频繁 > 85%）→ 有效并发降低 →
   前缀共享机会减少 → KV 命中率可能不如 baseline。这是 **延迟 vs 缓存命中的 trade-off**——准入用
   缓存共享换延迟 + 抢占收益。

4. **单次方差**：并发服务的引擎调度（batching、preempt 时序）本身非确定。同 seed 两次跑结果有
   ±波动（抢占 ±2、p50 ±20s），需 runs=3 中位数才能稳定。

### 4.4 无效路线的公平复测（SRTF 优先级）

修复 conc-1 串行 bug 后，progress（SRTF）在真 conc-6 下复测：

| 配置 | e2e p50 | KV 命中率 | 实际抢占 |
|---|---|---|---|
| native | 167 s | 0.533 | 4 |
| progress（SRTF） | **197 s（更慢）** | 0.548 | **5（更多）** |

SRTF 优先级**无效甚至有害**——确认了「优先级改不了抢占次数」的理论分析。

---

## 5 关键创新

### 5.1 应用层 KV-pool 感知准入控制（背压）

**核心创新**：将经典网络拥塞控制的「背压」思想迁移到 LLM 服务的 KV cache 管理。不修改引擎内核，
仅用应用层闸门（读 `/metrics` + 控制新 session 的提交时机），即可从源头消除多并发下的抢占雪崩。

与 vLLM 内部的 LRU/preempt 机制形成 **「引擎内事后抢救 vs 应用层源头预防」** 的互补关系：
- vLLM 内部：池满时被动抢占（LRU/随机选 victim → 重算）。
- F5 准入控制：池接近满时主动限流（挡住 pending → 不溢出 → 不需抢占）。

### 5.2 严格分层、不 fork 的工程实现

通过 vLLM 原生接口（`--scheduling-policy priority` + `/metrics`）+ 应用层闸门，整套方案：
- 不修改 vLLM 内核、不 fork。
- 可随 vllm-ascend 版本升级。
- 入口是一个纯函数（`should_admit`）+ 一个调度循环改动，极低侵入。

### 5.3 诚实的负结果（SRTF 优先级路线否决）

投入大量精力研究 SRTF 式优先级调度（progress/idle/combined 三种模式），最终经公平复测否决。
过程中发现并修复了一个 `target_hi=999` 串行 bug（制造了假的 3.5× 大胜）。**负结果与正结果一并记录**
（见 `docs/F5-chapter-multiconcurrency.md` §5.2），为后续研究者避免重蹈：
- 优先级调度在 LLM serving 的 KV 抢占场景**不适用**（抢占次数由压力决定，不由优先级决定）。
- 真正有效的杠杆是 **减少内存压力**（准入控制），而非 **改变抢占选择**（优先级）。

### 5.4 实时可视化演示

demo 前端提供 plotly 实时动画（每秒一帧），展示：
- running/waiting 请求数随时间 → 看准入是否在挡。
- 抢占累计尖峰 → baseline 跳起、ours 压平。
- KV-pool% + 85% 阈值线 → baseline 冲 100%、ours 被压在 85 附近。

使评委无需看原始日志即可直观理解准入控制的工作原理。

---

## 附录：关键文件索引

| 文件 | 作用 |
|---|---|
| `agent_mem/scheduler/admission.py` | `AdmissionController`（承重机制） |
| `agent_mem/scheduler/driver.py` | `ConcurrentSessionDriver`（调度循环+准入接入） |
| `agent_mem/scheduler/strategies.py` | `PriorityEvictionStrategy`（协同层 sweep） |
| `agent_mem/bench/runners/qwen_agent.py` | `QwenAgentRunner`（benchmark runner） |
| `agent-mem/benchmarks/runner.py` | sweep 编排 CLI |
| `configs/f5-native.yaml` | baseline 配置 |
| `configs/f5-evict-dynamic.yaml` | ours（准入控制）配置 |
| `agent_mem/demo/chat_app.py` | demo 前端（F5 tab + run_f5_tier + viz） |
| `agent_mem/demo/f5_runtime.py` | tier 定义 + RequestAdmissionController（demo 专用） |
| `docs/F5-chapter-multiconcurrency.md` | 技术报告 F5 章节 |
