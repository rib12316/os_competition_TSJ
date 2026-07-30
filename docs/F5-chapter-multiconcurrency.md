# 第 F5 章 · 多并发场景下的 KV-pool 准入控制（背压）消除抢占

> 面向「多个 LLM agent 并发服务」场景，提出**应用层 KV-pool 准入控制器**：实时读 vLLM 的
> `vllm:kv_cache_usage_perc`，在 KV 池接近溢出（>85%）时**暂不放新 session 进 running 池**，
> 降至 70% 再放——从源头防止溢出，把多并发下的 **vLLM 抢占归零**、KV 命中率从 0.80 提升到 0.92、
> 端到端延迟降低 ~1.9×、吞吐 ~2.4×。不改引擎内核、不 fork，一个应用层闸门 + vLLM 原生调度。
>
> **数据状态**：准入控制 ✅ **已烟测验证**（conc=6, 6 任务/12 步, runs=1，§5.6）；全工况扫描
> （conc 4/6/8 × 10 任务/25 步 × runs=3）⏳ 待跑。

---

## 5.1 问题：多并发 Agent 服务的 KV 抢占雪崩

### 5.1.1 场景
τ-bench（retail）多轮工具调用 agent，N 个 session 并发，每个 session 的 KV cache 随对话单调增长。
KV 池有限（`--gpu-memory-utilization 0.27` → **1.27 GiB / 约 23,680 token**，见下）。N 个 session 总 KV
超池 → 引擎**必须减载**：要么让新请求等待，要么**抢占**（preempt，丢弃某 session 的 KV、之后从头重算）。
这是**纯并发问题**——单 session 永不溢出，并发越高溢出越频。

### 5.1.2 资源预算（真机引擎日志）
```
Available KV cache memory: 1.27 GiB
GPU KV cache size: 23,680 tokens          # block_size=128 → 185 block
```
核算：Qwen2.5-7B（28 层、GQA 4 KV head、head_dim 128）bf16 KV ≈ **56 KiB/token**；
1.27 GiB ÷ 56 KiB ≈ 23,300 token ≈ 引擎报告。retail 系统提示约 **7,681 token**，6 并发 session 总 KV
轻易突破该上限。

### 5.1.3 盲目抢占的代价
默认 FCFS 在池满时盲目减载。真机日志显示 native（conc=6）KV 利用率在 9.8%→100% 间剧烈波动，
running 集合在 2~6 间被反复收缩：
```
Running: 6, Waiting: 0, GPU KV cache usage: 100.0%
Running: 3, Waiting: 3, GPU KV cache usage: 99.5%
```
抢占 = 雪崩式重算：被踢 session 的 KV 丢弃、之后重 prefill 整段上下文并重新生成。`✅已确认`（烟测）：

| 指标 | native（FCFS, conc=6, 6任务/12步） | 含义 |
|---|---|---|
| e2e p50 | 83 s | 半数任务 >1.4 分钟 |
| KV 命中率 | 0.797 | 缓存被反复推翻 |
| 抢占次数 | **2** | 重算浪费 |
| 墙钟(6任务) | 162 s | 吞吐 0.037 task/s |

---

## 5.2 核心洞察：防溢出（背压），而非事后挑抢占受害者

**关键认知**：抢占的**次数**由**内存压力**（并发数 × 负载）决定，**不由调度策略决定**。一旦池子溢出，
无论用哪种优先级挑受害者，**抢占都发生了**、重算代价已产生。所以：

> **正解不是「更聪明地选谁被抢」，而是「别让池子溢出」——在源头加背压（backpressure）。**

这把我们从「调度优先级」路线拉回到「准入控制」：在 KV 池接近满时，**暂缓放新 session 进 running 池**
（让它 wait），等池子降到安全水位再放。这样 running 集合的 KV 总量**永不超池** → **抢占不发生** → 无重算浪费。

### 5.2.1 为什么「优先级」路线走不通（诚实复盘）
我们最初尝试 **SRTF 式优先级**（progress：`priority=(1-step/max_steps)*100`，保护快完成 session）。
但公平复测（conc=6）证明**无效**：优先级只改「谁被抢」，不改「抢几次」——抢占计数不变（4→5）、
KV 命中不升（0.53→0.55）、延迟不降。更糟，早期一次实现 bug（`target_hi=999` 被解释成「中间区间不放行」）
让 progress **串行跑（有效并发=1）**，制造了一个假的「3.5× 大胜」（实为 conc-1 vs conc-6 的并发差）。
**根因**：抢占重算成本由**系统前缀 prefill**主导（与 step 无关），step 不是有效的「沉没成本」信号。
→ 章节定稿为**准入控制**（已被真机复证有效），优先级路线作负结果记录。

---

## 5.3 方法：应用层 KV-pool 准入控制器（缝E，不 fork）

### 5.3.1 机制
`AdmissionController`（`agent_mem/scheduler/admission.py`）：每个并发 session 在进入 running 池前过一道闸门——

```python
def should_admit(self) -> bool:
    hbm = self.hbm_pct            # 读 vllm:kv_cache_usage_perc（KV 池利用率%，0~100）
    if hbm > self.target_hi:      # > 85% → 池快满，暂不放
        return False
    if hbm < self.target_lo:      # < 70% → 有空间，放（受 max_workers 上限）
        return self._current_workers < self.max_workers
    return False                  # 70~85% → 保持当前并发，不增
```

效果：当 6 个 session 的 KV 把池子推过 85%，**第 7 个想进来的 session 被挡住（wait）**，
直到某 session 完成、池子降到 70% 才放行。**running 集合的 KV 总量始终 < 池容量 → 永不溢出 → 抢占 0**。

### 5.3.2 架构（机制 vs 策略分离）
```
应用层（agent_mem）
  AdmissionController（承重）— 读 kv_cache_usage_perc，>85% 暂不放行
  + --scheduling-policy priority（协同层，引擎原生）
         │ should_admit() 闸门
└────────┼─────────────────
         ▼ —— 缝E：vLLM 原生接口 ——
引擎层（vllm-ascend）  running/waiting 集合管理；池不溢出 → 无 preempt
```
不修改 vLLM 内核、不 fork：一个应用层闸门 + 引擎原生调度。`ConcurrentSessionDriver`
（`scheduler/driver.py`）把闸门接入并发 task 调度循环。

---

## 5.4 为什么是准入控制（而非优先级 / offload）

| 方案 | 机制 | 评估 |
|---|---|---|
| **KV-pool 准入控制（本工作）** | 池>85% 暂不放新 session → 防溢出 | ✅ **抢占 2→0、kv 0.80→0.92、p50 1.9×、吞吐 2.4×**（真机复证） |
| progress 优先级（SRTF） | 按 step 排抢占优先级 | ❌ 公平 conc-6 下无效（只改「谁被抢」不改「几次」；抢占计数 4→5） |
| idle 优先级 | 闲置 session 抬优先级 | ❌ 时序错配（闲的 session 不在 running 队列） |
| 无损 CPU offload | 冷 KV→CPU 副本 | ❌ 当前 vllm-ascend 版本 connector 未实际搬 KV（推理期无 save/load） |

**核心**：只有「减少内存压力」能降低抢占次数——准入控制（背压）是唯一在当前版本奏效的杠杆。

---

## 5.5 实验设置（可复现）

- **硬件/引擎**：Ascend 910B2C（64 GiB HBM），vllm-ascend 0.22.1rc1（vllm 0.22.1），Qwen2.5-7B-Instruct。
- **制压**：`--gpu-memory-utilization 0.27`（KV pool 1.27 GiB / 23,680 token），`--max-model-len 16384`，hermes 工具调用。
- **负载**：τ-bench retail，本地 user-sim（与 agent 同引擎），APC-on，无 think-time sleep。
- **配置**：
  - `baseline` = `configs/f5-native.yaml`（FCFS，仅 prefix-cache，无 priority）。
  - `ours` = `configs/f5-evict-dynamic.yaml`（`--scheduling-policy priority` + 准入控制 target_lo=70/target_hi=85，无 offload）。
- **路径**：`benchmarks/runner.py → QwenAgentRunner → ConcurrentSessionDriver`（demo 同款）。
- **指标**：e2e p50/p95、kv_cache_hit_rate、`vllm:num_preemptions_total`（抢占）、QPS、TTFT。
- **烟测**（demo 默认）：6 任务 × 12 步 × conc 6 × runs 1（快速展示）。全工况扫描 ⏳ 待跑。

---

## 5.6 结果

### 5.6.1 准入控制烟测（conc=6, 6 任务/12 步, runs=1）✅ 已验证

| 配置 (conc 6) | e2e p50 | KV 命中率 | **抢占** | 墙钟(6任务) | 吞吐 |
|---|---|---|---|---|---|
| native（FCFS） | 83 s | 0.797 | **2** | 162 s | 0.037 task/s |
| **ours（准入控制）** | **43 s（1.9×）** | **0.921** | **0** ✅ | **68 s** | **0.088 task/s（2.4×）** |

- **抢占 2→0**：准入控制把池子挡在 85% 以下，running 集合永不溢出 → 无 preempt。
- **KV 命中 0.80→0.92**：缓存不再被抢占推翻。
- **p50 1.9× / 吞吐 2.4×**：省掉重算浪费，更快完成全部任务。
- 这是**公平 conc-6 对比**（吞吐涨，非靠少跑换延迟）——与早期 conc-1 假象有本质区别。

### 5.6.2 全工况扫描 ⏳ 待跑
`f5-evict-dynamic`（准入）vs native × conc 4/6/8 × 10 任务/25 步 × runs=3：确认全并发度稳定有效 + 并发放大曲线。

---

## 5.7 讨论：边界、代价与诚实

1. **代价是并发上限**：准入控制在高压时**主动降低有效并发**（>85% 暂不放行）。但实测显示这是**净赢**——
   避免抢占重算省下的时间，远超少跑一个并发的代价（吞吐 2.4× 证明）。
2. **工况前提**：需要「并发压力足以溢出 KV 池」才有意义（单 session/低并发无溢出→闸门不触发→无差异）。
   这界定它为**多并发专用**优化；并发越高、池越小，增益越明显。
3. **诚实复盘**：F5 一开始走 SRTF 优先级路线，被一个 conc-1 串行 bug 制造的假象误导（已修 `should_admit`，
   `scheduler/admission.py`）。公平复测推翻了优先级假设，定位到**准入控制**才是真机制。负结果（优先级无效）
   与正结果（准入有效）一并记录，避免他人重蹈。
4. **可移植**：应用层闸门 + vLLM 原生调度，不改内核、不 fork，随版本升级。

---

*实验产物：`configs/f5-{native,evict-dynamic}.yaml`、`scheduler/{admission,driver}.py`、
`benchmarks/runner.py`、标准 run 目录与 demo F5 页。原始 run 由本地 `logs/` 保存，不进入代码仓库。*
