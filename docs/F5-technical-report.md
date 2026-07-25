# F5 — 并发场景动态资源回收与重分配（技术报告）

> 分支：`feat/f5-priority-evict` ｜ 赛题方向 1（KV Cache 生命周期管理）原话"动态资源回收与重分配"
> 状态：**Phase 0 + Phase 1（真机 A/B/C × runs=3）均完成**：KV-pool 准入控制消除抢占、KV 命中率 0.93 vs 0.46–0.48（+47pp）、p50 −21%、eviction-hit-idle 100%（3/3 run 全命中 idle）。

## 摘要

多个 agent session 并发跑在一个 vLLM-Ascend 引擎上，共享 HBM。每个 session 因多轮对话
持续累积 KV、寿命长；HBM 吃紧时 vLLM 原生调度**机械地**随机/LIFO 抢占某个请求（V1 默认
recompute = 被抢的 KV 全部清零、从头重算），不区分"忙 session"与"闲 session"，导致活跃
agent 延迟飙升甚至任务失败。F5 给这套回收装上**会话感知的策略层**：按 session 活跃度动态
分配调度优先级，让 vLLM **先回收闲的、保护忙的**，并在应用层用 HBM 驱动的准入控制把引擎
维持在安全带宽内。

关键结论（读 vllm-ascend / vLLM 源码得出）：**vLLM 没有给应用层"按 session 无损
offload/restore KV"的现成 API**——KV connector 全是调度器内部按内存压力自动触发的。因此
Phase 1 的务实机制是 **vLLM 原生 priority 抢占**（`--scheduling-policy priority`，HBM 满先
踢低优请求），由应用层把"idle session 的优先级抬高"喂给它；真正的无损 on-demand offload
需自定义 V1 KV connector，列为 Phase 2（stretch）。

## 1. 现象

并发跑 τ-bench retail（N 个 agent session × 多轮）时观察到的内存问题：

- 多 session 共用引擎 KV 池，**KV 随轮次持续累积**，单 session 寿命远超 single-shot。
- HBM 接近上限时 vLLM 触发 **preemption**：被踢请求的 KV 清零、后续重算（V1 默认
  `recompute`，`swap` 模式 V1 不支持）。被踢对象是**随机/LIFO**，与 session 价值无关。
- 表现：活跃 session 被误伤 → 延迟尖刺 / 超时 / 成功率下降；而真正闲置的 session 的 KV
  却还占着显存。

## 2. 根因

agent 推理的内存特征 = **长生命周期 + 并发 + 突发**：

1. **长生命周期**：单 session 跨几十轮，KV 是沉没成本，被抢占重算代价高。
2. **并发突发**：多 session 同时活跃时瞬时 KV 需求超 HBM，必须有人让步。
3. **价值异构**：同一时刻有的 session 在响应用户（高价值）、有的在等工具/用户（低价值）。

vLLM 的回收是**机制正确但策略盲**的：block 级回收/重算机制现成（PagedAttention + preempt），
但"踢谁"只看机械顺序，**不感知 session 活跃度/价值**。这正是赛题方向 1 要补的"动态资源
回收与重分配"。

## 3. 可行性深挖（决定机制选择）

> 读 `third_party/vllm` + `third_party/vllm-ascend` 源码的硬结论。

| 想法 | 现状 | 结论 |
|---|---|---|
| 应用层"按 session 无损 offload/restore KV" | `SimpleCPUOffloadConnector` / vllm-ascend `simple_kv_offload` 全是**调度器内部**按内存压力自动触发，无 `offload_request(id)` 等外部钩子 | ❌ 无现成 API，要做得自定义 V1 connector（Phase 2） |
| `--scheduling-policy priority` 抢占 | V1 调度器支持（`SchedulingPolicy.PRIORITY` + `PriorityRequestQueue`，用于 preempt 决策）；vllm-ascend 用上游 V1 调度器 | ✅ **可用**，HBM 满先踢低优请求（数值越大越先被踢） |
| `SimpleCPUOffloadConnector` lazy offload | `lazy_offload=True` 时自动把近淘汰块留 CPU 副本，cache hit 时便宜 reload | ✅ 降本（让被回收的 KV 恢复更便宜），需 `enable_prefix_caching` |
| APC 前缀复用 | V1 默认开，共享 system prompt KV 免重算 | ✅ 降本（重算只跑 agent 专属段） |

**选型**：Phase 1 走 **priority 抢占 + lazy offload + APC + 应用层准入控制** 的分层组合
（全部原生/应用层，不改 vLLM 源码）；无损 on-demand offload 留给 Phase 2 自定义 connector。

## 4. 机制（分层策略）

```
┌─ 应用层（agent_mem，我们写）──────────────────────────────────────┐
│  ConcurrentSessionDriver                                          │
│  ├─ AdmissionController   HBM 驱动动态并发（>hi 不放行，<lo 放）  │
│  ├─ SessionManager        追踪每 session 活跃时间（线程安全）     │
│  ├─ PriorityEvictionStrategy  后台 sweep：idle→抬 priority=100    │
│  │                           active→回落 priority=0（mark_active）│
│  ├─ EvictionTracker       记 eviction 次数 + 命中 idle 比例        │
│  └─ 回调注入 agent        priority_fn（每轮读当前 priority）      │
│                            on_turn_start（每轮标记活跃）          │
├─ 引擎层（vllm-ascend，不改源码）──────────────────────────────────┤
│  --scheduling-policy priority   HBM 满先抢 priority 大的=idle 的  │ ← 真·回收
│  SimpleCPUOffloadConnector (可选 lazy)  被抢 KV 留 CPU 副本便宜恢复│ ← 降本
│  APC 前缀复用                    共享 system prompt KV 免重算     │ ← 降本
└──────────────────────────────────────────────────────────────────┘
```

- **回收** = 优先级驱动抢占：idle session 的 priority 被抬到 100，vLLM 压力下先抢它。
- **重分配** = 腾出的 block 给高优（活跃/突发）session——它们 priority=0，受保护、优先入队。
- **降本** = lazy offload + APC 让被抢 KV 的恢复成本远低于全量重算。

## 5. 实现（Phase 0，纯应用层，无需 NPU）

> 全部接线进真实运行路径、脱离 NPU 可单测（fake runner + fake HBM + 注入时钟）。

| 组件 | 文件 | 说明 |
|---|---|---|
| `SessionManager` | `scheduler/session.py` | session 一等实体 + 状态机（ACTIVE/OFFLOADED/CHECKPOINTED/EVICTED）+ idle 追踪；**加 RLock** 支持 N 并发 session 线程 touch |
| `PriorityEvictionStrategy` | `scheduler/strategies.py` | idle 超阈值→抬 `metadata["priority"]`、上报 tracker；`mark_active` 回落。不改 KV（区别于无损 `IdleEvictionStrategy`，后者留 Phase 2 接 connector） |
| `EvictionTracker` | `scheduler/eviction.py` | 线程安全计数：evictions / idle_hits / idle_hit_rate |
| `AdmissionController` | `scheduler/admission.py` | HBM 准入闸门（`should_admit`）+ idle 驱逐；**HBM 读取可注入 `MemBackend`**（复用 `bench/mem_sampler` 的 TorchNpu/NpuSmi/Fake，不再硬编码 npu-smi）；加锁 + tracker |
| `ConcurrentSessionDriver` | `scheduler/driver.py` | 串起上述四者：准入闸门 + 后台 sweep 线程 + 把 `priority_fn`/`on_turn_start` 注入每个 agent；与引擎/τ-bench 解耦（注入 `task_runner`，故可单测） |
| `TauBenchAgent` | `agent/tau_bench_agent.py` | `priority_fn`（每轮读动态 priority 入 `extra_body`）+ `on_turn_start`（每轮标记活跃）；None 时退回静态 priority（后向兼容） |
| 引擎 flag 渲染 | `server/vllm_server.py` | 按 config 渲染 `--scheduling-policy priority`（缝A）+ lazy offload connector（缝C，复用 `kv/connector.render_kv_connector_args`） |
| 配置 | `config.py` + `configs/f5-*.yaml` | `engine.priority_scheduling` / `engine.kv_offload` / 白名单加 `priority-evict`；A/B/C 三组 yaml |

**测试**：新增 10 个 F5 单测（SessionManager 线程安全、PriorityEvictionStrategy 状态迁移、
AdmissionController 准入/驱逐 + `FakeBackend`、driver 端到端 fake runner、eviction 指标入
run summary），**全量 203 tests 绿**。

## 6. 验证

### 6.1 Phase 0（已完成，无 NPU）
单元测试覆盖策略/准入/driver/指标全链路（fake 回调 + 注入时钟 + fake HBM），证明接线正确、
线程安全、eviction 计数与 metrics 落盘正常。

### 6.2 Phase 1 真机结果（NPU 910B2C，已跑通）

**de-risk**（照 F1 int8 flag 假生效教训）：`--scheduling-policy priority` 在 vllm-ascend
0.22.1rc1 **被正常接受**——探针（`scripts/f5_priority_probe.py`）起引擎不报错、priority 经
`extra_body` 透传成功、抢占计数器 `vllm:num_preemptions_total` 存在。非 no-op。

**A/B/C 三组对照**（Qwen2.5-7B，KV pool **1.27 GiB** 刻意制压 = util 0.27 / max_model_len 16384，
6 并发 τ-bench retail × 25 steps，本地 user-sim，**每组 runs=3 取中位数**；对照报告
`logs/_summaries/20260725_f5-dynamic-reclaim_comparison.md`）：

| 指标 | A FCFS | B priority-static | C 动态回收（priority + KV-pool 准入） | 效果 |
|---|---|---|---|---|
| vLLM preemption（实测） | 有 | 有 | **0** | 准入控制**消除抢占** |
| KV 命中率 | 0.48 | 0.46 | **0.93** | **+45–47pp**（抢占重算不吃缓存） |
| e2e p50 | 114 s | 109 s | **90 s** | C −21% vs A |
| e2e p95 | 165 s | 158 s | 155 s | 略优 |
| TTFT | 152 ms | 132 ms | **111 ms** | C −27% vs A |
| mem_peak_mb | 17399 | 17401 | 17401 | 同（pool 固定） |
| eviction-hit-idle | — | — | **100%（3 次 run 均 100%；3/7/4 次全命中 idle）** | 回收全准 |

**机制实证**：实时抓 `/metrics` 见——C 组 KV 到 ~72% 时 `num_requests_running` 从 6 降到 3
（**KV-pool 准入闸门触发**，KV 不再涨 → 不溢出 → 0 抢占）；A/B 组无闸门，KV 涨到 ~86–89% 溢出 →
preempt → 重算 → KV 命中率掉到 ~0.46。

**关键判读（诚实）**：**A ≈ B**——priority-static 给所有 session 同优先级时 ≈ FCFS，priority
**单独**几乎不增益；**真正的增益来自准入控制（C）**。单 run 里 B 出现过 270 s 的方差离群点，runs=3
中位数（B 109 s）更稳；而 **KV 命中率 0.46 → 0.93 是稳健的大信号**——准入控制让 KV 不溢出、保住
APC 缓存，A/B 的抢占重算把命中率打掉一半。

**诚实说明**：成功率 A/C ≈ 0、B=0.17（本地 7B user-sim 噪声，非 F5 问题；"成功率不掉"需外部
user-sim 如 mimo 补最终数）。采集写 `metrics.json` + `f5_driver_snapshot.json` sidecar。


## 7. 权衡

- **有损 vs 无损**：Phase 1 的 priority 抢占是**有损**回收（被抢 KV 重算，由 APC + lazy
  offload 降本）；真正无损 on-demand offload 需自定义 V1 connector（Phase 2，研究级、Ascend
  风险高）。务实路线先用有损拿到可交付数字，再视时间上无损。
- **准入控制代价**：HBM 高时降并发 → 牺牲瞬时吞吐换稳定（少 preempt 抖动）。阈值
  `target_lo/hi`（默认 70/85%）在真机调。
- **priority 粒度**：vLLM priority 是 per-request；session 活跃度经 `priority_fn` 每轮映射
  到当前请求。idle 但已 cached 的 KV 另由 APC 的 LRU 天然回收——F5 的增量是**保护活跃
  running session + 主动协调**，而非取代 APC。

## 8. 状态与路线

- ✅ **Phase 0**：policy 全接线 + 单测绿（`feat/f5-priority-evict`）。
- ✅ **Phase 1（真机跑通，A/B/C × runs=3）**：priority 探针确认 flag 非 no-op；**三组中位数**——
  KV-pool 准入控制消除抢占（C:0 vs A/B:有）、KV 命中率 **0.93 vs 0.46–0.48**、p50 −21%、TTFT −27%、
  eviction-hit-idle 100%（3/3 run）。诚实：A≈B（priority 单独不增益），增益来自准入控制。
- 🔭 **收尾（可选）**：runs=3 取中位数、外部 user-sim 补成功率、A 组（FCFS）对照、lazy offload
  connector 名；Phase 2 自定义 V1 connector 做真·无损 on-demand offload/restore。

> 配套：实验设计 `docs/F5-experiment-design.md`、计划 `plans/nested-prancing-metcalfe.md`。
