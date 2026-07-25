# 指标目录（metrics-catalog）— agent-mem-v1 统计指标全集

> 整合 F1+F2+F3+F4+F5+baseline-tot 后，项目可采集/应采集的所有指标。
> **用途**：从这里挑选最终答辩 before/after 与技术报告要呈现的指标。
> 结构：**A. 现有指标穷举表（拉平）** → **B. 每功能「必要性」补充指标（counterfactual）** → **C. 推荐核心集** → **D. 采集实现现状**。
> 图例：✅=v1 已接采集 · 🔧=代码有但需 NPU/真跑 · ❌=尚未采集（B 部分提议）

---

## A. 现有指标穷举表（按类别拉平，去重）

### A1. 显存 / 资源（赛题"显存↓"主战场）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `mem_peak_mb` | MB | metrics.py / mem_sampler (npu-smi) | 显存↓ (40分) | HBM 峰值（全 run max） | ✅ |
| `mem_avg_mb` | MB | baseline-tot sampler | 显存↓ | HBM 均值 | 🔧 |
| HBM 利用率 | % | npu-smi | 显存↓ | 瞬时 HBM 占用曲线 | 🔧 |
| KV pool 利用率 `kv_cache_usage_perc` | 0~1 | vLLM /metrics | 显存↓/生命周期 | KV cache pool 占用（触发 preempt 的直接信号，非总 HBM） | ✅ |
| KV 显存占用 (GiB) | GiB | F1 bench | 显存↓ (F1) | KV cache 绝对占用 | 🔧 |
| KV token 容量 | tokens | F1 bench | 显存↓ (F1) | 等同显存预算下可缓存的 token 数（C8 2× 的核心证据） | 🔧 |

### A2. KV cache 命中 / 复用（"长生命周期复用"叙事）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `kv_cache_hit_rate` | 0~1 | metrics.py / vLLM | 复用↑/TTFT↓ | KV cache 命中率（APC prefix 复用） | ✅ |
| `prefix_cache_hits_total` / `queries_total` | 计数 | vLLM /metrics | 复用↑ | prefix cache 命中/查询累计计数 | ✅ |
| 抢占次数 `num_preemptions_total` | 计数 | vLLM /metrics | 生命周期 (F5) | V1 recompute 抢占计数（F5 准入控制的目标：→0） | ✅ |
| `num_requests_running` | 计数 | vLLM /metrics | 生命周期 (F5) | 运行中请求数（admission 闸门信号） | ✅ |

### A3. 延迟 / 吞吐（赛题"延迟↓"）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `e2e_latency_p50_ms` | ms | metrics.py | 延迟↓ (40分) | 端到端延迟中位数 | ✅ |
| `e2e_latency_p95_ms` | ms | metrics.py | 延迟↓ | 尾延迟（尖刺敏感） | ✅ |
| `ttft_ms` | ms | metrics.py / stream | 延迟↓ | 首 token 时间（prefix cache 直接收益） | ✅ |
| `tpot_ms` | ms | baseline-tot | 延迟↓ | 每 token 生成时间（decode 速度） | 🔧 |
| `qps` | req/s | metrics.py | 延迟↓ | 吞吐 | ✅ |
| 吞吐 tok/s | tok/s | F1/F4 bench | 延迟↓ | token 级吞吐 | 🔧 |

### A4. 任务质量（赛题"成功率不掉"红线）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `task_success_rate` | 0~1 | metrics.py / τ-bench reward | 成功率 (红线) | τ-bench 任务成功率（≤2pp 红线） | ✅ |
| HotpotQA 答案正确率 | 0~1 | baseline-tot 判分 | 质量 (F7对照) | 4 级宽松匹配 exact→F1≥0.7 | 🔧 |
| Game24 solved/total | 比例 | (ToT，已删) | — | — | ❌ |
| 检索正确率 (retrieval) | 正确/总数 | F3 probe | 质量 (F3) | fetch_tool_result 取回正确片段 | 🔧 |
| 量化输出误差 (quant err) | 相对误差 | F1 verify_c8_rope | 质量 (F1) | C8 quant→dequant roundtrip | 🔧 |

### A5. Prompt / 上下文（"上下文压缩"方向）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `prompt_tokens`（变换后） | tokens | usage_log / measure_prompt_pair | 压缩↓ (F2) | 实际发给引擎的 token（middleware 后） | ✅ |
| 基线 prompt tokens（未压） | tokens | measure_prompt_pair baseline | 压缩↓ (F2) | 未压缩对照（paired 计量） | ✅ |
| `sent_tokens` | tokens/步 | F2 event log | 压缩↓ (F2) | 每步实际发送 | 🔧 |
| `compression_ratio` | 比例 | compress.py | 压缩↓ (F2) | 压缩比（保留率） | ✅ |
| 工具结果 token 节省 | tokens | F3 | 工具数据↓ (F3) | 外化后 context 减少量（−93%） | 🔧 |
| fetch 延迟 | ms | F3 | 工具数据↓ (F3) | fetch_tool_result p95 | 🔧 |
| 累计 prompt 节省 | % | F3 LongBench | 工具数据↓ (F3) | 整轮 prompt 累计降幅（−66%） | 🔧 |

### A6. F5 动态调度（"动态资源回收"方向）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| `evictions` | 计数 | metrics.py / EvictionTracker | 回收 (F5) | priority 抬升（回收）次数 | ✅ |
| `idle_hits` | 计数 | metrics.py | 回收 (F5) | 其中命中 idle session 次数 | ✅ |
| `idle_hit_rate` | 0~1 | metrics.py | 回收精准 (F5) | 回收命中 idle 比例（越高越准，真机 100%） | ✅ |

### A7. 多路径决策 / baseline（"分支推理"测量）

| 指标 | 单位 | 来源 | 赛题命中 | 说明 | 状态 |
|---|---|---|---|---|---|
| MCTS 搜索深度 | 层 | baseline-tot | 分支 (F7对照) | 平均搜索深度 | 🔧 |
| 分支数 n | 数 | baseline-tot 扫描 | 分支 (F7对照) | propose 分支因子（1/2/4/8） | 🔧 |
| `gen_tokens` | tokens | baseline-tot | — | 总生成 token | 🔧 |

---

## B. 每功能「必要性」补充指标（counterfactual — 证明功能不可或缺）

> 现有指标多在"开了功能后"的单点测。**必要性指标**回答："若没有这个功能会怎样"——把优化前后的**质变临界点**量化，让评委一眼看到"非做不可"。这些大多需补采集（❌）。

### F1 — C8 int8 KV 量化
- **必要性指标 ❌：OOM 临界并发数** —— 固定 HBM 预算下，无 C8 vs 有 C8 各能撑多少并发 session 不 OOM（预期 2×）。比"显存砍半"更直观证明"显存是瓶颈、量化解锁并发"。
- **必要性指标 ❌：等并发下的 KV 溢出率** —— 无 C8 时 KV pool 溢出/preempt 频次 vs C8。
- 现有可代用：`KV token 容量` 775K→1.55M（已是 2× 证据，但偏静态）。

### F2 — Prompt 压缩
- **必要性指标 ❌：context 溢出轮次** —— 无压缩时第几轮 context 超 `max_model_len`（任务失败）；有压缩延后/避免。证明"长 session 必爆、压缩是解药"。
- **必要性指标 ❌：prefill 计算量节省** —— 压缩前后 prefill FLOPs/token 差（TTFT 收益的根因，比 TTFT 本身更"机制")。
- 现有可代用：`prompt_tokens` −19%、TTFT（已有）。

### F3 — 工具数据 lazy-load
- **必要性指标 ❌：单次工具调用 KV 增量** —— 长工具结果直接进 KV vs 引用模式的每步 KV 增量（预期差 ~90%）。证明"工具数据是 KV 膨胀主因、外化是根治"。
- **必要性指标 ❌：N 步后 context 增长曲线** —— raw vs lazy 两条曲线对比（随工具调用发散）。
- 现有可代用：工具结果 −93%、检索正确率（已有）。

### F4 — LMCache 分层
- **必要性指标 ❌：可承载 KV 上限** —— 无分层时 KV 超 HBM 即 OOM 的阈值 vs 有分层（KV 可溢出到 CPU/Disk）。证明"分层让 KV 容量突破物理 HBM"。
- **必要性指标 ❌：offload/rewarm 时延占比** —— 冷 KV 搬运往返占延迟比（证明分层代价可控）。
- 现有可代用：p50 −21%、QPS +32%（已有，但未显式对照"无分层 OOM"）。

### F5 — 动态资源回收
- **必要性指标 ❌：抢占风暴指标** —— 无准入控制时单位时间 preempt 次数 + 活跃 session 被误伤次数（vs 准入控制 →0）。这是 F5 最强叙事（KV 命中 0.46→0.93 已部分体现）。
- **必要性指标 ❌：尾延迟尖刺消除率** —— 无策略时 p95/p50 比（抖动）vs 有策略（平稳）。证明"回收策略消除延迟尖刺"。
- **必要性指标 ❌：progress 优先级 SRTF 增益** —— 完成近完成 session 的速度差（p50 3.5× 的机制证据）。
- 现有可代用：`num_preemptions_total`→0、`kv_cache_hit_rate` 0.46→0.93、`idle_hit_rate` 100%（已有，强）。

### baseline-tot（F7 对照组）
- **必要性指标 ❌：分支数 vs KV 命中率斜率** —— n=1/2/4/8 时 KV 命中率提升曲线，量化"分支推理的 KV 复用潜力"（= F7 若做能收割的上限）。⚠️ 现有数据有**累计假象**（引擎不重启，计数器跨组累加），需改成**每组独立重启引擎**重测才可信。

---

## C. 推荐核心集（答辩 + before/after 必采，最小集）

> 用最少指标覆盖赛题 4 个评分维度的每条主张。**标 ⭐ = 不可让步**。

| 维度 | 主张 | 必采指标 | 来源功能 |
|---|---|---|---|
| 应用效果·显存↓ | 显存有效降低 | ⭐ `mem_peak_mb` + KV token 容量(2×) | F1 |
| 应用效果·延迟↓ | 延迟优化 | ⭐ `e2e_latency_p50_ms` + `ttft_ms` + `qps` | 全局 |
| 应用效果·成功率 | 成功率基本不降 | ⭐ `task_success_rate`（≤2pp 红线） | 全局 |
| 长生命周期 | 动态资源回收 | ⭐ `num_preemptions_total`(→0) + `kv_cache_hit_rate`(↑) + `idle_hit_rate` | F5 |
| Prompt 压缩 | 上下文压缩 | `prompt_tokens`(↓) + `compression_ratio` | F2 |
| 工具数据 | 按需加载 | 工具结果 token 节省(↓) + 检索正确率(持平) | F3 |
| 分层内存 | 冷热分离 | `mem_peak_mb` + 可承载 KV 上限 | F4 |
| 通用性 | 多模型/任务 | 同指标 × {Qwen2.5-7B, MiniCPM3} × {retail, airline} | 全局 |

---

## D. 采集实现现状（v1 代码层）

| 采集点 | 实现 | 状态 |
|---|---|---|
| `metrics.json`（6+3 指标） | `agent_mem/metrics.py` RunMetrics + bench runner | ✅ 已接 |
| vLLM Prometheus 抓取 | `bench/vllm_metrics.py`（prefix cache / preempt / kv_usage） | ✅ 已接 |
| HBM 采样 | `bench/mem_sampler`（npu-smi / fake） | ✅ 已接（NPU 关时 fake） |
| Prompt paired 计量 | `agent/usage_log.py` measure_prompt_pair | ✅ 已接 |
| F5 driver snapshot | `scheduler/` EvictionTracker → run summary sidecar | ✅ 已接 |
| F1 C8 bench | `scripts/bench_c8.py` + `kv/c8.py` | 🔧 需 NPU |
| F3 检索质量 probe | `docs/F3-*` 脚本（结果已删，脚本在 baseline/f3 分支） | 🔧 需重跑 |
| baseline-tot 指标 | `scripts/run_baseline.py` MemSampler + vllm_metrics | 🔧 需 NPU |
| **B 部分必要性指标** | 多数未采集 | ❌ 需补采集脚本 |

> **诚实提醒**：baseline-tot 的 KV 命中率为跨组累计值（不可直接引用）；F5 成功率因本地 user-sim 噪声 ≈0（需外部 mimo user-sim 补最终数）。补这两项是出可信 before/after 的前提。
