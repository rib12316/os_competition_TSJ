# F7 — 分支 KV 共享（测量版）

> 主责：P1 | 缝：F | 难度：高 | 方向：测量 vllm APC 隐式 CoW，不造新机制

## 1. 赛题锚点

命中赛题「分支推理内存共享」方向。vllm 没有公开 fork API（tree-attention #3960），
所以 F7 不做新机制——**测量 vllm Automatic Prefix Caching (APC) 在分支 workload 下
已有的 block 级 CoW 共享效果**，作为「分支共享」的技术叙事。

## 2. vllm APC 机制简介

vllm V1 对每个 KV cache block 计算 hash（SHA256/xxhash），hash 相同 → 同一 block。
新请求的 prompt 前缀与已缓存 block hash 匹配 → 直接复用（零拷贝）。
多请求共享同一 prefix block，写入时分配新 block —— **隐式 CoW**。

```
请求 A: [prompt: "Solve: 2x+1=5\n"]      → hashes: [h1, h2, h3]
请求 B: [prompt: "Solve: 2x+1=5\nStep1:"] → hashes: [h1, h2, h3, h4]
                                                    ↑ 前 3 个 hash 相同 → CoW 复用
```

### 关键代码路径

| 组件 | 文件 | 作用 |
|---|---|---|
| `PrefixCacheStats` | `vllm/v1/metrics/stats.py` | 记录 hit/queries 统计 |
| `find_longest_cache_hit` | `vllm/v1/core/kv_cache_manager.py` | 查找最长前缀命中 |
| `get_num_common_prefix_blocks` | 同上 | 计算多请求共享的 block 数 |
| `vllm:prefix_cache_hits_total` | `/metrics` 端点 | Prometheus 指标（我们已采集） |

## 3. 实现方案

### 3.1 核心思路

构造分支 workload：N 个请求共享一个长 prompt 前缀，各自在不同分支继续。
对比两组：

- **对照 A（独立）**：N 个独立请求，前缀各不相同 → KV 命中率 ≈ 0%
- **对照 B（分支）**：N 个请求共享同一前缀 → KV 命中率 ≈ (N-1)/N

取 `/metrics` 两次算 `kv_cache_hit_rate` 差值，证明 APC 的 CoW 共享。

### 3.2 Workload 设计

```
对照 A（独立，无共享）：
  请求 1: [prefix_A 4000 tokens] + [task prompt]
  请求 2: [prefix_B 4000 tokens] + [task prompt]
  请求 3: [prefix_C 4000 tokens] + [task prompt]
  → 三个前缀不同，APC 无命中

对照 B（分支，共享前缀）：
  请求 1: [SHARED 4000 tokens] + [branch_A prompt]
  请求 2: [SHARED 4000 tokens] + [branch_B prompt]
  请求 3: [SHARED 4000 tokens] + [branch_C prompt]
  → 请求 2,3 的前缀命中请求 1 的缓存
```

### 3.3 为什么不造真 ToT / self-consistency

- 真 ToT 需要 agent 多轮交互 + 评估器，验证链路长、不稳定
- 真 self-consistency 需要 reward 模型
- **构造的 prompt 序列**效果等价、可控、可复现，不依赖 agent 成功率

### 3.4 Runner 实现

```python
class BranchingKVShareRunner(Runner):
    """发送分支 prompt 到引擎，测量 APC CoW 共享"""
    
    def __init__(self, engine_url, model, prefix_tokens=4000, n_branches=4):
        ...
    
    def run_all(self, cfg) -> list[TaskRunResult]:
        # 1. 构造共享前缀（用模型 tokenizer 生成）
        # 2. 对照 A：逐个发 n 个独立请求，抓 /metrics
        # 3. 对照 B：逐个发 n 个分支请求（共享前缀），抓 /metrics
        # 4. 返回 [独立_metrics, 分支_metrics] → 由 compare 算 delta
```

## 4. 任务拆解

| # | 步骤 | 依赖 | 工作量 |
|---|---|---|---|
| 1 | 调研文档 + 设计（本文档） | 无 | ✅ 当前 |
| 2 | 实现 `BranchingKVShareRunner` | 步骤 1 | 2-3h |
| 3 | 写 `configs/f7-branch-share.yaml` | 步骤 1 | 10min |
| 4 | 写单测（prompt 构造 + result 结构） | 步骤 2 | 1h |
| 5 | NPU 烟测（起引擎，验证 prefix match） | NPU 开 | 1h |
| 6 | 跑对照 A vs B benchmark | 步骤 5 | 30min |
| 7 | 整理对照报告 + 文档 | 步骤 6 | 30min |

## 5. 实现注意

- **不动的**：`bench/runner.py`、`config.py`、`vllm_server.py`
- **新增的**：`bench/workloads/branching_runner.py`、`configs/f7-branch-share.yaml`、测试
- **需要 tokenizer**：构造 prompt 必须走模型 tokenizer 算 token 数，字符串长度 ≠ token 数
- **需要 NPU**：APC 只在真引擎上工作，stub/dry-run 无法验证
- **公平对照**：A 和 B 的总 token 数必须一致，prefix + task 设计要对称
- **prefix 长度**：4000 tokens 足够撑满一个 block（block_size=128，约 32 个 block）

## 6. 参考

- vllm Prefix Caching 设计：https://docs.vllm.ai/en/stable/design/prefix_caching/
- `vllm/v1/core/kv_cache_manager.py`：`find_longest_cache_hit`, `get_num_common_prefix_blocks`
- `vllm/v1/metrics/stats.py`：`PrefixCacheStats`
- `vllm/v1/metrics/loggers.py`：`vllm:prefix_cache_hits_total` Prometheus 指标
