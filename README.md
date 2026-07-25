# LATS MCTS × HotpotQA 多路径决策实验

赛题 14：面向智能体的内存管理系统 — Baseline 实验

## 环境

| 项目 | 值 |
|------|-----|
| 硬件 | Ascend 910B2 (HBM 65536 MB) |
| 模型 | Qwen2.5-7B-Instruct (vLLM, port 8000) |
| Python | 3.11 (.venv-lats, openai==0.27.7) |
| 数据集 | HotpotQA distractor (90447 train / 7405 validation) |
| 主仓库 | /data/os_competition_TSJ |

## 文件结构

```
/tmp/multi-path/
├── README.md                          # 本文件
├── scripts/
│   ├── offline_hotpot_env.py          # 离线 HotpotQA 环境 (TF-IDF 检索)
│   ├── lats_hotpot_offline.py         # MCTS 核心 + 批量实验入口
│   ├── run_baseline.py                # Baseline 全量实验 (断点续跑)
│   └── debug_single_case.py           # 单例调试 (详细树过程)
└── logs-lats/                         # 实验结果输出
    ├── baseline_checkpoint.json       # 断点文件 (可续跑)
    ├── baseline_full.json             # 完整结果 (per-task + summary)
    └── baseline_summary.json          # 汇总 (快速查看)
```

## 快速开始

### 1. 单例调试 (看 MCTS 树过程)

```bash
cd /tmp/multi-path
source /data/os_competition_TSJ/.venv-lats/bin/activate

# 跑第 0 题，2 分支，8 轮迭代
python scripts/debug_single_case.py 0 2 8

# 或使用主脚本的 debug 模式
python scripts/lats_hotpot_offline.py --debug 5
```

输出包括：每轮的 selection path → LLM action 提议 → rollout 值 → 树结构 → 所有 Finish 尝试 → 详细树节点。

### 2. Baseline 实验

```bash
# 跑 100 条 (快速验证)
python scripts/run_baseline.py --num-tasks 100 --fresh

# 跑 validation 全集 (7405 条 ~20h，建议后台跑)
nohup python scripts/run_baseline.py --fresh > logs-lats/baseline_run.log 2>&1 &

# 查看进度
tail -f logs-lats/baseline_run.log

# 中途断了可以续跑
python scripts/run_baseline.py --resume
```

### 3. 切换实验组

```bash
# 低分支 (n=2)
python scripts/run_baseline.py --group C1 --num-tasks 100 --fresh

# 高分支 (n=8)
python scripts/run_baseline.py --group C3 --num-tasks 100 --fresh

# 无共享基线
python scripts/run_baseline.py --group A --num-tasks 100 --fresh
```

## 实验组配置

| Group | 分支数 | prefix_cache | 说明 |
|-------|--------|-------------|------|
| A | 2 | False | 无共享基线 |
| B | 1 | True | 单分支基准 |
| C1 | 2 | True | 低分支 |
| C2 | 4 | True | **中分支 (baseline)** |
| C3 | 8 | True | 高分支 |

## 采集指标

### 系统指标
| 指标 | 说明 | 来源 |
|------|------|------|
| HBM peak/avg/min | 显存峰值/均值/谷值 (MB) | npu-smi, 每 2s 采样 |
| KV cache avg/peak | KV 缓存使用率 (%) | vLLM /metrics |
| KV cache hit rate | 前缀缓存命中率 | vLLM prefix_cache_hits/queries |
| TTFT | 平均首 token 延迟 (ms) | vLLM time_to_first_token |
| TPOT | 平均每 token 生成时间 (ms) | vLLM request_time_per_output_token |
| E2E latency | 平均端到端延迟 (ms) | vLLM e2e_request_latency |
| Throughput | 任务吞吐 (tasks/min) | 总 wall time / 任务数 |
| Tokens/sec | token 生成速度 | 总生成 token / 总时间 |

### MCTS 树指标
| 指标 | 说明 |
|------|------|
| avg_nodes_per_task | 平均每任务树节点数 |
| avg_branches_per_expansion | 平均每扩展节点的子节点数 |
| avg_max_depth | 平均树最大深度 |
| avg_best_depth | 平均成功解答深度 |
| avg_expansions_per_task | 平均每任务扩展次数 |

### 任务指标
| 指标 | 说明 |
|------|------|
| success_rate | 任务成功率 (lenient match) |
| wall_time | 每任务耗时 (avg/median/min/max) |
| failure_analysis | 失败分类 (无 Finish / Finish 错误) |

## MCTS 算法 (v3)

```
while iter < max_iters:
    Selection:  从 root 沿 UCT 选到叶子 (跳过 is_terminal 节点)
    Expansion:  LLM 提议 n_branches 个 action (Finish 永不被截断)
    Rollout:    每个 action 执行确定性 rollout (temp=0, max 2 步)
    Backprop:   向上传播 rollout value
```

关键改进 (相对 v1)：
- **Finish 不截断**: `_parse_actions` 保证 Finish action 始终保留
- **确定性 rollout**: temperature=0，选第一个 action，不依赖随机数
- **TF-IDF 检索**: 纯 Python 实现，替代简单词匹配
- **多级答案匹配**: 精确 → 子串 → token-F1 → 归一化
- **全树答案提取**: 扫描所有 Finish 节点，按 reward 排序取最佳

## 结果格式

```json
{
  "config": {"group": "C2", "n_branches": 4, "max_iters": 8},
  "summary": {
    "success_rate": 0.75,
    "hbm_peak_mb": 60293,
    "avg_ttft_ms": 32.0,
    "avg_tpot_ms": 14.2,
    "kv_cache_hit_rate": 0.7796
  },
  "per_task": [
    {
      "task_id": 0,
      "question": "Which magazine was started first...",
      "ground_truth": "Arthur's Magazine",
      "final_answer": "Arthur's Magazine was started first...",
      "success": true,
      "best_depth": 1,
      "wall_time_s": 2.0,
      "tree": {"total_nodes": 3, "max_depth": 1}
    }
  ]
}
```
