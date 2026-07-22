# F5 — 并发场景动态资源回收与重分配（技术报告）

> 分支：`feat/f5-priority-evict` | 日期：2026-07-22

## 1. 问题

并发跑 τ-bench 任务时，多个 agent session 共用 vllm 引擎，各自产生 KV cache 争抢
HBM。vllm 原生处理方式：pool 满时随机踢掉某个 request（scheduler preemption），
被踢的 request 全部 KV 清零、从头重算。没有 session 级别的资源管理。

## 2. 目标

在应用层实现并发控制 + idle 驱逐，不修改 vllm 源码：

- 多个 session 并发跑 τ-bench 任务
- HBM 低于阈值 → 放行更多 session，提高并发度
- HBM 超过阈值 → 驱逐 idle 最久的 session，释放 KV
- 被驱逐的 session 回到任务队列，等 HBM 降下来后重跑（任务不丢失）

## 3. 架构

```
┌─ 应用层（我们写）─────────────────────┐
│                                      │
│  AdmissionController                 │
│  ├─ 读 HBM（npu-smi）               │
│  ├─ HBM < 70% → 放行新 session      │
│  ├─ HBM > 85% → 驱逐 idle 最久      │
│  ├─ 驱逐后任务回队列                  │
│  └─ 不碰 vllm                       │
│                                      │
│  QwenAgentRunner（内部并发）          │
│  ├─ max_concurrency=N → N task 并发  │
│  └─ ThreadPoolExecutor              │
│                                      │
├─ vllm 引擎（不动）───────────────────┤
│  FCFS scheduler + APC + block_pool   │
└──────────────────────────────────────┘
```

## 4. 实现

### 4.1 AdmissionController（`scheduler/admission.py`，新增 ~160 行）

```python
ctrl = AdmissionController(
    target_lo=70,    # HBM < 70% → 放行
    target_hi=85,    # HBM > 85% → 驱逐
    min_workers=1, max_workers=6,
    idle_timeout_s=30, interval=3,
)

# 主循环
while pending:
    if ctrl.should_admit():           # HBM < 70% 且有 slot
        t = pending.pop(0)
        executor.submit(t)            # 放行
        ctrl.admit(f"task-{t.id}")

    sid = ctrl.evict_idle()           # idle > 30s → 驱逐
    if sid: pending.insert(0, sid)    # 回队首

    time.sleep(3)
```

### 4.2 QwenAgentRunner 内部并发（`bench/runners/qwen_agent.py`）

```python
class QwenAgentRunner:
    def __init__(self, max_concurrency=1, ...):
        self.max_concurrency = max_concurrency

    def run_all(self, cfg):
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            futures = {ex.submit(run_task, t): t for t in tasks}
            for f in as_completed(futures):
                results.append(f.result())
        return results
```

### 4.3 CLI 参数（`benchmarks/runner.py`）

```bash
--max-concurrency N    # 并发 task 数（默认 1 = 串行）
```

## 5. 实验设计（待 NPU 恢复后执行）

两组对照，同一引擎、同一 workload（10 τ-bench retail tasks × 25 max_steps）：

| 组 | 策略 | 并发度 | 驱逐 |
|---|---|---|---|
| A | vllm 原生 FCFS | 固定 5 | vllm 随机踢 |
| B | AdmissionController | 动态 1-6 | 按 idle 时间驱逐 |

对比指标：mem_peak、总完成时间、P50/P95 延迟、成功率、驱逐是否命中 idle。

## 6. 当前状态

- 代码实现：✅ 完成（admission.py + qwen_agent.py 并发 + CLI）
- 测试：191 passed（未引入新失败）
- 实验：⏳ 待 NPU 恢复
