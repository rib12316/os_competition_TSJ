# F4 — LMCache Ascend 集成指南

> **状态**：✅ 代码集成完成 + NPU 验证通过 + benchmark 有数据。
> 结论：LMCache 在当前 workload 下的收益在**延迟和 QPS**（-21%/-32%），内存峰持平（KV 不超 pool，offload 不触发）。

## Benchmark 结果（2026-07-20）

| 场景 | 指标 | baseline | f4-lmcache | delta |
|---|---|---|---|---|
| 单 agent (10 tasks) | P50 延迟 | 152.8s | 120.8s | **-21%** ✅ |
| 单 agent (10 tasks) | 成功率 | 0.20 | 0.40 | **+20pp** |
| 单 agent (10 tasks) | mem_peak | 57911 MB | 58050 MB | +0.2% |
| 并发 4 agent | qps | 0.028 | 0.037 | **+32%** |
| 并发 4 agent | mem_peak | 31755 MB | 31911 MB | +0.5% |

> 内存峰不降的原因：τ-bench retail 任务 prompt ~7K tokens，并发 ≤4 session，
> KV cache pool 够用，LMCache offload 不会触发。需要更长序列或多并发才能体现内存收益。

## 验证摘要（2026-07-20 NPU 真机）

| 步骤 | 结果 | 关键操作 |
|---|---|---|
| 环境确认 | ✅ | 910B2C, CANN 9.0.0, torch_npu OK |
| 安装 lmcache_ascend | ✅ | 需 `SOC_VERSION=Ascend910B2C`；用 venv pip 装 |
| lmcache 版本 | ⚠️ | 0.5.1 CUDA 版需降级为 0.4.4（`NO_CUDA_EXT=1`） |
| import 验证 | ✅ | `check_lmcache_ascend()` 通过 |
| connector 注册 | ✅ | 起引擎时 vllm-ascend 插件加载 LMCacheAscendConnector |
| 引擎烟测 | ✅ | 正常启动，LMCache 用 `non_cuda_equivalents` 后端，API 正常 |
| F4 对照 benchmark | ❌ | NPU HBM 92% 被前序引擎占用，容器无法 reset |

> **已知约束**：容器内无法 reset NPU（`npu-smi set -t reset` 不可用）。
> 多档 benchmark 需在每档之间留足 HBM 释放时间（>5min），或单次只跑一档并重启容器。

## 前提

- NPU 已启动（`npu-smi info` 正常）
- 仓库根目录 `/data/os_competition_TSJ`
- 虚拟环境 `.venv` 已激活

## 原理（一句话）

vllm-ascend 0.22.1rc1 的 `KVConnectorFactory` 已注册 `LMCacheAscendConnector`。
本项目只需在 yaml 里指定 `connector: LMCacheAscendConnector`，`vllm_server.py`
自动翻译成 `--kv-transfer-config` JSON。剩下的就是安装 `lmcache_ascend` 包。

---

## Step 1：确认环境

```bash
# 应在仓库根目录
pwd                    # → /data/os_competition_TSJ
npu-smi info           # → 应显示 NPU 信息，不报错
echo $ASCEND_HOME_PATH # → /usr/local/Ascend/cann-9.0.0
.venv/bin/python -c "import torch_npu; print('torch_npu OK')"
```

**预期**：NPU 正常，`torch_npu` 可 import。

---

## Step 2：试直接装 lmcache_ascend（先不动现有 lmcache）

当前环境有 `lmcache 0.5.1`（pip 装的）。先不卸载它，直接试装 `lmcache_ascend`，看 pip 解依赖时是否报冲突。

```bash
cd /tmp
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend
.venv/bin/python -m pip install --no-build-isolation -e .
```

**预期分支**：
- ✅ 安装成功 → **跳到 Step 4**
- ❌ 报 lmcache 版本冲突（如 `lmcache>=0.4.3,<0.5.0` vs 已装 0.5.1）→ **走 Step 3**
- ❌ 编译报错（CMake / CANN 头文件 / SoC 检测失败）→ **看报错信息，评论区贴出来**

---

## Step 3（仅 Step 2 版本冲突时）：降级 lmcache

```bash
pip uninstall lmcache -y
NO_CUDA_EXT=1 pip install lmcache==0.4.4
```

然后回到 Step 2 重装 `lmcache_ascend`。

---

## Step 4：验证 lmcache_ascend 可用

```bash
.venv/bin/python -c "from agent_mem.kv.lmcache_check import check_lmcache_ascend; check_lmcache_ascend()"
```

**预期**：无输出 = 成功。报 `RuntimeError` → lmcache_ascend 没装上，回到 Step 2 看报错。

再验证 vllm 的 factory 能找到它：

```bash
.venv/bin/python -c "
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
conn = KVConnectorFactory.resolve_connector('LMCacheAscendConnector')
print(type(conn).__name__)
"
```

**预期**：打印 `LMCacheConnectorV1`。

---

## Step 5：真机烟测（起引擎 + 单次推理）

**终端 A** — 起引擎：

```bash
cd /data/os_competition_TSJ
PYTHONPATH="agent-mem/src:$PYTHONPATH" .venv/bin/python -m agent_mem.server.vllm_server \
    --config agent-mem/configs/f4-lmcache.yaml \
    --model-path models/Qwen2.5-7B-Instruct \
    --port 8000
```

看引擎日志，找 LMCacheAscendConnector 相关行：
- 正常：应出现 `LMCacheAscendConnector`、`kv_transfer_config` 等字样
- 异常：`KeyError` 或 `Unknown connector` → factory 没注册上

**终端 B** — 发一次请求确认引擎正常响应：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"say hi"}],"max_tokens":10}'
```

**预期**：返回 JSON 含 `choices[0].message.content`。

---

## Step 6：跑 benchmark（可选，验证 before/after 效果）

参考 mimo-trial 的跑法（user-sim 用外部 API）：

```bash
cd /data/os_competition_TSJ

# 确认环境变量
echo $MIMO_KEY   # API key，未设则 export MIMO_KEY='tp-...'

# baseline 档
PYTHONPATH="agent-mem/src:$PYTHONPATH" .venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/baseline.yaml \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root logs-mimo \
    --user-model mimo-v2.5-pro \
    --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"

# F4 档
PYTHONPATH="agent-mem/src:$PYTHONPATH" .venv/bin/python agent-mem/benchmarks/runner.py \
    --config agent-mem/configs/f4-lmcache.yaml \
    --runner qwen-agent --engine-url http://127.0.0.1:8000/v1 \
    --model-name Qwen2.5-7B-Instruct --max-tasks 10 --runs 1 --max-steps 25 \
    --device npu --log-root logs-mimo \
    --user-model mimo-v2.5-pro \
    --user-api-base https://token-plan-cn.xiaomimimo.com/v1 \
    --user-api-key "$MIMO_KEY"

# 对照报告
PYTHONPATH="agent-mem/src:$PYTHONPATH" .venv/bin/python agent-mem/benchmarks/runner.py \
    --compare --study f4-trial --log-root logs-mimo
```

---

## 快速参考

| 组件 | 要求 | 当前 |
|---|---|---|
| CANN | >= 8.2 | 9.0.0 |
| vllm-ascend | >= 0.14.0 | 0.22.1rc1 |
| torch-npu | >= 2.7.1 | 2.10.0 |
| lmcache | 0.4.4（报冲突才降级） | 0.5.1 |
| lmcache-ascend | main | 待装 |

## 降级/兜底

如果 lmcache_ascend 最终装不上：用已有的 `configs/kv_offload.yaml`（`--kv-offloading-backend native`）作为 F4 的 native offload 变体。
