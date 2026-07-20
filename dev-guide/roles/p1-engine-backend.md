# P1 — 引擎 / 后端开发

> 角色主线：**推理引擎启停 + KV/分层/部署机制**。你是整个系统的「底座」——agent 和 benchmark 都跑在你提供的 OpenAI 兼容 API 之上。
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)（v2）。

## 你的协作边界

- **产出接口**：① 引擎 OpenAI 兼容 API（含 tool_call）；② 引擎 `/metrics` 端点（P3 抓 KV 命中率）；③ 每个功能的 yaml preset（缝A/B/C）。
- **消费接口**：P3 的 `configs/*.yaml` schema + stub server（无 NPU 时联调）。
- **依赖**：你不被 P2 阻塞（agent 调你，不是你调 agent）；只依赖 P3 的 config schema。
- **审查**：你的 PR 由 **P3 审查**。

## 你负责的功能（5 个，各自独立）

> 每个功能 = 一条 feature 分支 + 一份独立 yaml + 自己的 before/after。彼此无前置依赖，可任意并行。

### F8 — 多硬件对照（缝B，最先做，ROI 最高）

- **做什么**：同一 agent loop + 同一 yaml，`backend: vllm`(CPU) vs `vllm-ascend`(NPU) 各跑一遍对比，命中"适配不同框架/硬件"20 分项（顶替已作废的 SGLang 双引擎）。
- **落点**：`config.py` 的 backend 白名单（已有 vllm / vllm-ascend）；两套 yaml。
- **现状**：✅ config/yaml/文档已就绪。纯 backend 字段切换，见 docs/F8-multi-hw.md。

### F1 — int8 KV 量化（缝A，待真机验证）

- **做什么**：`--kv-cache-dtype int8`（一行 flag），KV 显存砍半。
- **落点**：`configs/f1-int8.yaml`（走 `EngineConfig.extra_args`）。
- **⚠️ 高风险**：int8 是 Ascend 上 KV 量化的唯一支持格式，但曾在 0.13rc1 被移除（[#5630](https://github.com/vllm-project/vllm-ascend/issues/5630)），0.22.1rc1 是否恢复**未知**。
- **第一步动作（需 NPU）**：起 vllm-ascend 加 `--kv-cache-dtype int8`，看是否报错。一锤定音。
- **备选**：若不支持，降级为 max-num-seqs / gpu-memory-utilization 调参顶替"显存↓"叙事。
- **工作量**：验证 0.5 天 + benchmark 1 天。

### F4 — LMCache 分层（缝C）

- **做什么**：`--enable-lmcache` + 一份 `LMCACHE_CONFIG_FILE` yaml，NPU↔CPU↔Disk 三级。命中"分层内存与异构存储"。
- **落点**：`kv/`（`lmcache.yaml`）；本仓已带 `lmcache_integration/`。
- **开源借鉴**：[LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)（社区插件，2025-11 官方支持 NPU）。
- **工作量**：配置 + 调试 1–2 天。

### F7 — 分支 KV 共享（测量版）（缝F）

- **做什么**：构造分支 workload（self-consistency / tree-of-thought），**测量 vLLM APC 的隐式 block 级 CoW 共享**，而非造新 fork API（vLLM 无公开 fork API，[tree-attention #3960](https://github.com/vllm-project/vllm/issues/3960)）。
- **落点**：`bench/workloads/`（自带 ToT driver，走 `Runner` Protocol）。
- **开源借鉴**：[Tree of Thoughts](https://github.com/princeton-nlp/tree-of-thought-llm)；vLLM [Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/)。
- **工作量**：1 周+ · 难度高（自带 driver + 构造对照）。时间余量不足可只做文档叙事。

### F9 — openEuler / Ascend 部署（缝G）

- **做什么**：openEuler = 用 CANN 的 openeuler24.03 镜像作基底 + deploy 脚本；Ascend = 本栈已是 vllm-ascend（已就绪）。命中赛题"至少一个国内开源 OS"硬性要求。
- **落点**：`deploy/openeuler/`、`deploy/ascend/`、`docker/Dockerfile`（当前是 TODO 占位）。
- **工作量**：1 天。

## 已完成的 MVP 基座（你的产出，已 merge）

- [x] vllm-ascend serve 封装（`server/vllm_server.py`，spawn 子进程 + `/health` 轮询）
- [x] prefix-cache 开关（baseline 对照显式 `--no-enable-prefix-caching`）
- [x] `/metrics` 端点（`vllm:gpu_prefix_cache_hits_total`）
- [x] 从 config 驱动启动参数（`config.py` 的 backend 白名单）
- [ ] 多模型：Qwen3-0.6B 已跑通；Qwen2.5-7B / MiniCPM3 需下载后切换（命中"多模型"项）

## 交付标准（每个功能完成时自检）

- [ ] 启停脚本可复现：yaml 驱动，不依赖手动改代码。
- [ ] 接口契约不变（OpenAI API + `/metrics`），否则通知 P2/P3。
- [ ] before/after 数字交 P3（同 seed、3 次中位数）。
- [ ] 成功率下降 ≤ 2pp。
- [ ] `ruff check` + `pytest`（P3 的 CI 会卡）。
- [ ] 真机步骤前确认 NPU 已启动（**NPU 默认停着，暂停等用户**）。
