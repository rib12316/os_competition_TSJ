# P1 — 引擎 / 后端开发

> 角色主线：**推理引擎启停 + KV/分层/国产化机制**。你是整个系统的「底座」——agent 和 benchmark 都跑在你提供的 OpenAI 兼容 API 之上。
>
> 协作契约见 [`README.md`](README.md)；总方案 [`../../docs/技术设计方案.md`](../../docs/技术设计方案.md)。

## 你的协作边界

- **产出接口（给别人用）**：① 引擎 OpenAI 兼容 API（含 tool_call）；② 引擎 `/metrics` 端点（P3 抓 KV 命中率）。
- **消费接口（别人给你）**：P3 的 `configs/*.yaml` schema（你的启动参数从这里读）。
- **依赖**：MVP 你**不被 P2 阻塞**（agent 调你，不是你调 agent）；你只依赖 P3 的 config schema，schema 没定之前先用 stub。
- **审查**：你的 PR 由 **P3 审查**。

---

## MVP 任务清单（必做地基）

> 命中：功能完整性（跑得通的引擎）+ 基础应用效果。落点：[`agent-mem/src/agent_mem/server/`](../../agent-mem/src/agent_mem/server/)。

- [ ] **vLLM V1 启动封装**：`src/agent_mem/server/vllm_server.py`，封装 `vllm serve`，默认开启 prefix cache（V1 默认即开，无需 flag）。
- [ ] **baseline 开关**：支持 `--no-enable-prefix-caching` 显式关闭，供三档对照（对应 [`configs/baseline.yaml`](../../agent-mem/configs/baseline.yaml)）。
- [ ] **从 config 驱动启动参数**：读 `configs/*.yaml` 的 `engine.backend / model / extra_args`，不要硬编码。
- [ ] **OpenAI 兼容 API 验证**：`/v1/chat/completions` 能正常返回，且 **tool calling 可用**（赛题核心：agent 靠这个做多步工具调用）。
- [ ] **`/metrics` 端点暴露**：保证 Prometheus 端点可达，确认 `vllm:gpu_prefix_cache_hits_total` 存在（P3 要抓）。
- [ ] **多模型支持**：Qwen2.5-7B-Instruct + MiniCPM3-4B 都能起（命中「多模型」评分项）。
  - ⚠️ **模型可用性现状**：本地目前只下载了 `models/Qwen3-0.6B/`（见 git log）。**先用 Qwen3-0.6B 跑通冒烟链路**（dev/调试快），真实 benchmark 再切换到 Qwen2.5-7B / MiniCPM3（需你规划模型下载，落地到 [`models/`](../../models/)）。
- [ ] **健康检查 / 就绪探针**：serve 起来后能被 compose / benchmark 轮询直到就绪（避免 P3 的 harness 连不上）。
- [ ] **Docker 基础镜像**：配合 P3 完善 [`docker/compose.yml`](../../agent-mem/docker/compose.yml) 的 `vllm` service（device 挂载、端口、volume）。
- [ ] **NPU/GPU 约束**：真机跑前确认设备已启动 —— **NPU 默认是停的，真跑前暂停等用户启动设备**，不要假设在线。

### MVP 集成时序（你的位置）

1. P3 先 merge 契约 + stub（day 1）→ 你拿到 config schema。
2. 你开 `feat/p1-engine` 实现 vLLM serve → PR → **P3 审查** → merge。
3. P2 此前对着 stub 写 agent，你 merge 后 P2 切真实引擎联调。

---

## 扩展任务（M1–M11，按 ROI 排序）

> 这些都是「你的」模块。做完一个就在本节勾掉，并按 [叙事模板](README.md#每个优化模块的交付标准叙事模板所有人通用) 附 before/after 数字。

### 高 ROI（优先做）

- [ ] **M8 国产化 — openEuler 容器化（必做，1 天）**：openEuler 24.0 上 vLLM 容器化部署。落点 [`deploy/openeuler/`](../../agent-mem/deploy/openeuler/)。直接命中赛题「至少一个国内开源 OS」硬性要求。
- [ ] **M1 KV FP8 量化（极低难度）**：`--kv-cache-dtype fp8`，KV 显存砍半。落点 [`kv/`](../../agent-mem/src/agent_mem/kv/)。配合 P3 做 baseline → +prefix → +FP8 三档递进。
- [ ] **M5 SGLang 双引擎（2 天）**：[`server/`](../../agent-mem/src/agent_mem/server/) 加 SGLang 后端，RadixAttention 替代 vLLM hash-based prefix cache。**协作**：P2 用同 agent loop 在 SGLang 跑通；P3 出 block 级 vs token 级对比。命中「分支推理内存共享」+「适配不同框架」。
- [ ] **M4 LMCache 分层（1–2 天）**：集成 LMCache 作 V1 KV connector，GPU↔CPU↔SSD 三级。落点 [`kv/`](../../agent-mem/src/agent_mem/kv/)（`lmcache-local.yaml`）。命中「分层内存与异构存储」。

### 中 ROI

- [ ] **M8 国产化 — Ascend NPU（3–5 天，建议做）**：vLLM-Ascend v0.18.0 + CANN 9.0.0。落点 [`deploy/ascend/`](../../agent-mem/deploy/ascend/)。⠂**NPU 启停由用户控制**，适配前先确认设备 + CANN 环境（见 [`../install-status.md`](../install-status.md)）。
- [ ] **M11 Semantic Router（文档为主，0.5 天）**：评估 vLLM SR Iris 在 agent 工作流的适用性，写进技术前瞻文档；有余力再做原型。

### 低 ROI（按时间余量）

- [ ] **M7 显式分支 CoW（1 周+，高难度）**：vLLM PagedAttention block manager 加 explicit fork，block 级 Copy-on-Write。**前置**：需先完成 M5，且要刻意构造 tree-search 工作流，否则收益和默认 prefix sharing 重叠。
- [ ] **M9 Mooncake 分布式 KV 池（1 周+）**：仅多机场景启用，需 RDMA。单机**不要做**。

---

## 交付标准（每个任务完成时自检）

- [ ] 启停脚本可复现：`configs/*.yaml` 驱动，不依赖手动改代码。
- [ ] 接口契约不变（OpenAI API + `/metrics`），否则通知 P2/P3。
- [ ] before/after 数字交 P3（同 seed、3 次中位数）。
- [ ] 成功率下降 ≤ 2 个百分点。
- [ ] `ruff check` + `pytest`（P3 的 CI 会卡）。
