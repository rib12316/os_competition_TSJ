# 开发路线图

> 总方案：`docs/技术设计方案.md`（MVP 基础地基 + M1–M11 模块菜单）。
> 评分指挥棒：功能完整性 30% / 应用效果 40%（**必须 before/after 数字**）/ 代码规范 20% / 文档 10%。

## 优先级（按 ROI，从高到低）

```
M8(openEuler) ＞ M1 ＞ M5 ＞ M4 ＞ M2 ＞ M3 ＞ M6 ＞ M8(Ascend) ＞ M10 ＞ M11 ＞ M7 ＞ M9
```

## 阶段 0 — 基础设施（本次任务）
- [x] 目录脚手架（agent-mem / docs / third_party / logs / dev-guide）
- [x] 日志命名规范
- [ ] uv 隔离环境 + MVP 克隆 + 全量安装（Phase B/C/D）
- [ ] git + SSH 远程（Phase E）

## 阶段 1 — MVP（必做地基）
- [ ] vLLM V1 启动脚本（默认开 prefix cache）
- [ ] Qwen-Agent 包装层：ReAct + 多轮工具调用，直连 OpenAI 兼容 API
- [ ] Benchmark harness：τ-bench retail/airline，采集 6 指标
- [ ] 三档对照：baseline（`--no-enable-prefix-caching`）/ prefix_cache / optimized
- [ ] Docker compose 一键启动
- 命中：功能完整性 + 基础应用效果（约 65–70%）

## 阶段 2 — 高 ROI 模块
- [ ] **M8 openEuler 容器化**（1 天，命中国产 OS 硬性要求）
- [ ] **M1 KV FP8 量化**（`--kv-cache-dtype fp8`，KV 显存砍半）
- [ ] **M5 SGLang 双引擎**（分支共享 + 适配不同框架 20 分）
- [ ] **M4 LMCache 分层**（GPU↔CPU↔SSD，V1 标准 connector）
- 累计约 78–90%

## 阶段 3 — 深化（按时间余量）
- [ ] M2 Prompt 压缩（LLMLingua，配成功率 ablation）
- [ ] M3 session-aware 调度（复用 SharedStorageConnector）
- [ ] M6 工具数据 lazy-load
- [ ] M8 Ascend 适配（NPU 启动后 + CANN 9.0.1）
- [ ] M10 session KV checkpoint/恢复
- [ ] M7 显式分支 CoW（需配合 tree-search 工作流）
- [ ] M9 Mooncake 分布式 KV 池（仅多机）
- [ ] M11 Semantic Router（技术前瞻，文档为主）

## 每个优化模块的交付标准（叙事模板）
1. **现象**：agent 场景观察到的具体内存问题（带数据）
2. **根因**：为什么（agent 特征）
3. **机制**：优化方案（技术原理）
4. **验证**：before/after 数字（显存↓、延迟↓、成功率不掉）
5. **权衡**：代价（如压缩的成功率损失）

> 日志一律按 `dev-guide/log-naming-convention.md` 落盘，before/after 各 3 次取中位数。
