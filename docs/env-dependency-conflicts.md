# 环境 / 依赖冲突处理记录

> 给团队留档：本仓库共享主 venv（`.venv`）里跑着 vllm / vllm-ascend / litellm / tau-bench /
> qwen-agent / transformers / lmcache 等一堆重依赖，各自的版本要求互相冲突，且 venv 是
> **全队共享**的——任何人在上面装东西都可能改坏别人。本文记录已遇到的冲突及解法，供后续参照。

## 核心原则：能隔离就别动主 venv

主 venv 被 vllm 0.22 / vllm-ascend **锁死**（transformers 5.x、特定 torch/openai 等），动一个就可能
连环崩引擎。所以遇到冲突时，**优先"隔离"而非"在主 venv 里改版本"**：

| 冲突类型 | 解法 |
|---|---|
| 某功能用的库和主 venv 的核心库版本互斥（如 llmlingua vs transformers） | **开独立 venv + 子进程**调用，主 venv 一点不动 |
| 库之间只是缺个别属性（如 litellm vs aiohttp） | **运行时打补丁**（import 前补上缺失符号），不改任何包 |
| 主 venv 自身被改坏（如 openai 被降级） | 恢复主 venv 到一致状态（慎用，见末节） |

---

## 冲突 1：transformers 5.x ↔ llmlingua 4.x（已解决 · 隔离 venv）

### 现象
F2 要用 `llmlingua`（LongLLMLingua prompt 压缩）。装上后一调 `compress_prompt` 就崩：

```
File ".../llmlingua/prompt_compressor.py", line 1659
    for k, v in past_key_values
RuntimeError: ...  # 实为 ValueError: too many values to unpack
```

### 根因
- `llmlingua==0.2.2`（PyPI 最新，已停更）只兼容 **transformers 4.x**——它假设 `model(...)` 返回的
  `past_key_values` 是"元组套元组"（legacy 格式），有 5 处 `for k,v in past_key_values`。
- 主 venv 的 **transformers 5.14.1** 把 `past_key_values` 改成了 `Cache` 对象 → 5 处全崩。
- transformers 5.x 被 **vllm 0.22 / vllm-ascend / lmcache 锁定**，**不能降级**（降了引擎栈崩）。
- 没有适配 5.x 的新版 llmlingua。

**结论：llmlingua 无法和主 venv 同进程共存。**

### 解法：独立 venv + 子进程（主 venv 零改动）

不在主 venv 装 llmlingua、也不降主 venv 的 transformers，而是**另开一个 venv 专门跑压缩**：

```bash
# 1) 建隔离 venv（transformers 4.x + llmlingua + torch-cpu）
uv venv /data/os_competition_TSJ/.venv-compress
uv pip install --python .venv-compress/bin/python \
  "transformers==4.43.4" llmlingua \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cpu
# 结果：transformers 4.43.4 + torch 2.x cpu + llmlingua 0.2.2
```

> 选 `transformers==4.43.4`：`<4.44` 才返回 legacy 格式 `past_key_values`（4.44 起默认 DynamicCache，
> 又会崩）。torch 用 CPU 版（压缩小模型放 CPU，NPU 让给主 LLM）。

2) **常驻 worker 脚本**（在隔离 venv 里跑）：`agent_mem/middleware/_compress_worker.py`
   - 行式 JSON over stdin/stdout：启动读一行 config → 加载 `PromptCompressor`（**模型只加载一次**）→
     循环读 `{"args":[...],"kw":{...}}` → 调 `compress_prompt(*args,**kw)` → 回 `{"result":...}`。
   - 它是 `compress_prompt` 的**透明代理**（原样转发参数），所以调用方代码无需分支。

3) **主 venv 侧的客户端**：`agent_mem/middleware/compress.py` 里的 `_SubprocessCompressor`
   - 懒拉起 worker（`subprocess.Popen`，stderr 重定向到文件避免 PIPE 死锁），duck-type 成
     `PromptCompressor`（暴露同签名 `compress_prompt(*args,**kw)`）。
   - 模型加载一次、跨多次压缩复用；主 venv 进程**完全不 import llmlingua/transformers**。

```
f2-compress.yaml (backend=subprocess, worker_venv=.venv-compress/bin/python)
   → CompressMiddleware._get_compressor() → _SubprocessCompressor
        → stdin/stdout JSON → _compress_worker.py（在 .venv-compress 里）
              → llmlingua.PromptCompressor(gpt2)  # transformers 4.x，主 venv 不受影响
```

`.venv-compress/` 是机器本地，**不进 git**（应加进 `.gitignore`）。需在别处重建时，按上面命令重装即可。

---

## 冲突 2：litellm 1.92 ↔ aiohttp 3.8.4（已解决 · 运行时补丁）

### 现象
tau-bench 的 user-sim 走 litellm，`from litellm import completion` 即崩：

```
File ".../litellm/llms/custom_httpx/aiohttp_transport.py", line 24
AttributeError: module 'aiohttp' has no attribute 'ConnectionTimeoutError'
```

### 根因
litellm 1.92.0 的异常映射表引用了 `aiohttp.ConnectionTimeoutError` / `aiohttp.SocketTimeoutError`，
但主 venv 的 aiohttp 3.8.4 已移除这两个类。aiohttp 被 vllm/lmcache/litellm/dashscope 共用，**不能随便改**。

### 解法：运行时补丁（不改包）
litellm 只是拿这两个类**填映射表**（不实际抛），所以 import 前把缺失属性补上即可。用隔离 wrapper
`/tmp/f2-shim/run_bench.py`（仅对 benchmark 进程生效，靠 PYTHONPATH 注入，**不改任何包**）：

```python
import asyncio, aiohttp, runpy, sys
for n, v in [("ConnectionTimeoutError", asyncio.TimeoutError),
             ("SocketTimeoutError", asyncio.TimeoutError)]:
    if not hasattr(aiohttp, n):
        setattr(aiohttp, n, v)
sys.argv = [".../benchmarks/runner.py"] + sys.argv[1:]
runpy.run_path(".../benchmarks/runner.py", run_name="__main__")
```

---

## 冲突 3：litellm 1.92 ↔ openai（当前 · 待定）

### 现象
`from litellm import completion` 崩在 `from openai._models import BaseModel`：

```
ModuleNotFoundError: No module named 'openai._models'
```
且 agent 侧 `tau_bench_adapter.py:215` 用 `OpenAI(base_url=...)`（需 openai≥1.x）。

### 根因
主 venv 的 openai 是 **0.28.1**（古老版，无 `OpenAI` 类、无 `_models`），和 litellm 1.92（需 openai≥1.x）
本就冲突。更糟：openai 在会话期间**被降级过**（极可能是队友在这台共享机器上装 qwen-agent 等带的，
qwen-agent 老版钉 `openai==0.28.*`）——mimo 版 baseline 档（success=0.33）是降级**前**跑通的，
f2 档就触发了。

### 选项（未定，等指令）
- **(a) 升级主 venv 的 openai 到 ≥1.x**（如 2.46.0）：一行修复，但**动共享 venv**，可能影响队友
  （若他们正依赖 0.28.1）。⚠️ 这是"升级"不是"降级"——0.28.1 才是坏的那个。
- **(b) 把整个 benchmark 跑在独立 venv**（openai≥1.x + litellm + tau-bench，仿冲突 1 的隔离思路）：
  不动主 venv，但要重建一个较重的 benchmark venv。
- **(c) 治本**：团队约定不在主 venv 上各自 `pip install`，改用 per-feature 子 venv + 锁定依赖清单。

---

## 已踩坑清单（汇总）

| # | 冲突 | 触发点 | 解法 | 状态 |
|---|------|--------|------|------|
| 1 | transformers 5.x ↔ llmlingua 4.x | F2 压缩 | 隔离 `.venv-compress` + 子进程 worker | ✅ 已解决 |
| 2 | litellm ↔ aiohttp 3.8.4 | tau-bench user-sim | 运行时补丁（wrapper） | ✅ 已解决 |
| 3 | litellm ↔ openai 0.28.1 | tau-bench user-sim + agent | 待定（升 openai 或隔离） | ⏸ 等指令 |
| 附 | 模型目录残留 F1 C8 产物 | 引擎启动崩 `_prepare_c8_scales` | 改名 `.c8bak` 还原 stock | ✅ 已处理 |

## 建议（给团队）

1. **主 venv 视为只读**：不在上面直接 `pip install`；新功能用独立子 venv（见冲突 1）。
2. **每个隔离 venv 记录重建命令**（本文 + 各 F 文档），机器本地、不进 git、`.gitignore` 屏蔽。
3. **锁定依赖清单**：主 venv 的关键版本（vllm/vllm-ascend/transformers/torch/litellm/openai）
   写进 `third_party/VERSIONS.md` 或 `dev-guide/install-status.md`，避免被无意改坏。
4. 冲突优先级：**隔离子 venv > 运行时补丁 > 动主 venv**。
