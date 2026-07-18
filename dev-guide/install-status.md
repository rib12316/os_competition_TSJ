# 安装状态与已知问题

> 安装日志：`logs/_installs/20260718-020621_uv-install/`（`install.log` + `uv-freeze.txt`）。
> 路线：**CUDA-free 全量安装** —— 目标是 Ascend，故 torch 用 CPU 版、vllm 用 empty-target 源码构建，全程**零 nvidia CUDA 包**。

## 当前状态（✅ 基础设施 + CPU 环境就绪）

| 项目 | 状态 | 说明 |
|---|---|---|
| 目录脚手架 | ✅ | agent-mem / docs / third_party / logs / dev-guide |
| uv + `.venv` (py3.11) | ✅ | 与系统 Python 隔离 |
| MVP 克隆 | ✅ | vllm@v0.22.1 / vllm-ascend@v0.22.1rc1(+子模块) / qwen-agent / tau-bench |
| torch + torch-npu | ✅ | CPU torch + Ascend 后端（运行需 NPU） |
| vllm (empty target) | ✅ 可导入 | 可编辑装入，`import vllm` 正常 |
| qwen-agent / tau-bench / agent-mem | ✅ | 均可导入；pytest 2 passed、ruff 通过 |
| **vllm-ascend** | ⏸️ **DEFER** | **按用户要求暂不装**，等 NPU 启动 |
| **triton-ascend** | ⏸️ **DEFER** | 随 vllm-ascend 一起，NPU 启动后装 |
| git + SSH + remote | ✅ | 已 push 到 `github.com:rib12316/os_competition_TSJ` |

## 版本快照（uv pip freeze，共 200 包，0 个 nvidia 包）

```
# 核心
torch==2.10.0+cpu
torch-npu==2.10.0.post2
-e file:///data/os_competition_TSJ/third_party/vllm            # vllm 0.22.1+empty
-e file:///data/os_competition_TSJ/third_party/tau-bench       # tau-bench 0.1.0
-e file:///data/os_competition_TSJ/agent-mem                   # agent-mem 0.0.1
qwen-agent==0.0.34
transformers==5.14.1
numpy==2.3.5 / pyyaml==6.0.3 / soundfile==0.14.0
# 开发工具链
ruff==0.15.22 / pytest==9.1.1 / pytest-cov==7.1.0
mkdocs==1.6.1 / mkdocs-material==9.7.7
# DEFER（未装）
# vllm-ascend==0.22.1rc1 / triton-ascend==3.2.1   ← NPU 启动后装
```
完整 freeze：`logs/_installs/20260718-020621_uv-install/uv-freeze.txt`。

## 安装要点（CUDA-free，实测有效）

```bash
export UV_LINK_MODE=copy                          # uv 缓存在 / 而 venv 在 /data，跨文件系统
export TORCH_DEVICE_BACKEND_AUTOLOAD=0            # 构建期禁用 torch_npu 自动加载，否则报错
# 1) CPU torch（无 nvidia 包）+ torch-npu
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-npu==2.10.0.post2 --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
# 2) vllm 源码 empty-target（需预装构建依赖）
uv pip install setuptools wheel packaging ninja setuptools_rust setuptools_scm
cd third_party/vllm && VLLM_TARGET_DEVICE=empty uv pip install -e . --no-build-isolation
# 3) 纯 CPU 三件套
uv pip install qwen-agent soundfile               # soundfile 是 qwen-agent 隐式依赖
uv pip install -e third_party/tau-bench
uv pip install -e 'agent-mem[dev]'
```

## 已知问题 / 待办

### ⏸️ vllm-ascend / triton-ascend（DEFER，NPU 启动后处理）
- 用户指示：NPU 未启动前不装 vllm-ascend（装上也跑不起来）。
- **安装坑（已查明，待 NPU 后用）**：`uv pip install vllm-ascend==0.22.1rc1` 会失败 —— uv 对华为云 variant 索引里带芯片后缀的 wheel（`...x86_64-910b.whl`）做严格 platform 校验，认为"无兼容 wheel"，且提示的 `--index-strategy unsafe-best-match` 无效（PyPI 上 vllm-ascend 只到 0.18.0）。
- **可行修复**：直接下载目标 wheel 本地安装（绕过索引解析），例如
  `curl -O https://mirrors.huaweicloud.com/ascend/repos/pypi/variant/vllm-ascend/vllm_ascend-0.22.1rc1-cp311-cp311-manylinux_2_24_x86_64-910b.whl && uv pip install ./vllm_ascend-...910b.whl`
  （若 uv 仍拒标签，把文件名 `-910b` 去掉再装）。芯片型号默认 `910b`（Atlas A2），**NPU 启动后用 `npu-smi info` 确认**。

### ⚠️ CANN 版本不匹配
- 需求 vllm-ascend 0.22.1rc1 → **CANN 9.0.1**；本机 `/usr/local/Ascend/cann-8.5.1`（**8.5.1**）。
- 待办：NPU 启动后按 `environment-setup.md`「CANN 手动升级」升级（系统级 `.run` 包）。

### ⚠️ NPU 未启动
- `/dev/davinci_manager`、`/dev/hisi_hdc` 在，但无 `/dev/davinci0`；`npu-smi info` 报 `dcmi ... ret -8005`。
- 用户：需要时再启动。运行时验证全部 defer。

### 其它
- **构建变量**：`TORCH_DEVICE_BACKEND_AUTOLOAD=0`（构建/非运行期）；运行 NPU 时勿设。
- **soundfile**：qwen-agent 隐式依赖（`utils.py` 无条件 import），已补装。
- **pytest 路径**：在 `agent-mem/` 下运行 `pytest`（根目录跑会误收集 `third_party/*/tests`）。
- 体积：`.venv` 约 2.1G；`/data` 余量充足。
