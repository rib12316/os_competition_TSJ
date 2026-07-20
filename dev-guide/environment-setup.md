# 环境搭建记录

> 隔离环境：仓库根目录 `./.venv`（Python 3.11），与系统 Python 隔离。所有 uv 命令在仓库根目录执行。

## 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 按 installer 提示 source 环境变量（通常 ~/.local/bin/env 或 ~/.bashrc）
```

## 2. 创建隔离环境

```bash
cd /data/os_competition_TSJ
uv venv .venv --python 3.11
source .venv/bin/activate      # 或后续统一用 .venv/bin/python
```

## 3. 第三方克隆（MVP 核心，shallow）

```bash
cd /data/os_competition_TSJ/third_party
git clone --depth 1 --branch v0.22.1   https://github.com/vllm-project/vllm
git clone --depth 1 --branch v0.22.1rc1 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && git submodule update --init --recursive && cd ..
git clone --depth 1 https://github.com/QwenLM/Qwen-Agent qwen-agent
git clone --depth 1 https://github.com/sierra-org/tau-bench
```

> 克隆后回填 `third_party/VERSIONS.md`（tag + commit hash + 时间）。

## 4. 全量安装到 uv 环境（CUDA-free，实测可复现）

> 官方文档：https://docs.vllm.ai/projects/ascend/en/latest/installation.html
> 权威脚本（带 exit code 记录）：`logs/_installs/<最新>/run.sh`。本节为精简版。

```bash
cd /data/os_competition_TSJ
export PATH="$HOME/.local/bin:$PATH"          # uv
export UV_LINK_MODE=copy                       # cache 与 venv 跨盘，避免 hardlink 警告
export TORCH_DEVICE_BACKEND_AUTOLOAD=0         # 构建/非运行期禁用 torch_npu 自动加载
source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 注入 CANN 9.0.0 编译/运行环境

# Phase A —— CPU 侧，无需 NPU
uv pip install setuptools wheel packaging ninja setuptools-rust==1.13.0 setuptools-scm==10.2.0   # vllm --no-build-isolation 构建依赖（setuptools-rust/scm 不可少）
uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu   # +cpu，零 nvidia 包
uv pip install torch-npu==2.10.0.post2
( cd third_party/vllm && VLLM_TARGET_DEVICE=empty uv pip install -e . --no-build-isolation )   # empty-target 源码构建，非 PyPI vllm
uv pip install qwen-agent
( cd third_party/tau-bench && uv pip install -e . )
uv pip install -e 'agent-mem[dev]'             # 本项目 + 开发工具链(ruff/pytest/mkdocs)

# Phase B —— NPU 启动后（见 install-status.md「安装要点」）：
#   vllm-ascend 0.22.1rc1（华为云 variant wheel 的 -910b 后缀需去掉再 --no-deps 装）
#   triton-ascend 3.2.1（覆盖补丁 libtriton.so，使 triton.backends 含 'ascend'）
#   ⚠️ vllm-ascend 是 --no-deps 装，须手动补两个运行时依赖（否则 engine init 报 ModuleNotFoundError）：
#      uv pip install numba==0.60.0                                   # 兼容 numpy 1.26.4；policy_flashlb 硬 import
#      uv pip install torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu   # 必须钉 0.25.0！0.28 要 torch2.13 会撑爆 torch_npu（torch↔tv: 2.10↔0.25）
```

**PyPI 下大包卡死时的回退**（实测 triton 188MB 在 files.pythonhosted 会 socket 挂起）：
```bash
# 任意 uv pip install 命令追加 --index-url https://mirrors.aliyun.com/pypi/simple/  # 阿里云全量 PyPI 镜像，45s 跑完
```

**torch 版本解析失败时的回退**：
```bash
uv pip install torch==2.10.0 torch-npu==2.10.0.post2   # 先固定 torch，再装 vllm-ascend
```

## 5. 版本快照

安装完成后 `uv pip freeze` 写入 [`install-status.md`](install-status.md)。

## 6. 910b ops 补齐（CANN 系统级，**最关键**）⭐

> 若 `is_available()=True` 但推理报 `aclnnXxx failed, error 561103` / `Parse dynamic kernel config fail`，**根因是 910b 算子 kernel 缺失**（系统只装了 310p ops，装错芯片）。校验：`find …/cann-9.0.0/opp -ipath '*ascend910b*' -name '*.o' | wc -l` 应为 **9390**，若只有个位数即缺失。
>
> 华为官网 nnrt/ops 下载需登录（403），**唯一免认证渠道是官方 Docker 镜像层**：

```bash
# 1) 取 quay.io 匿名 token（公开仓库 ascend/cann）
TOKEN=$(curl -s "https://quay.io/v2/auth?service=quay.io&scope=repository:ascend/cann:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
# 2) 下载 9.0.0-910b 镜像的 ops 层（4.1GB，~5min @12MB/s；sha256 应为 1add324ee0a12edb09aa8919e423ce1f…）
curl -sL -H "Authorization: Bearer $TOKEN" \
  "https://quay.io/v2/ascend/cann/blobs/sha256:1add324ee0a12edb09aa8919e423ce1f2461c30e584e4f90b04f5d0243e41c76" \
  -o /data/cann_layer.tar.gz
# 3) 仅提取 910b ops 到系统（注意：kernel/config 与 data/tiling 不可少，否则 561103 不解）
tar -xzf /data/cann_layer.tar.gz --directory=/ --wildcards \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/op_host/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/op_api/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/data/tiling/Ascend910B/*" \
  "usr/local/Ascend/cann-9.0.0/opp/built-in/data/op/Ascend910B*" \
  "usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64/libopapi*"
# 4) 校验: kernel .o = 9390；然后 rm -f /data/cann_layer.tar.gz
```

## 参考：CANN 手动升级（NPU 启动后，系统级，独立于 uv 环境）

```bash
# 官方 9.0.1 工具链 + 910b ops + nnal
wget --header="Referer: https://www.hiascend.com/" \
  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-toolkit_9.0.1_linux-$(uname -i).run
# ./Ascend-cann-toolkit_9.0.1_linux-*.run --full  &&  source .../set_env.sh
# 同理 Ascend-cann-910b-ops_9.0.1 与 Ascend-cann-nnal_9.0.1
```
