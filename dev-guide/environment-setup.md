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

## 4. 全量安装到 uv 环境（遵循 vllm-ascend 官方文档）

> 版本 > 0.11.0：**不设**清华全局镜像；仅 vllm-ascend / triton-ascend 用华为云 extra-index。
> 官方文档：https://docs.vllm.ai/projects/ascend/en/latest/installation.html

```bash
cd /data/os_competition_TSJ
# 注入 CANN 编译/运行环境变量（本机 CANN 8.5.1，仅作编译环境）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

uv pip install vllm==0.22.1
uv pip install vllm-ascend==0.22.1rc1 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant
uv pip install triton-ascend==3.2.1 \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi   # 必须最后装
uv pip install qwen-agent
uv pip install -e third_party/tau-bench
uv pip install -e agent-mem[dev]                                         # 本项目 + 开发工具链
```

**torch 版本解析失败时的回退**：
```bash
uv pip install torch==2.10.0 torch-npu==2.10.0.post2   # 先固定 torch，再装 vllm-ascend
```

## 5. 版本快照

安装完成后 `uv pip freeze` 写入 [`install-status.md`](install-status.md)。

## 参考：CANN 手动升级（NPU 启动后，系统级，独立于 uv 环境）

```bash
# 官方 9.0.1 工具链 + 910b ops + nnal
wget --header="Referer: https://www.hiascend.com/" \
  https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.1/Ascend-cann-toolkit_9.0.1_linux-$(uname -i).run
# ./Ascend-cann-toolkit_9.0.1_linux-*.run --full  &&  source .../set_env.sh
# 同理 Ascend-cann-910b-ops_9.0.1 与 Ascend-cann-nnal_9.0.1
```
