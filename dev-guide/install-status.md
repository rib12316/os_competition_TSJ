# 安装状态与已知问题

> 路线：**CUDA-free 全量安装** —— 目标 Ascend，torch 用 CPU 版、vllm 用 empty-target 源码构建，全程**零 nvidia CUDA 包**。
> NPU 已启动后，补装 vllm-ascend + triton-ascend，全栈可用。

## 重建状态（2026-07-19，CANN 升 9.0.0 后）⭐ 最新 —— ✅ 全栈就绪，smoke 通过

> 换镜像/升 CANN 后 base python 由 3.11.14 → **3.11.15**，旧 `.venv` 软链失效（uv 也丢），需整体重建。
> CANN 由 **8.5.1 → 9.0.0**。日志：`logs/_installs/20260719-024304_rebuild-cann9-phaseA/`（Phase A）、`logs/_installs/20260719-035337_rebuild-cann9-phaseB/`（Phase B/C，含 smoke1-4）。

**最终栈（实测 smoke OK，Qwen3-0.6B 在 Ascend910B2C 推理 26s）**

| 项目 | 状态 | 说明 |
|---|---|---|
| CANN 9.0.0 + **910b ops**（9390 kernel） | ✅ | **关键**：原先只装了 `Ascend-cann-310p-ops`（装错芯片！），910b ops 从 quay 镜像层补齐（见下） |
| torch 2.10.0+cpu + torch-npu 2.10.0.post2 | ✅ | CANN 9.0.0 下 `is_available()=True`，count=1，Ascend910B2C |
| vllm 0.22.1+empty（editable） | ✅ | 构建需补 `setuptools-rust==1.13.0`+`setuptools-scm==10.2.0` |
| **vllm-ascend 0.22.1rc1** | ✅ | wheel `--no-deps`；**须补 numba==0.60.0 + torchvision==0.25.0+cpu**（见下） |
| **triton-ascend 3.2.1**（补丁 libtriton） | ✅ | sha256 `5cc746d72fc383cd…`；triton.backends 含 `ascend` |
| qwen-agent / tau-bench / agent-mem[dev] | ✅ | ruff 全过、pytest 2 passed |
| **nvidia-* CUDA 包** | ✅ **0 个** | CUDA-free 达成 |

**⚠️ 三个真根因（旧栈 smoke 失败的真相，均与 CANN 版本无关）**：
1. **910b ops 缺失**（最致命）：系统只装了 `Ascend-cann-310p-ops`，缺 910b 算子 kernel → `aclnnInplaceZero failed, error 561103` / `Parse dynamic kernel config fail`。`is_available()` 能过（驱动层），但一执行真算子就崩。**修复**：从 `quay.io/ascend/cann:9.0.0-910b-openeuler24.03-py3.11` 镜像层提取（华为官网 nnrt 需登录 403，镜像层是唯一免认证渠道），见 environment-setup.md「910b ops 补齐」。
2. **numba 缺失**：vllm-ascend `--no-deps` 装导致其依赖 `numba` 没拉进来（`policy_flashlb.py` 硬 import）。`uv pip install numba==0.60.0`（兼容 numpy 1.26.4）。
3. **torchvision 缺失 + 版本陷阱**：vllm 源码构建没拉 torchvision（`vllm/v1/worker/block_table.py` import）。**必须钉 `torchvision==0.25.0+cpu`** —— 不钉则 uv 拉 0.28.0（要 torch 2.13.0），把 torch 升级从而撑爆 torch_npu 2.10.0.post2。torch↔torchvision 映射：2.10↔0.25。

**重建踩坑**：① PyPI 下大包（triton 188MB）会 socket 卡死（0% CPU），**改用阿里云镜像** `--index-url https://mirrors.aliyun.com/pypi/simple/` 45s 跑完。② vllm `--no-build-isolation` 构建依赖须预装（含 setuptools-rust/scm）。③ triton-ascend 补丁 libtriton 须 zipfile 强制覆盖（uv 严格不覆盖），判据用 **sha256 比对**而非 `nm -D|grep ascend`（后者恒为 0，误报）。

---

## 历史状态（CANN 8.5.1 时期，已被上方重建状态取代）

| 项目 | 状态 | 说明 |
|---|---|---|
| 目录脚手架 / uv `.venv` (py3.11) | ✅ | 与系统 Python 隔离 |
| MVP 克隆 | ✅ | vllm@v0.22.1 / vllm-ascend@v0.22.1rc1 / qwen-agent / tau-bench |
| torch 2.10.0+cpu + torch-npu 2.10.0.post2 | ✅ | NPU 可用（`is_available()=True`, count=1） |
| vllm 0.22.1+empty（editable） | ✅ | 可导入 |
| **vllm-ascend 0.22.1rc1** | ✅ | ascend 平台插件已注册；`import vllm_ascend` 正常 |
| **triton-ascend 3.2.1** | ✅ | triton 已注册后端 `['ascend']` |
| qwen-agent / tau-bench / agent-mem | ✅ | pytest/ruff 通过 |
| git + SSH + remote | ✅ | `github.com:rib12316/os_competition_TSJ` |
| **CANN** | ✅ **8.5.1 实测够用** | 驱动 26.0.rc1 足够；**无需升级 9.0.1** |

**硬件**：芯片 910B2C（Atlas A2，64GB HBM），设备 `/dev/davinci15`，驱动 26.0.rc1。

## 版本快照

```
# 推理栈
torch==2.10.0+cpu            torch-npu==2.10.0.post2
-e third_party/vllm          # vllm 0.22.1+empty
vllm-ascend==0.22.1rc1       triton==3.2.0   triton-ascend==3.2.1
# agent / benchmark
-e third_party/tau-bench     qwen-agent==0.0.34   -e agent-mem
transformers==5.14.1         numpy==1.26.4（triton-ascend 钉版）  soundfile==0.14.0
# 开发工具链
ruff==0.15.22  pytest==9.1.1  pytest-cov==7.1.0  mkdocs==1.6.1  mkdocs-material==9.7.7
# nvidia-* CUDA 包：0 个 ✓
```

## vllm-ascend / triton-ascend 安装要点（实测）

> 日志：`logs/_installs/20260719-010839_vllm-ascend/`。NPU 启动后操作。

1. **vllm-ascend**：`uv pip install vllm-ascend==0.22.1rc1` 会失败（uv 对华为云 variant 索引里带 `-910b` 后缀的 wheel 做 platform 校验，认为无兼容 wheel）。
   - 修复：直接下 wheel 本地装（绕过索引解析）：
     ```bash
     curl -O https://mirrors.huaweicloud.com/ascend/repos/pypi/variant/vllm-ascend/vllm_ascend-0.22.1rc1-cp311-cp311-manylinux_2_24_x86_64-910b.whl
     # 去掉文件名 -910b 后缀使 platform tag 合法，再 --no-deps 装
     uv pip install --no-deps ./vllm_ascend-0.22.1rc1-cp311-cp311-manylinux_2_24_x86_64.whl
     ```
2. **triton-ascend**：`uv pip install triton-ascend==3.2.1 --extra-index-url .../pypi` 能装上，但 uv/pip **不会**用 triton-ascend 的补丁 `libtriton.so`（含 ascend 符号）覆盖 PyPI triton 的同名文件（文件归属 triton，严格不覆盖）→ `import triton` 报 `cannot import name 'ascend'`。
   - 修复：下 triton-ascend wheel，用 python zipfile 把其 `triton/*` 文件**覆盖**到 site-packages：
     ```bash
     curl -O https://mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/triton_ascend-3.2.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
     python -c "import zipfile; z=zipfile.ZipFile('triton_ascend-...whl'); [open(f'site-packages/{n}','wb').write(z.read(n)) for n in z.namelist() if n.startswith('triton/') and not n.endswith('/')]"
     ```
   - 关键：运行前必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（libtriton 注册 ascend 子模块需 CANN 运行时在 LD_LIBRARY_PATH）。
3. **运行期**：`unset TORCH_DEVICE_BACKEND_AUTOLOAD`（仅构建期才需设 0）。

## 已知点 / 后续

- `import triton_ascend` 报 `No module named 'triton_ascend'` 是**正常的** —— triton-ascend 集成为 `triton.backends.ascend`（已注册），无顶层模块。
- `transformers` 当前 5.14.1（vllm 拉的），vllm-ascend 声明 `==5.5.4`；`--no-deps` 装未强制降级，运行若有 API 不兼容再按需降。
- **下一步（MVP）**：用 vllm-ascend 跑通真实推理（下载小模型如 Qwen3-0.6B，`vllm serve`），再进入 roadmap 的 agent loop + benchmark。
