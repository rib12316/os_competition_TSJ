"""复现性信息采集（git commit + 关键版本快照）。

被 :mod:`agent_mem.bench.run_dir` 用来写 ``git_commit.txt`` / ``env.txt``，
让每个 run 目录从 day-1 就具备可复现性。
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

# 写入 env.txt 的关键包（用 importlib.metadata 查版本，避免直接 import 重包）
_TRACKED_PACKAGES = (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "tau_bench",
    "pyyaml",
    "openai",
    "httpx",
    "litellm",
)


def repo_root() -> Path:
    """从本文件向上找 ``.git`` 定位仓库根；找不到回退到 ``src`` 上三级。"""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return here.parents[3]  # src/agent_mem/repro.py → os_competition_TSJ


def git_commit(cwd: Path | None = None) -> str:
    """当前 HEAD commit hash；git 不可用/浅克隆时返回 ``"unknown"``。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or repo_root()),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or "unknown"
    except Exception:
        return "unknown"


def collect_env() -> dict[str, str]:
    """收集 python + 关键包版本（缺失记 ``not-installed``）。"""
    env: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in _TRACKED_PACKAGES:
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            env[pkg] = "not-installed"
    return env


def format_env_text(env: dict[str, str] | None = None) -> str:
    """把 env dict 渲染成 ``key=value`` 文本（``env.txt`` 内容）。"""
    env = env if env is not None else collect_env()
    return "\n".join(f"{k}={v}" for k, v in env.items()) + "\n"
