"""F4 · LMCache Ascend 可用性检查（纯 Python，无 NPU 时可 import）。

在 vllm-ascend 起引擎前调用 :func:`check_lmcache_ascend` 确认
``lmcache_ascend`` 包是否已安装。未安装时给出清晰的安装指引。
"""

from __future__ import annotations


def is_lmcache_ascend_available() -> bool:
    """``lmcache_ascend`` 包是否可 import。"""
    try:
        import lmcache_ascend  # noqa: F401
        return True
    except ImportError:
        return False


def check_lmcache_ascend() -> None:
    """检查 lmcache_ascend 可用性；不可用时抛 RuntimeError 并给出安装说明。

    NPU 开时调用此函数做前置检查，避免引擎启动后发现 connector 不存在。
    """
    if is_lmcache_ascend_available():
        return
    raise RuntimeError(
        "lmcache_ascend 未安装，无法使用 LMCacheAscendConnector。\n"
        "NPU 上的安装步骤见 docs/F4-lmcache-ascend.md。\n"
        "简版步骤：\n"
        "  1. cd /tmp && git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git\n"
        "  2. cd LMCache-Ascend && .venv/bin/python -m pip install --no-build-isolation -e .\n"
    )