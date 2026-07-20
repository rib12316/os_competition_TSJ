"""缝C · LMCache 配置解析（F4 分层存储）。

把 ``configs/*.yaml`` 里 ``engine.lmcache`` 段翻译成：

- CLI flag：``--enable-lmcache``（由 :func:`agent_mem.server.vllm_server.build_serve_args` 在
  ``lmcache.enabled`` 时附加）。
- 环境变量：``LMCACHE_CONFIG_FILE``（指向本包打包的模板或用户自定 yaml）。

模板见 :data:`LMCACHE_TEMPLATE`（本目录 ``lmcache_local.yaml``）。``config_file`` 可写：

- 省略 / ``"local"`` / ``"default"`` → 用打包模板（:func:`resolve_lmcache_config`）；
- 任意路径 → 原样解析（相对路径按 cwd）。

解析是纯函数，无需 NPU / LMCache 安装，可单测。
"""

from __future__ import annotations

from pathlib import Path

from agent_mem.config import AppConfig, LmCacheConfig

# 打包在本目录的模板文件名
LMCACHE_TEMPLATE = "lmcache_local.yaml"
# 视为「用打包模板」的别名
_TEMPLATE_ALIASES = frozenset({"local", "default", "builtin", ""})


def _packaged_template_path() -> Path:
    """返回打包模板的绝对路径（本模块同目录下的 ``lmcache_local.yaml``）。"""
    return Path(__file__).resolve().parent / LMCACHE_TEMPLATE


def resolve_lmcache_config(spec: str | None) -> Path | None:
    """把 ``engine.lmcache.config_file`` 解析成绝对路径。

    - ``None`` / ``"local"`` / ``"default"`` → 打包模板；
    - 其它字符串 → 按路径解析（不校验存在性——真机才知文件在哪）。
    """
    if spec is None or spec.strip() in _TEMPLATE_ALIASES:
        return _packaged_template_path()
    return Path(spec).expanduser()


def lmcache_serve_flag(lm: LmCacheConfig) -> list[str]:
    """``lm.enabled`` 时返回 ``["--enable-lmcache"]``，否则空列表。"""
    return ["--enable-lmcache"] if lm.enabled else []


def lmcache_env(cfg: AppConfig) -> dict[str, str]:
    """返回需注入引擎子进程的 LMCache 环境变量（键→值）。

    仅当 ``lmcache.enabled`` 且 config 解析成功时给 ``LMCACHE_CONFIG_FILE``。
    """
    lm = cfg.engine.lmcache
    if not lm.enabled:
        return {}
    path = resolve_lmcache_config(lm.config_file)
    if path is None:
        return {}
    return {"LMCACHE_CONFIG_FILE": str(path)}
