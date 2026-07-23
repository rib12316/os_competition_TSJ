"""模型输入 token 的公共 JSONL 记录器，与任何 middleware/优化方法无关。"""

from __future__ import annotations

import json
import os
import threading
import time

from agent_mem.middleware import MiddlewareContext

_LOG_LOCK = threading.Lock()


def log_prompt_tokens(ctx: MiddlewareContext, prompt_tokens: int | None) -> None:
    """记录服务端 ``usage.prompt_tokens``；未配置日志路径时零开销返回。"""
    path = os.environ.get("PROMPT_TOKEN_LOG", "")
    if not path:
        return
    payload = {
        "session_id": ctx.session_id,
        "step": ctx.step,
        "prompt_tokens": prompt_tokens,
        "source": (
            "response.usage.prompt_tokens" if prompt_tokens is not None else "unavailable"
        ),
        "ts": time.time(),
    }
    try:
        with _LOG_LOCK:
            with open(path, "a") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass
