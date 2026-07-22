"""隔离 venv 里常驻的压缩 worker（transformers 4.x + llmlingua）。

为什么单独一个进程：主 venv 的 transformers 被 vllm 锁在 5.x，而 llmlingua 0.2.2
只兼容 transformers 4.x（4.x 返回 legacy 格式的 past_key_values）。两者无法共存于
同一解释器，于是把"真压缩"放进一个独立 venv 的常驻子进程，``CompressMiddleware``
通过 stdin/stdout 与它通信。模型只加载一次、跨多次压缩复用。

协议（行式 JSON over stdin/stdout；stderr 走日志文件不参与协议）：
  1) 启动：读一行 config ``{"model_name","use_llmlingua2","device"}``，
     加载 ``PromptCompressor``，回写 ``{"ready": true}``。
  2) 循环：每行一个请求 ``{"args": [...], "kw": {...}}``，调
     ``comp.compress_prompt(*args, **kw)``，回写 ``{"result": {...}}`` 或 ``{"error": "..."}``。
  3) 收到 ``EXIT`` 行退出。

它只是 ``compress_prompt`` 的**透明代理**——调用方（``_compress_cold``）用和在进程内
完全一样的参数调 ``compress_prompt``，worker 原样转发，因此不产生分支逻辑重复。

由 ``_SubprocessCompressor``（compress.py）用隔离 venv 的 python 拉起。
"""

from __future__ import annotations

import json
import sys
import traceback


def main() -> None:
    config_line = sys.stdin.readline()
    if not config_line:
        return
    config = json.loads(config_line)

    from llmlingua import PromptCompressor

    comp = PromptCompressor(
        model_name=config.get("model_name"),
        use_llmlingua2=config.get("use_llmlingua2", False),
        device_map=config.get("device", "cpu"),
    )
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "EXIT":
            break
        try:
            req = json.loads(line)
            result = comp.compress_prompt(*req.get("args", []), **req.get("kw", {}))
            sys.stdout.write(json.dumps({"result": result}) + "\n")
        except Exception as e:  # noqa: BLE001 — 任何异常都回写给调用方，不杀 worker
            sys.stdout.write(
                json.dumps({"error": repr(e), "trace": traceback.format_exc()}) + "\n"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
