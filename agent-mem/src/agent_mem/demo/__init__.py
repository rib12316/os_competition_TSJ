"""agent-mem 演示前端（Gradio）：Qwen-Agent 对话 + 实时显存/指标监控 + 历史 before/after。

启动::

    PYTHONPATH="agent-mem/src:$PYTHONPATH" \\
    python -m agent_mem.demo --engine-url http://127.0.0.1:8000/v1 --model Qwen2.5-7B-Instruct

访问：服务绑 ``127.0.0.1:7860``；在笔记本上经 SSH 端口转发打开::

    ssh -L 7860:localhost:7860 <user>@<server>   # 然后浏览器开 http://localhost:7860

数据层见 :mod:`agent_mem.demo.monitor`（纯 Python，无 GUI 依赖，可单测）。
"""

from agent_mem.demo.monitor import (  # noqa: F401
    HistoryConfig,
    LiveMonitor,
    Sample,
    engine_status,
    load_history,
)
