"""通用 agent CLI 测试（对 stub server 跑通）。"""

from __future__ import annotations

import pytest

# cli.py 顶层 import openai —— 未装时整个模块收集会崩，故先 importorskip。
pytest.importorskip("openai")

from agent_mem import cli  # noqa: E402
from agent_mem.server import stub_openai  # noqa: E402


def test_cli_runs_against_stub(capsys):
    srv, base_url = stub_openai.start_background(port=0)
    try:
        rc = cli.main([
            "算一下 2 加 3",
            "--engine-url", base_url,
            "--model", "Qwen3-0.6B",
            "--max-steps", "2",
        ])
        assert rc == 0
        out = capsys.readouterr()
        # session_id 打到 stderr
        assert "session_id=" in out.err
        # 最终回答打到 stdout（stub 一直返回 tool_call，2 步后截断）
        assert out.out.strip()
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_trace_shows_tool_call(capsys):
    srv, base_url = stub_openai.start_background(port=0)
    try:
        cli.main([
            "search for x",
            "--engine-url", base_url,
            "--max-steps", "1",
        ])
        out = capsys.readouterr()
        # stub 带 tools 返回 tool_call → trace 里应见 call tool
        assert "call tool" in out.err
    finally:
        srv.shutdown()
        srv.server_close()
