"""agent: Qwen-Agent 包装层（MVP）。

ReAct + Function Call，多轮对话编排；工具：search / python 等。
客户端通过 OpenAI-compatible HTTP 协议直连本地 vLLM 推理引擎。
"""
