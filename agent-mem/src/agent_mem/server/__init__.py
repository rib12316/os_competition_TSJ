"""server: 推理引擎启动封装（缝A flag 预设 / 缝B 引擎后端 / 缝C KV connector）。

封装 vLLM V1 与 vLLM-Ascend 的启动参数与 OpenAI 兼容入口（SGLang 在 v2 作废：
不支持 Ascend）。``build_serve_args`` 把 config 翻译成 CLI flag，``engine_env`` 注入
LMCache 等环境变量。
"""
