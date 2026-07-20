"""benchmark 任务适配器子包（τ-bench 等）。

注意：本包的 ``__init__`` 刻意不 re-export 任何符号，避免在 import 子包时
意外触发 ``tau_bench`` 等重依赖的加载。请直接从具体模块导入，例如::

    from agent_mem.bench.tasks.tau_bench_adapter import list_tasks, TaskInfo
"""
