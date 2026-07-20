"""缝F · 自带工作流的研究/demo 落点（F7 分支 KV 共享 · 测量版）。

F7 不造新 fork API，而是构造分支 workload（self-consistency / tree-of-thought），
**测量** vLLM Automatic Prefix Caching 的隐式 block 级 CoW 共享（block 引用计数复用）。

稳定接口已是现成的 :class:`agent_mem.bench.runner.Runner` Protocol——F7 在此实现一个
``TreeOfThoughtsRunner(Runner)``（自带 ToT driver + 构造对照），不改 runner 主体。

当前为占位（F7 优先级最低、难度高，1 周+）。落地参考：
- vLLM APC 设计：https://docs.vllm.ai/en/stable/design/prefix_caching/
- workload 借鉴：https://github.com/princeton-nlp/tree-of-thought-llm
"""
