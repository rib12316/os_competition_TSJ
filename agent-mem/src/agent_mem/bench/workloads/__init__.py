"""缝F · 自带工作流的研究/demo 落点（F7 分支 KV 共享 · 测量版）。

F7 不造新 fork API，而是构造分支 workload，**验证** vLLM APC 的 block 级 CoW
在多分支推理下的显存节省效果。

两档对照（各独立起引擎、独立 benchmark）：
- f7-branch-indep：N 个请求前缀各不相同 → APC 无法共享 → mem_peak 高
- f7-branch-share：N 个请求共享同一前缀 → APC CoW 复用 → mem_peak 低

Runner：:class:`BranchingKVShareRunner`（见 branching_runner.py）。
"""

from agent_mem.bench.workloads.branching_runner import BranchingKVShareRunner

__all__ = ["BranchingKVShareRunner"]
