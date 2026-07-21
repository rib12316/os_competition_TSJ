"""F7 · 分支 KV 共享 Runner —— 验证 vllm APC 的多分支 CoW 共享。

两档对照：
- f7-indep（独立前缀）：N 个请求，前缀各不相同 → APC 无法共享
- f7-share（分支共享）：N 个请求，共享同一前缀 → APC 自动 CoW

每档单独起引擎、单独跑 benchmark、单独采集 mem_peak 和 kv_cache_hit_rate。
差值 = APC 的分支共享收益。
"""

from __future__ import annotations

import time

from agent_mem.bench.runner import Runner
from agent_mem.bench.tasks.tau_bench_adapter import TaskRunResult
from agent_mem.config import AppConfig


def _make_prefix_text(tokenizer, n_tokens: int, seed: int) -> str:
    """生成一段可复现的 token 序列作为 prompt 前缀。

    用 filler 文本 tokenize 后循环截取到目标长度，保证不同 seed 产生不同前缀。
    """
    fillers = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a subset of artificial intelligence. ",
        "The history of computing dates back to ancient times. ",
        "Data structures and algorithms form the foundation of computer science. ",
        "The Renaissance period marked a turning point in European history. ",
    ]
    base = fillers[seed % len(fillers)]
    tokens = tokenizer.encode(base * 100)
    # 循环扩展到需要长度
    while len(tokens) < n_tokens:
        tokens = tokens + tokens
    return tokenizer.decode(tokens[:n_tokens])


class BranchingKVShareRunner:
    """分支 KV 共享测量 Runner。

    ``shared_prefix`` 控制前缀是否共享：
    - ``False``：每个分支独立前缀 → APC 命中率低、mem_peak 高
    - ``True``：所有分支共享同一前缀 → APC CoW 复用、mem_peak 低
    """

    def __init__(
        self,
        *,
        engine_url: str,
        model: str,
        model_path: str,
        shared_prefix: bool = False,
        n_branches: int = 4,
        prefix_tokens: int = 3000,
        max_tokens: int = 256,
        api_key: str = "stub",
    ):
        self.engine_url = engine_url
        self.model = model
        self.shared_prefix = shared_prefix
        self.n_branches = n_branches
        self.prefix_tokens = prefix_tokens
        self.max_tokens = max_tokens
        self.api_key = api_key

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def name(self) -> str:
        return "branching-kv-share"

    def _send_one(self, prompt: str) -> tuple[float, str | None]:
        """发一次请求，返回 (延迟秒, 错误消息或 None)。"""
        import httpx

        t0 = time.monotonic()
        try:
            r = httpx.post(
                f"{self.engine_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=600,
            )
            r.raise_for_status()
            return time.monotonic() - t0, None
        except Exception as e:
            return time.monotonic() - t0, repr(e)

    def run_all(self, cfg: AppConfig) -> list[TaskRunResult]:
        results: list[TaskRunResult] = []

        if self.shared_prefix:
            # 对照 B：共享前缀（分支推理）
            prefix_text = _make_prefix_text(
                self.tokenizer, self.prefix_tokens, seed=42
            )
            for i in range(self.n_branches):
                prompt = f"{prefix_text}\n\nBranch {i}: continue the text in a different way."
                lat_s, err = self._send_one(prompt)
                results.append(TaskRunResult(
                    task_id=i,
                    reward=float(not err),
                    success=err is None,
                    latency_ms=lat_s * 1000,
                    n_steps=1,
                    error=err,
                ))
        else:
            # 对照 A：独立前缀（无共享）
            for i in range(self.n_branches):
                unique = _make_prefix_text(
                    self.tokenizer, self.prefix_tokens, seed=100 + i
                )
                prompt = f"{unique}\n\nContinue the text naturally."
                lat_s, err = self._send_one(prompt)
                results.append(TaskRunResult(
                    task_id=i,
                    reward=float(not err),
                    success=err is None,
                    latency_ms=lat_s * 1000,
                    n_steps=1,
                    error=err,
                ))

        return results
