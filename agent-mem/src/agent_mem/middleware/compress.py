"""缝D · F2 Prompt / 上下文压缩中间件（基于 LLMLingua / LongLLMLingua）。

挂在 :meth:`transform_messages`：发引擎**前**把 agent 的**冷历史**压短，热尾与
system 原样保留。命中赛题"降显存/降延迟"——更短的 prompt → 更少的 prefill 与 KV。

设计要点（实现者必读）
^^^^^^^^^^^^^^^^^^^^^^

- **方案 A：单一方法 + 触发门**。``method`` 选一个压缩器（默认
  ``longllmlingua``，最适合长会话 agent 的 question-aware 压缩）；``trigger_tokens``
  是**压/不压**的开关——冷历史太短就**直接放行**（短上下文没有 lost-in-the-middle
  问题，压了反而白费延迟）。不是"两种方法分场景"。
- **正典不动**：只变换发给引擎的副本（详见 ``base.py``），压缩**无序可恢复**。
- **tool_call 配对安全**：绝不留下孤立的 ``role=tool`` 消息。做法——把整段冷历史
  压成**一条**文本消息（冷的 assistant ``tool_calls`` 与冷的 tool 结果**一起**进
  文本，互不残留引用），热尾原样保留且边界 snap 到完整 ``tool_call→tool`` 组。
- **压缩器进程级单例**：``llmlingua`` 的 ``PromptCompressor`` 要加载小模型（几百
  MB~2GB），必须**只加载一次**。缓存在 ``self._compressor``（实例由
  ``build_middlewares`` 构造一次、跨 session/step 复用），懒加载。**别**放进
  ``ctx.scratch``（那是每 session 一份，会反复加载）。
- **惰性 import**：``llmlingua`` 是重依赖（torch/transformers），import 写在
  ``_get_compressor`` 内部，未安装时不影响包导入与其它 F / 单测。

红线：压过头伤成功率，必须配 ablation，``task_success_rate`` 下降 ≤ 2pp。
"""

from __future__ import annotations

from typing import Any

from agent_mem.middleware.base import BaseMiddleware, MiddlewareContext

# 支持的压缩方法（同一套配置切，便于 ablation）
_METHODS: set[str] = {"llmlingua", "longllmlingua", "llmlingua2"}

# 粗略 chars→tokens 估计（英文 ~4 char/token；仅用于触发门判定，无需精确）
_CHARS_PER_TOKEN = 4


def _msg_to_text(m: dict) -> str:
    """把一条 message 压成给压缩器的文本（丢结构、留语义）。

    - ``tool_calls`` 是结构化的，不进压缩文本；但保留"调用了哪些工具"的梗概，
      免得冷历史里 assistant 轮次完全空白。
    - 空 content 返回空串（调用方会过滤掉）。
    """
    content = m.get("content") or ""
    if m.get("role") == "assistant" and m.get("tool_calls"):
        names = [
            tc.get("function", {}).get("name")
            for tc in m["tool_calls"]
            if isinstance(tc, dict)
        ]
        calls = ", ".join(n for n in names if n)
        if calls:
            content = (content + f" [called: {calls}]").strip()
    return content


class CompressMiddleware(BaseMiddleware):
    """F2 Prompt 压缩中间件。

    配置（yaml ``middleware.options.compress``）::

        compress:
          method: longllmlingua        # llmlingua | longllmlingua | llmlingua2
          rate: 0.4                    # 保留比例（0.4=保留 40%）
          trigger_tokens: 4000         # 冷历史 < 此值不压（压/不压门）
          keep_hot: 6                  # 热尾保留条数（snap 到完整 tool 组）
          device: cpu                  # 压缩器小模型放 CPU，NPU 让给主 LLM
          model_name: null             # null=PromptCompressor 默认模型
          # —— LongLLMLingua 专属（method=longllmlingua 才用）——
          condition_in_question: after_condition
          dynamic_context_compression_ratio: 0.3
          condition_compare: true
          reorder_context: none        # agent 轨迹默认保时序；ablation 可试 sort
    """

    name = "compress"

    def __init__(
        self,
        method: str = "longllmlingua",
        rate: float = 0.5,
        trigger_tokens: int = 4000,
        keep_hot: int = 6,
        device: str = "cpu",
        model_name: str | None = None,
        condition_in_question: str = "after_condition",
        dynamic_context_compression_ratio: float = 0.3,
        condition_compare: bool = True,
        reorder_context: str = "none",
        force_tokens: list[str] | None = None,
        history_role: str = "system",
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method 必须是 {sorted(_METHODS)}，得到 {method!r}")
        if not (0.0 < rate <= 1.0):
            raise ValueError(f"rate 必须在 (0, 1]，得到 {rate}")
        if keep_hot < 1:
            raise ValueError("keep_hot 必须 >= 1（至少保留 1 条热尾）")
        if trigger_tokens < 0:
            raise ValueError("trigger_tokens 必须 >= 0")

        self.method = method
        self.rate = rate
        self.trigger_tokens = trigger_tokens
        self.keep_hot = keep_hot
        self.device = device
        self.model_name = model_name
        self.condition_in_question = condition_in_question
        self.dynamic_context_compression_ratio = dynamic_context_compression_ratio
        self.condition_compare = condition_compare
        self.reorder_context = reorder_context
        self.force_tokens = force_tokens if force_tokens is not None else ["\n", "?", "."]
        self.history_role = history_role
        self._compressor: Any = None  # 懒加载，进程级单例

    # ---- 压缩器加载（惰性、缓存）----

    def _get_compressor(self) -> Any:
        """懒加载 ``llmlingua.PromptCompressor``，缓存在实例上（只加载一次）。"""
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as e:  # pragma: no cover - 依赖缺失友好提示
                raise ImportError(
                    "F2 压缩需要可选依赖 llmlingua："
                    "pip install 'agent-mem[compress]' 或 pip install llmlingua"
                ) from e
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                use_llmlingua2=(self.method == "llmlingua2"),
                device_map=self.device,
            )
        return self._compressor

    # ---- 主钩子 ----

    def transform_messages(
        self, messages: list[dict], ctx: MiddlewareContext
    ) -> list[dict]:
        if not messages:
            return list(messages)

        sys_msgs = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        # 太短：热尾都凑不齐，全留
        if len(rest) <= self.keep_hot:
            return list(messages)

        # 切冷/热；热尾边界 snap 到完整 tool_call→tool 组（避免孤立 tool）
        split = len(rest) - self.keep_hot
        while split > 0 and rest[split].get("role") == "tool":
            split -= 1  # 把 tool 的 caller(assistant) 一起拉进热尾
        cold, hot = rest[:split], rest[split:]

        # 冷历史 → 文本块
        chunks = [t for t in (_msg_to_text(m) for m in cold) if t]
        if not chunks:
            return list(messages)

        # 触发门：冷历史短就不压（省延迟）
        if self._est_tokens(chunks) < self.trigger_tokens:
            return list(messages)

        # question = 最近一条带内容的 user 消息（LongLingua 的相关性锚点）
        question = next(
            (
                m.get("content") or ""
                for m in reversed(rest)
                if m.get("role") == "user" and m.get("content")
            ),
            "",
        )

        compressed = self._compress_cold(chunks, question)

        out: list[dict] = list(sys_msgs)
        if compressed:
            out.append(
                {
                    "role": self.history_role,
                    "content": f"[compressed history]\n{compressed}",
                }
            )
        out.extend(hot)
        return out

    # ---- 压缩分发 ----

    def _compress_cold(self, chunks: list[str], question: str) -> str:
        c = self._get_compressor()
        if self.method == "longllmlingua":
            # LongLLMLingua：按块传入，question-aware 打分 + 分段动态率
            res = c.compress_prompt(
                chunks,
                question=question,
                rate=self.rate,
                rank_method="longllmlingua",
                condition_in_question=self.condition_in_question,
                dynamic_context_compression_ratio=self.dynamic_context_compression_ratio,
                condition_compare=self.condition_compare,
                reorder_context=self.reorder_context,
            )
        else:
            # llmlingua / llmlingua2：拼成一段文本，按 rate 压
            prompt = "\n\n".join(chunks)
            kw: dict[str, Any] = dict(prompt=prompt, rate=self.rate, force_tokens=self.force_tokens)
            if question:
                kw["question"] = question
            res = c.compress_prompt(**kw)
        return res.get("compressed_prompt", "")

    @staticmethod
    def _est_tokens(chunks: list[str]) -> int:
        """粗估 token 数（仅用于触发门判定）。"""
        return sum(len(c) for c in chunks) // _CHARS_PER_TOKEN
