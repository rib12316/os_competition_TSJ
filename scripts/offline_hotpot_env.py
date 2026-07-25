"""离线 HotpotQA 环境 — 替代 Wikipedia API，用本地 distractor 数据集"""
import re
import random
from typing import Any


class OfflineHotpotEnv:
    """离线 HotpotQA 环境，从 HF distractor 数据集加载上下文文档。

    模拟 WikiEnv 的接口：reset(idx) → 初始问题 + 上下文文档
    step(action) → 搜索/查找/回答
    """

    def __init__(self, split: str = "train", max_examples: int = 100):
        from datasets import load_dataset
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        self.data = ds.select(range(min(len(ds), max_examples)))
        self._idx = 0
        self._context = None  # [{title, sentences: [str, ...]}, ...]
        self._question = ""
        self._answer = ""
        self._done = False
        self._steps = 0

    def reset(self, idx: int | None = None) -> str:
        """重置到第 idx 个问题，返回问题文本"""
        if idx is None:
            idx = random.randint(0, len(self.data) - 1)
        self._idx = idx
        ex = self.data[idx]
        self._question = ex["question"]
        self._answer = ex["answer"]
        ctx = ex["context"]
        # HF distractor: title 是 list[str]，sentences 是 list[list[str]]
        n_docs = len(ctx["title"])
        self._context = [
            {"title": ctx["title"][i], "sentences": ctx["sentences"][i]}
            for i in range(n_docs)
        ]
        self._done = False
        self._steps = 0
        titles = [doc["title"] for doc in self._context]
        return (
            f"Question: {self._question}\n\n"
            f"Available documents: {', '.join(titles)}\n"
            f"Use Search[query] to search across all documents, "
            f"Lookup[keyword] to find specific sentences, "
            f"or Finish[answer] when ready."
        )

    def step(self, action: str) -> tuple[str, int, bool, dict]:
        """执行 action，返回 (observation, reward, done, info)"""
        self._steps += 1
        action = action.strip()

        # Finish[answer]
        if action.startswith("Finish["):
            ans = action[len("Finish["):-1] if action.endswith("]") else action[7:]
            self._done = True
            r = 1 if self._is_correct(ans) else 0
            info = {"em": r}
            return (f"Answer recorded: {ans}\nCorrect answer: {self._answer}", r, True, info)

        # Search[query]
        if action.startswith("Search["):
            query = action[len("Search["):-1] if action.endswith("]") else action[7:]
            results = self._search(query)
            return (results, 0, False, {})

        # Lookup[keyword]
        if action.startswith("Lookup["):
            kw = action[len("Lookup["):-1] if action.endswith("]") else action[8:]
            results = self._lookup(kw)
            return (results, 0, False, {})

        return ("Invalid action. Use Search[query], Lookup[keyword], or Finish[answer].", 0, False, {})

    def _search(self, query: str, top_k: int = 3) -> str:
        """在所有文档中搜索匹配 query 的句子"""
        q_words = set(query.lower().split())
        scored = []
        for doc in self._context:
            for i, sent in enumerate(doc["sentences"]):
                sent_lower = sent.lower()
                score = sum(1 for w in q_words if w in sent_lower)
                if score > 0:
                    scored.append((score, doc["title"], i, sent))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return f"No results found for '{query}'."

        lines = [f"Search results for '{query}':"]
        for _, title, idx, sent in scored[:top_k]:
            lines.append(f"[{title}] S{idx+1}: {sent}")
        return "\n".join(lines)

    def _lookup(self, keyword: str, top_k: int = 5) -> str:
        """在所有文档中查找包含 keyword 的句子"""
        kw_lower = keyword.lower()
        results = []
        for doc in self._context:
            for i, sent in enumerate(doc["sentences"]):
                if kw_lower in sent.lower():
                    results.append((doc["title"], i, sent))
        if not results:
            return f"No sentences found containing '{keyword}'."

        lines = [f"Lookup results for '{keyword}':"]
        for title, idx, sent in results[:top_k]:
            lines.append(f"[{title}] S{idx+1}: {sent}")
        return "\n".join(lines)

    def _is_correct(self, answer: str) -> bool:
        """简单答案匹配（与 HotpotQA 的 EM 一致）"""
        return answer.lower().strip() == self._answer.lower().strip()

    @property
    def tools_info(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "Search",
                    "description": "Search across all documents for a query",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Lookup",
                    "description": "Find specific sentences containing a keyword",
                    "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Finish",
                    "description": "Submit your final answer",
                    "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                },
            },
        ]
