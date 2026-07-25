"""离线 HotpotQA 环境 — 替代 Wikipedia API，用本地 distractor 数据集。
v2: TF-IDF 检索替代简单词匹配，提升搜索相关性。
"""
import re
import math
from collections import defaultdict
from typing import Any


class OfflineHotpotEnv:
    """离线 HotpotQA 环境，从 HF distractor 数据集加载上下文文档。

    模拟 WikiEnv 的接口：reset(idx) → 初始问题 + 上下文文档
    step(action) → 搜索/查找/回答

    v2 改进：TF-IDF 加权检索替代简单词匹配。
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
        # TF-IDF 索引（每个 reset 重建）
        self._idf = {}          # term → idf
        self._doc_vecs = []     # [(title, sent_idx, sent_text, tfidf_vec)] — flat list

    def reset(self, idx: int | None = None) -> str:
        """重置到第 idx 个问题，返回问题文本"""
        if idx is None:
            import random
            idx = random.randint(0, len(self.data) - 1)
        self._idx = idx
        ex = self.data[idx]
        self._question = ex["question"]
        self._answer = ex["answer"]
        ctx = ex["context"]
        n_docs = len(ctx["title"])
        self._context = [
            {"title": ctx["title"][i], "sentences": ctx["sentences"][i]}
            for i in range(n_docs)
        ]
        self._done = False
        self._steps = 0

        # 构建 TF-IDF 索引
        self._build_index()

        titles = [doc["title"] for doc in self._context]
        return (
            f"Question: {self._question}\n\n"
            f"Available documents: {', '.join(titles)}\n"
            f"Use Search[query] to search across all documents, "
            f"Lookup[keyword] to find specific sentences, "
            f"or Finish[answer] when ready."
        )

    # ── TF-IDF 索引 ──
    def _tokenize(self, text: str) -> list[str]:
        """简单分词：小写 + 去标点 + 按空白分"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [t for t in text.split() if len(t) > 1]

    def _build_index(self):
        """构建 TF-IDF 索引：计算 IDF，向量化所有句子"""
        # 收集所有句子
        all_sents = []
        self._doc_vecs = []
        for doc in self._context:
            for i, sent in enumerate(doc["sentences"]):
                tokens = self._tokenize(sent)
                all_sents.append(tokens)
                self._doc_vecs.append((doc["title"], i, sent, None))  # vec 稍后填充

        n_docs = len(all_sents)
        # 计算 DF（document frequency）
        df = defaultdict(int)
        for tokens in all_sents:
            for term in set(tokens):
                df[term] += 1

        # 计算 IDF: log(N / df)，加 1 平滑
        self._idf = {
            term: math.log((n_docs + 1) / (df[term] + 1)) + 1
            for term in df
        }

        # 向量化每个句子
        for idx, (title, si, sent, _) in enumerate(self._doc_vecs):
            tokens = all_sents[idx]
            vec = self._vectorize_tokens(tokens)
            norm = math.sqrt(sum(v * v for v in vec.values()))
            self._doc_vecs[idx] = (title, si, sent, vec, norm)

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        """TF-IDF 向量（稀疏 dict），TF 用 log(1+count)"""
        vec = {}
        tf_map = defaultdict(int)
        for t in tokens:
            tf_map[t] += 1
        for t, cnt in tf_map.items():
            if t in self._idf:
                vec[t] = (1 + math.log(cnt)) * self._idf[t]
        return vec

    def _vectorize_query(self, query: str) -> dict[str, float]:
        """查询向量（同上）"""
        tokens = self._tokenize(query)
        return self._vectorize_tokens(tokens)

    def _cosine_sim(self, q_vec: dict[str, float],
                    d_vec: dict[str, float], d_norm: float) -> float:
        """余弦相似度"""
        if d_norm == 0:
            return 0.0
        dot = sum(q_vec.get(t, 0) * w for t, w in d_vec.items())
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return 0.0
        return dot / (q_norm * d_norm)

    # ── 搜索 / 查找 ──
    def _search(self, query: str, top_k: int = 5) -> str:
        """TF-IDF + 余弦相似度检索"""
        q_vec = self._vectorize_query(query)
        scored = []
        for title, si, sent, d_vec, d_norm in self._doc_vecs:
            sim = self._cosine_sim(q_vec, d_vec, d_norm)
            if sim > 0:
                scored.append((sim, title, si, sent))
        scored.sort(key=lambda x: -x[0])

        if not scored:
            return f"No results found for '{query}'."

        lines = [f"Search results for '{query}':"]
        for sim, title, idx, sent in scored[:top_k]:
            lines.append(f"[{title}] S{idx+1}: {sent}")
        return "\n".join(lines)

    def _lookup(self, keyword: str, top_k: int = 5) -> str:
        """精确关键词匹配（保留原来用途：查找特定句子）"""
        kw_lower = keyword.lower()
        results = []
        for doc in self._context:
            for i, sent in enumerate(doc["sentences"]):
                if kw_lower in sent.lower():
                    results.append((doc["title"], i, sent))
        if results:
            lines = [f"Lookup results for '{keyword}':"]
            for title, idx, sent in results[:top_k]:
                lines.append(f"[{title}] S{idx+1}: {sent}")
            return "\n".join(lines)
        # fallback: 当 Lookup 找不到时，用 TF-IDF 搜索
        return self._search(keyword, top_k=top_k)

    # ── 动作执行 ──
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

    def _is_correct(self, answer: str) -> bool:
        """v3: 多级匹配 — 精确 → 子串 → token-F1 → 归一化子串"""
        gt = self._answer.lower().strip()
        pred = answer.lower().strip()
        # 1. 精确匹配
        if pred == gt:
            return True
        # 2. 子串匹配（双向）
        if gt in pred:
            return True
        if len(pred) > 5 and pred in gt:
            return True
        # 3. Token 级 F1 匹配（去停用词）
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "on",
                     "at", "to", "for", "and", "or", "by", "with", "from", "as", "has"}
        gt_tokens = set(self._tokenize(gt)) - stopwords
        pred_tokens = set(self._tokenize(pred)) - stopwords
        if gt_tokens and pred_tokens:
            overlap = gt_tokens & pred_tokens
            precision = len(overlap) / len(pred_tokens)
            recall = len(overlap) / len(gt_tokens)
            if precision > 0 and recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                if f1 >= 0.7:
                    return True
        # 4. 归一化后子串匹配（去标点、去空白）
        def normalize(s):
            return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s)).strip()
        gt_norm = normalize(gt)
        pred_norm = normalize(pred)
        if len(pred_norm) > 5 and pred_norm in gt_norm:
            return True
        if len(gt_norm) > 5 and gt_norm in pred_norm:
            return True
        return False

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
