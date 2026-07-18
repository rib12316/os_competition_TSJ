"""kv: KV Cache 量化与分层存储（M1 / M4 / M9）。

- FP8 KV 量化（M1）
- LMCache GPU↔CPU↔SSD 三级（M4）
- 分布式 KV 池 Mooncake（M9，仅多机）
"""
