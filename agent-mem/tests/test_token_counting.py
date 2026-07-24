"""Shared token-counting helpers."""

from agent_mem.token_counting import count_text_chunks, estimate_text_tokens


class _CharacterTokenizer:
    @staticmethod
    def encode(text, add_special_tokens=False):
        return list(text)


def test_text_chunks_include_serialization_boundaries():
    assert count_text_chunks(_CharacterTokenizer(), ["ab", "cd"]) == 6


def test_unicode_fallback_counts_cjk_individually():
    assert estimate_text_tokens(["你好世界"]) == 4
    assert estimate_text_tokens(["abcd"]) == 1
