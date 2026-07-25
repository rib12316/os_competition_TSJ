"""Apply Qwen2 C8 KV scale-loading patch at interpreter startup.

Imported automatically in every Python process whose sys.path includes this dir
(``agent_mem/kv/c8patch/`` — put it on PYTHONPATH via
:func:`agent_mem.server.vllm_server.engine_env` when ``engine.c8.patch_qwen2``).
Gated by ``QWEN2_C8_PATCH=1``.

Why this exists: vLLM V1 loads the model in a **spawned EngineCore subprocess**,
so an in-process monkeypatch in the api_server parent does NOT propagate to the
child where ``load_weights`` actually runs. ``sitecustomize`` runs at every
interpreter startup (parent + spawned child, since PYTHONPATH & env are
inherited), so the patch lands in the EngineCore too.

vllm-ascend's ``patch_gqa_c8.py`` only covers Qwen3/Glm4Moe/MiniMaxM2; this adds
Qwen2 (Qwen2.5) via the same ``_patched_causal_lm_load_weights`` interceptor.
``get_cache_scale`` is suffix-based (model-agnostic), so no other change needed.
"""
import os
import sys


def _apply():
    try:
        import vllm_ascend.patch.worker.patch_gqa_c8 as pg
        from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    except Exception as e:  # noqa: BLE001 — non-vllm context (pytest/calibrate/etc.)
        return
    fn = Qwen2ForCausalLM.load_weights
    if getattr(fn, "_qwen2_c8_patched", False):
        return
    _orig = fn

    def _patched(self, weights, _orig=_orig):
        return pg._patched_causal_lm_load_weights(self, weights, _orig)

    _patched._qwen2_c8_patched = True
    Qwen2ForCausalLM.load_weights = _patched
    print(
        "[qwen2_c8 sitecustomize] patched Qwen2ForCausalLM.load_weights "
        "for C8 KV scale loading",
        file=sys.stderr,
        flush=True,
    )


if os.environ.get("QWEN2_C8_PATCH") == "1":
    _apply()
