#!/usr/bin/env python3
"""Compile a generic system policy into a validated runtime artifact.

The model is used only offline. Runtime loading and validation live in
``agent_mem.middleware.policy`` and never call an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent_mem.middleware.policy import (
    deterministic_literals,
    extract_policy_units,
    make_policy_artifact,
    validate_policy_artifact,
)

DEFAULT_API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("compiler response must be a JSON object")
    return value


def _prompt(units: list[str], source_chars: int) -> str:
    numbered = [{"id": f"u{index:04d}", "source": text}
                for index, text in enumerate(units, start=1)]
    return (
        "You are a safety-critical policy compiler. Rewrite the supplied system policy into "
        "short, explicit rules for an LLM agent. Return JSON only with this shape: "
        '{"units":[{"id":"u0001","compact":"...","kind":"rule"}],'
        '"protected_literals":["..."]}. '
        "You MUST return exactly one unit for every input id, in the same order. "
        "Do not drop a unit, merge units, invent facts, or change scope. Preserve MUST, MUST NOT, "
        "ONLY, NEVER and UNLESS semantics; preserve numbers, quoted strings, enum values, IDs, "
        "statuses, dates and time windows exactly. Keep each compact rule self-contained. "
        "For headings use a short heading. Each compact value must be a concise rule fragment, "
        "not an explanation and not a copy of its source. The joined compact values MUST use at "
        f"most {max(1, int(source_chars * 0.70))} characters (70% of the source). "
        "protected_literals must contain safety-critical words or phrases that must remain "
        "verbatim and must be substrings of the source.\n\n"
        f"INPUT_UNITS={json.dumps(numbered, ensure_ascii=False)}"
    )


def compile_with_openai(
    *,
    source: str,
    policy_id: str,
    model: str,
    api_base: str,
    api_key: str,
) -> dict[str, Any]:
    units = extract_policy_units(source)
    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Output valid JSON only. Never omit a supplied policy unit.",
            },
            {"role": "user", "content": _prompt(units, len(source))},
        ],
    )
    content = response.choices[0].message.content or ""
    generated = _parse_json(content)
    generated_units = generated.get("units")
    if not isinstance(generated_units, list):
        raise ValueError("compiler response has no units list")
    by_id = {
        str(item.get("id")): item
        for item in generated_units
        if isinstance(item, dict)
    }
    compact_units: list[dict[str, Any]] = []
    expected_ids = [f"u{index:04d}" for index in range(1, len(units) + 1)]
    for unit_id, source_unit in zip(expected_ids, units):
        item = by_id.get(unit_id)
        if item is None:
            raise ValueError(f"compiler omitted policy unit {unit_id}")
        compact_units.append({
            "id": unit_id,
            "source": source_unit,
            "compact": str(item.get("compact") or "").strip(),
            "kind": str(item.get("kind") or "rule"),
        })
    declared = generated.get("protected_literals") or []
    protected = deterministic_literals(source)
    for value in declared:
        value = str(value)
        if value in source and value not in protected:
            protected.append(value)
    artifact = make_policy_artifact(
        policy_id=policy_id,
        source=source,
        compact_units=compact_units,
        protected_literals=protected,
        compiler={"provider": "openai-compatible", "model": model, "api_base": api_base},
    )
    errors = validate_policy_artifact(source, artifact)
    if errors:
        raise ValueError("compiled policy failed validation: " + "; ".join(errors))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default="MIMO_KEY")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    source = args.input.read_text(encoding="utf-8")
    artifact = compile_with_openai(
        source=source,
        policy_id=args.policy_id,
        model=args.model,
        api_base=args.api_base,
        api_key=api_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "policy_id": artifact["policy_id"],
        "source_sha256": artifact["source_sha256"],
        "source_chars": len(source),
        "compact_chars": len(artifact["compact_policy"]),
        "units": len(artifact["source_units"]),
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
