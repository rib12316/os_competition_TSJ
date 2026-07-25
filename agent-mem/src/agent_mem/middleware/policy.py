"""Offline-compiled system policy artifacts.

The compiler may use an LLM, but runtime only loads a validated, hashed artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class PolicyArtifactError(ValueError):
    """Raised when a compiled policy cannot be proven compatible with its source."""


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_MODAL_PHRASES = (
    "must not", "must", "only", "never", "unless", "cannot", "do not", "at most", "before",
)
_SAFETY_TERMS = (
    "pending", "processed", "delivered", "cancelled", "confirmation",
    "authenticated", "refund", "payment", "permission", "private",
)


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def extract_policy_units(source: str) -> list[str]:
    """Split policy into stable source units for compiler citations.

    Markdown/list lines stay intact. Plain paragraphs are split at sentence boundaries,
    which gives the offline compiler a small unit to rewrite without trusting it to
    decide what can be dropped.
    """
    units: list[str] = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#") or _LIST_PREFIX.match(line):
            units.append(line)
            continue
        parts = [part.strip() for part in _SENTENCE_SPLIT.split(line) if part.strip()]
        units.extend(parts or [line])
    return units


def deterministic_literals(source: str) -> list[str]:
    """Return literals whose spelling is cheap and important to preserve."""
    found: list[str] = []
    patterns = [
        r"\"([^\"]+)\"",
        r"(?<![A-Za-z])'([^'\n]+)'(?![A-Za-z])",
        r"#[A-Za-z0-9_-]+",
        r"\b\d+(?:[.-]\d+)*\b",
        r"\b[A-Z][A-Z0-9_-]{2,}\b",
    ]
    for pattern in patterns:
        found.extend(re.findall(pattern, source))
    lowered = source.lower()
    found.extend(term for term in _SAFETY_TERMS if re.search(rf"\b{re.escape(term)}\b", lowered))
    unique: list[str] = []
    for value in found:
        if value and value not in unique:
            unique.append(value)
    return unique


def render_policy_units(units: list[dict[str, Any]]) -> str:
    return "\n".join(str(unit.get("compact") or "").strip() for unit in units).strip()


def validate_policy_artifact(
    source: str,
    artifact: dict[str, Any],
    *,
    require_shorter: bool = True,
) -> list[str]:
    """Return deterministic validation errors; an empty list means valid."""
    errors: list[str] = []
    if artifact.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not str(artifact.get("policy_id") or "").strip():
        errors.append("policy_id is required")
    if artifact.get("source_sha256") != source_sha256(source):
        errors.append("source_sha256 does not match the supplied system policy")

    expected_units = extract_policy_units(source)
    units = artifact.get("source_units")
    if not isinstance(units, list) or not units:
        errors.append("source_units must be a non-empty list")
        units = []
    actual_sources = [str(unit.get("source") or "") for unit in units if isinstance(unit, dict)]
    if actual_sources != expected_units:
        errors.append("source_units must exactly cover the source policy in order")

    ids: set[str] = set()
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            errors.append(f"source_units[{index}] must be an object")
            continue
        unit_id = str(unit.get("id") or "")
        compact = str(unit.get("compact") or "").strip()
        if not unit_id or unit_id in ids:
            errors.append(f"source_units[{index}] has a missing or duplicate id")
        ids.add(unit_id)
        if not compact:
            errors.append(f"source_units[{index}] has an empty compact rule")
            continue
        source_unit = str(unit.get("source") or "")
        for literal in deterministic_literals(source_unit):
            if literal not in compact:
                errors.append(
                    f"source_units[{index}] is missing protected literal {literal!r}"
                )
        compact_lower = compact.lower()
        source_lower = source_unit.lower()
        for phrase in _MODAL_PHRASES:
            if phrase in source_lower and phrase not in compact_lower:
                errors.append(f"source_units[{index}] is missing modal phrase {phrase!r}")

    rendered = render_policy_units(units)
    compact_policy = str(artifact.get("compact_policy") or "").strip()
    if compact_policy != rendered:
        errors.append("compact_policy must equal the deterministic rendering of source_units")
    if not compact_policy:
        errors.append("compact_policy is empty")
    if require_shorter and len(compact_policy) >= len(source):
        errors.append("compact_policy is not shorter than the source")

    literals = deterministic_literals(source)
    declared = artifact.get("protected_literals") or []
    if not isinstance(declared, list):
        errors.append("protected_literals must be a list")
        declared = []
    for literal in literals + [str(value) for value in declared]:
        if literal not in source:
            errors.append(f"protected literal is not present in source: {literal!r}")
        elif literal not in compact_policy:
            errors.append(f"protected literal is missing from compact policy: {literal!r}")

    return errors


def load_policy_artifact(
    path: str | Path,
    source: str | None = None,
    *,
    require_shorter: bool = True,
) -> dict[str, Any]:
    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolicyArtifactError(f"cannot load policy artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict):
        raise PolicyArtifactError("policy artifact root must be an object")
    if source is not None:
        errors = validate_policy_artifact(source, artifact, require_shorter=require_shorter)
        if errors:
            raise PolicyArtifactError("; ".join(errors))
    return artifact


def make_policy_artifact(
    *,
    policy_id: str,
    source: str,
    compact_units: list[dict[str, Any]],
    protected_literals: list[str] | None = None,
    compiler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the artifact shape used by both the compiler and runtime."""
    units = [
        {
            "id": str(unit["id"]),
            "source": str(unit["source"]),
            "compact": str(unit["compact"]).strip(),
            "kind": str(unit.get("kind") or "rule"),
        }
        for unit in compact_units
    ]
    return {
        "schema_version": 1,
        "policy_id": policy_id,
        "source_sha256": source_sha256(source),
        "source_units": units,
        "compact_policy": render_policy_units(units),
        "protected_literals": list(protected_literals or []),
        "compiler": dict(compiler or {}),
    }
