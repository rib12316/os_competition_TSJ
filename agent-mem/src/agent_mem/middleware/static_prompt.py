"""固定 system prompt 与工具描述的确定性精简。"""

from __future__ import annotations

import copy
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agent_mem.middleware.policy import PolicyArtifactError, load_policy_artifact

COMPACT_RETAIL_POLICY = """# Retail support policy
Scope: help the authenticated user with their profile, orders and related products; cancel or modify pending orders; return or exchange delivered orders; or change their default address.

Global rules:
- First authenticate by finding user_id from email or from name plus ZIP, even if a user_id was supplied. Serve only that user and reject requests about others.
- Before any database-changing action, state all details and obtain explicit yes/no confirmation.
- Use only user/tool facts and these rules. Do not invent procedures, recommendations or opinions.
- Make at most one tool call per turn and do not also reply to the user. Transfer to a human iff the request is outside available actions.
- Times are 24-hour EST. Each profile contains email, default address, user_id and payment methods; payment methods are gift card, PayPal or credit card. The store has 50 product types with variant items/options. Product IDs and item IDs are distinct. Order status is pending, processed, delivered or cancelled. Modify/exchange tools may be called once, so collect every item first.

Actions:
- Cancel: require pending status; confirm order_id and reason (only "no longer needed" or "ordered by mistake"). Set cancelled; refund gift card immediately, otherwise in 5-7 business days.
- Modify pending order: only shipping address, payment method or item options. A new payment method must differ from the original; a gift card must cover the total. The order remains pending. Refund the original method immediately for gift card, otherwise in 5-7 business days.
- Modify items: call once; same product type, different option only. Confirm all items. Require a payment method for price difference; gift card balance must suffice. Afterwards status is "pending (items modifed)" and no further modify/cancel is allowed.
- Return: require delivered status. Confirm order_id, all item IDs and refund method, which must be the original method or an existing gift card. Set "return requested" and email return instructions.
- Exchange: require delivered status. Confirm all item IDs. Exchange only for available items of the same product with a different option. Require payment method for price difference; gift card balance must suffice. Set "exchange requested", email return instructions, and do not create a new order."""

_CONFIRMATION_FRAGMENT = "explicit user confirmation (yes/no)"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def compact_system_messages(
    messages: list[dict], *, mode: str, conventions: list[str],
    policy_artifact_path: str = "", policy_artifact_strict: bool = False,
) -> tuple[list[dict], bool, dict[str, Any]]:
    """替换已识别的 retail policy，并把共享工具约定只注入一次。"""
    out = [copy.deepcopy(message) for message in messages]
    compacted = False
    policy_meta: dict[str, Any] = {
        "policy_mode": mode,
        "policy_artifact_status": "not_configured",
    }
    for message in out:
        if message.get("role") != "system":
            continue
        content = str(message.get("content") or "")
        if mode == "retail_compact" and content.lstrip().startswith("# Retail agent policy"):
            message["content"] = COMPACT_RETAIL_POLICY
            compacted = True
            policy_meta["policy_artifact_status"] = "builtin_retail"
        elif mode == "compiled":
            try:
                artifact = load_policy_artifact(policy_artifact_path, content)
            except PolicyArtifactError as exc:
                policy_meta["policy_artifact_status"] = "fallback_original"
                policy_meta["policy_artifact_error"] = str(exc)
                if policy_artifact_strict:
                    raise
            else:
                message["content"] = artifact["compact_policy"]
                compacted = True
                policy_meta["policy_artifact_status"] = "applied"
                policy_meta["policy_id"] = artifact["policy_id"]
                policy_meta["policy_artifact_path"] = str(Path(policy_artifact_path))
        if conventions:
            suffix = "# Shared tool conventions\n" + "\n".join(conventions)
            message["content"] = f"{message.get('content') or ''}\n\n{suffix}"
        return out, compacted, policy_meta
    if conventions:
        out.insert(0, {
            "role": "system",
            "content": "# Shared tool conventions\n" + "\n".join(conventions),
        })
    return out, compacted, policy_meta


def _description_slots(tools: list[dict]) -> list[tuple[dict, str]]:
    slots: list[tuple[dict, str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            description = value.get("description")
            if isinstance(description, str) and description.strip():
                slots.append((value, description.strip()))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(tools)
    return slots


def optimize_tool_descriptions(
    tools: list[dict], *, system_covers_confirmation: bool
) -> tuple[list[dict], list[str], dict[str, int]]:
    """共享完全重复描述，并移除 system 已覆盖的 confirmation 重复句。"""
    out = copy.deepcopy(tools)
    slots = _description_slots(out)

    confirmation_removed = 0
    if system_covers_confirmation:
        for owner, description in slots:
            sentences = _SENTENCE_SPLIT.split(description)
            kept = [sentence for sentence in sentences
                    if _CONFIRMATION_FRAGMENT not in sentence.lower()]
            if len(kept) != len(sentences):
                confirmation_removed += 1
                owner["description"] = " ".join(kept).strip()

    slots = _description_slots(out)
    counts = Counter(description for _, description in slots)
    candidates = []
    for description, count in counts.items():
        if count < 2:
            continue
        reference_cost = count * len("See C00.") + len(description) + len("C00: ")
        if count * len(description) - reference_cost >= 20:
            candidates.append((description, count))
    candidates.sort(key=lambda item: (-item[1], item[0]))

    convention_ids = {
        description: f"C{idx}"
        for idx, (description, _) in enumerate(candidates, start=1)
    }
    replaced = 0
    for owner, description in slots:
        convention_id = convention_ids.get(description)
        if convention_id:
            owner["description"] = f"See {convention_id}."
            replaced += 1
    conventions = [
        f"{convention_ids[description]}: {description}"
        for description, _ in candidates
    ]
    return out, conventions, {
        "tool_descriptions_replaced": replaced,
        "tool_conventions": len(conventions),
        "confirmation_sentences_removed": confirmation_removed,
    }
