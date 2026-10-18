"""Shadow-only memory coverage audit for daily continuity."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .resolver import resolve_continuity_items


_ELIGIBLE_KINDS = {
    "event",
    "relation_delta",
    "user_fact",
    "correction",
    "commitment",
    # Compatibility with pre-v2 adapters.
    "fact",
    "state_delta",
}
_OPEN_STATUSES = {"", "active", "open", "pending", "unresolved"}


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                raw = method()
            except TypeError:
                continue
            if isinstance(raw, Mapping):
                return dict(raw)
    return {}


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value or "").lower())


def _tokens(value: Any) -> set[str]:
    text = _normalize(value)
    if not text:
        return set()
    out = {text}
    for size in (2, 3):
        for index in range(max(0, len(text) - size + 1)):
            out.add(text[index:index + size])
    return out


def _memory_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        raw_payload = row.get("payload_json")
        if isinstance(raw_payload, str) and raw_payload.strip():
            try:
                parsed = json.loads(raw_payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = {}
            payload = parsed if isinstance(parsed, Mapping) else {}
    return payload or {}


def _memory_key(row: Mapping[str, Any]) -> str:
    direct = row.get("canonical_key") or row.get("claim_key")
    if direct:
        return _normalize(direct)
    return _normalize(_memory_payload(row).get("canonical_key"))


def _memory_actor(row: Mapping[str, Any]) -> str:
    return str(row.get("actor") or _memory_payload(row).get("actor") or "").strip().lower()


def _coverage_match(item: Mapping[str, Any], memory_rows: list[dict[str, Any]]) -> str:
    item_key = _normalize(item.get("canonical_key"))
    item_actor = str(item.get("actor", "") or "").strip().lower()
    item_text = _normalize(item.get("text"))
    item_tokens = _tokens(item_text)
    for row in memory_rows:
        memory_id = str(row.get("item_id") or row.get("memory_id") or "")
        memory_key = _memory_key(row)
        memory_actor = _memory_actor(row)
        actors_match = bool(
            item_actor
            and memory_actor
            and item_actor == memory_actor
        )
        if (
            item_key
            and memory_key
            and item_key == memory_key
            and actors_match
        ):
            return memory_id or "canonical_key_match"
        if not actors_match:
            continue
        memory_text = _normalize(row.get("text") or row.get("summary"))
        if not item_text or not memory_text:
            continue
        if item_text in memory_text or memory_text in item_text:
            shorter = min(len(item_text), len(memory_text))
            if shorter >= 8:
                return memory_id or "text_containment"
        memory_tokens = _tokens(memory_text)
        union = item_tokens | memory_tokens
        if union and len(item_tokens & memory_tokens) / len(union) >= 0.72:
            return memory_id or "semantic_token_match"
    return ""


def build_shadow_memory_write_plan(
    resolution_or_items: Any,
    memory_rows: Any,
    *,
    character: str = "",
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Audit durable coverage and return candidates without applying writes."""
    resolved = _mapping(resolution_or_items)
    if isinstance(resolved.get("items"), list):
        items = [dict(row) for row in resolved["items"] if isinstance(row, Mapping)]
    else:
        items = resolve_continuity_items(resolution_or_items)["items"]
    existing = [dict(row) for row in memory_rows if isinstance(row, Mapping)] if isinstance(memory_rows, (list, tuple)) else []
    records: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    covered: list[dict[str, str]] = []

    for item in items:
        item_id = str(item.get("item_id", "") or "")
        kind = str(item.get("kind", "") or "").lower()
        status = str(item.get("status", "") or "active").lower()
        confidence = float(item.get("confidence", 0.0) or 0.0)
        if kind == "open_thread":
            suppressed.append({"item_id": item_id, "reason": "transient_open_thread"})
            continue
        if kind not in _ELIGIBLE_KINDS:
            suppressed.append({"item_id": item_id, "reason": "unsupported_kind"})
            continue
        if status not in _OPEN_STATUSES:
            suppressed.append({"item_id": item_id, "reason": f"closed_status:{status}"})
            continue
        if confidence < float(min_confidence):
            suppressed.append({"item_id": item_id, "reason": "low_confidence"})
            continue
        coverage_id = _coverage_match(item, existing)
        if coverage_id:
            covered.append({"item_id": item_id, "memory_id": coverage_id})
            suppressed.append({"item_id": item_id, "reason": "covered_by_existing_memory"})
            continue
        requested_layer = str(item.get("memory_layer", "") or "").lower()
        target_layer = requested_layer if requested_layer in {"episodic", "semantic"} else "episodic"
        records.append({
            "memory_type": "daily_continuity_candidate",
            "character": str(character or item.get("source_character", "") or "")[:40],
            "layer": target_layer,
            "memory_role": "event" if target_layer == "episodic" else "knowledge",
            "engram_type": "continuity_claim",
            "item_type": kind,
            "text": str(item.get("text", "") or "")[:320],
            "payload": {
                "continuity_item_id": item_id,
                "canonical_key": str(item.get("canonical_key", "") or "")[:120],
                "actor": str(item.get("actor", "unknown") or "unknown")[:20],
                "due_at_ms": int(item.get("due_at_ms", 0) or 0),
                "due_precision": str(item.get("due_precision", "unknown") or "unknown")[:20],
                "source_ref_ids": list(item.get("source_ref_ids", []) or [])[:12],
                "coverage_audit": "uncovered",
                "shadow_only": True,
            },
            "salience": float(item.get("salience", 0.5) or 0.5),
            "confidence": confidence,
            "source": "daily_continuity_shadow_coverage",
            "evidence_id": item_id,
        })
    return {
        "contract_version": "daily_continuity_memory_write_plan_v2",
        "shadow_only": True,
        "apply": False,
        "records": records,
        "suppressed": suppressed,
        "coverage_audit": {
            "existing_memory_count": len(existing),
            "covered": covered,
            "covered_count": len(covered),
            "candidate_count": len(records),
            "durable_writes_performed": 0,
        },
    }


def audit_memory_coverage(
    resolution_or_items: Any,
    memory_rows: Any,
    *,
    character: str = "",
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    """Named audit entry point; equivalent to the shadow plan builder."""
    return build_shadow_memory_write_plan(
        resolution_or_items,
        memory_rows,
        character=character,
        min_confidence=min_confidence,
    )


__all__ = ["audit_memory_coverage", "build_shadow_memory_write_plan"]
