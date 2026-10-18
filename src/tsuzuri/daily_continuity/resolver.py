"""Pure lifecycle resolution for Daily Continuity Capsule items."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import hashlib
import re
from typing import Any, Mapping, Protocol
from tsuzuri.persona_registry import agent_names, text_variants, user_name


_OPEN_STATUSES = {"", "active", "open", "pending", "unresolved"}
_CLOSED_STATUSES = {"cancelled", "fulfilled", "resolved", "superseded", "archived"}
_LIFECYCLE_KINDS = {"correction", "commitment", "open_thread"}


class MappingLike(Protocol):
    def to_dict(self) -> Mapping[str, Any]: ...


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        raw = asdict(value)
        return dict(raw) if isinstance(raw, Mapping) else {}
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


def _rows(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _text(value: Any, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    raw = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[str] = []
    for item in raw:
        text = _text(item, 160)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _timestamp_ms(row: Mapping[str, Any]) -> int:
    for key in (
        "source_at_ms",
        "created_at_ms",
        "updated_at_ms",
        "timestamp_ms",
        "timestamp",
    ):
        try:
            value = float(row.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            if key == "timestamp" and value < 1e11:
                value *= 1000
            return int(value)
    for key in ("created_at", "updated_at", "as_of_world"):
        raw = _text(row.get(key, ""), 80)
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        return int(parsed.timestamp() * 1000)
    return 0


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff:_-]+", "", str(value or "").lower())[:120]


def _fallback_key(kind: str, text: str) -> str:
    normalized = _normalized_key(text)
    digest = hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()[:14]
    return f"{kind}:{digest}"


def _actor(value: Any) -> str:
    actor = _text(value, 20).lower()
    return actor if actor in {"user", "character", "shared", "other"} else "unknown"


def _actors_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_actor = _actor(left.get("actor"))
    right_actor = _actor(right.get("actor"))
    return left_actor != "unknown" and left_actor == right_actor


def _same_explicit_claim(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("canonical_key_explicit")
        and right.get("canonical_key_explicit")
        and left.get("canonical_key") == right.get("canonical_key")
        and _actors_compatible(left, right)
    )


def _item_id(row: Mapping[str, Any], *, kind: str, text: str, suffix: str) -> str:
    explicit = _text(row.get("item_id") or row.get("continuity_id") or row.get("fact_id"), 120)
    if explicit:
        return explicit
    raw = f"{kind}|{suffix}|{text}".encode("utf-8", errors="ignore")
    return "dc-" + hashlib.sha1(raw).hexdigest()[:16]


def _adapt_item(value: Any, *, kind: str = "", suffix: str = "") -> dict[str, Any] | None:
    row = _mapping(value)
    if not row and isinstance(value, str):
        row = {"text": value}
    item_kind = _text(row.get("kind") or kind or "fact", 40).lower()
    text = _text(row.get("text") or row.get("fact") or row.get("summary"), 360)
    if not text:
        return None
    explicit_canonical_key = _normalized_key(row.get("canonical_key") or row.get("claim_key"))
    canonical_key = explicit_canonical_key
    canonical_key = canonical_key or _fallback_key(item_kind, text)
    status = _text(row.get("status") or "active", 30).lower()
    if status not in _OPEN_STATUSES | _CLOSED_STATUSES:
        status = "active"
    try:
        confidence = max(0.0, min(1.0, float(row.get("confidence", 0.6) or 0.6)))
    except (TypeError, ValueError):
        confidence = 0.6
    try:
        salience = max(0.0, min(1.0, float(row.get("salience", 0.5) or 0.5)))
    except (TypeError, ValueError):
        salience = 0.5
    return {
        "item_id": _item_id(row, kind=item_kind, text=text, suffix=suffix),
        "canonical_key": canonical_key,
        "canonical_key_explicit": bool(explicit_canonical_key),
        "kind": item_kind,
        "text": text,
        "status": status,
        "created_at_ms": _timestamp_ms(row),
        "source_ref_ids": _string_list(
            row.get("source_ref_ids") or row.get("source_message_ids") or row.get("source_ref_id")
        ),
        "subjects": _string_list(row.get("subjects"), limit=8),
        "actor": _actor(row.get("actor")),
        "topics": _string_list(row.get("topics"), limit=8),
        "supersedes": _string_list(row.get("supersedes"), limit=12),
        "supersedes_source_message_ids": _string_list(
            row.get("supersedes_source_message_ids"), limit=12
        ),
        "temporal_status": _text(row.get("temporal_status") or "unknown", 20).lower(),
        "due_at_ms": max(0, int(row.get("due_at_ms", 0) or 0)),
        "due_text": _text(row.get("due_text"), 80),
        "due_precision": _text(row.get("due_precision") or "unknown", 20).lower(),
        "confidence": confidence,
        "salience": salience,
        "source_character": _text(row.get("source_character") or row.get("character"), 40),
        "source_time_period": _text(
            row.get("source_time_period") or row.get("date_world"), 80
        ),
        "memory_layer": _text(row.get("memory_layer"), 20).lower(),
        "resolution_reason": _text(row.get("resolution_reason"), 120),
    }


def _items_from_card(card_value: Any, *, card_index: int) -> list[dict[str, Any]]:
    card = _mapping(card_value)
    if not card or bool(card.get("archived", False)):
        return []
    common = {
        "created_at_ms": card.get("created_at_ms", 0),
        "confidence": card.get("confidence", 0.6),
        "salience": card.get("salience", 0.5),
        "source_character": card.get("character", ""),
        "source_ref_ids": [
            card.get("user_id", ""),
            *(_rows(card.get("user_ids"))),
            card.get("assistant_id", ""),
        ],
    }
    out: list[dict[str, Any]] = []
    for kind, field in (
        ("correction", "corrections"),
        ("commitment", "commitments"),
        ("open_thread", "open_threads"),
    ):
        for index, value in enumerate(_rows(card.get(field))):
            raw = {**common, **(_mapping(value) or {"text": value})}
            raw.setdefault(
                "actor",
                "user" if kind == "correction" else "character" if kind == "commitment" else "shared",
            )
            item = _adapt_item(value=raw, kind=kind, suffix=f"card:{card_index}:{field}:{index}")
            if item:
                out.append(item)
    return out


def _extract_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (list, tuple)):
        raw_rows = list(value)
        container: dict[str, Any] = {}
    else:
        container = _mapping(value)
        raw_rows = _rows(container.get("continuity_items") or container.get("items"))
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = _mapping(raw)
        if row and not row.get("kind") and any(
            key in row for key in ("corrections", "commitments", "open_threads")
        ):
            out.extend(_items_from_card(row, card_index=index))
            continue
        item = _adapt_item(raw, suffix=f"item:{index}")
        if item:
            out.append(item)
    for card_index, card in enumerate(
        _rows(container.get("turn_fact_cards") or container.get("fact_cards"))
    ):
        out.extend(_items_from_card(card, card_index=card_index))
    for kind, field in (
        ("correction", "corrections"),
        ("commitment", "commitments"),
        ("open_thread", "open_threads"),
    ):
        for index, raw in enumerate(_rows(container.get(field))):
            item = _adapt_item(raw, kind=kind, suffix=f"capsule:{field}:{index}")
            if item:
                out.append(item)
    return out


def _ledger_events(ledger_state: Any) -> list[dict[str, Any]]:
    if isinstance(ledger_state, (list, tuple)):
        rows = [_mapping(item) for item in ledger_state if _mapping(item)]
        if any(
            any(key in row for key in ("corrections", "commitments", "open_threads", "state_delta"))
            for row in rows
        ):
            return []
        return rows
    state = _mapping(ledger_state)
    events = [_mapping(item) for item in _rows(state.get("events")) if _mapping(item)]
    for item_id, status in (_mapping(state.get("item_status"))).items():
        events.append({"item_id": item_id, "status": status})
    for canonical_key, status in (_mapping(state.get("canonical_status"))).items():
        events.append({"canonical_key": canonical_key, "status": status})
    return events


def _ledger_fact_cards(ledger_state: Any) -> list[dict[str, Any]]:
    if isinstance(ledger_state, (list, tuple)):
        rows = [_mapping(item) for item in ledger_state if _mapping(item)]
        return [
            row for row in rows
            if any(key in row for key in ("corrections", "commitments", "open_threads", "state_delta"))
        ]
    state = _mapping(ledger_state)
    return [
        _mapping(item)
        for item in _rows(state.get("fact_cards") or state.get("turn_fact_cards"))
        if _mapping(item)
    ]


def _correction_negated_terms(text: Any) -> list[str]:
    terms: list[str] = []
    for match in re.finditer(
        r"(?:不是|并非|而非|不在)\s*([一-龥A-Za-z0-9·_-]{2,20})",
        str(text or ""),
    ):
        term = match.group(1).strip("的了是为在")
        if term and term not in terms:
            terms.append(term)
    return terms


def _correction_replacements(text: Any) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for clause in re.split(r"[，,。；;！？!?\n]+", str(text or "")):
        match = re.search(
            r"([一-龥A-Za-z0-9·_-]{2,24}?)(?:已经|现在)?"
            r"(?:改成|改为|改到|换成|更正为|更新为)\s*"
            r"([一-龥A-Za-z0-9·_-]{2,24})",
            clause,
        )
        if not match:
            continue
        subject = re.sub(
            r"^(?:对了|其实|等等|另外|还有)", "", match.group(1)
        ).strip("把将的了是为在 ")
        replacement = match.group(2).rstrip("的了吧呢啊呀哦 ")
        if len(subject) >= 2 and len(replacement) >= 2:
            replacements.append((subject, replacement))
    return list(dict.fromkeys(replacements))


def _correction_value_replacements(text: Any) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?:原先|原来|之前)(?:的|是|为)?\s*"
        r"([一-龥A-Za-z0-9·:_-]{2,24})\s*[，,；;]\s*"
        r"(?:现在|最终)?\s*(?:改成|改为|改到|换成|更正为|更新为)\s*"
        r"([一-龥A-Za-z0-9·:_-]{2,24})",
        str(text or ""),
    ):
        old_value = match.group(1).strip("的了是为在 ")
        new_value = match.group(2).rstrip("的了吧呢啊呀哦 ")
        if old_value and new_value:
            replacements.append((old_value, new_value))
    for match in re.finditer(
        r"([一-龥A-Za-z0-9·:_-]{2,24})\s*是口误\s*[，,；;]\s*"
        r"(?:正确)?(?:地点|时间)?\s*(?:是|为)\s*"
        r"([一-龥A-Za-z0-9·:_-]{2,24})",
        str(text or ""),
    ):
        old_value = re.sub(
            r"^.*(?:先前说的|刚才说的|之前说的|说的)",
            "",
            match.group(1),
        )
        replacements.append((old_value, match.group(2)))
    return list(dict.fromkeys(replacements))


def _correction_supersedes_text(correction_text: str, target_text: str) -> bool:
    for term in _correction_negated_terms(correction_text):
        if term in target_text and not re.search(
            rf"(?:不是|并非|而非|不在)\s*{re.escape(term)}", target_text
        ):
            return True
    for subject, replacement in _correction_replacements(correction_text):
        if subject not in target_text or replacement in target_text:
            continue
        tail = target_text.split(subject, 1)[1]
        if re.search(r"(?:是|为|在|定在|安排在|选在|改成|改为|改到|换成)", tail):
            return True
    for old_value, new_value in _correction_value_replacements(correction_text):
        if old_value in target_text and new_value not in target_text:
            return True
    return False


def _correction_value_supersedes_text(correction_text: str, target_text: str) -> bool:
    return any(
        old_value in target_text and new_value not in target_text
        for old_value, new_value in _correction_value_replacements(correction_text)
    )


def _fold_claim_variants(text: str) -> str:
    """Fold variant spellings of agent names onto the canonical name (already lower-cased text)."""
    for variant, canonical in text_variants().items():
        text = text.replace(variant.lower(), canonical)
    return text


def _claim_tokens(value: Any) -> set[str]:
    text = _fold_claim_variants(str(value or "").lower())
    recommendation_request = bool(
        re.search(r"询问|问问|问|意见|推荐|找.{0,8}推荐", text)
    )
    route_task = "路线" in text and bool(
        re.search(r"整理|规划|排好|安排|路线内容", text)
    )
    text = re.sub(
        r"(?:已经|还没|没有|答应|承诺|确认|完成|取消|作废|不再|会|要|将)",
        "",
        text,
    )
    compact = _normalized_key(text)
    tokens: set[str] = set(re.findall(r"[a-z0-9_-]{2,}", compact))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", compact))
    for size in (2, 3):
        for index in range(max(0, len(chinese) - size + 1)):
            tokens.add(chinese[index:index + size])
    if recommendation_request:
        tokens.add("recommendation_request")
    if route_task:
        tokens.add("route_task")
    return tokens


def _authority_matches_item(authority: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    if not _actors_compatible(authority, item):
        return False
    authority_key = str(authority.get("canonical_key", "") or "")
    if authority_key and _same_explicit_claim(authority, item):
        return True
    superseded_refs = set(authority.get("supersedes_source_message_ids", []) or [])
    if superseded_refs.intersection(item.get("source_ref_ids", []) or []):
        return True
    authority_refs = set(authority.get("source_ref_ids", []) or [])
    if authority_refs.intersection(item.get("source_ref_ids", []) or []):
        return True
    shared = _claim_tokens(authority.get("text", "")) & _claim_tokens(item.get("text", ""))
    return len(shared) >= 2


def _explicit_authority_matches_item(
    authority: Mapping[str, Any], item: Mapping[str, Any]
) -> bool:
    if not _actors_compatible(authority, item):
        return False
    authority_key = str(authority.get("canonical_key", "") or "")
    if authority_key and _same_explicit_claim(authority, item):
        return True
    superseded_refs = set(authority.get("supersedes_source_message_ids", []) or [])
    if superseded_refs.intersection(item.get("source_ref_ids", []) or []):
        return True
    authority_refs = set(authority.get("source_ref_ids", []) or [])
    if authority_refs.intersection(item.get("source_ref_ids", []) or []):
        return True
    return False


def _infer_state_actor(text: str, source_character: str = "") -> str:
    if user_name() in text:
        return "user"
    character_names = set(agent_names())
    if source_character:
        character_names.add(source_character)
    if any(name and name in text for name in character_names):
        return "character"
    return "unknown"


def _apply_fact_card_resolution(items: list[dict[str, Any]], ledger_state: Any) -> None:
    for card_index, card in enumerate(_ledger_fact_cards(ledger_state)):
        authorities = _items_from_card(card, card_index=card_index)
        for authority in authorities:
            if authority["kind"] == "correction":
                for item in items:
                    if item["created_at_ms"] > authority["created_at_ms"] > 0:
                        continue
                    if not (
                        _explicit_authority_matches_item(authority, item)
                        or _correction_value_supersedes_text(
                            authority["text"], item["text"]
                        )
                        or (
                            _actors_compatible(authority, item)
                            and _correction_supersedes_text(authority["text"], item["text"])
                        )
                    ):
                        continue
                    item["status"] = "superseded"
                    item["resolution_reason"] = f"ledger_correction:{authority['item_id']}"
                continue
            if authority["status"] not in _CLOSED_STATUSES:
                continue
            for item in items:
                if item["kind"] != authority["kind"]:
                    continue
                if not _authority_matches_item(authority, item):
                    continue
                item["status"] = authority["status"]
                item["resolution_reason"] = f"ledger_{authority['status']}:{authority['item_id']}"

        for action_index, action in enumerate(_rows(card.get("character_action_updates"))):
            action_row = _mapping(action)
            transition = _text(action_row.get("transition"), 20).lower()
            if transition not in {"complete", "cancel"}:
                continue
            action_text = _text(action_row.get("event_text"), 240)
            if not action_text:
                continue
            action_authority = {
                "text": action_text,
                "actor": "character",
                "canonical_key": "",
                "canonical_key_explicit": False,
                "source_ref_ids": [],
                "supersedes_source_message_ids": [],
            }
            status = "fulfilled" if transition == "complete" else "cancelled"
            for item in items:
                if item["kind"] not in {"commitment", "open_thread"}:
                    continue
                if item["status"] not in _OPEN_STATUSES:
                    continue
                if not _authority_matches_item(action_authority, item):
                    continue
                item["status"] = status
                item["resolution_reason"] = (
                    f"ledger_character_action:{card_index}:{action_index}:{transition}"
                )

        state_delta = _text(card.get("state_delta"), 360)
        if not state_delta:
            continue
        if re.search(r"取消|作废|不再|不用", state_delta):
            inferred_status = "cancelled"
        elif re.search(
            r"已完成|完成了|履约|兑现|办完|买完|带到|带来|"
            r"已经.{0,16}(?:完成|确认完|带到|带来|交给|交付|买完|做好|做完)",
            state_delta,
        ):
            inferred_status = "fulfilled"
        else:
            continue
        state_authority = {
            "text": state_delta,
            "actor": _infer_state_actor(state_delta, _text(card.get("character"), 40)),
            "canonical_key": _normalized_key(card.get("canonical_key")),
            "canonical_key_explicit": bool(card.get("canonical_key")),
            "source_ref_ids": _string_list(
                [card.get("user_id", ""), *(_rows(card.get("user_ids"))), card.get("assistant_id", "")]
            ),
            "supersedes_source_message_ids": _string_list(
                card.get("supersedes_source_message_ids")
            ),
        }
        for item in items:
            if item["kind"] not in {"commitment", "open_thread"}:
                continue
            if item["status"] not in _OPEN_STATUSES:
                continue
            if not _authority_matches_item(state_authority, item):
                continue
            item["status"] = inferred_status
            item["resolution_reason"] = f"ledger_state_delta:{card_index}"


def resolve_continuity_items(
    capsule_or_items: Any,
    *,
    ledger_state: Any = None,
    as_of_ms: int = 0,
) -> dict[str, Any]:
    """Resolve correction and lifecycle state without writing any ledger."""
    items = _extract_items(capsule_or_items)
    by_id = {item["item_id"]: item for item in items}

    for event in _ledger_events(ledger_state):
        status = _text(event.get("status"), 30).lower()
        if status not in _OPEN_STATUSES | _CLOSED_STATUSES:
            continue
        target_id = _text(event.get("item_id") or event.get("target_id"), 120)
        target_key = _normalized_key(event.get("canonical_key"))
        target_actor = _actor(event.get("actor"))
        for item in items:
            if target_id and item["item_id"] != target_id:
                continue
            if target_key and item["canonical_key"] != target_key:
                continue
            if target_key and not target_id and (
                target_actor == "unknown" or item["actor"] != target_actor
            ):
                continue
            if not target_id and not target_key:
                continue
            item["status"] = status or "active"
            item["resolution_reason"] = _text(event.get("reason"), 120) or "ledger_state"

    _apply_fact_card_resolution(items, ledger_state)

    for item in sorted(items, key=lambda row: (row["created_at_ms"], row["item_id"])):
        for target_id in item.get("supersedes", []):
            target = by_id.get(target_id)
            if (
                target
                and target["item_id"] != item["item_id"]
                and _actors_compatible(item, target)
            ):
                target["status"] = "superseded"
                target["resolution_reason"] = f"superseded_by:{item['item_id']}"
        superseded_source_ids = set(item.get("supersedes_source_message_ids", []))
        if superseded_source_ids:
            for target in items:
                if target["item_id"] == item["item_id"]:
                    continue
                if not superseded_source_ids.intersection(target.get("source_ref_ids", [])):
                    continue
                if not _actors_compatible(item, target):
                    continue
                target["status"] = "superseded"
                target["resolution_reason"] = f"source_corrected_by:{item['item_id']}"
        if item["kind"] != "correction" or item["status"] in _CLOSED_STATUSES:
            continue
        for target in items:
            if target["item_id"] == item["item_id"]:
                continue
            if not _same_explicit_claim(item, target):
                continue
            if target["created_at_ms"] > item["created_at_ms"] > 0:
                continue
            if target["kind"] == "correction" and target["created_at_ms"] == item["created_at_ms"]:
                continue
            target["status"] = "superseded"
            target["resolution_reason"] = f"corrected_by:{item['item_id']}"

    latest_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in sorted(items, key=lambda row: (row["created_at_ms"], row["item_id"])):
        identity = (item["actor"], item["kind"], item["canonical_key"])
        previous = latest_by_identity.get(identity)
        if previous and previous["text"] == item["text"] and previous["status"] in _OPEN_STATUSES:
            previous["status"] = "superseded"
            previous["resolution_reason"] = f"duplicate_of:{item['item_id']}"
        latest_by_identity[identity] = item

    effective_as_of_ms = max(0, int(as_of_ms or 0))
    for item in items:
        due_at_ms = int(item.get("due_at_ms", 0) or 0)
        if due_at_ms and effective_as_of_ms:
            item["due_state"] = (
                "overdue_needs_confirmation"
                if due_at_ms < effective_as_of_ms and item["status"] in _OPEN_STATUSES
                else "upcoming"
            )
        else:
            item["due_state"] = "unknown"
        item["is_unresolved"] = bool(
            item["kind"] in _LIFECYCLE_KINDS and item["status"] in _OPEN_STATUSES
        )
    ordered = sorted(items, key=lambda row: (row["created_at_ms"], row["item_id"]))
    unresolved = [dict(item) for item in ordered if item["is_unresolved"]]
    return {
        "contract_version": "daily_continuity_resolution_v2",
        "as_of_ms": effective_as_of_ms,
        "items": [dict(item) for item in ordered],
        "unresolved_items": unresolved,
        "resolved_count": len(ordered) - len(unresolved),
        "unresolved_count": len(unresolved),
    }


def resolve_continuity(
    capsule: Any,
    *,
    ledger_state: Any = None,
    as_of_ms: int = 0,
) -> dict[str, Any]:
    """Capsule-oriented alias used by integration adapters."""
    return resolve_continuity_items(capsule, ledger_state=ledger_state, as_of_ms=as_of_ms)


__all__ = ["MappingLike", "resolve_continuity", "resolve_continuity_items"]
