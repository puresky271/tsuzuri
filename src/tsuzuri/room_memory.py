"""Multi-party conversation memory: rooms where the user talks with several agents at once.

A room keeps one shared history; each agent reads it from its own point of view and
the room accumulates TurnFact cards attributed to whichever agent spoke. This module
holds the room logic that does not depend on any transport:

- ``build_group_context``      — room history -> LLM messages for one speaker, with
                                 other agents' lines attributed as ``[name]：`` user turns.
- ``room_history_for_workspace`` — normalize room entries for the TurnFact / workspace layer.
- ``merge_room_fact_card``     — add one speaker's card to the room's card window.
- ``room_fact_memory_meta`` / ``write_room_fact_memory`` — persist a room fact into the
                                 speaker's memory (or the collective scope for a shared room).

Presence (who is physically together) is supplied by the host; this module only
consumes it.
"""
from __future__ import annotations

from typing import Any, Callable

from tsuzuri.persona_registry import persona_regex


def _entry_ts_ms(entry: dict | None) -> int:
    raw = (entry or {}).get("timestamp", 0) or (entry or {}).get("ts", 0) or 0
    try:
        val = float(raw)
        if val <= 0:
            return 0
        if val < 1e11:
            val *= 1000.0
        return int(val)
    except Exception:
        return 0


def build_group_context(history: list, limit: int = 80, *, current_character: str = "") -> list[dict]:
    """Room history -> chat messages from ``current_character``'s point of view.

    Own lines stay ``assistant`` turns (style continuity); other agents' lines become
    ``user`` turns prefixed ``[name]：`` so the model never mistakes them for its own;
    a user message that imitates that prefix is escaped. ``current_character=""``
    keeps the legacy behaviour (every agent line is an assistant turn).
    """
    msgs = []
    current = str(current_character or "").strip()
    speaker_prefix = persona_regex(r"^\s*\[{AGENTS}\]：")
    for entry in history[-limit:]:
        role = entry.get("role", "user")
        character = entry.get("character", "")
        content = entry.get("content", "")
        if not content:
            continue
        if role == "user":
            # Reserve [角色]： for system speaker attribution. Escape a user
            # supplied lookalike so it cannot be mistaken for another speaker.
            if speaker_prefix.match(content):
                content = "[用户原话]：" + content
            msgs.append({"role": "user", "content": content})
        elif role == "assistant" and character:
            if current and character != current:
                msgs.append({"role": "user", "content": f"[{character}]：{content}"})
            else:
                msgs.append({"role": "assistant", "content": content, "name": character})
        elif role == "system":
            msgs.append({"role": "system", "content": content})
    return msgs


def room_history_for_workspace(room_history: list, current_character: str, *, room_id: str = "") -> list:
    """Room entries -> workspace/TurnFact-compatible entries.

    Adds ``timestamp`` (ms) and ``id`` (fact-window ordering and card coverage depend on
    them) and keeps ``character`` for speaker attribution in group-mode rendering.
    Legacy entries without an id get a deterministic ``{room_id}-{ts}-{i}`` id.
    """
    del current_character  # attribution happens at render time
    out: list[dict] = []
    for idx, entry in enumerate(room_history or []):
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", "") or "").strip().lower()
        content = str(entry.get("content", "") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        row = dict(entry)
        ts_ms = _entry_ts_ms(row)
        if ts_ms <= 0:
            ts_ms = idx + 1  # legacy entry without a timestamp: keep order
        row["timestamp"] = ts_ms
        if not str(row.get("id", "") or ""):
            row["id"] = f"{room_id or 'room'}-{ts_ms}-{idx}"
        out.append(row)
    return out


def merge_room_fact_card(cards: Any, card: dict, *, recent: int = 80, reserve: int = 24, limit: int = 104) -> tuple[list[dict], str]:
    """Add one speaker's TurnFact card to a room's window; returns ``(cards, state_summary)``.

    Several speakers share one user message per round, so de-duplication keys on
    ``(user_id, character)`` as well as ``assistant_id``.
    """
    from tsuzuri.turn_fact_ledger import build_fact_state_summary, lifecycle_card_window, sanitize_fact_cards

    existing = sanitize_fact_cards(cards if isinstance(cards, list) else [], limit=limit)
    incoming_user_id = str(card.get("user_id", "") or "")
    incoming_char = str(card.get("character", "") or "")
    kept = [
        row for row in existing
        if str(row.get("assistant_id", "") or "") != str(card.get("assistant_id", "") or "")
        and (
            not incoming_user_id
            or str(row.get("user_id", "") or "") != incoming_user_id
            or str(row.get("character", "") or "") != incoming_char
        )
    ]
    kept.append(card)
    merged = sanitize_fact_cards(lifecycle_card_window(kept, recent=recent, reserve=reserve), limit=limit)
    return merged, build_fact_state_summary(merged)


def room_fact_memory_meta(room_id: str, assistant_entry: dict, card: dict, *,
                          source: str = "web_multi_room",
                          fact_rewrite_llm_caller: Callable[[str], str] | None = None) -> dict:
    context = assistant_entry.get("room_context", {}) if isinstance(assistant_entry, dict) else {}
    context = context if isinstance(context, dict) else {}
    return {
        "source": source, "room_id": room_id,
        "participants": list(context.get("participants", [])), "speaker_role": "speaker",
        "dev_mode": True, "chat_mode": "chat", "channel": context.get("channel", "line"),
        "co_present_chars": [row["character"] for row in context.get("members", []) if row.get("co_located")],
        "turn_fact_card": card, "fact_rewrite_llm_caller": fact_rewrite_llm_caller,
    }


def room_fact_text(card: Any) -> str:
    if not isinstance(card, dict):
        return ""
    return str(card.get("fact") or card.get("state_delta") or "").strip()


def room_fact_is_writable(character: str, assistant_entry: dict, card: Any) -> bool:
    """Only LLM-extracted facts from this speaker's own reply are persisted."""
    return bool(
        isinstance(card, dict)
        and card.get("model") != "rule_fallback"
        and room_fact_text(card)
        and card.get("assistant_id") == assistant_entry.get("id")
        and card.get("character") == character
    )


def write_room_fact_memory(core, room_id: str, character: str, user_entry: dict, assistant_entry: dict,
                           card: dict, *, shared_room: bool = False,
                           fact_rewrite_llm_caller: Callable[[str], str] | None = None,
                           source: str = "web_multi_room") -> bool:
    """Persist one room fact once (raw dialogue stays payload evidence only).

    ``shared_room=True`` writes the fact to the collective scope; otherwise it lands in
    the speaker's own memory with room provenance.
    """
    if core is None or not room_fact_is_writable(character, assistant_entry, card):
        return False
    meta = room_fact_memory_meta(room_id, assistant_entry, card, source=source,
                                 fact_rewrite_llm_caller=fact_rewrite_llm_caller)
    if shared_room:
        core.append_collective_memory_event(
            event_type="group_turn", text=room_fact_text(card),
            payload={"meta": {k: v for k, v in meta.items() if not callable(v)},
                     "turn_fact_card": card, "user_text": user_entry.get("content", ""),
                     "assistant_reply": assistant_entry.get("content", "")},
            salience=card.get("salience", 0.6), confidence=card.get("confidence", 0.6),
        )
    else:
        core.append_turn_event(
            character, str(user_entry.get("content") or ""),
            str(assistant_entry.get("content") or ""), event_nodes=[], meta=meta, memory_snapshot={},
        )
    return True


__all__ = [
    "build_group_context",
    "merge_room_fact_card",
    "room_fact_is_writable",
    "room_fact_memory_meta",
    "room_fact_text",
    "room_history_for_workspace",
    "write_room_fact_memory",
]
