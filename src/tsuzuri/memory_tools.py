"""Bounded pre-reply memory tools for the main developer 1v1 chat path."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


from tsuzuri.secrets_loader import env_bool as _env_enabled
from tsuzuri.persona_registry import collective_scope

TOOL_SCHEMAS = (
    {
        "name": "memory_recall",
        "description": "Recall up to three governed memory cues.",
        "args": {"query": "string", "scope": "char|collective|auto", "top_k": "1..3"},
    },
    {
        "name": "memory_lookup",
        "description": "Look up answer-bearing semantic facts only.",
        "args": {"query": "string", "top_k": "1..5"},
    },
    {
        "name": "memory_propose_write",
        "description": "Propose a fact, state delta, or commitment through server-side gates.",
        "args": {
            "kind": "fact|state_delta|commitment",
            "text": "string",
            "evidence": "exact substring from the current user message",
        },
    },
)

_VALID_TOOLS = {str(row["name"]): row for row in TOOL_SCHEMAS}
_VALID_SCOPES = {"char", "collective", "auto"}
_VALID_WRITE_KINDS = {"fact", "state_delta", "commitment"}
_PROACTIVE_TRIGGER_KINDS = {"chat_followup", "delayed_followup", "short_silence"}
_PROACTIVE_MIN_SALIENCE = 0.75
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_MEMORY_RECALL_SIGNAL_RE = re.compile(
    r"(还记得|记不记得|记得吗|回忆|想起|之前|以前|上次|那次|说过|提过|聊过|谈过|讨论过|"
    r"刚才|前面|这件事|那件事|这段|那段|后来|后续|接着说|继续说)"
)
_MEMORY_RECALL_DIRECT_RE = re.compile(
    r"(还记得|记不记得|记得吗|回忆|想起|说过什么|提过什么|聊过什么|谈过什么|讨论过什么|"
    r"后来(?:怎么样|如何)|有后续)"
)
_MEMORY_RECALL_QUESTION_RE = re.compile(r"(?:[?？]|什么|哪(?:里|儿)?|哪个|哪些|怎么|怎样|是否|有没有)")
_MEMORY_RECALL_REFERENCE_RE = re.compile(
    r"(?:之前|以前|上次|那次|刚才|前面|后来|后续).{0,12}(?:说过|提过|聊过|谈过|讨论过|提到|那件事|这件事|那段|这段)"
)


def _has_memory_recall_signal(user_text: str) -> bool:
    query = re.sub(r"\s+", "", str(user_text or "")).strip()
    if not query:
        return False
    if _MEMORY_RECALL_DIRECT_RE.search(query):
        return True
    if not _MEMORY_RECALL_SIGNAL_RE.search(query):
        return False
    if _MEMORY_RECALL_REFERENCE_RE.search(query):
        return True
    has_temporal = bool(re.search(r"(?:之前|以前|上次|那次|刚才|前面|后来|后续)", query))
    return has_temporal and bool(_MEMORY_RECALL_QUESTION_RE.search(query))


@dataclass
class ToolRoundResult:
    activated: bool = False
    activation_reason: str = ""
    decision_called: bool = False
    records: list[dict] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    rejected_calls: list[dict] = field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0
    approx_tokens: int = 0
    error: str = ""


def memory_turn_tools_enabled() -> bool:
    return _env_enabled("MEM_TURN_TOOLS_ENABLE", False)


def memory_turn_tools_write_enabled() -> bool:
    return _env_enabled("MEM_TURN_TOOLS_WRITE_ENABLE", False)


def memory_turn_tools_model() -> str:
    return str(os.environ.get("MEM_TURN_TOOLS_MODEL", "deepseek-flash") or "deepseek-flash").strip()


def memory_turn_tools_max_calls() -> int:
    try:
        value = int(os.environ.get("MEM_TURN_TOOLS_MAX_CALLS", "2") or 2)
    except (TypeError, ValueError):
        value = 2
    return max(1, min(2, value))


def should_activate_tool_round(
    user_text: str,
    pack: dict | None,
    *,
    is_guest: bool,
    channel_kind: str,
) -> tuple[bool, str]:
    if is_guest:
        return False, "guest_blocked"
    if str(channel_kind or "").strip().lower() != "web_1v1":
        return False, "channel_blocked"

    memory_pack = pack if isinstance(pack, dict) else {}
    read_bundle = memory_pack.get("memory_read_bundle", {})
    read_bundle = read_bundle if isinstance(read_bundle, dict) else {}
    analysis = read_bundle.get("query_analysis", {})
    analysis = analysis if isinstance(analysis, dict) else {}
    recall_signal = _has_memory_recall_signal(str(user_text or ""))
    explicit = bool(analysis.get("explicit_recall_intent", False)) and recall_signal
    if explicit:
        return True, "explicit_recall_intent"

    # Bounded-slot questions such as "晚饭吃什么" or "明天几点集训"
    # are ordinary fact queries, not memory-tool requests.  Only a recall
    # phrase is strong enough to activate the planner when analysis missed
    # the signal (short queries are intentionally conservative there).
    plan = memory_pack.get("memory_surface_plan", {})
    plan = plan if isinstance(plan, dict) else {}
    suppression_reason = str(plan.get("suppression_reason", "") or "").strip()
    if suppression_reason in {"no_consolidated_memory_rows", "no_safe_memory_rows", "no_relevant_rows"} and recall_signal:
        return True, f"surface_suppressed:{suppression_reason[:80]}"
    if recall_signal:
        return True, "memory_recall_signal"
    return False, "no_activation_signal"


def _one_line(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"<[^>]{0,80}>", "", text).strip()
    return text[:limit]


def _extract_json_object(text: str) -> str:
    source = str(text or "").strip()
    match = _JSON_FENCE_RE.search(source)
    if match:
        source = str(match.group(1) or "").strip()
    left = source.find("{")
    right = source.rfind("}")
    return source[left:right + 1] if left >= 0 and right > left else source


def parse_tool_calls(raw_response: str, *, max_calls: int = 2) -> tuple[list[dict], list[dict], str]:
    try:
        payload = json.loads(_extract_json_object(raw_response))
    except Exception as exc:
        return [], [], f"json_decode_error:{type(exc).__name__}"
    if not isinstance(payload, dict) or not isinstance(payload.get("calls", []), list):
        return [], [], "calls_not_list"

    accepted: list[dict] = []
    rejected: list[dict] = []
    limit = max(1, min(2, int(max_calls or 2)))
    for index, raw in enumerate(payload.get("calls", [])):
        if len(accepted) >= limit:
            rejected.append({"index": index, "reason": "max_calls_exceeded"})
            continue
        if not isinstance(raw, dict):
            rejected.append({"index": index, "reason": "call_not_object"})
            continue
        tool = str(raw.get("tool", "") or "").strip()
        args = raw.get("args", {})
        if tool not in _VALID_TOOLS:
            rejected.append({"index": index, "tool": tool, "reason": "unknown_tool"})
            continue
        if not isinstance(args, dict):
            rejected.append({"index": index, "tool": tool, "reason": "args_not_object"})
            continue
        clean: dict[str, Any]
        if tool == "memory_recall":
            query = _one_line(args.get("query"), 240)
            scope = str(args.get("scope", "auto") or "auto").strip().lower()
            if not query or scope not in _VALID_SCOPES:
                rejected.append({"index": index, "tool": tool, "reason": "invalid_recall_args"})
                continue
            try:
                top_k = int(args.get("top_k", 3) or 3)
            except (TypeError, ValueError):
                top_k = 3
            clean = {"query": query, "scope": scope, "top_k": max(1, min(3, top_k))}
        elif tool == "memory_lookup":
            query = _one_line(args.get("query"), 240)
            if not query:
                rejected.append({"index": index, "tool": tool, "reason": "invalid_lookup_args"})
                continue
            try:
                top_k = int(args.get("top_k", 5) or 5)
            except (TypeError, ValueError):
                top_k = 5
            clean = {"query": query, "top_k": max(1, min(5, top_k))}
        else:
            kind = str(args.get("kind", "fact") or "fact").strip().lower()
            text = _one_line(args.get("text"), 240)
            evidence = _one_line(args.get("evidence"), 320)
            if kind not in _VALID_WRITE_KINDS or not text or not evidence:
                rejected.append({"index": index, "tool": tool, "reason": "invalid_write_args"})
                continue
            clean = {"kind": kind, "text": text, "evidence": evidence}
        accepted.append({"tool": tool, "args": clean})
    return accepted, rejected, ""


def _build_decision_prompt(user_text: str, pack: dict | None, *, max_calls: int, write_enabled: bool) -> str:
    memory_pack = pack if isinstance(pack, dict) else {}
    plan = memory_pack.get("memory_surface_plan", {})
    plan = plan if isinstance(plan, dict) else {}
    rows = plan.get("selected_surface_rows", [])
    selected_count = len(rows) if isinstance(rows, list) else 0
    state = {
        "selected_surface_count": selected_count,
        "suppression_reason": str(plan.get("suppression_reason", "") or "")[:120],
        "surface_mode": str(plan.get("surface_mode", "") or "")[:40],
        "write_enabled": bool(write_enabled),
    }
    return (
        "You are a memory tool planner. Return JSON only: "
        '{"calls":[{"tool":"memory_recall","args":{...}}]}.\n'
        f"Choose zero to {max_calls} calls. Do not answer the user. "
        "Use tools only when the current reply needs missing memory evidence. "
        "For memory_propose_write, evidence must be an exact substring of CURRENT_USER and "
        "the proposal must be directly grounded by it.\n"
        f"TOOLS={json.dumps(TOOL_SCHEMAS, ensure_ascii=False)}\n"
        f"MEMORY_STATE={json.dumps(state, ensure_ascii=False)}\n"
        f"CURRENT_USER={json.dumps(_one_line(user_text, 600), ensure_ascii=False)}"
    )


def _explicit_query(query: str) -> str:
    try:
        from tsuzuri.memory_core import memory_explicit_query_has_bounded_slot

        if memory_explicit_query_has_bounded_slot(query):
            return query
    except Exception:
        pass
    return f"你还记得我之前说过的{query}吗？"


def _row_timestamp_ms(row: dict) -> int:
    raw = row.get("created_at_ms", 0)
    try:
        value = float(raw or 0)
        if value > 0:
            return int(value * 1000 if value < 1e11 else value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(row.get("created_at", "") or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    except Exception:
        return int(time.time() * 1000)


def _records_from_rows(tool: str, rows: list[dict], character: str, *, top_k: int) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = _one_line(row.get("text"), 360)
        normalized = re.sub(r"\W+", "", text.lower())
        if not text or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        item_id = str(row.get("item_id", "") or row.get("evidence_id", "") or "").strip()
        ref_tail = item_id[:24] or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        try:
            salience = float(row.get("salience", 0.5) or 0.5)
        except (TypeError, ValueError):
            salience = 0.5
        try:
            confidence = float(row.get("confidence", row.get("_confidence", 0.6)) or 0.6)
        except (TypeError, ValueError):
            confidence = 0.6
        records.append({
            "text": text,
            "kind": "fact",
            "source_channel": "memory_recall",
            "source_ref_id": f"memtool:{tool}:{ref_tail}",
            "source_character": character,
            "created_at_ms": _row_timestamp_ms(row),
            "salience": max(0.0, min(1.0, salience)),
            "confidence": max(0.0, min(1.0, confidence)),
        })
        if len(records) >= top_k:
            break
    return records


def _execute_read_tool(
    tool: str,
    args: dict,
    character: str,
    *,
    provider: Any,
) -> tuple[list[dict], dict]:
    query = str(args.get("query", "") or "").strip()
    top_k = int(args.get("top_k", 3) or 3)
    scope = str(args.get("scope", "char") or "char") if tool == "memory_recall" else "char"
    target_character = collective_scope() if scope == "collective" else character
    retrieval_query = _explicit_query(query)
    pack = provider.build_memory_context_pack(
        target_character,
        retrieval_query,
        query_bundle={"query_text": retrieval_query, "character": target_character},
        max_items=max(6, top_k * 3),
        risk_bundle={"memory_read_scope": "developer"},
    )
    pack = pack if isinstance(pack, dict) else {}
    plan = pack.get("memory_surface_plan", {})
    plan = plan if isinstance(plan, dict) else {}
    rows = plan.get("selected_surface_rows", [])
    rows = [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if tool == "memory_lookup":
        try:
            from tsuzuri.memory_core import memory_explicit_row_is_answer_bearing

            rows = [
                row for row in rows
                if str(row.get("layer", "") or "").strip().lower() == "semantic"
                and float(row.get("confidence", row.get("_confidence", 0.0)) or 0.0) >= 0.58
                and memory_explicit_row_is_answer_bearing(row, retrieval_query)
            ]
        except Exception:
            rows = []
    records = _records_from_rows(tool, rows, character, top_k=top_k)
    return records, {
        "status": "ok" if records else "empty",
        "target_character": target_character,
        "result_ids": [str(row.get("source_ref_id", "") or "") for row in records],
        "result_count": len(records),
    }


def _execute_propose_write(
    args: dict,
    character: str,
    current_user_text: str,
    *,
    write_enabled: bool,
    store: Any | None,
) -> tuple[list[dict], dict]:
    text = str(args.get("text", "") or "").strip()
    evidence = str(args.get("evidence", "") or "").strip()
    if not write_enabled:
        return [], {
            "status": "rejected", "rejected_reason": "write_disabled",
        }
    compact_user = re.sub(r"\s+", " ", str(current_user_text or "")).strip()
    compact_evidence = re.sub(r"\s+", " ", evidence).strip()
    if not compact_evidence or compact_evidence not in compact_user:
        return [], {
            "status": "rejected", "rejected_reason": "evidence_not_in_user_text",
        }
    try:
        from tsuzuri.memory_core import (
            canonicalize_developer_fact,
            fact_is_dialogue_residue,
            rewrite_fact_via_llm,
            get_memory_core,
        )
        from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH

        if fact_is_dialogue_residue(text):
            return [], {
                "status": "rejected", "rejected_reason": "dialogue_residue",
            }
        expected_fact = rewrite_fact_via_llm(
            canonicalize_developer_fact(text, character=character),
            character=character,
        )
        if not expected_fact:
            return [], {
                "status": "rejected", "rejected_reason": "proposal_normalization_failed",
            }
        active_store = store or get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)
        proposal_meta = {
            "latest_user_msg": f"记住，{compact_user}",
            "semantic_facts": [text],
            "memory_plan": {"mode_scope": "normal", "source_kind": "memory_tool_proposal"},
            "memory_tool_proposal": {
                "kind": str(args.get("kind", "fact") or "fact"),
                "evidence": compact_evidence,
            },
        }
        plan = active_store.plan_turn_memory_writes(
            character,
            compact_user,
            "",
            event_nodes=[],
            meta=proposal_meta,
        )
        records = plan.get("records", []) if isinstance(plan, dict) else []
        expected_key = re.sub(r"\s+", "", expected_fact).rstrip("。.!！")
        accepted_records = []
        for row in records:
            if not isinstance(row, dict):
                continue
            payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
            row_fact = str(payload.get("fact", "") or row.get("text", "") or "").strip()
            row_key = re.sub(r"\s+", "", row_fact).rstrip("。.!！")
            if (
                str(row.get("layer", "") or "") == "semantic"
                and str(row.get("memory_type", "") or "") in ("shared_fact", "character_projection")
                and row_key == expected_key
            ):
                accepted_records.append(dict(row))
        accepted_facts = {
            str((row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}).get("fact", "") or row.get("text", "") or "").strip()
            for row in accepted_records
        }
        accepted_records.extend(
            dict(row) for row in records
            if isinstance(row, dict)
            and str(row.get("engram_type", "") or "") == "collective_user_fact"
            and str((row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}).get("fact", "") or "").strip() in accepted_facts
        )
        if not accepted_records:
            suppressed = plan.get("suppressed", []) if isinstance(plan, dict) else []
            reject_reason = next(
                (
                    str(row.get("reason", "") or "")
                    for row in suppressed
                    if isinstance(row, dict)
                    and row.get("reason")
                    and re.sub(r"\s+", "", str(row.get("text", "") or "")).rstrip("。.!！") == expected_key
                ),
                "policy_rejected",
            )
            return [], {
                "status": "rejected", "rejected_reason": reject_reason,
            }
        write_plan = {
            "records": accepted_records,
            "suppressed": list(plan.get("suppressed", []) or []),
            "shared_payload": dict(plan.get("shared_payload", {}) or {}),
            "topic_tags": list(plan.get("topic_tags", []) or []),
        }
        active_store.append_turn_event(
            character,
            compact_user,
            "",
            event_nodes=[],
            meta={**proposal_meta, "memory_write_plan": write_plan},
            memory_snapshot={},
        )
        written_ids = [
            hashlib.sha256(
                f"{row.get('character')}|{row.get('layer')}|{row.get('text')}".encode("utf-8")
            ).hexdigest()[:16]
            for row in accepted_records
        ]
        return [], {
            "status": "accepted", "result_ids": written_ids, "written_count": len(accepted_records),
        }
    except Exception as exc:
        return [], {
            "status": "error", "rejected_reason": f"{type(exc).__name__}:{exc}"[:160],
        }


def mem_entity_proactive_enabled() -> bool:
    return os.getenv("MEM_ENTITY_PROACTIVE_ENABLE", "0").strip() == "1"


def _select_proactive_evidence(context: dict, evidence_source: str) -> tuple[str, dict]:
    source = str(evidence_source or "turn_fact_card").strip().lower()
    cards = context.get("turn_fact_cards", []) if isinstance(context.get("turn_fact_cards", []), list) else []
    recent_user = context.get("recent_user_msgs", []) if isinstance(context.get("recent_user_msgs", []), list) else []
    if source == "recent_user_text":
        evidence = next((str(value or "").strip() for value in reversed(recent_user) if str(value or "").strip()), "")
        return evidence, {
            "fact": str(context.get("fact", "") or evidence).strip(),
            "salience": context.get("salience", 0.0),
            "confidence": context.get("confidence", 0.7),
            "topic_labels": list(context.get("topic_labels", []) or []),
        }
    for raw in reversed(cards):
        if not isinstance(raw, dict):
            continue
        scope = str(raw.get("audience_scope", "") or "")
        if scope and scope != "developer":
            continue
        fact = _one_line(raw.get("fact"), 320)
        if fact:
            return fact, dict(raw)
    return "", {}


def propose_proactive_memory_record(
    character: str,
    *,
    trigger_kind: str,
    context: dict,
    evidence_source: str = "turn_fact_card",
    store: Any | None = None,
) -> dict:
    """Persist a high-signal proactive fact before reply generation.

    This is an orchestrator API. It is not exposed to the character LLM.
    """
    if not mem_entity_proactive_enabled():
        return {"ok": True, "status": "disabled"}
    trigger = str(trigger_kind or "").strip().lower()
    if trigger not in _PROACTIVE_TRIGGER_KINDS:
        return {"ok": True, "status": "ineligible_trigger", "trigger_kind": trigger}
    payload = context if isinstance(context, dict) else {}
    if bool(payload.get("is_guest", False)):
        return {"ok": True, "status": "guest_noop"}
    evidence, card = _select_proactive_evidence(payload, evidence_source)
    fact = _one_line(card.get("fact"), 320)
    try:
        salience = float(card.get("salience", 0.0) or 0.0)
        confidence = float(card.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return {"ok": True, "status": "low_signal"}
    if not fact or not evidence or salience < _PROACTIVE_MIN_SALIENCE or confidence < 0.65:
        return {"ok": True, "status": "low_signal"}

    try:
        from tsuzuri.memory_core import (
            canonicalize_developer_fact,
            fact_is_dialogue_residue,
            rewrite_fact_via_llm,
            get_memory_core,
        )
        from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH

        if fact_is_dialogue_residue(fact):
            return {"ok": True, "status": "rejected", "rejected_reason": "dialogue_residue"}
        expected_fact = rewrite_fact_via_llm(
            canonicalize_developer_fact(fact, character=character),
            character=character,
        )
        if not expected_fact:
            return {"ok": True, "status": "rejected", "rejected_reason": "normalization_failed"}
        active_store = store or get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)
        normalized_fact = re.sub(r"\s+", "", expected_fact)
        fingerprint = hashlib.sha256(
            f"{str(character or '').strip()}|{normalized_fact}".encode("utf-8")
        ).hexdigest()[:24]
        evidence_id = f"proactive:{fingerprint}"
        with active_store._lock:
            conn = active_store._connect()
            try:
                exists = conn.execute(
                    "SELECT 1 FROM memory_items WHERE evidence_id=? AND source='proactive_record' "
                    "AND archived=0 LIMIT 1;",
                    (evidence_id,),
                ).fetchone()
            finally:
                conn.close()
        if exists is not None:
            return {"ok": True, "status": "idempotent", "evidence_id": evidence_id}

        topic_labels = [
            str(value or "").strip()
            for value in (card.get("topic_labels", []) if isinstance(card.get("topic_labels", []), list) else [])
            if str(value or "").strip()
        ][:6]
        plan_user_text = f"记住，{evidence}"
        proposal_meta = {
            "latest_user_msg": plan_user_text,
            "semantic_facts": [fact],
            "turn_fact_card": {
                "fact": fact,
                "salience": salience,
                "confidence": confidence,
                "topic_labels": topic_labels,
            },
            "memory_plan": {"mode_scope": "normal", "source_kind": "proactive_record"},
            "proactive_memory_record": {
                "trigger_kind": trigger,
                "evidence_source": str(evidence_source or "turn_fact_card"),
                "evidence": evidence,
            },
        }
        plan = active_store.plan_turn_memory_writes(
            character,
            plan_user_text,
            "",
            event_nodes=[],
            meta=proposal_meta,
        )
        records = plan.get("records", []) if isinstance(plan, dict) else []
        expected_key = re.sub(r"\s+", "", expected_fact).rstrip("。.!！")
        accepted: list[dict] = []
        accepted_facts: set[str] = set()
        for raw in records:
            if not isinstance(raw, dict) or str(raw.get("layer", "") or "") != "semantic":
                continue
            row_payload = raw.get("payload", {}) if isinstance(raw.get("payload", {}), dict) else {}
            row_fact = str(row_payload.get("fact", "") or raw.get("text", "") or "").strip()
            if re.sub(r"\s+", "", row_fact).rstrip("。.!！") != expected_key:
                continue
            row = dict(raw)
            row["source"] = "proactive_record"
            row["evidence_id"] = evidence_id
            accepted.append(row)
            accepted_facts.add(row_fact)
        for raw in records:
            if not isinstance(raw, dict) or str(raw.get("layer", "") or "") != "collective":
                continue
            row_payload = raw.get("payload", {}) if isinstance(raw.get("payload", {}), dict) else {}
            row_fact = str(row_payload.get("fact", "") or raw.get("text", "") or "").strip()
            if row_fact not in accepted_facts:
                continue
            row = dict(raw)
            row["source"] = "proactive_record"
            row["evidence_id"] = evidence_id
            accepted.append(row)
        if not accepted:
            return {"ok": True, "status": "rejected", "rejected_reason": "policy_rejected"}
        write_plan = {
            "records": accepted,
            "suppressed": list(plan.get("suppressed", []) or []),
            "shared_payload": dict(plan.get("shared_payload", {}) or {}),
            "topic_tags": list(plan.get("topic_tags", []) or []),
        }
        write_result = active_store.append_turn_event(
            character,
            plan_user_text,
            "",
            event_nodes=[],
            meta={**proposal_meta, "memory_write_plan": write_plan},
            memory_snapshot={},
        )
        return {
            "ok": True,
            "status": "accepted",
            "evidence_id": evidence_id,
            "written_count": len(accepted),
            "write_result": write_result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "rejected_reason": f"{type(exc).__name__}:{exc}"[:160],
        }


def run_memory_tool_round(
    character: str,
    user_text: str,
    pack: dict | None,
    *,
    llm_caller: Callable[[str], str],
    session_id: str = "",
    activation_reason: str = "manual",
    provider: Any | None = None,
    write_enabled: bool | None = None,
    write_store: Any | None = None,
    model: str = "",
    max_calls: int | None = None,
) -> ToolRoundResult:
    del session_id  # Provenance is owned by the caller diagnostics; never sent to retrieval.
    started = time.perf_counter()
    result = ToolRoundResult(
        activated=True,
        activation_reason=str(activation_reason or "manual"),
        model=str(model or memory_turn_tools_model()),
    )
    call_limit = max_calls if max_calls is not None else memory_turn_tools_max_calls()
    allow_write = memory_turn_tools_write_enabled() if write_enabled is None else bool(write_enabled)
    prompt = _build_decision_prompt(user_text, pack, max_calls=call_limit, write_enabled=allow_write)
    try:
        result.decision_called = True
        raw = str(llm_caller(prompt) or "").strip()
    except Exception as exc:
        result.error = f"llm_caller_error:{type(exc).__name__}"
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result
    result.approx_tokens = max(1, (len(prompt) + len(raw) + 3) // 4)
    calls, rejected, parse_error = parse_tool_calls(raw, max_calls=call_limit)
    result.rejected_calls.extend(rejected)
    if parse_error:
        result.error = parse_error
        result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    if provider is None and any(call["tool"] in ("memory_recall", "memory_lookup") for call in calls):
        try:
            from tsuzuri.memory_provider_registry import get_active_memory_provider

            provider = get_active_memory_provider()
        except Exception as exc:
            result.error = f"provider_unavailable:{type(exc).__name__}"
            result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
            return result

    for call in calls:
        call_started = time.perf_counter()
        tool = str(call.get("tool", "") or "")
        args = dict(call.get("args", {}) or {})
        try:
            if tool in ("memory_recall", "memory_lookup"):
                records, trace = _execute_read_tool(tool, args, character, provider=provider)
            else:
                records, trace = _execute_propose_write(
                    args,
                    character,
                    user_text,
                    write_enabled=allow_write,
                    store=write_store,
                )
        except Exception as exc:
            records = []
            trace = {"status": "error", "rejected_reason": f"{type(exc).__name__}:{exc}"[:160]}
        result.records.extend(records)
        result.calls.append({
            "tool": tool,
            "args": args,
            **trace,
            "latency_ms": round((time.perf_counter() - call_started) * 1000.0, 2),
        })
    result.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return result


def run_memory_tool_round_if_active(
    character: str,
    user_text: str,
    pack: dict | None,
    *,
    llm_caller: Callable[[str], str],
    session_id: str,
    is_guest: bool,
    channel_kind: str,
    provider: Any | None = None,
    write_store: Any | None = None,
    model: str = "",
) -> ToolRoundResult:
    if not memory_turn_tools_enabled():
        return ToolRoundResult(activation_reason="feature_disabled")
    active, reason = should_activate_tool_round(
        user_text,
        pack,
        is_guest=is_guest,
        channel_kind=channel_kind,
    )
    if not active:
        return ToolRoundResult(activation_reason=reason)
    return run_memory_tool_round(
        character,
        user_text,
        pack,
        llm_caller=llm_caller,
        session_id=session_id,
        activation_reason=reason,
        provider=provider,
        write_store=write_store,
        model=model,
    )


__all__ = [
    "TOOL_SCHEMAS",
    "ToolRoundResult",
    "memory_turn_tools_enabled",
    "memory_turn_tools_max_calls",
    "memory_turn_tools_model",
    "memory_turn_tools_write_enabled",
    "mem_entity_proactive_enabled",
    "parse_tool_calls",
    "propose_proactive_memory_record",
    "run_memory_tool_round",
    "run_memory_tool_round_if_active",
    "should_activate_tool_round",
]
