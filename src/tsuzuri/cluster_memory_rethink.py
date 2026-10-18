"""Parent-only cluster editing. Production records shadow proposals, never applies them."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import date
from typing import Callable

from tsuzuri.cluster_memory import (
    NARRATIVE_MAX_CHARS,
    NARRATIVE_MIN_CHARS,
    _extract_json_object,
    select_cluster_rethink_candidates,
)
from tsuzuri.prompts.cluster_rethink_prompt import VERSION, build_cluster_rethink_prompt, build_cluster_edit_audit_prompt
from tsuzuri.prompts.memory_rethink_prompt import render_canon_summary
from tsuzuri.secrets_loader import env_bool
from tsuzuri.time_source import utc_now_naive
from tsuzuri.persona_registry import is_agent

TYPES = ("event_cluster", "topic_cluster")
PARENT_FIELDS = ("item_id", "character", "layer", "item_type", "text", "payload_json", "member_ids_json", "cluster_meta_json", "topic_labels_json", "status", "archived", "confidence")
MEMBER_FIELDS = ("item_id", "character", "layer", "item_type", "text", "payload_json", "cue_json", "evidence_id", "created_at", "status", "archived", "cluster_memory_parent_id")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def cluster_rethink_enabled() -> bool:
    return env_bool("MEM_CLUSTER_MEMORY_ENABLE", False) and env_bool("MEM_CLUSTER_RETHINK_ENABLE", False)


def load_cluster_edit_snapshot(core, character: str, item_id: str) -> dict:
    """Load the parent and every member in a single consistent read transaction."""
    with core._lock:
        conn = core._connect()
        try:
            conn.execute("BEGIN")
            row = conn.execute("SELECT * FROM memory_items WHERE item_id=? AND character=?", (item_id, character)).fetchone()
            if row is None or row["item_type"] not in TYPES or row["layer"] != "semantic" or row["status"] != "active" or row["archived"]:
                raise ValueError("parent_not_editable")
            parent = {key: row[key] for key in PARENT_FIELDS}
            ids = json.loads(parent["member_ids_json"])
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
                raise ValueError("invalid_member_provenance")
            if len(ids) > 100:
                raise ValueError("complete_evidence_exceeds_budget")
            members = []
            for mid in ids:
                member = conn.execute("SELECT * FROM memory_items WHERE item_id=?", (mid,)).fetchone()
                if member is None or member["character"] != character or member["item_type"] in TYPES or member["layer"] not in {"episodic", "semantic"}:
                    raise ValueError("missing_or_foreign_member")
                if member["status"] != "active" or member["archived"]:
                    raise ValueError("inactive_member_requires_review")
                if member["cluster_memory_parent_id"] != item_id:
                    raise ValueError("membership_mismatch")
                members.append({key: member[key] for key in MEMBER_FIELDS})
            return {"parent": parent, "members": members}
        finally:
            conn.rollback()
            conn.close()


def validate_cluster_edit(action: dict, snapshot: dict) -> None:
    if not isinstance(action, dict) or action.get("finish_edits") is not True:
        raise ValueError("unfinished_or_invalid_action")
    if set(action) != {"finish_edits", "action_type", "target_item_id", "evidence_item_ids", "reason", "replacement"}:
        raise ValueError("unexpected_action_fields")
    if action["target_item_id"] != snapshot["parent"]["item_id"]:
        raise ValueError("parent_only_target")
    op = action["action_type"]
    if op not in {"rewrite", "supersede", "archive", "no_change"}:
        raise ValueError("unsupported_action")
    evidence = action["evidence_item_ids"]
    known = {m["item_id"] for m in snapshot["members"]}
    if not isinstance(evidence, list) or any(not isinstance(i, str) or i not in known for i in evidence):
        raise ValueError("evidence_outside_snapshot")
    if not isinstance(action["reason"], str) or not action["reason"].strip() or len(action["reason"]) > 500:
        raise ValueError("reason_required")
    if op != "no_change" and not evidence:
        raise ValueError("edit_requires_evidence")
    new = action["replacement"]
    if op in {"no_change", "archive"}:
        if new is not None:
            raise ValueError("unexpected_replacement")
        return
    fields = {"text", "title", "capsule", "timeline", "people", "places", "start_at", "end_at"}
    if not isinstance(new, dict) or set(new) != fields:
        raise ValueError("complete_replacement_required")
    for key, low, high in (("text", NARRATIVE_MIN_CHARS, NARRATIVE_MAX_CHARS), ("title", 1, 30), ("capsule", 1, 60)):
        if not isinstance(new[key], str) or not low <= len(new[key].strip()) <= high:
            raise ValueError(f"invalid_{key}_budget")
    for key in ("people", "places"):
        if not isinstance(new[key], list) or len(new[key]) > 10 or any(not isinstance(v, str) or not 1 <= len(v) <= 40 for v in new[key]):
            raise ValueError(f"invalid_{key}")
    for key in ("start_at", "end_at"):
        if not isinstance(new[key], str) or len(new[key]) != 10:
            raise ValueError("invalid_dates")
        date.fromisoformat(new[key])
    if new["start_at"] > new["end_at"]:
        raise ValueError("reversed_dates")
    if not isinstance(new["timeline"], list) or not 1 <= len(new["timeline"]) <= 8:
        raise ValueError("invalid_timeline")
    last = new["start_at"]
    for entry in new["timeline"]:
        if not isinstance(entry, dict) or set(entry) != {"date", "detail", "significance"}:
            raise ValueError("invalid_timeline_entry")
        date.fromisoformat(entry["date"])
        if not last <= entry["date"] <= new["end_at"]:
            raise ValueError("timeline_outside_range")
        last = entry["date"]
        if any(not isinstance(entry[k], str) or not 1 <= len(entry[k]) <= 80 for k in ("detail", "significance")):
            raise ValueError("invalid_timeline_text")


def project_cluster_edit(snapshot: dict, action: dict) -> dict:
    """Produce a complete shadow revision. No database mutation is performed."""
    validate_cluster_edit(action, snapshot)
    row = copy.deepcopy(snapshot["parent"])
    op = action["action_type"]
    if op == "no_change":
        return row
    meta = json.loads(row["cluster_meta_json"])
    meta["rethink"] = {"version": VERSION, "operation": op, "before_hash": _hash(snapshot), "evidence_ids": action["evidence_item_ids"], "reason": action["reason"]}
    row["cluster_meta_json"] = _json(meta)
    if op == "archive":
        row.update(status="archived_by_cluster_rethink", archived=1)
    else:
        new = action["replacement"]
        payload = json.loads(row["payload_json"])
        payload.update({key: value for key, value in new.items() if key != "text"})
        row.update(text=new["text"], payload_json=_json(payload))
        # Generic memory writes rebuild indexes from the new text on replay.
        row.update(embedding_text="", embedding_sparse_text="", embedding_model_id="", embedding_dim=0)
    return row


def _call(system: str, user: str, caller: Callable | None) -> dict:
    if caller is not None:
        raw = caller(system, user)
    else:
        from tsuzuri.memory_rethink_engine import _call_rethink_llm
        raw, meta = _call_rethink_llm(system, user)
        if not raw:
            raise ValueError(meta.get("fallback_reason") or "llm_unavailable")
    if isinstance(raw, str):
        raw = _extract_json_object(raw)
    if not isinstance(raw, dict):
        raise ValueError("invalid_json_object")
    return raw


def run_cluster_rethink_shadow(core, character: str, *, llm_caller=None, auditor=None,
                               max_items: int = 3, enabled: bool | None = None) -> dict:
    """Review each changed evidence snapshot once; failed proposals remain retryable."""
    result = {"status": "disabled", "shadow": True, "llm_calls": 0, "records": []}
    if not (cluster_rethink_enabled() if enabled is None else enabled):
        return result
    if not is_agent(character):
        result["status"] = "invalid_scope"
        return result
    result["status"] = "complete"
    candidates = select_cluster_rethink_candidates(core, character, max_items=200)
    profile = core.build_character_canon_profile(character)
    if not isinstance(profile, dict) or not profile.get("anchors"):
        result["status"] = "canon_unavailable"
        return result
    canon = render_canon_summary(profile)
    reviewed = 0
    for candidate in candidates:
        if reviewed >= max(0, min(10, max_items)):
            break
        iid = candidate["item_id"]
        record = {"item_id": iid, "status": "rejected"}
        try:
            snapshot = load_cluster_edit_snapshot(core, character, iid)
            fingerprint = _hash({"version": VERSION, "snapshot": snapshot, "canon": canon})
            key = f"cluster_rethink.shadow.{character}.{iid}.{fingerprint}"
            previous = core.get_meta_state_json(key, {})
            if isinstance(previous, dict) and previous.get("status") == "validated":
                record["status"] = "unchanged_evidence"
                result["records"].append(record)
                continue
            record.update(snapshot=snapshot, fingerprint=fingerprint, trigger="unreviewed_evidence")
            system, user = build_cluster_rethink_prompt(snapshot, canon)
            record["prompt"] = {"system": system, "user": user}
            reviewed += 1
            result["llm_calls"] += 1
            action = _call(system, user, llm_caller)
            record["action"] = action
            try:
                validate_cluster_edit(action, snapshot)
            except ValueError as validation_error:
                if str(validation_error) not in {"invalid_text_budget", "invalid_title_budget", "invalid_capsule_budget", "complete_replacement_required"}:
                    raise
                record["first_action"] = action
                feedback = "\n上次输出未通过结构校验：" + str(validation_error) + "。请重新对照原始完整证据输出合规对象，保留所有关键事实。"
                record["repair_feedback"] = feedback
                result["llm_calls"] += 1
                action = _call(system + feedback, user, llm_caller)
                record["action"] = action
                validate_cluster_edit(action, snapshot)
            if action["action_type"] != "no_change":
                audit_system, audit_user = build_cluster_edit_audit_prompt(snapshot, action)
                audit = _call(audit_system, audit_user, auditor)
                record["audit"] = audit
                if audit.get("verdict") != "faithful" or not audit.get("reason"):
                    raise ValueError("edit_not_supported_by_audit")
            if load_cluster_edit_snapshot(core, character, iid) != snapshot:
                raise ValueError("stale_snapshot")
            record.update(status="validated", shadow_after=project_cluster_edit(snapshot, action))
        except Exception as exc:
            record.update(status="rejected", reason=str(exc))
        record["recorded_at"] = utc_now_naive().isoformat()
        result["records"].append(record)
        # Only the dedicated audit key is written. Generic rethink cannot apply it.
        if "fingerprint" in record:
            core.set_meta_state_json(f"cluster_rethink.shadow.{character}.{iid}.{record['fingerprint']}", record)
    return result
