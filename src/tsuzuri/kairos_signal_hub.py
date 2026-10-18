from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tsuzuri.kairos_state import snapshot_kairos_session
from tsuzuri.memory_core import get_memory_core
from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH
from tsuzuri.runtime_state import get_runtime_state_store
from tsuzuri.persona_registry import resolve_agent


def _get_memory_core_store():
    return get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_text(value) -> str:
    return " ".join(str(value or "").split()).strip()


def _shorten(value, limit: int) -> str:
    text = _normalize_text(value)
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + "..."


def _string_list(value, *, limit: int = 0) -> list[str]:
    rows = value if isinstance(value, list) else []
    out: list[str] = []
    for item in rows:
        text = _normalize_text(item)
        if text:
            out.append(text)
    if limit > 0:
        return out[:limit]
    return out


def _normalize_note_pool(value, *, limit: int = 8) -> list[dict[str, object]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, object]] = []
    for row in rows:
        if isinstance(row, str):
            text = _normalize_text(row)
            if not text:
                continue
            out.append(
                {
                    "note": text[:64],
                    "axis": "general",
                    "direction": "stable",
                    "strength": 0.15,
                    "support_count": 1.0,
                    "updated_at": "",
                }
            )
        elif isinstance(row, dict):
            text = _normalize_text(row.get("note", row.get("text", "")))
            if not text:
                continue
            out.append(
                {
                    "note": text[:64],
                    "axis": _normalize_text(row.get("axis", "general"))[:32] or "general",
                    "direction": _normalize_text(row.get("direction", "stable"))[:32] or "stable",
                    "strength": round(_safe_float(row.get("strength", row.get("support", 0.15)), 0.15), 3),
                    "support_count": round(max(0.0, _safe_float(row.get("support_count", row.get("support", 1.0)), 1.0)), 3),
                    "updated_at": _normalize_text(row.get("updated_at", row.get("last_seen", "")))[:40],
                }
            )
        if limit > 0 and len(out) >= limit:
            break
    return out


def _maybe_dict(value) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _maybe_list_of_dicts(value) -> list[dict[str, object]]:
    rows = value if isinstance(value, list) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _latest_dict(rows: list[dict[str, object]]) -> dict[str, object]:
    return dict(rows[0]) if rows else {}


def _extract_voice_tone_constraints(durable_character) -> list[str]:
    """C 阶段：从 durable_character.voice_tone_summary 提取可注入 prompt 的约束行。
    voice_tone_summary 格式：
        {tone_observation: str, voice_anchors: list[str], drift_pattern: str, updated_at: str}
    """
    import os as _os
    if (_os.getenv("KAIROS_VOICE_TONE_EVOLUTION", "1") or "1").strip().lower() in ("0", "false", "no"):
        return []
    if not isinstance(durable_character, dict):
        return []
    vts = durable_character.get("voice_tone_summary", {})
    if not isinstance(vts, dict):
        return []
    lines: list[str] = []
    obs = str(vts.get("tone_observation", "") or "").strip()
    anchors = vts.get("voice_anchors", []) or []
    drift = str(vts.get("drift_pattern", "") or "").strip()
    if obs:
        lines.append(f"【近期语气总结】{obs[:120]}")
    if isinstance(anchors, list) and anchors:
        anchors_str = "、".join(str(a)[:40] for a in anchors[:3] if a)
        if anchors_str:
            lines.append(f"维持语气锚点：{anchors_str}")
    if drift:
        lines.append(f"避免最近偏移：{drift[:80]}")
    return lines[:3]


def _normalize_guardrails(value) -> list[str]:
    rows = value if isinstance(value, list) else []
    out: list[str] = []
    for row in rows:
        if isinstance(row, str):
            text = _normalize_text(row)
        elif isinstance(row, dict):
            text = _normalize_text(row.get("text", row.get("note", row.get("summary", ""))))
        else:
            text = ""
        if text:
            out.append(text[:96])
    return out


def _build_canon_risk_summary(
    latest_ooc: dict[str, object],
    *,
    canon_alignment_score: float | None,
) -> str:
    if latest_ooc:
        ooc_score = round(_safe_float(latest_ooc.get("ooc_score", 0.0), 0.0), 3)
        recommendation = _normalize_text(
            _maybe_dict(latest_ooc.get("penalty_recommendation", {})).get("recommendation", "")
        )
        parts = [f"recent OOC risk {ooc_score:.3f}"]
        if canon_alignment_score is not None:
            parts.append(f"canon alignment {canon_alignment_score:.3f}")
        if recommendation:
            parts.append(f"recommendation: {recommendation}")
        return "; ".join(parts)
    if canon_alignment_score is not None:
        return f"canon alignment {canon_alignment_score:.3f}"
    return ""


@dataclass(frozen=True)
class KairosSignalRequest:
    session_id: str
    channel_kind: str = ""
    character: str = ""
    memory_snapshot: dict[str, object] = field(default_factory=dict)
    runtime_snapshot: dict[str, object] = field(default_factory=dict)
    kairos_snapshot: dict[str, object] = field(default_factory=dict)
    canon_profile: dict[str, object] = field(default_factory=dict)
    recent_ooc_audits: list[dict[str, object]] = field(default_factory=list)
    recent_evolution_runs: list[dict[str, object]] = field(default_factory=list)
    durable_session: dict[str, object] = field(default_factory=dict)
    durable_character: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KairosSignalBundle:
    session_id: str
    channel_kind: str
    character: str
    behavior_guardrails: dict[str, object] = field(default_factory=dict)
    quality_feedback: dict[str, object] = field(default_factory=dict)
    canon_feedback: dict[str, object] = field(default_factory=dict)
    memory_management: dict[str, object] = field(default_factory=dict)
    review_flags: list[str] = field(default_factory=list)


def build_kairos_signal_bundle(request: KairosSignalRequest) -> KairosSignalBundle:
    session_id = _normalize_text(request.session_id) or "default"
    channel_kind = _normalize_text(request.channel_kind)
    character = resolve_agent(_normalize_text(request.character))
    core = _get_memory_core_store()

    memory_snapshot = _maybe_dict(request.memory_snapshot) or _maybe_dict(core.load_json_snapshot())
    runtime_snapshot = _maybe_dict(request.runtime_snapshot) or _maybe_dict(get_runtime_state_store().snapshot(session_id))
    kairos_snapshot = _maybe_dict(request.kairos_snapshot) or _maybe_dict(snapshot_kairos_session(session_id))
    canon_profile = _maybe_dict(request.canon_profile) or _maybe_dict(core.build_character_canon_profile(character))
    recent_ooc_audits = _maybe_list_of_dicts(request.recent_ooc_audits) or _maybe_list_of_dicts(core.get_recent_ooc_audits(character, limit=5))
    recent_evolution_runs = _maybe_list_of_dicts(request.recent_evolution_runs) or _maybe_list_of_dicts(core.get_recent_evolution_runs(character, limit=5))
    durable_session = _maybe_dict(request.durable_session) or _maybe_dict(core.get_domain_json(f"kairos.session.{session_id}", {}))
    durable_character = _maybe_dict(request.durable_character) or _maybe_dict(core.get_domain_json(f"kairos.character.{character}", {}))

    self_eval_stats = _maybe_dict(memory_snapshot.get("self_eval_stats", {}))
    self_eval_log = _maybe_list_of_dicts(memory_snapshot.get("self_eval_log", []))
    self_eval_exemplars = _maybe_list_of_dicts(memory_snapshot.get("self_eval_exemplars", []))
    character_states = _maybe_dict(memory_snapshot.get("character_states", {}))
    character_state = _maybe_dict(character_states.get(character, {}))
    evolved_traits = _maybe_dict(character_state.get("evolved_traits", {}))

    injected_constraints = _string_list(self_eval_stats.get("injected_constraints", []), limit=5)
    # C 阶段：voice_tone_summary 注入（来自 durable_character、KAIROS dream 写入）
    _voice_tone_lines = _extract_voice_tone_constraints(durable_character)
    if _voice_tone_lines:
        injected_constraints = injected_constraints + _voice_tone_lines
    note_pool = _normalize_note_pool(evolved_traits.get("note_pool", []), limit=8)
    drift_axes = {
        str(key): round(_safe_float(value, 0.0), 3)
        for key, value in _maybe_dict(evolved_traits.get("drift_axes", {})).items()
        if str(key).strip()
    }
    canon_alignment_score = None
    if "canon_alignment_score" in self_eval_stats:
        canon_alignment_score = round(_safe_float(self_eval_stats.get("canon_alignment_score", 0.0), 0.0), 3)

    latest_eval = _latest_dict(list(reversed(self_eval_log)))
    latest_ooc = _latest_dict(recent_ooc_audits)
    latest_evolution = _latest_dict(recent_evolution_runs)
    canon_claims = _maybe_list_of_dicts(canon_profile.get("claims", []))
    canon_anchors = _maybe_dict(canon_profile.get("anchors", {}))

    filtered_exemplars: list[dict[str, object]] = []
    for exemplar in self_eval_exemplars:
        exemplar_character = _normalize_text(exemplar.get("character", ""))
        if exemplar_character and exemplar_character != character:
            continue
        filtered_exemplars.append(
            {
                "character": exemplar_character or character,
                "date": _normalize_text(exemplar.get("date", ""))[:20],
                "overall": round(_safe_float(exemplar.get("overall", 0.0), 0.0), 3),
                "user_snippet": _shorten(exemplar.get("user_snippet", ""), 80),
                "response_snippet": _shorten(exemplar.get("response_snippet", ""), 120),
            }
        )
        if len(filtered_exemplars) >= 4:
            break

    canon_risk_summary = _build_canon_risk_summary(latest_ooc, canon_alignment_score=canon_alignment_score)
    behavior_guardrails = {
        "injected_constraints": injected_constraints,
        "note_pool_hints": note_pool,
        "drift_axes": drift_axes,
        "durable_guardrails": _normalize_guardrails(durable_character.get("current_guardrails", [])),
        "canon_risk": {
            "summary": canon_risk_summary,
            "latest_ooc_score": round(_safe_float(latest_ooc.get("ooc_score", 0.0), 0.0), 3) if latest_ooc else 0.0,
            "canon_alignment_score": canon_alignment_score,
            "high_conflict_claims": list(latest_ooc.get("high_conflict_claims", []) if isinstance(latest_ooc.get("high_conflict_claims", []), list) else []),
        },
    }
    quality_feedback = {
        "stats": {
            "total_evals": int(_safe_float(self_eval_stats.get("total_evals", 0), 0.0)),
            "recent_overall_ema": round(_safe_float(self_eval_stats.get("recent_overall_ema", 0.0), 0.0), 3),
            "recent_overall_trend": round(_safe_float(self_eval_stats.get("recent_overall_trend", 0.0), 0.0), 3),
            "recent_ooc_rate": round(_safe_float(self_eval_stats.get("recent_ooc_rate", 0.0), 0.0), 3),
            "canon_penalty_load": round(_safe_float(self_eval_stats.get("canon_penalty_load", 0.0), 0.0), 3),
            "canon_alignment_score": canon_alignment_score,
        },
        "latest_eval": latest_eval,
        "exemplars": filtered_exemplars,
    }
    canon_feedback = {
        "canon_profile": {
            "claim_count": len(canon_claims),
            "anchor_types": sorted(str(key) for key in canon_anchors.keys() if str(key).strip()),
            "updated_at": _normalize_text(canon_profile.get("updated_at", ""))[:40],
        },
        "recent_ooc_audits": recent_ooc_audits[:5],
        "latest_ooc_audit": latest_ooc,
        "canon_alignment_score": canon_alignment_score,
        "canon_risk_summary": canon_risk_summary,
    }
    memory_management = {
        "recent_evolution_runs": recent_evolution_runs[:5],
        "latest_evolution": latest_evolution,
        "durable_session": durable_session,
        "durable_character": durable_character,
        "runtime_snapshot": runtime_snapshot,
        "kairos_snapshot": kairos_snapshot,
        "style_exemplars": filtered_exemplars[:3],
    }

    review_flags: list[str] = []
    if not self_eval_stats:
        review_flags.append("missing_self_eval_stats")
    if not injected_constraints:
        review_flags.append("missing_injected_constraints")
    if not note_pool and not drift_axes:
        review_flags.append("missing_evolved_traits")
    if not canon_claims:
        review_flags.append("missing_canon_claims")
    if not recent_ooc_audits:
        review_flags.append("missing_ooc_audits")
    if not recent_evolution_runs:
        review_flags.append("missing_evolution_runs")
    if not filtered_exemplars:
        review_flags.append("missing_exemplars")
    if not durable_session:
        review_flags.append("missing_durable_session")
    if not durable_character:
        review_flags.append("missing_durable_character")

    return KairosSignalBundle(
        session_id=session_id,
        channel_kind=channel_kind or str(runtime_snapshot.get("channel_kind", "") or ""),
        character=character,
        behavior_guardrails=behavior_guardrails,
        quality_feedback=quality_feedback,
        canon_feedback=canon_feedback,
        memory_management=memory_management,
        review_flags=sorted(set(review_flags)),
    )


__all__ = [
    "KairosSignalRequest",
    "KairosSignalBundle",
    "build_kairos_signal_bundle",
]
