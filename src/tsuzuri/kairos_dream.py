from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from tsuzuri import memory_host as memory_host
from tsuzuri.kairos_signal_hub import KairosSignalRequest, build_kairos_signal_bundle
from tsuzuri.kairos_state import ensure_kairos_session, get_kairos_state_store, record_kairos_dream, snapshot_kairos_session
from tsuzuri.memory_core import get_memory_core
from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH
from tsuzuri.runtime_state import get_runtime_state_store
from tsuzuri.secrets_loader import sync_env_from_secrets
from tsuzuri.persona_registry import collective_scope, resolve_agent
from tsuzuri.time_source import WORLD_TZ, now_world, utc_now


logger = logging.getLogger(__name__)


def _require_hook(name: str):
    """Return host hook ``name``; raise ImportError when absent (same path as a missing app module)."""
    fn = memory_host.get(name)
    if fn is None:
        raise ImportError(f"memory host hook {name!r} not registered")
    return fn


@dataclass(frozen=True)
class KairosBriefRequest:
    """Dream-cycle brief request handed to the ``build_dream_brief`` host hook (duck-typed)."""

    session_id: str
    channel_kind: str
    character: str
    history: list[dict[str, object]] = field(default_factory=list)
    mode_flags: dict[str, object] = field(default_factory=dict)
    session_context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class KairosBriefResult:
    """Empty brief used when no ``build_dream_brief`` hook is registered."""

    brief_text: str = ""
    summary: dict[str, object] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    source_usage: dict[str, object] = field(default_factory=dict)


def build_kairos_brief(request: KairosBriefRequest):
    """World/schedule brief via host hook ``build_dream_brief``; empty brief when absent."""
    fn = memory_host.get("build_dream_brief")
    if fn is None:
        return KairosBriefResult()
    return fn(request)


def _utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def _kairos_logical_day_world(now: datetime | None = None) -> str:
    """Return the KAIROS day using the shared 02:30 WORLD_TZ rollover."""
    world = WORLD_TZ
    current = now or utc_now()
    local = current.astimezone(world)
    if (local.hour, local.minute) < (2, 30):
        local -= timedelta(days=1)
    return local.strftime("%Y-%m-%d")


def _get_memory_core_store():
    return get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)


def _load_recent_working_memory_texts(character: str, *, limit: int = 20) -> list[str]:
    """Load only active working rows for the user-profile consolidation prompt."""
    from tsuzuri.sqlite_reliability import connect_sqlite

    conn = connect_sqlite(str(MEMORY_CORE_DB_PATH), timeout=10)
    try:
        rows = conn.execute(
            """
            SELECT text FROM memory_items
            WHERE character=? AND layer='working'
              AND archived=0 AND status='active'
            ORDER BY created_at DESC LIMIT ?;
            """,
            (str(character or "").strip(), max(1, min(100, int(limit or 20)))),
        ).fetchall()
        return [str(row[0] or "")[:80] for row in rows if row[0]]
    finally:
        conn.close()


def _llm_response_diagnostics(response) -> str:
    """Return non-content metadata for diagnosing structured LLM calls."""
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    reasoning = str(
        getattr(message, "reasoning_content", "")
        or getattr(message, "reasoning", "")
        or ""
    )
    return (
        f"finish_reason={str(getattr(choice, 'finish_reason', '') or '?')!r}; "
        f"usage={getattr(response, 'usage', None)!s}; reasoning_chars={len(reasoning)}"
    )


def _run_forgetting_lifecycle(core, character: str) -> dict:
    """Run the gated, once-per-KAIROS-day Phase E/P4 lifecycle job."""
    from tsuzuri.memory_core import mem_forgetting_apply_enabled, mem_forgetting_enabled

    if not mem_forgetting_enabled():
        return {"ok": True, "status": "disabled", "dry_run": True}
    logical_day = _kairos_logical_day_world()
    meta_key = f"kairos.forgetting.{str(character or '').strip()}.{logical_day}"
    previous = core.get_meta_state_json(meta_key, {}) or {}
    if isinstance(previous, dict) and previous.get("terminal"):
        return {
            "ok": True,
            "status": "idempotent",
            "dry_run": bool(previous.get("dry_run", True)),
            "meta_key": meta_key,
            "frozen_applied": 0,
            "tombstone_applied": 0,
        }

    try:
        frozen_days = float(os.getenv("MEM_FORGETTING_FROZEN_DAYS", "90") or "90")
    except (TypeError, ValueError):
        frozen_days = 90.0
    try:
        tombstone_days = float(os.getenv("MEM_FORGETTING_TOMBSTONE_DAYS", "180") or "180")
    except (TypeError, ValueError):
        tombstone_days = 180.0
    result = core.advance_forgetting_lifecycle(
        character,
        frozen_min_days=frozen_days,
        tombstone_min_days=tombstone_days,
        dry_run=not mem_forgetting_apply_enabled(),
    )
    if bool(result.get("ok", False)):
        core.set_meta_state_json(
            meta_key,
            {
                "terminal": True,
                "status": str(result.get("status", "") or ""),
                "dry_run": bool(result.get("dry_run", True)),
                "frozen_candidates": int(result.get("frozen_candidates", 0) or 0),
                "frozen_applied": int(result.get("frozen_applied", 0) or 0),
                "tombstone_candidates": int(result.get("tombstone_candidates", 0) or 0),
                "tombstone_applied": int(result.get("tombstone_applied", 0) or 0),
                "completed_at": _utc_now_iso(),
            },
        )
        try:
            from tsuzuri.memory_api import get_memory_api

            get_memory_api(core).record_budget_event(
                character,
                "forgetting_advances",
                amount=(
                    int(result.get("frozen_applied", 0) or 0)
                    + int(result.get("tombstone_applied", 0) or 0)
                ),
            )
        except Exception:
            pass
    result["meta_key"] = meta_key
    return result


def _daily_continuity_coverage_meta_key(character: str) -> str:
    return f"kairos.daily_continuity_coverage.{str(character or '').strip()}"


def _run_daily_continuity_coverage_shadow(core, session_id: str, character: str) -> dict:
    """Audit expired capsule coverage and route candidates through MemoryWritePlan."""
    try:
        from tsuzuri.chat_compact import load_daily_continuity_capsules_for_character
        from tsuzuri.daily_continuity.coverage import build_shadow_memory_write_plan
        from tsuzuri.daily_continuity.resolver import resolve_continuity_items

        target_character = str(character or "").strip()
        capsules = load_daily_continuity_capsules_for_character(
            target_character,
            min_age_days=15,
            max_age_days=45,
        )
        if not capsules:
            return {"status": "no_expired_capsules", "shadow_only": True}
        coverage_key = hashlib.sha256(
            json.dumps(
                [
                    {
                        "character": target_character,
                        "date_world": capsule.get("date_world", ""),
                        "source": (capsule.get("source", {}) or {}).get("archive_sha256", ""),
                        "generator": capsule.get("generator_version", ""),
                        "items_sha256": hashlib.sha256(
                            json.dumps(
                                capsule.get("items", []) or [],
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        ).hexdigest(),
                    }
                    for capsule in capsules
                    if isinstance(capsule, dict)
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        meta_key = _daily_continuity_coverage_meta_key(target_character)
        previous = core.get_meta_state_json(meta_key, {}) or {}
        if isinstance(previous, dict) and previous.get("coverage_key") == coverage_key:
            cached = dict(previous)
            cached["status"] = "idempotent"
            cached["shadow_only"] = True
            return cached

        continuity_items: list[dict] = []
        for capsule in capsules:
            if not isinstance(capsule, dict):
                continue
            for raw_item in capsule.get("items", []) or []:
                if not isinstance(raw_item, dict):
                    continue
                item = dict(raw_item)
                item.setdefault("date_world", str(capsule.get("date_world", "") or ""))
                item.setdefault("source_character", str(capsule.get("character", "") or ""))
                continuity_items.append(item)
        resolution = resolve_continuity_items(continuity_items)
        memory_rows = core._query_items(character, limit=500, query_text="")
        shadow_candidates = build_shadow_memory_write_plan(
            resolution,
            memory_rows,
            character=character,
        )
        planned_item_ids: list[str] = []
        planned_record_count = 0
        planned_layers: dict[str, int] = {}
        for candidate in (shadow_candidates.get("records", []) or [])[:8]:
            if not isinstance(candidate, dict):
                continue
            text = str(candidate.get("text", "") or "").strip()
            item_id = str(candidate.get("evidence_id", "") or "").strip()
            if not text:
                continue
            write_plan = core.plan_turn_memory_writes(
                character,
                text,
                "",
                event_nodes=[],
                meta={
                    "semantic_facts": [text],
                    "latest_user_msg": text,
                    "memory_plan": {"mode_scope": "daily_continuity_shadow"},
                    "daily_continuity_candidate": dict(candidate.get("payload", {}) or {}),
                },
            )
            records = write_plan.get("records", []) if isinstance(write_plan, dict) else []
            planned_record_count += len(records) if isinstance(records, list) else 0
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    continue
                layer = str(record.get("layer", "") or "unknown")
                planned_layers[layer] = planned_layers.get(layer, 0) + 1
            if item_id:
                planned_item_ids.append(item_id)
        audit = dict(shadow_candidates.get("coverage_audit", {}) or {})
        payload = {
            "status": "audited",
            "shadow_only": True,
            "apply": False,
            "coverage_key": coverage_key,
            "capsule_count": len(capsules),
            "item_count": len(continuity_items),
            "covered_count": int(audit.get("covered_count", 0) or 0),
            "candidate_count": int(audit.get("candidate_count", 0) or 0),
            "memory_write_plan_item_ids": planned_item_ids,
            "memory_write_plan_record_count": planned_record_count,
            "memory_write_plan_layers": planned_layers,
            "durable_writes_performed": 0,
            "audited_at": _utc_now_iso(),
        }
        core.set_meta_state_json(meta_key, payload)
        return payload
    except Exception as exc:
        return {
            "status": "audit_failed",
            "shadow_only": True,
            "apply": False,
            "reason": f"{type(exc).__name__}: {exc}"[:500],
            "durable_writes_performed": 0,
        }


def _core_summary_meta_key(character: str, logical_day: str) -> str:
    return f"kairos.core_summary.rebuilt.{str(character or '').strip()}.{logical_day}"


def _build_core_summary_prompt(character: str, items: list[dict], profile_slice: dict) -> str:
    evidence_lines = []
    for item in items[:80]:
        text = str(item.get("text", "") or "").replace("\n", " ").strip()[:220]
        if not text:
            continue
        evidence_lines.append(
            f"- [{item.get('kind', 'fact')}/{item.get('status', 'active')}] {text}"
        )
    evidence_block = "\n".join(evidence_lines) or "（没有可用的跨日事实）"
    return (
        f"你负责维护{character}对长期对话对象的跨日核心背景摘要。\n"
        "这不是角色回复，也不是角色设定摘要。只整理下面 continuity 证据中稳定、仍有效、"
        "以后多轮都可能需要的用户事实与关系背景。\n"
        "不得复制或改写 character profile canon，不得把计划写成已发生，不得保留已 superseded/closed 的事实，"
        "不确定就省略。输出一段不超过 360 个汉字的中性事实文本，不要标题、列表或 JSON。\n\n"
        f"只读角色 profile 切片（仅用于防止主客体/canon 混淆，不得写进摘要）：\n"
        f"{json.dumps(profile_slice, ensure_ascii=False, sort_keys=True)}\n\n"
        f"已消解 continuity 证据：\n{evidence_block}"
    )


def _run_core_summary_rebuild(core, character: str, *, llm_caller=None) -> dict:
    """Rebuild one fixed-ID living summary; caller owns the enable gate."""
    logical_day = _kairos_logical_day_world()
    meta_key = _core_summary_meta_key(character, logical_day)
    previous = core.get_meta_state_json(meta_key, {}) or {}
    if isinstance(previous, dict) and previous.get("terminal"):
        return {
            "ok": True,
            "status": "idempotent",
            "item_id": str(previous.get("item_id", "") or ""),
            "tokens": 0,
            "meta_key": meta_key,
        }
    try:
        from tsuzuri.chat_compact import load_daily_continuity_capsules_for_character
        from tsuzuri.daily_continuity.resolver import resolve_continuity_items

        capsules = load_daily_continuity_capsules_for_character(
            character,
            min_age_days=0,
            max_age_days=14,
        )
        raw_items: list[dict] = []
        for capsule in capsules:
            if not isinstance(capsule, dict):
                continue
            for item in capsule.get("items", []) or []:
                if isinstance(item, dict):
                    raw_items.append(dict(item))
        resolution = resolve_continuity_items(raw_items)
        resolved_items = [
            dict(item)
            for item in (resolution.get("items", []) or [])
            if isinstance(item, dict)
            and str(item.get("status", "active") or "active") not in {
                "superseded", "resolved", "completed", "cancelled", "expired", "closed"
            }
        ]
        # Phase 4(2026-08-16):event/topic cluster 的向上胶囊作为补充证据源。
        # capsule 是设计给上游消费的一句话概括;总开关门控,无 cluster 行时零影响。
        try:
            from tsuzuri.cluster_memory import load_cluster_capsule_evidence, mem_cluster_memory_enabled

            if mem_cluster_memory_enabled():
                resolved_items.extend(load_cluster_capsule_evidence(core, character))
        except Exception as _clu_err:
            _tsz_emit(f"[KairosDream] {character}: cluster capsule evidence skipped: {_clu_err}", flush=True)
        if not resolved_items:
            return {"ok": True, "status": "no_input", "tokens": 0, "meta_key": meta_key}

        profile_slice: dict = {}
        try:
            sliced = _require_hook("agent_profile_slice")(character)
            if isinstance(sliced, dict):
                profile_slice = dict(sliced)
        except Exception:
            profile_slice = {}
        prompt = _build_core_summary_prompt(character, resolved_items, profile_slice)
        tokens = 0
        model = ""
        if callable(llm_caller):
            summary = str(llm_caller(prompt) or "").strip()
            model = "injected_caller"
        else:
            api_key = str(os.getenv("LLM_API_KEY", "") or "").strip()
            base_url = str(os.getenv("LLM_BASE_URL", "") or "").strip()
            model = str(
                os.getenv("MEM_CORE_SUMMARY_MODEL", "")
                or os.getenv("DAILY_CONTINUITY_MODEL", "")
                or "deepseek-flash"
            ).strip()
            if not (api_key and base_url and model):
                return {"ok": False, "status": "llm_unavailable", "tokens": 0, "meta_key": meta_key}
            from tsuzuri.llm_client import shared_openai_client

            client = shared_openai_client(api_key=api_key, base_url=base_url, timeout=60)
            summary = ""
            for _max_tokens in (1000, 1600):
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=_max_tokens,
                    temperature=0.2,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                usage = getattr(response, "usage", None)
                tokens += int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
                summary = str(response.choices[0].message.content or "").strip()
                if summary:
                    break
        if summary.startswith("```"):
            summary = summary.strip("`").strip()
            if summary.lower().startswith("text"):
                summary = summary[4:].strip()
        summary = " ".join(summary.split())[:2000]
        if not summary:
            return {"ok": False, "status": "empty_summary", "tokens": tokens, "meta_key": meta_key}
        evidence_ids = [str(item.get("item_id", "") or "") for item in resolved_items if item.get("item_id")]
        write = core.upsert_core_summary(
            character,
            summary,
            payload={
                "owner": "core_summary_rebuild",
                "logical_day": logical_day,
                "model": model,
                "source_item_ids": evidence_ids[:120],
                "capsule_count": len(capsules),
            },
            evidence_id=hashlib.sha256(
                json.dumps(evidence_ids, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
        )
        if not bool(write.get("ok", False)):
            return {"ok": False, "status": "write_failed", "tokens": tokens, "meta_key": meta_key}
        terminal = {
            "terminal": True,
            "item_id": str(write.get("item_id", "") or ""),
            "rebuilt_at": _utc_now_iso(),
            "logical_day": logical_day,
        }
        core.set_meta_state_json(meta_key, terminal)
        return {
            "ok": True,
            "status": "rebuilt",
            "item_id": terminal["item_id"],
            "tokens": tokens,
            "meta_key": meta_key,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "fallback_reason": f"{type(exc).__name__}:{exc}"[:200],
            "tokens": 0,
            "meta_key": meta_key,
        }


def _run_entity_rebuild(core, character: str) -> dict:
    """Rebuild per-character and collective derived entities once per KAIROS day."""
    target = str(character or "").strip()
    logical_day = _kairos_logical_day_world()
    meta_key = f"kairos.entity_rebuilt.{target}.{logical_day}"
    previous = core.get_meta_state_json(meta_key, {}) or {}
    if isinstance(previous, dict) and previous.get("terminal"):
        return {"ok": True, "status": "idempotent", "meta_key": meta_key}
    try:
        from tsuzuri.memory_entities import rebuild_entities

        character_result = rebuild_entities(core, target)
        collective_result = rebuild_entities(core, collective_scope())
        if not character_result.get("ok") or not collective_result.get("ok"):
            raise RuntimeError("entity projection rejected scope")
        result = {
            "terminal": True,
            "status": "rebuilt",
            "entity_count": int(character_result.get("entity_count", 0) or 0),
            "link_count": int(character_result.get("link_count", 0) or 0),
            "collective_entity_count": int(collective_result.get("entity_count", 0) or 0),
            "collective_link_count": int(collective_result.get("link_count", 0) or 0),
            "completed_at": _utc_now_iso(),
        }
        core.set_meta_state_json(meta_key, result)
        return {"ok": True, "meta_key": meta_key, **result}
    except Exception as exc:
        return {
            "ok": False,
            "status": "failed",
            "fallback_reason": f"{type(exc).__name__}:{exc}"[:200],
            "meta_key": meta_key,
        }


def _ensure_llm_env_from_secrets() -> None:
    """Load missing standalone KAIROS config through the shared secrets loader."""
    try:
        sync_env_from_secrets()
    except Exception as exc:
        # Standalone KAIROS must preserve its prior best-effort config behavior.
        logger.warning("KAIROS secrets-to-env sync skipped: %s", type(exc).__name__)


@dataclass(frozen=True)
class KairosDreamRequest:
    session_id: str
    channel_kind: str
    character: str
    runtime_snapshot: dict[str, object] = field(default_factory=dict)
    history_tail: list[dict[str, object]] = field(default_factory=list)
    force: bool = False


@dataclass(frozen=True)
class KairosDreamResult:
    ok: bool
    summary: dict[str, object] = field(default_factory=dict)
    domain_writes: list[str] = field(default_factory=list)
    meta_writes: list[str] = field(default_factory=list)
    refreshed_brief: dict[str, object] = field(default_factory=dict)


def _normalize_guardrail_values(signal_bundle) -> list[str]:
    behavior = signal_bundle.behavior_guardrails if isinstance(getattr(signal_bundle, "behavior_guardrails", {}), dict) else {}
    injected = behavior.get("injected_constraints", []) if isinstance(behavior.get("injected_constraints", []), list) else []
    note_pool = behavior.get("note_pool_hints", []) if isinstance(behavior.get("note_pool_hints", []), list) else []
    out: list[str] = []
    seen: set[str] = set()
    for row in injected:
        text = str(row or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text[:96])
    for row in note_pool:
        if not isinstance(row, dict):
            continue
        text = str(row.get("note", "") or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text[:96])
    return out[:6]


def _signal_bundle_summary(signal_bundle) -> dict[str, object]:
    behavior = signal_bundle.behavior_guardrails if isinstance(getattr(signal_bundle, "behavior_guardrails", {}), dict) else {}
    canon = signal_bundle.canon_feedback if isinstance(getattr(signal_bundle, "canon_feedback", {}), dict) else {}
    memory_management = signal_bundle.memory_management if isinstance(getattr(signal_bundle, "memory_management", {}), dict) else {}
    quality = signal_bundle.quality_feedback if isinstance(getattr(signal_bundle, "quality_feedback", {}), dict) else {}
    return {
        "guardrail_count": len(behavior.get("injected_constraints", []) if isinstance(behavior.get("injected_constraints", []), list) else []),
        "note_pool_count": len(behavior.get("note_pool_hints", []) if isinstance(behavior.get("note_pool_hints", []), list) else []),
        "ooc_audit_count": len(canon.get("recent_ooc_audits", []) if isinstance(canon.get("recent_ooc_audits", []), list) else []),
        "evolution_count": len(memory_management.get("recent_evolution_runs", []) if isinstance(memory_management.get("recent_evolution_runs", []), list) else []),
        "exemplar_count": len(quality.get("exemplars", []) if isinstance(quality.get("exemplars", []), list) else []),
        "review_flags": list(signal_bundle.review_flags if isinstance(signal_bundle.review_flags, list) else []),
    }


def _synthesize_voice_tone_summary(character: str, signal_bundle, history_tail: list, now_iso: str) -> dict[str, object]:
    """C 阶段：综合最近 self_eval、ooc_audit、style_audit 信号、生成 voice_tone_summary。
    输出会写入 durable_character.voice_tone_summary、然后被 kairos_signal_hub 读取注入到下次 prompt。

    drift floor 0.15：当 average drift < 0.15、不更新（防过修）。

    返回 {tone_observation, voice_anchors, drift_pattern, updated_at} 或空 dict。
    """
    import os as _os
    if (_os.getenv("KAIROS_VOICE_TONE_EVOLUTION", "1") or "1").strip().lower() in ("0", "false", "no"):
        return {}
    try:
        # 收集 signal_bundle 中的 quality + canon 反馈
        quality = signal_bundle.quality_feedback if hasattr(signal_bundle, "quality_feedback") else {}
        if not isinstance(quality, dict):
            quality = {}
        canon = signal_bundle.canon_feedback if hasattr(signal_bundle, "canon_feedback") else {}
        if not isinstance(canon, dict):
            canon = {}
        stats = quality.get("stats", {}) if isinstance(quality.get("stats", {}), dict) else {}
        latest_eval = quality.get("latest_eval", {}) if isinstance(quality.get("latest_eval", {}), dict) else {}
        latest_ooc = canon.get("latest_ooc_audit", {}) if isinstance(canon.get("latest_ooc_audit", {}), dict) else {}

        # 计算平均 drift
        ooc_score = float(latest_ooc.get("ooc_score", 0.0) or 0.0)
        eval_overall = float(latest_eval.get("overall", 0.0) or 0.0)
        recent_avg = float(stats.get("recent_avg", 0.0) or 0.0)
        avg_drift = max(0.0, ooc_score)  # ooc_score 是主要 drift signal

        if avg_drift < 0.15 and recent_avg >= 0.7:
            # 状态稳定、不更新
            return {}

        # 从 latest_eval / latest_ooc 提取 issues
        issues = []
        for key in ("issues", "drift_axes", "ooc_features"):
            val = latest_ooc.get(key) if isinstance(latest_ooc, dict) else None
            if isinstance(val, list):
                issues.extend(str(x)[:40] for x in val[:5] if x)
            elif isinstance(val, dict):
                issues.extend(f"{k}:{round(float(v),2) if isinstance(v,(int,float)) else v}" for k, v in list(val.items())[:5])

        # 用 LLM 生成 voice_tone_summary（短 prompt、温度 0.4）
        try:
            api_key = (_os.getenv("LLM_API_KEY", "") or "").strip()
            base_url = (_os.getenv("LLM_BASE_URL", "") or "").strip()
            model = (_os.getenv("LLM_MODEL", "") or "").strip()
            if not (api_key and base_url and model):
                raise ImportError("LLM not configured")
            from tsuzuri.llm_client import shared_openai_client
            client = shared_openai_client(api_key=api_key, base_url=base_url)
            history_excerpt = "\n".join(
                str(h.get("text", h)[:60]) if isinstance(h, dict) else str(h)[:60]
                for h in (history_tail[-6:] if isinstance(history_tail, list) else [])
            )
            prompt = (
                f"角色：{character}。最近对话观测到这些 drift 信号：\n"
                f"OOC 风险分：{round(ooc_score, 2)}\n"
                f"近期评估均分：{round(recent_avg, 2)}\n"
                f"问题点：{'; '.join(issues[:5]) if issues else '无明显特征'}\n"
                f"\n最近回复样本：\n{history_excerpt[:400]}\n\n"
                f"请输出严格 JSON（不加 markdown 不解释）：\n"
                f"{{\"tone_observation\":\"<一句话观察、≤40字>\","
                f"\"voice_anchors\":[\"<锚点1>\",\"<锚点2>\"],"
                f"\"drift_pattern\":\"<漂移模式描述、≤40字>\"}}"
            )
            text = ""
            for _max_tokens in (400, 800):
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=_max_tokens,
                    temperature=0.4,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                text = (resp.choices[0].message.content or "").strip()
                if text:
                    break
            # 尝试解析 JSON（去 markdown 包装）
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            import json as _json
            parsed = _json.loads(text.strip())
            if isinstance(parsed, dict) and parsed.get("tone_observation"):
                return {
                    "tone_observation": str(parsed.get("tone_observation", ""))[:200],
                    "voice_anchors": [str(a)[:60] for a in (parsed.get("voice_anchors", []) if isinstance(parsed.get("voice_anchors", []), list) else [])][:3],
                    "drift_pattern": str(parsed.get("drift_pattern", ""))[:200],
                    "updated_at": now_iso,
                    "ooc_score": round(ooc_score, 3),
                    "recent_avg": round(recent_avg, 3),
                }
        except Exception as _e:
            _tsz_emit(f"[VoiceToneSummary] LLM failed: {_e}, fallback to rule-based", flush=True)

        # 规则版 fallback
        tone_obs_parts = []
        if avg_drift > 0.3:
            tone_obs_parts.append(f"OOC 偏高 ({round(avg_drift, 2)})")
        if recent_avg < 0.6 and recent_avg > 0:
            tone_obs_parts.append(f"近期评分偏低 ({round(recent_avg, 2)})")
        tone_obs = "、".join(tone_obs_parts) or f"语气稳定 (drift {round(avg_drift, 2)})"

        return {
            "tone_observation": tone_obs[:200],
            "voice_anchors": ["保持角色 SSOT 风格", "短句优先 + 不炫技"],
            "drift_pattern": "; ".join(issues[:3])[:200] if issues else "",
            "updated_at": now_iso,
            "ooc_score": round(ooc_score, 3),
            "recent_avg": round(recent_avg, 3),
        }
    except Exception as _e:
        _tsz_emit(f"[VoiceToneSummary] failed: {_e}", flush=True)
        return {}


def _merge_character_domain(existing_value, *, session_id: str, character: str = "", summary: dict[str, object] = None, signal_bundle=None, history_tail: list = None, now_iso: str = "") -> dict[str, object]:
    payload = dict(existing_value if isinstance(existing_value, dict) else {})
    recent_sessions = payload.get("recent_sessions", [])
    if not isinstance(recent_sessions, list):
        recent_sessions = []
    recent_sessions = [str(x or "") for x in recent_sessions if str(x or "").strip()]
    if session_id in recent_sessions:
        recent_sessions.remove(session_id)
    recent_sessions.insert(0, session_id)
    payload["recent_sessions"] = recent_sessions[:8]
    payload["latest_session_id"] = session_id
    payload["last_dream_at"] = now_iso
    payload["latest_brief_summary"] = dict(summary if isinstance(summary, dict) else {})
    payload["current_guardrails"] = _normalize_guardrail_values(signal_bundle)
    payload["quality_snapshot"] = {
        "stats": dict(signal_bundle.quality_feedback.get("stats", {}) if isinstance(signal_bundle.quality_feedback, dict) and isinstance(signal_bundle.quality_feedback.get("stats", {}), dict) else {}),
        "latest_eval": dict(signal_bundle.quality_feedback.get("latest_eval", {}) if isinstance(signal_bundle.quality_feedback, dict) and isinstance(signal_bundle.quality_feedback.get("latest_eval", {}), dict) else {}),
    }
    payload["canon_snapshot"] = {
        "canon_alignment_score": signal_bundle.canon_feedback.get("canon_alignment_score", None) if isinstance(signal_bundle.canon_feedback, dict) else None,
        "canon_profile": dict(signal_bundle.canon_feedback.get("canon_profile", {}) if isinstance(signal_bundle.canon_feedback, dict) and isinstance(signal_bundle.canon_feedback.get("canon_profile", {}), dict) else {}),
        "latest_ooc_audit": dict(signal_bundle.canon_feedback.get("latest_ooc_audit", {}) if isinstance(signal_bundle.canon_feedback, dict) and isinstance(signal_bundle.canon_feedback.get("latest_ooc_audit", {}), dict) else {}),
        "canon_risk_summary": str(signal_bundle.canon_feedback.get("canon_risk_summary", "") if isinstance(signal_bundle.canon_feedback, dict) else ""),
    }
    payload["memory_review"] = {
        "latest_evolution": dict(signal_bundle.memory_management.get("latest_evolution", {}) if isinstance(signal_bundle.memory_management, dict) and isinstance(signal_bundle.memory_management.get("latest_evolution", {}), dict) else {}),
        "signal_bundle_summary": _signal_bundle_summary(signal_bundle),
    }
    payload["style_exemplars"] = list(signal_bundle.memory_management.get("style_exemplars", []) if isinstance(signal_bundle.memory_management, dict) and isinstance(signal_bundle.memory_management.get("style_exemplars", []), list) else [])
    # C 阶段：voice tone summary（drift 触发时更新、稳定时保留旧的）
    _new_vts = _synthesize_voice_tone_summary(
        character=character or str(payload.get("character", "")),
        signal_bundle=signal_bundle,
        history_tail=history_tail or [],
        now_iso=now_iso,
    )
    if _new_vts:
        payload["voice_tone_summary"] = _new_vts
        _tsz_emit(f"[VoiceToneSummary] {payload.get('character', '?')} updated: {_new_vts.get('tone_observation', '')[:60]}", flush=True)
    return payload


def run_kairos_dream_cycle(request: KairosDreamRequest) -> KairosDreamResult:
    # 2026-05-17: 独立调用（py -c）时 env 没设、自动从 secrets.toml 兜底
    _ensure_llm_env_from_secrets()
    session_id = str(request.session_id or "").strip() or "default"
    channel_kind = str(request.channel_kind or "").strip()
    character = resolve_agent(request.character)
    runtime_snapshot = dict(request.runtime_snapshot if isinstance(request.runtime_snapshot, dict) else {})
    history_tail = list(request.history_tail if isinstance(request.history_tail, list) else [])
    store = get_kairos_state_store()
    ensure_kairos_session(
        session_id,
        channel_kind=channel_kind,
        character=character,
    )
    session_snapshot = snapshot_kairos_session(session_id)
    if (not bool(session_snapshot.get("kairos_enabled", False))) and (not bool(request.force)):
        return KairosDreamResult(
            ok=False,
            summary={"reason_code": "kairos_disabled"},
            domain_writes=[],
            meta_writes=[],
            refreshed_brief={},
        )
    if not store.mark_dream_active(session_id):
        return KairosDreamResult(
            ok=False,
            summary={"reason_code": "dream_already_active"},
            domain_writes=[],
            meta_writes=[],
            refreshed_brief={},
        )

    try:
        if not runtime_snapshot:
            runtime_snapshot = get_runtime_state_store().snapshot(session_id)
        core = _get_memory_core_store()
        memory_snapshot = core.load_json_snapshot()
        signal_bundle = build_kairos_signal_bundle(
            KairosSignalRequest(
                session_id=session_id,
                channel_kind=channel_kind,
                character=character,
                memory_snapshot=memory_snapshot,
                runtime_snapshot=runtime_snapshot,
                kairos_snapshot=session_snapshot,
                canon_profile=core.build_character_canon_profile(character),
                recent_ooc_audits=core.get_recent_ooc_audits(character, limit=5),
                recent_evolution_runs=core.get_recent_evolution_runs(character, limit=5),
                durable_session=core.get_domain_json(f"kairos.session.{session_id}", {}),
                durable_character=core.get_domain_json(f"kairos.character.{character}", {}),
            )
        )
        brief_result = build_kairos_brief(
            KairosBriefRequest(
                session_id=session_id,
                channel_kind=channel_kind,
                character=character,
                history=history_tail,
                mode_flags=runtime_snapshot.get("mode_flags", {}) if isinstance(runtime_snapshot.get("mode_flags", {}), dict) else {},
                session_context={
                    "projection_summary": runtime_snapshot.get("last_projection_summary", {}) if isinstance(runtime_snapshot.get("last_projection_summary", {}), dict) else {},
                    "prompt_summary": runtime_snapshot.get("last_prompt_summary", {}) if isinstance(runtime_snapshot.get("last_prompt_summary", {}), dict) else {},
                    "agent_summary": runtime_snapshot.get("last_agent_summary", {}) if isinstance(runtime_snapshot.get("last_agent_summary", {}), dict) else {},
                    "continuity_hint": f"dream refresh for {channel_kind or 'session'}",
                    "last_brief_summary": session_snapshot.get("last_brief_summary", {}) if isinstance(session_snapshot.get("last_brief_summary", {}), dict) else {},
                },
            )
        )
        brief_payload = asdict(brief_result)
        now_iso = _utc_now_iso()
        signal_summary = _signal_bundle_summary(signal_bundle)

        meta_writes: list[str] = []
        domain_writes: list[str] = []

        daily_continuity_coverage = _run_daily_continuity_coverage_shadow(
            core,
            session_id,
            character,
        )
        if daily_continuity_coverage.get("status") == "audited":
            meta_writes.append(_daily_continuity_coverage_meta_key(character))

        meta_session_key = f"kairos.session.{session_id}"
        meta_session_value = {
            "session_id": session_id,
            "channel_kind": channel_kind,
            "character": character,
            "last_dream_at": now_iso,
            "turn_count": int(session_snapshot.get("turn_count", 0) or 0),
            "prompt_summary": runtime_snapshot.get("last_prompt_summary", {}) if isinstance(runtime_snapshot.get("last_prompt_summary", {}), dict) else {},
            "projection_summary": runtime_snapshot.get("last_projection_summary", {}) if isinstance(runtime_snapshot.get("last_projection_summary", {}), dict) else {},
            "agent_summary": runtime_snapshot.get("last_agent_summary", {}) if isinstance(runtime_snapshot.get("last_agent_summary", {}), dict) else {},
            "signal_summary": signal_summary,
            "review_flags": list(signal_bundle.review_flags if isinstance(signal_bundle.review_flags, list) else []),
        }
        core.set_meta_state_json(meta_session_key, meta_session_value)
        meta_writes.append(meta_session_key)

        session_domain_key = f"kairos.session.{session_id}"
        session_domain_value = {
            "session_id": session_id,
            "channel_kind": channel_kind,
            "character": character,
            "updated_at": now_iso,
            "turn_count": int(session_snapshot.get("turn_count", 0) or 0),
            "runtime_snapshot": runtime_snapshot,
            "history_tail": history_tail[-6:],
            "durable_brief": brief_payload,
            "signal_bundle_summary": signal_summary,
        }
        core.set_domain_json(session_domain_key, session_domain_value)
        domain_writes.append(session_domain_key)

        character_domain_key = f"kairos.character.{character}"
        character_domain_existing = core.get_domain_json(character_domain_key, {})
        character_domain_value = _merge_character_domain(
            character_domain_existing,
            session_id=session_id,
            character=character,
            summary=brief_result.summary,
            signal_bundle=signal_bundle,
            history_tail=history_tail,
            now_iso=now_iso,
        )
        core.set_domain_json(character_domain_key, character_domain_value)
        domain_writes.append(character_domain_key)

        # ═══════════════════════════════════════════════════════════
        # KAIROS 主动触发：演化 + 审计 + 关系更新（闭环）
        # ═══════════════════════════════════════════════════════════
        _dream_evolution_result = {}
        _dream_rethink_result = {}
        _dream_core_summary_result = {}
        _dream_forgetting_result = {}
        _dream_audit_result = {}
        _dream_relationship_update = {}
        # 2026-05-19: 在外 try 之前初始化、防外 try 任何步骤抛异常跳到 except 后、
        # L914 引用 recent_replies UnboundLocalError。原 recent_replies 在 try 内 L618 才定义。
        recent_replies: list = []
        try:
            # Rethink must see the evidence snapshot that existed before evolution
            # or maintenance can change lifecycle state.
            _rethink_evidence_ready = True
            try:
                _rethink_cursor = core.get_rethink_cursor(character)
                _rethink_evidence_snapshot = core.select_rethink_evidence(
                    character,
                    since_ts=str(_rethink_cursor.get("last_turn_ts", "") or ""),
                )
            except Exception as _evidence_err:
                _rethink_evidence_ready = False
                _rethink_evidence_snapshot = []
                _dream_rethink_result = {
                    "ok": False,
                    "status": "evidence_snapshot_failed",
                    "fallback_reason": str(_evidence_err)[:160],
                }

            # ── 1) 记忆演化：propose → apply ──
            evo = core.propose_memory_evolution(character)
            _dream_evolution_result = evo
            evo_actions = evo.get("actions", [])
            if evo_actions and not evo.get("shadow", True):
                # shadow 模式已过期 → 真正应用演化
                apply_result = core.apply_evolution_plan(run_id=evo.get("run_id"))
                _dream_evolution_result["applied"] = apply_result
                domain_writes.append(f"kairos.evolution.{character}")

            # ── 1.3) Letta-inspired bounded semantic rethink loop ──
            try:
                from tsuzuri.memory_core import mem_rethink_enabled
                _rethink_enabled = mem_rethink_enabled()
                if _rethink_enabled and _rethink_evidence_ready:
                    _rethink_day = _kairos_logical_day_world()
                    _rethink_meta_key = f"memory_rethink.last_terminal.{character}.{_rethink_day}"
                    _previous_rethink = core.get_meta_state_json(_rethink_meta_key, {}) or {}
                    if isinstance(_previous_rethink, dict) and _previous_rethink.get("terminal"):
                        _dream_rethink_result = {
                            "ok": True,
                            "status": "idempotent",
                            "shadow": bool(_previous_rethink.get("shadow", True)),
                            "run_id": str(_previous_rethink.get("run_id", "") or ""),
                        }
                    else:
                        from tsuzuri.memory_rethink_engine import propose_memory_rethink_loop
                        _dream_rethink_result = propose_memory_rethink_loop(
                            core,
                            character,
                            evidence=_rethink_evidence_snapshot,
                        )
                        if bool(_dream_rethink_result.get("ok", False)):
                            core.set_meta_state_json(
                                _rethink_meta_key,
                                {
                                    "terminal": True,
                                    "run_id": str(_dream_rethink_result.get("run_id", "") or ""),
                                    "status": str(_dream_rethink_result.get("status", "") or ""),
                                    "shadow": bool(_dream_rethink_result.get("shadow", True)),
                                    "completed_at": _utc_now_iso(),
                                },
                            )
                            # P2 预算：每 pass 的 LLM 调用与实际 action 分开计数。
                            try:
                                from tsuzuri.memory_api import get_memory_api as _get_memory_api

                                _memory_api = _get_memory_api(core)
                                for _pass_result in (_dream_rethink_result.get("pass_results", []) or []):
                                    if not isinstance(_pass_result, dict):
                                        continue
                                    _memory_api.record_budget_event(
                                        character,
                                        "rethink_loop_llm_calls",
                                        amount=1,
                                        tokens=int(_pass_result.get("llm_total_tokens", 0) or 0),
                                    )
                                _memory_api.record_budget_event(
                                    character,
                                    "rethink_actions",
                                    amount=int(_dream_rethink_result.get("actions_applied_count", 0) or 0),
                                    tokens=0,
                                )
                            except Exception:
                                pass  # 记账失败绝不阻断 KAIROS 主路径
                        if _dream_rethink_result.get("run_id"):
                            domain_writes.append(f"kairos.memory_rethink.{character}")
                elif not _rethink_enabled:
                    _dream_rethink_result = {"ok": True, "status": "disabled", "shadow": False}
            except Exception as _rethink_err:
                _dream_rethink_result = {
                    "ok": False,
                    "status": "failed",
                    "fallback_reason": str(_rethink_err)[:160],
                    "shadow": True,
                }
                _tsz_emit(f"[KAIROS][MemoryRethink] failed for {character}: {_rethink_err}", flush=True)

            # ── 1.5) Tier 2 (2026-05-04): evidence pool maintenance ──
            # 把 working_ttl + ephemeral_noise + low_value_generic + profile_covered
            # 4 类碎语清掉。原本无自动 caller、Phase A 时手动跑过一次后再没跑。
            # 现挂进 KAIROS 凌晨 dream cycle、每晚跑一次。默认只审计；显式
            # MEM_EVIDENCE_MAINTENANCE_APPLY=1 才真归档。
            try:
                _maintenance_apply = str(
                    os.getenv("MEM_EVIDENCE_MAINTENANCE_APPLY", "0") or "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                _maint_report = core.maintain_memory_evidence_pool(
                    dry_run=not _maintenance_apply, export_json=False,
                )
                _archived_summary = _maint_report.get("archived", {}) if isinstance(_maint_report, dict) else {}
                _total_archived = sum(int(v or 0) for v in _archived_summary.values() if isinstance(v, (int, float)))
                if _total_archived > 0:
                    _tsz_emit(
                        f"[KAIROS][MemMaintenance] {character}: archived {_total_archived} items "
                        f"(working_ttl={_archived_summary.get('working_ttl', 0)} "
                        f"profile_covered={_archived_summary.get('profile_covered', 0)} "
                        f"exclude={_archived_summary.get('exclude', 0)})",
                        flush=True,
                    )
                _dream_evolution_result["maintenance"] = {
                    "archived": _archived_summary,
                    "total_archived": _total_archived,
                    "dry_run": not _maintenance_apply,
                }
            except Exception as _maint_err:
                _tsz_emit(f"[KAIROS][MemMaintenance] failed for {character}: {_maint_err}", flush=True)

            # ── 1.6) Phase E/P4: archived -> frozen -> tombstone ──
            try:
                _dream_forgetting_result = _run_forgetting_lifecycle(core, character)
                if _dream_forgetting_result.get("status") not in {"disabled", "idempotent"}:
                    _tsz_emit(
                        f"[KAIROS][Forgetting] {character}: "
                        f"frozen={_dream_forgetting_result.get('frozen_applied', 0)} "
                        f"tombstone={_dream_forgetting_result.get('tombstone_applied', 0)} "
                        f"skipped={_dream_forgetting_result.get('skipped_protected', 0)} "
                        f"dry_run={_dream_forgetting_result.get('dry_run', True)}",
                        flush=True,
                    )
            except Exception as _forgetting_err:
                _dream_forgetting_result = {
                    "ok": False,
                    "status": "failed",
                    "dry_run": True,
                    "fallback_reason": str(_forgetting_err)[:160],
                }
                _tsz_emit(f"[KAIROS][Forgetting] failed for {character}: {_forgetting_err}", flush=True)

            # ── 1.7) Phase Q.X (2026-05-04): cross-character daily digest broadcast ──
            # speaker=character 与开发者对话总结、broadcast 到其他角色 episodic 层
            # 用「听 X 说：」前缀显式 secondhand 视角。idempotent via meta state last_run.
            # 各角色串行调用、每天每角色 1 次 LLM（KAIROS 凌晨集中跑）。
            #
            # 日期语义修复 (2026-05-06): KAIROS dream 窗口在凌晨 01-05 WORLD_TZ 触发、要
            # digest 的是"刚刚结束的那一天"（昨天）、不是 wall-clock 的今天——凌晨 02:00
            # 时今天才过 2 小时、working 层基本空、digest 会一直 skip。catchup 启动时
            # (mock dream_dt=02:00、real wall-clock 可能 09:00) 也是同一语义：补跑昨天。
            # 变量名仍叫 _today_world_str 避免下文大量重命名、但语义已是 "target_digest_date"。
            try:
                from datetime import timedelta as _td_world
                _now_world = now_world()
                # digest 的目标日期 = 当前 WORLD_TZ 时间的"前一天"
                _today_world_str = (_now_world - _td_world(days=1)).strftime("%Y-%m-%d")
                _digest_meta_key = f"cross_char_digest.last_run.{character}"
                _digest_meta = core.get_meta_state_json(_digest_meta_key, {}) if hasattr(core, "get_meta_state_json") else {}
                if not isinstance(_digest_meta, dict):
                    _digest_meta = {}
                _last_digest_date = str(_digest_meta.get("date", "") or "")
                if _last_digest_date == _today_world_str:
                    # 今天已经跑过、skip
                    pass
                else:
                    _digest_result = core.propose_daily_cross_char_digest(
                        speaker_char=character,
                        date_str_world=_today_world_str,
                        llm_caller=None,  # propose 内部走 env-LLM fallback
                    )
                    _written = int(_digest_result.get("written_count", 0) or 0) if isinstance(_digest_result, dict) else 0
                    if _written > 0:
                        # 持久化 last_run 标记、防同一晚多次 KAIROS tick 重跑
                        if hasattr(core, "set_meta_state_json"):
                            core.set_meta_state_json(
                                _digest_meta_key,
                                {
                                    "date": _today_world_str,
                                    "written_count": _written,
                                    "summary": str(_digest_result.get("summary", "") or "")[:120],
                                    "receivers": list(_digest_result.get("receivers", []) or []),
                                },
                            )
                        domain_writes.append(f"kairos.cross_char_digest.{character}")
                        _tsz_emit(
                            f"[KAIROS][CrossCharDigest] {character} → broadcast to "
                            f"{_digest_result.get('receivers', [])}: "
                            f"{str(_digest_result.get('summary', ''))[:60]}",
                            flush=True,
                        )
                    elif isinstance(_digest_result, dict) and _digest_result.get("skipped"):
                        # skipped 不算失败、记 last_run 避免今天反复尝试
                        if hasattr(core, "set_meta_state_json"):
                            core.set_meta_state_json(
                                _digest_meta_key,
                                {
                                    "date": _today_world_str,
                                    "written_count": 0,
                                    "skipped": _digest_result.get("skipped", ""),
                                },
                            )
                    _dream_evolution_result["cross_char_digest"] = _digest_result if isinstance(_digest_result, dict) else {}
            except Exception as _digest_err:
                _tsz_emit(f"[KAIROS][CrossCharDigest] failed for {character}: {_digest_err}", flush=True)

            # ── 1.8) Phase Q.Y (2026-05-04): 周日凌晨关系反思 ──
            # 仅周日 (isoweekday==7) dream cycle 跑、每个角色生成对用户的关系反思。
            # caller_system_prompt 走 宿主应用.build_pet_system_prompt_for_api 完整 chat 链路、
            # 防角色 voice 投射到对用户的描述。idempotent: meta state per (char, week_iso)。
            try:
                from datetime import datetime as _dt_refl
                _world_refl = WORLD_TZ
                if _world_refl is not None:
                    _today_refl = now_world()
                else:
                    from datetime import timedelta as _td_refl, timezone as _tz_refl
                    _today_refl = now_world()

                # 仅周日 (isoweekday==7) 跑反思
                if _today_refl.isoweekday() == 7:
                    _week_iso_refl = _today_refl.strftime("%G-W%V")
                    _refl_meta_key = f"relationship_reflection.last_run.{character}.{_week_iso_refl}"
                    _refl_meta = core.get_meta_state_json(_refl_meta_key, {}) if hasattr(core, "get_meta_state_json") else {}
                    if not isinstance(_refl_meta, dict):
                        _refl_meta = {}
                    if _refl_meta.get("done"):
                        # 本周已经跑过、skip
                        _dream_evolution_result["relationship_reflection"] = {"skipped": "already_done", "week": _week_iso_refl}
                    else:
                        # build 完整 chat-time system prompt（含全 21 block）
                        _refl_sys_prompt = ""
                        try:
                            _refl_sys_prompt = _require_hook("agent_system_prompt")(character) or ""
                        except Exception as _bsp_err:
                            _tsz_emit(f"[KAIROS][Reflection] build_pet_system_prompt failed for {character}: {_bsp_err}", flush=True)
                            # 不致命、propose 会 fallback 到 isolated pet_reflection compose

                        _refl_result = core.propose_weekly_relationship_reflection(
                            character, week_iso=_week_iso_refl,
                            caller_system_prompt=_refl_sys_prompt,
                        )
                        if isinstance(_refl_result, dict) and _refl_result.get("written"):
                            domain_writes.append(f"kairos.reflection.{character}")
                            _tsz_emit(
                                f"[KAIROS][Reflection] {character} (week={_week_iso_refl}) → "
                                f"{str(_refl_result.get('reflection_text', ''))[:60]}",
                                flush=True,
                            )
                        elif isinstance(_refl_result, dict) and _refl_result.get("skipped"):
                            _tsz_emit(
                                f"[KAIROS][Reflection] {character} (week={_week_iso_refl}) "
                                f"skipped: {_refl_result.get('skipped')}",
                                flush=True,
                            )
                        _dream_evolution_result["relationship_reflection"] = _refl_result if isinstance(_refl_result, dict) else {}
            except Exception as _refl_err:
                _tsz_emit(f"[KAIROS][Reflection] failed for {character}: {_refl_err}", flush=True)

            # ── 1.9) Phase Q.Y.1 (2026-05-04): 反思 → user_profile fallback process ──
            # 每天 dream cycle 扫描该 character 自己的反思、找 7+ 天未 process 的、direct process。
            # 用户回复反思走另一条 path（message_board reply hook 立即触发 process、Phase 4）、
            # 这里仅处理"超过 7 天没人回复"的 fallback 情况。
            # 一条反思 process 一次性（process_reflection_into_user_profile 内含幂等检查）。
            try:
                from tsuzuri.memory_core_impl.reflection_sink import REFLECTION_SOURCE, get_reflection_sink

                _cutoff_ts_refl = time.time() - 7 * 86400  # 7 days ago
                _refl_rows = get_reflection_sink(core.db_path).list_before(
                    author=character, source=REFLECTION_SOURCE, created_before=_cutoff_ts_refl,
                )

                _processed_count = 0
                for _refl_row in _refl_rows:
                    _refl_meta_obj = _refl_row.get("meta") or {}
                    if not isinstance(_refl_meta_obj, dict):
                        continue
                    if bool(_refl_meta_obj.get("processed_into_profile", False)):
                        continue  # 已 process（无论是用户 reply 触发还是之前 fallback）
                    # direct process (无用户 reply)
                    _proc_result = core.process_reflection_into_user_profile(
                        message_id=str(_refl_row.get("message_id", "") or ""),
                        user_reply_text="",
                    )
                    if isinstance(_proc_result, dict) and _proc_result.get("processed"):
                        _processed_count += 1
                        domain_writes.append(f"kairos.reflection_processed.{character}")
                        _tsz_emit(
                            f"[KAIROS][Reflection.Process] {character} reflection "
                            f"{str(_refl_row.get('message_id', '') or '')[:8]} fallback processed (7+ day, no user reply)",
                            flush=True,
                        )
                if _processed_count > 0:
                    _dream_evolution_result["reflection_fallback_processed"] = _processed_count
            except Exception as _scan_err:
                _tsz_emit(f"[KAIROS][Reflection.Process] scan failed for {character}: {_scan_err}", flush=True)

            # ── 2) OOC 审计：收集回复 → 审计 ──
            recent_replies = []
            try:
                eval_log = memory_snapshot.get("self_eval_log", [])
                if isinstance(eval_log, list):
                    for item in reversed(eval_log):
                        if not isinstance(item, dict):
                            continue
                        if str(item.get("character", "") or "").strip() != character:
                            continue
                        reply = str(item.get("response_snippet", "") or "").strip()
                        if reply:
                            recent_replies.append({"reply_text": reply, "source": "self_eval_log"})
                            if len(recent_replies) >= 24:
                                break
            except Exception:
                pass
            if recent_replies:
                audit = core.audit_recent_ooc_against_canon(character, recent_replies, shadow=False)
                _dream_audit_result = audit
                domain_writes.append(f"kairos.audit.{character}")
                # 应用 canon penalty 到记忆（drift_axes / note_pool / alignment_score）
                try:
                    _apply_canon_penalty_to_memory = _require_hook("apply_canon_penalty")
                    _save_dev_memory = _require_hook("save_memory_snapshot")
                    penalty = audit.get("penalty_recommendation", {})
                    if penalty and audit.get("ooc_score", 0.0) > 0:
                        memory_snapshot = _apply_canon_penalty_to_memory(memory_snapshot, character, audit, penalty)
                        _save_dev_memory(memory_snapshot)
                        domain_writes.append(f"kairos.penalty.{character}")
                except Exception:
                    pass

            # ── 3) 关系演化：基于信号更新 familiarity/trust ──
            try:
                rel_domain_key = "long_term.character_relationships"
                rel_data = core.get_domain_json(rel_domain_key, {})
                if isinstance(rel_data, dict) and character in rel_data:
                    char_rel = rel_data[character]
                    turn_count = int(session_snapshot.get("turn_count", 0) or 0)
                    # 根据对话轮数缓慢增长 familiarity
                    if turn_count > 0 and turn_count % 10 == 0:
                        old_fam = char_rel.get("familiarity", 0)
                        if old_fam < 10:
                            char_rel["familiarity"] = old_fam + 1
                    # 根据 OOC 分数调整 trust（分数低=好=加信任）
                    ooc_score = _dream_audit_result.get("ooc_score", 0.0) if _dream_audit_result else 0.0
                    if ooc_score < 0.1 and recent_replies:
                        old_trust = char_rel.get("trust", 0)
                        if old_trust < 10:
                            char_rel["trust"] = old_trust + 1
                    # 更新 conversation_count 兜底（确保不卡在0）
                    if turn_count > 0 and char_rel.get("conversation_count", 0) == 0:
                        char_rel["conversation_count"] = max(1, turn_count // 5)
                    # 2026-05-09 v4：每天 KAIROS dream cycle 跑一次 shared_topics decay。
                    # 所有 promoted topic 的 score *= 0.92、score < 0.30 降级回 candidate；
                    # 同时把已满足晋升条件的 candidate 移入 shared_topics（reinforce 路径）。
                    try:
                        from datetime import datetime as _dt_kd
                        _decay_shared_topics_daily = _require_hook("decay_shared_topics")
                        _maybe_promote_topic_candidates = _require_hook("promote_topic_candidates")
                        _now_kd = utc_now().astimezone().replace(tzinfo=None)
                        _decay_shared_topics_daily(char_rel, _now_kd)
                        _maybe_promote_topic_candidates(char_rel, _now_kd)
                    except Exception as _topic_decay_err:
                        _tsz_emit(f"[KairosDream] {character}: topic decay/promote err: {_topic_decay_err}")
                    rel_data[character] = char_rel
                    core.set_domain_json(rel_domain_key, rel_data)
                    _dream_relationship_update = char_rel
                    domain_writes.append(f"kairos.relationship.{character}")

                    # 2026-05-09 v4：把 persona_continuity[c].relation_tier 按当前 fam/trust/depth/conv
                    # 重新派生（宿主应用._pet_relation_tier_from_count）。conv 涨到阈值会自动从
                    # 初识→相熟→好友→挚交。原本 tier 只在初次构建时写、长期不更新。
                    try:
                        _pet_relation_tier_from_count = _require_hook("relation_tier")
                        _save_dm_kd = _require_hook("save_memory_snapshot")
                        _conv_for_tier = int(char_rel.get("conversation_count", 0) or 0)
                        _new_tier, _ = _pet_relation_tier_from_count(_conv_for_tier, char_rel)
                        _pc_root = memory_snapshot.setdefault("persona_continuity", {}) if isinstance(memory_snapshot, dict) else {}
                        _pc_for_char = _pc_root.setdefault(character, {}) if isinstance(_pc_root, dict) else {}
                        _old_tier = str(_pc_for_char.get("relation_tier", "") or "")
                        if _new_tier and _new_tier != _old_tier:
                            _pc_for_char["relation_tier"] = _new_tier
                            _save_dm_kd(memory_snapshot)
                            _tsz_emit(f"[KairosDream] {character}: relation_tier {_old_tier!r}→{_new_tier!r} (conv={_conv_for_tier} fam={char_rel.get('familiarity')} trust={char_rel.get('trust')})")
                            domain_writes.append(f"kairos.tier.{character}")
                    except Exception as _tier_err:
                        _tsz_emit(f"[KairosDream] {character}: relation_tier derive err: {_tier_err}")
            except Exception:
                pass

            # ── 4) 动机层演化：衰减过期目标 + 从记忆中种子初始动机 ──
            try:
                # Decay stale motivations
                _dormant = core.decay_all_motivations(character)
                if _dormant > 0:
                    _tsz_emit(f"[KairosDream] {character}: {_dormant} motivations went dormant")

                # Seed canon motivations if none exist yet
                _active = core.get_active_motivations(character, top_k=1)
                if not _active:
                    try:
                        get_character_motivation_core = _require_hook("agent_motivation_core")
                        _canon_goals = get_character_motivation_core(character)
                        if _canon_goals:
                            _seeded = core.seed_canon_motivations(character, _canon_goals)
                            _tsz_emit(f"[KairosDream] {character}: seeded {_seeded} canon motivations")
                    except Exception:
                        pass
                domain_writes.append(f"kairos.motivation.{character}")
            except Exception:
                pass

            # ── 4.5) Phase O.2 (2026-05-02) episodic 时间衰减（半衰期 60 天）──
            # 走 dream 窗、和 motivation decay 并列。保护规则：salience>=0.7 / access_count>3 / 高显著度永生。
            try:
                _decay_result = core.decay_all_episodic(
                    character,
                    half_life_days=60.0,
                    archive_threshold=0.10,
                    salience_protect=0.70,
                    access_protect=3,
                    dry_run=False,
                )
                _scanned = int(_decay_result.get("scanned", 0))
                _decayed = int(_decay_result.get("decayed", 0))
                _protected = int(_decay_result.get("skipped_protected", 0))
                if _decayed > 0 or _scanned > 0:
                    _tsz_emit(f"[KairosDream] {character}: episodic decay scanned={_scanned} "
                          f"decayed={_decayed} protected={_protected}")
                if _decayed > 0:
                    domain_writes.append(f"kairos.episodic_decay.{character}")
            except Exception as _ed_err:
                _tsz_emit(f"[KairosDream] {character}: episodic decay error: {_ed_err}")

            # ── 4.6) Phase Q.1.C (2026-05-03) cluster + summarize ──
            # 借鉴 MemOS reorganizer：把高度相似的 episodic 聚簇 + LLM 提炼 parent summary
            # 写到 semantic 层、children 标 parent_id（不删 children、保留具体细节）
            try:
                _cluster_result = core.cluster_and_summarize_memories(
                    character,
                    source_layer="episodic",
                    summary_layer="semantic",
                    similarity_threshold=0.62,
                    min_cluster_size=4,
                    max_clusters_per_run=3,  # dream 一次最多 3 个 parent、控成本
                    dry_run=False,
                )
                _summarized = int(_cluster_result.get("clusters_summarized", 0))
                _scanned = int(_cluster_result.get("scanned", 0))
                if _summarized > 0:
                    _parents = _cluster_result.get("parents_created", [])
                    _previews = "; ".join(p.get("summary_preview", "")[:40] for p in _parents)
                    _tsz_emit(f"[KairosDream] {character}: cluster→summary scanned={_scanned} created={_summarized} ({_previews})")
                    domain_writes.append(f"kairos.cluster_summary.{character}")
                elif _scanned > 0:
                    # 跑了但没满足 cluster 阈值、记录一下不报错
                    _tsz_emit(f"[KairosDream] {character}: cluster scanned={_scanned} 0 满足阈值")
            except Exception as _cs_err:
                _tsz_emit(f"[KairosDream] {character}: cluster+summarize error: {_cs_err}")

            # ── 4.6b) cluster memory(事件/话题叙事聚类,2026-08-16)──
            # 与 4.6 的 embedding-only ≤15字 parent 不同:topic/salience 候选 +
            # 时间窗分桶 + LLM 叙事契约 + significance/审计门槛,走完整写链。
            # 默认关(MEM_CLUSTER_MEMORY_CREATE_ENABLE);02:30 WORLD_TZ 逻辑日幂等。
            try:
                from tsuzuri.cluster_memory import mem_cluster_memory_create_enabled, run_kairos_cluster_memory

                if mem_cluster_memory_create_enabled():
                    _cm_result = run_kairos_cluster_memory(core, character)
                    _cm_created = int(_cm_result.get("created", 0))
                    _cm_status = str(_cm_result.get("status", "") or "")
                    if _cm_created > 0:
                        _cm_titles = "; ".join(
                            str(p.get("title", ""))[:30] for p in _cm_result.get("parents_created", [])
                        )
                        _tsz_emit(
                            f"[KairosDream] {character}: cluster_memory created={_cm_created} ({_cm_titles})",
                            flush=True,
                        )
                        domain_writes.append(f"kairos.cluster_memory.{character}")
                    elif _cm_status not in ("idempotent",):
                        _tsz_emit(f"[KairosDream] {character}: cluster_memory {_cm_status}", flush=True)
            except Exception as _cm_err:
                _tsz_emit(f"[KairosDream] {character}: cluster_memory error: {_cm_err}")

            try:
                from tsuzuri.cluster_memory_rethink import run_cluster_rethink_shadow

                _cm_edit = run_cluster_rethink_shadow(core, character)
                if _cm_edit.get("llm_calls"):
                    domain_writes.append(f"kairos.cluster_rethink_shadow.{character}")
                    _tsz_emit(f"[ClusterRethink] {character}: {_cm_edit['status']} calls={_cm_edit['llm_calls']}")
            except Exception as _cm_edit_err:
                _tsz_emit(f"[ClusterRethink] {character}: {_cm_edit_err}")

            # ── 4.7) P2 per-character living summary ──
            try:
                from tsuzuri.memory_core import mem_core_summary_enabled

                if mem_core_summary_enabled():
                    _dream_core_summary_result = _run_core_summary_rebuild(core, character)
                    if _dream_core_summary_result.get("status") == "rebuilt":
                        domain_writes.append(f"kairos.core_summary.{character}")
                        meta_writes.append(str(_dream_core_summary_result.get("meta_key", "") or ""))
                        try:
                            from tsuzuri.memory_api import get_memory_api as _get_memory_api

                            _get_memory_api(core).record_budget_event(
                                character,
                                "core_summary_rebuilds",
                                amount=1,
                                tokens=int(_dream_core_summary_result.get("tokens", 0) or 0),
                            )
                        except Exception:
                            pass
                else:
                    _dream_core_summary_result = {"ok": True, "status": "disabled"}
            except Exception as _core_summary_err:
                _dream_core_summary_result = {
                    "ok": False,
                    "status": "failed",
                    "fallback_reason": str(_core_summary_err)[:160],
                }
                _tsz_emit(
                    f"[KAIROS][CoreSummary] failed for {character}: {_core_summary_err}",
                    flush=True,
                )

            # ── 4.7) P3 derived entity projection (same maintenance slot) ──
            try:
                from tsuzuri.memory_entities import mem_entity_rebuild_enabled

                if mem_entity_rebuild_enabled():
                    _dream_entity_result = _run_entity_rebuild(core, character)
                    if _dream_entity_result.get("status") == "rebuilt":
                        domain_writes.append(f"kairos.entity.{character}")
                        meta_writes.append(str(_dream_entity_result.get("meta_key", "") or ""))
                        try:
                            from tsuzuri.memory_api import get_memory_api as _get_memory_api

                            _get_memory_api(core).record_budget_event(
                                character,
                                "entity_rebuilds",
                                amount=1,
                            )
                        except Exception:
                            pass
                else:
                    _dream_entity_result = {"ok": True, "status": "disabled"}
            except Exception as _entity_err:
                _dream_entity_result = {
                    "ok": False,
                    "status": "failed",
                    "fallback_reason": str(_entity_err)[:160],
                }
                _tsz_emit(f"[KAIROS][Entity] failed for {character}: {_entity_err}", flush=True)

        except Exception as _maintenance_err:
            # Maintenance remains best-effort, but a whole-chain failure must be observable.
            _tsz_emit(
                f"[KairosDream] {character}: maintenance chain error: "
                f"{type(_maintenance_err).__name__}: {_maintenance_err}",
                flush=True,
            )

        # Character-private impressions evolve from old entries plus new exchanges.
        try:
            from tsuzuri.developer_profile import evolve_profile
            _profile_flags = dict(runtime_snapshot.get("mode_flags", {}) or {})
            _profile_result = evolve_profile(core, character, mode_flags=_profile_flags)
            _dream_evolution_result["developer_profile"] = _profile_result
        except Exception as exc:
            _dream_evolution_result["developer_profile"] = {"status": "failed", "error": type(exc).__name__}
            logger.warning("Developer profile evolution failed for %s: %s", character, type(exc).__name__)

        result = KairosDreamResult(
            ok=True,
            summary={
                "reason_code": "dream_completed",
                "developer_profile": _dream_evolution_result.get("developer_profile", {}),
                "session_id": session_id,
                "character": character,
                "turn_count": int(session_snapshot.get("turn_count", 0) or 0),
                "refreshed_sections": list(brief_result.sections.keys()),
                "guardrail_count": int(signal_summary.get("guardrail_count", 0) or 0),
                "review_flags": list(signal_bundle.review_flags if isinstance(signal_bundle.review_flags, list) else []),
                "evolution": {
                    "run_id": _dream_evolution_result.get("run_id", ""),
                    "actions": len(_dream_evolution_result.get("actions", [])),
                    "shadow": _dream_evolution_result.get("shadow", True),
                    "applied": bool(_dream_evolution_result.get("applied", {})),
                },
                "memory_rethink": {
                    "run_id": _dream_rethink_result.get("run_id", ""),
                    "status": _dream_rethink_result.get("status", ""),
                    "shadow": _dream_rethink_result.get("shadow", True),
                    "proposed": _dream_rethink_result.get("actions_proposed_count", 0),
                    "simulated_or_applied": _dream_rethink_result.get("actions_applied_count", 0),
                    "rejected": _dream_rethink_result.get("actions_rejected_count", 0),
                    "fallback_reason": _dream_rethink_result.get("fallback_reason", ""),
                },
                "core_summary": {
                    "status": _dream_core_summary_result.get("status", ""),
                    "item_id": _dream_core_summary_result.get("item_id", ""),
                    "fallback_reason": _dream_core_summary_result.get("fallback_reason", ""),
                },
                "forgetting": {
                    "status": _dream_forgetting_result.get("status", ""),
                    "dry_run": _dream_forgetting_result.get("dry_run", True),
                    "frozen_candidates": _dream_forgetting_result.get("frozen_candidates", 0),
                    "frozen_applied": _dream_forgetting_result.get("frozen_applied", 0),
                    "tombstone_candidates": _dream_forgetting_result.get("tombstone_candidates", 0),
                    "tombstone_applied": _dream_forgetting_result.get("tombstone_applied", 0),
                    "fallback_reason": _dream_forgetting_result.get("fallback_reason", ""),
                },
                "ooc_audit": {
                    "audit_id": _dream_audit_result.get("audit_id", ""),
                    "ooc_score": _dream_audit_result.get("ooc_score", 0.0),
                    "replies_checked": len(recent_replies) if recent_replies else 0,
                },
                "relationship": _dream_relationship_update,
                "daily_continuity_coverage": daily_continuity_coverage,
            },
            domain_writes=domain_writes,
            meta_writes=meta_writes,
            refreshed_brief=brief_payload,
        )
        record_kairos_dream(
            session_id,
            {
                "ok": result.ok,
                "summary": result.summary,
                "domain_writes": result.domain_writes,
                "meta_writes": result.meta_writes,
                "refreshed_brief": result.refreshed_brief,
                "created_at": utc_now().timestamp(),
                "turn_count": int(session_snapshot.get("turn_count", 0) or 0),
            },
        )
        # test_agent nightly smoke hook — DISABLED 2026-04-25 by user request.
        # test_agent has moved to interactive wizard mode and shouldn't auto-fire
        # from KAIROS dream cycle. To re-enable: uncomment below + set env
        # TEST_AGENT_NIGHTLY_HOOK=1.
        # try:
        #     from test_agent_lib import kairos_hook as _ta_hook
        #     _ta_hook.maybe_trigger_nightly_smoke(session_id=session_id, character=character)
        # except Exception:
        #     pass
        return result
    finally:
        store.mark_dream_finished(session_id)


__all__ = [
    "KairosDreamRequest",
    "KairosDreamResult",
    "run_kairos_dream_cycle",
]
