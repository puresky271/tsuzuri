"""Store mixin: memory evolution proposals, application and rollback.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import sqlite3
import uuid
from typing import Callable
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import (
    _safe_float, _safe_int, _utc_now_iso, mem_evolve_min_shadow_days, mem_evolve_shadow_only,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _lexical_similarity, _memory_row_age_hours,
    _memory_row_allows_evolution_semantic_promotion, _memory_row_has_durable_turn_signal,
    _memory_row_is_trusted_fact_observation, _memory_row_is_working,
)
from tsuzuri.memory_core_impl.features import (
    _compute_typed_score, _extract_memory_features, _is_cold_start_fast_track,
    _typed_gate_decision, CONSOLIDATION_SCORE_CANDIDATE_MIN, CONSOLIDATION_SCORE_FAST_TRACK,
    CONSOLIDATION_STATE_ARCHIVE_CANDIDATE, CONSOLIDATION_STATE_CANDIDATE,
    CONSOLIDATION_STATE_KEY_EVENT_PROMOTED, CONSOLIDATION_STATE_PROFILE_PROMOTED,
    CONSOLIDATION_STATE_RECALL_ARCHIVE, CONSOLIDATION_STATE_TRANSIENT,
    CONSOLIDATION_VALID_STATES,
)
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads, _parse_dt
from tsuzuri.memory_core_impl.meta_policy import _infer_memory_meta_from_row
from tsuzuri.persona_registry import resolve_agent

class _EvolutionMixin:
    """Memory evolution proposals, application and rollback."""


    def _evolution_actions_from_items(self, items: list[dict], max_actions: int = 8, evolution_policy: dict | None = None) -> list[dict]:
        rows = [dict(x) for x in items if isinstance(x, dict)]
        actions: list[dict] = []
        seen_pairs = set()
        now = utc_now_naive()
        ep = evolution_policy if isinstance(evolution_policy, dict) else {}
        duplicate_sim_threshold = _safe_float(ep.get("duplicate_sim_threshold", 0.78), 0.78)
        promote_salience_min = _safe_float(ep.get("promote_salience_min", 0.72), 0.72)
        promote_access_min = _safe_int(ep.get("promote_access_min", 2), 2)
        stale_days = _safe_float(ep.get("stale_days", 21), 21.0)
        stale_importance_max = _safe_float(ep.get("stale_importance_max", 0.42), 0.42)
        stale_access_max = _safe_int(ep.get("stale_access_max", 1), 1)
        rewrite_min_chars = _safe_int(ep.get("rewrite_min_chars", 120), 120)
        rewrite_confidence_max = _safe_float(ep.get("rewrite_confidence_max", 0.55), 0.55)
        # 2026-05-04: Phase Q.W — working → episodic promotion thresholds.
        # 旧设计只有 episodic→semantic、working turn 永久滞留 working layer 导致
        # explicit recall probe 模式下被双倍惩罚 (-0.14)、direct_pool 过滤又
        # 卡住、用户问"你还记得 X 吗" 时 4 天前的 working turn 召回失败。
        # 新分支让满足 (老化 >=24h + 任一保留信号) 的 working turn 升级 episodic、
        # 解放 working penalty。比 episodic→semantic 阈值更宽（working 该流动得快）。
        working_promote_age_hours = _safe_float(ep.get("working_promote_age_hours", 24.0), 24.0)
        working_promote_salience_min = _safe_float(ep.get("working_promote_salience_min", 0.50), 0.50)
        working_promote_importance_min = _safe_float(ep.get("working_promote_importance_min", 0.50), 0.50)
        working_promote_access_min = _safe_int(ep.get("working_promote_access_min", 1), 1)
        working_archive_grace_hours = _safe_float(ep.get("working_archive_grace_hours", 168.0), 168.0)

        # 1) merge duplicates
        for i in range(min(len(rows), 90)):
            a = rows[i]
            ta = str(a.get("text", "") or "")
            if len(ta) < 8:
                continue
            for j in range(i + 1, min(len(rows), i + 20)):
                b = rows[j]
                if str(a.get("layer", "") or "") != str(b.get("layer", "") or ""):
                    continue
                sim = _lexical_similarity(ta, str(b.get("text", "") or ""))
                if sim < duplicate_sim_threshold:
                    continue
                pair = tuple(sorted([str(a.get("item_id", "")), str(b.get("item_id", ""))]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                winner = a if _safe_float(a.get("importance", 0.0), 0.0) >= _safe_float(b.get("importance", 0.0), 0.0) else b
                loser = b if winner is a else a
                actions.append(
                    {
                        "op": "merge",
                        "item_ids": [str(winner.get("item_id", "")), str(loser.get("item_id", ""))],
                        "reason": "high lexical overlap duplicate",
                        "priority": 1,
                    }
                )

        # 1.5) Layer 2 state machine: aged working turn → candidate / promote / stale_archive
        # （Phase Q.W2 / Option E、2026-05-04）
        #
        # 旧设计 (Phase Q.W、已 deprecated)：sal/imp >=0.50 OR access_count>=1
        # → default 0.50 永远过、所有 24h+ turn 一视同仁升 episodic（用户原 bug）
        #
        # 中间 fix (Phase Q.W.fix1) 改 strict `>` → 实际 hit rate 0%、过 strict
        #
        # **本设计 (Option E、用户拍板 2026-05-04 完整 4 层)**：
        #   1. 读 payload.memory_policy.memory_features (Layer 1 写入)
        #   2. 读 payload.memory_policy.confirmed_reuse_count (Layer 3 写入)
        #   3. State machine 决策：
        #      ephemeral_penalty=1.0           → stale_archive (强档 kill)
        #      score < 0.45                    → stale_archive
        #      0.45 <= score < 0.70 (无 reuse) → candidate state (留 working、等 reuse)
        #      score >= 0.70 + reuse>=1        → typed_gate
        #      score >= 0.85 OR cold_start     → typed_gate (fast-track)
        #   4. typed_gate 决定 target_layer + engram_type:
        #      profile (semantic/user_profile_fact)、key_event、recall_archive
        #
        # candidate state 通过 op="set_consolidation_state" 写回 payload、不改 layer。
        # apply 路径：promote 走现有分支、archive 走 stale 分支、candidate 是新 op。
        for r in rows:
            if not _memory_row_is_working(r):
                continue
            if str(r.get("layer", "") or "") != "working":
                continue  # 已不在 working layer、跳过
            text = str(r.get("text", "") or "")
            if not text:
                continue
            # 2026-05-04 audit fix：移除 ephemeral_noise pre-check 的 continue
            # 旧设计：跳过 ephemeral noise（"嗯"/"好"等）让其他 archive 路径处理
            # 新设计：Layer 1 ephemeral_penalty=1.0 已 cover greeting/command/transient、
            # 应该走 Layer 2 archive 分支显式 archive。
            # 仍保留 _memory_working_text_is_ephemeral_noise 作为 backwards-compat 信号、
            # 但**不**直接 continue，让 ephemeral_penalty=1.0 → archive 分支 fire。
            age_h = _memory_row_age_hours(r)
            if age_h < working_promote_age_hours:
                continue  # < 24h、还在 transient 期、不评估

            _iid = str(r.get("item_id", ""))
            _char = str(r.get("character", ""))

            # Read Layer 1 features + Layer 3 reuse signals from payload
            _features = {}
            _typed_score = 0.0
            _cold_start = False
            _confirmed_reuse = 0
            _current_state = CONSOLIDATION_STATE_TRANSIENT
            _payload: dict = {}
            try:
                _payload_str = str(r.get("payload_json", "") or "")
                if _payload_str:
                    _payload = _json_loads(_payload_str, {})
                    if isinstance(_payload, dict):
                        _mem_policy = _payload.get("memory_policy", {})
                        if isinstance(_mem_policy, dict):
                            _features = _mem_policy.get("memory_features", {}) or {}
                            _typed_score = float(_mem_policy.get("typed_score", 0.0) or 0.0)
                            _cold_start = bool(_mem_policy.get("cold_start_fast_track", False))
                            _confirmed_reuse = int(_mem_policy.get("confirmed_reuse_count", 0) or 0)
                            _current_state = str(_mem_policy.get("consolidation_state", CONSOLIDATION_STATE_TRANSIENT) or CONSOLIDATION_STATE_TRANSIENT)
            except Exception:
                pass

            # Backwards compat: row predates Layer 1 (no features) → recompute on the fly
            if not _features:
                if " | assistant:" in text:
                    # Reconstruct user_text + assistant from text field (fallback)
                    parts = text.split(" | assistant:", 1)
                    _ut = parts[0].replace("user:", "", 1).strip()
                    _at = parts[1].strip() if len(parts) > 1 else ""
                    _features = _extract_memory_features(
                        _ut, _at, character=_char, store=self,
                    )
                    _typed_score = _compute_typed_score(_features)
                    _cold_start = _is_cold_start_fast_track(_features)
                else:
                    # 2026-08-01 误杀修复：payload 无 Layer 1 features 且 text 不是
                    # 原始对话格式（"user:… | assistant:…"）的行，是 2026-05-09~08-01
                    # meta 序列化 bug 的受害者（payload 被清空、text 为第三人称摘要）。
                    # 对摘要文本兜底重算的 typed_score 没有判别力（该路径打分恒低于
                    # archive 阈值，2026-08-01 一轮 62/62 全灭）；unknown ≠ low value，
                    # 留在 candidate 等待一次性策展，不再 score-kill。
                    if _current_state != CONSOLIDATION_STATE_CANDIDATE:
                        actions.append({
                            "op": "set_consolidation_state",
                            "item_ids": [_iid],
                            "target_state": CONSOLIDATION_STATE_CANDIDATE,
                            "reason": (
                                "legacy payload wiped + non-turn text; unscorable, "
                                "hold for curation instead of score-kill"
                            ),
                            "priority": 4,
                        })
                        try:
                            _tsz_emit(f"[MemEvolve] working→candidate (unscorable legacy hold): char={_char} text={text[:40]!r}", flush=True)
                        except Exception:
                            pass
                    continue

            _ephemeral_strong = float(_features.get("ephemeral_penalty", 0.0)) >= 1.0
            _durable_turn_signal = _memory_row_has_durable_turn_signal(
                r, _payload if isinstance(_payload, dict) else {}
            )

            if _durable_turn_signal:
                actions.append({
                    "op": "promote",
                    "item_ids": [_iid],
                    "target_layer": "episodic",
                    "target_state": CONSOLIDATION_STATE_CANDIDATE,
                    "reason": "structured correction/durable-commitment/explicit-request TurnFact",
                    "priority": 1,
                })
                continue

            # ── Decision branch 1: ephemeral_penalty=1 → direct stale_archive ──
            if _ephemeral_strong:
                actions.append({
                    "op": "archive",
                    "item_ids": [_iid],
                    "reason": f"ephemeral_penalty=1.0 (greeting/command/transient)",
                    "priority": 3,
                })
                try:
                    _tsz_emit(f"[MemEvolve] working→stale_archive (ephemeral): char={_char} text={text[:40]!r}", flush=True)
                except Exception:
                    pass
                continue

            # ── Decision branch 2: score < 0.45 → stale_archive ──
            if _typed_score < CONSOLIDATION_SCORE_CANDIDATE_MIN:
                if age_h < working_archive_grace_hours:
                    if _current_state != CONSOLIDATION_STATE_ARCHIVE_CANDIDATE:
                        actions.append({
                            "op": "set_consolidation_state",
                            "item_ids": [_iid],
                            "target_state": CONSOLIDATION_STATE_ARCHIVE_CANDIDATE,
                            "reason": (
                                f"low typed_score ({_typed_score:.2f}); defer archive until "
                                f"{working_archive_grace_hours:.0f}h grace"
                            ),
                            "priority": 4,
                        })
                else:
                    actions.append({
                        "op": "archive",
                        "item_ids": [_iid],
                        "reason": (
                            f"low typed_score after grace ({_typed_score:.2f} < "
                            f"{CONSOLIDATION_SCORE_CANDIDATE_MIN})"
                        ),
                        "priority": 3,
                    })
                try:
                    _low_score_outcome = (
                        "archive_after_grace" if age_h >= working_archive_grace_hours else "archive_candidate"
                    )
                    _tsz_emit(
                        f"[MemEvolve] working→{_low_score_outcome} (low_score={_typed_score:.2f}): "
                        f"char={_char} text={text[:40]!r}",
                        flush=True,
                    )
                except Exception:
                    pass
                continue

            # ── Decision branch 3: typed_gate ──
            _target_layer, _target_engram, _gate_reason = _typed_gate_decision(
                _features, _typed_score, reuse_count=_confirmed_reuse,
            )

            if _target_layer is None:
                # Score 0.45-0.70 (无 reuse)、留 candidate state
                if _current_state != CONSOLIDATION_STATE_CANDIDATE:
                    actions.append({
                        "op": "set_consolidation_state",
                        "item_ids": [_iid],
                        "target_state": CONSOLIDATION_STATE_CANDIDATE,
                        "reason": f"score={_typed_score:.2f} in candidate range, awaiting reuse signal",
                        "priority": 4,
                    })
                    try:
                        _tsz_emit(f"[MemEvolve] working→candidate (score={_typed_score:.2f}): char={_char} text={text[:40]!r}", flush=True)
                    except Exception:
                        pass
                continue

            # ── Decision branch 4: promote with typed target ──
            try:
                _tsz_emit(
                    f"[MemEvolve] working→{_target_layer}/{_target_engram} promote: char={_char} "
                    f"score={_typed_score:.2f} cold={_cold_start} reuse={_confirmed_reuse} "
                    f"reason={_gate_reason} text={text[:40]!r}",
                    flush=True,
                )
            except Exception:
                pass
            actions.append({
                "op": "promote",
                "item_ids": [_iid],
                "target_layer": _target_layer,
                "target_engram_type": _target_engram,
                "target_state": (
                    CONSOLIDATION_STATE_PROFILE_PROMOTED if _target_engram == "user_profile_fact"
                    else CONSOLIDATION_STATE_KEY_EVENT_PROMOTED if _target_engram == "key_event"
                    else CONSOLIDATION_STATE_RECALL_ARCHIVE
                ),
                "reason": f"typed_gate: {_gate_reason} (age={age_h:.1f}h, score={_typed_score:.2f})",
                "priority": 2,
            })

        # 2) promote strong episodic to semantic
        for r in rows:
            if str(r.get("layer", "") or "") != "episodic":
                continue
            sal = _safe_float(r.get("salience", 0.0), 0.0)
            acc = _safe_int(r.get("access_count", 0), 0)
            payload = _json_loads(str(r.get("payload_json", "") or "{}"), {})
            if not _memory_row_allows_evolution_semantic_promotion(
                r, payload if isinstance(payload, dict) else {}
            ):
                continue
            trusted_fact = _memory_row_is_trusted_fact_observation(
                r, payload if isinstance(payload, dict) else {}
            )
            if sal >= promote_salience_min and (acc >= promote_access_min or trusted_fact):
                actions.append(
                    {
                        "op": "promote",
                        "item_ids": [str(r.get("item_id", ""))],
                        "target_layer": "semantic",
                        "reason": (
                            "trusted fact observation"
                            if trusted_fact and acc < promote_access_min
                            else "high salience and repeatedly retrieved"
                        ),
                        "priority": 1 if trusted_fact else 2,
                    }
                )

        # 3) archive stale low-value
        for r in rows:
            payload = _json_loads(str(r.get("payload_json", "") or "{}"), {})
            if _memory_row_has_durable_turn_signal(r, payload if isinstance(payload, dict) else {}):
                continue
            updated = _parse_dt(str(r.get("updated_at", "") or ""))
            if updated is None:
                continue
            age_days = (now - updated).total_seconds() / 86400.0
            imp = _safe_float(r.get("importance", 0.0), 0.0)
            acc = _safe_int(r.get("access_count", 0), 0)
            if (age_days >= stale_days and imp < stale_importance_max and acc <= stale_access_max) or (age_days >= 90 and acc == 0):
                actions.append(
                    {
                        "op": "archive",
                        "item_ids": [str(r.get("item_id", ""))],
                        "reason": "stale low-value memory",
                        "priority": 3,
                    }
                )

        # 4) rewrite low-confidence long memories
        for r in rows:
            tx = str(r.get("text", "") or "")
            if len(tx) < rewrite_min_chars:
                continue
            conf = _safe_float(r.get("confidence", 0.0), 0.0)
            if conf < rewrite_confidence_max:
                rewritten = tx[:120].rstrip(" ,，。") + " (compressed)"
                actions.append(
                    {
                        "op": "rewrite",
                        "item_ids": [str(r.get("item_id", ""))],
                        "rewrite_text": rewritten,
                        "reason": "reduce verbosity and improve retrievability",
                        "priority": 4,
                    }
                )

        # Reserve capacity for layer transitions so merge pressure cannot starve
        # either semantic promotion or the working lifecycle. A target may
        # participate in only one action/run.
        limit = max(1, int(max_actions or 1))
        promotions = [action for action in actions if str(action.get("op", "")) == "promote"]
        semantic_promotions = [
            action for action in promotions
            if str(action.get("target_layer", "") or "") == "semantic"
        ]
        layer_by_id = {
            str(row.get("item_id", "") or ""): str(row.get("layer", "") or "")
            for row in rows
        }
        working_lifecycle = [
            action for action in actions
            if str(action.get("op", "")) in {"promote", "archive", "set_consolidation_state"}
            and any(
                layer_by_id.get(str(item_id or ""), "") == "working"
                for item_id in (action.get("item_ids", []) or [])
            )
        ]
        selected: list[dict] = []
        claimed: set[str] = set()

        def _take(action: dict) -> bool:
            ids = {
                str(item_id or "").strip()
                for item_id in (action.get("item_ids", []) if isinstance(action.get("item_ids", []), list) else [])
                if str(item_id or "").strip()
            }
            if not ids or claimed.intersection(ids) or len(selected) >= limit:
                return False
            selected.append(action)
            claimed.update(ids)
            return True

        for action in sorted(semantic_promotions, key=lambda item: int(item.get("priority", 9) or 9))[:1]:
            _take(action)
        for action in working_lifecycle[:2]:
            _take(action)
        for action in sorted(promotions, key=lambda item: int(item.get("priority", 9) or 9))[:2]:
            _take(action)
        for action in actions:
            _take(action)
        for action in promotions:
            _take(action)
        return selected[:limit]

    def _effective_shadow_mode(self) -> tuple[bool, dict]:
        now_dt = utc_now_naive()
        base_shadow = bool(mem_evolve_shadow_only())
        min_days = max(0, int(mem_evolve_min_shadow_days()))
        meta = self._get_meta_json("evolution_shadow_window", {})
        if not isinstance(meta, dict):
            meta = {}
        started_at = _parse_dt(str(meta.get("started_at", "") or ""))
        if started_at is None:
            started_at = now_dt
            self._set_meta_json(
                "evolution_shadow_window",
                {"started_at": _utc_now_iso(), "min_days": min_days},
            )
        elapsed_days = max(0.0, (now_dt - started_at).total_seconds() / 86400.0)
        forced_shadow = bool(elapsed_days < float(min_days))
        return (bool(base_shadow or forced_shadow), {"base_shadow": base_shadow, "forced_shadow": forced_shadow, "elapsed_days": round(elapsed_days, 3), "min_days": int(min_days)})

    def _query_evolution_items(
        self,
        character: str,
        *,
        working_age_hours: float = 24.0,
        working_archive_grace_hours: float = 168.0,
        recent_limit: int = 260,
        working_limit: int = 800,
    ) -> list[dict]:
        """Build a bounded evolution pool without hiding aged working rows."""
        char = resolve_agent(character)
        with self._lock:
            conn = self._connect()
            try:
                oldest_working = conn.execute(
                    """
                    SELECT item_id, character, layer, memory_role, engram_type,
                           item_type, text, payload_json, cue_json, lifecycle_json,
                           salience, confidence, novelty, importance, status, source,
                           evidence_id, created_at, updated_at, last_accessed_at,
                           access_count, archived
                    FROM memory_items
                    WHERE archived=0 AND status='active' AND importance > 0.15
                          AND COALESCE(item_type, '') != 'core_summary'
                          AND layer='working'
                          AND (character=? OR character='' OR character='*')
                    ORDER BY updated_at ASC
                    LIMIT ?;
                    """,
                    (char, max(20, min(800, int(working_limit or 800)))),
                ).fetchall()
            finally:
                conn.close()

        aged_working: list[dict] = []
        for row in oldest_working:
            item = dict(row)
            age_hours = _memory_row_age_hours(item)
            if age_hours < max(0.0, float(working_age_hours or 0.0)):
                continue
            payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
            policy = payload.get("memory_policy", {}) if isinstance(payload, dict) else {}
            policy = policy if isinstance(policy, dict) else {}
            state = str(policy.get("consolidation_state", "") or "")
            reuse_count = _safe_int(policy.get("confirmed_reuse_count", 0), 0)
            typed_score = _safe_float(policy.get("typed_score", 0.0), 0.0)
            cold_start = bool(policy.get("cold_start_fast_track", False))
            if (
                state == CONSOLIDATION_STATE_CANDIDATE
                and reuse_count <= 0
                and typed_score < CONSOLIDATION_SCORE_FAST_TRACK
                and not cold_start
            ):
                continue
            if (
                state == CONSOLIDATION_STATE_ARCHIVE_CANDIDATE
                and age_hours < max(0.0, float(working_archive_grace_hours or 0.0))
            ):
                continue
            aged_working.append(item)
        recent = self._query_items(char, limit=recent_limit)
        combined: list[dict] = []
        seen: set[str] = set()
        for row in [*aged_working, *recent]:
            item = dict(row)
            item_id = str(item.get("item_id", "") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            combined.append(item)
        return combined

    def propose_memory_evolution(
        self,
        character: str,
        *,
        llm_reviewer: Callable[[dict], dict] | None = None,
        max_actions: int = 8,
    ) -> dict:
        char = resolve_agent(character)
        shadow, shadow_meta = self._effective_shadow_mode()
        profile = self._load_policy_profile()
        ep = profile.get("evolution_policy", {}) if isinstance(profile.get("evolution_policy", {}), dict) else {}
        max_actions_cfg = _safe_int(ep.get("max_actions", max_actions), max_actions)
        working_age_hours = _safe_float(ep.get("working_promote_age_hours", 24.0), 24.0)
        working_archive_grace_hours = _safe_float(ep.get("working_archive_grace_hours", 168.0), 168.0)
        items = self._query_evolution_items(
            char,
            working_age_hours=working_age_hours,
            working_archive_grace_hours=working_archive_grace_hours,
        )
        actions = self._evolution_actions_from_items(items, max_actions=max_actions_cfg, evolution_policy=ep)

        duplicate_pairs = 0
        duplicate_sim_threshold = _safe_float(ep.get("duplicate_sim_threshold", 0.78), 0.78)
        for i in range(min(len(items), 80)):
            for j in range(i + 1, min(len(items), i + 16)):
                if _lexical_similarity(str(items[i].get("text", "") or ""), str(items[j].get("text", "") or "")) > duplicate_sim_threshold:
                    duplicate_pairs += 1
        redundancy = 1.0 - _clamp(duplicate_pairs / max(1.0, float(len(items) * 1.2)))
        retrievability = _clamp(sum(_safe_float(x.get("importance", 0.5), 0.5) for x in items[:40]) / max(1.0, float(min(40, len(items)))))
        consistency = _clamp(0.55 * redundancy + 0.45 * (1.0 - _clamp(duplicate_pairs / 25.0)))
        relevance = _clamp(sum(_safe_float(x.get("salience", 0.5), 0.5) for x in items[:40]) / max(1.0, float(min(40, len(items)))))
        evaluation = {
            "relevance": round(relevance, 3),
            "consistency": round(consistency, 3),
            "retrievability": round(retrievability, 3),
            "redundancy": round(redundancy, 3),
            "quality": round(_clamp(0.30 * relevance + 0.28 * consistency + 0.24 * retrievability + 0.18 * redundancy), 3),
            "reviewer": "heuristic",
            "shadow_window": shadow_meta,
        }

        fallback_reason = ""
        if callable(llm_reviewer) and actions:
            try:
                llm_eval = llm_reviewer(
                    {
                        "character": char,
                        "actions": actions,
                        "base_evaluation": evaluation,
                        "memory_stats": {"item_count": len(items), "duplicate_pairs": duplicate_pairs},
                    }
                )
                if isinstance(llm_eval, dict):
                    for k in ("relevance", "consistency", "retrievability", "redundancy", "quality"):
                        if k in llm_eval:
                            evaluation[k] = round(_clamp(_safe_float(llm_eval.get(k, evaluation[k]), evaluation[k])), 3)
                    evaluation["reviewer"] = str(llm_eval.get("reviewer", "llm+heuristic") or "llm+heuristic")
                    if str(llm_eval.get("notes", "") or "").strip():
                        evaluation["notes"] = str(llm_eval.get("notes", "") or "").strip()[:300]
            except Exception:
                fallback_reason = "llm_reviewer_error"

        run_id = "evo_" + uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        base_hash = hashlib.md5(_json_dumps({"character": char, "n": len(items)}).encode("utf-8", errors="ignore")).hexdigest()[:16]
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO evolution_runs
                    (run_id, character, shadow, status, proposal_json, evaluation_json, applied_actions_json, base_snapshot_hash, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        run_id,
                        char,
                        1 if shadow else 0,
                        "skipped_empty" if not actions else ("shadow_planned" if shadow else "planned"),
                        _json_dumps(actions),
                        _json_dumps(evaluation),
                        "[]",
                        base_hash,
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        self._log_policy_decision(
            decision_kind="evolution",
            phase="proposal",
            character=char,
            run_id=run_id,
            input_payload={
                "item_count": len(items),
                "max_actions": max_actions_cfg,
                "duplicate_threshold": duplicate_sim_threshold,
                "shadow": bool(shadow),
                "shadow_meta": shadow_meta,
            },
            llm_payload={"reviewer": str(evaluation.get("reviewer", ""))},
            fallback_reason=fallback_reason,
            final_payload={
                "action_count": len(actions),
                "quality": _safe_float(evaluation.get("quality", 0.0), 0.0),
            },
        )
        return {"run_id": run_id, "shadow": shadow, "actions": actions, "evaluation": evaluation}

    def _snapshot_item_row(self, conn: sqlite3.Connection, item_id: str) -> dict:
        iid = str(item_id or "").strip()
        if not iid:
            return {"exists": False}
        row = conn.execute(
            """
            SELECT item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json,
                   salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived
            FROM memory_items
            WHERE item_id=?
            LIMIT 1;
            """,
            (iid,),
        ).fetchone()
        if row is None:
            return {"exists": False}
        return {"exists": True, "row": {k: row[k] for k in row.keys()}}

    def _update_payload_memory_policy(self, conn: sqlite3.Connection, item_id: str, updates: dict) -> bool:
        """Phase Q.W2 / Option E: surgical update of payload.memory_policy fields.

        Reads current payload_json, deep-updates memory_policy sub-dict, writes back.
        Returns True if write succeeded.
        """
        iid = str(item_id or "").strip()
        if not iid or not isinstance(updates, dict) or not updates:
            return False
        row = conn.execute(
            "SELECT payload_json FROM memory_items WHERE item_id=? LIMIT 1;",
            (iid,)
        ).fetchone()
        if row is None:
            return False
        try:
            payload = _json_loads(str(row[0] or "{}"), {})
            if not isinstance(payload, dict):
                payload = {}
            mem_policy = payload.get("memory_policy", {})
            if not isinstance(mem_policy, dict):
                mem_policy = {}
            for k, v in updates.items():
                mem_policy[k] = v
            payload["memory_policy"] = mem_policy
            conn.execute(
                "UPDATE memory_items SET payload_json=? WHERE item_id=?;",
                (_json_dumps(payload), iid)
            )
            return True
        except Exception:
            return False

    def _restore_item_snapshot(self, conn: sqlite3.Connection, snapshot: dict, now: str) -> None:
        snap = snapshot if isinstance(snapshot, dict) else {}
        exists = bool(snap.get("exists", False))
        row = snap.get("row", {}) if isinstance(snap.get("row", {}), dict) else {}
        item_id = str(row.get("item_id", "") or "")
        if not exists:
            if item_id:
                conn.execute("DELETE FROM memory_items WHERE item_id=?;", (item_id,))
            return
        if not item_id:
            return
        conn.execute(
            """
            INSERT INTO memory_items
            (item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                character=excluded.character,
                layer=excluded.layer,
                memory_role=excluded.memory_role,
                engram_type=excluded.engram_type,
                item_type=excluded.item_type,
                text=excluded.text,
                payload_json=excluded.payload_json,
                cue_json=excluded.cue_json,
                lifecycle_json=excluded.lifecycle_json,
                salience=excluded.salience,
                confidence=excluded.confidence,
                novelty=excluded.novelty,
                importance=excluded.importance,
                status=excluded.status,
                source=excluded.source,
                evidence_id=excluded.evidence_id,
                created_at=excluded.created_at,
                updated_at=excluded.updated_at,
                last_accessed_at=excluded.last_accessed_at,
                access_count=excluded.access_count,
                archived=excluded.archived;
            """,
            (
                item_id,
                str(row.get("character", "") or ""),
                str(row.get("layer", "episodic") or "episodic"),
                str(row.get("memory_role", "") or self._normalize_memory_role(str(row.get("layer", "episodic") or "episodic"))),
                str(row.get("engram_type", "") or str(row.get("item_type", "note") or "note")),
                str(row.get("item_type", "note") or "note"),
                str(row.get("text", "") or ""),
                str(row.get("payload_json", "{}") or "{}"),
                str(row.get("cue_json", "{}") or "{}"),
                str(row.get("lifecycle_json", "{}") or "{}"),
                float(_safe_float(row.get("salience", 0.5), 0.5)),
                float(_safe_float(row.get("confidence", 0.5), 0.5)),
                float(_safe_float(row.get("novelty", 0.5), 0.5)),
                float(_safe_float(row.get("importance", 0.5), 0.5)),
                str(row.get("status", "active") or "active"),
                str(row.get("source", "restore") or "restore"),
                str(row.get("evidence_id", "") or ""),
                str(row.get("created_at", now) or now),
                str(row.get("updated_at", now) or now),
                str(row.get("last_accessed_at", "") or ""),
                int(_safe_int(row.get("access_count", 0), 0)),
                int(_safe_int(row.get("archived", 0), 0)),
            ),
        )

    def apply_evolution_plan(
        self,
        *,
        run_id: str | None = None,
        plan: list[dict] | None = None,
        shadow_only: bool | None = None,
    ) -> dict:
        now = _utc_now_iso()
        actions = [dict(x) for x in (plan or []) if isinstance(x, dict)]
        _run_id = str(run_id or "").strip()
        if _run_id and not actions:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute("SELECT proposal_json, shadow FROM evolution_runs WHERE run_id=? LIMIT 1;", (_run_id,)).fetchone()
                finally:
                    conn.close()
            if row is not None:
                actions = [dict(x) for x in _json_loads(str(row["proposal_json"] or "[]"), []) if isinstance(x, dict)]
                if shadow_only is None:
                    shadow_only = bool(_safe_int(row["shadow"], 1))

        if shadow_only is None:
            shadow_only, _shadow_meta = self._effective_shadow_mode()
        else:
            _shadow_meta = {"forced_shadow": False}
        if not actions:
            if _run_id:
                with self._lock:
                    conn = self._connect()
                    try:
                        conn.execute("UPDATE evolution_runs SET status=?, updated_at=? WHERE run_id=?;", ("skipped_empty", now, _run_id))
                        conn.commit()
                    finally:
                        conn.close()
            self._log_policy_decision(
                decision_kind="evolution",
                phase="apply",
                character="",
                run_id=_run_id,
                input_payload={"action_count": 0},
                fallback_reason="empty_actions",
                final_payload={"applied": 0, "shadow_only": bool(shadow_only)},
            )
            return {"ok": True, "applied": 0, "shadow_only": bool(shadow_only)}

        if shadow_only:
            if _run_id:
                with self._lock:
                    conn = self._connect()
                    try:
                        conn.execute(
                            "UPDATE evolution_runs SET status=?, applied_actions_json=?, updated_at=? WHERE run_id=?;",
                            ("shadow_only", _json_dumps(actions), now, _run_id),
                        )
                        conn.commit()
                    finally:
                        conn.close()
            self._log_policy_decision(
                decision_kind="evolution",
                phase="apply",
                character="",
                run_id=_run_id,
                input_payload={"action_count": len(actions)},
                fallback_reason="shadow_mode",
                final_payload={"applied": 0, "shadow_only": True, "shadow_meta": _shadow_meta},
            )
            return {"ok": True, "applied": 0, "shadow_only": True}

        applied = []
        action_logs = 0
        with self._lock:
            conn = self._connect()
            try:
                for idx, act in enumerate(actions):
                    op = str(act.get("op", "") or "").strip().lower()
                    ids = [str(x or "").strip() for x in (act.get("item_ids", []) if isinstance(act.get("item_ids", []), list) else []) if str(x or "").strip()]
                    if not ids:
                        continue
                    if op == "merge" and len(ids) >= 2:
                        winner = ids[0]
                        for loser in ids[1:]:
                            _before = self._snapshot_item_row(conn, loser)
                            conn.execute(
                                "UPDATE memory_items SET archived=1, status='merged', updated_at=? WHERE item_id=?;",
                                (now, loser),
                            )
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO memory_links(src_item_id, dst_item_id, link_type, weight, meta_json, created_at)
                                VALUES (?, ?, 'merge_into', 0.95, ?, ?);
                                """,
                                (loser, winner, _json_dumps({"reason": act.get("reason", "")}), now),
                            )
                            _after = self._snapshot_item_row(conn, loser)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    _run_id,
                                    int(idx),
                                    op,
                                    loser,
                                    _json_dumps(_before),
                                    _json_dumps(_after),
                                    "applied",
                                    now,
                                ),
                            )
                            action_logs += 1
                        applied.append({"op": op, "item_ids": ids})
                    elif op == "promote":
                        target = str(act.get("target_layer", "semantic") or "semantic").strip()
                        target_role = self._normalize_memory_role(target)
                        # Phase Q.W2 / Option E Layer 4: target_engram_type + consolidation_state
                        target_engram = str(act.get("target_engram_type", "") or "").strip()
                        target_state = str(act.get("target_state", "") or "").strip()
                        _before_map = {iid: self._snapshot_item_row(conn, iid) for iid in ids}
                        # Update layer + bump importance + (optional) engram_type
                        if target_engram:
                            conn.execute(
                                f"UPDATE memory_items SET layer=?, memory_role=?, engram_type=?, importance=MAX(importance, 0.7), updated_at=? WHERE archived=0 AND status='active' AND item_id IN ({','.join(['?'] * len(ids))});",
                                (target, target_role, target_engram, now, *ids),
                            )
                        else:
                            conn.execute(
                                f"UPDATE memory_items SET layer=?, memory_role=?, importance=MAX(importance, 0.7), updated_at=? WHERE archived=0 AND status='active' AND item_id IN ({','.join(['?'] * len(ids))});",
                                (target, target_role, now, *ids),
                            )
                        # Write consolidation_state to payload.memory_policy if specified
                        if target_state:
                            for iid in ids:
                                self._update_payload_memory_policy(conn, iid, {
                                    "consolidation_state": target_state,
                                    "consolidation_state_at": now,
                                })
                        promoted_ids: list[str] = []
                        for iid in ids:
                            promoted_row = conn.execute(
                                "SELECT * FROM memory_items WHERE item_id=? AND archived=0 AND status='active' LIMIT 1;",
                                (iid,),
                            ).fetchone()
                            if promoted_row is None or str(promoted_row["layer"] or "") != target:
                                continue
                            promoted = {key: promoted_row[key] for key in promoted_row.keys()}
                            payload = _json_loads(str(promoted.get("payload_json", "") or "{}"), {})
                            payload = payload if isinstance(payload, dict) else {}
                            old_meta = payload.get("memory_meta", {}) if isinstance(payload.get("memory_meta", {}), dict) else {}
                            payload.pop("memory_meta", None)
                            promoted["memory_role"] = target_role
                            promoted["layer"] = target
                            memory_meta = _infer_memory_meta_from_row(promoted, payload)
                            memory_meta["mode_scope"] = str(old_meta.get("mode_scope", memory_meta.get("mode_scope", "normal")) or "normal")
                            memory_meta["transient_context_only"] = False
                            memory_meta["promotion_candidate"] = False
                            payload["memory_meta"] = memory_meta
                            cue = self._build_cue_bundle(
                                str(promoted.get("text", "") or ""),
                                character=str(promoted.get("character", "") or ""),
                                payload=payload,
                                meta=payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {},
                                engram_type=str(promoted.get("engram_type", "") or ""),
                                memory_role=target_role,
                            )
                            lifecycle = self._build_lifecycle(
                                created_at=str(promoted.get("created_at", "") or now),
                                updated_at=now,
                                source="memory_evolution",
                                access_count=_safe_int(promoted.get("access_count", 0), 0),
                                last_retrieved_at=str(promoted.get("last_accessed_at", "") or ""),
                                state="consolidation",
                            )
                            conn.execute(
                                "UPDATE memory_items SET payload_json=?, cue_json=?, lifecycle_json=? WHERE item_id=?;",
                                (_json_dumps(payload), _json_dumps(cue), _json_dumps(lifecycle), iid),
                            )
                            _after = self._snapshot_item_row(conn, iid)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, iid, _json_dumps(_before_map.get(iid, {"exists": False})), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                            promoted_ids.append(iid)
                        applied.append({"op": op, "item_ids": promoted_ids, "target_layer": target,
                                        "target_engram_type": target_engram or None,
                                        "target_state": target_state or None})
                    elif op == "set_consolidation_state":
                        # Phase Q.W2 / Option E Layer 2: candidate state transition (no layer change)
                        target_state = str(act.get("target_state", "") or "").strip()
                        if not target_state or target_state not in CONSOLIDATION_VALID_STATES:
                            applied.append({"op": op, "item_ids": ids, "skipped": "invalid_target_state"})
                            continue
                        _before_map = {iid: self._snapshot_item_row(conn, iid) for iid in ids}
                        for iid in ids:
                            self._update_payload_memory_policy(conn, iid, {
                                "consolidation_state": target_state,
                                "consolidation_state_at": now,
                            })
                            _after = self._snapshot_item_row(conn, iid)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, iid, _json_dumps(_before_map.get(iid, {"exists": False})), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                        applied.append({"op": op, "item_ids": ids, "target_state": target_state})
                    elif op == "demote":
                        target = str(act.get("target_layer", "episodic") or "episodic").strip()
                        _before_map = {iid: self._snapshot_item_row(conn, iid) for iid in ids}
                        conn.execute(
                            f"UPDATE memory_items SET layer=?, importance=MIN(importance, 0.55), updated_at=? WHERE item_id IN ({','.join(['?'] * len(ids))});",
                            (target, now, *ids),
                        )
                        for iid in ids:
                            _after = self._snapshot_item_row(conn, iid)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, iid, _json_dumps(_before_map.get(iid, {"exists": False})), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                        applied.append({"op": op, "item_ids": ids, "target_layer": target})
                    elif op == "archive":
                        _before_map = {iid: self._snapshot_item_row(conn, iid) for iid in ids}
                        conn.execute(
                            f"UPDATE memory_items SET archived=1, status='archived', updated_at=? WHERE item_id IN ({','.join(['?'] * len(ids))});",
                            (now, *ids),
                        )
                        for iid in ids:
                            _after = self._snapshot_item_row(conn, iid)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, iid, _json_dumps(_before_map.get(iid, {"exists": False})), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                        applied.append({"op": op, "item_ids": ids})
                    elif op == "drop":
                        _before_map = {iid: self._snapshot_item_row(conn, iid) for iid in ids}
                        conn.execute(
                            f"DELETE FROM memory_items WHERE item_id IN ({','.join(['?'] * len(ids))});",
                            (*ids,),
                        )
                        for iid in ids:
                            _after = self._snapshot_item_row(conn, iid)
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, iid, _json_dumps(_before_map.get(iid, {"exists": False})), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                        applied.append({"op": op, "item_ids": ids})
                    elif op == "rewrite":
                        new_text = str(act.get("rewrite_text", "") or "").strip()
                        if not new_text:
                            base_row = conn.execute("SELECT text FROM memory_items WHERE item_id=? LIMIT 1;", (ids[0],)).fetchone()
                            src = str(base_row["text"] or "") if base_row is not None else ""
                            new_text = (src[:120].rstrip(" ,，。") + " (rewritten)") if src else ""
                        if new_text:
                            _before = self._snapshot_item_row(conn, ids[0])
                            conn.execute(
                                "UPDATE memory_items SET text=?, updated_at=?, confidence=MAX(confidence, 0.58), novelty=MIN(novelty, 0.42) WHERE item_id=?;",
                                (new_text[:240], now, ids[0]),
                            )
                            _after = self._snapshot_item_row(conn, ids[0])
                            conn.execute(
                                """
                                INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (_run_id, int(idx), op, ids[0], _json_dumps(_before), _json_dumps(_after), "applied", now),
                            )
                            action_logs += 1
                            applied.append({"op": op, "item_ids": [ids[0]]})
                if _run_id:
                    conn.execute(
                        "UPDATE evolution_runs SET status=?, applied_actions_json=?, updated_at=? WHERE run_id=?;",
                        ("applied", _json_dumps(applied), now, _run_id),
                    )
                conn.commit()
            finally:
                conn.close()
        self._log_policy_decision(
            decision_kind="evolution",
            phase="apply",
            character="",
            run_id=_run_id,
            input_payload={"action_count": len(actions)},
            final_payload={"applied": len(applied), "action_logs": action_logs, "shadow_only": False},
        )
        return {"ok": True, "applied": len(applied), "shadow_only": False, "actions": applied}

    def rollback_evolution_run(self, run_id: str) -> dict:
        _run_id = str(run_id or "").strip()
        if not _run_id:
            return {"ok": False, "error": "missing_run_id"}
        now = _utc_now_iso()
        reverted = 0
        skipped = 0
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT action_id, item_id, before_json, status, reverted_at
                    FROM evolution_action_log
                    WHERE run_id=?
                    ORDER BY action_id DESC;
                    """,
                    (_run_id,),
                ).fetchall()
                if not rows:
                    row = conn.execute("SELECT status FROM evolution_runs WHERE run_id=? LIMIT 1;", (_run_id,)).fetchone()
                    status = str(row["status"] or "") if row is not None else ""
                    return {"ok": True, "run_id": _run_id, "reverted": 0, "skipped": 0, "status": status or "no_action_log"}

                for r in rows:
                    rid = int(_safe_int(r["action_id"], 0))
                    rv_at = str(r["reverted_at"] or "").strip()
                    if rv_at:
                        skipped += 1
                        continue
                    before = _json_loads(str(r["before_json"] or "{}"), {})
                    try:
                        self._restore_item_snapshot(conn, before if isinstance(before, dict) else {}, now)
                        conn.execute(
                            """
                            UPDATE evolution_action_log
                            SET status='reverted', reverted_at=?
                            WHERE action_id=?;
                            """,
                            (now, rid),
                        )
                        reverted += 1
                    except Exception:
                        skipped += 1
                        conn.execute(
                            """
                            UPDATE evolution_action_log
                            SET status='rollback_failed'
                            WHERE action_id=?;
                            """,
                            (rid,),
                        )
                conn.execute(
                    """
                    UPDATE evolution_runs
                    SET status=?, updated_at=?
                    WHERE run_id=?;
                    """,
                    ("rolled_back", now, _run_id),
                )
                conn.commit()
            finally:
                conn.close()
        self._log_policy_decision(
            decision_kind="evolution",
            phase="rollback",
            run_id=_run_id,
            input_payload={},
            final_payload={"reverted": reverted, "skipped": skipped},
        )
        return {"ok": True, "run_id": _run_id, "reverted": int(reverted), "skipped": int(skipped)}
