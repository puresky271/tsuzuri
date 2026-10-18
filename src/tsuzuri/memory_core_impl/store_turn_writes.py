"""Store mixin: turn event writes, write planning and guest-safe profile views.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import re
import sqlite3
from datetime import timedelta
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import _safe_float, _utc_now_iso
from tsuzuri.memory_core_impl.memory_rows import (
    _ASSISTANT_ONLY_FACT_RE, _EXPLICIT_MEMORY_WRITE_RE, _lexical_similarity,
    _memory_row_is_surfaceable_working, _norm_text, _PATH4_CHITCHAT_FACT_RE,
    _PATH4_ONESHOT_PROMISE_RE, _PATH4_TRANSIENT_DAILY_RE,
)
from tsuzuri.memory_core_impl.features import (
    _canonicalize_developer_fact, _compute_typed_score, _dedupe_short_strings,
    _derive_semantic_facts_from_user_text, _EMOTIONAL_DIRECT_SEMANTIC_LABELS,
    _EMOTIONAL_DIRECT_SEMANTIC_MIN_SALIENCE, _extract_entities_from_text,
    _extract_memory_features, _extract_topic_tags, _fact_is_dialogue_residue,
    _fact_is_durable_from_user, _is_cold_start_fast_track, _is_collective_memory_text,
    _rewrite_fact_via_llm, _topic_from_semantic_fact, CONSOLIDATION_STATE_TRANSIENT,
)
from tsuzuri.memory_core_impl.recall_scoring import (
    _clamp, _json_dumps, _json_loads, _json_safe_meta, _recency_score,
)
from tsuzuri.memory_core_impl.meta_policy import (
    _COLLECTIVE_SCOPE_MEMBERS, _CURRENT_STATUS_RE, _derive_time_anchor, _DEVELOPER_FOCUS_RE,
    _infer_memory_meta_from_row, _memory_query_analysis, _memory_record_meta,
    _SUBJECTIVE_PROJECTION_RE, _verify_fact_grounding,
)
from tsuzuri.persona_registry import collective_scope, resolve_agent, setting

# Words that mark a turn as group-level (collective) context. Deployments extend
# these via settings "collective_context_keywords" / "collective_event_keywords" /
# "collective_activity_keywords" (the last selects the activity event type).
_DEFAULT_COLLECTIVE_CONTEXT_KEYWORDS = ("全员", "我们几个")
_DEFAULT_COLLECTIVE_EVENT_KEYWORDS = ("全员", "我们几个")


class _TurnWritesMixin:
    """Turn event writes, write planning and guest-safe profile views."""


    def append_turn_event(
        self,
        character: str,
        user_text: str,
        assistant_reply: str,
        *,
        event_nodes: list[dict] | None = None,
        meta: dict | None = None,
        memory_snapshot: dict | None = None,
    ) -> dict:
        char = resolve_agent(character)
        now = _utc_now_iso()
        meta = meta if isinstance(meta, dict) else {}
        write_plan = meta.get("memory_write_plan") if isinstance(meta.get("memory_write_plan", {}), dict) else {}
        if not write_plan:
            write_plan = self.plan_turn_memory_writes(
                char,
                user_text,
                assistant_reply,
                event_nodes=event_nodes,
                meta=meta,
            )
        records = [dict(x) for x in (write_plan.get("records", []) if isinstance(write_plan.get("records", []), list) else []) if isinstance(x, dict)]
        shared_payload = dict(write_plan.get("shared_payload", {}) if isinstance(write_plan.get("shared_payload", {}), dict) else {})
        topic_tags = list(write_plan.get("topic_tags", []) if isinstance(write_plan.get("topic_tags", []), list) else [])
        collective_written = 0
        semantic_written = 0
        event_nodes_written = 0
        def _write_turn(conn: sqlite3.Connection) -> None:
            nonlocal collective_written, semantic_written, event_nodes_written
            collective_written = 0
            semantic_written = 0
            event_nodes_written = 0
            for rec in records:
                rec_type = str(rec.get("memory_type", "") or "").strip()
                text = str(rec.get("text", "") or "").strip()
                if not text:
                    continue
                layer = str(rec.get("layer", "") or "").strip() or "semantic"
                item_type = str(rec.get("item_type", "") or "").strip() or "note"
                engram_type = str(rec.get("engram_type", "") or "").strip() or item_type
                rec_char = str(rec.get("character", "") or char).strip() or char
                created_at = str(rec.get("created_at", "") or now)
                updated_at = str(rec.get("updated_at", "") or now)
                source = str(rec.get("source", "") or "turn_append")
                payload = dict(rec.get("payload", {}) if isinstance(rec.get("payload", {}), dict) else {})
                payload["memory_meta"] = dict(rec.get("memory_meta", {}) if isinstance(rec.get("memory_meta", {}), dict) else {})
                cue_payload = dict(rec.get("cue_payload", {}) if isinstance(rec.get("cue_payload", {}), dict) else {})
                cue_meta = dict(rec.get("cue_meta", {}) if isinstance(rec.get("cue_meta", {}), dict) else {})
                self._upsert_item(
                    {
                        "item_id": self._make_item_id(rec_char, layer, item_type, text, created_at),
                        "character": rec_char,
                        "layer": layer,
                        "memory_role": str(rec.get("memory_role", "") or self._normalize_memory_role(layer)),
                        "engram_type": engram_type,
                        "item_type": item_type,
                        "text": text[:320],
                        "payload_json": _json_dumps(payload),
                        "cue_json": _json_dumps(self._build_cue_bundle(text, character=rec_char, payload=cue_payload, meta=cue_meta, engram_type=engram_type, memory_role=str(rec.get("memory_role", "") or self._normalize_memory_role(layer)))),
                        "lifecycle_json": _json_dumps(self._build_lifecycle(created_at=created_at, updated_at=updated_at, source=source, access_count=0, state=("encoding" if rec_type == "working_turn" else "consolidation"))),
                        "topic_labels_json": _json_dumps(rec.get("topic_labels", []) or []),
                        "salience": float(_clamp(_safe_float(rec.get("salience", 0.5), 0.5))),
                        "confidence": float(_clamp(_safe_float(rec.get("confidence", 0.5), 0.5))),
                        "novelty": float(_clamp(_safe_float(rec.get("novelty", 0.4), 0.4))),
                        "importance": float(_clamp(_safe_float(rec.get("importance", 0.5), 0.5))),
                        "source": source,
                        "evidence_id": str(rec.get("evidence_id", "") or ""),
                        "created_at": created_at,
                        "updated_at": updated_at,
                    },
                    conn=conn,
                )
                if layer == "collective":
                    collective_written += 1
                elif layer == "semantic":
                    semantic_written += 1
                elif layer == "episodic":
                    event_nodes_written += 1

            if isinstance(memory_snapshot, dict):
                proc = memory_snapshot.get("_brain_procedural", {}).get(char, {}) if isinstance(memory_snapshot.get("_brain_procedural"), dict) else {}
                rules = proc.get("rules_v2", []) if isinstance(proc, dict) and isinstance(proc.get("rules_v2", []), list) else []
                for r in rules[:18]:
                    if not isinstance(r, dict):
                        continue
                    tx = str(r.get("text", "") or "").strip()
                    if not tx:
                        continue
                    added_at = str(r.get("added_at", "") or now)
                    strength = _clamp(_safe_float(r.get("strength", 0.55), 0.55))
                    self._upsert_item(
                        {
                            "item_id": self._make_item_id(char, "procedural", "rule", tx, added_at),
                            "character": char,
                            "layer": "procedural",
                            "memory_role": "control",
                            "engram_type": "strategy_rule",
                            "item_type": "rule",
                            "text": tx[:220],
                            "payload_json": _json_dumps({
                                **dict(r),
                                "memory_meta": _memory_record_meta(
                                    memory_type="suppressed_candidate",
                                    owner_scope=f"character:{char}",
                                    evidence_tier="medium",
                                    source_kind="strategy_rule",
                                    temporal_validity="persistent",
                                    surface_default="abstain",
                                    mode_scope="normal",
                                ),
                            }),
                            "cue_json": _json_dumps(self._build_cue_bundle(tx, character=char, payload=shared_payload, meta={"topics": topic_tags}, engram_type="strategy_rule", memory_role="control")),
                            "lifecycle_json": _json_dumps(self._build_lifecycle(created_at=added_at, updated_at=now, source="turn_append", access_count=0, state="update")),
                            "salience": strength,
                            "confidence": max(0.55, strength),
                            "novelty": 0.20,
                            "importance": max(0.58, strength),
                            "source": "turn_append",
                            "created_at": added_at,
                            "updated_at": now,
                        },
                        conn=conn,
                    )

        self._run_write_transaction(_write_turn)

        return {
            "ok": True,
            "character": char,
            "event_nodes_written": int(event_nodes_written),
            "semantic_written": int(semantic_written),
            "collective_written": int(collective_written),
            "write_plan": {
                "record_count": len(records),
                "suppressed_count": len(write_plan.get("suppressed", []) if isinstance(write_plan.get("suppressed", []), list) else []),
            },
        }

    def append_collective_memory_event(
        self,
        *,
        event_type: str,
        text: str,
        payload: dict | None = None,
        salience: float = 0.6,
        confidence: float = 0.6,
        novelty: float = 0.4,
        importance: float = 0.6,
        created_at: str = "",
    ) -> dict:
        tx = str(text or "").strip()
        if not tx:
            return {"ok": False, "error": "empty_text"}
        now = _utc_now_iso()
        ts = str(created_at or now).strip() or now
        pl = payload if isinstance(payload, dict) else {}
        self._upsert_item(
            {
                "item_id": self._make_item_id(collective_scope(), "collective", str(event_type or "group_consensus"), tx, ts),
                "character": collective_scope(),
                "layer": "collective",
                "memory_role": "collective",
                "engram_type": str(event_type or "group_consensus"),
                "item_type": str(event_type or "group_consensus"),
                "text": tx[:260],
                "payload_json": _json_dumps(pl),
                "cue_json": _json_dumps(self._build_cue_bundle(tx, character=collective_scope(), payload=pl, meta={"topics": pl.get("topics", []) if isinstance(pl.get("topics", []), list) else []}, engram_type=str(event_type or "group_consensus"), memory_role="collective")),
                "lifecycle_json": _json_dumps(self._build_lifecycle(created_at=ts, updated_at=now, source="collective_append", access_count=0, state="consolidation")),
                "salience": _clamp(_safe_float(salience, 0.6)),
                "confidence": _clamp(_safe_float(confidence, 0.6)),
                "novelty": _clamp(_safe_float(novelty, 0.4)),
                "importance": _clamp(_safe_float(importance, 0.6)),
                "source": "collective_append",
                "created_at": ts,
                "updated_at": now,
            }
        )
        return {"ok": True, "event_type": str(event_type or "group_consensus"), "created_at": ts}

    @staticmethod
    def _static_verify_fact_grounding(fact: str, user_text: str) -> str:
        return _verify_fact_grounding(fact, user_text)

    def plan_turn_memory_writes(
        self,
        character: str,
        user_text: str,
        assistant_reply: str,
        *,
        event_nodes: list[dict] | None = None,
        meta: dict | None = None,
    ) -> dict:
        char = resolve_agent(character)
        now = _utc_now_iso()
        node_list = [x for x in (event_nodes or []) if isinstance(x, dict)]
        meta = meta if isinstance(meta, dict) else {}
        raw_facts = meta.get("semantic_facts", []) if isinstance(meta.get("semantic_facts", []), list) else []
        fact_topics = [_topic_from_semantic_fact(str(x or "")) for x in raw_facts]
        topic_tags = _dedupe_short_strings(
            list(meta.get("topic_tags", []) if isinstance(meta.get("topic_tags", []), list) else [])
            + [x for x in fact_topics if x],
            limit=10,
        )
        situation_model = meta.get("situation_model", {}) if isinstance(meta.get("situation_model", {}), dict) else {}
        schedule_grounding = situation_model.get("schedule_grounding", {}) if isinstance(situation_model.get("schedule_grounding", {}), dict) else {}
        turn_fact_card = meta.get("turn_fact_card", {}) if isinstance(meta.get("turn_fact_card", {}), dict) else {}
        # Fine-grained topic labels from TurnFactCard LLM extraction; propagated to all records
        _fact_topic_labels: list[str] = [
            str(x or "").strip() for x in
            (turn_fact_card.get("topic_labels", []) if isinstance(turn_fact_card.get("topic_labels", []), list) else [])
            if str(x or "").strip()
        ][:6]

        def _turn_fact_retrieval_text(*keys: str) -> str:
            for key in keys:
                tx = re.sub(r"\s+", " ", str(turn_fact_card.get(key, "") or "").strip())
                if tx:
                    return tx[:260]
            return ""

        shared_payload = {
            # 2026-08-16 cluster Phase4-B2: meta["location"] 兜底——chat_server web 主路径
            # 没有 situation_model，由 _write_turn_to_memory 从 turn state snapshot
            # (live_location → schedule_location) 直接供给；为空则保持空(诚实未知)。
            "location": str(schedule_grounding.get("expected_location", "") or situation_model.get("location", "") or meta.get("location", "") or ""),
            "school_phase": str(schedule_grounding.get("school_phase", "") or situation_model.get("school_phase", "") or ""),
            "session_phase": str(schedule_grounding.get("session_phase", "") or situation_model.get("session_phase", "") or ""),
            "group_status": str(schedule_grounding.get("expected_activity", "") or situation_model.get("group_status", "") or ""),
        }
        mode_scope = str((meta.get("memory_plan", {}) if isinstance(meta.get("memory_plan", {}), dict) else {}).get("mode_scope", "normal") or "normal")
        records: list[dict] = []
        suppressed: list[dict] = []
        latest_user_msg = str(meta.get("latest_user_msg", "") or user_text or "").strip()
        explicit_memory_request = bool(_EXPLICIT_MEMORY_WRITE_RE.search(latest_user_msg))
        high_salience_event = any(_safe_float(n.get("salience", 0.0), 0.0) >= 0.60 for n in node_list)
        working_promotion_candidate = bool(explicit_memory_request or high_salience_event)

        # ── Phase Q.W2 / Option E Layer 1: feature extraction at write time ──
        # 11 features + typed_score + cold_start flag + initial consolidation_state
        # 写入 payload.memory_policy.consolidation_*、Layer 2 state machine 后续读取
        # 2026-05-09: thread character/store 给 embedding_novelty 用
        _features = _extract_memory_features(
            latest_user_msg, assistant_reply, meta=meta,
            character=char, store=self,
        )
        _typed_score = _compute_typed_score(_features)
        _is_cold_start = _is_cold_start_fast_track(_features)
        # 初始 state：所有新 working turn 都是 transient（age < 24h、未到 evaluation 窗口）
        _initial_consolidation_state = CONSOLIDATION_STATE_TRANSIENT

        # P2 (2026-05-04 audit): topic-level reuse detection
        # 写入 turn 时 extract topic_tags + entities、Layer 3 chat 路径用于 reuse 匹配
        _topic_tags_user = _extract_topic_tags(latest_user_msg)
        _topic_tags_combined = _extract_topic_tags(f"{latest_user_msg} {assistant_reply}")
        _entities_user = _extract_entities_from_text(latest_user_msg)
        _entities_combined = _extract_entities_from_text(f"{latest_user_msg} {assistant_reply}")

        turn_fact_text = _turn_fact_retrieval_text("fact", "state_delta", "near_summary")
        turn_text = turn_fact_text or f"user:{str(user_text or '').strip()[:140]} | assistant:{str(assistant_reply or '').strip()[:180]}"
        records.append(
            {
                "memory_type": "working_turn",
                "character": char,
                "layer": "working",
                "memory_role": "working",
                "engram_type": "turn_exchange",
                "item_type": "turn",
                "text": turn_text[:320],
                "payload": {
                    "user_text": str(user_text or "")[:300],
                    "assistant_reply": str(assistant_reply or "")[:300],
                    "turn_fact_card": dict(turn_fact_card) if turn_fact_card else {},
                    "source_trust": {
                        "user_text": "verified",
                        "assistant_reply": "unverified",
                        "turn_fact_card": "llm_extracted" if turn_fact_card else "",
                    },
                    "memory_policy": {
                        "transient_context_only": True,
                        "promotion_candidate": working_promotion_candidate,
                        "promotion_reason": (
                            "explicit_user_memory_request"
                            if explicit_memory_request
                            else "high_salience_event"
                            if high_salience_event
                            else ""
                        ),
                        "assistant_reply_verified": False,
                        # ── Phase Q.W2 / Option E Layer 1+2: typed features + state ──
                        "memory_features": _features,
                        "typed_score": round(_typed_score, 3),
                        "cold_start_fast_track": bool(_is_cold_start),
                        "consolidation_state": _initial_consolidation_state,
                        "consolidation_state_at": now,
                        # Layer 3 reuse signals (initialized; bumped by chat_server)
                        "context_exposure_count": 0,
                        "confirmed_reuse_count": 0,
                        "direct_recall_hit_count": 0,
                        "assistant_reference_accepted_count": 0,
                    },
                    "meta": _json_safe_meta(meta),
                },
                "cue_payload": shared_payload,
                "cue_meta": {"topics": topic_tags},
                "topic_labels": _fact_topic_labels,
                "salience": 0.50,
                "confidence": 0.70,
                "novelty": 0.45,
                "importance": 0.50,
                "source": "turn_append",
                "created_at": now,
                "updated_at": now,
                "evidence_id": "",
                "memory_meta": _memory_record_meta(
                    memory_type="working_turn",
                    owner_scope=f"character:{char}",
                    evidence_tier="medium",
                    source_kind="turn_exchange",
                    temporal_validity="transient",
                    surface_default="silent_grounding",
                    mode_scope=mode_scope,
                    transient_context_only=True,
                    promotion_candidate=working_promotion_candidate,
                    assistant_reply_verified=False,
                ),
            }
        )

        # 2026-05-09 fix: collective layer 写入独立于 event_nodes
        # 旧逻辑：collective detection 写在 `for n in node_list:` 循环里、依赖 event_nodes 非空
        # 但 chat_server 主对话路径传 event_nodes=[]、循环不跑、collective 永远不写
        # 新逻辑：turn-text 级检测、独立于 event_nodes、每 turn 最多写 1 条 collective
        try:
            _turn_text_collective = f"{str(latest_user_msg or '')} {str(assistant_reply or '')}".strip()
            _collective_kw = tuple(setting("collective_context_keywords", _DEFAULT_COLLECTIVE_CONTEXT_KEYWORDS))
            _coll_kw_hit = any(k in _turn_text_collective for k in _collective_kw)
            _coll_member_hit = [m for m in _COLLECTIVE_SCOPE_MEMBERS
                               if m in _turn_text_collective and m != char]
            _coll_school_phase = bool(shared_payload.get("school_phase"))
            _coll_session_phase = bool(shared_payload.get("session_phase"))
            # 触发条件（2026-06-02 再收紧·collective 必须有其他成员在场）：
            # - 提到 **2+ 个其他成员** → 跨成员事件强信号、不限长度
            # - 1 个成员**配上** keyword 或 phase → 仍算 group 相关（"和角色D去 活动场地"）
            # - **单成员无事件词 / 仅 keyword / 仅 phase** → **不升 collective**
            #   仅 keyword 命中（如角色C一个人和 user 闲聊"活动场地 后面的活动室"）不再
            #   误升为 group-shared——否则会泄漏进角色B/角色A等其他成员的召回。
            #   私聊到 1 个成员 ≠ 集体事件（"我向某角色坦白一件私事"）同样不升。
            _coll_should_write = False
            if len(_coll_member_hit) >= 2:
                _coll_should_write = True
            elif len(_coll_member_hit) >= 1 and (_coll_kw_hit or _coll_school_phase or _coll_session_phase):
                # 1 个成员 + 事件词 = 仍是 group-relevant（如 "和角色D一起去 活动场地 集训"）
                _coll_should_write = True
            # Phase 5.4 (2026-05-24): multi-room LINE channel 不写 collective
            # 共在 colocated 自然写、跨地理 LINE = 各自在不同位置、不是 group 集体事件、
            # 写 collective 会让 KAIROS dream 误判为「group 一起做某事」、污染集体记忆。
            _meta_channel = str(meta.get("channel", "") or "").strip().lower()
            if _coll_should_write and _meta_channel == "line":
                _coll_should_write = False
            if _coll_should_write:
                if True:  # 保留缩进（旧 if 长度 >= 10 块）
                    _coll_engram = (
                        "group_session_event"
                        if any(k in _turn_text_collective for k in (setting("collective_activity_keywords", []) or []))
                        else "school_timeline_anchor"
                    )
                    records.append({
                        "memory_type": "collective_event",
                        "character": collective_scope(),
                        "layer": "collective",
                        "memory_role": "collective",
                        "engram_type": _coll_engram,
                        "item_type": _coll_engram,
                        # 用 user_text 作主体（不带 assistant prefix）
                        "text": str(latest_user_msg or "")[:260],
                        "payload": {
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "source_character": char,
                            "evidence": "turn_text_detection",
                            "mentioned_members": _coll_member_hit,
                            "topics": topic_tags[:10],
                            **shared_payload,
                        },
                        "cue_payload": {
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "source_character": char,
                            "topics": topic_tags[:10],
                            **shared_payload,
                        },
                        "cue_meta": {"topics": topic_tags[:10]},
                        "topic_labels": _fact_topic_labels,
                        "salience": 0.62,
                        "confidence": 0.66,
                        "novelty": 0.30,
                        "importance": 0.62,
                        "source": "collective_turn_append",
                        "created_at": now,
                        "updated_at": now,
                        "evidence_id": "",
                        "memory_meta": _memory_record_meta(
                            memory_type="collective_event",
                            owner_scope="collective",
                            evidence_tier="medium",
                            source_kind="collective_event",
                            temporal_validity="recency_weighted",
                            surface_default="silent_grounding",
                            mode_scope=mode_scope,
                        ),
                    })
        except Exception as _coll_err:
            _tsz_emit(f"[Collective] turn-text detection error: {_coll_err}", flush=True)

        # ── Phase R.W (2026-06-16): web 主路径 episodic 补写、纠正写入倒转 ──
        # chat_server 的 append_turn_event 传 event_nodes=[]、episodic 永远写 0
        # （实测 working 1489 vs episodic turn_append 221 的根因）。当本轮是真实对话
        # （有 user 文本）、caller 没显式 block retrievable 写入（宿主应用 的 present-self-state
        # 抑制会设 retrievable_write_block_reason、保持不写）、且非 guest 时、从 turn 合成
        # 1 个 episodic node、让 episodic 重新成为即时高写入层。宿主应用 正常路径传非空
        # event_nodes、不进此分支（无双写）。synthetic 标记让下方循环跳过 collective 子写
        # （turn-text collective 块已独立覆盖、避免重复 collective）。
        if (
            not node_list
            and latest_user_msg
            and len(latest_user_msg.strip()) >= 2
            and not str(meta.get("retrievable_write_block_reason", "") or "").strip()
            and mode_scope != "guest"
        ):
            _syn_text = _turn_fact_retrieval_text("fact", "state_delta", "near_summary") or latest_user_msg.strip()
            node_list = [{
                "id": hashlib.md5(f"{now}|{_syn_text[:120]}|synth_episodic".encode("utf-8")).hexdigest()[:12],
                "ts": now,
                "intent": "chat",
                "valence": 0.0,
                "surprise": 0.45,
                "salience": 0.50,
                "actors": ["user"],
                "causal_links": [],
                "text": _syn_text[:200],
                "_synthetic": True,
            }]

        for n in node_list:
            txt = str(n.get("text", "") or "").strip()
            if not txt:
                continue
            # 2026-09-12: 纯短问句 node 不落 episodic。
            # perception 把每个用户问句标成 intent="qa" 事件 node（"你还记得我是谁吗"），
            # 原文作为 event 写入后成为召回回声噪声（与疑问句 query 词面同形、顶掉
            # 真正的事实行）；问题携带的事实由 TurnFact/semantic 通道另行蒸馏，
            # working 层也保留了完整 turn。>30 字的长问句可能携带计划/上下文、放行。
            if (
                str(n.get("intent", "") or "") == "qa"
                and len(txt) <= 30
                and re.search(r"(吗|呢|？|\?)\s*$", txt)
            ):
                continue
            ts = str(n.get("ts", "") or now)
            salience = _clamp(_safe_float(n.get("salience", 0.5), 0.5))
            surprise = _clamp(_safe_float(n.get("surprise", 0.5), 0.5))
            confidence = _clamp(0.45 + 0.35 * (1.0 - abs(_safe_float(n.get("valence", 0.0), 0.0))))
            novelty = surprise
            importance = _clamp(0.45 * salience + 0.35 * surprise + 0.20 * confidence)
            records.append(
                {
                    "memory_type": "episodic_event",
                    "character": char,
                    "layer": "episodic",
                    "memory_role": "episodic",
                    "engram_type": "event",
                    "item_type": "event",
                    "text": txt[:260],
                    # 2026-08-16 cluster Phase4-B2: turn_id↔evidence_id 映射——
                    # evidence_id 取 node id,turn_fact_turn_id 取本轮 fact card 的 turn_id,
                    # 写时两者都在手,供 cluster→turn 溯源(无 card 时为空=未知)。
                    "payload": {**dict(n), "turn_fact_turn_id": str(turn_fact_card.get("turn_id", "") or "")},
                    "cue_payload": {**shared_payload, "actors": n.get("actors", [])},
                    "cue_meta": {"topics": topic_tags},
                    "topic_labels": _fact_topic_labels,
                    "salience": salience,
                    "confidence": confidence,
                    "novelty": novelty,
                    "importance": importance,
                    "source": "turn_append",
                    "created_at": ts,
                    "updated_at": now,
                    "evidence_id": str(n.get("id", "") or ""),
                    "memory_meta": _memory_record_meta(
                        memory_type="episodic_event",
                        owner_scope=f"character:{char}",
                        evidence_tier=("medium" if confidence >= 0.58 else "low"),
                        source_kind="episodic_event",
                        temporal_validity="recency_weighted",
                        surface_default="light_callback",
                        mode_scope=mode_scope,
                    ),
                }
            )
            # Collective memory detection: cue-based (2026-06-02 收紧·必须有其他成员在场)
            # 旧版：1 个成员就升、后又用 `_keyword_collective or ...` short-circuit 掉
            #   精心算出的 _cue_collective gate → keyword-only 仍能误升（角色C一人聊 活动场地）
            # 新版：只认 _cue_collective（2+ 成员 OR 1 成员 + 事件词/phase）、
            #   keyword-only / phase-only / 单成员无事件词都不升 collective
            # 2026-06-16 (Phase R.W): synthetic（web 补写）node 跳过 collective 子写——
            # turn-text collective 块（上方独立分支）已覆盖此 turn、避免重复 collective。
            if n.get("_synthetic"):
                continue
            _keyword_collective = any(k in txt for k in (setting("collective_event_keywords", _DEFAULT_COLLECTIVE_EVENT_KEYWORDS) or []))
            _phase_collective = bool(shared_payload.get("session_phase") or shared_payload.get("school_phase"))
            _entities_in_text = [m for m in _COLLECTIVE_SCOPE_MEMBERS if m in txt and m != char]
            _cue_collective = False
            if len(_entities_in_text) >= 2:
                _cue_collective = True
            elif len(_entities_in_text) >= 1 and (_keyword_collective or _phase_collective):
                # 1 成员 + 事件词/phase = 仍是 group-relevant
                _cue_collective = True
            if _cue_collective:
                records.append(
                    {
                        "memory_type": "collective_event",
                        "character": collective_scope(),
                        "layer": "collective",
                        "memory_role": "collective",
                        "engram_type": ("group_session_event" if any(k in txt for k in (setting("collective_activity_keywords", []) or [])) else "school_timeline_anchor"),
                        "item_type": ("group_session_event" if any(k in txt for k in (setting("collective_activity_keywords", []) or [])) else "school_timeline_anchor"),
                        "text": txt[:260],
                        "payload": {
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "source_character": char,
                            "event_id": str(n.get("id", "") or ""),
                            "topics": topic_tags[:10],
                            **shared_payload,
                        },
                        "cue_payload": {
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "source_character": char,
                            "event_id": str(n.get("id", "") or ""),
                            "topics": topic_tags[:10],
                            **shared_payload,
                        },
                        "cue_meta": {"topics": topic_tags[:10]},
                        "salience": _clamp(0.60 + 0.25 * salience),
                        "confidence": confidence,
                        "novelty": novelty,
                        "importance": _clamp(0.58 + 0.24 * importance),
                        "source": "collective_append",
                        "created_at": ts,
                        "updated_at": now,
                        "evidence_id": str(n.get("id", "") or ""),
                        "memory_meta": _memory_record_meta(
                            memory_type="collective_event",
                            owner_scope="collective",
                            evidence_tier=("high" if confidence >= 0.72 else "medium"),
                            source_kind="collective_event",
                            temporal_validity="recency_weighted",
                            surface_default="silent_grounding",
                            mode_scope=mode_scope,
                        ),
                    }
                )

        # 2026-09-07 (方案 D): 显式记忆请求优先复用当轮 TurnFact card 的 fact——
        # turn_fact_primary LLM 已做过说话者/听者归因，是该轮事实的单一真源。
        # 原路径 _derive_semantic_facts_from_user_text 把整句原话裸拼「用户」前缀
        # （"你知道吗，虽然我们隔得很远，但我会想起你…"），呼格歧义会让
        # 下游 rewrite LLM 主客颠倒（会把"我"和"你"对调后直写 semantic）。
        # card 缺失/超短/超长/对话残留时回退原 verbatim 路径（fail-safe）。
        card_sourced_facts: set[str] = set()
        derived_facts: list[str] = []
        _card_fact_text = ""
        if explicit_memory_request and isinstance(turn_fact_card, dict):
            _card_fact_text = re.sub(r"\s+", " ", str(turn_fact_card.get("fact", "") or "").strip())
        if (
            _card_fact_text
            and 6 <= len(_card_fact_text) <= 120
            and not _fact_is_dialogue_residue(_card_fact_text)
        ):
            derived_facts = [_card_fact_text]
            card_sourced_facts.add(_card_fact_text)
        else:
            derived_facts = _derive_semantic_facts_from_user_text(latest_user_msg)
        facts = _dedupe_short_strings(list(raw_facts) + derived_facts, limit=12, max_len=160)
        turn_confidence = _clamp(_safe_float(meta.get("turn_confidence", 0.67), 0.67))

        # 2026-05-09 Path 4 fix: Layer 1 ephemeral + chitchat fact pre-filter
        # 之前的 semantic 写入路径完全 bypass Layer 1+4 gate、导致 chitchat 告辞
        # （"嗯，好，那我先去看书了，下次再聊~"）和 assistant 应答（"是吗，那可太好了"）
        # 都成了 "fact"。这里加两层拦截：
        #   1) Layer 1 ephemeral_penalty (短问候/命令/transient)
        #   2) _PATH4_CHITCHAT_FACT_RE: 长 chitchat 告辞 / 应答口头禅（这类不被 ephemeral 覆盖、
        #      因 ephemeral 设计成短句、historical 污染样本都是 16+ 字）
        #
        # 不加 typed_score 低分阈值：因 fact 经 _canonicalize_developer_fact 后多为
        # 第三人称形式（"用户最近在忙工作"），不匹配 _USER_FACT_PATTERN_RE 的"我..."
        # 模式、score 自然偏低、加阈值会误杀合法 dev fact。
        # 保留原 _verify_fact_grounding / _fact_is_durable_from_user / _ASSISTANT_ONLY_FACT_RE
        # 作为后续 layer。explicit_memory_request 时绕过本 filter。

        # 2026-05-09: LLM rewrite caller (可选、由 chat_server 注入到 meta)
        # 用于把 verbose 原话 fact 抽象成简短 callback-able 陈述
        _fact_rewrite_caller = None
        if isinstance(meta, dict):
            _maybe_caller = meta.get("fact_rewrite_llm_caller")
            if callable(_maybe_caller):
                _fact_rewrite_caller = _maybe_caller

        for raw_fact in facts[:12]:
            _rf = str(raw_fact or "").strip()
            # 2026-06-02: 对话残留前置拦截（在 canonicalize 加主语前、对原文判）。
            # 修 ordering bug：canonicalize 会把「嗯，话说…」加主语成「用户嗯，话说…」、
            # 架空 _rewrite_fact_via_llm 里 ^口头语 引语拦截。这里对原文直接拦。
            rewriteable_long_residue = bool(callable(_fact_rewrite_caller) and len(_rf) > 25)
            if not explicit_memory_request and _fact_is_dialogue_residue(_rf) and not rewriteable_long_residue:
                suppressed.append({"memory_type": "suppressed_candidate", "text": _rf[:120], "reason": "dialogue_residue"})
                _tsz_emit(f"[Path4Filter] BLOCKED dialogue residue: {_rf[:60]}", flush=True)
                continue
            fact = _canonicalize_developer_fact(_rf, character=char)
            if not fact:
                continue
            if _rf in card_sourced_facts:
                # TurnFact card fact 已是归因正确的第三人称抽象命题，不再过 rewrite
                # （二次 LLM 压缩只会引入新的主客错位/信息丢失风险）。下游
                # grounding / durable / collective 等闸口仍全部执行。
                if fact[-1] not in "。.！!？?":
                    fact = fact + "。"
            else:
                # 2026-05-09 Path 4: LLM rewrite verbose fact → concise callback form
                # 仅 > 25 字才调 LLM、短句直接放行
                fact = _rewrite_fact_via_llm(
                    fact, llm_caller=_fact_rewrite_caller, character=char,
                )
                if not fact:
                    continue
            topic_only = _topic_from_semantic_fact(fact)
            if topic_only:
                suppressed.append({"memory_type": "suppressed_candidate", "text": fact[:120], "reason": "topic_tag_only"})
                continue

            # 2026-05-09 Layer 1 ephemeral + chitchat fact pre-filter (Path 4 fix)
            if not explicit_memory_request:
                _fact_features = _extract_memory_features(
                    fact, "", character=char, store=self,
                )
                if float(_fact_features.get("ephemeral_penalty", 0.0)) >= 1.0:
                    suppressed.append({
                        "memory_type": "suppressed_candidate",
                        "text": fact[:120],
                        "reason": "layer1_ephemeral_chitchat",
                    })
                    _tsz_emit(f"[Path4Filter] BLOCKED ephemeral chitchat fact: {fact[:60]}", flush=True)
                    continue
                # 2026-05-12: 一次性承诺/认知 ack 拦截
                if _PATH4_ONESHOT_PROMISE_RE.search(fact):
                    suppressed.append({
                        "memory_type": "suppressed_candidate",
                        "text": fact[:120],
                        "reason": "path4_oneshot_promise",
                    })
                    _tsz_emit(f"[Path4Filter] BLOCKED oneshot promise/ack: {fact[:60]}", flush=True)
                    continue
                if _PATH4_CHITCHAT_FACT_RE.search(fact):
                    suppressed.append({
                        "memory_type": "suppressed_candidate",
                        "text": fact[:120],
                        "reason": "path4_chitchat_fact",
                    })
                    _tsz_emit(f"[Path4Filter] BLOCKED chitchat-pattern fact: {fact[:60]}", flush=True)
                    continue
                if _PATH4_TRANSIENT_DAILY_RE.search(fact):
                    suppressed.append({
                        "memory_type": "suppressed_candidate",
                        "text": fact[:120],
                        "reason": "path4_transient_daily",
                    })
                    _tsz_emit(f"[Path4Filter] BLOCKED transient daily fact: {fact[:60]}", flush=True)
                    continue

            # 事实验证：检查 fact 是否有 user_text 依据
            _grounding = _verify_fact_grounding(fact, str(user_text or ""))
            if _grounding == "ungrounded":
                suppressed.append({"memory_type": "suppressed_candidate", "text": fact[:120], "reason": "ungrounded_hallucination"})
                _tsz_emit(f"[FactVerify] BLOCKED ungrounded fact: {fact[:60]}")
                continue
            durable_user_fact = _fact_is_durable_from_user(fact, latest_user_msg, explicit_request=explicit_memory_request)
            fact_type = "shared_fact"
            owner_scope = "shared"
            source_kind = "shared_fact"
            surface_default = "silent_grounding"
            evidence_tier = "medium"
            if _DEVELOPER_FOCUS_RE.search(fact):
                if _SUBJECTIVE_PROJECTION_RE.search(fact) or _SUBJECTIVE_PROJECTION_RE.search(latest_user_msg):
                    fact_type = "character_projection"
                    owner_scope = f"character:{char}"
                    source_kind = "character_projection"
                    surface_default = "light_callback"
                elif bool(meta.get("prompt_inferred", False)) or bool(meta.get("runtime_inferred", False)):
                    suppressed.append({"memory_type": "suppressed_candidate", "text": fact[:120], "reason": "prompt_runtime_inferred"})
                    continue
                elif _CURRENT_STATUS_RE.search(fact):
                    evidence_tier = "low"
            if not durable_user_fact and source_kind == "shared_fact" and _ASSISTANT_ONLY_FACT_RE.search(fact):
                suppressed.append({"memory_type": "suppressed_candidate", "text": fact[:120], "reason": "assistant_only_unverified"})
                continue
            fact_collective = _is_collective_memory_text(fact, character=char) or (
                durable_user_fact and _is_collective_memory_text(latest_user_msg, character=char)
            )
            # ── Phase R.W (2026-06-16): turn-time semantic 直写降级、纠正写入倒转 ──
            # 「semantic 过度写入」的主因是每条 LLM 抽的 user_observation 都即时落 semantic、
            # 绕过 KAIROS 蒸馏（实测 turn_append semantic 689 vs episodic 221）。健康模型：
            #   - 显式记忆请求（"记住 X"、explicit_memory_request）→ 仍即时落 semantic
            #     （用户明确要的持久知识、稀有高价值、当轮就要能召回）
            #   - 其余自动抽取 fact → 落 episodic（temporal recency_weighted、会自然衰减）、
            #     由 KAIROS episodic→semantic promote（salience≥0.60 且被召回过 access≥1）
            #     慢慢蒸馏成永久 semantic。salience 保持 0.60-0.62 → 仍满足 promote 阈值。
            # promote 时只改 layer episodic→semantic、不重写 memory_type/engram、所以这条
            # 记录「出生为 episodic 观察、被召回+salient 后毕业为 semantic fact」链路自洽。
            if explicit_memory_request:
                _fact_layer = "semantic"
                _fact_role = "semantic"
                _fact_temporal = "persistent"
                _fact_engram = ("projection" if fact_type == "character_projection" else "fact")
            else:
                _fact_layer = "episodic"
                _fact_role = "episodic"
                _fact_temporal = "recency_weighted"
                _fact_engram = ("projection" if fact_type == "character_projection" else "fact_observation")
            records.append(
                {
                    "memory_type": fact_type,
                    "character": char,
                    "layer": _fact_layer,
                    "memory_role": _fact_role,
                    "engram_type": _fact_engram,
                    "item_type": _fact_engram,
                    "text": fact[:240],
                    "payload": {"fact": fact, "meta": _json_safe_meta(meta)},
                    "cue_payload": shared_payload,
                    "cue_meta": {"topics": topic_tags},
                    "salience": (0.60 if fact_type == "character_projection" else 0.62),
                    "confidence": (max(0.58, turn_confidence - 0.07) if fact_type == "character_projection" else max(0.64, turn_confidence)),
                    "novelty": 0.34,
                    "importance": (0.58 if fact_type == "character_projection" else 0.66),
                    "source": "turn_append",
                    "created_at": now,
                    "updated_at": now,
                    "evidence_id": "",
                    "memory_meta": _memory_record_meta(
                        memory_type=fact_type,
                        owner_scope=owner_scope,
                        evidence_tier=evidence_tier,
                        source_kind=source_kind,
                        temporal_validity=_fact_temporal,
                        surface_default=surface_default,
                        mode_scope=mode_scope,
                    ),
                }
            )
            if fact_collective:
                records.append(
                    {
                        "memory_type": "collective_event",
                        "character": collective_scope(),
                        "layer": "collective",
                        "memory_role": "collective",
                        "engram_type": "collective_user_fact",
                        "item_type": "collective_user_fact",
                        "text": fact[:240],
                        "payload": {
                            "fact": fact,
                            "source_character": char,
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "topics": topic_tags[:10],
                            "meta": _json_safe_meta(meta),
                        },
                        "cue_payload": {
                            "source_character": char,
                            "member_scope": sorted(_COLLECTIVE_SCOPE_MEMBERS),
                            "topics": topic_tags[:10],
                            **shared_payload,
                        },
                        "cue_meta": {"topics": topic_tags[:10]},
                        "salience": 0.72,
                        "confidence": max(0.64, turn_confidence),
                        "novelty": 0.36,
                        "importance": 0.72,
                        "source": "collective_append",
                        "created_at": now,
                        "updated_at": now,
                        "evidence_id": "",
                        "memory_meta": _memory_record_meta(
                            memory_type="collective_event",
                            owner_scope="collective",
                            evidence_tier=("high" if turn_confidence >= 0.72 else "medium"),
                            source_kind="collective_event",
                            temporal_validity="persistent",
                            surface_default="silent_grounding",
                            mode_scope=mode_scope,
                            promotion_candidate=True,
                            promotion_reason="collective_user_statement",
                        ),
                    }
                )

        # ── 2026-08-02: 重大情感话题直写 semantic（用户拍板标准）
        # 触发：turn_fact_card.topic_labels 命中情感组 且 card.salience ≥ 0.6。
        # 标准来源：876 条历史 keep 记忆的 LLM 初筛（受控词表打标 + 情感显著度），
        # 该组合命中 ~13%、逐标签抽查几乎零误伤（家人管控/迷茫坦承/表白/重逢级）。
        # 与 Phase R.W「自动 fact 先落 episodic 等蒸馏」分工不同：那条管
        # user_observation 类事实；本条管重大情感话题——稀有、当轮就要能召回、
        # 等演化蒸馏太慢且可能中间被 recency 衰减。
        _emotional_card = meta.get("turn_fact_card", {}) if isinstance(meta.get("turn_fact_card", {}), dict) else {}
        _emotional_labels = {
            str(x or "").strip()
            for x in (_emotional_card.get("topic_labels", []) if isinstance(_emotional_card.get("topic_labels", []), list) else [])
            if str(x or "").strip()
        }
        _emotional_hit_labels = _emotional_labels & _EMOTIONAL_DIRECT_SEMANTIC_LABELS
        _emotional_fact = str(_emotional_card.get("fact", "") or "").strip()
        _emotional_salience = _safe_float(_emotional_card.get("salience", 0.0), 0.0)
        if (
            _emotional_fact
            and _emotional_hit_labels
            and _emotional_salience >= _EMOTIONAL_DIRECT_SEMANTIC_MIN_SALIENCE
            and not explicit_memory_request  # 显式请求已走 semantic fact 直写、避免同轮重复
        ):
            _emotional_dup = False
            try:
                _dup_cutoff = (utc_now_naive() - timedelta(days=30)).isoformat()
                _dup_conn = self._connect()
                try:
                    for _dr in _dup_conn.execute(
                        "SELECT text FROM memory_items WHERE character=? AND layer='semantic' AND archived=0 AND created_at >= ? LIMIT 200;",
                        (char, _dup_cutoff),
                    ).fetchall():
                        if _lexical_similarity(_emotional_fact, str(_dr["text"] or "")) >= 0.80:
                            _emotional_dup = True
                            break
                finally:
                    _dup_conn.close()
            except Exception:
                pass
            if _emotional_dup:
                suppressed.append({
                    "memory_type": "suppressed_candidate",
                    "text": _emotional_fact[:120],
                    "reason": "emotional_key_event_dup",
                })
            else:
                records.append(
                    {
                        "memory_type": "key_event",
                        "character": char,
                        "layer": "semantic",
                        "memory_role": "semantic",
                        "engram_type": "key_event",
                        "item_type": "key_event",
                        "text": _emotional_fact[:240],
                        "payload": {
                            "fact": _emotional_fact,
                            "topic_labels": sorted(_emotional_hit_labels),
                            "turn_fact_turn_id": str(_emotional_card.get("turn_id", "") or ""),
                            "card_salience": _emotional_salience,
                            "meta": _json_safe_meta(meta),
                        },
                        "cue_payload": shared_payload,
                        "cue_meta": {"topics": topic_tags},
                        "salience": max(0.62, min(0.90, _emotional_salience)),
                        "confidence": max(0.64, _safe_float(_emotional_card.get("confidence", turn_confidence), turn_confidence)),
                        "novelty": 0.34,
                        "importance": max(0.66, min(0.90, _emotional_salience)),
                        "source": "turn_append",
                        "created_at": now,
                        "updated_at": now,
                        "evidence_id": str(_emotional_card.get("turn_id", "") or ""),
                        "memory_meta": _memory_record_meta(
                            memory_type="key_event",
                            owner_scope="shared",
                            evidence_tier="high",
                            source_kind="emotional_key_event",
                            temporal_validity="persistent",
                            surface_default="light_callback",
                            mode_scope=mode_scope,
                        ),
                    }
                )
                try:
                    _tsz_emit(
                        f"[EmotionalSemantic] {char}: direct semantic write "
                        f"labels={sorted(_emotional_hit_labels)} sal={_emotional_salience:.2f} "
                        f"text={_emotional_fact[:40]!r}",
                        flush=True,
                    )
                except Exception:
                    pass

        return {
            "records": records,
            "suppressed": suppressed,
            "shared_payload": shared_payload,
            "topic_tags": topic_tags,
        }

    def _build_guest_safe_profile_view_from_typed(
        self,
        character: str,
        query_text: str = "",
        *,
        query_analysis: dict | None = None,
        max_items: int = 4,
    ) -> dict:
        analysis = query_analysis if isinstance(query_analysis, dict) else _memory_query_analysis(query_text, {}, "guest")
        query = str(query_text or "").strip()
        char = resolve_agent(character)
        rows: list[dict] = []
        projection_rows: list[dict] = []
        seen_shared = set()
        seen_projection = set()
        for raw in self._query_items(char, limit=max(80, int(max_items or 4) * 26)):
            decorated = self._decorate_memory_row(raw)
            memory_type = str(decorated.get("memory_type", "") or "").strip()
            if memory_type not in ("shared_fact", "character_projection"):
                continue
            text = str(decorated.get("text", "") or "").strip()
            if (not text) or (not _DEVELOPER_FOCUS_RE.search(text)):
                continue
            status = str(decorated.get("status", "active") or "active").strip().lower()
            if status in ("superseded", "contradicted", "guest_blocked", "surface_blocked", "archived"):
                continue
            key = _norm_text(text)
            if not key:
                continue
            relevance = _lexical_similarity(query, text) if query else 0.0
            confidence = _clamp(_safe_float(decorated.get("confidence", decorated.get("_confidence", 0.5)), 0.5))
            importance = _clamp(_safe_float(decorated.get("importance", 0.5), 0.5))
            recency = _recency_score(str(decorated.get("updated_at", "") or ""), half_life_hours=240.0)
            evidence_tier = str(decorated.get("evidence_tier", "") or "low").strip().lower()
            payload = _json_loads(str(decorated.get("payload_json", "") or "{}"), {})
            item_anchor = _derive_time_anchor(text, payload if isinstance(payload, dict) else {})
            if item_anchor == "immediate":
                continue
            tier_bonus = {"high": 0.20, "medium": 0.12, "low": 0.05}.get(evidence_tier, 0.04)
            if memory_type == "shared_fact" and evidence_tier not in ("high", "medium"):
                continue
            if memory_type == "character_projection":
                if not (analysis.get("personality_query", False) or analysis.get("explicit_recall_intent", False)):
                    continue
                if key in seen_projection:
                    continue
                seen_projection.add(key)
            else:
                if key in seen_shared:
                    continue
                seen_shared.add(key)
            type_bonus = 0.08 if (memory_type == "character_projection") else 0.0
            score = 0.34 * relevance + 0.24 * confidence + 0.18 * importance + 0.12 * recency + tier_bonus + type_bonus
            row = dict(decorated)
            row["score"] = round(score, 4)
            row["relevance"] = round(relevance, 4)
            row["_confidence"] = round(confidence, 4)
            row["source"] = "guest_safe_profile_typed"
            if memory_type == "character_projection":
                projection_rows.append(row)
            else:
                rows.append(row)
        rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        projection_rows.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        rows = rows[: max(1, int(max_items or 4))]
        summary_lines = [str(row.get("text", "") or "").strip()[:120] for row in rows if str(row.get("text", "") or "").strip()]
        for row in projection_rows[:2]:
            text = str(row.get("text", "") or "").strip()[:120]
            if text and text not in summary_lines:
                summary_lines.append(text)
        detected = bool(rows or summary_lines)
        return {
            "detected": detected,
            "rows": rows,
            "summary_lines": summary_lines[:6],
            "query_analysis": analysis,
            "suppression_reason": "" if detected else "no_typed_safe_profile_match",
            "source": ("typed" if detected else ""),
            "read_policy": "stable_only",
            "legacy_fallback_used": False,
        }

    def _build_guest_safe_profile_view_from_legacy(
        self,
        character: str,
        query_text: str = "",
        *,
        query_analysis: dict | None = None,
        max_items: int = 4,
    ) -> dict:
        snap = self.load_json_snapshot()
        if not isinstance(snap, dict):
            snap = {}
        query = str(query_text or "").strip()
        analysis = query_analysis if isinstance(query_analysis, dict) else _memory_query_analysis(query, {}, "guest")

        lt = snap.get("long_term", {}) if isinstance(snap.get("long_term", {}), dict) else {}
        profile = lt.get("developer_profile", {}) if isinstance(lt.get("developer_profile", {}), dict) else {}
        rels = lt.get("character_relationships", {}) if isinstance(lt.get("character_relationships", {}), dict) else {}
        rel = rels.get(character, {}) if isinstance(rels.get(character, {}), dict) else {}
        rows: list[dict] = []

        prefs = []
        categories = profile.get("categories", {}) if isinstance(profile.get("categories"), dict) else {}
        for cat in categories.values():
            if isinstance(cat, list):
                for item in cat:
                    if isinstance(item, dict):
                        trait = str(item.get("trait", "") or "").strip()
                        if trait:
                            prefs.append(trait[:28])
                    elif str(item or "").strip():
                        prefs.append(str(item or "").strip()[:28])
        if not prefs:
            prefs = [str(x or "").strip()[:28] for x in (profile.get("known_preferences", []) if isinstance(profile.get("known_preferences", []), list) else []) if str(x or "").strip()]
        if prefs and (analysis.get("personality_query", False) or analysis.get("explicit_recall_intent", False)):
            rows.append(
                {
                    "item_id": "guest_safe_profile_pref",
                    "text": "开发者给你的整体印象：" + "、".join(prefs[:3]),
                    "score": 0.74,
                    "relevance": 0.72,
                    "cue_match": 0.66,
                    "temporal_score": 0.62,
                    "_confidence": 0.62,
                    "source": "guest_safe_profile_legacy",
                    "memory_meta": _memory_record_meta(
                        memory_type="shared_fact",
                        owner_scope="shared",
                        evidence_tier="medium",
                        source_kind="guest_safe_profile",
                        temporal_validity="persistent",
                        surface_default="light_callback",
                        mode_scope="guest",
                    ),
                }
            )
        pref_char = str(profile.get("preferred_character", "") or "").strip()
        if pref_char and (analysis.get("personality_query", False) or analysis.get("explicit_recall_intent", False)):
            rows.append(
                {
                    "item_id": "guest_safe_profile_pref_char",
                    "text": f"开发者平时更常和{pref_char}聊天。",
                    "score": 0.68,
                    "relevance": 0.63,
                    "cue_match": 0.60,
                    "temporal_score": 0.58,
                    "_confidence": 0.58,
                    "source": "guest_safe_profile_legacy",
                    "memory_meta": _memory_record_meta(
                        memory_type="shared_fact",
                        owner_scope="shared",
                        evidence_tier="medium",
                        source_kind="guest_safe_profile",
                        temporal_validity="persistent",
                        surface_default="light_callback",
                        mode_scope="guest",
                    ),
                }
            )
        typical_hours = str(profile.get("typical_hours", "") or "").strip()
        if typical_hours and bool(re.search(r"(熬夜|作息|几点|晚睡|夜里|半夜)", query)):
            rows.append(
                {
                    "item_id": "guest_safe_profile_hours",
                    "text": f"你对开发者作息的印象比较模糊，但他常在{typical_hours}附近活跃。",
                    "score": 0.55,
                    "relevance": 0.54,
                    "cue_match": 0.40,
                    "temporal_score": 0.36,
                    "_confidence": 0.42,
                    "source": "guest_safe_profile_legacy",
                    "memory_meta": _memory_record_meta(
                        memory_type="shared_fact",
                        owner_scope="shared",
                        evidence_tier="low",
                        source_kind="guest_safe_profile",
                        temporal_validity="persistent",
                        surface_default="light_callback",
                        mode_scope="guest",
                    ),
                }
            )

        familiarity = _safe_float(rel.get("familiarity", 0.0), 0.0)
        trust = _safe_float(rel.get("trust", 0.0), 0.0)
        playfulness = _safe_float(rel.get("playfulness", 0.0), 0.0)
        relation_bits = []
        if familiarity >= 7:
            relation_bits.append("你们很熟")
        elif familiarity >= 4:
            relation_bits.append("你们比较熟悉")
        if trust >= 7:
            relation_bits.append("你挺信任他")
        if playfulness >= 6:
            relation_bits.append("提到他时会自然一点")
        if relation_bits and (analysis.get("personality_query", False) or analysis.get("explicit_recall_intent", False)):
            rows.append(
                {
                    "item_id": "guest_safe_profile_projection",
                    "text": "你对开发者的主观感觉：" + "，".join(relation_bits) + "。",
                    "score": 0.76,
                    "relevance": 0.70,
                    "cue_match": 0.68,
                    "temporal_score": 0.60,
                    "_confidence": 0.65,
                    "source": "guest_safe_profile_legacy",
                    "memory_meta": _memory_record_meta(
                        memory_type="character_projection",
                        owner_scope=f"character:{resolve_agent(character)}",
                        evidence_tier="medium",
                        source_kind="guest_safe_profile",
                        temporal_validity="persistent",
                        surface_default="light_callback",
                        mode_scope="guest",
                    ),
                }
            )

        rows = rows[: max(1, int(max_items or 4))]
        summary_lines = [str(row.get("text", "") or "").strip()[:120] for row in rows if str(row.get("text", "") or "").strip()]
        return {
            "detected": bool(rows),
            "rows": rows,
            "summary_lines": summary_lines[:6],
            "query_analysis": analysis,
            "suppression_reason": "" if rows else "no_safe_profile_match",
            "source": ("legacy_fallback" if rows else ""),
        }

    def _build_guest_safe_profile_view(self, character: str, query_text: str = "", query_bundle: dict | None = None, max_items: int = 4) -> dict:
        query = str(query_text or "").strip()
        analysis = _memory_query_analysis(query, query_bundle, "guest")
        if not analysis.get("developer_focus", False):
            return {
                "detected": False,
                "rows": [],
                "summary_lines": [],
                "query_analysis": analysis,
                "source": "",
                "read_policy": "",
                "legacy_fallback_used": False,
            }
        if analysis.get("meta_instruction_topic", False):
            return {
                "detected": False,
                "rows": [],
                "summary_lines": [],
                "query_analysis": analysis,
                "suppression_reason": "meta_instruction_topic_guard",
                "source": "",
                "read_policy": "meta_topic_no_memory",
                "legacy_fallback_used": False,
            }
        elif analysis.get("guest_developer_experience_query", False):
            return {
                "detected": False,
                "rows": [],
                "summary_lines": [],
                "query_analysis": analysis,
                "suppression_reason": "guest_shared_experience_requires_specific_memory",
                "source": "",
                "read_policy": "deep_history_only",
                "legacy_fallback_used": False,
            }
        if analysis.get("current_status_query", False) and not analysis.get("personality_query", False) and not analysis.get("explicit_recall_intent", False):
            return {
                "detected": False,
                "rows": [],
                "summary_lines": [],
                "query_analysis": analysis,
                "suppression_reason": "current_status_requires_turn_evidence",
                "source": "",
                "read_policy": "stable_only",
                "legacy_fallback_used": False,
            }
        typed_view = self._build_guest_safe_profile_view_from_typed(
            character,
            query,
            query_analysis=analysis,
            max_items=max_items,
        )
        if bool(typed_view.get("detected", False)):
            return typed_view
        return {
            "detected": False,
            "rows": [],
            "summary_lines": [],
            "query_analysis": analysis,
            "suppression_reason": str(typed_view.get("suppression_reason", "") or "no_safe_profile_match"),
            "source": "",
            "read_policy": "stable_only",
            "legacy_fallback_used": False,
        }

    def _decorate_memory_row(self, row: dict | None = None) -> dict:
        item = dict(row or {}) if isinstance(row, dict) else {}
        payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
        infer_payload = payload if isinstance(payload, dict) else {}
        infer_item = item
        if bool(item.get("_hybrid_promoted_layer", False)) and str(item.get("layer", "") or "").strip().lower() == "semantic":
            infer_payload = dict(infer_payload)
            existing_meta = infer_payload.get("memory_meta", {}) if isinstance(infer_payload.get("memory_meta", {}), dict) else {}
            if existing_meta:
                promoted_meta = dict(existing_meta)
                promoted_meta.update(
                    {
                        "memory_type": "shared_fact",
                        "source_kind": "shared_fact",
                        "temporal_validity": "persistent",
                        "surface_default": "silent_grounding",
                        "transient_context_only": False,
                        "promotion_candidate": False,
                    }
                )
                infer_payload["memory_meta"] = promoted_meta
            infer_item = dict(item)
            infer_item["memory_role"] = "semantic"
            infer_item["engram_type"] = "fact"
            infer_item["item_type"] = "fact"
        meta = _infer_memory_meta_from_row(infer_item, infer_payload)
        if _memory_row_is_surfaceable_working(item):
            meta = dict(meta)
            meta["temporal_validity"] = "transient"
            meta["surface_default"] = "continuity_recall"
            meta["promotion_candidate"] = False
        item.update(meta)
        item["memory_meta"] = dict(meta)
        return item
