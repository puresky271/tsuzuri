"""Query analysis, memory metadata inference, policy profiles and grounding signals.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import _embedding_row_matches_current, _safe_float
from tsuzuri.memory_core_impl.memory_rows import (
    _lexical_similarity, _memory_query_allows_system_origin,
    _memory_query_is_meta_instruction_topic, _memory_row_has_durable_turn_signal,
    _memory_row_is_explicit_recall_archive, _recall_focus_text,
)
from tsuzuri.memory_core_impl.features import (
    _canonicalize_developer_fact, _fact_is_dialogue_residue, _infer_topic_labels_from_query,
    _recall_focus_keyword_score, _rewrite_fact_via_llm,
)
from tsuzuri.memory_core_impl.explicit_recall import (
    _memory_explicit_query_has_bounded_slot, _memory_explicit_row_is_answer_bearing,
    _memory_row_evidence_role,
)
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads, _parse_dt
from tsuzuri.persona_registry import agent_alias_view, agent_set_view, collective_scope, live_pattern, persona_regex, resolve_agent

_EXPLICIT_RECALL_RE = re.compile(r"(还记得|记不记得|记得吗|上次|以前|之前|那次|说过|提过|提到|想起|回忆|昨天|前天|前几天|那天)")
_DEVELOPER_FOCUS_RE = live_pattern(r"(开发者|程序员|维护者|作者|{USER})")
_CURRENT_STATUS_RE = re.compile(r"(今天|现在|刚刚|刚才|见到|在吗|在哪|在哪里|几点|此刻|眼下|目前|正在)")
_RECENT_STABLE_STATUS_RE = re.compile(r"(最近|近来|近况|忙|熬夜|作息|状态)")
_PERSONALITY_RE = re.compile(r"(了解|认识|性格|印象|关系|是什么样的人|怎么样的人)")
_SUBJECTIVE_PROJECTION_RE = re.compile(r"(觉得|感觉|印象|担心|在意|关心|信任|熟悉|亲近|贴心|细心)")
_SURFACE_REPLY_RECALL_RE = re.compile(r"(上次|以前|之前|那次|还记得|想起|回想|说过|提过)")
_SCHEDULE_QUERY_RE = re.compile(r"(几点睡|什么时候睡|作息|熬夜|晚睡|夜里|半夜|睡觉时间|睡得早|睡得晚|几点起|什么时候起)")
_GUEST_SELF_FOCUS_RE = re.compile(
    r"(?:你(?:知\s*道|记\s*得|清\s*楚|认\s*识|觉\s*得).{0,12}?我|我(?:平时|一般|通常|最近|现在|今天|是不是|会不会|常不常|到底))"
)
_GUEST_DEV_FOLLOWUP_RE = re.compile(r"(你和他|和他之间|他之间|你们之间|和他.*?(事情|经历|故事)|关于他|他的事)")
_SHARED_EXPERIENCE_RE = re.compile(r"(印象深刻|难忘|发生过什么|共同经历|什么事情|哪件事|有什么事|经历过什么)")

# gossip 审计表字段长度控制
_GOSSIP_AUDIT_PREVIEW_LEN = 60
_GOSSIP_AUDIT_HASH_LEN = 16


def _memory_record_meta(
    *,
    memory_type: str,
    owner_scope: str,
    evidence_tier: str,
    source_kind: str,
    temporal_validity: str,
    surface_default: str,
    mode_scope: str = "normal",
    **extra,
) -> dict:
    data = {
        "memory_type": str(memory_type or "").strip() or "shared_fact",
        "owner_scope": str(owner_scope or "").strip() or "shared",
        "evidence_tier": str(evidence_tier or "").strip() or "low",
        "source_kind": str(source_kind or "").strip() or "runtime",
        "temporal_validity": str(temporal_validity or "").strip() or "persistent",
        "surface_default": str(surface_default or "").strip() or "silent_grounding",
        "mode_scope": str(mode_scope or "").strip() or "normal",
    }
    for key, value in (extra or {}).items():
        if str(key or "").strip():
            data[str(key)] = value
    return data


def _memory_payload_meta(payload: dict | None = None) -> dict:
    row = payload if isinstance(payload, dict) else {}
    meta = row.get("memory_meta", {}) if isinstance(row.get("memory_meta", {}), dict) else {}
    return dict(meta) if isinstance(meta, dict) else {}


def _infer_memory_meta_from_row(row: dict | None = None, payload: dict | None = None) -> dict:
    item = row if isinstance(row, dict) else {}
    pl = payload if isinstance(payload, dict) else {}
    existing = _memory_payload_meta(pl)
    if existing:
        meta = _memory_record_meta(
            memory_type=str(existing.get("memory_type", "") or ""),
            owner_scope=str(existing.get("owner_scope", "") or ""),
            evidence_tier=str(existing.get("evidence_tier", "") or ""),
            source_kind=str(existing.get("source_kind", "") or ""),
            temporal_validity=str(existing.get("temporal_validity", "") or ""),
            surface_default=str(existing.get("surface_default", "") or ""),
            mode_scope=str(existing.get("mode_scope", "normal") or "normal"),
        )
        for key, value in existing.items():
            if key not in meta:
                meta[key] = value
        return meta
    layer = str(item.get("layer", "") or "").strip().lower()
    role = str(item.get("memory_role", "") or "").strip().lower()
    engram = str(item.get("engram_type", "") or "").strip().lower()
    text = str(item.get("text", "") or "").strip()
    confidence = _safe_float(item.get("confidence", 0.5), 0.5)
    if _memory_row_is_explicit_recall_archive(item, pl):
        return _memory_record_meta(
            memory_type="explicit_recall_archive",
            owner_scope="shared",
            evidence_tier=("medium" if confidence >= 0.58 else "low"),
            source_kind="source_backed_archive",
            temporal_validity="archival",
            surface_default="explicit_recall",
            promotion_candidate=False,
        )
    if layer == "working" or role == "working":
        return _memory_record_meta(
            memory_type="working_turn",
            owner_scope=f"character:{resolve_agent(item.get('character', ''))}",
            evidence_tier="medium",
            source_kind="turn_exchange",
            temporal_validity="transient",
            surface_default="silent_grounding",
            transient_context_only=True,
            promotion_candidate=False,
        )
    if role == "collective" or str(item.get("character", "") or "").strip() == collective_scope():
        return _memory_record_meta(
            memory_type="collective_event",
            owner_scope="collective",
            evidence_tier=("high" if confidence >= 0.72 else "medium"),
            source_kind="collective_event",
            temporal_validity="recency_weighted",
            surface_default="silent_grounding",
        )
    if layer == "episodic":
        return _memory_record_meta(
            memory_type="episodic_event",
            owner_scope=f"character:{resolve_agent(item.get('character', ''))}",
            evidence_tier=("medium" if confidence >= 0.58 else "low"),
            source_kind="episodic_event",
            temporal_validity="recency_weighted",
            surface_default="light_callback",
        )
    if layer == "semantic":
        if _DEVELOPER_FOCUS_RE.search(text) and (_SUBJECTIVE_PROJECTION_RE.search(text) or engram == "projection"):
            return _memory_record_meta(
                memory_type="character_projection",
                owner_scope=f"character:{resolve_agent(item.get('character', ''))}",
                evidence_tier=("medium" if confidence >= 0.60 else "low"),
                source_kind="character_projection",
                temporal_validity="persistent",
                surface_default="light_callback",
            )
        return _memory_record_meta(
            memory_type="shared_fact",
            owner_scope="shared",
            evidence_tier=("medium" if confidence >= 0.64 else "low"),
            source_kind="shared_fact",
            temporal_validity="persistent",
            surface_default="silent_grounding",
        )
    if layer == "procedural" or role == "control":
        return _memory_record_meta(
            memory_type="suppressed_candidate",
            owner_scope=f"character:{resolve_agent(item.get('character', ''))}",
            evidence_tier="medium",
            source_kind="strategy_rule",
            temporal_validity="persistent",
            surface_default="abstain",
        )
    return _memory_record_meta(
        memory_type="shared_fact",
        owner_scope="shared",
        evidence_tier=("medium" if confidence >= 0.64 else "low"),
        source_kind="runtime",
        temporal_validity="persistent",
        surface_default="silent_grounding",
    )


def _memory_query_analysis(query_text: str, query_bundle: dict | None = None, read_scope: str = "normal") -> dict:
    qb = query_bundle if isinstance(query_bundle, dict) else {}
    query = str(query_text or "").strip()
    recent_user_msgs = [str(x or "").strip() for x in (qb.get("recent_user_msgs", []) if isinstance(qb.get("recent_user_msgs", []), list) else []) if str(x or "").strip()]
    recent_assistant_msgs = [str(x or "").strip() for x in (qb.get("recent_assistant_msgs", []) if isinstance(qb.get("recent_assistant_msgs", []), list) else []) if str(x or "").strip()]
    recent_blob = " ".join((recent_user_msgs + recent_assistant_msgs)[-5:])
    intent_bucket = str(qb.get("intent_bucket", "") or "").strip().lower()
    temporal_mode = str(qb.get("temporal_query_mode", "recent_general") or "recent_general").strip().lower()
    # Score-based explicit recall detection (not just regex bool)
    # 2026-05-16 修：删 ("说的", 0.5) — 太通用、命中"你说的没错/他说的对/这就是说的"等
    # 日常虚词、不是回忆查询。日志实测「嗯，你说的没错」误触发 explicit_recall。
    # 同步加 phrase 级精确匹配："你说过的"/"上次说的"/"之前说的" 这种真指向过去的才算。
    _recall_keywords = [
        ("还记得", 0.9), ("记不记得", 0.9), ("记得吗", 0.85), ("想起", 0.8), ("回忆", 0.8),
        ("上次", 0.6), ("以前", 0.5), ("之前", 0.5), ("那次", 0.7), ("那天", 0.6),
        ("说过", 0.7), ("提过", 0.7), ("提到", 0.6), ("提起", 0.6),
        ("昨天", 0.5), ("前天", 0.5), ("前几天", 0.5), ("上回", 0.65),
        ("聊过", 0.7), ("谈过", 0.7), ("讨论过", 0.7),
        ("你说过的", 0.7), ("上次说的", 0.7), ("之前说的", 0.7), ("你说的那", 0.6),
        ("有后续", 0.6), ("怎么样了", 0.5), ("后来", 0.4), ("结果", 0.4),
        ("来着", 0.5),
    ]
    _recall_score = 0.0
    for _kw, _kwscore in _recall_keywords:
        if _kw in query:
            _recall_score = max(_recall_score, _kwscore)
    # 2026-05-16 修：短 query (< 8 字) 防 ack 误判。
    # 「嗯」「好好好，知道了」「你说的没错」等纯同意/打招呼短句即使命中 0.4-0.6
    # 弱信号、也不是真的回忆查询。只有命中高信号 (>=0.7) 才算明确回忆意图。
    # ≥ 8 字时按原规则、保 backwards compat。
    if len(query.strip()) < 8:
        explicit_recall = bool(_recall_score >= 0.7)
    else:
        explicit_recall = bool(_recall_score >= 0.4)
    # Also store the score for fine-grained use
    developer_focus = bool(_DEVELOPER_FOCUS_RE.search(query))
    contextual_developer_focus = False
    if (not developer_focus) and (read_scope == "guest") and recent_blob and _DEVELOPER_FOCUS_RE.search(recent_blob):
        if _GUEST_DEV_FOLLOWUP_RE.search(query):
            developer_focus = True
            contextual_developer_focus = True
    current_status_query = bool(_CURRENT_STATUS_RE.search(query))
    stable_status_query = bool(_RECENT_STABLE_STATUS_RE.search(query))
    schedule_query = bool(_SCHEDULE_QUERY_RE.search(query))
    shared_experience_query = bool(_SHARED_EXPERIENCE_RE.search(query))
    personality_query = bool(_PERSONALITY_RE.search(query)) and (not shared_experience_query)
    self_focus_query = bool(_GUEST_SELF_FOCUS_RE.search(query)) and (not developer_focus)
    fact_like = bool(intent_bucket in ("retrieve", "verify", "planning", "fact")) or current_status_query or stable_status_query
    emotional_query = bool(intent_bucket in ("emotional", "support", "deep_talk"))
    meta_instruction_topic = _memory_query_is_meta_instruction_topic(query)
    system_origin_query = _memory_query_allows_system_origin(query)
    guest_self_fact_query = bool(
        read_scope == "guest"
        and self_focus_query
        and fact_like
        and (not personality_query)
        and (not explicit_recall)
    )
    guest_developer_experience_query = bool(
        read_scope == "guest"
        and developer_focus
        and shared_experience_query
        and (not personality_query)
    )
    return {
        "query_text": query[:240],
        "intent_bucket": intent_bucket,
        "temporal_query_mode": temporal_mode,
        "explicit_recall_intent": explicit_recall,
        "explicit_recall_score": round(_recall_score, 2),
        "developer_focus": developer_focus,
        "contextual_developer_focus": contextual_developer_focus,
        "current_status_query": current_status_query,
        "stable_status_query": stable_status_query,
        "personality_query": personality_query,
        "schedule_query": schedule_query,
        "shared_experience_query": shared_experience_query,
        "self_focus_query": self_focus_query,
        "guest_self_fact_query": guest_self_fact_query,
        "guest_developer_experience_query": guest_developer_experience_query,
        "fact_like_query": fact_like,
        "emotional_query": emotional_query,
        "meta_instruction_topic": meta_instruction_topic,
        "system_origin_query": system_origin_query,
        "strict_focus": bool(read_scope in ("guest", "pure") or developer_focus or fact_like or explicit_recall),
    }


# Public cross-module memory predicates. Keep the private implementations for
# internal callers and extensions that still use the historical spellings.
embedding_row_matches_current = _embedding_row_matches_current
lexical_similarity = _lexical_similarity
recall_focus_text = _recall_focus_text
memory_row_has_durable_turn_signal = _memory_row_has_durable_turn_signal
infer_topic_labels_from_query = _infer_topic_labels_from_query
canonicalize_developer_fact = _canonicalize_developer_fact
fact_is_dialogue_residue = _fact_is_dialogue_residue
rewrite_fact_via_llm = _rewrite_fact_via_llm
recall_focus_keyword_score = _recall_focus_keyword_score
memory_row_evidence_role = _memory_row_evidence_role
memory_explicit_row_is_answer_bearing = _memory_explicit_row_is_answer_bearing
memory_explicit_query_has_bounded_slot = _memory_explicit_query_has_bounded_slot
memory_query_analysis = _memory_query_analysis

_MIGRATION_DOMAIN_PATHS = [
    "today",
    "recent_days",
    "long_term.key_events",
    "long_term.developer_profile",
    "long_term.character_relationships",
    # gossip_queue 自 2026-08-14 退役：gossip 存储 SSOT 迁往 shared_info 表
    # （见 gossip_runtime.py）；保留该域仅为兼容存量快照数据，不再有新写入。
    "gossip_queue",
    "offline_diary",
    "daily_digest",
    "developer_diary",
    "pending_tasks",
    "completed_tasks",
    "user_emotion_log",
    "user_emotion_trend",
    "self_eval_log",
    "self_eval_stats",
    "self_eval_exemplars",
    "_topic_fingerprints",
    "_micro_reflections",
    "_town_projection",
    "_town_mailbox_seen",
]

_NATIVE_MECHANISM_PATHS = list(_MIGRATION_DOMAIN_PATHS)

_MECHANISM_PHASES = {
    # Phase 1: high-frequency interaction memory
    "gossip_queue": "phase1",
    "offline_diary": "phase1",
    "daily_digest": "phase1",
    "developer_diary": "phase1",
    "pending_tasks": "phase1",
    "completed_tasks": "phase1",
    "user_emotion_log": "phase1",
    "user_emotion_trend": "phase1",
    "_topic_fingerprints": "phase1",
    "_micro_reflections": "phase1",
    "_town_projection": "phase1",
    "_town_mailbox_seen": "phase1",
    # Phase 2: long-term profile and self-eval
    "today": "phase2",
    "recent_days": "phase2",
    "long_term.key_events": "phase2",
    "long_term.developer_profile": "phase2",
    "long_term.character_relationships": "phase2",
    "self_eval_log": "phase2",
    "self_eval_stats": "phase2",
    "self_eval_exemplars": "phase2",
}

_DEFAULT_POLICY_PROFILE_V2 = {
    "version": "v2",
    "retrieval_policy": {
        "weights": {
            "lexical_weight": 0.30,
            "recency_weight": 0.18,
            "importance_weight": 0.16,
            "source_weight": 0.08,
            "novelty_weight": 0.08,
            "llm_weight": 0.20,
            "temporal_weight": 0.10,
            "spatial_weight": 0.08,
            "cue_match_weight": 0.12,
            "embedding_weight": 0.18,
            "motivation_weight": 0.08,
        },
        "min_relevance": 0.10,
        "mmr_lambda": 0.65,
        "llm_selector": {
            "enabled": True,
            "top_k": 8,
        },
    },
    "evolution_policy": {
        "max_actions": 8,
        "duplicate_sim_threshold": 0.78,       # 降低: 0.90→0.78 捕获语义相近的记忆
        "promote_salience_min": 0.72,           # 当前产品阈值：避免单次触达永久化
        "promote_access_min": 2,                # 当前产品阈值：要求重复真实复用
        "stale_days": 30,                       # 30天未访问才归档（间歇使用友好）
        "stale_importance_max": 0.35,           # importance < 0.35 才归档（保护更多记忆）
        "stale_access_max": 1,
        "rewrite_min_chars": 120,               # 降低: 180→120 更多长记忆可压缩
        "rewrite_confidence_max": 0.55,
    },
    "proactive_policy": {
        "same_session_window_sec": 20 * 60,
        "global_window_sec": 2 * 3600,
        "max_sent_same_session": 1,
        "max_sent_global": 3,
        "idle_followup_min_idle_sec": 8 * 60,
        "risk_block_threshold": 0.66,
        "quality_block_threshold": 0.36,
        "min_send_interval_sec": 10 * 60,
        "schedule_conflict_block_threshold": 0.60,
    },
    "safety_fallbacks": {
        "hard_safety": {
            "block_goodbye": True,
            "block_if_user_last_speaker": True,
            "block_high_risk_silent_threshold": 0.92,
        },
        "on_llm_error": "conservative_block",
        "on_llm_invalid_json": "conservative_block",
    },
}

_MEMORY_ROLE_MAP = {
    "working": "working",
    "episodic": "episodic",
    "semantic": "semantic",
    "procedural": "control",
    "collective": "collective",
}

_COLLECTIVE_SCOPE_MEMBERS = agent_set_view()

# Phase Q.X (2026-05-04): 5 主角的名字别名集合 — 用于 cross_char_daily_digest
# query 命中检测、给"told_by 是被提及角色"的转述记忆 +0.10 boost。
# 包含中文名 / 罗马音 / 昵称 / 「ちゃん」尾词。注意"角色B"单字粒度低、可能误命中
# （如"灯笼"），简版先实现、看实际命中率再细化（plan Phase 4 risk 已记录）。
_CHARACTER_NAME_ALIASES = agent_alias_view()


def _query_mentions_character(query: str, char: str) -> bool:
    """query 文本是否提及某主角（命中任一别名即 True）。"""
    aliases = _CHARACTER_NAME_ALIASES.get(str(char or "").strip(), [str(char or "")])
    q = str(query or "").lower()
    return any(a.lower() in q for a in aliases if a)

_CANON_SOURCE_WEIGHTS = {
    "official_bang_dream_sources": 1.0,
    "moegirl_sources": 0.72,
}
_CANON_CLAIM_TYPES = {
    "speech_style",
    "lexical_marker",
    "interaction_style",
    "school_behavior",
    "group_behavior",
    "persona_boundary",
}

_GUEST_DEEP_EXPERIENCE_MIN_AGE_HOURS = 36.0


def _stable_hash(v) -> str:
    return hashlib.md5(_json_dumps(v).encode("utf-8", errors="ignore")).hexdigest()[:12]


def _tokenize_grounding_text(text: str) -> list[str]:
    s = str(text or "").strip().lower()
    if not s:
        return []
    parts = re.findall(r"[\w\u4e00-\u9fff:]+", s)
    return [p for p in parts if p]


def _overlap_ratio(a: list[str], b: list[str]) -> float:
    sa = {str(x or "").strip() for x in (a or []) if str(x or "").strip()}
    sb = {str(x or "").strip() for x in (b or []) if str(x or "").strip()}
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb)) / float(max(1, len(sa | sb)))


def _normalize_time_anchor(anchor: str) -> str:
    a = str(anchor or "").strip().lower()
    if a in ("immediate", "near_future", "future", "recent", "historical"):
        return a
    return "recent"


def _time_anchor_compat(query_anchor: str, item_anchor: str) -> float:
    qa = _normalize_time_anchor(query_anchor)
    ia = _normalize_time_anchor(item_anchor)
    if qa == ia:
        return 1.0
    nearby = {
        ("immediate", "recent"),
        ("near_future", "future"),
        ("recent", "historical"),
    }
    if (qa, ia) in nearby or (ia, qa) in nearby:
        return 0.62
    return 0.28


def _derive_time_anchor(text: str, payload: dict | None = None) -> str:
    src = f"{str(text or '')} {_json_dumps(payload if isinstance(payload, dict) else {})}".lower()
    if re.search(r"(明天|稍后|等会|待会|之后|下次|下周|next|later|tomorrow)", src):
        return "near_future"
    if re.search(r"(现在|刚刚|正在|今天|目前|this morning|today|now)", src):
        return "immediate"
    if re.search(r"(以前|之前|上次|去年|曾经|过去|yesterday|last week|before)", src):
        return "historical"
    return "recent"


def _derive_grounding_signals(query_text: str, query_bundle: dict | None, payload: dict | None = None) -> dict:
    qb = query_bundle if isinstance(query_bundle, dict) else {}
    pl = payload if isinstance(payload, dict) else {}
    query_text_s = str(query_text or "").strip()
    query_anchor = _normalize_time_anchor(str(qb.get("time_anchor", "") or "recent"))
    query_tokens = _tokenize_grounding_text(query_text_s)
    query_entities = [str(x or "").strip().lower() for x in (qb.get("entities", []) if isinstance(qb.get("entities", []), list) else []) if str(x or "").strip()]
    payload_text = " ".join(
        [
            str(pl.get("location", "") or ""),
            str(pl.get("activity", "") or ""),
            str(pl.get("school", "") or ""),
            str(pl.get("place_name", "") or ""),
        ]
    ).strip()
    payload_tokens = _tokenize_grounding_text(payload_text)
    payload_entities = [x for x in payload_tokens if len(x) >= 2][:10]
    item_anchor = _derive_time_anchor(query_text_s, pl)
    return {
        "query_anchor": query_anchor,
        "item_anchor": item_anchor,
        "query_tokens": query_tokens[:16],
        "payload_tokens": payload_tokens[:16],
        "query_entities": query_entities[:10],
        "payload_entities": payload_entities[:10],
    }


def _derive_text(v) -> str:
    if isinstance(v, str):
        return v[:320]
    if not isinstance(v, dict):
        return str(v)[:320]
    for k in ("text", "summary", "event", "task", "content", "about", "raw", "feeling", "emotion", "title"):
        s = str(v.get(k, "") or "").strip()
        if s:
            return s[:320]
    return _json_dumps(v)[:320]


def _row_primary_dt(row: dict | None = None) -> datetime | None:
    item = row if isinstance(row, dict) else {}
    cue = _json_loads(str(item.get("cue_json", "") or "{}"), {})
    if isinstance(cue, dict):
        for key in ("at", "ts", "created_at", "updated_at"):
            dt = _parse_dt(str(cue.get(key, "") or ""))
            if dt is not None:
                return dt
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    if isinstance(payload, dict):
        dt = _parse_dt(_derive_ts(payload))
        if dt is not None:
            return dt
    for key in ("created_at", "updated_at", "last_accessed_at"):
        dt = _parse_dt(str(item.get(key, "") or ""))
        if dt is not None:
            return dt
    return None


def _is_recent_guest_experience_row(row: dict | None = None, min_age_hours: float = _GUEST_DEEP_EXPERIENCE_MIN_AGE_HOURS) -> bool:
    item = row if isinstance(row, dict) else {}
    temporal_validity = str(item.get("temporal_validity", "") or "").strip().lower()
    if temporal_validity == "transient":
        return True
    dt = _row_primary_dt(item)
    if dt is not None:
        age_h = max(0.0, (utc_now_naive() - dt).total_seconds() / 3600.0)
        return age_h < float(min_age_hours)
    text = str(item.get("text", "") or "").strip()
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    anchor = _derive_time_anchor(text, payload if isinstance(payload, dict) else {})
    if anchor == "immediate":
        return True
    return temporal_validity in ("recency_weighted", "")


def _derive_ts(v) -> str:
    if not isinstance(v, dict):
        return ""
    for k in ("at", "timestamp", "updated_at", "completed_at", "created_at", "to", "from"):
        ts = str(v.get(k, "") or "").strip()
        if ts:
            return ts
    date = str(v.get("date", "") or "").strip()
    time_v = str(v.get("time", "") or "").strip()
    if date and time_v:
        return f"{date}T{time_v}"
    return date


def _derive_weight(v) -> float:
    if not isinstance(v, dict):
        return 0.0
    for k in ("importance", "overall", "intensity", "ew", "weight", "trust", "confidence"):
        vv = v.get(k, None)
        if vv is None:
            continue
        try:
            return float(vv)
        except Exception:
            continue
    return 0.0


def _derive_confidence(v) -> float:
    if not isinstance(v, dict):
        return 0.0
    for k in ("confidence", "trust", "strength", "reliability"):
        vv = v.get(k, None)
        if vv is None:
            continue
        try:
            return _clamp(float(vv), 0.0, 1.0)
        except Exception:
            continue
    return 0.0


def _parse_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if not s:
        return bool(default)
    return s not in ("0", "false", "off", "no")


def _canon_weight_for_source(source_type: str) -> float:
    return float(_CANON_SOURCE_WEIGHTS.get(str(source_type or "").strip(), 0.55))


def _deep_merge_dict(base: dict, override: dict) -> dict:
    out = _json_loads(_json_dumps(base), {})
    if not isinstance(out, dict):
        out = {}
    src = override if isinstance(override, dict) else {}
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k), v)
        else:
            out[k] = _json_clone(v)
    return out


def _normalize_policy_profile(raw_cfg: dict | None) -> dict:
    cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    if str(cfg.get("version", "") or "").strip().lower() == "v2":
        return _deep_merge_dict(_DEFAULT_POLICY_PROFILE_V2, cfg)

    # Backward compatibility: lift v1 keys into v2 sections.
    out = _deep_merge_dict(_DEFAULT_POLICY_PROFILE_V2, {})
    if isinstance(cfg.get("retrieval"), dict):
        out["retrieval_policy"]["weights"] = _deep_merge_dict(out["retrieval_policy"].get("weights", {}), cfg.get("retrieval", {}))
    if isinstance(cfg.get("evolution"), dict):
        out["evolution_policy"] = _deep_merge_dict(out.get("evolution_policy", {}), cfg.get("evolution", {}))
    for k in ("retrieval_policy", "evolution_policy", "proactive_policy", "safety_fallbacks"):
        if isinstance(cfg.get(k), dict):
            out[k] = _deep_merge_dict(out.get(k, {}), cfg.get(k, {}))
    out["version"] = "v2"
    return out


def _json_clone(v):
    return _json_loads(_json_dumps(v), None)


def _path_get(data: dict, path: str):
    cur = data if isinstance(data, dict) else {}
    for p in [x for x in str(path or "").split(".") if str(x or "").strip()]:
        if not isinstance(cur, dict):
            return None
        if p not in cur:
            return None
        cur = cur.get(p)
    return cur


def _path_set(data: dict, path: str, value) -> None:
    if not isinstance(data, dict):
        return
    parts = [x for x in str(path or "").split(".") if str(x or "").strip()]
    if not parts:
        return
    cur = data
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = _json_clone(value)


# ── 事实验证层：防止 LLM 幻觉写入 semantic 记忆 ──

import re as _fv_re

# 角色自身状态描述模式——这些不需要 user_text 验证
_FACT_SELF_STATE_RE = _fv_re.compile(
    r"^(喜欢|善于|在意|珍视|对时间|会主动|关心|对食物|独立|谦虚|做事|礼貌|说话|分享)",
)

# 提及用户做了具体事情的模式——需要 user_text 验证
_FACT_USER_ACTION_RE = live_pattern(
    r"(用户|开发者|{USER}|他|对方|你).{0,6}(说|提到|做|吃|去|买|看|玩|来|走|学|写|发|问|找|试|用|参加|通宵|打游戏|面试|工作|出门|回家|下雨)",
)


def _verify_fact_grounding(fact: str, user_text: str) -> str:
    """检查 semantic fact 是否有 user_text 中的依据。
    返回: 'grounded' / 'inferred' / 'ungrounded'
    """
    fact = str(fact or "").strip()
    user = str(user_text or "").strip().lower()
    if not fact:
        return "inferred"

    # 角色自身状态/话题标签 → 允许（不涉及用户事实）
    if _FACT_SELF_STATE_RE.search(fact):
        return "inferred"

    # 不涉及用户行为 → 允许
    if not _FACT_USER_ACTION_RE.search(fact):
        return "inferred"

    # 涉及用户行为 → 检查 user_text 中是否有关键词支撑
    # 人称归一化：fact 中的"用户/开发者/用户/他/对方"在 user_text 中对应"我"
    _fact_normalized = persona_regex(r"(用户|开发者|{USER}|他|对方)").sub("", fact.lower())
    # 提取 2-gram（滑动窗口2字）+ 完整词（2字以上连续中文）
    _stop = {"可能", "常常", "经常", "今天", "昨天", "最近", "已经", "一直", "比较", "非常", "有点", "而且", "但是", "因为", "所以"}
    _cn_chars = _fv_re.findall(r"[\u4e00-\u9fff]+", _fact_normalized)
    _bigrams = set()
    for seg in _cn_chars:
        if seg in _stop:
            continue
        for i in range(len(seg) - 1):
            bg = seg[i:i+2]
            if bg not in _stop:
                _bigrams.add(bg)
    if not _bigrams:
        return "inferred"

    # 至少有 1 个 bigram 在 user_text 中出现
    _matched = sum(1 for bg in _bigrams if bg in user)
    if _matched >= 1:
        return "grounded"

    return "ungrounded"
