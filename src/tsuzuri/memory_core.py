# Compatibility facade: implementation lives in the ``memory_core_impl`` package.
# Every former top-level name is re-exported here; attribute writes on this
# module (tests' ``patch.object(memory_core, ...)``) propagate to the submodules.
from __future__ import annotations

from tsuzuri.memory_core_impl.config import (
    _current_embedding_spec, _emb_cos, _emb_declared_dim, _emb_declared_model_id, _emb_decode,
    _emb_encode, _emb_encode_both, _emb_encode_dense, _emb_encode_sparse, _emb_is_enabled,
    _emb_is_m3_mode, _emb_sparse_dot, _emb_sparse_to_text, _emb_text_to_sparse, _emb_to_text,
    _embedding_callable, _embedding_engine_available, _embedding_import_error,
    _embedding_row_matches_current, _safe_float, _safe_int, _utc_now_iso,
    mem_actr_recency_enabled, mem_context_dual_read_enabled, mem_context_primary_enabled,
    mem_core_summary_enabled, mem_core_summary_render_enabled, mem_evolve_min_shadow_days,
    mem_evolve_shadow_only, mem_forgetting_apply_enabled, mem_forgetting_enabled,
    mem_json_bootstrap_fallback_enabled, mem_json_online_export_enabled,
    mem_lateral_inhibition_enabled, mem_llm_selector_enabled, mem_motivation_boost_enabled,
    mem_policy_llm_enabled, mem_rethink_enabled, mem_rethink_intraday_enabled, mem_rethink_k,
    mem_rethink_max_actions, mem_rethink_max_iterations, mem_rethink_min_shadow_days,
    mem_rethink_ooc_threshold, mem_rethink_shadow_only, mem_snapshot_backup_enabled,
    mem_snapshot_backup_interval_hours, mem_snapshot_backup_keep,
    mem_spreading_activation_enabled, mem_sqlite_enabled,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _ASSISTANT_ONLY_FACT_RE, _bigrams, _bm25_score_pair, _COLD_START_IDENTITY_RE,
    _COLLECTIVE_ADDRESS_RE, _collective_row_third_person_self_referent, _DISCLOSURE_USER_RE,
    _DURABLE_COMMITMENT_RE, _EMOTION_CONNECTION_OR_PREFERENCE_ALONE_RE,
    _EMOTION_TEMPORAL_ANCHOR, _EMOTION_WORDS_CONNECTION, _EMOTION_WORDS_PREFERENCE,
    _EMOTION_WORDS_SEEKING, _EMOTION_WORDS_VULNERABILITY, _EMOTIONAL_REACTIVE_RE,
    _EMOTIONAL_VALUE_BOUND_RE, _EPHEMERAL_COMMAND_RE, _EPHEMERAL_GREETING_RE,
    _EPHEMERAL_MEMORY_TEXT_RE, _EPHEMERAL_TRANSIENT_RE,
    _EVOLUTION_SEMANTIC_PROMOTION_BLOCKED_TYPES, _EXPLICIT_MEMORY_INTENT_RE,
    _EXPLICIT_MEMORY_WRITE_RE, _FOCUS_GENERIC_CHUNKS, _GENERIC_MEMORY_SURFACE_EXACT,
    _GENERIC_MEMORY_SURFACE_RE, _GENERIC_RE, _lexical_similarity,
    _LOW_VALUE_MEMORY_GENERALIZATION_RE, _memory_explicit_archive_matches_query,
    _memory_payload_evidence_policy, _memory_query_allows_system_origin,
    _memory_query_hits_removed_recall_tombstone, _memory_query_is_meta_instruction_topic,
    _memory_row_age_hours, _memory_row_allows_evolution_semantic_promotion,
    _memory_row_blocked_by_meta_topic, _memory_row_has_durable_turn_signal,
    _memory_row_is_auto_fact_observation, _memory_row_is_explicit_recall_archive,
    _memory_row_is_surfaceable_working, _memory_row_is_trusted_fact_observation,
    _memory_row_is_working, _memory_snapshot_text_index, _memory_surface_text_is_generic,
    _memory_text_is_low_value_generalization, _memory_working_text_is_ephemeral_noise,
    _NEGATED_DURABLE_COMMITMENT_RE, _norm_text, _PATH4_CHITCHAT_FACT_RE,
    _PATH4_ONESHOT_PROMISE_RE, _PATH4_TRANSIENT_DAILY_RE,
    _RARE_UPPER_TOKEN_RE, _REASSURANCE_ASSISTANT_RE, _recall_focus_text, _RECALL_NOISE_RE,
    _RECALL_QUOTED_RE, _rrf_fuse, _SPECIFICITY_PLACE_RE, _SPECIFICITY_PROPER_RE,
    _SPECIFICITY_TIME_RE, _STANCE_RE,
    _SYSTEM_INSTRUCTION_TOPIC_RE, _SYSTEM_META_TOPIC_RE, _SYSTEM_ORIGIN_RECALL_ANCHOR_RE,
    _text_mentions_system_origin,
    _TOPIC_ONLY_FACT_RE, _turn_fact_card_has_durable_commitment, _USER_FACT_NEGATIVE_RE,
    _USER_FACT_PATTERN_RE, _VAGUE_META_TOPIC_RE, _WORKING_EPHEMERAL_NOISE_RE, use_rrf_fusion,
)
from tsuzuri.memory_core_impl.features import (
    _baseline_subject_normalize, _candidate_is_memory_probe_text, _canonicalize_developer_fact,
    _compute_digit_density, _compute_embedding_novelty, _compute_rare_char_ratio,
    _compute_typed_score, _dedupe_short_strings, _derive_semantic_facts_from_user_text,
    _detect_negation, _developer_display_name, _EMOTIONAL_DIRECT_SEMANTIC_LABELS,
    _EMOTIONAL_DIRECT_SEMANTIC_MIN_SALIENCE, _ENTITY_EXTRACT_RES, _extract_entities_from_text,
    _extract_memory_features, _extract_topic_tags, _fact_is_dialogue_residue,
    _fact_is_durable_from_user, _FACT_RESIDUE_LEAD_FILLER_RE, _FACT_RESIDUE_QUESTION_RE,
    _FACT_RESIDUE_SUBJECT_PREFIX_RE, _FACT_REWRITE_PROMPT_TEMPLATE,
    _infer_topic_labels_from_query, _is_cold_start_fast_track, _is_collective_memory_text,
    _is_reuse_worthy_topic_overlap, _NEGATION_RE, _recall_focus_keyword_score,
    _rewrite_fact_via_llm, _TOPIC_CATEGORIES, _TOPIC_DEVELOPER_RE, _TOPIC_EMOTION_RE,
    _topic_from_semantic_fact, _TOPIC_HOBBY_RE, _TOPIC_LABEL_KEYWORDS, _TOPIC_LIFE_VALUE_RE,
    _topic_overlap, _TOPIC_RELATIONSHIP_SELF_RE, _TOPIC_SCHEDULE_RE, _TOPIC_STANCE_VALUE_RE,
    _TOPIC_WORK_STUDY_RE, _typed_gate_decision, CONSOLIDATION_SCORE_CANDIDATE_MIN,
    CONSOLIDATION_SCORE_FAST_TRACK, CONSOLIDATION_SCORE_PROMOTE,
    CONSOLIDATION_STATE_ARCHIVE_CANDIDATE, CONSOLIDATION_STATE_CANDIDATE,
    CONSOLIDATION_STATE_KEY_EVENT_PROMOTED, CONSOLIDATION_STATE_PROFILE_PROMOTED,
    CONSOLIDATION_STATE_RECALL_ARCHIVE, CONSOLIDATION_STATE_STALE_ARCHIVE,
    CONSOLIDATION_STATE_TRANSIENT, CONSOLIDATION_VALID_STATES,
)
from tsuzuri.memory_core_impl.explicit_recall import (
    _apply_memory_correction_dominance, _BIRTHDAY_DATE_ASK_RE, _BIRTHDAY_FACT_RE,
    _dedupe_memory_rows_by_item_id, _direct_recall_match_score,
    _drink_additive_positive_answer, _DRINK_ADDITIVE_QUERY_RE, _EXPLICIT_DATE_ANSWER_RE,
    _EXPLICIT_DATE_QUERY_RE, _EXPLICIT_INTERROGATIVE_RE, _explicit_recall_answer_budget,
    _EXPRESSION_ANSWER_RE, _EXPRESSION_QUERY_RE, _FLAVOR_PREFERENCE_QUERY_RE,
    _FORGETTING_ELIGIBLE_LAYERS, _FORGETTING_PROTECT_ENGRAM_TYPES, _forgetting_protected,
    _GENERIC_QUERY_NOISE_RE, _GENERIC_QUERY_PREDICATE_RE, _HOMETOWN_PLACE_ANSWER_RE,
    _LEGACY_CORRECTION_RE, _memory_explicit_query_has_bounded_slot,
    _memory_explicit_row_is_answer_bearing, _memory_generic_query_answer_matches,
    _memory_query_answer_anchors, _memory_row_can_surface_nonexplicit,
    _memory_row_canonical_key, _memory_row_effective_timestamp,
    _memory_row_effective_timestamp_key, _memory_row_evidence_role, _memory_row_score,
    _memory_should_run_expensive_reranker, _profile_promoted_probe_user_answer,
    _query_requires_first_person_fact_ownership, _select_explicit_surface_rows,
)
from tsuzuri.memory_core_impl.recall_scoring import (
    _ACT_R_DECAY, _ACT_R_NOISE, _ACT_R_THRESHOLD, _actr_recency_score,
    _apply_affect_memory_boost, _apply_lateral_inhibition, _apply_spreading_activation,
    _bigram_jaccard, _clamp, _CLAUSE_CONJUNCTION_IDIOMS, _CLAUSE_CONJUNCTION_MIN_SIDE_LEN,
    _CLAUSE_CONJUNCTIONS, _CLAUSE_LEAD_RE, _CLAUSE_PUNCT_RE, _CO_STRENGTH_CACHE,
    _CO_STRENGTH_DEFAULT, _co_strength_key, _co_strength_load_from_sqlite, _CO_STRENGTH_LR,
    _CO_STRENGTH_MAX_ENTRIES, _co_strength_persist, _compute_motivation_boost,
    _expand_query_semantically, _extract_payload_emotional_weight, _extract_query_keywords,
    _find_co_occurring, _get_co_store, _get_co_strength, _HAS_JIEBA, _hebbian_update,
    _json_dumps, _json_loads, _json_safe_meta, _LATERAL_INHIBIT_SIM_THRESHOLD,
    _llm_expand_query, _memory_recency_timestamp, _parse_dt,
    _payload_marks_non_retrievable_present_state, _QUERY_KEYWORD_MAX_LEN,
    _QUERY_KEYWORD_MIN_LEN, _QUERY_KEYWORD_STOP, _recency_score, _rule_expand_query,
    _semantic_query_expand_enabled, _source_prior, _split_query_clauses,
    _split_query_semantics, _tokenize_goal_words, _TOPIC_EXPANSIONS, _word_jaccard,
)
from tsuzuri.memory_core_impl.meta_policy import (
    _CANON_CLAIM_TYPES, _CANON_SOURCE_WEIGHTS, _canon_weight_for_source,
    _CHARACTER_NAME_ALIASES, _COLLECTIVE_SCOPE_MEMBERS, _CURRENT_STATUS_RE, _deep_merge_dict,
    _DEFAULT_POLICY_PROFILE_V2, _derive_confidence, _derive_grounding_signals, _derive_text,
    _derive_time_anchor, _derive_ts, _derive_weight, _DEVELOPER_FOCUS_RE, _EXPLICIT_RECALL_RE,
    _FACT_SELF_STATE_RE, _FACT_USER_ACTION_RE, _GOSSIP_AUDIT_HASH_LEN,
    _GOSSIP_AUDIT_PREVIEW_LEN, _GUEST_DEEP_EXPERIENCE_MIN_AGE_HOURS, _GUEST_DEV_FOLLOWUP_RE,
    _GUEST_SELF_FOCUS_RE, _infer_memory_meta_from_row, _is_recent_guest_experience_row,
    _json_clone, _MECHANISM_PHASES, _memory_payload_meta, _memory_query_analysis,
    _memory_record_meta, _MEMORY_ROLE_MAP, _MIGRATION_DOMAIN_PATHS, _NATIVE_MECHANISM_PATHS,
    _normalize_policy_profile, _normalize_time_anchor, _overlap_ratio, _parse_bool, _path_get,
    _path_set, _PERSONALITY_RE, _query_mentions_character, _RECENT_STABLE_STATUS_RE,
    _row_primary_dt, _SCHEDULE_QUERY_RE, _SHARED_EXPERIENCE_RE, _stable_hash,
    _SUBJECTIVE_PROJECTION_RE, _SURFACE_REPLY_RECALL_RE, _time_anchor_compat,
    _tokenize_grounding_text, _verify_fact_grounding, canonicalize_developer_fact,
    embedding_row_matches_current, fact_is_dialogue_residue, infer_topic_labels_from_query,
    lexical_similarity, memory_explicit_query_has_bounded_slot,
    memory_explicit_row_is_answer_bearing, memory_query_analysis, memory_row_evidence_role,
    memory_row_has_durable_turn_signal, recall_focus_keyword_score, recall_focus_text,
    rewrite_fact_via_llm,
)
from tsuzuri.memory_core_impl.store_storage import _StorageMixin
from tsuzuri.memory_core_impl.store_items import _ItemsMixin
from tsuzuri.memory_core_impl.store_turn_writes import _TurnWritesMixin
from tsuzuri.memory_core_impl.store_retrieval import _RetrievalMixin
from tsuzuri.memory_core_impl.store_evolution import _EvolutionMixin
from tsuzuri.memory_core_impl.store_reflection import _ReflectionMixin
from tsuzuri.memory_core_impl.store_rethink import _RethinkMixin
from tsuzuri.memory_core_impl.store_lifecycle import _LifecycleMixin
from tsuzuri.memory_core_impl.store import (
    _APPOINTMENT_PATTERN, _CONFIRM_PATTERN, _MEMORY_CORE_LOCK, _parse_appointment_date,
    _WEEKDAY_MAP, extract_appointment, get_due_appointments, get_memory_core,
    get_someday_appointments, mark_appointment_reminded, MemoryCoreStore, save_appointment,
)

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable
from tsuzuri.reliable_io import interprocess_file_lock
from tsuzuri.sqlite_reliability import connect_sqlite, is_sqlite_busy_error
from tsuzuri.secrets_loader import env_bool as _bool_env
from tsuzuri.time_source import local_now, now_world, utc_now, utc_now_naive
import re as _fv_re
import re as _appt_re
from datetime import datetime as _appt_dt, timedelta as _appt_td

from tsuzuri.module_facade import reexport_optional as _reexport_optional

# Names bound only on some paths (loop variables, one-sided branches).
_reexport_optional(globals(), 'tsuzuri.memory_core_impl', 'recall_scoring', ('_jieba_mod',))

from tsuzuri.module_facade import install_facade as _install_facade

_install_facade(
    __name__,
    package='tsuzuri.memory_core_impl',
    submodules=('config', 'memory_rows', 'features', 'explicit_recall', 'recall_scoring', 'meta_policy', 'store_storage', 'store_items', 'store_turn_writes', 'store_retrieval', 'store_evolution', 'store_reflection', 'store_rethink', 'store_lifecycle', 'store'),
    forwarded={
        '_CO_STRENGTH_LOADED': 'recall_scoring',
        '_MEMORY_CORE_SINGLETON': 'store'
    },
)
