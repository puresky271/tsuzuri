"""Store mixin: memory read planning, retrieval and context packs.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import os
import re
import sqlite3
import time
import uuid
from typing import Callable
from tsuzuri.memory_core_impl.config import (
    _current_embedding_spec, _emb_cos, _emb_decode, _emb_encode, _emb_encode_both,
    _emb_encode_sparse, _emb_is_m3_mode, _emb_sparse_to_text, _emb_to_text,
    _embedding_callable, _embedding_row_matches_current, _safe_float, _safe_int, _utc_now_iso,
    mem_actr_recency_enabled, mem_lateral_inhibition_enabled, mem_llm_selector_enabled,
    mem_motivation_boost_enabled, mem_spreading_activation_enabled,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _bm25_score_pair, _collective_row_third_person_self_referent, _lexical_similarity,
    _memory_explicit_archive_matches_query, _memory_query_hits_removed_recall_tombstone,
    _memory_row_age_hours, _memory_row_blocked_by_meta_topic,
    _memory_row_is_auto_fact_observation, _memory_row_is_explicit_recall_archive,
    _memory_row_is_surfaceable_working, _memory_row_is_working,
    _memory_surface_text_is_generic, _memory_working_text_is_ephemeral_noise,
    _PATH4_CHITCHAT_FACT_RE, _recall_focus_text, _rrf_fuse, _text_mentions_system_origin,
    use_rrf_fusion,
)
from tsuzuri.memory_core_impl.features import (
    _candidate_is_memory_probe_text, _developer_display_name, _extract_memory_features,
    _infer_topic_labels_from_query, _recall_focus_keyword_score,
)
from tsuzuri.memory_core_impl.explicit_recall import (
    _apply_memory_correction_dominance, _dedupe_memory_rows_by_item_id,
    _direct_recall_match_score, _EXPLICIT_DATE_ANSWER_RE, _EXPLICIT_DATE_QUERY_RE,
    _explicit_recall_answer_budget, _memory_explicit_query_has_bounded_slot,
    _memory_explicit_row_is_answer_bearing, _memory_row_can_surface_nonexplicit,
    _memory_row_canonical_key, _memory_row_evidence_role,
    _memory_should_run_expensive_reranker, _profile_promoted_probe_user_answer,
    _select_explicit_surface_rows,
)
from tsuzuri.memory_core_impl.recall_scoring import (
    _actr_recency_score, _apply_affect_memory_boost, _apply_lateral_inhibition,
    _apply_spreading_activation, _clamp, _compute_motivation_boost, _expand_query_semantically,
    _extract_payload_emotional_weight, _json_dumps, _json_loads, _memory_recency_timestamp,
    _payload_marks_non_retrievable_present_state, _recency_score, _rule_expand_query,
    _semantic_query_expand_enabled, _source_prior, _split_query_semantics,
)
from tsuzuri.memory_core_impl.meta_policy import (
    _derive_grounding_signals, _DEVELOPER_FOCUS_RE, _EXPLICIT_RECALL_RE,
    _infer_memory_meta_from_row, _is_recent_guest_experience_row, _memory_query_analysis,
    _normalize_policy_profile, _overlap_ratio, _parse_bool, _query_mentions_character,
    _SURFACE_REPLY_RECALL_RE, _time_anchor_compat,
)
from tsuzuri.persona_registry import collective_scope, resolve_agent


def _fts5_quoted_query(text: str) -> str:
    """Turn free text into an FTS5 MATCH expression that cannot be a syntax error.

    原文直接进 MATCH 会被引号/+/-/: 当成查询语法报错；按空白切开逐段加引号，
    保留裸查询的隐式 AND 语义。
    """
    return " ".join('"' + part.replace('"', '""') + '"' for part in str(text or "").split())


class _RetrievalMixin:
    """Memory read planning, retrieval and context packs."""


    def plan_memory_read(
        self,
        character: str,
        query_text: str,
        *,
        query_bundle: dict | None = None,
        max_items: int = 6,
        llm_selector: Callable[[str, list[dict]], dict[str, float]] | None = None,
        world_signals: dict | None = None,
        risk_bundle: dict | None = None,
    ) -> dict:
        rb = risk_bundle if isinstance(risk_bundle, dict) else {}
        read_scope = str(rb.get("memory_read_scope", "normal") or "normal").strip().lower() or "normal"
        analysis = _memory_query_analysis(query_text, query_bundle if isinstance(query_bundle, dict) else {}, read_scope)
        retrieval_query_bundle = dict(query_bundle) if isinstance(query_bundle, dict) else {}
        retrieval_query_bundle["_memory_read_scope"] = read_scope
        retrieval_query_bundle["_guest_developer_focus"] = bool(
            read_scope == "guest" and analysis.get("developer_focus", False)
        )
        ret = self.retrieve_memory_context(
            character,
            query_text,
            query_bundle=retrieval_query_bundle,
            max_items=max_items,
            llm_selector=llm_selector,
            touch_selected=False,
            log_retrieval=True,
        )
        base_rows = [
            self._decorate_memory_row(x)
            for x in (ret.get("rows", []) if isinstance(ret.get("rows", []), list) else [])
            if isinstance(x, dict) and str(x.get("item_type", "") or "") != "core_summary"
        ]
        collective_rows = [self._decorate_memory_row(x) for x in (ret.get("collective_rows", []) if isinstance(ret.get("collective_rows", []), list) else []) if isinstance(x, dict)]
        guest_safe_profile_view = self._build_guest_safe_profile_view(character, query_text, query_bundle=query_bundle, max_items=max_items)

        filtered_rows: list[dict] = []
        blocked_rows: list[dict] = []
        recent_guest_experience_blocked = False
        for row in base_rows + collective_rows:
            r = dict(row)
            memory_type = str(r.get("memory_type", "") or "")
            owner_scope = str(r.get("owner_scope", "") or "")
            evidence_tier = str(r.get("evidence_tier", "") or "").strip().lower()
            source_kind = str(r.get("source_kind", "") or "").strip().lower()
            text = str(r.get("text", "") or "")
            if analysis.get("meta_instruction_topic", False) and _memory_row_blocked_by_meta_topic(r):
                r["blocked_reason"] = "meta_instruction_topic_guard"
                blocked_rows.append(r)
                continue
            # 2026-05-11 加：collective 层 3rd-person 自指 filter
            # 防止历史 collective 记录里 "我向某角色坦白" 这种私聊误升记录
            # 在 角色B chat 召回时被 surface（视角错位、canon 灾难）
            if _collective_row_third_person_self_referent(r, character):
                r["blocked_reason"] = "collective_third_person_self_reference"
                blocked_rows.append(r)
                continue
            if read_scope == "pure":
                if memory_type != "working_turn":
                    r["blocked_reason"] = "pure_scope"
                    blocked_rows.append(r)
                    continue
            elif read_scope == "guest":
                if analysis.get("developer_focus", False):
                    if analysis.get("guest_developer_experience_query", False):
                        allow_guest_character_memory = bool(
                            (
                                owner_scope == f"character:{resolve_agent(character)}"
                                and memory_type == "episodic_event"
                                and source_kind == "episodic_event"
                            )
                            or (
                                owner_scope == "collective"
                                and memory_type == "collective_event"
                                and source_kind == "collective_event"
                            )
                        )
                        if allow_guest_character_memory and _is_recent_guest_experience_row(r):
                            allow_guest_character_memory = False
                            recent_guest_experience_blocked = True
                            r["blocked_reason"] = "guest_recent_experience_blocked"
                            blocked_rows.append(r)
                            continue
                    else:
                        allow_guest_character_memory = bool(
                            memory_type == "shared_fact"
                            and evidence_tier in ("high", "medium")
                            and _DEVELOPER_FOCUS_RE.search(text)
                        )
                else:
                    allow_guest_character_memory = bool(
                        owner_scope == f"character:{resolve_agent(character)}"
                        and (not analysis.get("guest_self_fact_query", False))
                        and (not analysis.get("current_status_query", False))
                        and (
                            memory_type == "working_turn"
                            or (
                                memory_type == "episodic_event"
                                and source_kind == "episodic_event"
                            )
                            or (
                                memory_type == "character_projection"
                                and source_kind == "character_projection"
                                and (not analysis.get("fact_like_query", False))
                            )
                        )
                    )
                if not allow_guest_character_memory:
                    r["blocked_reason"] = "guest_scope_restricted"
                    blocked_rows.append(r)
                    continue
            if analysis.get("fact_like_query", False) and memory_type == "character_projection":
                r["blocked_reason"] = "projection_not_for_fact_query"
                blocked_rows.append(r)
                continue
            if str(r.get("source_kind", "") or "") in ("prompt_runtime", "town_projection"):
                r["blocked_reason"] = "runtime_inferred"
                blocked_rows.append(r)
                continue
            filtered_rows.append(r)

        if read_scope == "guest" and bool(guest_safe_profile_view.get("detected", False)):
            for row in guest_safe_profile_view.get("rows", []) if isinstance(guest_safe_profile_view.get("rows", []), list) else []:
                if isinstance(row, dict):
                    filtered_rows.append(self._decorate_memory_row(row))

        filtered_rows = _dedupe_memory_rows_by_item_id(filtered_rows)
        # Cluster narratives have a dedicated workspace surface. Keep them out
        # of the generic memory surface so they cannot compete with ordinary
        # facts during provider selection; cluster_memory.py owns their
        # deterministic trigger and CLUSTER_MEMORY_FRAME rendering.
        filtered_rows = [
            row for row in filtered_rows
            if str(row.get("item_type", "") or "").strip().lower()
            not in {"event_cluster", "topic_cluster"}
        ]
        filtered_rows.sort(key=lambda x: float(x.get("score", x.get("collective_score", 0.0)) or 0.0), reverse=True)
        top = filtered_rows[0] if filtered_rows else {}
        top_score = float(top.get("score", top.get("collective_score", 0.0)) or 0.0) if isinstance(top, dict) else 0.0
        top_tier = str(top.get("evidence_tier", "") or "") if isinstance(top, dict) else ""
        surfaceable_rows = [
            r for r in filtered_rows
            if not _memory_surface_text_is_generic(str(r.get("text", "") or ""))
            and (not _memory_row_is_working(r) or _memory_row_is_surfaceable_working(r))
        ]
        selected_surface_rows: list[dict] = []
        surface_mode = "silent_grounding"
        trigger_basis = "none"
        suppression_reason = ""
        style_hint = "先接住眼前这句，不要像翻档案。"

        if analysis.get("meta_instruction_topic", False):
            surface_mode = "abstain" if analysis.get("explicit_recall_intent", False) else "silent_grounding"
            trigger_basis = "none"
            selected_surface_rows = []
            suppression_reason = "meta_instruction_topic_guard"
            style_hint = "对方在说系统指令、提示词或模糊话题时，不要从长期记忆里牵出系统来历、账号事故或无关旧事。"
        elif analysis.get("system_origin_query", False):
            origin_rows = [
                r for r in filtered_rows
                if (
                    float(r.get("score", r.get("collective_score", 0.0)) or 0.0) >= 0.30
                    and not _memory_surface_text_is_generic(str(r.get("text", "") or ""))
                    and _text_mentions_system_origin(str(r.get("text", "") or ""))
                )
            ]
            if origin_rows:
                surface_mode = "explicit_recall"
                trigger_basis = "system_origin_query"
                selected_surface_rows = _select_explicit_surface_rows(origin_rows, limit=2, query_text=query_text)
                style_hint = "对方明确追问系统来历时，只依据选中的事实回答；不要补写未记录的开发过程或动机。"
            else:
                surface_mode = "abstain"
                trigger_basis = "none"
                suppression_reason = "no_system_origin_evidence"
                style_hint = "没有足够的系统来历事实时直接说不确定，不要从其他系统话题推断。"
        elif analysis.get("guest_developer_experience_query", False):
            experience_rows = [
                r for r in filtered_rows
                if (
                    str(r.get("memory_type", "") or "") in ("episodic_event", "collective_event")
                    and str(r.get("source_kind", "") or "") in ("episodic_event", "collective_event")
                )
            ]
            if experience_rows:
                surface_mode = "explicit_recall"
                trigger_basis = "user_explicit_recall"
                selected_surface_rows = experience_rows[:1]
                style_hint = "访客明确追问你和开发者之间的经历时，只提一条较早、真实、把握高的旧经历；不要掺今天/最近的碎片，也不要讲成日志流水账。"
            else:
                surface_mode = "abstain"
                suppression_reason = ("guest_shared_experience_recent_only" if recent_guest_experience_blocked else "guest_shared_experience_requires_specific_memory")
                style_hint = (
                    "访客追问你和开发者之间发生过什么具体事时，如果你能想到的只有今天/最近的碎片，就不要往这轮里带；没有更稳的旧记忆线索就直接说想不起特别具体的。"
                    if recent_guest_experience_blocked
                    else "访客追问你和开发者之间发生过什么具体事时，直接收口到想不起特别具体的；不要现编场景，也不要把仓里的旧条目包装成能讲给访客听的共同经历。"
                )
                trigger_basis = "none"
        elif not filtered_rows:
            if analysis.get("guest_self_fact_query", False):
                surface_mode = "abstain"
                suppression_reason = "guest_self_fact_requires_turn_evidence"
                style_hint = "访客问到他自己的作息、习惯或近况时，没有这段对话里的明确信息就直接说不清楚，不要靠熟悉感去猜。"
            else:
                surface_mode = "abstain" if (analysis.get("developer_focus", False) or analysis.get("explicit_recall_intent", False)) else "silent_grounding"
                if analysis.get("explicit_recall_intent", False):
                    suppression_reason = "no_consolidated_memory_rows"
                else:
                    suppression_reason = ("no_safe_memory_rows" if surface_mode == "abstain" else "no_relevant_rows")
                style_hint = "没有合适线索时就不要硬提过去的事。"
            trigger_basis = "none"
        elif analysis.get("guest_self_fact_query", False):
            direct_rows = [
                r for r in filtered_rows
                if str(r.get("memory_type", "") or "") == "working_turn"
                and float(r.get("score", r.get("collective_score", 0.0)) or 0.0) >= 0.72
            ]
            if direct_rows:
                surface_mode = "light_callback"
                trigger_basis = "continuity_followup"
                selected_surface_rows = direct_rows[:1]
                style_hint = "只有当对方刚在这段对话里亲口提过，才能保守复述；别替对方下定论，也别补具体时刻。"
            else:
                surface_mode = "abstain"
                trigger_basis = "none"
                suppression_reason = "guest_self_fact_requires_turn_evidence"
                style_hint = "访客问到他自己的作息、习惯或近况时，没有这段对话里的明确信息就直接说不清楚，不要靠熟悉感去猜。"
        else:
            # ── 分数驱动的记忆注入（不再依赖正则判断用户意图）──
            # 原则：召回引擎已经通过 8 维评分选出了最相关的记忆，
            # 如果分数够高就注入，不需要用户说"还记得"之类的触发词。
            _is_explicit = bool(analysis.get("explicit_recall_intent", False))
            _explicit_focus = _recall_focus_text(query_text) if _is_explicit else ""
            _explicit_direct_rows = [
                r for r in filtered_rows
                if (
                    (not _memory_row_is_working(r) or bool(r.get("working_direct_recall_lane", False)))
                    and not _memory_surface_text_is_generic(str(r.get("text", "") or ""))
                    and (
                        bool(r.get("working_direct_recall_lane", False))
                        or bool(r.get("_hybrid_structured_hit", False))
                        or float(r.get("score", r.get("collective_score", 0.0)) or 0.0) >= 0.30
                        or _recall_focus_keyword_score(_explicit_focus, str(r.get("text", "") or "")) >= 0.45
                        or _lexical_similarity(_explicit_focus, str(r.get("text", "") or "")) >= 0.18
                    )
                )
            ]
            if _is_explicit and (top_score >= 0.30 or _explicit_direct_rows):
                # 用户明确在聊过去的事 → 放宽阈值，积极注入
                surface_mode = "explicit_recall"
                trigger_basis = "user_explicit_recall"
                # 2026-05-09 P5: limit 与 mention_budget (=2) 对齐、不再选 3 条让第 3 条半泄漏
                selected_surface_rows = _select_explicit_surface_rows(
                    _explicit_direct_rows or filtered_rows,
                    limit=_explicit_recall_answer_budget(query_text),
                    query_text=query_text,
                    require_answer_bearing=True,
                )
                if selected_surface_rows:
                    if any(_memory_row_is_surfaceable_working(r) for r in selected_surface_rows):
                        style_hint = "对方在聊刚才或近期说过的事，可以自然接上短期连续性线索；不要讲成长期记忆或沉淀档案。"
                    else:
                        style_hint = "对方在聊过去的事，自然地引用你记得的细节，像老朋友回忆共同经历一样。"
                else:
                    surface_mode = "abstain"
                    trigger_basis = "none"
                    suppression_reason = "no_consolidated_memory_rows"
                    style_hint = "显式回忆时没有沉淀过的旧记忆线索，就不要拿刚刚的 working 对话或泛化话题硬答。"
            else:
                # Evidence-tier-aware thresholds: high confidence memories surface more easily
                _nonexplicit_surface_rows = [
                    r for r in surfaceable_rows
                    if _memory_row_can_surface_nonexplicit(r, analysis)
                ]
                _surface_top = _nonexplicit_surface_rows[0] if _nonexplicit_surface_rows else {}
                _surface_top_score = float(_surface_top.get("score", _surface_top.get("collective_score", 0.0)) or 0.0) if isinstance(_surface_top, dict) else 0.0
                _surface_top_tier = str(_surface_top.get("evidence_tier", "") or top_tier)
                # 2026-05-09 P2 fix: high evidence 阈值 0.30 → 0.40（弱关联不应自动 callback）
                _tier_threshold = {"high": 0.40, "medium": 0.45, "low": 0.55}.get(_surface_top_tier, 0.45)

                # 2026-05-09 P4: 过滤掉 chitchat / ephemeral 的 working rows
                # 防止 "晚安"/"嗯"/"下次再聊" 类闲聊 turn 撞词面重合 0.30 → 成 visible_cue
                def _row_is_chitchat(_r):
                    _txt = str(_r.get("text", "") or "")
                    if not _txt:
                        return False
                    # 提取 user 部分（"user:X | assistant:Y" 格式）
                    _ut = _txt
                    if "user:" in _txt:
                        _parts = _txt.split(" | assistant:", 1)
                        _ut = _parts[0].replace("user:", "", 1).strip()
                    if _PATH4_CHITCHAT_FACT_RE.search(_ut):
                        return True
                    _feats = _extract_memory_features(_ut, "")
                    return float(_feats.get("ephemeral_penalty", 0.0)) >= 1.0

                _nonexplicit_surface_rows = [
                    r for r in _nonexplicit_surface_rows
                    if not (_memory_row_is_working(r) and _row_is_chitchat(r))
                ]
                # 重算 top（filter 后可能为空）
                _surface_top = _nonexplicit_surface_rows[0] if _nonexplicit_surface_rows else {}
                _surface_top_score = float(_surface_top.get("score", _surface_top.get("collective_score", 0.0)) or 0.0) if isinstance(_surface_top, dict) else 0.0

                _continuity_rows = [
                    r for r in _nonexplicit_surface_rows
                    if bool(r.get("working_continuity_lane", False))
                    and float(r.get("score", r.get("collective_score", 0.0)) or 0.0) >= 0.30
                ]
                _direct_fact_rows = [
                    r for r in _nonexplicit_surface_rows
                    if analysis.get("fact_like_query", False)
                    and _memory_explicit_query_has_bounded_slot(query_text)
                    and _memory_explicit_row_is_answer_bearing(r, query_text)
                ]
                if _direct_fact_rows:
                    surface_mode = "light_callback"
                    trigger_basis = "direct_topic_match"
                    _fact_limit = _explicit_recall_answer_budget(query_text)
                    selected_surface_rows = _select_explicit_surface_rows(
                        _direct_fact_rows,
                        limit=_fact_limit,
                        query_text=query_text,
                        require_answer_bearing=True,
                    )
                    style_hint = (
                        "直接使用最贴题、能回答当前问题的事实；复合问题可组合互补细节，不要顺带扩展到其他旧话题。"
                        if _fact_limit > 1
                        else "直接使用最贴题、能回答当前问题的事实；不要顺带扩展到其他旧话题。"
                    )
                elif _continuity_rows:
                    surface_mode = "light_callback"
                    trigger_basis = "continuity_followup"
                    selected_surface_rows = _continuity_rows[:1]
                    style_hint = "自然接上对方刚才或这几天提过的事；这是短期连续性，不要说成长期记忆。"
                elif _surface_top_score >= _tier_threshold:
                    # 高相关度记忆 → 注入
                    surface_mode = "light_callback"
                    trigger_basis = (
                        "continuity_followup"
                        if any(bool(r.get("working_continuity_lane", False)) for r in _nonexplicit_surface_rows[:3])
                        else ("emotional_resonance" if analysis.get("emotional_query", False) else "direct_topic_match")
                    )
                    selected_surface_rows = _nonexplicit_surface_rows[:3]
                    if any(_memory_row_is_surfaceable_working(r) for r in selected_surface_rows):
                        style_hint = "自然接上对方刚才或这几天提过的事；这是短期连续性，不要说成长期记忆。"
                    else:
                        style_hint = "你记得这件事。像朋友一样自然地提起——不要说'我记得你说过'这种格式化措辞，而是直接用你知道的信息回应，就像你本来就知道这件事一样。"
                elif _surface_top_score >= 0.25:
                    # 中等相关度 → 背景认知（仅设 mode、不填 rows、防 P1 leak 进 auto_greet / runtime cache）
                    # 2026-05-09 P1 fix: silent_grounding 不再填 selected_surface_rows
                    # 历史 bug：rows 通过 pack.memory_rows / runtime cache 漏给 QQ auto_greet
                    # 等 downstream 路径、即使 should_surface=False 也会被随机当 callback hint
                    surface_mode = "silent_grounding"
                    trigger_basis = "background_relevance"
                    selected_surface_rows = []
                    suppression_reason = "background_only"
                    style_hint = "没有特别合适的旧线索；不要主动把过去的事搬出来。"
                else:
                    # 低相关度 → 不注入
                    surface_mode = "silent_grounding"
                    trigger_basis = "none"
                    selected_surface_rows = []
                    suppression_reason = "low_relevance"
                    style_hint = "没有相关记忆，不要编造。"

        mention_budget = {
            "explicit_recall": _explicit_recall_answer_budget(query_text),
            "light_callback": 1,     # 高相关 → 自然提 1 条
            "silent_grounding": 0,   # 背景认知 → 不直接提
            "abstain": 0,
        }.get(surface_mode, 0) if selected_surface_rows else 0
        # 2026-09-12: 疑问句查询 + 多条互补线索时、light_callback 允许自然带 2 条——
        # 复合问题（"她喜欢什么"= 类别 A + 类别 B）单条线索答不全；仍硬封顶 2、
        # 避免闲聊变翻档案。explicit_recall 路径由 answer budget 自适应（1-4）。
        if (
            surface_mode == "light_callback"
            and mention_budget == 1
            and len(selected_surface_rows) >= 2
            and re.search(r"(什么|哪|怎么|为什么|为何|多少|呢|吗|？|\?)", str(query_text or ""))
        ):
            mention_budget = 2
        if (
            read_scope == "guest"
            and analysis.get("guest_developer_experience_query", False)
            and surface_mode == "explicit_recall"
        ):
            mention_budget = min(int(mention_budget or 0), 1)
        memory_surface_plan = {
            "should_surface": bool(mention_budget > 0),
            "surface_mode": surface_mode,
            "trigger_basis": trigger_basis,
            "timing_fit": round(top_score, 3),
            "mention_budget": int(mention_budget),
            "selected_memory_ids": [str(r.get("item_id", "") or "") for r in selected_surface_rows if isinstance(r, dict)],
            "selected_surface_rows": [dict(r) for r in selected_surface_rows if isinstance(r, dict)],
            "suppression_reason": suppression_reason,
            "style_hint": style_hint,
            "strict_focus": bool(analysis.get("strict_focus", False)),
            "read_scope": read_scope,
            "top_evidence_tier": top_tier,
        }
        self._log_policy_decision(
            decision_kind="memory_surface",
            phase="plan",
            character=character,
            run_id=str(ret.get("run_id", "") or ""),
            input_payload={
                "read_scope": read_scope,
                "query": str(query_text or "")[:220],
                "retrieved_count": len(base_rows) + len(collective_rows),
                "filtered_count": len(filtered_rows),
            },
            llm_payload={},
            fallback_reason=suppression_reason,
            final_payload={
                "surface_mode": surface_mode,
                "trigger_basis": trigger_basis,
                "selected_ids": [str(r.get("item_id", "") or "") for r in selected_surface_rows if isinstance(r, dict)],
            },
        )
        return {
            "read_scope": read_scope,
            "query_analysis": analysis,
            "retrieval_run_id": str(ret.get("run_id", "") or ""),
            "reranker_used": bool(ret.get("reranker_used", False)),
            "reranker_eligible": bool(ret.get("reranker_eligible", False)),
            "retrieved_rows": base_rows,
            "collective_rows": collective_rows,
            "filtered_rows": filtered_rows,
            "blocked_rows": blocked_rows,
            "cue_bundle": ret.get("cue_bundle", {}) if isinstance(ret.get("cue_bundle", {}), dict) else {},
            "guest_safe_profile_view": guest_safe_profile_view,
            "memory_surface_plan": memory_surface_plan,
            "surface_rows": selected_surface_rows,
            "world_signals": world_signals if isinstance(world_signals, dict) else {},
            "risk_bundle": rb,
            "guest_developer_read_policy": (
                "deep_history_only"
                if read_scope == "guest" and bool(analysis.get("guest_developer_experience_query", False))
                else ("stable_only" if read_scope == "guest" and bool(analysis.get("developer_focus", False)) else "")
            ),
        }

    def retrieve_collective_memory_context(
        self,
        character: str,
        query_text: str,
        *,
        query_bundle: dict | None = None,
        max_items: int = 4,
        explicit_recall_probe: bool = False,  # Phase G.M5 (2026-05-02): collective gate 与 episodic 对称
    ) -> list[dict]:
        _ = resolve_agent(character)
        qb = query_bundle if isinstance(query_bundle, dict) else {}
        query = str(query_text or "").strip()
        if len(query) < 2:
            return []
        # [QueryExpand 规则版切分] collective 检索同样回喂候选池
        _coll_semantics = _split_query_semantics(query, str(character or ""))
        try:
            _qvs = _expand_query_semantically(query, str(character or ""))
        except Exception:
            _qvs = [query]
        _coll_extra: list[str] = []
        if _semantic_query_expand_enabled():
            for _cs in (_qvs[1:] + _coll_semantics.get("clauses", []) + _coll_semantics.get("keywords", [])
                        + _rule_expand_query(query, str(character or ""))):
                _cs2 = str(_cs or "").strip()
                if len(_cs2) >= 2 and _cs2 != query and _cs2 not in _coll_extra:
                    _coll_extra.append(_cs2)
        rows = self._query_items(collective_scope(), limit=max(120, int(max_items) * 12), query_text=query,
                                 extra_search_texts=_coll_extra[:6])
        # A 阶段：collective retrieval 也用多变体 query
        _query_emb = None
        _query_embs: list = []
        _emb_active = False
        # Phase G.M2 (2026-05-02): 走 module-level _emb_* 单点引用
        _emb_decode_fn = _emb_decode  # local alias for downstream code
        _emb_cos_fn = _emb_cos
        if _embedding_callable():
            try:
                for _qv in _qvs:
                    _qe = self._encode_memory_query_dense(_qv)
                    if _qe is not None:
                        _query_embs.append(_qe)
                if _query_embs:
                    _query_emb = _query_embs[0]
                    _emb_active = True
            except Exception:
                pass
        cands = []
        for r in rows:
            if str(r.get("memory_role", "") or "").strip().lower() != "collective":
                continue
            tx = str(r.get("text", "") or "").strip()
            if not tx:
                continue
            cue = _json_loads(str(r.get("cue_json", "") or "{}"), {})
            relevance = _lexical_similarity(query, tx)
            # [QueryExpand 变体感知 lexical] collective 同 episodic：同义/子句变体可抬升词法 rel
            if relevance < 0.20 and _coll_extra:
                for _v in _coll_extra:
                    _vr = _lexical_similarity(_v, tx)
                    if _vr > relevance:
                        relevance = _vr
            topic_match = _overlap_ratio(
                [str(x or "").strip() for x in (qb.get("topic_tags", []) if isinstance(qb.get("topic_tags", []), list) else qb.get("topics", []) if isinstance(qb.get("topics", []), list) else []) if str(x or "").strip()],
                [str(x or "").strip() for x in (cue.get("topics", []) if isinstance(cue.get("topics", []), list) else []) if str(x or "").strip()],
            )
            cue_match = _overlap_ratio(
                [str(x or "").strip() for x in (qb.get("entities", []) if isinstance(qb.get("entities", []), list) else []) if str(x or "").strip()],
                [str(x or "").strip() for x in (cue.get("entities", []) if isinstance(cue.get("entities", []), list) else []) if str(x or "").strip()],
            )
            temporal = _time_anchor_compat(str(qb.get("time_anchor", "recent") or "recent"), str(cue.get("time_anchor", "recent") or "recent"))
            phase_match = 0.0
            if str(qb.get("session_phase", "") or "").strip() and str(qb.get("session_phase", "") or "").strip() == str(cue.get("session_phase", "") or "").strip():
                phase_match = max(phase_match, 0.45)
            if str(qb.get("school_phase", "") or "").strip() and str(qb.get("school_phase", "") or "").strip() == str(cue.get("school_phase", "") or "").strip():
                phase_match = max(phase_match, 0.35)
            if str(qb.get("location_anchor", "") or "").strip() and str(qb.get("location_anchor", "") or "").strip() == str(cue.get("location_anchor", "") or "").strip():
                phase_match = max(phase_match, 0.35)
            # Embedding similarity for collective (semantic recall) - A 阶段多变体 max
            _emb_score = 0.0
            if _emb_active:
                _item_emb_text = str(r.get("embedding_text", "") or "")
                if _item_emb_text and _embedding_row_matches_current(r):
                    try:
                        _item_emb = _emb_decode_fn(_item_emb_text)
                        if _item_emb is not None:
                            _emb_score = max(
                                (max(0.0, _emb_cos_fn(_qe, _item_emb)) for _qe in _query_embs),
                                default=0.0
                            )
                    except Exception:
                        pass
            # Biomimetic gate: weak lexical AND weak semantic → 不召回
            # (same logic as episodic retrieve: 无关记忆不被无故唤起)
            # Phase G.M5 (2026-05-02): collective gate 与 episodic 对称、explicit_recall_probe
            # 时 emb 阈值放松到 0.50（之前不区分、用户主动唤起 collective 仍被 0.55 拦截）
            _coll_emb_gate = 0.50 if explicit_recall_probe else 0.55
            if relevance < 0.045 and _emb_score < _coll_emb_gate and max(topic_match, cue_match, phase_match) < 0.20:
                continue
            # Score formula: embedding takes ~0.18 weight from lexical-heavy side,
            # remaining factors re-normalized to ~0.94 total
            # Original: 0.36*rel + 0.22*cue + 0.12*topic + 0.14*temp + 0.10*phase + 0.14*sal + 0.12*imp
            # New    : 0.20*rel + 0.18*emb + 0.21*cue + 0.11*topic + 0.13*temp + 0.09*phase + 0.13*sal + 0.11*imp
            score = _clamp(
                0.20 * relevance
                + 0.18 * _emb_score
                + 0.21 * cue_match
                + 0.11 * topic_match
                + 0.13 * temporal
                + 0.09 * phase_match
                + 0.13 * _clamp(_safe_float(r.get("salience", 0.5), 0.5))
                + 0.11 * _clamp(_safe_float(r.get("importance", 0.5), 0.5))
            )
            # Phase B.1 (2026-05-01): biomimetic dampening
            # 与 episodic 主路径对称，弱 lex + 弱 emb 时降权防止 sal/imp 撑分
            # emb 阈值 0.40：BAAI bge 对任意短句仍给 0.15-0.30 余弦
            if (relevance < 0.10
                    and _emb_score < 0.40
                    and cue_match < 0.20
                    and topic_match < 0.20
                    and phase_match < 0.20):
                score *= 0.30
            if score < 0.15:  # raised from 0.10 (stricter, biomimetic: 无关记忆真的不召回)
                continue
            item = {k: r.get(k) for k in r.keys()}
            item["source"] = "memory_core:collective"
            item["collective_score"] = round(score, 3)
            item["embedding_score"] = round(_emb_score, 3)
            cands.append(item)
        cands.sort(key=lambda x: float(x.get("collective_score", 0.0) or 0.0), reverse=True)
        return cands[: max(1, int(max_items or 4))]

    def _load_policy_profile(self) -> dict:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT config_json FROM policy_profiles WHERE active=1 LIMIT 1;").fetchone()
            finally:
                conn.close()
        if row is None:
            return _normalize_policy_profile({})
        return _normalize_policy_profile(_json_loads(str(row["config_json"] or ""), {}))

    def _log_policy_decision(
        self,
        *,
        decision_kind: str,
        phase: str = "",
        character: str = "",
        run_id: str = "",
        input_payload: dict | None = None,
        llm_payload: dict | None = None,
        fallback_reason: str = "",
        final_payload: dict | None = None,
    ) -> None:
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO policy_decision_logs
                    (decision_kind, phase, character, run_id, input_json, llm_json, fallback_reason, final_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(decision_kind or "").strip()[:48],
                        str(phase or "").strip()[:48],
                        str(character or "").strip()[:24],
                        str(run_id or "").strip()[:40],
                        _json_dumps(input_payload if isinstance(input_payload, dict) else {}),
                        _json_dumps(llm_payload if isinstance(llm_payload, dict) else {}),
                        str(fallback_reason or "").strip()[:120],
                        _json_dumps(final_payload if isinstance(final_payload, dict) else {}),
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def log_memory_context_diff(
        self,
        *,
        character: str,
        retrieval_run_id: str,
        legacy_rows: list[dict] | list[str] | None,
        context_rows: list[dict] | list[str] | None,
        diff: dict | None = None,
        severity: str = "info",
    ) -> None:
        legacy = legacy_rows if isinstance(legacy_rows, list) else []
        context = context_rows if isinstance(context_rows, list) else []
        d = diff if isinstance(diff, dict) else {}
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO memory_context_diff_logs
                    (character, retrieval_run_id, legacy_json, context_json, diff_json, severity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(character or "").strip()[:24],
                        str(retrieval_run_id or "").strip()[:40],
                        _json_dumps(legacy),
                        _json_dumps(context),
                        _json_dumps(d),
                        str(severity or "info").strip()[:24],
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def _query_items(self, character: str, limit: int = 220, query_text: str = "", extra_search_texts=()) -> list[dict]:
        char = resolve_agent(character)
        lim = max(20, min(800, int(limit or 220)))
        search_texts: list[str] = []
        qt = str(query_text or "").strip()
        if qt:
            search_texts.append(qt)
            focus = _recall_focus_text(qt) if bool(_EXPLICIT_RECALL_RE.search(qt)) else ""
            if focus and focus not in search_texts:
                search_texts.append(focus)
        # [QueryExpand 规则版切分] 子句/关键词/同义变体回喂候选池——
        # 老记忆不在最近 220 条窗口内、只能靠 FTS/LIKE 拉进来，
        # 变体不进 search_texts 的话扩展永远碰不到老记忆（结构性失效）。
        for _et in (extra_search_texts or ()):
            _ets = str(_et or "").strip()
            if len(_ets) >= 2 and _ets not in search_texts:
                search_texts.append(_ets)
        with self._lock:
            conn = self._connect()
            try:
                # Try FTS5 pre-filter if query_text is provided
                fts_ids: set[str] = set()
                # FTS search for items containing query keywords
                for _search in search_texts:
                    if len(_search) < 2:
                        continue
                    _fts_query = _fts5_quoted_query(_search)
                    if not _fts_query:
                        continue
                    try:
                        fts_rows = conn.execute(
                            """SELECT f.item_id FROM memory_items_fts AS f
                               JOIN memory_items AS m ON m.item_id=f.item_id
                               WHERE f.text MATCH ?
                                 AND COALESCE(m.item_type, '') NOT IN ('event_cluster', 'topic_cluster')
                               LIMIT ?""",
                            (_fts_query, lim * 2),
                        ).fetchall()
                        fts_ids.update(str(r[0]) for r in fts_rows)
                    except Exception:
                        continue  # FTS table may not exist (SQLite without FTS5)

                rows = conn.execute(
                    """
                    SELECT item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json,
                           salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, embedding_text, embedding_sparse_text, embedding_model_id, embedding_dim, topic_labels_json
                    FROM memory_items
                    WHERE archived=0 AND status='active' AND importance > 0.15
                          AND COALESCE(item_type, '') NOT IN ('core_summary', 'event_cluster', 'topic_cluster')
                          AND (character=? OR character='' OR character='*')
                    ORDER BY updated_at DESC
                    LIMIT ?;
                    """,
                    (char, lim),
                ).fetchall()

                # FTS can find old explicit-recall targets that are outside the
                # recency window. Pull those rows into the candidate pool too.
                if fts_ids:
                    try:
                        _existing_ids = {str(r["item_id"]) for r in rows}
                        missing_ids = [iid for iid in sorted(fts_ids) if iid and iid not in _existing_ids]
                        if missing_ids:
                            placeholders = ",".join(["?"] * len(missing_ids))
                            fts_rows = conn.execute(
                                f"""
                                SELECT item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json,
                                       salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, embedding_text, embedding_sparse_text, embedding_model_id, embedding_dim, topic_labels_json
                                FROM memory_items
                                WHERE archived=0 AND status='active'
                                      AND COALESCE(item_type, '') NOT IN ('core_summary', 'event_cluster', 'topic_cluster')
                                      AND (character=? OR character='' OR character='*' OR character=?)
                                      AND item_id IN ({placeholders})
                                ORDER BY updated_at DESC
                                LIMIT ?;
                                """,
                                (char, collective_scope(), *missing_ids, max(20, lim // 2)),
                            ).fetchall()
                            rows.extend(fts_rows)
                    except Exception:
                        pass

                # Keyword fallback: LIKE search for query keywords not covered by recency window
                _kw_ids: set[str] = set()
                if search_texts:
                    try:
                        import jieba as _jieba_kw
                        _kw_tokens: list[str] = []
                        for _search in search_texts:
                            if len(_search) >= 2:
                                _kw_tokens.append(_search)
                            _kw_tokens.extend([w for w in _jieba_kw.cut(_search) if len(w) >= 2])
                        _seen_kw: set[str] = set()
                        _kw_tokens = [
                            w for w in _kw_tokens
                            if not (w in _seen_kw or _seen_kw.add(w))
                        ]
                        _existing_ids = {str(r["item_id"]) for r in rows}
                        for _kw in _kw_tokens[:12]:
                            _kw_rows = conn.execute(
                                """SELECT item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json,
                                          salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, embedding_text, embedding_sparse_text, embedding_model_id, embedding_dim, topic_labels_json
                                   FROM memory_items
                                   WHERE archived=0 AND status='active'
                                         AND COALESCE(item_type, '') NOT IN ('core_summary', 'event_cluster', 'topic_cluster')
                                         AND (character=? OR character='' OR character='*')
                                         AND text LIKE ?
                                   ORDER BY updated_at DESC LIMIT 10""",
                                (char, f"%{_kw}%"),
                            ).fetchall()
                            for _kr in _kw_rows:
                                _kid = str(_kr["item_id"])
                                if _kid not in _existing_ids:
                                    rows.append(_kr)
                                    _existing_ids.add(_kid)
                                    _kw_ids.add(_kid)
                    except Exception:
                        pass
            finally:
                conn.close()
        out = []
        for r in rows:
            item = {k: r[k] for k in r.keys()}
            if fts_ids and item.get("item_id") in fts_ids:
                item["_fts_hit"] = True
            if _kw_ids and item.get("item_id") in _kw_ids:
                item["_fts_hit"] = True  # treat keyword hits same as FTS hits
            out.append(item)
        return out

    def _encode_memory_query_dense(self, text: str):
        return _emb_encode(text) if callable(_emb_encode) else None

    def _encode_memory_query_sparse(self, text: str):
        return _emb_encode_sparse(text) if callable(_emb_encode_sparse) else None

    def _mmr_select(self, rows: list[dict], k: int, lambda_val: float = 0.76) -> list[dict]:
        pool = [dict(x) for x in rows if isinstance(x, dict)]
        if not pool:
            return []
        target = max(1, int(k or 1))
        lv = _clamp(lambda_val, 0.30, 0.95)
        selected: list[dict] = []
        while pool and len(selected) < target:
            best_idx = -1
            best_score = -1e9
            for i, c in enumerate(pool):
                base = _safe_float(c.get("score", 0.0), 0.0)
                if not selected:
                    mmr = base
                else:
                    penalty = 0.0
                    c_text = str(c.get("text", "") or "")
                    for s in selected:
                        penalty = max(penalty, _lexical_similarity(c_text, str(s.get("text", "") or "")))
                    mmr = lv * base - (1.0 - lv) * penalty
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(pool.pop(best_idx))
        return selected

    def retrieve_memory_context(
        self,
        character: str,
        query_text: str,
        *,
        query_bundle: dict | None = None,
        max_items: int = 6,
        llm_selector: Callable[[str, list[dict]], dict[str, float]] | None = None,
        touch_selected: bool = True,
        log_retrieval: bool = True,
    ) -> dict:
        # 2026-05-16 性能定位：MemRecall 总耗时
        _t_recall = time.perf_counter()
        char = resolve_agent(character)
        query = str(query_text or "").strip()
        if len(query) < 2:
            return {"run_id": "", "rows": [], "collective_rows": [], "cue_bundle": {}}
        if _memory_query_hits_removed_recall_tombstone(query):
            return {"run_id": "", "rows": [], "collective_rows": [], "cue_bundle": {}}
        # 2026-05-23 cache miss 调试：把 caller 文件 + 函数名 + 行号打出来、
        # 定位 web turn 内 2 次 MemRecall 的第二个神秘 caller（max_items/qb 不同导致 cache key miss）。
        # toggle: env MEM_RECALL_TRACE=1 开启（默认关、避免污染正常日志）。
        try:
            import os as _os_trace
            if _os_trace.environ.get("MEM_RECALL_TRACE", "0").strip().lower() in ("1", "true", "on", "yes"):
                import traceback as _tb
                _stack = _tb.extract_stack(limit=8)[:-1]  # 去掉本帧
                _caller_chain = " ← ".join(f"{_f.name}:{_f.lineno}" for _f in _stack[-4:])
                _qb_keys = sorted(list((query_bundle or {}).keys()))[:6] if isinstance(query_bundle, dict) else []
                _tsz_emit(
                    f"[MemRecall·trace] {char} Q='{query[:40]}' max_items={max_items} "
                    f"qb_keys={_qb_keys} caller={_caller_chain}",
                    flush=True,
                )
        except Exception:
            pass

        # 2026-09-12: env MEM_RECALL_DEBUG=<正则> 时、对命中的候选行打印
        # 门禁失败原因 / 终分构成（漏斗诊断用，默认关）。
        _mem_debug_pat = os.environ.get("MEM_RECALL_DEBUG", "").strip()
        try:
            _mem_debug_re = re.compile(_mem_debug_pat) if _mem_debug_pat else None
        except re.error:
            _mem_debug_re = None

        # 2026-05-17 turn-scoped TTL dedup
        # 同 (char, query) 在 60s 内重复召回 → 复用结果（caller 按自己 max_items slice）。
        #
        # 2026-05-23 cache key 简化（实测 trace 数据驱动）：
        #   trace log 抓到 web reply 路径两个真实 caller：
        #     ① `_build_dev_memory_prompt:21664` → max_items=10、qb 6 keys
        #     ② `kairos_brief.build_kairos_brief:384` → max_items=4、qb 只 2 keys
        #   原 cache key = (char, query, max_items, qb_fingerprint) → qb 不同永远 miss、
        #   两次 retrieve + reranker 全跑（warm 1-3s × 2、cold ~ 32s + ~ 14s）。
        #
        #   简化为 (char, query)：两个 caller 都拿 "top-N most relevant memories" 这一固定语义、
        #   即使 qb 不同也只影响 grounding/cue secondary、core rows 集合高度重叠。
        #   命中条件加 `len(cached_rows) >= max_items` 防 caller A 先 cache 少量、
        #   caller B 后命中只拿到不够数量的回退 case。
        #
        #   实际生产顺序：reply 主路径 (max_items=10) 总比 kairos_brief (max_items=4) 先跑、
        #   所以 kairos_brief 命中条件天然满足、第二次 retrieve 直接 CACHED 返回。
        # Phase C (2026-07-28): affect_state 不同会影响 scoring，纳入 cache key 指纹
        _affect_state_for_cache = (
            query_bundle.get("affect_state") if isinstance(query_bundle, dict) else None
        )
        _affect_fingerprint: tuple = ()
        if isinstance(_affect_state_for_cache, dict):
            _affect_fingerprint = (
                round(float(_affect_state_for_cache.get("valence", 0.0) or 0.0), 1),
                round(float(_affect_state_for_cache.get("arousal", 0.0) or 0.0), 1),
            )
        _read_scope_for_cache = str(
            query_bundle.get("_memory_read_scope", "normal")
            if isinstance(query_bundle, dict)
            else "normal"
        ).strip().lower() or "normal"
        _cache_key = (char, query, _affect_fingerprint, _read_scope_for_cache)
        _cache_now = time.time()
        _slice_n = max(1, int(max_items or 0))
        with self._recall_inflight_lock:
            _cached_entry = self._recall_inflight_cache.get(_cache_key)
            if _cached_entry and (_cache_now - _cached_entry[0]) < self._recall_inflight_ttl_sec:
                _cached_result = _cached_entry[1]
                _cached_rows_full = _cached_result.get("rows", []) if isinstance(_cached_result, dict) else []
                _cached_coll_full = _cached_result.get("collective_rows", []) if isinstance(_cached_result, dict) else []
                _cached_max_items = int(_cached_result.get("cached_max_items", 0) or 0) if isinstance(_cached_result, dict) else 0
                # 2026-05-23 命中条件双判：
                #   (1) cached rows 数 >= caller 期望  → 数据本身就够、命中
                #   (2) cached_max_items >= slice_n   → cached 时 retrieve cap 就比 caller 大、
                #                                       即使 rows 数较少（包括 0 rows）也命中、
                #                                       因为再 retrieve 一次结果完全相同（同 query 同/更小 cap）
                # 修复前 bug case: 第一次 caller max_items=10 retrieve 返 0 rows、第二次
                # caller max_items=4 永远不命中（0 < 4）、重跑也是 0 rows、白烧 1-2s。
                _cache_hit = (len(_cached_rows_full) >= _slice_n) or (_cached_max_items >= _slice_n)
                if _cache_hit:
                    if log_retrieval:
                        try:
                            import os as _os_log_c
                            _silent_c = _os_log_c.environ.get("MEM_RECALL_SILENT", "").strip().lower() in ("1", "true", "on", "yes")
                            _log_on_c = _os_log_c.environ.get("MEM_RECALL_LOG", "1").strip().lower() not in ("0", "false", "off", "no")
                            if _log_on_c and not _silent_c:
                                _qd_c = (query or "")[:60].replace("\n", " ")
                                _age_ms = (_cache_now - _cached_entry[0]) * 1000
                                _tsz_emit(f"[MemRecall] {char} Q='{_qd_c}' -> CACHED {len(_cached_rows_full)} rows (caller max_items={max_items}, cached_cap={_cached_max_items}, collective={len(_cached_coll_full)}) age={_age_ms:.0f}ms", flush=True)
                        except Exception:
                            pass
                    # 浅 copy 防 caller 改 rows 列表污染 cache + slice 到 caller 期待数量
                    return {
                        "run_id": str(_cached_result.get("run_id", "") or ""),
                        "rows": list(_cached_rows_full)[:_slice_n],
                        "collective_rows": list(_cached_coll_full)[:_slice_n],
                        "cue_bundle": dict(_cached_result.get("cue_bundle", {}) or {}),
                        "reranker_used": bool(_cached_result.get("reranker_used", False)),
                        "reranker_eligible": bool(_cached_result.get("reranker_eligible", False)),
                    }
                # cached cap 和 rows 都不够 caller 期望、不命中、走下面完整 build（之后会覆盖 cache 为更大 cap 集合）

        profile = self._load_policy_profile()
        rpol = profile.get("retrieval_policy", {}) if isinstance(profile, dict) else {}
        rw = rpol.get("weights", {}) if isinstance(rpol.get("weights", {}), dict) else {}
        w_lex = _safe_float(rw.get("lexical_weight", 0.30), 0.30)
        w_rec = _safe_float(rw.get("recency_weight", 0.18), 0.18)
        w_imp = _safe_float(rw.get("importance_weight", 0.16), 0.16)
        w_src = _safe_float(rw.get("source_weight", 0.08), 0.08)
        w_nov = _safe_float(rw.get("novelty_weight", 0.08), 0.08)
        w_tmp = _safe_float(rw.get("temporal_weight", 0.10), 0.10)
        w_spa = _safe_float(rw.get("spatial_weight", 0.08), 0.08)
        w_cue = _safe_float(rw.get("cue_match_weight", 0.12), 0.12)
        w_emb = _safe_float(rw.get("embedding_weight", 0.18), 0.18)
        w_mot = _safe_float(rw.get("motivation_weight", 0.08), 0.08)
        # Phase B (2026-05-14): BM25 (sparse) 加权维度、默认权重 0.20、仅 M3 + USE_RRF_FUSION 启用
        w_bm25 = _safe_float(rw.get("bm25_weight", 0.20), 0.20)
        # 2026-05-16 修：min_relevance 0.045→0.10 + 强制 floor。
        # 之前 0.045 太低、让 rel=0 的 semantic 高 importance 行直接通过 gate、
        # 配合 RRF + reranker 顶到 score=0.99（"用户对他人学业有比较心态" case）。
        # 用 max() 强制 floor、不管 db 持久化的 policy 怎样、min 至少 0.10。
        min_relevance = max(0.10, _safe_float(rpol.get("min_relevance", 0.10), 0.10))
        mmr_lambda = _safe_float(rpol.get("mmr_lambda", 0.76), 0.76)
        llm_cfg = rpol.get("llm_selector", {}) if isinstance(rpol.get("llm_selector", {}), dict) else {}
        llm_enabled = bool(mem_llm_selector_enabled()) and _parse_bool(llm_cfg.get("enabled", True), True)
        w_llm = _safe_float(rw.get("llm_weight", 0.12), 0.12) if llm_enabled else 0.0
        fallback_reason = ""
        retrieval_analysis = _memory_query_analysis(query, query_bundle if isinstance(query_bundle, dict) else {}, "normal")
        explicit_recall_probe = bool(retrieval_analysis.get("explicit_recall_intent", False))
        recall_focus = _recall_focus_text(query) if explicit_recall_probe else ""
        # Phase Q.X.3 (2026-05-04) — source-asking query 豁免 cross_char_digest decay
        # 上一轮 LLM reply 引用 cross_char_digest（"听 X 说用户..."）后、Phase Q.X.1 把
        # importance 从 0.70 → 0.55 防复读。但用户立刻追问"谁告诉你这件事的？" 时、
        # 那条 episodic 因 importance 降分被 Filter A/B 过滤、角色就找不回 source、
        # 退化成无关 working turn 凑回复。检测 source-asking 关键词、命中时
        # 把 _cross_char_digest_match 强制 True、走豁免 + score boost +0.30 通道。
        # 不动 decay 本身（保留它防多日复读的核心目的）。
        #
        # 关键词分两类：
        #   (a) 直接 source-asking——用户不记得名字、用代词追问
        #       "谁告诉你"、"听谁说的"、"从哪知道"、"怎么知道"
        #   (b) 转述动词句式（"和你说"这一大类）——用户在追问角色间转述
        #       "和你说"、"和你讲"、"和你提"、"跟你说"、"告诉你"、"对你说"、"给你说" 等
        #       这些通常出现在 follow-up（"角色C和你说过这事？"、"她跟你提到过吗"）。
        _SOURCE_ASKING_KEYWORDS = (
            # (a) 直接 source-asking 句式
            "谁告诉", "谁说的", "谁说过", "谁跟你说",
            "听谁说", "听谁讲", "从谁", "从哪听", "从哪知道",
            "怎么知道", "哪儿听", "哪里听", "哪听到", "哪听说",
            "在哪听", "是谁说", "是谁告诉", "听说什么", "怎么听说",
            # (b) 转述动词句式（user → character）
            "和你说", "和你讲", "和你提", "和你聊", "和你说过",
            "跟你说", "跟你讲", "跟你提", "跟你聊", "跟你说过",
            "对你说", "对你讲", "对你提",
            "告诉你", "告诉过你",
            "给你说", "给你讲", "给你提",
            # Phase Q.X.5 (2026-05-04): 转述动词句式（character → user）
            # user 视角下、自己作为接收方："角色C和我说" / "她告诉我"
            "和我说", "和我讲", "和我提", "和我聊", "和我说过",
            "跟我说", "跟我讲", "跟我提", "跟我聊", "跟我说过",
            "对我说", "对我讲", "对我提",
            "告诉我", "告诉过我",
            "给我说", "给我讲", "给我提",
        )
        _source_query_intent = any(kw in query for kw in _SOURCE_ASKING_KEYWORDS)
        # Phase Q.X.5 (2026-05-04) — user 用第一人称代词指代自己（"我"≡开发者本名）
        # user 不会自称"用户"、所以"角色C告诉你我的事吗" / "你跟角色E说过我吗"
        # 这种 query 含的"我"应该被识别为隐含提及 user 名字。命中时、同样把含
        # user 名字的 cross_char_digest（receiver / sender 两路径）视为 match。
        #
        # Phase Q.X.5.1: 句式扩展、覆盖「把我的XX告诉XX」这种结构
        # （单字"我"会误激活——"我去"/"我觉得"/"我也是"这些日常用语；
        # "我的"+任意中文也容易误激活——"我的天"/"我的妈"/"我的歌"。
        # 折衷：用 phrase 限定 self-reference 出现在"作为对话主体/事件对象"语境）。
        _SELF_PRONOUN_KEYWORDS = (
            # 实词组合（self 作 topic）
            "我的事", "我的事情", "我的话", "我的情况", "我的状态",
            "我的想法", "我的打算", "我的计划", "我的烦恼", "我的近况",
            "我自己", "我那边", "我这边", "我怎么", "关于我",
            # 末尾代词（self 作 topic 的疑问尾）
            "我吗", "我啊", "我呢",
            # 转述动词 + 我（self 作 receiver/object）
            "提到我", "说我", "讲我", "聊我", "问我", "记得我",
            # Phase Q.X.5.1 句式：把我的XX...（self 作 patient + 实词隔断）
            # "把我的备考的事告诉角色C" / "把我的近况说给角色E" 等结构
            "把我的", "把我",
            # 反过来：XX把YY告诉给我
            "告诉给我", "说给我", "讲给我",
        )
        _user_self_reference = any(kw in query for kw in _SELF_PRONOUN_KEYWORDS)
        _dev_name = _developer_display_name()
        grounding = _derive_grounding_signals(query, query_bundle if isinstance(query_bundle, dict) else {}, {})
        grounding = _derive_grounding_signals(query, query_bundle if isinstance(query_bundle, dict) else {}, {})
        cue_bundle = {
            "entities": [str(x or "").strip() for x in (query_bundle.get("entities", []) if isinstance(query_bundle, dict) and isinstance(query_bundle.get("entities", []), list) else []) if str(x or "").strip()][:12],
            "topics": [
                str(x or "").strip()
                for x in (
                    (query_bundle.get("topic_tags", []) if isinstance(query_bundle, dict) and isinstance(query_bundle.get("topic_tags", []), list) else [])
                    + (query_bundle.get("recent_user_msgs", []) if isinstance(query_bundle, dict) and isinstance(query_bundle.get("recent_user_msgs", []), list) else [])
                    + ([query] if query else [])
                )
                if str(x or "").strip()
            ][:8],
            "time_anchor": str(query_bundle.get("time_anchor", "recent") or "recent") if isinstance(query_bundle, dict) else "recent",
            "emotion_anchor": str(query_bundle.get("intent_bucket", "") or "") if isinstance(query_bundle, dict) else "",
            "session_phase": str(query_bundle.get("session_phase", "") or "") if isinstance(query_bundle, dict) else "",
            "school_phase": str(query_bundle.get("school_phase", "") or "") if isinstance(query_bundle, dict) else "",
            "location_anchor": str(query_bundle.get("location_anchor", "") or "") if isinstance(query_bundle, dict) else "",
        }

        candidate_limit = max(220, int(max_items) * 26)
        if explicit_recall_probe:
            # Explicit recall probes need a wider first pass: semantic matching
            # cannot rescue old memories that never enter the candidate pool.
            candidate_limit = max(candidate_limit, 800)
        # [QueryExpand 规则版切分 (2026-08-04)] 提前计算变体——
        # 1) _query_variants：embedding 变体（≤3，含 recall_focus），预算紧、embed 成本高
        # 2) _lex_variants：词法变体（≤6，子句 + 关键词 + 同义替换 + recall_focus），
        #    词法 rel 便宜，同时服务候选池 FTS/LIKE 扩展 + 词法门禁。
        #    必须排在 _query_items 之前，否则老记忆永远进不了池。
        _query_semantics = _split_query_semantics(query, char)
        try:
            _query_variants = _expand_query_semantically(query, char)
        except Exception:
            _query_variants = [query]
        if recall_focus and recall_focus != query and recall_focus not in _query_variants:
            _query_variants.insert(1, recall_focus)
            _query_variants = _query_variants[:3]
        _lex_variants: list[str] = list(_query_variants)
        if _semantic_query_expand_enabled():
            for _lv in (_query_semantics.get("clauses", []) + _query_semantics.get("keywords", [])
                        + _rule_expand_query(query, char)
                        + ([recall_focus] if recall_focus else [])):
                _lv2 = str(_lv or "").strip()
                if len(_lv2) >= 2 and _lv2 != query and _lv2 not in _lex_variants:
                    _lex_variants.append(_lv2)
                    if len(_lex_variants) >= 6:
                        break
        rows = self._query_items(char, limit=candidate_limit, query_text=query,
                                 extra_search_texts=_lex_variants[1:])
        cands = []
        temporal_query_mode = str(query_bundle.get("temporal_query_mode", "recent_general") or "recent_general") if isinstance(query_bundle, dict) else "recent_general"
        _use_actr = mem_actr_recency_enabled()

        # Compute query embedding(s) for all candidates — A 阶段：多变体 query expansion
        _query_emb = None  # primary embedding (kept for backward compat)
        _query_embs: list = []  # all variants
        _emb_active = False
        _backfill_ids: list[tuple[str, str, str, str, int]] = []
        # Phase G.M2 (2026-05-02): 走 module-level _emb_* 单点引用
        if w_emb > 0.0 and _embedding_callable():
            try:
                for _qv in _query_variants:
                    _qe = self._encode_memory_query_dense(_qv)
                    if _qe is not None:
                        _query_embs.append(_qe)
                if _query_embs:
                    _query_emb = _query_embs[0]
                    _emb_active = True
                    if len(_query_variants) > 1:
                        _tsz_emit(f"[QueryExpand] {char} '{query[:40]}' → +{len(_query_variants)-1} variants: {_query_variants[1:]}", flush=True)
            except Exception:
                pass

        # Phase B (2026-05-14): BM25 (sparse) query encode
        # 仅在 EMBEDDING_MODEL_M3=true + USE_RRF_FUSION=true 双 gate 时启用
        _query_sparse = None
        _bm25_active = False
        if use_rrf_fusion() and _emb_is_m3_mode and _emb_encode_sparse and _embedding_callable():
            try:
                if _emb_is_m3_mode():
                    _query_sparse = self._encode_memory_query_sparse(query)
                    if _query_sparse:
                        _bm25_active = True
            except Exception as _e:
                _tsz_emit(f"[BM25] query encode failed: {_e}", flush=True)

        # Motivation layer: fetch active goals once per retrieval
        _motivations: list[dict] = []
        if w_mot > 0.0 and mem_motivation_boost_enabled():
            try:
                _motivations = self.get_active_motivations(char, top_k=3) or []
            except Exception:
                _motivations = []

        for r in rows:
            # Living summaries have a dedicated CURRENT_STATE_FRAME channel.
            # Keep this API-level guard even though SQL excludes them, because
            # cached or alternate candidate providers may bypass _query_items.
            if str(r.get("item_type", "") or "") == "core_summary":
                continue
            tx = str(r.get("text", "") or "").strip()
            if not tx:
                continue
            generic_surface = _memory_surface_text_is_generic(tx)
            if explicit_recall_probe and generic_surface:
                continue
            is_working_turn = _memory_row_is_working(dict(r))
            relevance = _lexical_similarity(query, tx)
            # [QueryExpand 变体感知 lexical (2026-08-04)] 主 query rel 低于门禁时，
            # 用 _lex_variants（子句/关键词/同义替换/recall_focus）里的最高 rel 覆盖——
            # "演唱会"→"演出" 这类同义桥接在没加载 embedding 时也能过门禁、让切分真正生效。
            if relevance < min_relevance and len(_lex_variants) > 1:
                for _v in _lex_variants[1:]:
                    _vr = _lexical_similarity(_v, tx)
                    if _vr > relevance:
                        relevance = _vr
            focus_relevance = _lexical_similarity(recall_focus, tx) if recall_focus else 0.0
            focus_keyword_score = _recall_focus_keyword_score(recall_focus, tx) if recall_focus else 0.0
            _gate_payload = _json_loads(str(r.get("payload_json", "") or "{}"), {})
            _gate_meta = _infer_memory_meta_from_row(
                dict(r), _gate_payload if isinstance(_gate_payload, dict) else {}
            )
            _guest_developer_candidate = bool(
                isinstance(query_bundle, dict)
                and query_bundle.get("_guest_developer_focus", False)
                and _DEVELOPER_FOCUS_RE.search(tx)
                and str(_gate_meta.get("memory_type", "") or "")
                in ("episodic_event", "collective_event", "shared_fact")
            )
            _cross_char_digest_match = False
            try:
                _sk = str(_gate_payload.get("source_kind", "") or "")
                if _sk == "cross_char_daily_digest":
                    _td = str(_gate_payload.get("told_by", "") or "").strip()
                    # 2026-09-12: 仅在"转述语境"下才把 told_by 提及当作 source 命中——
                    # 单纯提到角色名（"你知道角色A喜欢什么吗"）时角色A是话题而非来源，
                    # +0.30 boost 会让二手 digest 盖过角色自己的一手记忆。
                    _cross_char_digest_match = bool(
                        (
                            _td
                            and _query_mentions_character(query, _td)
                            and re.search(r"(说|提|告诉|讲|聊|谈)", query)
                        )
                        or _source_query_intent
                    )
                elif _sk == "cross_char_daily_digest_self_record":
                    _to_list = _gate_payload.get("told_to", []) or []
                    _cross_char_digest_match = bool(
                        isinstance(_to_list, list)
                        and any(
                            _query_mentions_character(query, str(_to or "").strip())
                            for _to in _to_list
                            if str(_to or "").strip()
                        )
                    ) or _source_query_intent
                if (
                    not _cross_char_digest_match
                    and _user_self_reference
                    and _dev_name
                    and _sk in ("cross_char_daily_digest", "cross_char_daily_digest_self_record")
                    and _dev_name in tx
                ):
                    _cross_char_digest_match = True
            except Exception:
                _cross_char_digest_match = False
            # Biomimetic gate: lexical OR strong-semantic gate.
            # - Strong lexical match (>=min_relevance) passes directly.
            # - Lexical weak but embedding strong (cos >=0.55) passes —— this is the
            #   "direct memory probe" case: user asks 「你还记得 X 吗」and emb matches X.
            # - Both weak → 记忆睡着、不被打扰（人不记得一个月前朋友说的每句话）
            # Preview embedding is cached in _prefetched_emb to avoid double compute downstream.
            _prefetched_emb = None
            # 2026-05-16: 移除 FTS hit 后门豁免（原: rel>=min OR _fts_hit）。
            # FTS5 只是 SQL 加速索引、字面命中不保证语义相关。日志 case：
            # query「下次吧、我要是看到了...」rel=0.05 但 FTS 命中"我下午没课"
            # 这种含「我」字的旧 episodic、score 0.86 顶到 r1。
            # 移除后语义弱的条目走下方 line 7459 的 emb cosine fallback 自适应阈值、
            # 这才是真正的"短 query → 长 item"语义路径。
            _passes_gate = bool(
                relevance >= min_relevance
                or bool(r.get("_hybrid_dense_hit", False))
                or bool(r.get("_hybrid_sparse_hit", False))
                or bool(r.get("_hybrid_structured_hit", False))
                or _guest_developer_candidate
                or _cross_char_digest_match
                or (
                    explicit_recall_probe
                    and focus_keyword_score >= 0.45
                )
            )
            # Phase B.2 (2026-05-01) / Phase G.S1 (2026-05-02): biomimetic gate 自适应阈值
            # - 用户直问 "你还记得 X 吗"（explicit_recall_probe=True）时、emb 阈值放松到 0.50：
            #   用户主动唤起、应该更愿意把语义相邻的记忆推进 pool
            # - 普通对话场景仍用 0.55：避免无关记忆被无故唤起
            # 阈值 0.50 是经过 0.45 试错后的中庸值（0.45 把 emb 0.45-0.55 noise 拉进
            # pool、挤出真 canon、canon 反掉 10pp）
            # Phase G.S1: 之前两侧都是 0.55、explicit_recall_probe 完全无作用、B.2 自适应
            # 形同未接、为修复审计 S1 重新启用真正的二元分支。
            # Phase A (2026-05-14): bge-m3 real-world calibration
            # bge-m3 sentence-pair cosine: 同义 0.81-0.86, 无关≤0.52
            # BUT real memory items are 100-500 char conversations, queries are 3-30
            # chars - this asymmetry crushes cosine by 0.20-0.35 vs sentence pairs.
            # Real distribution (query→长文本): median~0.33-0.46, p95~0.50-0.58.
            # Gate at 0.42 = ~p40 for related, ~p95 for unrelated (safe).
            _emb_gate = 0.38 if explicit_recall_probe else 0.42  # Phase A: real-calibrated (曾 bge-small 0.50/0.55)
            if not _passes_gate and _emb_active:
                _item_emb_text_pre = str(r.get("embedding_text", "") or "")
                if _item_emb_text_pre and _embedding_row_matches_current(r):
                    try:
                        _item_emb_pre = _emb_decode(_item_emb_text_pre)
                        if _item_emb_pre is not None:
                            # A 阶段：多变体 query 取 max cosine、不只 primary
                            _emb_pre_score = max((_emb_cos(_qe, _item_emb_pre) for _qe in _query_embs), default=0.0)
                            r["_dbg_emb_pre"] = round(_emb_pre_score, 3)
                            if _emb_pre_score >= _emb_gate:
                                _passes_gate = True
                                _prefetched_emb = _item_emb_pre
                    except Exception:
                        pass
            if not _passes_gate:
                if _mem_debug_re is not None and _mem_debug_re.search(str(r.get("text", "") or "")):
                    _tsz_emit(
                        f"[MemDebug] GATE-FAIL rel={relevance:.3f} min_rel={min_relevance} "
                        f"emb={_safe_float(r.get('_dbg_emb_pre', 0.0), 0.0):.3f} gate={_emb_gate} "
                        f"dense={bool(r.get('_hybrid_dense_hit'))} sparse={bool(r.get('_hybrid_sparse_hit'))} "
                        f"struct={bool(r.get('_hybrid_structured_hit'))} text={str(r.get('text',''))[:60]}",
                        flush=True,
                    )
                continue
            # Dynamic half-life: important memories decay slower
            _item_imp = _safe_float(r.get("importance", 0.5), 0.5)
            if _use_actr:
                rec = _actr_recency_score(
                    str(r.get("created_at", "") or r.get("updated_at", "") or ""),
                    str(r.get("lifecycle_json", "") or ""),
                    _safe_int(r.get("access_count", 0), 0),
                )
            else:
                _dynamic_hl = 72.0 + _item_imp * 144.0  # range: 72h (low) to 216h (high)
                rec = _recency_score(_memory_recency_timestamp(dict(r)), half_life_hours=_dynamic_hl)
            imp = _clamp(0.5 * _safe_float(r.get("importance", 0.5), 0.5) + 0.25 * _safe_float(r.get("confidence", 0.5), 0.5) + 0.25 * _safe_float(r.get("salience", 0.5), 0.5))
            src = _source_prior(str(r.get("layer", "") or ""))
            nov = _clamp(_safe_float(r.get("novelty", 0.4), 0.4))
            payload = _json_loads(str(r.get("payload_json", "") or "{}"), {})
            explicit_archive = _memory_row_is_explicit_recall_archive(dict(r), payload if isinstance(payload, dict) else {})
            if explicit_archive and not explicit_recall_probe:
                continue
            if explicit_archive and explicit_recall_probe and not _memory_explicit_archive_matches_query(dict(r), query, recall_focus):
                continue
            if temporal_query_mode == "present" and _payload_marks_non_retrievable_present_state(payload):
                continue
            row_grounding = _derive_grounding_signals(tx, query_bundle if isinstance(query_bundle, dict) else {}, payload if isinstance(payload, dict) else {})
            cue = _json_loads(str(r.get("cue_json", "") or "{}"), {})
            cue_match = _overlap_ratio(
                cue_bundle.get("entities", []),
                cue.get("entities", []) if isinstance(cue.get("entities", []), list) else [],
            )
            if str(cue.get("session_phase", "") or "").strip() and str(query_bundle.get("time_anchor", "") or "").strip() in ("today", "recent", "near_future"):
                cue_match = max(cue_match, 0.42)
            temporal = _time_anchor_compat(str(grounding.get("query_anchor", "recent")), str(row_grounding.get("item_anchor", "recent")))
            spatial = _overlap_ratio(
                list(grounding.get("query_entities", [])) + list(grounding.get("query_tokens", [])),
                list(row_grounding.get("payload_entities", [])) + list(row_grounding.get("payload_tokens", [])),
            )
            # Adaptive lexical weight: short queries need higher lexical weight
            _qlen = len(query.replace(" ", ""))
            _eff_lex = w_lex + 0.08 if _qlen <= 4 else (w_lex - 0.06 if _qlen > 12 else w_lex)
            # FTS5 hit bonus
            _fts_bonus = 0.08 if r.get("_fts_hit") else 0.0
            # Embedding similarity (semantic recall)
            _emb_score = 0.0
            _eff_emb = 0.0
            if _emb_active:
                # Reuse prefetched embedding from biomimetic gate if available
                _item_emb = _prefetched_emb
                _row_embedding_current = _embedding_row_matches_current(r)
                _m3_sparse_missing = bool(
                    callable(_emb_is_m3_mode)
                    and _emb_is_m3_mode()
                    and not str(r.get("embedding_sparse_text", "") or "")
                )
                if _item_emb is None and _row_embedding_current:
                    _item_emb_text = str(r.get("embedding_text", "") or "")
                    if _item_emb_text:
                        try:
                            _item_emb = _emb_decode(_item_emb_text)
                        except Exception:
                            pass
                # Backfill stale/missing dense metadata and M3 sparse in one forward pass.
                if (
                    (not _row_embedding_current or _item_emb is None or _m3_sparse_missing)
                    and tx
                    and not bool(r.get("_skip_embedding_backfill", False))
                ):
                    try:
                        dense, sparse = _emb_encode_both(tx)
                        model_id, declared_dim = _current_embedding_spec()
                        actual_dim = len(dense) if dense is not None else 0
                        if dense is not None and (declared_dim <= 0 or actual_dim == declared_dim):
                            _item_emb = dense
                            _backfill_ids.append(
                                (
                                    str(r.get("item_id", "")),
                                    _emb_to_text(dense),
                                    _emb_sparse_to_text(sparse) if sparse else "",
                                    model_id,
                                    actual_dim,
                                )
                            )
                    except Exception as exc:
                        _tsz_emit(f"[MemRecall] lazy embedding backfill failed: {exc}", flush=True)
                if _item_emb is not None:
                    # A 阶段：多变体 query 取 max cosine、深层语义联想
                    _emb_score = max(
                        (max(0.0, _emb_cos(_qe, _item_emb)) for _qe in _query_embs),
                        default=0.0
                    )
                    _eff_emb = w_emb
            _hybrid_dense_score = _safe_float(r.get("_hybrid_dense_score", 0.0), 0.0)
            if _hybrid_dense_score > _emb_score:
                _emb_score = _hybrid_dense_score
                _eff_emb = w_emb
            _recall_match = _direct_recall_match_score({
                "relevance": relevance,
                "focus_relevance": focus_relevance,
                "embedding_score": _emb_score,
                "cue_match": cue_match,
                "focus_keyword_score": focus_keyword_score,
                "_fts_hit": bool(r.get("_fts_hit")),
                "recall_focus": recall_focus,
                "candidate_probe_text": _candidate_is_memory_probe_text(tx),
                "explicit_recall_archive": explicit_archive,
            })
            working_age_hours = _memory_row_age_hours(dict(r)) if is_working_turn else 999999.0
            working_focus = recall_focus if explicit_recall_probe and recall_focus else query
            working_focus_relevance = _lexical_similarity(working_focus, tx) if working_focus else 0.0
            working_focus_keyword_score = _recall_focus_keyword_score(working_focus, tx) if working_focus else 0.0
            working_same_character = str(r.get("character", "") or "").strip() == char
            working_ephemeral_noise = _memory_working_text_is_ephemeral_noise(tx) if is_working_turn else False
            # 2026-05-04 修复：原版 working_direct_lane 4 个 signal 不含 emb、对中文真同义
            # 但词面不同的 working turn 全部 fail（FTS 中文 tokenizer 坏 + 词法/focus_kw 阈值
            # 对中文偏严）。后果：3-4 天前的 working turn 即使 emb 高也会在 direct_pool 阶段
            # 被踢出（line 5760-5776 require working_direct_recall_lane=True）。
            # 加 emb signal: emb_score >= 0.60 让语义强相关的 working turn 进 direct_pool。
            working_direct_lane = bool(
                explicit_recall_probe
                and is_working_turn
                and working_same_character
                and working_age_hours <= 168.0
                and not generic_surface
                and not working_ephemeral_noise
                and (
                    bool(r.get("_fts_hit"))
                    or working_focus_keyword_score >= 0.30
                    or working_focus_relevance >= 0.22
                    or _recall_match >= 0.42
                    or _emb_score >= 0.60  # NEW: 语义强相关也算 direct lane
                )
            )
            working_continuity_lane = bool(
                (not explicit_recall_probe)
                and is_working_turn
                and working_same_character
                and working_age_hours <= 72.0
                and not generic_surface
                and not working_ephemeral_noise
                and (
                    bool(r.get("_fts_hit"))
                    or working_focus_keyword_score >= 0.28
                    or working_focus_relevance >= 0.20
                    or relevance >= 0.16
                )
            )
            # Phase Q.X (2026-05-04): cross_char_daily_digest 在 episodic 层、
            # 不享受 working_direct_lane 豁免；但当 query 提及 told_by 角色名时
            # （"agent_c 提过…"）这条记忆**就是用户找的目标**、Filter A/B 应放行。
            # Phase Q.X.3 (2026-05-04): 当 user 是 source-asking query
            # （"谁告诉你"、"听谁说"、"从哪知道"）时、所有 cross_char_digest 都视为 match、
            # 让被 decay 过的转述记忆能找回 source、不被 Filter A/B 过滤掉。
            # Phase Q.X.4 (2026-05-04): 兼容 sender 自己的 self_record。
            #   · receiver 路径（cross_char_daily_digest）：told_by mention 命中
            #     例："角色C说用户在熬夜" / 或 user 用 source-asking 句式
            #   · sender 路径（cross_char_daily_digest_self_record）：told_to mention 命中
            #     例：用户问角色C「你和角色E说过这事吗？」、角色C自己有 self_record
            #     told_to=[角色A、角色D、角色B、角色E] → query 含「角色E」即命中
            strong_semantic_recall = bool(explicit_recall_probe and _emb_score >= 0.70)
            _hybrid_direct_hit = bool(
                r.get("_hybrid_structured_hit", False)
                or (r.get("_hybrid_dense_hit", False) and _hybrid_dense_score >= 0.52)
                or r.get("_hybrid_sparse_hit", False)
            )
            _recall_floor = 0.15 if focus_keyword_score >= 0.45 else (0.18 if explicit_archive else 0.24)
            if explicit_recall_probe and _recall_match < _recall_floor and not strong_semantic_recall and not _hybrid_direct_hit and not working_direct_lane and not _cross_char_digest_match:
                continue
            # 2026-05-04 修复：原阈值 (emb>=0.70 AND recall_match>=0.30) 对中文 BAAI
            # bge-small-zh-v1.5 偏严——真同义但词面不同的中文句子 cosine 普遍 0.60-0.70、
            # 0.70 卡掉大量真相关项；与上层 biomimetic gate 的 0.55 阈值脱节
            # （gate 说"emb>=0.55 进 pool"、Filter B 又说"必须 0.70 才算 semantic"）。
            # 实证 case：query=「备考复习」、target=「准备考试」、emb=0.665、recall_match=0.279
            # 应该召回但被 skip。降阈值到 (emb>=0.60 AND recall_match>=0.20) 以恢复中文召回。
            # Phase A: bge-m3 real-calibrated. p95 of related≈0.58, so 0.52=~p85 safe cutoff
            semantic_recall_pass = _emb_score >= 0.52 and _recall_match >= 0.20  # Phase A: real-calibrated (曾 0.68 plan/0.60 old)
            archive_keyword_pass = bool(explicit_archive and focus_keyword_score >= 0.45 and _recall_match >= 0.18)
            if explicit_recall_probe and recall_focus and focus_keyword_score < 0.18 and _recall_match < 0.36 and not semantic_recall_pass and not strong_semantic_recall and not archive_keyword_pass and not _hybrid_direct_hit and not working_direct_lane and not _cross_char_digest_match:
                continue
            # When embedding active: reduce lexical weight to make room (softened 0.55→0.40)
            _eff_lex_adj = max(0.10, _eff_lex - _eff_emb * 0.40) if _eff_emb > 0 else _eff_lex
            # Motivation boost: active character goals influence recall
            _mot_score = 0.0
            if w_mot > 0.0 and _motivations:
                _mot_score = _compute_motivation_boost(tx, str(r.get("cue_json", "") or ""), _motivations)
            # Phase B (2026-05-14): BM25 sparse 作为一个独立加权维度
            # sparse dot product 范围通常 0-30、用 /10 归一化到 [0, ~1]
            _bm25_score_raw = 0.0
            _bm25_norm = 0.0
            _eff_bm25 = 0.0
            if _bm25_active:
                _item_sparse_text = (
                    str(r.get("embedding_sparse_text", "") or "")
                    if _embedding_row_matches_current(r)
                    else ""
                )
                if _item_sparse_text:
                    _bm25_score_raw = _bm25_score_pair(_query_sparse, _item_sparse_text)
                    _bm25_norm = min(1.0, _bm25_score_raw / 10.0)
                    _eff_bm25 = w_bm25
            _hybrid_sparse_score = _safe_float(r.get("_hybrid_sparse_score", 0.0), 0.0)
            if _hybrid_sparse_score > _bm25_score_raw:
                _bm25_score_raw = _hybrid_sparse_score
                _bm25_norm = min(1.0, _bm25_score_raw / 10.0)
                _eff_bm25 = w_bm25

            # ── Phase R.W2 (2026-06-16): auto-fact episodic OOS guard ──
            # 写入倒转修复后，非显式 semantic_facts 会先落 episodic/fact_observation，
            # 等 KAIROS/复用信号再蒸馏。数量上升后，不能让这类“未毕业事实观察”仅凭
            # recency/importance/source 等结构信号在无关 query 中出水面；必须有足够
            # query evidence（词面 / focus / embedding / BM25 / cue）才允许进入排序。
            if _memory_row_is_auto_fact_observation(dict(r), payload if isinstance(payload, dict) else {}):
                if explicit_recall_probe:
                    _auto_fact_evidence = bool(
                        relevance >= 0.08
                        or focus_relevance >= 0.10
                        or focus_keyword_score >= 0.18
                        or _emb_score >= 0.45
                        or _bm25_norm >= 0.12
                        or cue_match >= 0.20
                    )
                else:
                    _auto_fact_evidence = bool(
                        relevance >= 0.12
                        or _emb_score >= 0.50
                        or _bm25_norm >= 0.18
                        or cue_match >= 0.28
                    )
                if not _auto_fact_evidence:
                    continue

            score = (
                _eff_lex_adj * relevance
                + _eff_emb * _emb_score
                + _eff_bm25 * _bm25_norm
                + w_rec * rec
                + w_imp * imp
                + w_src * src
                + w_nov * nov
                + w_cue * cue_match
                + w_tmp * temporal
                + w_spa * spatial
                + w_mot * _mot_score
                + _fts_bonus
            )
            # Phase B.1 (2026-05-01): biomimetic dampening
            # 当 lexical 与 embedding 都很弱（query 跟 candidate 真无相关性），
            # rec/imp/src/nov 等结构性信号不应独自把 score 撑到 0.5+。
            # 乘 0.30 让真正"无关记忆睡着"，不影响 working_direct_lane 等加分通道。
            # 不影响 _fts_hit（FTS 命中视为 lex 强信号）。
            #
            # emb 阈值 0.40 而非 0.10：BAAI bge-small-zh-v1.5 对任意短句仍给
            # 0.15-0.30 余弦，但真命中 canon 普遍 ≥0.60，0.40 是 gap 中点。
            if (relevance < 0.10
                    and _emb_score < 0.40
                    and not bool(r.get("_fts_hit"))
                    and cue_match < 0.20):
                score *= 0.30
            # 2026-09-12: 疑问句查询下、词面(rel<0.05)与语义(emb<0.58)都与问题无关
            # 的行不应凭 recency/importance/新颖度等结构先验排前——实证"下一个过
            # 生日的是你？"被当晚高 recency 闲聊行压顶、生日事实行沉底。
            # 阈值 0.58 取自 bge-m3 query→长文本无关分布 p95 上沿。
            _question_shaped_gate = bool(
                explicit_recall_probe
                or re.search(r"(什么|哪|怎么|为什么|为何|多少|吗|呢|？|\?)", query_text)
            )
            if _question_shaped_gate and relevance < 0.05 and _emb_score < 0.58 and not bool(r.get("_fts_hit")):
                score *= 0.5
            # Phase A (2026-05-14): bge-m3 high-cosine bonus
            # 当 bge-m3 语义匹配非常强（≥0.70）时, 加 0.05 直接 bonus
            # 防止语义最强项被 recency/imp/src 合起来 42% 权重淹没
            # 实证：cos=0.753 的 target 被 cos=0.43 项以 0.0027 挤出 top-3
            if _emb_active and _emb_score >= 0.70:
                score = min(1.5, score + 0.05)
            if working_direct_lane:
                score = min(1.5, score + 0.10)
            elif working_continuity_lane:
                score = min(1.5, score + 0.08)
            elif is_working_turn:
                score = max(0.0, score - (0.14 if explicit_recall_probe else 0.08))
            elif generic_surface:
                score = max(0.0, score - 0.18)
            elif explicit_archive and explicit_recall_probe:
                score = min(1.5, score + 0.10)

            # Topic label boost: 细粒度话题标签乘法提升
            # 仅当 item 有 LLM 标注的 topic_labels 且与 query 有命中时触发；
            # 用乘法而非加法，保持无标签记忆行为不变。
            _item_topic_labels = _json_loads(str(r.get("topic_labels_json", "[]") or "[]"), [])
            if _item_topic_labels and isinstance(_item_topic_labels, list):
                _qtl = _infer_topic_labels_from_query(query)
                if _qtl:
                    _tl_overlap = len(set(_item_topic_labels) & set(_qtl))
                    if _tl_overlap >= 2:
                        score = min(1.8, score * 1.35)
                    elif _tl_overlap == 1:
                        score = min(1.8, score * 1.18)
            _entity_boost = _safe_float(r.get("_entity_boost", 0.0), 0.0)
            if _entity_boost > 0.0:
                score = min(1.8, score * _entity_boost)

            if bool(r.get("_hybrid_structured_hit", False)):
                score = min(1.5, score + (0.16 if explicit_recall_probe else 0.10))

            # Phase Q.X (2026-05-04): cross_char_daily_digest 召回 boost
            # 当 query 提及 told_by 角色名时（"agent_c 之前提过…"、"角色B说什么"）、
            # 给 told_by 匹配的转述记忆 boost、让 LLM 更容易召回到「听 X 说」。
            # 同时记 _cross_char_digest_match flag、让下游 direct_pool 过滤豁免。
            # payload 已在前面（gate 路径）parse、这里复用 payload var。
            # _cross_char_digest_match 已在 Filter A 之前算好（line ~5601 上方）。
            # Phase Q.X.3 (2026-05-04): 统一 +0.30 boost（不再区分 source-asking
            # vs told_by mention）。理由：
            #  · working turns 普遍 score=0.40+、cross_char_digest decay 后 base ~0.16、
            #    +0.10 仍排不到 top。
            #  · 命中条件已经够严：(a) told_by 角色名出现在 query 或 (b) source-asking
            #    句式（"和你说"/"谁告诉你"/"听谁说"等）。两者都是用户 actively pull
            #    那条 source、推到 top 不会噪。
            #  · cross_char_digest 数量本身少（每角色每天 1-3 条）、副作用可控。
            if _cross_char_digest_match:
                score = min(1.5, score + 0.30)

            # Phase C (2026-07-28): 情绪-记忆双向耦合
            # 当 query_bundle 里有 affect_state 时、按角色当前情绪态对记忆分数做乘法调整。
            # emotional_weight 从 payload.memory_policy.memory_features 提取；
            # payload 已在本轮 line ~8405 parse、这里直接复用。
            _affect_state_qb = (query_bundle.get("affect_state") if isinstance(query_bundle, dict) else None)
            if _affect_state_qb and isinstance(_affect_state_qb, dict):
                _ew = _extract_payload_emotional_weight(payload if isinstance(payload, dict) else {})
                score = _apply_affect_memory_boost(score, _ew, _affect_state_qb)

            # 2026-09-12 召回迭代: 回声行降权。
            # 文本本身是一条"回忆提问"（如"话说，角色A记得我们认识多久了吗"）时，
            # 它与疑问句 query 词面/语义同形、reranker 也会给高分，但它只是
            # 问题的回声、不承载答案，会盖过真正的事实行。profile 提升过的
            # probe answer 除外。
            if _candidate_is_memory_probe_text(tx) and not _profile_promoted_probe_user_answer(
                tx, query_text, r, payload if isinstance(payload, dict) else {}
            ):
                score *= 0.55
            if _mem_debug_re is not None and _mem_debug_re.search(tx):
                _tsz_emit(
                    f"[MemDebug] SCORED s={score:.4f} rel={relevance:.3f} emb={_emb_score:.3f} "
                    f"rec={rec:.3f} imp={imp:.2f} dense={bool(r.get('_hybrid_dense_hit'))} "
                    f"struct={bool(r.get('_hybrid_structured_hit'))} text={tx[:60]}",
                    flush=True,
                )
            # 2026-09-12 召回迭代: 显式回忆 / 疑问句查询时二手转述（"听 X 说"digest）
            # 轻度降权，让一手经历行优先；source-asking 查询（"谁告诉你的"）豁免——
            # 那类问题要的就是 digest 本身。非 source-asking 时它仍是唯一信息源
            # 可凭底分留在池中。
            _question_shaped = bool(
                explicit_recall_probe
                or re.search(r"(什么|哪|怎么|为什么|为何|多少|吗|呢|？|\?)", query_text)
            )
            if (
                _question_shaped
                and not _source_query_intent
                and str(r.get("item_type", "") or "")
                in ("cross_char_daily_digest", "cross_char_daily_digest_self_record")
            ):
                score *= 0.85
            # 2026-09-12: 时长/日期类疑问句的答案有明确形态（"四个多月"/"7月24日"）。
            # 给命中答案形态的行一个有界加成，防止"我也记不清了"这类非答案回声
            # 凭 reranker 同形高分压过真正的事实行。
            if _question_shaped:
                if re.search(r"(多久|多长时间|几年|几个月|几天)", query_text) and re.search(
                    r"(\d+|一|两|三|四|五|六|七|八|九|十|半)个?(多)?(年|月|天|周|星期|小时)", tx
                ):
                    score += 0.15
                elif _EXPLICIT_DATE_QUERY_RE.search(query_text) and _EXPLICIT_DATE_ANSWER_RE.search(tx):
                    score += 0.12
            # 2026-09-12: 问题不能回答问题。短问句行（早期对话残留："可以呀，
            # 你都有什么计划？"）与疑问句 query 词面同形、但没有答案信息，降权。
            if _question_shaped and len(tx) <= 30 and re.search(r"(吗|呢|？|\?)\s*$", tx):
                score *= 0.7

            cands.append(
                {
                    "item_id": str(r.get("item_id", "") or ""),
                    "character": str(r.get("character", "") or ""),
                    "layer": str(r.get("layer", "") or ""),
                    "memory_role": str(r.get("memory_role", "") or ""),
                    "item_type": str(r.get("item_type", "") or ""),
                    "text": tx,
                    "source": "memory_core:" + str(r.get("layer", "") or ""),
                    "evidence_id": str(r.get("evidence_id", "") or ""),
                    "payload_json": str(r.get("payload_json", "") or ""),
                    "cue_json": str(r.get("cue_json", "") or ""),
                    "lifecycle_json": str(r.get("lifecycle_json", "") or ""),
                    "created_at": str(r.get("created_at", "") or ""),
                    "updated_at": str(r.get("updated_at", "") or ""),
                    "last_accessed_at": str(r.get("last_accessed_at", "") or ""),
                    "status": str(r.get("status", "") or ""),
                    "access_count": int(_safe_int(r.get("access_count", 0), 0)),
                    "relevance": relevance,
                    "focus_relevance": round(focus_relevance, 4),
                    "focus_keyword_score": round(focus_keyword_score, 3),
                    "recency": rec,
                    "importance": imp,
                    "novelty": nov,
                    "cue_match": cue_match,
                    "temporal_score": temporal,
                    "spatial_score": spatial,
                    "heuristic_score": float(score),
                    "score": float(score),
                    "embedding_score": round(_emb_score, 3),
                    "bm25_score": round(_bm25_score_raw, 4),
                    "bm25_normalized": round(_bm25_norm, 4),
                    "direct_recall_match": round(_recall_match, 3),
                    "motivation_score": round(_mot_score, 3),
                    "affect_emotional_weight": round(_extract_payload_emotional_weight(payload if isinstance(payload, dict) else {}), 3),
                    "explicit_recall_archive": bool(explicit_archive),
                    "generic_surface": bool(generic_surface),
                    "working_continuity_lane": bool(working_continuity_lane),
                    "working_direct_recall_lane": bool(working_direct_lane),
                    "working_age_hours": round(float(working_age_hours), 2) if is_working_turn else None,
                    "working_focus_keyword_score": round(float(working_focus_keyword_score), 3),
                    "working_focus_relevance": round(float(working_focus_relevance), 4),
                    "canonical_key": _memory_row_canonical_key(dict(r)),
                    "evidence_role": _memory_row_evidence_role(dict(r)),
                    "_hybrid_dense_hit": bool(r.get("_hybrid_dense_hit", False)),
                    "_hybrid_sparse_hit": bool(r.get("_hybrid_sparse_hit", False)),
                    "_hybrid_structured_hit": bool(r.get("_hybrid_structured_hit", False)),
                    "_hybrid_promoted_layer": bool(r.get("_hybrid_promoted_layer", False)),
                    # Phase Q.X (2026-05-04) — cross_char_digest 命中标记、direct_pool 豁免用
                    "_cross_char_digest_match": bool(_cross_char_digest_match),
                    "_confidence": _clamp(_safe_float(r.get("confidence", 0.5), 0.5)),
                }
            )

        cands = _apply_memory_correction_dominance(cands)
        # Flush backfill: persist dense+sparse features and their version metadata atomically.
        if _backfill_ids:
            try:
                def _write_backfill(conn: sqlite3.Connection) -> int:
                    updated = 0
                    for item_id, dense_text, sparse_text, model_id, dimension in _backfill_ids:
                        cursor = conn.execute(
                            """
                            UPDATE memory_items
                            SET embedding_text=?, embedding_sparse_text=?,
                                embedding_model_id=?, embedding_dim=?
                            WHERE item_id=? AND (
                                embedding_text='' OR embedding_text IS NULL
                                OR embedding_model_id!=? OR embedding_dim!=?
                                OR (? != '' AND (embedding_sparse_text='' OR embedding_sparse_text IS NULL))
                            );
                            """,
                            (
                                dense_text,
                                sparse_text,
                                model_id,
                                dimension,
                                item_id,
                                model_id,
                                dimension,
                                sparse_text,
                            ),
                        )
                        updated += max(0, int(cursor.rowcount))
                    return updated

                updated = int(self._run_write_transaction(_write_backfill) or 0)
                _tsz_emit(f"[MemRecall] backfilled {updated} embeddings for {char}", flush=True)
            except Exception as exc:
                _tsz_emit(f"[MemRecall] embedding backfill write failed: {exc}", flush=True)

        cands.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        # Phase B (2026-05-14): RRF rerank — 合并 dense 路 + BM25 路独立 rank
        # 启用条件：USE_RRF_FUSION=true + M3 sparse active + 候选 >= 4
        # 输出：cand["rrf_score"] / cand["rrf_rank"] + cands 按 RRF 重排
        # 影响：下游 selected_pool / top_for_llm 都按 RRF 顺序、score 字段保持 heuristic 不动
        if _bm25_active and len(cands) >= 4:
            try:
                dense_ranked = sorted(cands, key=lambda c: -float(c.get("heuristic_score", 0.0) or 0.0))
                bm25_ranked = sorted(cands, key=lambda c: -float(c.get("bm25_score", 0.0) or 0.0))
                dense_ids = [str(c.get("item_id", "")) for c in dense_ranked]
                bm25_ids = [str(c.get("item_id", "")) for c in bm25_ranked]
                rrf_pairs = _rrf_fuse([dense_ids, bm25_ids], k=60)
                rrf_score_map = {iid: s for iid, s in rrf_pairs}
                rrf_rank_map = {iid: rank for rank, (iid, _) in enumerate(rrf_pairs)}
                for _c in cands:
                    _iid = str(_c.get("item_id", ""))
                    _c["rrf_score"] = round(rrf_score_map.get(_iid, 0.0), 5)
                    _c["rrf_rank"] = rrf_rank_map.get(_iid, 9999)
                # 用 RRF rank 重排（score 字段仍是 heuristic、下游 LLM blend 公式不破坏）
                cands.sort(key=lambda c: c.get("rrf_rank", 9999))
            except Exception as _e:
                _tsz_emit(f"[RRF] rerank failed: {_e}", flush=True)

        # Cognitive enrichment: spreading activation → lateral inhibition
        # Runs on the full candidate pool before LLM re-ranking, so the LLM
        # sees a more diverse and coherent candidate set.
        if len(cands) >= 3:
            if mem_spreading_activation_enabled():
                try:
                    _apply_spreading_activation(cands, boost_factor=0.10)
                except Exception:
                    pass
            if mem_lateral_inhibition_enabled():
                try:
                    _apply_lateral_inhibition(cands, inhibit_factor=0.06)
                except Exception:
                    pass
            # Sync heuristic_score to reflect post-adjustment scores, then re-sort
            # 2026-05-14: blend RRF score (0.15 weight) so RRF reorder survives enrichment sort
            for _c in cands:
                _base = float(_c.get("score", 0.0) or 0.0)
                _rrf = float(_c.get("rrf_score", 0.0) or 0.0)
                # Normalize rrf_score: typical range ~0.005-0.025, scale to ~0-1
                _rrf_norm = min(1.0, max(0.0, _rrf * 40.0))
                _c["heuristic_score"] = round(_base * 0.85 + _rrf_norm * 0.15, 4)
            cands.sort(key=lambda x: float(x.get("heuristic_score", 0.0) or 0.0), reverse=True)

        llm_map: dict[str, float] = {}
        # Phase C (2026-05-14): Cross-encoder reranker 优先于 LLM rerank
        # 启用条件：USE_CROSS_ENCODER_RERANK=true + FlagEmbedding + reranker 模型可加载
        # 启用时：把 reranker_score 写进 llm_map（沿用现有 w_llm blend 公式）、
        # 替代 LLM rerank（节省 1-2s 延迟 + API cost、精度通常 +5-10pp）
        # 失败时：自动 fallback 到 LLM rerank（如果可用）
        _reranker_used = False
        _expensive_rerank_eligible = _memory_should_run_expensive_reranker(cands, retrieval_analysis)
        if cands and w_llm > 0.0 and _expensive_rerank_eligible:
            try:
                from tsuzuri import reranker_engine as _re_eng
                if _re_eng.is_reranker_ready() and _re_eng.is_reranker_enabled():
                    _re_top_k = _re_eng.get_top_k()
                    _re_pool = cands[: max(6, min(_re_top_k, len(cands)))]
                    _docs = [str(c.get("text", "") or "") for c in _re_pool]
                    # 2026-05-16 性能定位：cross-encoder reranker 单次推理耗时
                    _t_rerank = time.perf_counter()
                    _scores = _re_eng.rerank_pairs(query, _docs)
                    _rerank_ms = (time.perf_counter() - _t_rerank) * 1000
                    if _scores is not None and len(_scores) == len(_re_pool):
                        # Reranker logit 范围 ~[-15, +15]、归一化到 [0, 1]
                        # 用 sigmoid: 1 / (1 + exp(-x))
                        import math
                        for _c, _s in zip(_re_pool, _scores):
                            _norm = 1.0 / (1.0 + math.exp(-_s))
                            llm_map[str(_c.get("item_id", ""))] = _norm
                            _c["reranker_score"] = round(float(_s), 4)
                        _reranker_used = True
                        _tsz_emit(f"[Reranker] reranked {len(_re_pool)} candidates t={_rerank_ms:.0f}ms", flush=True)
            except Exception as _e:
                _tsz_emit(f"[Reranker] rerank failed (will fallback to LLM): {_e}", flush=True)
                _reranker_used = False

        if not _reranker_used and cands and callable(llm_selector) and w_llm > 0.0 and llm_enabled and _expensive_rerank_eligible:
            top_for_llm = [dict(x) for x in cands[: min(16, max(6, int(max_items) * 3))]]
            try:
                llm_map = llm_selector(query, top_for_llm) or {}
            except Exception:
                llm_map = {}
                fallback_reason = "llm_selector_error"
        elif (not _reranker_used) and w_llm > 0.0 and _expensive_rerank_eligible and (not callable(llm_selector)):
            fallback_reason = "llm_selector_unavailable"
        elif cands and w_llm > 0.0 and not _expensive_rerank_eligible:
            fallback_reason = "reranker_not_needed"

        for c in cands:
            llm_score = _clamp(_safe_float(llm_map.get(str(c.get("item_id", "")), 0.0), 0.0))
            c["llm_score"] = llm_score
            c["score"] = _clamp((1.0 - w_llm) * _safe_float(c.get("heuristic_score", 0.0), 0.0) + w_llm * llm_score, 0.0, 1.5)
            # 2026-05-16 修 B：rel<0.05 hard penalty score×0.5。
            # 日志 case：semantic「用户对他人学业有比较心态」emb=0.44 rel=0.00 但 score=0.99
            # （high importance + RRF + reranker 累计推顶）→ 跨多个无关 query 都召回到 r1。
            # rel<0.05 表示 query 与 row 几乎完全无关、即使其他维度高、也不该顶 top。
            # explicit_recall 路径下方会用 _direct_recall_match 重算 score、本 penalty 自动失效（合理：用户在 recall、低 rel 可能仍有用）。
            if _safe_float(c.get("relevance", 0.0), 0.0) < 0.05:
                c["score"] = round(c["score"] * 0.5, 4)

        if explicit_recall_probe:
            for c in cands:
                recall_match = _safe_float(c.get("direct_recall_match", 0.0), 0.0)
                # 2026-09-12: direct_recall_match 依赖 focus 词面重合，对长 focus /
                # 答案词不在 query 里的问题（"挑的生日礼物是什么"→"抹茶糖"）系统性
                # 偏低；此时高强度 embedding 是唯一直接证据，用作 recall_match 的
                # 下限（封顶 0.65，不压过真正的词面直中）。
                _emb_c = _safe_float(c.get("embedding_score", 0.0), 0.0)
                if _emb_c >= 0.55:
                    recall_match = max(recall_match, min(0.65, _emb_c * 0.85))
                archive_bonus = 0.06 if bool(c.get("explicit_recall_archive", False)) else 0.0
                working_bonus = 0.08 if bool(c.get("working_direct_recall_lane", False)) else 0.0
                structured_bonus = 0.18 if bool(c.get("_hybrid_structured_hit", False)) else 0.0
                # Phase Q.X (2026-05-04): cross_char_digest 命中时加 0.10 直接 bonus
                # （explicit_recall_probe final score 公式里 heuristic_score 系数 0.16、
                # 在 cands.append 前加的 +0.10 boost 会被稀释成 +0.016——这里专属 bonus
                # 与 working_bonus 同款、保证 query 提及 told_by 时 cross_char_digest 真排前）
                cross_char_bonus = 0.10 if bool(c.get("_cross_char_digest_match", False)) else 0.0
                c["score"] = _clamp(
                    0.68 * recall_match
                    + 0.16 * _safe_float(c.get("score", 0.0), 0.0)
                    + 0.08 * _safe_float(c.get("importance", 0.0), 0.0)
                    + 0.04 * _safe_float(c.get("cue_match", 0.0), 0.0)
                    + 0.04 * _safe_float(c.get("llm_score", 0.0), 0.0)
                    + archive_bonus
                    + working_bonus
                    + structured_bonus
                    + cross_char_bonus,
                    0.0,
                    1.5,
                )

        cands.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
        selected_pool = cands[: max(20, int(max_items) * 8)]
        # 2026-09-12 召回迭代: 语义强但词面/时效弱的行保底进入 MMR 池。
        # 12 维加权里 lexical+recency 占比高，emb 高分的旧行/长文本行可能排在
        # top-N 之外、永远到不了 MMR——这是"答案在库里却漏召回"的结构性来源。
        _pool_seen = {str(c.get("item_id", "") or "") for c in selected_pool}
        _emb_overflow = sorted(
            (
                c
                for c in cands
                if str(c.get("item_id", "") or "") not in _pool_seen
                and _safe_float(c.get("embedding_score", 0.0), 0.0) >= 0.55
            ),
            key=lambda x: _safe_float(x.get("embedding_score", 0.0), 0.0),
            reverse=True,
        )[:12]
        if _emb_overflow:
            selected_pool = selected_pool + _emb_overflow
        if explicit_recall_probe:
            direct_pool = [
                c
                for c in selected_pool
                if (
                    (not _memory_row_is_working(c) or bool(c.get("working_direct_recall_lane", False)))
                    and not bool(c.get("generic_surface", False))
                    and (
                        bool(c.get("working_direct_recall_lane", False))
                        or bool(c.get("explicit_recall_archive", False))
                        or bool(c.get("_hybrid_structured_hit", False))
                        or bool(c.get("_fts_hit", False))
                        or bool(c.get("_cross_char_digest_match", False))  # Phase Q.X: cross_char 命中豁免
                        or _safe_float(c.get("direct_recall_match", 0.0), 0.0) >= 0.30
                        or _safe_float(c.get("focus_keyword_score", 0.0), 0.0) >= 0.18
                        or _safe_float(c.get("focus_relevance", 0.0), 0.0) >= 0.16
                        # 2026-09-12: 语义强证据行豁免——focus/direct 信号对长 focus
                        # 复合问题系统性偏弱，emb≥0.55 的一手行不应因此被挤出
                        # 显式回忆的 MMR 池（实证：抹茶糖/出意外行的死因）。
                        or _safe_float(c.get("embedding_score", 0.0), 0.0) >= 0.55
                    )
                )
            ]
            stable_direct_pool = [c for c in direct_pool if not _memory_row_is_working(c)]
            working_direct_pool = [c for c in direct_pool if bool(c.get("working_direct_recall_lane", False))]
            if stable_direct_pool:
                selected = self._mmr_select(stable_direct_pool, k=max_items, lambda_val=mmr_lambda)
                if len(selected) < max_items and working_direct_pool:
                    used = {str(x.get("item_id", "") or "") for x in selected}
                    selected.extend(self._mmr_select([c for c in working_direct_pool if str(c.get("item_id", "") or "") not in used], k=max_items - len(selected), lambda_val=mmr_lambda))
            else:
                stable_pool = direct_pool or [
                    c for c in selected_pool
                    if (not _memory_row_is_working(c) or bool(c.get("working_direct_recall_lane", False)))
                    and not bool(c.get("generic_surface", False))
                ]
                selected = self._mmr_select(stable_pool, k=max_items, lambda_val=mmr_lambda) if stable_pool else []
            if len(selected) < max_items:
                used = {str(x.get("item_id", "") or "") for x in selected}
                filler = [
                    c for c in selected_pool
                    if str(c.get("item_id", "") or "") not in used
                    and (not _memory_row_is_working(c) or bool(c.get("working_direct_recall_lane", False)))
                    and not bool(c.get("generic_surface", False))
                ]
                selected.extend(self._mmr_select(filler, k=max_items - len(selected), lambda_val=mmr_lambda))
        else:
            selected = self._mmr_select(selected_pool, k=max_items, lambda_val=mmr_lambda)
        # Phase G.M5 (2026-05-02): 把 explicit_recall_probe 传到 collective 路径、
        # gate 与 episodic 对称
        collective_rows = self.retrieve_collective_memory_context(
            char, query,
            query_bundle=query_bundle if isinstance(query_bundle, dict) else {},
            max_items=max(2, int(max_items // 2) or 2),
            explicit_recall_probe=explicit_recall_probe,
        )
        collective_ids = {str(x.get("item_id", "") or "") for x in collective_rows if isinstance(x, dict)}

        run_id = "ret_" + uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        selected_ids = {str(x.get("item_id", "") or "") for x in selected} | collective_ids
        if log_retrieval or (touch_selected and selected_ids):
            with self._lock:
                conn = self._connect()
                try:
                    if log_retrieval:
                        for c in cands[: min(48, len(cands))]:
                            cid = str(c.get("item_id", "") or "")
                            conn.execute(
                                """
                                INSERT INTO retrieval_logs
                                (run_id, character, query_text, query_bundle_json, candidate_item_id, candidate_text, source, score, relevance, recency, importance, llm_score, selected, evidence_id, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    run_id,
                                    char,
                                    query,
                                    _json_dumps(query_bundle if isinstance(query_bundle, dict) else {}),
                                    cid,
                                    str(c.get("text", "") or "")[:300],
                                    str(c.get("source", "") or ""),
                                    float(_safe_float(c.get("score", 0.0), 0.0)),
                                    float(_safe_float(c.get("relevance", 0.0), 0.0)),
                                    float(_safe_float(c.get("recency", 0.0), 0.0)),
                                    float(_safe_float(c.get("importance", 0.0), 0.0)),
                                    float(_safe_float(c.get("llm_score", 0.0), 0.0)),
                                    1 if cid in selected_ids else 0,
                                    str(c.get("evidence_id", "") or ""),
                                    now,
                                ),
                            )
                        for c in collective_rows[: min(18, len(collective_rows))]:
                            cid = str(c.get("item_id", "") or "")
                            conn.execute(
                                """
                                INSERT INTO retrieval_logs
                                (run_id, character, query_text, query_bundle_json, candidate_item_id, candidate_text, source, score, relevance, recency, importance, llm_score, selected, evidence_id, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                                """,
                                (
                                    run_id,
                                    char,
                                    query,
                                    _json_dumps(query_bundle if isinstance(query_bundle, dict) else {}),
                                    cid,
                                    str(c.get("text", "") or "")[:300],
                                    "memory_core:collective",
                                    float(_safe_float(c.get("collective_score", 0.0), 0.0)),
                                    float(_safe_float(c.get("collective_score", 0.0), 0.0)),
                                    float(_recency_score(str(c.get("updated_at", "") or ""))),
                                    float(_safe_float(c.get("importance", 0.0), 0.0)),
                                    0.0,
                                    1,
                                    str(c.get("evidence_id", "") or ""),
                                    now,
                                ),
                            )
                    if touch_selected and selected_ids:
                        placeholders = ",".join(["?"] * len(selected_ids))
                        selected_rows = conn.execute(
                            f"SELECT item_id, access_count, lifecycle_json FROM memory_items WHERE item_id IN ({placeholders});",
                            list(selected_ids),
                        ).fetchall()
                        for sr in selected_rows:
                            current_lifecycle = _json_loads(str(sr["lifecycle_json"] or "{}"), {})
                            refreshed = self._refresh_lifecycle_on_retrieval(current_lifecycle if isinstance(current_lifecycle, dict) else {}, now=now)
                            next_access_count = int(max(0, _safe_int(sr["access_count"], 0))) + 1
                            conn.execute(
                                """
                                UPDATE memory_items
                                SET access_count=?, last_accessed_at=?, lifecycle_json=?
                                WHERE item_id=?;
                                """,
                                (
                                    next_access_count,
                                    now,
                                    _json_dumps(refreshed),
                                    str(sr["item_id"] or ""),
                                ),
                            )
                    conn.commit()
                finally:
                    conn.close()

        self._log_policy_decision(
            decision_kind="retrieval",
            phase="final",
            character=char,
            run_id=run_id,
            input_payload={
                "query": query[:220],
                "candidate_count": len(cands),
                "max_items": int(max_items),
                "min_relevance": round(min_relevance, 4),
                "mmr_lambda": round(mmr_lambda, 4),
                "weights": {
                    "lexical_weight": w_lex,
                    "recency_weight": w_rec,
                        "importance_weight": w_imp,
                        "source_weight": w_src,
                        "novelty_weight": w_nov,
                        "temporal_weight": w_tmp,
                        "spatial_weight": w_spa,
                        "llm_weight": w_llm,
                    },
                },
            llm_payload={"score_count": len(llm_map)},
            fallback_reason=fallback_reason,
            final_payload={
                "selected_count": len(selected),
                "selected_ids": [str(x.get("item_id", "") or "") for x in selected[:12]],
                "collective_ids": [str(x.get("item_id", "") or "") for x in collective_rows[:12]],
            },
        )

        # 2026-05-04 修复：把 [MemRecall] log 移到 retrieve_memory_context 出口、
        # 不管 caller 是 宿主应用 旧路径还是 chat_server provider 新路径都能看到。
        # toggle: env MEM_RECALL_LOG=0 全局关闭；env MEM_RECALL_SILENT=1 临时静默
        # （后者由 KAIROS health_check / dev skill 等程序内部短期路径设置、try/finally
        # 清掉、不影响主对话路径的 log 可见性）。
        try:
            import os as _os_log
            _silent = _os_log.environ.get("MEM_RECALL_SILENT", "").strip().lower() in ("1", "true", "on", "yes")
            _log_enabled = _os_log.environ.get("MEM_RECALL_LOG", "1").strip().lower() not in ("0", "false", "off", "no")
            if _log_enabled and not _silent:
                _qd = (query or "")[:60].replace("\n", " ")
                _recall_ms = (time.perf_counter() - _t_recall) * 1000
                if selected:
                    _tsz_emit(f"[MemRecall] {char} Q='{_qd}' -> {len(selected)} rows  (collective={len(collective_rows)}) t={_recall_ms:.0f}ms", flush=True)
                    for _i, _r in enumerate(selected[:6]):
                        _t = (_r.get("text", "") or "")[:60].replace("\n", " ")
                        _layer = str(_r.get("layer", "?"))
                        _sc = float(_r.get("score", 0.0) or 0.0)
                        _em = float(_r.get("embedding_score", 0.0) or 0.0)
                        _re = float(_r.get("relevance", 0.0) or 0.0)
                        _tsz_emit(f"  r{_i+1} {_layer:10s} score={_sc:.2f} emb={_em:.2f} rel={_re:.2f}  {_t}", flush=True)
                else:
                    _tsz_emit(f"[MemRecall] {char} Q='{_qd}' -> 0 rows  (collective={len(collective_rows)}) t={_recall_ms:.0f}ms", flush=True)
        except Exception:
            pass

        _result_payload = {
            "run_id": run_id,
            "rows": selected,
            "collective_rows": collective_rows,
            "cue_bundle": cue_bundle,
            "cached_max_items": int(max_items or 0),
            "reranker_used": bool(_reranker_used),
            "reranker_eligible": bool(_expensive_rerank_eligible),
        }
        # 2026-05-17 写入 turn-scoped cache（10s TTL）
        # 关键：用 exit 时间戳（time.time() 重取）而非 entry _cache_now。
        # 这样 bge-m3 / reranker 冷启动 30s+ 的耗时不进 TTL 窗口、
        # 同 turn 内的并发重复 caller 仍能命中 cache。
        # 2026-05-23 加 cached_max_items 字段：记录本次 retrieve 是用什么 max_items 调的、
        # 让 0-rows 结果在 cap 足够时仍能 cache hit（见 cache hit 分支注释）。
        _cache_exit_ts = time.time()
        try:
            with self._recall_inflight_lock:
                # poor-man GC: 超 64 项时清掉过期 entry
                if len(self._recall_inflight_cache) > 64:
                    for _k in list(self._recall_inflight_cache.keys()):
                        if (_cache_exit_ts - self._recall_inflight_cache[_k][0]) >= self._recall_inflight_ttl_sec:
                            self._recall_inflight_cache.pop(_k, None)
                self._recall_inflight_cache[_cache_key] = (_cache_exit_ts, dict(_result_payload))
        except Exception:
            pass
        return _result_payload

    def build_memory_context_pack(
        self,
        character: str,
        query_text: str,
        query_bundle: dict | None = None,
        max_items: int = 6,
        *,
        llm_selector: Callable[[str, list[dict]], dict[str, float]] | None = None,
        world_signals: dict | None = None,
        risk_bundle: dict | None = None,
    ) -> dict:
        read_bundle = self.plan_memory_read(
            character,
            query_text,
            query_bundle=query_bundle if isinstance(query_bundle, dict) else {},
            max_items=max_items,
            llm_selector=llm_selector,
            world_signals=world_signals,
            risk_bundle=risk_bundle,
        )
        surface_rows = [dict(x) for x in (read_bundle.get("surface_rows", []) if isinstance(read_bundle.get("surface_rows", []), list) else []) if isinstance(x, dict)]
        filtered_rows = [dict(x) for x in (read_bundle.get("filtered_rows", []) if isinstance(read_bundle.get("filtered_rows", []), list) else []) if isinstance(x, dict)]
        evidence_ids = []
        for r in surface_rows:
            if not isinstance(r, dict):
                continue
            eid = str(r.get("evidence_id", "") or "").strip()
            if eid:
                evidence_ids.append(eid)
        pack = {
            "retrieval_run_id": str(read_bundle.get("retrieval_run_id", "") or ""),
            "selected_items": int(len(surface_rows)),
            "memory_rows": surface_rows,
            "collective_rows": [dict(x) for x in (read_bundle.get("collective_rows", []) if isinstance(read_bundle.get("collective_rows", []), list) else []) if isinstance(x, dict)],
            "cue_bundle": read_bundle.get("cue_bundle", {}) if isinstance(read_bundle.get("cue_bundle", {}), dict) else {},
            "query_bundle": dict(query_bundle) if isinstance(query_bundle, dict) else {},
            "world_signals": world_signals if isinstance(world_signals, dict) else {},
            "risk_bundle": risk_bundle if isinstance(risk_bundle, dict) else {},
            "evidence_ids": evidence_ids[:80],
            "grounding_bundle": _derive_grounding_signals(query_text, query_bundle if isinstance(query_bundle, dict) else {}, world_signals if isinstance(world_signals, dict) else {}),
            "memory_read_bundle": read_bundle if isinstance(read_bundle, dict) else {},
            "memory_surface_plan": dict(read_bundle.get("memory_surface_plan", {}) if isinstance(read_bundle.get("memory_surface_plan", {}), dict) else {}),
            "guest_safe_profile_view": dict(read_bundle.get("guest_safe_profile_view", {}) if isinstance(read_bundle.get("guest_safe_profile_view", {}), dict) else {}),
            "filtered_memory_rows": filtered_rows[:12],
            "blocked_memory_rows": [dict(x) for x in (read_bundle.get("blocked_rows", []) if isinstance(read_bundle.get("blocked_rows", []), list) else []) if isinstance(x, dict)][:12],
            "memory_read_scope": str(read_bundle.get("read_scope", "") or ""),
            "persona_rows": [],
            "thread_rows": [],
            "commitment_rows": [],
            "reference_rows": [],
            "created_at": _utc_now_iso(),
        }
        return pack

    def commit_memory_surface(
        self,
        character: str,
        retrieval_run_id: str,
        memory_surface_plan: dict | None = None,
        *,
        assistant_reply: str = "",
        surfaced: bool | None = None,
        surfaced_ids: list[str] | None = None,
    ) -> dict:
        plan = dict(memory_surface_plan or {}) if isinstance(memory_surface_plan, dict) else {}
        char = resolve_agent(character)
        run_id = str(retrieval_run_id or "").strip()
        selected_rows = [dict(x) for x in (plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []) if isinstance(x, dict)]
        selected_ids = [
            str(x.get("item_id", "") or "").strip()
            for x in selected_rows
            if str(x.get("item_id", "") or "").strip()
        ]
        if not bool(plan.get("should_surface", False)) or (not selected_ids):
            return {
                "ok": True,
                "character": char,
                "retrieval_run_id": run_id,
                "committed": False,
                "reason": "surface_not_requested",
                "committed_ids": [],
            }
        explicit_ids = {
            str(x or "").strip()
            for x in (surfaced_ids if isinstance(surfaced_ids, list) else [])
            if str(x or "").strip()
        }
        reply = str(assistant_reply or "").strip()
        mode = str(plan.get("surface_mode", "") or "").strip().lower()
        used_ids = []
        if explicit_ids:
            used_ids = [mid for mid in selected_ids if mid in explicit_ids]
        elif surfaced is True:
            used_ids = list(selected_ids)
        elif surfaced is False:
            used_ids = []
        else:
            reply_l = reply.lower()
            reply_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", reply_l))
            for row in selected_rows:
                item_id = str(row.get("item_id", "") or "").strip()
                cue = str(row.get("text", "") or "").strip()
                if (not item_id) or (not cue):
                    continue
                cue_l = cue.lower()
                cue_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", cue_l))
                overlap = (len(cue_tokens & reply_tokens) / float(len(cue_tokens))) if cue_tokens else 0.0
                if (cue_l and cue_l in reply_l) or _lexical_similarity(reply_l, cue_l) >= 0.10 or overlap >= 0.34:
                    used_ids.append(item_id)
                    continue
                if (mode == "explicit_recall") and _SURFACE_REPLY_RECALL_RE.search(reply):
                    used_ids.append(item_id)
        used_ids = list(dict.fromkeys([mid for mid in used_ids if mid in selected_ids]))
        if not used_ids:
            self._log_policy_decision(
                decision_kind="memory_surface",
                phase="commit",
                character=char,
                run_id=run_id,
                input_payload={
                    "surface_mode": mode,
                    "selected_ids": selected_ids,
                    "assistant_reply": reply[:220],
                },
                llm_payload={},
                fallback_reason="surface_not_realized",
                final_payload={"committed": False, "committed_ids": []},
            )
            return {
                "ok": True,
                "character": char,
                "retrieval_run_id": run_id,
                "committed": False,
                "reason": "surface_not_realized",
                "committed_ids": [],
            }
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join(["?"] * len(used_ids))
                # 多取 payload_json + importance + salience、用于 Phase Q.X.1/Q.X.5 decay
                selected_db_rows = conn.execute(
                    f"SELECT item_id, access_count, lifecycle_json, payload_json, importance, salience FROM memory_items WHERE item_id IN ({placeholders});",
                    used_ids,
                ).fetchall()
                for sr in selected_db_rows:
                    current_lifecycle = _json_loads(str(sr["lifecycle_json"] or "{}"), {})
                    refreshed = self._refresh_lifecycle_on_retrieval(current_lifecycle if isinstance(current_lifecycle, dict) else {}, now=now)
                    next_access_count = int(max(0, _safe_int(sr["access_count"], 0))) + 1
                    iid = str(sr["item_id"] or "")

                    # Phase Q.X.1/Q.X.5 — cross_char_digest "成功应用"后衰减
                    # 触发：LLM 真在 reply 里复述了「听 X 说」/「我和 X 提到」型记忆。
                    # 设计哲学：朋友转告 vs 亲历——二手信息被用过后应快速淡化，不该
                    # 反复主导 retrieve、也不该 promote 到 semantic 变永久 fact。
                    #
                    # 旧 Q.X.1 只一次性 importance 0.70→0.55，但 salience 仍 0.65 ≥
                    # 当时旧阈值 promote_salience_min (0.60) + access_count ≥ 1 → 直接 promote
                    # 到 semantic，二手转告变 fact。
                    #
                    # 新 Q.X.5 (2026-05-06)：
                    #  · first delivery: imp 0.70→0.40 + sal 0.65→0.40（双双降到
                    #    promote 阈值之下，杜绝 promote）
                    #  · repeat delivery: imp ×0.80, sal ×0.80（累积衰减）
                    #  · 两个指标共用地板 0.15、防止彻底归零
                    #  · self_record（sender 视角）共享同一 decay 路径
                    _next_payload_json: str | None = None
                    _next_importance: float | None = None
                    _next_salience: float | None = None
                    try:
                        _pl = _json_loads(str(sr["payload_json"] or "{}"), {})
                        _sk = str(_pl.get("source_kind", "") or "") if isinstance(_pl, dict) else ""
                        if isinstance(_pl, dict) and _sk in ("cross_char_daily_digest", "cross_char_daily_digest_self_record"):
                            _old_imp = float(sr["importance"] or 0.0)
                            _old_sal = float(sr["salience"] or 0.0)
                            _delivered_count_prev = int(_pl.get("delivered_count", 0) or 0)
                            _pl["delivered_at"] = now
                            _pl["delivered_count"] = _delivered_count_prev + 1
                            _next_payload_json = _json_dumps(_pl)
                            _floor = self._CROSS_CHAR_DIGEST_DELIVERED_FLOOR
                            if _delivered_count_prev == 0:
                                # 首次 delivered：硬截到 first 常量（取 min 以防原值已更低）
                                _new_imp = min(_old_imp, self._CROSS_CHAR_DIGEST_DELIVERED_IMPORTANCE_FIRST)
                                _new_sal = min(_old_sal, self._CROSS_CHAR_DIGEST_DELIVERED_SALIENCE_FIRST)
                            else:
                                # repeat：multiplicative decay with floor
                                _decay = self._CROSS_CHAR_DIGEST_DELIVERED_REPEAT_DECAY
                                _new_imp = max(_floor, _old_imp * _decay)
                                _new_sal = max(_floor, _old_sal * _decay)
                            # 仅在实际下降时才 schedule 写入
                            if _new_imp < _old_imp - 1e-6:
                                _next_importance = _new_imp
                            if _new_sal < _old_sal - 1e-6:
                                _next_salience = _new_sal
                            if _next_importance is not None or _next_salience is not None:
                                _kind_tag = "self_record" if _sk.endswith("_self_record") else "broadcast"
                                _delivery_tag = "first" if _delivered_count_prev == 0 else f"repeat#{_delivered_count_prev + 1}"
                                _imp_str = f"{_old_imp:.2f}→{_new_imp:.2f}"
                                _sal_str = f"{_old_sal:.2f}→{_new_sal:.2f}"
                                _tsz_emit(
                                    f"[CrossCharDigest] {iid[:8]} delivered ({_kind_tag} {_delivery_tag}) → "
                                    f"imp {_imp_str}  sal {_sal_str}",
                                    flush=True,
                                )
                    except Exception as _decay_err:
                        _tsz_emit(f"[CrossCharDigest] decay parse failed for {iid[:8]}: {_decay_err}", flush=True)

                    if _next_payload_json is not None and (_next_importance is not None or _next_salience is not None):
                        # cross_char_digest delivered 路径：写 payload + 视情况写 imp/sal
                        # 用 COALESCE 写法既统一又避免不必要的字段触动
                        _final_imp = _next_importance if _next_importance is not None else float(sr["importance"] or 0.0)
                        _final_sal = _next_salience if _next_salience is not None else float(sr["salience"] or 0.0)
                        # 2026-05-13: delivered_count ≥ AUTO_ARCHIVE_COUNT 时直接 archive
                        # （二手记忆复用 3 次已彻底沉默、留 active 没意义、占召回排序位置）
                        _auto_arch = (
                            _pl.get("delivered_count", 0) >= self._CROSS_CHAR_DIGEST_DELIVERED_AUTO_ARCHIVE_COUNT
                        )
                        if _auto_arch:
                            conn.execute(
                                """
                                UPDATE memory_items
                                SET access_count=?, last_accessed_at=?, lifecycle_json=?, payload_json=?,
                                    importance=?, salience=?, archived=1, status='archived_delivered'
                                WHERE item_id=?;
                                """,
                                (next_access_count, now, _json_dumps(refreshed), _next_payload_json,
                                 _final_imp, _final_sal, iid),
                            )
                            _tsz_emit(
                                f"[CrossCharDigest] {iid[:8]} auto-archived after "
                                f"{_pl.get('delivered_count', 0)} deliveries (二手记忆复用上限到达)",
                                flush=True,
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE memory_items
                                SET access_count=?, last_accessed_at=?, lifecycle_json=?, payload_json=?,
                                    importance=?, salience=?
                                WHERE item_id=?;
                                """,
                                (next_access_count, now, _json_dumps(refreshed), _next_payload_json,
                                 _final_imp, _final_sal, iid),
                            )
                    elif _next_payload_json is not None:
                        # 非首次 delivered 且已经触底（imp/sal 都不再下降）、只更新 delivered_count
                        # 同样需要检查 auto-archive
                        _auto_arch = (
                            _pl.get("delivered_count", 0) >= self._CROSS_CHAR_DIGEST_DELIVERED_AUTO_ARCHIVE_COUNT
                        )
                        if _auto_arch:
                            conn.execute(
                                """
                                UPDATE memory_items
                                SET access_count=?, last_accessed_at=?, lifecycle_json=?, payload_json=?,
                                    archived=1, status='archived_delivered'
                                WHERE item_id=?;
                                """,
                                (next_access_count, now, _json_dumps(refreshed), _next_payload_json, iid),
                            )
                            _tsz_emit(
                                f"[CrossCharDigest] {iid[:8]} auto-archived after "
                                f"{_pl.get('delivered_count', 0)} deliveries (二手记忆复用上限到达)",
                                flush=True,
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE memory_items
                                SET access_count=?, last_accessed_at=?, lifecycle_json=?, payload_json=?
                                WHERE item_id=?;
                                """,
                                (next_access_count, now, _json_dumps(refreshed), _next_payload_json, iid),
                            )
                    else:
                        # 普通 row（非 cross_char_digest）走原路径
                        conn.execute(
                            """
                            UPDATE memory_items
                            SET access_count=?, last_accessed_at=?, lifecycle_json=?
                            WHERE item_id=?;
                            """,
                            (next_access_count, now, _json_dumps(refreshed), iid),
                        )
                conn.commit()
            finally:
                conn.close()
        self._log_policy_decision(
            decision_kind="memory_surface",
            phase="commit",
            character=char,
            run_id=run_id,
            input_payload={
                "surface_mode": mode,
                "selected_ids": selected_ids,
                "assistant_reply": reply[:220],
            },
            llm_payload={},
            fallback_reason="",
            final_payload={"committed": True, "committed_ids": used_ids},
        )
        return {
            "ok": True,
            "character": char,
            "retrieval_run_id": run_id,
            "committed": True,
            "reason": "",
            "committed_ids": used_ids,
            "committed_count": len(used_ids),
        }
