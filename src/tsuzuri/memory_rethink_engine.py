"""LLM-driven memory rethink engine — Phase R.2 (2026-07-25).

Letta sleeptime agent 借鉴。让 LLM 在 KAIROS 凌晨 dream cycle 中审视
semantic 层记忆并提出 rewrite / merge / archive / no_change 编辑建议。

主入口：propose_memory_rethink(core, character, ...) → dict

完整 pipeline：
    1. freeze readonly evidence + select semantic targets
    2. build_rethink_prompt(...) + call LLM
    3. parse batch and require finish_edits=true
    4. validate scope/evidence/duplicate targets + OOC
    5. record every action audit row
    6. apply_rethink_actions(...) in one SQLite transaction
    7. record an explicit terminal run state

设计约束：
    · 仅动 semantic 层（validate_action 强制）
    · canon 守门：prompt 注入 SSOT anchor + 后置 ooc_classifier 二审
    · action 类型只 rewrite / merge / archive / no_change（split 留 P2）
    · 默认 live apply；可用 MEM_RETHINK_SHADOW_ONLY=1 恢复 shadow，audit 始终完整
    · live 批次任一 action 失败则整批零写入；shadow 只记 simulated
    · 失败 fail-safe：parse / finish / canon / CAS 任一失败均留 audit
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from tsuzuri import memory_host as memory_host
from tsuzuri.persona_registry import resolve_agent
from tsuzuri.prompts.memory_rethink_prompt import build_rethink_prompt


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_VALID_ACTION_TYPES = {"rewrite", "merge", "archive", "insert", "no_change"}
_VALID_RESOLUTION_TYPES = {
    "explicit_correction",
    "temporal_update",
    "compatible_merge",
    "obsolete_duplicate",
    "evidence_distillation",
    "no_change",
}
_LLM_MAX_TOKENS = 1600      # 单次 rethink LLM 输出上限
_LLM_TEMPERATURE = 0.3      # 较低、记忆整理要稳重不要花活
_LLM_TIMEOUT_SECS = 60


# ─────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ParsedAction:
    """从 LLM JSON 输出解析出的单条 action（未验证）。"""
    action_type: str
    target_item_ids: list[str] = field(default_factory=list)
    resolution_type: str = ""
    evidence_item_ids: list[str] = field(default_factory=list)
    new_text: str = ""
    reason: str = ""
    parse_error: str = ""


@dataclass
class ValidatedAction:
    """validate + ooc 评分后的 action（可直接 apply）。"""
    parsed: ParsedAction
    validation_status: str = "pending"   # ok / scope_invalid / canon_violation / parse_error
    validation_reason: str = ""
    ooc_score: float = 0.0
    ooc_verdict: str = ""                # ok / warn / reject
    ooc_features: list[str] = field(default_factory=list)
    old_texts: list[str] = field(default_factory=list)


@dataclass
class ParsedRethinkBatch:
    actions: list[ParsedAction] = field(default_factory=list)
    summary: str = ""
    finish_edits: bool = False
    parse_error: str = ""


# ─────────────────────────────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────────────────────────────

def _call_rethink_llm(
    system_msg: str,
    user_msg: str,
    *,
    llm_caller: Callable[[str], str] | None = None,
) -> tuple[str, dict]:
    """调 LLM、返回 (raw_response, meta)。

    meta 包含 {model, total_tokens, fallback_reason, attempts}。
    优先 llm_caller 注入（test 用）、否则走 env LLM (DeepSeek Chat)。
    """
    meta: dict = {"model": "", "total_tokens": 0, "fallback_reason": "", "attempts": 0}

    if callable(llm_caller):
        try:
            combined = system_msg + "\n\n---\n\n" + user_msg
            meta["attempts"] += 1
            raw = str(llm_caller(combined) or "").strip()
            if raw:
                meta["model"] = "injected_caller"
                return raw, meta
            meta["fallback_reason"] = "llm_caller_empty"
        except Exception as exc:
            meta["fallback_reason"] = f"llm_caller_error:{exc}"[:120]

    # env LLM fallback (DeepSeek Chat 默认)
    api_key = (os.getenv("LLM_API_KEY", "") or "").strip()
    base_url = (os.getenv("LLM_BASE_URL", "") or "").strip()
    model = (os.getenv("LLM_MODEL", "") or "").strip()
    if not (api_key and base_url and model):
        meta["fallback_reason"] = meta["fallback_reason"] or "llm_env_missing"
        return "", meta
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=_LLM_TIMEOUT_SECS)
        meta["model"] = model
        for _ in range(2):
            meta["attempts"] += 1
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=_LLM_MAX_TOKENS,
                temperature=_LLM_TEMPERATURE,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = str(resp.choices[0].message.content or "").strip()
            try:
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    meta["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
            except Exception:
                pass
            if raw:
                return raw, meta
            _tsz_emit(f"[MemoryRethink] empty LLM response (attempt {meta['attempts']})", flush=True)
        meta["fallback_reason"] = "env_llm_empty_response_after_retry"
        return "", meta
    except Exception as exc:
        meta["fallback_reason"] = f"env_llm_error:{exc}"[:120]
        return "", meta


# ─────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> str:
    """从可能含 markdown fence 或前后说明的文本里拆出 JSON object 字串。"""
    if not text:
        return ""
    s = text.strip()
    # 1) 试 fence
    m = _JSON_FENCE_RE.search(s)
    if m:
        candidate = m.group(1).strip()
        if candidate:
            s = candidate
    # 2) 找第一个 { 到最后一个 } 之间
    lb = s.find("{")
    rb = s.rfind("}")
    if lb >= 0 and rb > lb:
        return s[lb : rb + 1]
    return s


def parse_rethink_batch(raw_response: str) -> ParsedRethinkBatch:
    """Parse the complete edit batch, including the explicit finish signal."""
    if not raw_response or not raw_response.strip():
        return ParsedRethinkBatch(parse_error="empty_response")
    json_str = _extract_json_object(raw_response)
    try:
        data = json.loads(json_str)
    except Exception as exc:
        return ParsedRethinkBatch(parse_error=f"json_decode_error:{exc}"[:120])
    if not isinstance(data, dict):
        return ParsedRethinkBatch(parse_error="top_level_not_object")

    summary = str(data.get("summary", "") or "").strip()[:200]
    finish_edits = data.get("finish_edits") is True
    raw_actions = data.get("actions", [])
    if not isinstance(raw_actions, list):
        return ParsedRethinkBatch(summary=summary, finish_edits=finish_edits, parse_error="actions_not_list")

    parsed: list[ParsedAction] = []
    for idx, raw_action in enumerate(raw_actions):
        if not isinstance(raw_action, dict):
            parsed.append(ParsedAction(action_type="", parse_error=f"action_{idx}_not_object"))
            continue
        action_type = str(raw_action.get("action_type", "") or "").strip().lower()
        if action_type not in _VALID_ACTION_TYPES:
            parsed.append(ParsedAction(
                action_type=action_type,
                parse_error=f"unknown_action_type:{action_type}",
            ))
            continue
        target_raw = raw_action.get("target_item_ids", [])
        target_ids = [
            str(value or "").strip()
            for value in (target_raw if isinstance(target_raw, list) else [])
            if str(value or "").strip()
        ]
        evidence_raw = raw_action.get("evidence_item_ids", [])
        evidence_ids = [
            str(value or "").strip()
            for value in (evidence_raw if isinstance(evidence_raw, list) else [])
            if str(value or "").strip()
        ]
        parsed.append(ParsedAction(
            action_type=action_type,
            target_item_ids=target_ids,
            resolution_type=str(raw_action.get("resolution_type", "") or "").strip().lower(),
            evidence_item_ids=evidence_ids,
            new_text=str(raw_action.get("new_text", "") or "").strip()[:2000],
            reason=str(raw_action.get("reason", "") or "").strip()[:500],
        ))
    return ParsedRethinkBatch(actions=parsed, summary=summary, finish_edits=finish_edits)


def parse_rethink_response(raw_response: str) -> tuple[list[ParsedAction], str, str]:
    """解析 LLM JSON 输出。

    Returns: (actions, summary, parse_error)
        actions     — list[ParsedAction]、空 list 表示 parse 失败 or 无 action
        summary     — LLM 写的本次总结
        parse_error — 整体解析错误（非单 action 错误）
    """
    batch = parse_rethink_batch(raw_response)
    return batch.actions, batch.summary, batch.parse_error


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def validate_action(
    action: ParsedAction,
    *,
    candidate_ids: set[str],
    evidence_ids: set[str] | None = None,
    allow_insert: bool = False,
) -> tuple[str, str]:
    """格式 + scope 校验。

    返回 (status, reason)：
        status ∈ {ok, parse_error, scope_invalid, format_invalid}
        reason 是人类可读说明
    """
    if action.parse_error:
        return ("parse_error", action.parse_error)
    if action.action_type not in _VALID_ACTION_TYPES:
        return ("format_invalid", f"unknown_action_type:{action.action_type}")
    if action.action_type == "insert" and not allow_insert:
        return ("format_invalid", "insert_not_available_in_single_batch")
    if action.resolution_type not in _VALID_RESOLUTION_TYPES:
        return ("format_invalid", f"unknown_resolution_type:{action.resolution_type}")
    tids = action.target_item_ids or []
    # 数量约束
    if action.action_type == "merge":
        if len(tids) < 2:
            return ("format_invalid", "merge_needs_>=2_targets")
    elif action.action_type == "insert":
        if tids:
            return ("format_invalid", "insert_requires_empty_targets")
    elif action.action_type in {"rewrite", "archive", "no_change"}:
        if len(tids) != 1:
            return ("format_invalid", f"{action.action_type}_needs_exactly_1_target")
    # target ids 必须都在候选池里（防 LLM 编造 id 改其他记忆）
    outside = [t for t in tids if t not in candidate_ids]
    if outside:
        return ("scope_invalid", f"targets_outside_candidate_pool:{outside[:3]}")
    outside_evidence = [item_id for item_id in action.evidence_item_ids if item_id not in (evidence_ids or set())]
    if outside_evidence:
        return ("scope_invalid", f"evidence_outside_readonly_pool:{outside_evidence[:3]}")
    # rewrite / merge 必须有 new_text
    if action.action_type in {"rewrite", "merge", "insert"} and not action.new_text:
        return ("format_invalid", f"{action.action_type}_requires_new_text")
    # archive 必须有 reason
    if action.action_type == "archive" and not action.reason:
        return ("format_invalid", "archive_requires_reason")
    # new_text 长度合理性（防 LLM 复制一大段或空 token）
    if action.new_text and len(action.new_text) < 3:
        return ("format_invalid", "new_text_too_short")
    if action.new_text and len(action.new_text) > 400:
        return ("format_invalid", "new_text_too_long")
    if action.resolution_type in {"explicit_correction", "temporal_update"}:
        if action.action_type != "rewrite" or not action.evidence_item_ids:
            return ("format_invalid", f"{action.resolution_type}_requires_evidence_bound_rewrite")
    if action.resolution_type == "compatible_merge" and action.action_type != "merge":
        return ("format_invalid", "compatible_merge_requires_merge")
    if action.resolution_type == "obsolete_duplicate" and action.action_type not in {"merge", "archive"}:
        return ("format_invalid", "obsolete_duplicate_requires_merge_or_archive")
    if action.action_type == "archive" and not action.evidence_item_ids:
        return ("format_invalid", "archive_requires_readonly_evidence")
    if action.action_type == "insert":
        if action.resolution_type != "evidence_distillation":
            return ("format_invalid", "insert_requires_evidence_distillation")
        if not action.evidence_item_ids:
            return ("format_invalid", "insert_requires_readonly_evidence")
    if action.action_type == "no_change" and action.resolution_type != "no_change":
        return ("format_invalid", "no_change_requires_no_change_resolution")
    return ("ok", "")


# ─────────────────────────────────────────────────────────────────────
# OOC 二审（canon 守门）
# ─────────────────────────────────────────────────────────────────────

def score_action_ooc(
    new_text: str,
    character: str,
    *,
    threshold: float = 0.45,
) -> tuple[float, str, list[str]]:
    """对 new_text 用 ooc_classifier 评 OOC 风险。

    Returns: (confidence, verdict, features_hit)
        confidence ∈ [0, 1]、越高越 OOC
        verdict ∈ {ok, warn, reject}
            ok     — confidence < threshold * 0.8
            warn   — threshold*0.8 ≤ confidence ≤ threshold
            reject — confidence > threshold
        features_hit — 命中的 OOC feature 名列表
    """
    if not new_text or not new_text.strip():
        return (0.0, "ok", [])
    scorer = memory_host.get("ooc_score")
    if scorer is None:
        # 无宿主 OOC 分类器：不额外拦截（canon 校验仍由 rethink 本身的门禁负责）
        return (0.0, "ok", ["ooc_classifier_unavailable"])
    try:
        result = scorer(new_text, resolve_agent(character))
        conf = float(getattr(result, "confidence", 0.0) or 0.0)
        feats = list(getattr(result, "features_hit", []) or [])
    except Exception as exc:
        # ooc_classifier 失败时保守判 warn
        return (0.5, "warn", [f"ooc_classifier_error:{exc}"[:80]])

    thr_warn = float(threshold) * 0.8
    thr_reject = float(threshold)
    if conf > thr_reject:
        verdict = "reject"
    elif conf >= thr_warn:
        verdict = "warn"
    else:
        verdict = "ok"
    return (conf, verdict, feats)


# ─────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────

def _propose_memory_rethink_pass(
    core,                                  # MemoryCore instance
    character: str,
    *,
    candidates: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    shadow: bool | None = None,
    llm_caller: Callable[[str], str] | None = None,
    k: int | None = None,
    max_actions: int | None = None,
    ooc_threshold: float | None = None,
    _run_id: str = "",
    _loop_pass: int = 0,
    _require_finish: bool = True,
    _complete_run: bool = True,
    _seen_targets: set[str] | None = None,
    _seen_insert_evidence_keys: set[str] | None = None,
    _loop_diff_context: str = "",
    _enable_insert: bool = False,
    _protocol_version: str = "rethink_v2",
    _loop_meta: dict | None = None,
) -> dict:
    """LLM 驱动 memory rethink 主入口。

    Args:
        core          — MemoryCore instance（提供 select_candidates / record / apply）
        character     — 角色名
        candidates    — 候选池（None 时自动从 core 选）
        shadow        — None 时走 core._effective_rethink_shadow_mode()
        llm_caller    — 注入 LLM caller（test 用）、None 走 env LLM
        k             — 候选数 override
        max_actions   — LLM 输出 action 上限 override
        ooc_threshold — OOC reject 阈值 override

    Returns:
        dict {
            ok, run_id, character, shadow,
            candidates_count, actions_proposed_count,
            actions_applied_count, actions_rejected_count,
            summary, fallback_reason,
            actions: [ {action_pk, action_type, validation_status, ooc_verdict, applied, ...} ]
        }
    """
    char = resolve_agent(character)

    # 默认参数从 env 取
    if k is None:
        try:
            from tsuzuri.memory_core import mem_rethink_k
            k = int(mem_rethink_k())
        except Exception:
            k = 15
    if max_actions is None:
        try:
            from tsuzuri.memory_core import mem_rethink_max_actions
            max_actions = int(mem_rethink_max_actions())
        except Exception:
            max_actions = 8
    if ooc_threshold is None:
        try:
            from tsuzuri.memory_core import mem_rethink_ooc_threshold
            ooc_threshold = float(mem_rethink_ooc_threshold())
        except Exception:
            ooc_threshold = 0.45

    # shadow 决策
    if shadow is None:
        try:
            shadow, _ = core._effective_rethink_shadow_mode()
            shadow = bool(shadow)
        except Exception:
            shadow = True

    # 1. 先冻结只读新证据，再按证据相关性选择可编辑 semantic 目标。
    if evidence is None:
        try:
            evidence = core.select_rethink_evidence(char)
        except Exception as exc:
            return {
                "ok": False,
                "fallback_reason": f"select_evidence_error:{exc}"[:200],
                "character": char,
            }
    if candidates is None:
        try:
            candidates = core.select_rethink_candidates(char, k=k, evidence=evidence)
        except Exception as exc:
            return {
                "ok": False,
                "fallback_reason": f"select_candidates_error:{exc}"[:200],
                "character": char,
            }

    if not candidates:
        # 空候选直接返回、不开 run
        return {
            "ok": True,
            "run_id": "",
            "character": char,
            "shadow": bool(shadow),
            "candidates_count": 0,
            "actions_proposed_count": 0,
            "actions_applied_count": 0,
            "actions_rejected_count": 0,
            "summary": "候选池为空、跳过",
            "fallback_reason": "no_candidates",
            "actions": [],
        }

    # 2. 开 run；loop 的后续 pass 复用同一个 run_id。
    run_id = str(_run_id or "").strip()
    if not run_id:
        try:
            run_id = core._record_rethink_run_open(
                char,
                candidates,
                shadow=bool(shadow),
                evidence=evidence,
                ooc_threshold=ooc_threshold,
                protocol_version=_protocol_version,
                loop_meta=_loop_meta,
            )
        except Exception as exc:
            return {
                "ok": False,
                "fallback_reason": f"record_run_open_error:{exc}"[:200],
                "character": char,
            }

    # 3. 拼 prompt + 调 LLM
    canon_profile: dict | None = None
    try:
        canon_profile = core.build_character_canon_profile(char)
    except Exception:
        canon_profile = None
    if not shadow and not (isinstance(canon_profile, dict) and canon_profile.get("anchors")):
        core._record_rethink_run_complete(
            run_id,
            status="failed",
            actions_proposed=0,
            actions_applied=0,
            actions_rejected=0,
            fallback_reason="canon_profile_unavailable",
            finish_confirmed=False,
        )
        return {
            "ok": False,
            "run_id": run_id,
            "character": char,
            "shadow": bool(shadow),
            "fallback_reason": "canon_profile_unavailable",
            "actions": [],
        }
    system_msg, user_msg = build_rethink_prompt(
        character=char,
        candidates=candidates,
        canon_profile=canon_profile,
        evidence=evidence,
        max_actions=int(max_actions),
        loop_diff_context=_loop_diff_context,
        enable_insert=bool(_enable_insert),
    )
    raw_response, llm_meta = _call_rethink_llm(system_msg, user_msg, llm_caller=llm_caller)

    if not raw_response:
        try:
            core._record_rethink_run_complete(
                run_id,
                status="failed",
                actions_proposed=0,
                actions_applied=0,
                actions_rejected=0,
                llm_model=str(llm_meta.get("model", "")),
                llm_total_tokens=int(llm_meta.get("total_tokens", 0)),
                llm_raw_response="",
                summary="",
                fallback_reason=str(llm_meta.get("fallback_reason", "llm_empty")),
            )
        except Exception:
            pass
        return {
            "ok": False,
            "run_id": run_id,
            "character": char,
            "shadow": bool(shadow),
            "candidates_count": len(candidates),
            "actions_proposed_count": 0,
            "actions_applied_count": 0,
            "actions_rejected_count": 0,
            "summary": "",
            "fallback_reason": str(llm_meta.get("fallback_reason", "llm_empty")),
            "actions": [],
        }

    # 4. parse response
    parsed_batch = parse_rethink_batch(raw_response)
    parsed_actions = parsed_batch.actions
    summary = parsed_batch.summary
    parse_error = parsed_batch.parse_error
    if parse_error and not parsed_actions:
        try:
            core._record_rethink_run_complete(
                run_id,
                status="failed",
                actions_proposed=0,
                actions_applied=0,
                actions_rejected=0,
                llm_model=str(llm_meta.get("model", "")),
                llm_total_tokens=int(llm_meta.get("total_tokens", 0)),
                llm_raw_response=raw_response,
                summary=summary,
                fallback_reason=f"parse_error:{parse_error}",
            )
        except Exception:
            pass
        return {
            "ok": False,
            "run_id": run_id,
            "character": char,
            "shadow": bool(shadow),
            "candidates_count": len(candidates),
            "actions_proposed_count": 0,
            "actions_applied_count": 0,
            "actions_rejected_count": 0,
            "summary": summary,
            "fallback_reason": f"parse_error:{parse_error}",
            "actions": [],
        }
    if not parsed_batch.finish_edits and _require_finish:
        core._record_rethink_run_complete(
            run_id,
            status="failed_incomplete",
            actions_proposed=len(parsed_actions),
            actions_applied=0,
            actions_rejected=len(parsed_actions),
            llm_model=str(llm_meta.get("model", "")),
            llm_total_tokens=int(llm_meta.get("total_tokens", 0)),
            llm_raw_response=raw_response,
            summary=summary,
            fallback_reason="finish_edits_missing",
            finish_confirmed=False,
        )
        return {
            "ok": False,
            "run_id": run_id,
            "character": char,
            "shadow": bool(shadow),
            "candidates_count": len(candidates),
            "actions_proposed_count": len(parsed_actions),
            "actions_applied_count": 0,
            "actions_rejected_count": len(parsed_actions),
            "summary": summary,
            "fallback_reason": "finish_edits_missing",
            "status": "failed_incomplete",
            "actions": [],
        }
    if len(parsed_actions) > int(max_actions):
        core._record_rethink_run_complete(
            run_id,
            status="failed",
            actions_proposed=len(parsed_actions),
            actions_applied=0,
            actions_rejected=len(parsed_actions),
            llm_model=str(llm_meta.get("model", "")),
            llm_total_tokens=int(llm_meta.get("total_tokens", 0)),
            llm_raw_response=raw_response,
            summary=summary,
            fallback_reason="max_actions_exceeded",
            finish_confirmed=True,
        )
        return {
            "ok": False,
            "run_id": run_id,
            "character": char,
            "shadow": bool(shadow),
            "fallback_reason": "max_actions_exceeded",
            "status": "failed",
            "actions": [],
        }

    # 5. 逐个 validate + ooc + record + apply
    candidate_ids = {str(c.get("item_id", "")) for c in candidates}
    readonly_evidence_ids = {str(item.get("item_id", "")) for item in (evidence or [])}
    text_by_id = {str(c.get("item_id", "")): str(c.get("text", "")) for c in candidates}
    # 加上 cluster member text 也算可见 old_texts
    for c in candidates:
        for m in (c.get("cluster_members", []) or []):
            mid = str(m.get("item_id", "") or "")
            if mid:
                text_by_id.setdefault(mid, str(m.get("text", "") or ""))
                candidate_ids.add(mid)

    action_results: list[dict] = []
    applied_n = 0
    rejected_n = 0
    seen_target_ids: set[str] = set()
    eligible_action_pks: list[int] = []

    for idx, pa in enumerate(parsed_actions):
        status, reason = validate_action(
            pa,
            candidate_ids=candidate_ids,
            evidence_ids=readonly_evidence_ids,
            allow_insert=bool(_enable_insert),
        )
        duplicate_targets = [
            target_id
            for target_id in pa.target_item_ids
            if target_id in seen_target_ids or target_id in (_seen_targets or set())
        ]
        if status == "ok" and duplicate_targets:
            status = "scope_invalid"
            reason = f"duplicate_target_in_run:{duplicate_targets[:3]}"
        insert_evidence_key = "|".join(sorted(pa.evidence_item_ids)) if pa.action_type == "insert" else ""
        if (
            status == "ok"
            and insert_evidence_key
            and insert_evidence_key in (_seen_insert_evidence_keys or set())
        ):
            status = "scope_invalid"
            reason = "duplicate_insert_evidence_in_loop"
        seen_target_ids.update(pa.target_item_ids)
        # OOC 评分（only for rewrite/merge with new_text）
        if status == "ok" and pa.action_type in {"rewrite", "merge", "insert"} and pa.new_text:
            ooc_score, ooc_verdict, ooc_features = score_action_ooc(
                pa.new_text, char, threshold=ooc_threshold,
            )
            if ooc_verdict == "reject":
                status = "canon_violation"
                reason = f"ooc_reject: features={ooc_features[:3]}"
            elif not shadow and any(str(feature).startswith("ooc_classifier_error:") for feature in ooc_features):
                status = "canon_unavailable"
                reason = "ooc_classifier_unavailable_live_fail_closed"
        else:
            ooc_score, ooc_verdict, ooc_features = (0.0, "ok", [])

        # snapshot 原 texts
        old_texts = [text_by_id.get(tid, "") for tid in pa.target_item_ids]
        if status == "ok" and pa.action_type == "archive" and not shadow:
            allow_archive = str(os.getenv("MEM_RETHINK_ALLOW_ARCHIVE", "0") or "0").strip().lower() in {
                "1", "true", "yes", "on",
            }
            if not allow_archive:
                status = "policy_blocked"
                reason = "live_archive_disabled"

        try:
            action_pk = core._record_rethink_action(
                run_id,
                idx,
                pa.action_type,
                target_item_ids=pa.target_item_ids,
                old_texts=old_texts,
                resolution_type=pa.resolution_type,
                evidence_item_ids=pa.evidence_item_ids,
                new_text=pa.new_text,
                reason=pa.reason,
                ooc_score=ooc_score,
                ooc_verdict=ooc_verdict,
                ooc_features=ooc_features,
                validation_status=status,
                validation_reason=reason,
                loop_pass=max(0, int(_loop_pass or 0)),
            )
        except Exception as exc:
            action_results.append({
                "action_index": idx,
                "action_type": pa.action_type,
                "validation_status": "record_error",
                "validation_reason": f"record_action_error:{exc}"[:200],
                "applied": False,
            })
            rejected_n += 1
            continue

        # 先完成整批 validate + audit，之后才进入一次性事务提交。
        should_apply = (status == "ok" and pa.action_type != "no_change")
        if should_apply:
            eligible_action_pks.append(action_pk)
        elif status != "ok":
            rejected_n += 1

        action_results.append({
            "action_index": idx,
            "action_pk": action_pk,
            "action_type": pa.action_type,
            "target_item_ids": pa.target_item_ids,
            "resolution_type": pa.resolution_type,
            "evidence_item_ids": pa.evidence_item_ids,
            "new_text": pa.new_text[:120],
            "reason": pa.reason[:120],
            "validation_status": status,
            "validation_reason": reason,
            "ooc_score": ooc_score,
            "ooc_verdict": ooc_verdict,
            "applied": False,
            "apply_error": "",
            "mutated_item_ids": [],
        })

    # 6. 整批提交：live 模式下任何无效 action 都阻断整批；shadow 仍可模拟其余有效项。
    batch_error = ""
    if eligible_action_pks and (shadow or rejected_n == 0):
        try:
            batch_result = core.apply_rethink_actions(eligible_action_pks, shadow=bool(shadow))
        except Exception as exc:
            batch_result = {"ok": False, "error": f"batch_apply_exception:{exc}"[:200], "actions": []}
        if bool(batch_result.get("ok", False)):
            applied_n = len(eligible_action_pks)
            result_by_pk = {
                int(item.get("action_pk", 0) or 0): item
                for item in (batch_result.get("actions", []) or [])
                if isinstance(item, dict)
            }
            for item in action_results:
                action_pk = int(item.get("action_pk", 0) or 0)
                if action_pk not in result_by_pk:
                    continue
                applied_result = result_by_pk[action_pk]
                item["applied"] = True
                item["mutated_item_ids"] = list(applied_result.get("mutated_item_ids", []) or [])
        else:
            batch_error = str(batch_result.get("error", "batch_apply_failed") or "batch_apply_failed")
            rejected_n += len(eligible_action_pks)
            for item in action_results:
                if int(item.get("action_pk", 0) or 0) in eligible_action_pks:
                    item["apply_error"] = batch_error
    elif eligible_action_pks and not shadow and rejected_n > 0:
        batch_error = "batch_blocked_by_rejected_action"
        rejected_n += len(eligible_action_pks)
        try:
            core.mark_rethink_actions_not_applied(
                eligible_action_pks,
                status="batch_blocked",
                error=batch_error,
            )
        except Exception:
            pass
        for item in action_results:
            if int(item.get("action_pk", 0) or 0) in eligible_action_pks:
                item["apply_error"] = batch_error

    if batch_error and not shadow:
        final_status = "failed_atomic"
    elif not shadow and rejected_n > 0:
        final_status = "failed_validation"
    elif applied_n == 0 and rejected_n == 0:
        # 全 no_change 或没产出实际改动
        final_status = "completed_no_change"
    elif shadow:
        final_status = "completed_partial_shadow" if rejected_n > 0 else "completed_shadow"
    else:
        final_status = "completed_applied" if applied_n > 0 else "completed_no_change"

    if _complete_run:
        try:
            core._record_rethink_run_complete(
                run_id,
                status=final_status,
                actions_proposed=len(parsed_actions),
                actions_applied=applied_n,
                actions_rejected=rejected_n,
                llm_model=str(llm_meta.get("model", "")),
                llm_total_tokens=int(llm_meta.get("total_tokens", 0)),
                llm_raw_response=raw_response,
                summary=summary,
                fallback_reason=str(llm_meta.get("fallback_reason", "")),
                finish_confirmed=bool(parsed_batch.finish_edits),
            )
        except Exception:
            pass

    return {
        "ok": True,
        "run_id": run_id,
        "character": char,
        "shadow": bool(shadow),
        "candidates_count": len(candidates),
        "actions_proposed_count": len(parsed_actions),
        "actions_applied_count": applied_n,
        "actions_rejected_count": rejected_n,
        "summary": summary,
        "fallback_reason": str(llm_meta.get("fallback_reason", "")),
        "status": final_status,
        "finish_confirmed": bool(parsed_batch.finish_edits),
        "llm_model": str(llm_meta.get("model", "")),
        "llm_total_tokens": int(llm_meta.get("total_tokens", 0)),
        "llm_raw_response": raw_response,
        "actions": action_results,
    }


def propose_memory_rethink(
    core,
    character: str,
    *,
    candidates: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    shadow: bool | None = None,
    llm_caller: Callable[[str], str] | None = None,
    k: int | None = None,
    max_actions: int | None = None,
    ooc_threshold: float | None = None,
) -> dict:
    """Run the original one-batch rethink contract.

    Missing ``finish_edits=true`` remains a hard ``failed_incomplete`` error.
    The loop-only insert action is deliberately unavailable here.
    """
    return _propose_memory_rethink_pass(
        core,
        character,
        candidates=candidates,
        evidence=evidence,
        shadow=shadow,
        llm_caller=llm_caller,
        k=k,
        max_actions=max_actions,
        ooc_threshold=ooc_threshold,
    )


def _loop_diff_summary(pass_index: int, actions: list[dict]) -> str:
    parts: list[str] = []
    for action in actions:
        if not isinstance(action, dict) or not bool(action.get("applied", False)):
            continue
        action_type = str(action.get("action_type", "") or "")
        targets = ",".join(str(value or "") for value in (action.get("target_item_ids", []) or []))
        new_text = str(action.get("new_text", "") or "").replace("\n", " ").strip()[:120]
        label = f"pass={pass_index} {action_type}"
        if targets:
            label += f" targets=[{targets}]"
        if new_text:
            label += f" new_text={new_text}"
        parts.append(label)
    return "\n".join(parts)[:2400]


def propose_memory_rethink_loop(
    core,
    character: str,
    *,
    evidence: list[dict[str, Any]] | None = None,
    shadow: bool | None = None,
    llm_caller: Callable[[str], str] | None = None,
    max_iterations: int | None = None,
    k: int | None = None,
    max_actions: int | None = None,
    ooc_threshold: float | None = None,
) -> dict:
    """Run bounded rethink passes while preserving the one-batch contract."""
    char = resolve_agent(character)
    if max_iterations is None:
        try:
            from tsuzuri.memory_core import mem_rethink_max_iterations

            max_iterations = int(mem_rethink_max_iterations())
        except Exception:
            max_iterations = 3
    iteration_limit = max(1, min(5, int(max_iterations or 3)))

    if shadow is None:
        try:
            shadow, _ = core._effective_rethink_shadow_mode()
            shadow = bool(shadow)
        except Exception:
            shadow = True

    cursor = core.get_rethink_cursor(char)
    frozen_evidence = evidence
    if frozen_evidence is None:
        try:
            frozen_evidence = core.select_rethink_evidence(
                char,
                since_ts=str(cursor.get("last_turn_ts", "") or ""),
            )
        except Exception as exc:
            return {
                "ok": False,
                "character": char,
                "shadow": bool(shadow),
                "status": "evidence_snapshot_failed",
                "fallback_reason": f"select_evidence_error:{exc}"[:200],
                "pass_results": [],
                "actions": [],
            }
    frozen_evidence = [dict(item) for item in (frozen_evidence or []) if isinstance(item, dict)]

    run_id = ""
    seen_targets: set[str] = set()
    seen_insert_evidence_keys: set[str] = set()
    pass_results: list[dict] = []
    all_actions: list[dict] = []
    raw_responses: list[str] = []
    diff_context = ""
    totals = {"proposed": 0, "applied": 0, "rejected": 0, "tokens": 0}
    finish_confirmed = False
    fallback_reason = ""
    final_summary = ""
    final_status = "failed_incomplete"

    for pass_index in range(iteration_limit):
        try:
            candidates = core.select_rethink_candidates(
                char,
                k=(15 if k is None else int(k)),
                evidence=frozen_evidence,
            )
        except Exception as exc:
            fallback_reason = f"select_candidates_error:{exc}"[:200]
            final_status = "failed"
            break
        if not candidates:
            finish_confirmed = True
            final_status = "completed_no_change" if not all_actions else (
                "completed_loop_shadow" if shadow else "completed_loop_applied"
            )
            final_summary = "候选池为空，loop 收敛"
            break

        result = _propose_memory_rethink_pass(
            core,
            char,
            candidates=candidates,
            evidence=frozen_evidence,
            shadow=shadow,
            llm_caller=llm_caller,
            k=k,
            max_actions=max_actions,
            ooc_threshold=ooc_threshold,
            _run_id=run_id,
            _loop_pass=pass_index,
            _require_finish=(pass_index == iteration_limit - 1),
            _complete_run=False,
            _seen_targets=seen_targets,
            _seen_insert_evidence_keys=seen_insert_evidence_keys,
            _loop_diff_context=diff_context,
            _enable_insert=True,
            _protocol_version="rethink_v3",
            _loop_meta={
                "max_iterations": iteration_limit,
                "cursor_before": dict(cursor),
            },
        )
        run_id = str(result.get("run_id", run_id) or run_id)
        actions = [dict(item) for item in (result.get("actions", []) or []) if isinstance(item, dict)]
        pass_diff = _loop_diff_summary(pass_index, actions)
        pass_result = {
            "pass_index": pass_index,
            "ok": bool(result.get("ok", False)),
            "status": str(result.get("status", "") or ""),
            "finish_confirmed": bool(result.get("finish_confirmed", False)),
            "actions_proposed_count": int(result.get("actions_proposed_count", 0) or 0),
            "actions_applied_count": int(result.get("actions_applied_count", 0) or 0),
            "actions_rejected_count": int(result.get("actions_rejected_count", 0) or 0),
            "llm_total_tokens": int(result.get("llm_total_tokens", 0) or 0),
            "diff_summary": pass_diff,
        }
        pass_results.append(pass_result)
        all_actions.extend(actions)
        raw_responses.append(str(result.get("llm_raw_response", "") or ""))
        totals["proposed"] += pass_result["actions_proposed_count"]
        totals["applied"] += pass_result["actions_applied_count"]
        totals["rejected"] += pass_result["actions_rejected_count"]
        totals["tokens"] += pass_result["llm_total_tokens"]
        final_summary = str(result.get("summary", "") or final_summary)

        pass_status = pass_result["status"]
        if not pass_result["ok"] or pass_status.startswith("failed"):
            fallback_reason = str(result.get("fallback_reason", "") or pass_status or "loop_pass_failed")
            final_status = pass_status or "failed"
            break

        for action in actions:
            if not bool(action.get("applied", False)):
                continue
            seen_targets.update(
                str(value or "").strip()
                for value in (action.get("target_item_ids", []) or [])
                if str(value or "").strip()
            )
            if str(action.get("action_type", "") or "") == "insert":
                evidence_key = "|".join(sorted(
                    str(value or "").strip()
                    for value in (action.get("evidence_item_ids", []) or [])
                    if str(value or "").strip()
                ))
                if evidence_key:
                    seen_insert_evidence_keys.add(evidence_key)
        if pass_diff:
            diff_context = (diff_context + "\n" + pass_diff).strip()[-2400:]
        if pass_result["finish_confirmed"]:
            finish_confirmed = True
            final_status = "completed_loop_shadow" if shadow else (
                "completed_loop_applied" if totals["applied"] else "completed_no_change"
            )
            break
        if (
            pass_result["actions_applied_count"] == 0
            and pass_result["actions_rejected_count"] == 0
        ):
            final_status = "failed_incomplete"
            fallback_reason = "loop_stalled_without_finish"
            break

    if not finish_confirmed and not final_status.startswith("failed"):
        final_status = "failed_incomplete"
        fallback_reason = fallback_reason or "finish_edits_missing_at_loop_limit"
    if not finish_confirmed and final_status == "failed_incomplete":
        fallback_reason = fallback_reason or "finish_edits_missing_at_loop_limit"

    rolled_back = False
    if run_id and final_status.startswith("failed") and not shadow and totals["applied"] > 0:
        try:
            revert_result = core.revert_rethink_run(
                run_id,
                reason="bounded_loop_failed_after_prior_pass",
            )
            rolled_back = bool(revert_result.get("ok", False))
            if rolled_back:
                totals["applied"] = 0
                final_status = "failed_loop_rolled_back"
        except Exception as exc:
            fallback_reason = (
                fallback_reason + f";loop_rollback_error:{type(exc).__name__}:{exc}"
            ).strip(";")[:200]

    if run_id:
        try:
            core._record_rethink_run_complete(
                run_id,
                status=final_status,
                actions_proposed=totals["proposed"],
                actions_applied=totals["applied"],
                actions_rejected=totals["rejected"],
                llm_model="loop",
                llm_total_tokens=totals["tokens"],
                llm_raw_response=json.dumps(raw_responses, ensure_ascii=False),
                summary=final_summary,
                fallback_reason=fallback_reason,
                finish_confirmed=finish_confirmed,
            )
        except Exception:
            pass

    ok = bool(finish_confirmed and not final_status.startswith("failed"))
    cursor_after = dict(cursor)
    if ok and frozen_evidence:
        latest = max(
            frozen_evidence,
            key=lambda item: str(item.get("updated_at", item.get("created_at", "")) or ""),
        )
        cursor_after = core.advance_rethink_cursor(
            char,
            last_turn_fact_card_id=str(latest.get("turn_card_id", "") or ""),
            last_turn_ts=str(latest.get("updated_at", latest.get("created_at", "")) or ""),
        )

    return {
        "ok": ok,
        "run_id": run_id,
        "character": char,
        "shadow": bool(shadow),
        "status": final_status,
        "finish_confirmed": finish_confirmed,
        "iterations": len(pass_results),
        "actions_proposed_count": totals["proposed"],
        "actions_applied_count": totals["applied"],
        "actions_rejected_count": totals["rejected"],
        "llm_total_tokens": totals["tokens"],
        "summary": final_summary,
        "fallback_reason": fallback_reason,
        "cursor_before": cursor,
        "cursor_after": cursor_after,
        "rolled_back": rolled_back,
        "pass_results": pass_results,
        "actions": all_actions,
    }


# 公开符号
__all__ = [
    "ParsedAction",
    "ParsedRethinkBatch",
    "parse_rethink_batch",
    "ValidatedAction",
    "parse_rethink_response",
    "validate_action",
    "score_action_ooc",
    "propose_memory_rethink",
    "propose_memory_rethink_loop",
]
