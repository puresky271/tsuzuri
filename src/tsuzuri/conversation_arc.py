"""Bounded, provenance-safe ConversationArc projection for turn facts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import os
import re

from tsuzuri.time_source import WORLD_TZ, world_tz_label
from tsuzuri.persona_registry import user_name
from typing import Any, Callable

from tsuzuri.turn_fact_ledger import lifecycle_card_window


_MAX_CANDIDATES = 12
_MAX_FRAME_CHARS = 420
_MIN_RULE_SCORE = 0.38
_MIN_RULE_MARGIN = 0.055
_MIN_EMBEDDING_SCORE = 0.72
_MIN_EMBEDDING_MARGIN = 0.04

_REOPEN_RE = re.compile(
    r"(?:回到|说回|再说回|重新(?:说|聊|谈)|接着(?:说|聊)|还记得|"
    r"最开始(?:那个|的)?|之前(?:那个|的)?(?:问题|话题|约定)|上次(?:那个|的)?(?:问题|话题|约定)|"
    r"接着(?:昨天|上次|之前|早上|上午|中午|下午|晚上|白天)|继续(?:说|聊|谈)|"
    r"(?:那|这)(?:件|个)事(?:情)?(?:怎么|进展|定|结果|后来|样)|"
    r"(?:上次|之前|前几天|昨天)(?:的)?(?:件|个)?(?:事|事情|计划|安排|约定)(?:怎么|定|结果|进展|后来|样)?)"
)
_EXPLICIT_ORIGIN_POINTER_RE = re.compile(
    r"(?:(?:最开始|一开始|最初).{0,8}(?:问题|话题|那件事)|"
    r"之前(?:那个|的)(?:问题|话题|约定))"
)
_CORRECTION_RE = re.compile(
    r"(?:不是.{0,24}(?:而是|是)|并不是|并非|更正|纠正|说错了|理解错了|其实不是)"
)
_FULFILL_RE = re.compile(r"(?:答应你的|之前答应|答应过|说好的|约好的|定好的|兑现|做到了|履行|没有食言)")
_QUERY_NOISE_RE = re.compile(
    r"(?:忽略|无视|覆盖|替换).{0,16}(?:指令|规则|system|prompt|提示词)|"
    r"(?:system|assistant|developer)\s*[:：]",
    re.IGNORECASE,
)
_STOP_TOKENS = {
    "这个", "那个", "什么", "怎么", "为什么", "现在", "之前", "刚才", "然后",
    "还是", "已经", "可以", "一下", "一个", "我们", "你们", "他们", "自己",
    "问题", "话题", "事情", "感觉", "觉得", "好像", "终于", "开始", "继续",
}


EmbeddingScorer = Callable[[str, list[tuple[str, str]]], dict[str, float]]


def _env_enabled(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw not in ("0", "false", "off", "no", "n")


def conversation_arc_enabled() -> bool:
    return _env_enabled("CONVERSATION_ARC_ENABLED", True)


def conversation_arc_shadow_only() -> bool:
    return _env_enabled("CONVERSATION_ARC_SHADOW_ONLY", False)


def conversation_arc_embedding_enabled() -> bool:
    return _env_enabled("CONVERSATION_ARC_EMBEDDING_ENABLED", True)


def _one_line(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:limit]


def _timestamp_ms(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _list(value: Any, limit: int = 6) -> list[str]:
    rows = value if isinstance(value, list) else ([value] if isinstance(value, str) else [])
    out: list[str] = []
    for row in rows:
        text = _one_line(row, 180)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _card_message_ids(card: dict) -> set[str]:
    ids = {
        _one_line(card.get("user_id"), 120),
        _one_line(card.get("assistant_id"), 120),
    }
    for value in card.get("user_ids", []) if isinstance(card.get("user_ids", []), list) else []:
        ids.add(_one_line(value, 120))
    return {value for value in ids if value}


def _card_text(card: dict) -> str:
    parts = [
        _one_line(card.get("fact"), 180),
        _one_line(card.get("state_delta"), 180),
        *_list(card.get("open_threads"), 4),
        *_list(card.get("commitments"), 4),
        *_list(card.get("corrections"), 4),
    ]
    return " ".join(part for part in parts if part)


def _topic_tokens(value: Any) -> set[str]:
    text = re.sub(r"https?://\S+", " ", str(value or "").lower())
    tokens: set[str] = set()
    for word in re.findall(r"[a-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text):
        if word in _STOP_TOKENS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            if len(word) <= 5:
                tokens.add(word)
            for size in (2, 3):
                for idx in range(max(0, len(word) - size + 1)):
                    gram = word[idx:idx + size]
                    if gram not in _STOP_TOKENS:
                        tokens.add(gram)
        else:
            tokens.add(word)
    return tokens


def _overlap(query_tokens: set[str], text: str) -> float:
    record_tokens = _topic_tokens(text)
    if not query_tokens or not record_tokens:
        return 0.0
    shared = query_tokens & record_tokens
    if not shared:
        return 0.0
    return min(1.0, len(shared) / max(3.0, min(len(query_tokens), len(record_tokens)) * 0.55))


def _relation_for_query(query: str) -> str:
    if _REOPEN_RE.search(query):
        return "reopens"
    if _CORRECTION_RE.search(query):
        return "corrects"
    if _FULFILL_RE.search(query):
        return "fulfills"
    return ""


def _candidate_score(card: dict, *, relation: str, query_tokens: set[str]) -> dict:
    text = _card_text(card)
    overlap = _overlap(query_tokens, text)
    salience = max(0.0, min(1.0, float(card.get("salience", 0.5) or 0.5)))
    confidence = max(0.0, min(1.0, float(card.get("confidence", 0.6) or 0.6)))
    has_open = bool(_list(card.get("open_threads"), 1))
    has_commitment = bool(_list(card.get("commitments"), 1))
    has_correction = bool(_list(card.get("corrections"), 1))
    structure = 0.0
    if relation == "reopens":
        structure = 0.22 if has_open else (0.08 if has_commitment else 0.025)
    elif relation == "fulfills":
        structure = 0.26 if has_commitment else 0.0
    elif relation == "corrects":
        structure = 0.18 if (has_open or has_commitment) else 0.06
    score = overlap * 0.5 + structure + salience * 0.1 + confidence * 0.08
    eligible = (
        (relation == "reopens" and has_open and overlap >= 0.025)
        or (
            overlap >= 0.08
            and (relation != "fulfills" or has_commitment)
        )
    )
    return {
        "card": card,
        "text": text,
        "overlap": round(overlap, 4),
        "score": round(score, 4),
        "eligible": eligible,
        "has_open": has_open,
        "has_commitment": has_commitment,
        "has_correction": has_correction,
    }


def _candidate_cards(
    fact_cards: list | None,
    *,
    current_character: str,
    audience_scope: str,
    recent_raw_message_ids: set[str],
    current_timestamp_ms: int,
) -> list[dict]:
    rows = [dict(row) for row in fact_cards or [] if isinstance(row, dict)]
    out: list[dict] = []
    for card in lifecycle_card_window(rows, recent=80, reserve=24):
        if card.get("archived"):
            continue
        character = _one_line(card.get("character"), 40)
        if character and current_character and character != current_character:
            continue
        scope = _one_line(card.get("audience_scope"), 160)
        if scope and audience_scope and scope != audience_scope:
            continue
        if recent_raw_message_ids.intersection(_card_message_ids(card)):
            continue
        card_ts = _timestamp_ms(card.get("created_at_ms"))
        if current_timestamp_ms > 0 and (card_ts <= 0 or card_ts >= current_timestamp_ms):
            continue
        if not _one_line(card.get("turn_id"), 80) or not _card_text(card):
            continue
        out.append(card)
    out.sort(
        key=lambda card: (
            -int(bool(_list(card.get("open_threads"), 1) or _list(card.get("commitments"), 1))),
            -float(card.get("salience", 0.5) or 0.5),
            -_timestamp_ms(card.get("created_at_ms")),
        )
    )
    return out[:_MAX_CANDIDATES]


def _default_embedding_scores(query: str, candidates: list[tuple[str, str]]) -> dict[str, float]:
    if not candidates:
        return {}
    try:
        from tsuzuri import embedding_engine as engine

        allow_cold = _env_enabled("CONVERSATION_ARC_EMBEDDING_ALLOW_COLD_LOAD", False)
        # Availability probes import the full ML stack; a cold shadow scorer
        # must return before paying that cost just to decide it cannot run.
        if (not allow_cold and getattr(engine, "_m3_model_instance", None) is None
                and getattr(engine, "_model_instance", None) is None):
            return {}
        if not engine.is_embedding_enabled():
            return {}
        warm_model = (
            getattr(engine, "_m3_model_instance", None)
            if engine.is_m3_mode()
            else getattr(engine, "_model_instance", None)
        )
        if warm_model is None and not allow_cold:
            return {}
        vectors = engine.encode_texts([query, *[text for _key, text in candidates]])
        if not vectors or vectors[0] is None:
            return {}
        query_vector = vectors[0]
        scores: dict[str, float] = {}
        for (key, _text), vector in zip(candidates, vectors[1:]):
            if vector is None:
                continue
            similarity = engine.cosine_similarity(query_vector, vector)
            if similarity is not None:
                scores[key] = max(0.0, min(1.0, float(similarity)))
        return scores
    except Exception:
        return {}


def _embedding_shadow(
    query: str,
    candidates: list[dict],
    scorer: EmbeddingScorer | None,
) -> dict:
    if not candidates:
        return {"selected": False, "reason": "no_candidates"}
    score_fn = scorer or _default_embedding_scores
    pairs = [(_one_line(card.get("turn_id"), 80), _card_text(card)) for card in candidates]
    try:
        scores = score_fn(query, pairs)
    except Exception as err:
        return {"selected": False, "reason": f"scorer_error:{type(err).__name__}"}
    ranked = sorted(
        ((key, float(value)) for key, value in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return {"selected": False, "reason": "embedding_unavailable"}
    top_id, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    selected = top_score >= _MIN_EMBEDDING_SCORE and margin >= _MIN_EMBEDDING_MARGIN
    return {
        "selected": selected,
        "reason": "semantic_candidate" if selected else "below_threshold_or_margin",
        "turn_id": top_id if selected else "",
        "score": round(top_score, 4),
        "margin": round(margin, 4),
        "candidate_count": len(ranked),
    }


def _source_period(card: dict) -> str:
    raw = _timestamp_ms(card.get("created_at_ms"))
    if raw <= 0:
        return "较早对话"
    try:
        dt = datetime.fromtimestamp(raw / 1000.0, tz=WORLD_TZ)
        return dt.strftime("%Y-%m-%d %H:%M ") + world_tz_label()
    except Exception:
        return "较早对话"


def _pivot_for(origin: dict, candidates: list[dict], query_tokens: set[str]) -> dict:
    origin_ts = _timestamp_ms(origin.get("created_at_ms"))
    origin_id = _one_line(origin.get("turn_id"), 80)
    ranked: list[tuple[float, int, dict]] = []
    for card in candidates:
        if _one_line(card.get("turn_id"), 80) == origin_id:
            continue
        ts = _timestamp_ms(card.get("created_at_ms"))
        if origin_ts and ts and ts <= origin_ts:
            continue
        has_correction = bool(_list(card.get("corrections"), 1))
        has_commitment = bool(_list(card.get("commitments"), 1))
        if not (has_correction or has_commitment):
            continue
        overlap = max(
            _overlap(query_tokens, _card_text(card)),
            _overlap(_topic_tokens(_card_text(origin)), _card_text(card)),
        )
        if overlap < 0.08:
            continue
        bonus = 0.3 if has_correction else 0.18
        ranked.append((overlap + bonus, ts, card))
    if not ranked:
        return {}
    ranked.sort(key=lambda row: (-row[0], -row[1], _one_line(row[2].get("turn_id"), 80)))
    return ranked[0][2]


def _render_frame(relation: str, origin: dict, pivot: dict | None = None) -> str:
    relation_line = {
        "reopens": "当前输入明确要求重新打开这条较早话题。",
        "corrects": "当前输入明确纠正了与这条较早话题有关的说法。",
        "fulfills": "当前输入明确提到正在兑现与这条较早话题有关的约定。",
    }.get(relation, "当前输入与这条较早话题重新相关。")
    head = [
        '<conversation-arc-frame authority="evidence">',
        "【本轮回扣的早期话题】",
        f"- 当前连接：{relation_line}",
    ]
    origin_prefix = f"- 起点：[来源：{_source_period(origin)}] "
    origin_text = _one_line(
        origin.get("fact") or origin.get("near_summary") or _card_text(origin),
        360,
    )
    pivot_prefix = ""
    pivot_text = ""
    if pivot:
        pivot_text = (
            _list(pivot.get("corrections"), 1)
            or _list(pivot.get("commitments"), 1)
            or [_one_line(pivot.get("state_delta") or pivot.get("fact"), 160)]
        )[0]
        if pivot_text:
            pivot_prefix = f"- 关键转折：[来源：{_source_period(pivot)}] "
    tail = [
        f"- 使用边界：这只是有来源的对话脉络，不得替{user_name()}判断内心、动机或关系性质；当前纠正与当前状态优先。",
        "</conversation-arc-frame>",
    ]

    line_count = len(head) + len(tail) + 1 + int(bool(pivot_prefix))
    fixed_chars = (
        sum(len(line) for line in head + tail)
        + len(origin_prefix)
        + len(pivot_prefix)
        + line_count - 1
    )
    content_budget = max(0, _MAX_FRAME_CHARS - fixed_chars)
    if pivot_prefix:
        origin_budget = min(len(origin_text), max(48, int(content_budget * 0.58)))
        pivot_budget = max(0, content_budget - origin_budget)
    else:
        origin_budget = content_budget
        pivot_budget = 0

    def clipped(value: str, budget: int) -> str:
        if len(value) <= budget:
            return value
        if budget <= 3:
            return value[:budget]
        return value[:budget - 3].rstrip() + "..."

    lines = [*head, origin_prefix + clipped(origin_text, origin_budget)]
    if pivot_prefix:
        lines.append(pivot_prefix + clipped(pivot_text, pivot_budget))
    lines.extend(tail)
    return "\n".join(lines)


def _empty(reason: str, *, enabled: bool, shadow_only: bool) -> dict:
    return {
        "enabled": enabled,
        "shadow_only": shadow_only,
        "selected": False,
        "projected": False,
        "reason": reason,
        "arc_id": "",
        "origin_turn_id": "",
        "pivot_turn_ids": [],
        "relation": "",
        "relation_authority": "",
        "score": 0.0,
        "margin": 0.0,
        "candidate_count": 0,
        "frame": "",
        "consumed_turn_ids": [],
        "embedding_shadow": {"selected": False, "reason": "not_run"},
    }


def compile_conversation_arc(
    current_user_text: str,
    fact_cards: list | None,
    recent_raw_message_ids: set[str] | None,
    *,
    current_character: str = "",
    audience_scope: str = "developer",
    current_timestamp_ms: int | None = None,
    proactive: bool = False,
    suppress_for_explicit_evidence: bool = False,
    enabled: bool | None = None,
    shadow_only: bool | None = None,
    embedding_enabled: bool | None = None,
    embedding_scorer: EmbeddingScorer | None = None,
) -> dict:
    """Build one bounded virtual arc; semantic matches remain shadow-only."""
    active = conversation_arc_enabled() if enabled is None else bool(enabled)
    shadow = conversation_arc_shadow_only() if shadow_only is None else bool(shadow_only)
    if not active:
        return _empty("disabled", enabled=False, shadow_only=shadow)
    query = _one_line(current_user_text, 1200)
    if not query:
        return _empty("empty_query", enabled=True, shadow_only=shadow)
    if proactive:
        return _empty("proactive_turn", enabled=True, shadow_only=shadow)
    if suppress_for_explicit_evidence:
        return _empty("explicit_evidence_owns_recall", enabled=True, shadow_only=shadow)
    if _QUERY_NOISE_RE.search(query):
        return _empty("instruction_like_query", enabled=True, shadow_only=shadow)

    candidates = _candidate_cards(
        fact_cards,
        current_character=_one_line(current_character, 40),
        audience_scope=_one_line(audience_scope, 160),
        recent_raw_message_ids=set(recent_raw_message_ids or set()),
        current_timestamp_ms=_timestamp_ms(current_timestamp_ms),
    )
    result = _empty("no_rule_relation", enabled=True, shadow_only=shadow)
    result["candidate_count"] = len(candidates)
    if not candidates:
        result["reason"] = "no_candidates_outside_recent_window"
        return result

    relation = _relation_for_query(query)
    query_tokens = _topic_tokens(query)
    if relation:
        ranked = [
            _candidate_score(card, relation=relation, query_tokens=query_tokens)
            for card in candidates
        ]
        if relation == "reopens" and _EXPLICIT_ORIGIN_POINTER_RE.search(query):
            open_rows = [row for row in ranked if row["has_open"]]
            if len(open_rows) == 1:
                open_rows[0]["eligible"] = True
        ranked = [row for row in ranked if row["eligible"]]
        ranked.sort(
            key=lambda row: (
                -float(row["score"]),
                -_timestamp_ms(row["card"].get("created_at_ms")),
                _one_line(row["card"].get("turn_id"), 80),
            )
        )
        if ranked:
            top = ranked[0]
            second_score = float(ranked[1]["score"]) if len(ranked) > 1 else 0.0
            margin = float(top["score"]) - second_score
            if float(top["score"]) >= _MIN_RULE_SCORE and margin >= _MIN_RULE_MARGIN:
                origin = top["card"]
                pivot = _pivot_for(origin, candidates, query_tokens)
                origin_id = _one_line(origin.get("turn_id"), 80)
                pivot_id = _one_line(pivot.get("turn_id"), 80) if pivot else ""
                arc_seed = f"{origin_id}|{pivot_id}|{relation}".encode("utf-8")
                projected = not shadow
                result.update({
                    "selected": True,
                    "projected": projected,
                    "reason": "rule_edge_selected",
                    "arc_id": "arc-" + hashlib.sha1(arc_seed).hexdigest()[:14],
                    "origin_turn_id": origin_id,
                    "pivot_turn_ids": [pivot_id] if pivot_id else [],
                    "relation": relation,
                    "relation_authority": "evidence",
                    "score": round(float(top["score"]), 4),
                    "margin": round(margin, 4),
                    "frame": _render_frame(relation, origin, pivot) if projected else "",
                    "consumed_turn_ids": [value for value in (origin_id, pivot_id) if value] if projected else [],
                })
                return result
            result["reason"] = "rule_below_threshold_or_margin"
            result["score"] = round(float(ranked[0]["score"]), 4)
            result["margin"] = round(margin, 4)

    use_embedding = (
        conversation_arc_embedding_enabled()
        if embedding_enabled is None
        else bool(embedding_enabled)
    )
    if use_embedding:
        result["embedding_shadow"] = _embedding_shadow(query, candidates, embedding_scorer)
    else:
        result["embedding_shadow"] = {"selected": False, "reason": "disabled"}
    return result


__all__ = [
    "compile_conversation_arc",
    "conversation_arc_embedding_enabled",
    "conversation_arc_enabled",
    "conversation_arc_shadow_only",
]
