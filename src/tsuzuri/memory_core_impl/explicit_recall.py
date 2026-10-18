"""Explicit-recall answer matching, correction dominance and reranker gating.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations

import re
from tsuzuri.persona_registry import live_pattern, user_name
from datetime import datetime, timezone
from tsuzuri.memory_core_impl.config import _safe_float, _safe_int
from tsuzuri.memory_core_impl.memory_rows import (
    _lexical_similarity, _LOW_VALUE_MEMORY_GENERALIZATION_RE,
    _memory_row_has_durable_turn_signal, _memory_row_is_explicit_recall_archive,
    _memory_row_is_surfaceable_working, _memory_row_is_working,
    _memory_surface_text_is_generic, _norm_text, _recall_focus_text,
)
from tsuzuri.memory_core_impl.features import (
    _candidate_is_memory_probe_text, _developer_display_name, _recall_focus_keyword_score,
    CONSOLIDATION_STATE_PROFILE_PROMOTED,
)

def _select_explicit_surface_rows(
    rows: list[dict],
    limit: int = 3,
    *,
    query_text: str = "",
    require_answer_bearing: bool = False,
) -> list[dict]:
    from tsuzuri.memory_core_impl.recall_scoring import _clamp
    ranked: list[tuple[float, dict]] = []
    _q = str(query_text or "").strip()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        if require_answer_bearing and _q and not _memory_explicit_row_is_answer_bearing(item, _q):
            continue
        base = _safe_float(item.get("score", item.get("collective_score", 0.0)), 0.0)
        generic = _memory_surface_text_is_generic(text)
        probe_like = _candidate_is_memory_probe_text(text)
        working = _memory_row_is_working(item)
        focus_keyword = _safe_float(item.get("focus_keyword_score", 0.0), 0.0)
        layer = str(item.get("layer", "") or "").strip().lower()
        rank = base
        surfaceable_working = _memory_row_is_surfaceable_working(item)
        # 2026-05-14: 防止自我强化循环 — 如果 working item 的 user 部分
        # 和当前 query 几乎一模一样（本轮对话回声、不是"记忆"）、重罚挤出。
        if _q and working and len(_q) >= 6:
            _user_part = ""
            if " | " in text:
                _left = text.split(" | ")[0].strip()
                if _left.startswith("user:"):
                    _user_part = _left[5:].strip()
                elif _left.startswith("对方曾说："):
                    _user_part = _left[5:].strip()
            if not _user_part:
                _user_part = text.split(" | ")[0].strip() if " | " in text else ""
            if _user_part and len(_user_part) >= 6:
                _overlap = _lexical_similarity(_q, _user_part)
                if _overlap >= 0.55:
                    rank -= 0.55  # 重罚：这是回声、不是记忆
        if generic:
            rank -= 0.18
        else:
            rank += 0.05
        if working and not surfaceable_working:
            rank -= 0.22
        elif surfaceable_working:
            rank += 0.04
        elif layer == "episodic" and not generic:
            rank += 0.07
        elif layer == "semantic" and not generic:
            rank += 0.03
        if focus_keyword > 0:
            rank += 0.08 * _clamp(focus_keyword)
        if probe_like:
            rank -= 0.06
        item["_surface_rank_score"] = round(rank, 4)
        ranked.append((rank, item))
    ranked.sort(key=lambda x: x[0], reverse=True)
    selected: list[dict] = []
    seen: set[str] = set()
    seen_ids: set[str] = set()
    selection_pool = [
        (rank, item)
        for rank, item in ranked
        if (not _memory_row_is_working(item) or _memory_row_is_surfaceable_working(item))
        and not _memory_surface_text_is_generic(str(item.get("text", "") or ""))
    ]
    # 2026-05-14 fix: 不再把 stable rows 无条件排到 working 前面。
    # stable 优先的本意是防闲聊 working 污染记忆注入、但它把高分 working
    # direct_recall（如"下次见面"）也推到队尾——limit=2 时低分 semantic 挤掉
    # 高分 working、log 实测泛泛的关心句替代了真正被问的那条记忆。
    # semantic/episodic 在评分阶段已有 +0.03/+0.07 bonus、不需要二次重排。
    for _, item in selection_pool:
        item_id = str(item.get("item_id", "") or "").strip()
        if item_id and item_id in seen_ids:
            continue
        key = _norm_text(str(item.get("text", "") or ""))[:100]
        if key and key in seen:
            continue
        if item_id:
            seen_ids.add(item_id)
        if key:
            seen.add(key)
        selected.append(item)
        if len(selected) >= max(1, int(limit or 1)):
            break
    return selected


_EXPLICIT_DATE_QUERY_RE = re.compile(r"(几月几号|几号|哪天|日期|什么时候)")
# 2026-09-12: 仅当 query 真的在问"生日是哪天"才把生日视作日期槽。
# "生日礼物/生日前夜/生日快到了"里的生日只是话题修饰，走日期事实门禁会把
# 真正的答案行（礼物、前夜轶事）全部挡掉、只放行生日日期事实（生日劫持）。
_BIRTHDAY_DATE_ASK_RE = re.compile(
    r"(?:几月几号|几号|哪天|哪日|何时|什么时候|日期).{0,8}生日"
    r"|生日(?!礼物|祝福|快乐|蛋糕|派对|聚餐|当天|惊喜|打算|的|前|后|会).{0,6}(?:是|在|哪天|哪日|几号|什么时候)"
)
_LEGACY_CORRECTION_RE = re.compile(r"(明确纠正|澄清|乌龙|并非|不是.+而是|不再是|应改为)")
_BIRTHDAY_FACT_RE = live_pattern(
    r"(?:{USER}|开发者|用户)(?:的)?生日(?:是|为|在)?\s*"
    r"(?:\d{1,2}月\d{1,2}(?:日|号)|"
    r"(?:一|二|三|四|五|六|七|八|九|十|十一|十二)月"
    r"(?:一|二|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四|二十五|二十六|二十七|二十八|二十九|三十|三十一)(?:日|号))"
)
_HOMETOWN_PLACE_ANSWER_RE = re.compile(r"(海边|海|河|湖|山|公园|街|院子|田野|森林|沙滩|港口|附近|地方)")
_EXPRESSION_QUERY_RE = re.compile(r"(什么表情|怎样的表情|平时.{0,8}表情)")
_EXPRESSION_ANSWER_RE = re.compile(r"(板着(?:张)?脸|面无表情|严肃|皱眉|笑|表情)")
_EXPLICIT_INTERROGATIVE_RE = re.compile(
    r"(什么|哪里|哪儿|哪天|哪种|哪个|哪一个|哪一种|多少|(?<!好)几(?:月|号|点|次|个|岁)?|怎么|如何|是否|有没有)")
_FLAVOR_PREFERENCE_QUERY_RE = re.compile(
    r"(?:什么|哪种)口味(?:的)?(?P<subject>[a-zA-Z0-9\u4e00-\u9fff]{2,20}?)(?:吗|呢|呀|啊|？|\?|$)"
)
_DRINK_ADDITIVE_QUERY_RE = re.compile(
    r"喝(?P<subject>[a-zA-Z0-9\u4e00-\u9fff]{2,12}?)(?:喜欢|通常|一般|会)?(?:加|放)什么"
)
_GENERIC_QUERY_PREDICATE_RE = re.compile(r"(留|加|放|送|带|买|做|说|叫|选|修|答应|回来)")
_GENERIC_QUERY_NOISE_RE = re.compile(
    r"你还记得|还记得|记不记得|记得吗|我之前|之前|我说过|说过|"
    r"不喜欢|喜欢|希望|自己|我的|我们|上次|那件事|这件事|那件|那次|"
    r"以后|平时|到底|曾经|一次|什么|哪里|哪儿|哪天|哪种|哪个|哪一个|"
    r"多少|几月几号|几号|几点|怎么|如何|是否|有没有|的吗|吗|呢|呀|啊|的|了|我|你|他|她"
)


def _query_requires_first_person_fact_ownership(query_text: str) -> bool:
    query = str(query_text or "")
    if "我" not in query:
        return False
    return bool(_EXPLICIT_INTERROGATIVE_RE.search(query) or _EXPLICIT_DATE_QUERY_RE.search(query))
_EXPLICIT_DATE_ANSWER_RE = re.compile(
    r"(?:\d{1,4}[年./-])?\d{1,2}(?:月|[./-])\d{1,2}(?:日|号)?|"
    r"(?:一|二|三|四|五|六|七|八|九|十|十一|十二)月"
    r"(?:一|二|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四|二十五|二十六|二十七|二十八|二十九|三十|三十一)(?:日|号)"
)


def _memory_row_evidence_role(row: dict | None = None) -> str:
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    payload = payload if isinstance(payload, dict) else {}
    card = payload.get("turn_fact_card", {}) if isinstance(payload.get("turn_fact_card", {}), dict) else {}
    role = str(item.get("evidence_role", "") or payload.get("evidence_role", "") or card.get("evidence_role", "")).strip().lower()
    if card.get("corrections"):
        return "correction"
    if _LEGACY_CORRECTION_RE.search(str(item.get("text", "") or "")):
        return "correction"
    return role or "observation"


_FORGETTING_ELIGIBLE_LAYERS = frozenset({"episodic"})
_FORGETTING_PROTECT_ENGRAM_TYPES = frozenset(
    {"explicit_recall_archive", "strategy_rule", "group_session_event"}
)
def _forgetting_protected(
    row: dict | None = None,
    payload: dict | None = None,
) -> tuple[bool, str]:
    """Return whether an archived row is ineligible for true forgetting."""
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    from tsuzuri.memory_core_impl.meta_policy import _memory_payload_meta
    try:
        item = dict(row or {})
    except Exception:
        item = {}
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    pl = pl if isinstance(pl, dict) else {}
    layer = str(item.get("layer", "") or "").strip().lower()
    if layer not in _FORGETTING_ELIGIBLE_LAYERS:
        return True, f"protected_layer:{layer or 'unknown'}"
    engram_type = str(item.get("engram_type", "") or "").strip().lower()
    if engram_type in _FORGETTING_PROTECT_ENGRAM_TYPES:
        return True, f"protected_engram_type:{engram_type}"
    if _memory_row_is_explicit_recall_archive(item, pl):
        return True, "explicit_recall_archive"
    if _memory_row_evidence_role(item) == "correction":
        return True, "correction"
    memory_meta = _memory_payload_meta(pl)
    if str(memory_meta.get("source_kind", "") or "").strip().lower() == "source_backed_archive":
        return True, "source_backed_archive"
    if _safe_float(item.get("salience", 0.0), 0.0) >= 0.70:
        return True, "high_salience"
    if _safe_int(item.get("access_count", 0), 0) > 0:
        return True, "accessed"
    return False, ""


def _memory_row_canonical_key(row: dict | None = None) -> str:
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    payload = payload if isinstance(payload, dict) else {}
    card = payload.get("turn_fact_card", {}) if isinstance(payload.get("turn_fact_card", {}), dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    key = str(
        item.get("canonical_key", "")
        or payload.get("canonical_key", "")
        or card.get("canonical_key", "")
        or meta.get("canonical_key", "")
        or ""
    ).strip().lower()
    if key:
        return key
    text = str(item.get("text", "") or "")
    if "谈恋爱" in text and re.search(r"(想|打算|意图|进展|乌龙|澄清|纠正)", text):
        return "legacy:user:relationship:intent"
    return ""


def _memory_query_answer_anchors(query_text: str) -> set[str]:
    from tsuzuri.memory_core_impl.recall_scoring import _tokenize_goal_words
    cleaned = _GENERIC_QUERY_NOISE_RE.sub(" ", str(query_text or ""))
    cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", " ", cleaned).strip()
    if not cleaned:
        return set()
    anchors = {
        token.strip().lower()
        for token in _tokenize_goal_words(cleaned, min_len=2)
        if len(token.strip()) >= 2
    }
    return {anchor for anchor in anchors if anchor not in {"时候", "地方", "最晚", "严重"}}


def _memory_generic_query_answer_matches(text: str, query_text: str) -> bool:
    text_norm = _norm_text(text).replace(" ", "")
    anchors = _memory_query_answer_anchors(query_text)
    if not text_norm:
        return False
    focus = _recall_focus_text(query_text)
    if focus and focus != str(query_text or "").strip() and _lexical_similarity(focus, text) >= 0.18:
        return True
    if not anchors:
        return False
    hits = {anchor for anchor in anchors if anchor in text_norm}
    if not hits:
        return False
    predicates = set(_GENERIC_QUERY_PREDICATE_RE.findall(str(query_text or "")))
    if predicates and not any(predicate in text_norm for predicate in predicates):
        return False
    return any(len(anchor) >= 3 for anchor in hits) or len(hits) >= 2


def _profile_promoted_probe_user_answer(text: str, query_text: str, row: dict, payload: dict) -> bool:
    """Allow a promoted raw turn only when its user segment states the requested answer."""
    policy = payload.get("memory_policy", {}) if isinstance(payload.get("memory_policy", {}), dict) else {}
    if not (
        str(row.get("layer", "") or "").strip().lower() == "semantic"
        and str(policy.get("consolidation_state", "") or "").strip().lower()
        == CONSOLIDATION_STATE_PROFILE_PROMOTED
    ):
        return False
    match = re.search(r"(?:^|\s)user:(.*?)(?:\s*\|\s*assistant:|$)", text, re.IGNORECASE | re.DOTALL)
    user_text = str(match.group(1) or "").strip() if match else ""
    if not user_text:
        return False
    if _EXPRESSION_QUERY_RE.search(query_text):
        return bool(_EXPRESSION_ANSWER_RE.search(user_text))
    return False


def _drink_additive_positive_answer(text: str, subject: str) -> bool:
    explicit = bool(re.search(r"(喜欢|爱|通常|一般|会).{0,16}(加|放)|(?:加|放).{0,16}(喜欢|爱)", text))
    if explicit:
        return True
    subject_pos = text.find(subject)
    if subject_pos < 0:
        return False
    for match in re.finditer(
        r"(?:喜欢|偏好|爱吃|爱喝)(?P<value>[^，,。；;]{1,20}?)(?=不喜欢|不爱|不要|不加|不放|讨厌|[，,。；;]|$)",
        text,
    ):
        value = re.sub(r"(?:加|放|和|与|及|、|的|口味|配料|小料)", "", str(match.group("value") or ""))
        value = value.replace(subject, "").strip()
        if len(_norm_text(value)) >= 2 and abs(match.start() - subject_pos) <= 40:
            return True
    return False


def _memory_explicit_row_is_answer_bearing(row: dict | None, query_text: str) -> bool:
    """Fail closed when an explicit recall candidate cannot answer the asked slot."""
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    from tsuzuri.memory_core_impl.meta_policy import _DEVELOPER_FOCUS_RE
    item = row if isinstance(row, dict) else {}
    text = str(item.get("text", "") or "").strip()
    query = str(query_text or "").strip()
    if not text or _memory_surface_text_is_generic(text):
        return False

    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    payload = payload if isinstance(payload, dict) else {}
    promoted_probe_answer = bool(
        _candidate_is_memory_probe_text(text)
        and _profile_promoted_probe_user_answer(text, query, item, payload)
    )
    if _candidate_is_memory_probe_text(text) and not promoted_probe_answer:
        return False
    role = _memory_row_evidence_role(item)
    direct = bool(
        role == "correction"
        or item.get("working_direct_recall_lane", False)
        or item.get("explicit_recall_archive", False)
        or item.get("_hybrid_structured_hit", False)
        or _safe_float(item.get("score", item.get("collective_score", 0.0)), 0.0) >= 0.30
        or _safe_float(item.get("direct_recall_match", 0.0), 0.0) >= 0.30
        or _safe_float(item.get("focus_keyword_score", 0.0), 0.0) >= 0.18
        or _safe_float(item.get("focus_relevance", 0.0), 0.0) >= 0.16
        or _safe_float(item.get("embedding_score", 0.0), 0.0) >= 0.52
        or _safe_float(item.get("bm25_normalized", 0.0), 0.0) >= 0.12
    )
    if not direct:
        return False

    # 2026-09-14: 第一人称事实问询（"我最喜欢的花是什么"）要求答案主体是开发者。
    # 该校验必须在所有 early-return True 之前——否则角色档案事实（学习成绩/生日）
    # 靠 embedding 高分从上面的 open-ended/emb 路径直接通过（郁金香 2026-09-14
    # 失败案例：档案行 0.72-0.76 挤掉真正的 developer-grounded 答案）。
    if _DEVELOPER_FOCUS_RE.search(query) or _query_requires_first_person_fact_ownership(query):
        about = str(payload.get("about", "") or "").strip().lower()
        canonical_key = _memory_row_canonical_key(item)
        source_trust = payload.get("source_trust", {}) if isinstance(payload.get("source_trust", {}), dict) else {}
        developer_grounded = bool(
            about in ("developer", "user", user_name())
            or canonical_key.startswith(("user:", "developer:"))
            or _developer_display_name() in text
            or (
                str(source_trust.get("user_text", "") or "").strip().lower() == "verified"
                and bool(str(payload.get("user_text", "") or "").strip())
            )
            or _memory_row_has_durable_turn_signal(item, payload)
            or promoted_probe_answer
        )
        if not developer_grounded:
            return False

    # For open-ended bounded slots ("开过什么玩笑", "做过哪些事") the
    # answer vocabulary is not in a closed rule table. Require both semantic
    # focus alignment and a meaningful embedding signal, rather than treating
    # any row that merely mentions the conversation as an answer.
    if (
        _memory_explicit_query_has_bounded_slot(query)
        and _safe_float(item.get("focus_relevance", 0.0), 0.0) >= 0.07
        and _safe_float(item.get("focus_keyword_score", 0.0), 0.0) >= 0.50
        and _safe_float(item.get("embedding_score", 0.0), 0.0) >= 0.60
    ):
        return True

    # Broad explicit-recall questions ("还记得我之前说过的面试吗？") do not
    # have a bounded answer slot, but a durable semantic/episodic event whose
    # focus anchor is an exact topic match is still answer-bearing.  Keep this
    # fallback away from working-turn echoes; those remain governed by the
    # dedicated working recall lanes above.
    layer = str(item.get("layer", "") or "").strip().lower()
    focus = _recall_focus_text(query)
    recall_signal = bool(re.search(r"(还记得|记不记得|记得吗|回忆|想起|之前|以前|上次|那次|说过|提过|聊过|谈过|讨论过)", query))
    if (
        recall_signal
        and layer in {"semantic", "episodic"}
        and focus
        and (
            _recall_focus_keyword_score(focus, text) >= 0.45
            or _lexical_similarity(focus, text) >= 0.18
        )
    ):
        return True

    # 2026-09-12: 开放式提问（"最想做的三件事是什么来着"）的答案词往往不在
    # query 里，focus/anchor 词面信号系统性失效；此时高强度 embedding 是唯一的
    # 直接证据。仅限 stable 层 + 回忆/疑问语境，且分数门槛高（bge-m3 实测
    # query→长文本无关分布 p95 ≈ 0.50-0.58，0.60 是安全相关区）。
    if (
        (recall_signal or _EXPLICIT_INTERROGATIVE_RE.search(query))
        and layer in {"semantic", "episodic"}
        and _safe_float(item.get("embedding_score", 0.0), 0.0) >= 0.60
    ):
        return True

    if _BIRTHDAY_DATE_ASK_RE.search(query):
        return bool(
            _memory_row_canonical_key(item).startswith(("user:birthday", "developer:birthday"))
            or _BIRTHDAY_FACT_RE.search(text)
        )
    flavor_match = _FLAVOR_PREFERENCE_QUERY_RE.search(query)
    if flavor_match:
        subject = str(flavor_match.group("subject") or "").strip()
        return bool(
            subject
            and subject in text
            and re.search(r"(喜欢|偏好|最爱|爱吃|爱喝|口味|味|夹心)", text)
        )
    drink_match = _DRINK_ADDITIVE_QUERY_RE.search(query)
    if drink_match:
        subject = str(drink_match.group("subject") or "").strip()
        positive = _drink_additive_positive_answer(text, subject)
        negative = bool(re.search(r"(不喜欢|不爱|不要|不加|不放|讨厌)", text))
        needs_negative = bool(re.search(r"(不喜欢|不爱|不要|不加|不放)", query))
        return bool(subject and subject in text and positive and (negative or not needs_negative))
    if _EXPLICIT_DATE_QUERY_RE.search(query) and role != "correction":
        return bool(
            _EXPLICIT_DATE_ANSWER_RE.search(text)
            and (
                _memory_generic_query_answer_matches(text, query)
                or _safe_float(item.get("embedding_score", 0.0), 0.0) >= 0.52
                or _recall_focus_keyword_score(_recall_focus_text(query), text) >= 0.30
            )
        )
    if "小时候" in query and "老家" in query and re.search(r"(什么地方|哪里|哪儿)", query):
        return bool(
            _memory_row_canonical_key(item) in (
                "user:hometown:environment",
                "developer:hometown:environment",
                "user:hometown:preference",
                "developer:hometown:preference",
            )
            or ("小时候" in text and "喜欢" in text and _HOMETOWN_PLACE_ANSWER_RE.search(text))
        )
    if "谈恋爱" in query:
        return "谈恋爱" in text and bool(
            role == "correction"
            or _memory_row_canonical_key(item).endswith("relationship:intent")
            or re.search(r"(想|打算|意图|乌龙|澄清|纠正)", text)
        )
    if "努力的意义" in query:
        return "努力" in text and "意义" in text
    if _EXPRESSION_QUERY_RE.search(query):
        return bool(_EXPRESSION_ANSWER_RE.search(text))
    return _memory_generic_query_answer_matches(text, query)


def _memory_explicit_query_has_bounded_slot(query_text: str) -> bool:
    query = str(query_text or "")
    return bool(
        _EXPLICIT_INTERROGATIVE_RE.search(query)
        or _EXPLICIT_DATE_QUERY_RE.search(query)
        or _BIRTHDAY_DATE_ASK_RE.search(query)
        or ("小时候" in query and "老家" in query and re.search(r"(什么地方|哪里|哪儿)", query))
        or ("谈恋爱" in query and re.search(r"(有没有|是否|到底)", query))
        or "努力的意义" in query
        or _EXPRESSION_QUERY_RE.search(query)
    )


def _explicit_recall_answer_budget(query_text: str) -> int:
    query = str(query_text or "")
    if _DRINK_ADDITIVE_QUERY_RE.search(query) and re.search(r"(不喜欢|不爱|不要|不加|不放)", query):
        return 2
    # Composite recall questions need enough evidence to cover each requested
    # part. Keep single-slot questions at one row so broad recall does not
    # become an excuse to surface side topics.
    if re.search(r"(哪些|哪几个|分别|两个|两件|多件|各自)", query):
        return 4
    if re.search(r"(开过什么|说过什么|做过什么|提过什么|发生过什么)", query):
        return 3
    if re.search(r"(?:为什么|为何).*(?:怎么|如何)|(?:怎么|如何).*(?:为什么|为何)", query):
        return 3
    if query.count("、") >= 1 or ("和" in query and re.search(r"(什么|哪|多少|怎么|如何)", query)):
        return 3
    if (
        _memory_explicit_query_has_bounded_slot(query)
    ):
        return 1
    return 2


def _memory_row_effective_timestamp(row: dict | None = None) -> str:
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    payload = payload if isinstance(payload, dict) else {}
    return str(
        payload.get("ts", "")
        or item.get("updated_at", "")
        or item.get("created_at", "")
        or ""
    )


def _memory_row_effective_timestamp_key(row: dict | None = None) -> tuple[int, float, str]:
    raw = _memory_row_effective_timestamp(row)
    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (1, parsed.astimezone(timezone.utc).timestamp(), raw)
    except (TypeError, ValueError, OverflowError):
        return (0, 0.0, raw)


def _apply_memory_correction_dominance(rows: list[dict]) -> list[dict]:
    """Suppress only facts at or before the newest correction for each fact key."""
    corrections: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or _memory_row_evidence_role(row) != "correction":
            continue
        key = _memory_row_canonical_key(row)
        if not key:
            continue
        previous = corrections.get(key)
        row_ts = _memory_row_effective_timestamp_key(row)
        previous_ts = _memory_row_effective_timestamp_key(previous) if previous else (0, 0.0, "")
        if previous is None or row_ts >= previous_ts:
            corrections[key] = row
    if not corrections:
        return rows
    resolved: list[dict] = []
    for row in rows:
        key = _memory_row_canonical_key(row)
        winner = corrections.get(key)
        if winner is None or row is winner:
            resolved.append(row)
            continue
        winner_ts_raw = _memory_row_effective_timestamp(winner)
        row_ts_raw = _memory_row_effective_timestamp(row)
        winner_ts = _memory_row_effective_timestamp_key(winner)
        row_ts = _memory_row_effective_timestamp_key(row)
        if row_ts_raw and winner_ts_raw and row_ts > winner_ts and _memory_row_evidence_role(row) != "correction":
            resolved.append(row)
    return resolved


def _memory_should_run_expensive_reranker(cands: list[dict], analysis: dict | None = None) -> bool:
    """Use the cross-encoder only for ambiguous fact/recall candidate sets."""
    if len(cands) < 4:
        return False
    info = analysis if isinstance(analysis, dict) else {}
    query_text = str(info.get("query_text", "") or "")
    plausible = [
        dict(row) for row in cands if isinstance(row, dict)
        and _safe_float(row.get("heuristic_score", row.get("score", 0.0)), 0.0) >= 0.24
        and (
            _safe_float(row.get("relevance", 0.0), 0.0) >= 0.10
            or _safe_float(row.get("focus_relevance", 0.0), 0.0) >= 0.16
            or _safe_float(row.get("focus_keyword_score", 0.0), 0.0) >= 0.18
            or _safe_float(row.get("embedding_score", 0.0), 0.0) >= 0.52
            or _safe_float(row.get("bm25_normalized", 0.0), 0.0) >= 0.12
            or bool(row.get("_hybrid_structured_hit", False))
        )
    ]
    if len(plausible) < 2:
        return False
    answer_required = bool(info.get("explicit_recall_intent", False) or info.get("fact_like_query", False))
    if answer_required:
        plausible = [
            row for row in plausible
            if query_text and _memory_explicit_row_is_answer_bearing(row, query_text)
        ]
        if len(plausible) < 2:
            return False
    ranked = sorted(
        plausible,
        key=lambda row: _safe_float(row.get("heuristic_score", row.get("score", 0.0)), 0.0),
        reverse=True,
    )
    if len(ranked) < 2:
        return False
    top = ranked[0]
    top_is_answer_bearing = bool(
        query_text and _memory_explicit_row_is_answer_bearing(top, query_text)
    )
    if _memory_row_evidence_role(top) == "correction" and (
        not answer_required or top_is_answer_bearing
    ):
        return False
    if bool(top.get("_hybrid_structured_hit", False)) and (
        not answer_required or top_is_answer_bearing
    ):
        return False
    margin = _safe_float(top.get("heuristic_score", top.get("score", 0.0)), 0.0) - _safe_float(
        ranked[1].get("heuristic_score", ranked[1].get("score", 0.0)), 0.0
    )
    trusted_structured = bool(
        (_memory_row_evidence_role(top) == "correction" or top.get("_hybrid_structured_hit", False))
        and (not answer_required or top_is_answer_bearing)
    )
    strong_direct = bool(
        trusted_structured
        or _safe_float(top.get("direct_recall_match", 0.0), 0.0) >= 0.72
        or _safe_float(top.get("embedding_score", 0.0), 0.0) >= 0.78
        or _safe_float(top.get("focus_keyword_score", 0.0), 0.0) >= 0.75
    )
    if strong_direct and margin >= 0.08 and (not answer_required or top_is_answer_bearing):
        return False
    if not (info.get("explicit_recall_intent", False) or info.get("fact_like_query", False)):
        top_signal = max(
            _safe_float(top.get("relevance", 0.0), 0.0),
            _safe_float(top.get("embedding_score", 0.0), 0.0),
            _safe_float(top.get("bm25_normalized", 0.0), 0.0),
        )
        return bool(top_signal >= 0.55 and margin < 0.06)
    return margin < 0.12


def _memory_row_score(row: dict | None = None) -> float:
    item = row if isinstance(row, dict) else {}
    return float(_safe_float(item.get("score", item.get("collective_score", 0.0)), 0.0))


def _dedupe_memory_rows_by_item_id(rows: list[dict]) -> list[dict]:
    best_by_id: dict[str, dict] = {}
    no_id_rows: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item_id = str(item.get("item_id", "") or "").strip()
        if not item_id:
            no_id_rows.append(item)
            continue
        prev = best_by_id.get(item_id)
        if prev is None or _memory_row_score(item) > _memory_row_score(prev):
            best_by_id[item_id] = item
    merged = list(best_by_id.values()) + no_id_rows
    merged.sort(key=lambda x: _memory_row_score(x), reverse=True)
    return merged


def _memory_row_can_surface_nonexplicit(row: dict | None = None, analysis: dict | None = None) -> bool:
    item = row if isinstance(row, dict) else {}
    info = analysis if isinstance(analysis, dict) else {}
    query_text = str(info.get("query_text", "") or "")
    if (
        info.get("fact_like_query", False)
        and _memory_explicit_query_has_bounded_slot(query_text)
        and not _memory_explicit_row_is_answer_bearing(item, query_text)
    ):
        return False
    if _memory_row_is_working(item):
        return _memory_row_is_surfaceable_working(item)
    memory_type = str(item.get("memory_type", "") or "").strip()
    source_kind = str(item.get("source_kind", "") or "").strip()
    if memory_type == "episodic_event" and source_kind == "episodic_event":
        # 2026-05-17 bug fix: filter 误杀高分具体 dialogue
        # 实证 case: Q='你家旁边的南北町公园里也有藤花吧' 召回 r1 北町好逛 score=0.70
        # 被本 filter 一刀切掉、surface 退到 r3 角色C集训 score=0.57 不贴题
        # 原意是过滤 KAIROS dream cycle 生成的"consolidated event" 抽象总结
        # 修复：低 value generic（"会..." "愿意..." 等抽象 trait）才过滤、具体 content 放行
        _text = str(item.get("text", "") or "").strip()
        if _LOW_VALUE_MEMORY_GENERALIZATION_RE.match(_text):
            return False
        # 也过滤太短的（< 12 字）— 通常是抽象短标签
        if len(_text) < 12:
            return False
        # Bounded fact questions may have a lower aggregate score when the
        # answer is split across recent episodic rows; answer-bearing evidence
        # is allowed through at a lower floor and is still capped by the
        # adaptive surface selector below.
        _sc = _memory_row_score(item)
        if (
            query_text
            and _memory_explicit_query_has_bounded_slot(query_text)
            and _memory_explicit_row_is_answer_bearing(item, query_text)
            and _sc >= 0.38
        ):
            return True
        if _sc >= 0.50:
            return True
        return False
    if memory_type == "collective_event" or source_kind == "collective_event":
        if _memory_row_score(item) >= 0.45:
            return True
        return bool(
            _memory_row_score(item) >= 0.34
            and (
                _safe_float(item.get("relevance", 0.0), 0.0) >= 0.10
                or _safe_float(item.get("direct_recall_match", 0.0), 0.0) >= 0.26
                or _safe_float(item.get("focus_keyword_score", 0.0), 0.0) >= 0.25
            )
        )
    return True


def _direct_recall_match_score(row: dict | None = None) -> float:
    """Score how directly a candidate answers an explicit memory probe.

    The normal retrieval score intentionally contains recency/access priors.
    For questions like "do you remember what I said about X", those priors can
    make a frequently recalled but weakly related memory outrank the actual X.
    """
    from tsuzuri.memory_core_impl.recall_scoring import _clamp
    item = row if isinstance(row, dict) else {}
    relevance = _clamp(_safe_float(item.get("relevance", 0.0), 0.0))
    focus_relevance = _clamp(_safe_float(item.get("focus_relevance", 0.0), 0.0))
    focus_keyword = _clamp(_safe_float(item.get("focus_keyword_score", 0.0), 0.0))
    emb = _clamp(_safe_float(item.get("embedding_score", 0.0), 0.0))
    cue = _clamp(_safe_float(item.get("cue_match", 0.0), 0.0))
    fts = 1.0 if bool(item.get("_fts_hit", False)) else 0.0
    score = 0.38 * emb + 0.18 * relevance + 0.24 * focus_relevance + 0.14 * focus_keyword + 0.04 * cue + 0.04 * fts
    if bool(item.get("explicit_recall_archive", False)):
        score += 0.08
    focus = str(item.get("recall_focus", "") or "").strip()
    if focus and focus_keyword <= 0.01 and focus_relevance <= 0.01 and emb < 0.70 and not fts:
        score *= 0.78
    if bool(item.get("candidate_probe_text", False)) and focus_relevance < 0.12:
        score -= 0.18
    return _clamp(score)
