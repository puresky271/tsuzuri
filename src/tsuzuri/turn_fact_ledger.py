from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Callable, Iterable
from datetime import datetime
from tsuzuri.time_source import WORLD_TZ, now_world, world_tz_label
from tsuzuri import memory_host as memory_host
from tsuzuri.persona_registry import agent_aliases, agent_names, collective_label, fold_text_variants, generation as _persona_generation, live_set, setting, user_gender, user_name, user_pronoun


logger = logging.getLogger(__name__)

# Audit trail for silent card loss (fact-workspace audit 2026-08-16, finding
# 1.4): every window/sanitize truncation must be countable. Counters are
# process-global; hydrate, persist, and summary paths all feed them.
_EVICTION_COUNTS = {
    "sanitize_over_limit": 0,
    "sanitize_empty_card": 0,
    "sanitize_invalid_row": 0,
    "window_aged_out": 0,
    "window_reserve_overflow": 0,
    "window_invalid_row": 0,
}


def get_fact_card_eviction_counts() -> dict:
    return dict(_EVICTION_COUNTS)


def _record_evictions(source: str, **counts: int) -> None:
    nonzero = {key: int(value) for key, value in counts.items() if value}
    if not nonzero:
        return
    for key, value in nonzero.items():
        _EVICTION_COUNTS[key] = _EVICTION_COUNTS.get(key, 0) + value
    logger.info(
        "turn_fact_cards eviction (%s): %s; cumulative=%s",
        source,
        nonzero,
        {key: _EVICTION_COUNTS.get(key, 0) for key in nonzero},
    )


def _third_party_example() -> str:
    """Name used for the third-party example in extraction prompts (setting "third_party_example_name")."""
    return str(setting("third_party_example_name", "") or "") or "小林"


def _user_gender_rule() -> str:
    """Pronoun guard for the primary user, e.g. "尤其X是男生，不要写成“她”。" ("" when gender unknown)."""
    gender = user_gender()
    if gender not in ("男", "女"):
        return ""
    wrong = "她" if gender == "男" else "他"
    return f"尤其{_dev_name()}是{gender}生，不要写成“{wrong}”。"


# Optional physical-distance guard between the user and the agent (deployment
# setting "remote_relationship_note"; placeholders {user} / {char}).
def _remote_relationship_rule(char: str) -> str:
    note = str(setting("remote_relationship_note", "") or "")
    return note.format(user=_dev_name(), char=char) if note else ""


def _dev_name() -> str:
    """Display name of the primary user (resolved at use time from the persona registry)."""
    return user_name()


_AUDIENCE_SCOPE_LIMIT = 160
_TEXT_LIMITS = {
    "near_summary": 180,
    "fact": 180,
    "state_delta": 180,
}
_LIST_ITEM_LIMIT = 120
# The discontinuity guard and raw-exchange renderer share this boundary. A
# single constant prevents a 30-59 minute turn from saying both "do not carry"
# and "here is the full previous exchange" in the same prompt.
CONTINUITY_RAW_MAX_GAP_MIN = 30
RECENT_DIALOGUE_DEFAULT_MAX_CHARS = 6000
RECENT_DIALOGUE_DEFAULT_MAX_MESSAGES = 14
_TOPIC_FOLLOWUP_RE = re.compile(
    r"^(那|那么|那如果|那要是|所以|然后|接着|继续|关于这个|关于这件事|这个问题|"
    r"那个问题|你刚才|你刚刚|你说的|上次|之前|刚才|这样的话)",
    re.IGNORECASE,
)

# Controlled vocabulary for topic_labels field.
# 受控词表：LLM 只能从这里选，不得自造词。
_TOPIC_LABELS_VOCAB = frozenset({
    # 学校
    "school_exam", "homework", "class_school", "graduation", "student_life",
    # 日常生活
    "food_cooking", "shopping", "outing", "home_life", "sleep_rest",
    "family", "friendship", "romance", "part_time_job",
    # 情绪/心理
    "conflict", "apology", "anxiety", "loneliness", "joy", "worry",
    # 成长/规划
    "future_plan", "creative_project",
    # 爱好
    "hobby_games", "hobby_anime", "hobby_books", "exercise", "health",
    # 开发/技术
    "dev_project",
    # 其他
    "travel", "money", "news_event",
})
_BASE_TOPIC_LABELS_VOCAB = _TOPIC_LABELS_VOCAB
# Deployment-specific labels (setting "topic_labels_vocab") extend the built-in vocabulary.
_TOPIC_LABELS_VOCAB = live_set("topic_labels_vocab", sorted(_BASE_TOPIC_LABELS_VOCAB))


def _topic_labels_vocab_str() -> str:
    return "/".join(sorted(_TOPIC_LABELS_VOCAB))


_TOPIC_LABELS_VOCAB_STR = _topic_labels_vocab_str()  # import-time snapshot (kept for compatibility)


_CHARACTER_ALIAS_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] | None = None
_CHARACTER_ALIAS_REPLACEMENTS_GENERATION: int | None = None

# TurnFact 抽取只强制归一 4 个高频外号 + 全员英文简称；其余别名（带「小」前缀、「酱」后缀的昵称
# 等）保留原文，避免在对话语境里被改写后丢失称呼语义（2026-10-14 用户拍板）。
# Aliases that TurnFact extraction always folds onto the canonical name (setting
# "turn_fact_alias_whitelist"); other nicknames are kept verbatim.
_DEFAULT_TURN_FACT_ALIAS_WHITELIST: frozenset[str] = frozenset()
_TURN_FACT_ALIAS_WHITELIST = _DEFAULT_TURN_FACT_ALIAS_WHITELIST


def _turn_fact_alias_whitelist() -> frozenset[str]:
    return frozenset(
        str(x).strip().lower()
        for x in setting("turn_fact_alias_whitelist", _DEFAULT_TURN_FACT_ALIAS_WHITELIST) or ()
        if str(x or "").strip()
    )


def _person_alias_catalog() -> dict:
    """Agent/person alias catalog (host hook ``person_alias_catalog``; registry aliases otherwise)."""
    catalog = memory_host.call("person_alias_catalog", default=None)
    if catalog is None:
        catalog = agent_aliases()
    return dict(catalog or {})


def _person_alias_to_name() -> dict[str, str]:
    alias_to_name: dict[str, str] = {}
    for canonical, aliases in _person_alias_catalog().items():
        for alias in aliases or ():
            alias_to_name[str(alias).strip().lower()] = canonical
    return alias_to_name


def _character_alias_replacements() -> tuple[tuple[re.Pattern[str], str], ...]:
    """Compile TurnFact actor aliases from the shared character catalog."""
    global _CHARACTER_ALIAS_REPLACEMENTS, _CHARACTER_ALIAS_REPLACEMENTS_GENERATION
    if (
        _CHARACTER_ALIAS_REPLACEMENTS is not None
        and _CHARACTER_ALIAS_REPLACEMENTS_GENERATION == _persona_generation()
    ):
        return _CHARACTER_ALIAS_REPLACEMENTS
    rows: list[tuple[str, str]] = []
    try:
        whitelist = _turn_fact_alias_whitelist()
        for canonical, aliases in _person_alias_catalog().items():
            canon = str(canonical or "").strip()
            for alias in aliases:
                name = str(alias or "").strip()
                if (
                    canon
                    and name
                    and name.lower() != canon.lower()
                    and name.lower() in whitelist
                ):
                    rows.append((name, canon))
    except Exception:
        rows = []
    compiled: list[tuple[re.Pattern[str], str]] = []
    seen: set[tuple[str, str]] = set()
    for alias, canonical in sorted(rows, key=lambda row: len(row[0]), reverse=True):
        key = (alias.lower(), canonical)
        if key in seen:
            continue
        seen.add(key)
        escaped = re.escape(alias)
        if alias.isascii():
            escaped = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
        compiled.append((re.compile(escaped, re.IGNORECASE), canonical))
    _CHARACTER_ALIAS_REPLACEMENTS = tuple(compiled)
    _CHARACTER_ALIAS_REPLACEMENTS_GENERATION = _persona_generation()
    return _CHARACTER_ALIAS_REPLACEMENTS


def _canonicalize_character_aliases(value: Any) -> str:
    """Replace known nicknames with canonical actor names for TurnFact only."""
    text = str(value or "")
    for pattern, canonical in _character_alias_replacements():
        text = pattern.sub(canonical, text)
    return text


def _restore_character_alias_evidence_span(value: Any, source_text: Any) -> str:
    """Map model-visible canonical names back to the exact source evidence."""
    evidence = str(value or "").strip()
    source = str(source_text or "")
    if not evidence or evidence in source:
        return evidence
    pattern_text = re.escape(evidence)
    try:
        for canonical, aliases in sorted(_person_alias_catalog().items(), key=lambda row: len(row[0]), reverse=True):
            canon = str(canonical or "").strip()
            if not canon or canon not in evidence:
                continue
            choices = sorted(
                {str(alias or "").strip() for alias in aliases if str(alias or "").strip()} | {canon},
                key=len,
                reverse=True,
            )
            alias_pattern = "(?:" + "|".join(re.escape(choice) for choice in choices) + ")"
            pattern_text = pattern_text.replace(re.escape(canon), alias_pattern)
    except Exception:
        return evidence
    match = re.search(pattern_text, source, flags=re.IGNORECASE)
    return match.group(0) if match else evidence


def _restore_update_evidence_aliases(raw_updates: Any, source_entry: dict) -> Any:
    if not isinstance(raw_updates, list):
        return raw_updates
    source_text = str((source_entry or {}).get("content", "") or "")
    restored: list[Any] = []
    for row in raw_updates:
        if not isinstance(row, dict):
            restored.append(row)
            continue
        item = dict(row)
        item["evidence_span"] = _restore_character_alias_evidence_span(
            item.get("evidence_span", ""), source_text
        )
        restored.append(item)
    return restored


# 第三方指代消解（2026-07-31）
# ---------------------------------------------------------------------------
# 讨论第三方时用的第三人称代词（"她/他"）先行词不在本轮两句话里，抽取器无法
# 消解。更危险的是确定性后处理：它原本无条件把"劝她/提醒她"改写成"劝用户"，
# 把第三方事实反转成关于开发者的假事实。因此代词修复必须先确认全文没有提到
# 第三方——歧义代词只是难读，假事实会进 FACT_WORKSPACE 和 semantic memory。
#
# guard 是文本级而非从句级的：dev_phrase_rewrites 命中的从句（"希望有人劝她
# 休息"）本身往往不含人名，只有整段才看得到先行词（"用户觉得角色F太累了"）。
_KINSHIP_THIRD_PARTY_TERMS: frozenset[str] = frozenset({
    "姐姐", "妹妹", "哥哥", "弟弟", "姊姊", "お姉", "お兄",
    "妈妈", "母亲", "母親", "爸爸", "父亲", "父親", "父母", "双亲", "两亲",
    "爷爷", "奶奶", "外公", "外婆", "祖父", "祖母", "祖父母",
    "叔叔", "阿姨", "舅舅", "伯母", "婶婶", "表姐", "表妹", "堂姐", "堂妹",
    "家人", "亲戚", "监护人",
    "老师", "先生", "教练", "顾问", "班主任",
    "前辈", "先輩", "後輩", "后辈", "同学", "同級生", "室友",
    "店长", "店長", "经理", "経理", "老板", "社长", "社長", "manager",
})

# 「角色B」既是角色名也是普通名词。只在不被灯具类词素包裹时算人名。
# Agent names that are also common words: name -> regex that matches the name only when it
# is used as a person (setting "agent_name_guards"). Checked for agents other than the speaker.

_THIRD_PARTY_ROSTER_CACHE: dict[str, frozenset[str]] = {}


def _load_person_name_roster() -> frozenset[str]:
    """Person names that count as an explicit third-party referent.

    Host hook ``known_person_names`` supplies the deployment roster (canon NPCs,
    full/given names, nicknames); without it the persona registry's agents and
    aliases plus ``setting("known_person_names")`` are used. Lazy + cached per
    persona generation. Single-character names are skipped: they are too
    collision prone to justify suppressing a real user-pronoun repair.
    """
    generation = _persona_generation()
    cached = _THIRD_PARTY_ROSTER_CACHE.get("__all__")
    if cached is not None and _THIRD_PARTY_ROSTER_CACHE.get("__generation__") == generation:
        return cached
    names: set[str] = set()
    try:
        hooked = memory_host.call("known_person_names", default=None)
    except Exception:
        hooked = None
    if hooked is not None:
        names.update(str(n or "") for n in hooked)
    else:
        for agent, aliases in agent_aliases().items():
            names.add(agent)
            names.update(aliases)
        names.update(str(n or "") for n in (setting("known_person_names", []) or []))
    roster = frozenset(n.strip() for n in names if len(n.strip()) >= 2)
    _THIRD_PARTY_ROSTER_CACHE["__all__"] = roster
    _THIRD_PARTY_ROSTER_CACHE["__generation__"] = generation
    return roster


def _text_mentions_third_party(text: str, character: str) -> bool:
    """True when the text names or implies somebody other than the two speakers."""
    probe = str(text or "")
    if not probe:
        return False
    char = _text(character, 20)
    for term in _KINSHIP_THIRD_PARTY_TERMS:
        if term and term in probe:
            return True
    # 「角色B」需要灯具消歧；作为当前角色时不算第三方。
    for _guarded_name, _guard in (setting("agent_name_guards", {}) or {}).items():
        if char != _guarded_name and _guard and re.search(_guard, probe):
            return True
    try:
        _alias_to_name = _person_alias_to_name()
    except Exception:
        _alias_to_name = {}
    for name in _load_person_name_roster():
        if name in (_dev_name(), char):
            continue
        if char and (name in char or char in name):
            continue
        # 当前角色自己的昵称（不含本名的那些）也不算第三方。
        if char and str(_alias_to_name.get(name.lower(), "") or "") == char:
            continue
        if name in probe:
            return True
    return False


def turn_fact_enabled() -> bool:
    raw = str(os.environ.get("TURN_FACT_ENABLED", "1") or "").strip().lower()
    return raw not in ("0", "false", "off", "no", "n")


def _text(value: Any, limit: int = 200) -> str:
    s = re.sub(r"\s+", " ", str(value or "").strip())
    if len(s) > limit:
        return s[: max(0, limit - 1)] + "…"
    return s


def _entry_text(entry: dict | None, limit: int = 500) -> str:
    return _text((entry or {}).get("content", ""), limit)


def _entry_id(entry: dict | None) -> str:
    return _text((entry or {}).get("id", ""), 120)


def _entry_source_user_ids(entry: dict | None) -> list[str]:
    source = (entry or {}).get("source_user_ids", [])
    values = source if isinstance(source, list) else []
    out: list[str] = []
    for value in [*values, _entry_id(entry)]:
        clean = _text(value, 120)
        if clean and clean not in out:
            out.append(clean)
    return out[:20]


def _interlocutor_label(*entries: dict | None) -> str:
    for entry in entries:
        scope = str((entry or {}).get("audience_scope", "") or "").strip()
        if scope.startswith("guest:"):
            return "访客"
    return _dev_name()


def _card_user_ids(card: dict | None) -> list[str]:
    source = (card or {}).get("user_ids", [])
    values = source if isinstance(source, list) else []
    out: list[str] = []
    for value in [*values, str((card or {}).get("user_id", "") or "")]:
        clean = _text(value, 120)
        if clean and clean not in out:
            out.append(clean)
    return out[:20]


def _entry_ts_ms(entry: dict | None) -> int:
    raw = (entry or {}).get("timestamp", 0) or (entry or {}).get("ts", 0) or 0
    try:
        val = float(raw)
        if val <= 0:
            return 0
        if val < 1e11:
            val *= 1000.0
        return int(val)
    except Exception:
        return 0


def _entry_ts_sec(entry: dict | None) -> float:
    ms = _entry_ts_ms(entry)
    return (float(ms) / 1000.0) if ms > 0 else 0.0


def _format_world_minute(ts_ms: int) -> str:
    try:
        sec = int(ts_ms) / 1000.0
        if sec <= 0:
            return ""
        # Keep this module dependency-light; timestamps are UTC ms, render in WORLD_TZ.
        return datetime.fromtimestamp(sec, tz=WORLD_TZ).strftime("%Y-%m-%d %H:%M ") + world_tz_label()
    except Exception:
        return ""


def _format_world_source_period(start_ms: int, end_ms: int = 0) -> str:
    start = _format_world_minute(start_ms)
    if not start:
        return ""
    end = _format_world_minute(end_ms) if end_ms and end_ms != start_ms else ""
    if not end or end == start:
        return f"{start} 前后"
    if start[:10] == end[:10]:
        return f"{start[:10]} {start[11:16]}-{end[11:16]} {world_tz_label()}"
    return f"{start} 至 {end}"


def _card_source_period(card: dict, history_ts_by_id: dict[str, int] | None = None) -> str:
    history_ts_by_id = history_ts_by_id or {}
    user_timestamps = [history_ts_by_id.get(value, 0) for value in _card_user_ids(card)]
    user_timestamps = [value for value in user_timestamps if value]
    user_ts = min(user_timestamps, default=0)
    assistant_ts = history_ts_by_id.get(str(card.get("assistant_id", "") or ""), 0)
    if user_ts and assistant_ts:
        return _format_world_source_period(min(user_ts, assistant_ts), max(user_ts, assistant_ts))
    if assistant_ts or user_ts:
        return _format_world_source_period(assistant_ts or user_ts)
    try:
        created = int(card.get("created_at_ms", 0) or 0)
    except Exception:
        created = 0
    return _format_world_source_period(created)


def _source_prefix(period: str) -> str:
    return f"[来源：{period}] " if period else ""


def _coerce_list(value: Any, *, limit: int = 6) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    out: list[str] = []
    for item in items:
        tx = _text(item, _LIST_ITEM_LIMIT)
        if tx and tx not in out:
            out.append(tx)
        if len(out) >= limit:
            break
    return out


def _clamp_float(value: Any, default: float) -> float:
    try:
        val = float(value)
    except Exception:
        val = default
    return max(0.0, min(1.0, val))


def _turn_id(character: str, user_id: str, assistant_id: str, created_at_ms: int, fact: str) -> str:
    base = f"{character}|{user_id}|{assistant_id}|{created_at_ms}|{fact[:80]}"
    return "tf-" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


_ANTECEDENT_CONTEXT_MAX_TURNS = 2
_ANTECEDENT_CONTEXT_MAX_CHARS = 420


def build_antecedent_context(prior_cards: Any) -> str:
    """Render the previous 1-2 turns so third-person pronouns have an antecedent.

    Uses the already-compressed `near_summary` of prior cards rather than raw
    dialogue: it is bounded, already actor-normalized, and costs far fewer
    tokens than replaying full exchanges. Cards are the same audience-filtered
    set the caller already holds, so this adds no new privacy surface.
    """
    rows = [c for c in (prior_cards or []) if isinstance(c, dict)] if isinstance(prior_cards, list) else []
    rows = [c for c in rows if not c.get("archived")]
    if not rows:
        return ""
    lines: list[str] = []
    used = 0
    for card in reversed(rows[-_ANTECEDENT_CONTEXT_MAX_TURNS:]):
        text = _text(
            _canonicalize_character_aliases(
                card.get("near_summary") or card.get("fact") or card.get("state_delta")
            ),
            _ANTECEDENT_CONTEXT_MAX_CHARS,
        )
        if not text:
            continue
        if used + len(text) > _ANTECEDENT_CONTEXT_MAX_CHARS and lines:
            break
        lines.append(f"- {text}")
        used += len(text)
    return "\n".join(reversed(lines))


def _extraction_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    marker = "【中段省略，不得补全】"
    head = (limit - len(marker)) // 2
    return text[:head] + marker + text[-(limit - len(marker) - head):]


def _extraction_anchor_tokens(character: str, user_entry: dict) -> set[str]:
    from tsuzuri.turn_context_workspace import _topic_tokens

    query = _canonicalize_character_aliases(_extraction_text(user_entry.get("content"), 2400))
    query = re.sub(re.escape(_dev_name()) + "|" + re.escape(character) + r"|那个|这个|之前|刚才|说回|还是|已经|好像|最近", " ", query)
    return _topic_tokens(query)


def build_extraction_context(character: str, user_entry: dict, assistant_entry: dict,
                             *, prior_cards: Any = None, prior_history: Any = None) -> str:
    """Select bounded, source-labelled evidence before the extraction worker runs."""
    from tsuzuri.turn_context_workspace import _topic_tokens

    scope = str(assistant_entry.get("audience_scope") or user_entry.get("audience_scope") or "developer")
    current_ids = {_entry_id(user_entry), _entry_id(assistant_entry), *_entry_source_user_ids(user_entry)} - {""}
    cutoff = _entry_ts_ms(user_entry) or _entry_ts_ms(assistant_entry)

    def eligible(row):
        if not isinstance(row, dict) or row.get("archived"):
            return False
        row_scope = str(row.get("audience_scope") or "")
        if row_scope not in (("", "developer") if scope == "developer" else (scope,)):
            return False
        if row.get("character") and row["character"] != character:
            return False
        ids = {_entry_id(row), str(row.get("assistant_id") or ""), *_card_user_ids(row)} - {""}
        if ids & current_ids:
            return False
        ts = _entry_ts_ms(row) or _entry_ts_ms({"timestamp": row.get("created_at_ms")})
        return not (cutoff and ts and ts >= cutoff)

    cards = [dict(c) for c in (prior_cards or []) if eligible(c)] if isinstance(prior_cards, list) else []
    history = [dict(r) for r in (prior_history or []) if eligible(r) and r.get("role") in ("user", "assistant")] if isinstance(prior_history, list) else []
    cards = cards[-104:]
    history = history[-400:]
    # Use user anchors, not an assistant's potentially invented answer, for old evidence.
    tokens = _extraction_anchor_tokens(character, user_entry)

    def card_text(row):
        parts = [str(row.get(k) or "") for k in ("near_summary", "fact", "state_delta")]
        for key in ("corrections", "commitments", "open_threads"):
            parts.extend(_coerce_list(row.get(key), limit=3))
        return "；".join(dict.fromkeys(p for p in parts if p))

    def score(text):
        shared = tokens & _topic_tokens(_canonicalize_character_aliases(text))
        return sum(len(t) for t in shared) if len(shared) >= 2 or any(len(t) >= 3 for t in shared) else 0

    candidates = []
    for kind, rows, recent_count in (("摘要", cards, 2), ("原文", history, 4)):
        for index, row in enumerate(rows):
            text = card_text(row) if kind == "摘要" else str(row.get("content") or "")
            if not text.strip():
                continue
            recent = index >= len(rows) - recent_count
            relevance = score(text)
            if recent or relevance:
                candidates.append((recent, relevance, index, kind, row, text))
    # Reserve near context, then spend the remainder on a few explicit old anchors.
    recent = [c for c in candidates if c[0]]
    older = sorted((c for c in candidates if not c[0]), key=lambda c: (c[1], c[2]), reverse=True)[:4]
    selected = []
    used = 0
    for item in recent + older:
        _, _, index, kind, row, text = item
        speaker = (_dev_name() if scope == "developer" else "访客") if row.get("role") == "user" else character
        label = f"{kind}/{speaker}" if kind == "原文" else kind
        source = _entry_id(row) or str(row.get("assistant_id") or row.get("turn_id") or index)
        ts = _entry_ts_ms(row) or _entry_ts_ms({"timestamp": row.get("created_at_ms")})
        line = f"- [{label} {source[:80]} {_format_world_minute(ts) if ts else '时间未知'}] " + _canonicalize_character_aliases(_extraction_text(text, 300))
        if used + len(line) + 1 > 2400:
            continue
        used += len(line) + 1
        selected.append((ts, index, line))
    return "\n".join(line for _, _, line in sorted(selected))


def _room_fact_context(assistant_entry: dict) -> str:
    """Render only the immutable room snapshot visible before this reply."""
    room = assistant_entry.get("room_context")
    if not isinstance(room, dict) or not room.get("room_id"):
        return ""
    participants = [_text(p, 40) for p in room.get("participants", []) if isinstance(p, str)]
    lines = [
        "【多人谈话语境】",
        "参与成员：" + "、".join(participants),
        "介质：" + _text(room.get("channel"), 20),
        _dev_name() + ("在现场。" if room.get("user_present") else "通过文字远程参与，不在角色现场。"),
        "以下是本角色回复前可见的发言，仅供确定话题、说话人和回应对象，不是本角色说过的话：",
    ]
    used = 0
    rows = room.get("preceding_messages", [])
    for row in rows[-12:] if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("audience_scope", "developer") not in ("", "developer"):
            continue
        if _entry_id(row) == _entry_id(assistant_entry):
            break
        if row.get("role") not in ("user", "assistant"):
            continue
        speaker = _dev_name() if row.get("role") == "user" else _text(row.get("character"), 40)
        if not speaker or (speaker != _dev_name() and speaker not in participants):
            continue
        line = f"[{speaker} / {_entry_id(row)[:80]}] " + _extraction_text(row.get("content", ""), 300)
        if used + len(line) > 2400:
            continue
        lines.append(line)
        used += len(line)
    return _canonicalize_character_aliases("\n".join(lines))


def build_fact_extraction_prompt(
    character: str,
    user_entry: dict,
    assistant_entry: dict,
    *,
    prior_cards: Any = None,
    prior_history: Any = None,
) -> str:
    user_text = _canonicalize_character_aliases(
        _extraction_text((user_entry or {}).get("content", ""), 2400)
    )
    assistant_text = _canonicalize_character_aliases(_extraction_text((assistant_entry or {}).get("content", ""), 2400))
    char = _text(_canonicalize_character_aliases(character), 20) or "当前角色"
    peer = _interlocutor_label(user_entry, assistant_entry)
    proactive = bool((assistant_entry or {}).get("proactive")) and not user_text
    proactive_kind = _text((assistant_entry or {}).get("proactive_kind", ""), 40) or "proactive"
    if proactive:
        input_block = (
            f"{peer}本轮没有发送新消息。\n"
            f"触发类型：{proactive_kind}\n"
            f"{char}主动接续对话：{assistant_text}"
        )
        turn_rule = (
            f"- 这是{char}主动发出的消息，不得虚构{peer}本轮说过任何话；"
            f"fact 应记录{char}主动提起、确认或推进了什么。"
        )
    else:
        input_block = f"{peer}：{user_text}\n{char}：{assistant_text}"
        turn_rule = f"- fact 必须是一句可召回事实，优先描述“{peer}和{char}之间发生/确认/推进了什么”。"
    antecedent = build_extraction_context(character, user_entry, assistant_entry,
                                         prior_cards=prior_cards, prior_history=prior_history)
    if antecedent:
        input_block = (
            "【前文】（带来源的旧摘要与原文，仅用于消解指代、补全省略和识别纠正对象；"
            "不要把前文内容当成本轮发生的事写进事实卡）\n"
            f"{antecedent}\n\n"
            "【本轮】\n"
            f"{input_block}"
        )
    actor_rule = (
        f"- 对话双方实名写“{_dev_name()}”和“{char}”；这不是事实主体白名单，第三方仍须保留自己的姓名或称谓。{_user_gender_rule()}\n"
        + _remote_relationship_rule(char)
        if peer == _dev_name()
        else (
            f"- 对话双方写“访客”和“{char}”，第三方保留自己的姓名或称谓；禁止用代词猜测主体身份。\n"
            f"- 不知道访客的姓名、性别和所在地；不得把两端写成同处一地，也不得虚构递送或触碰。"
        )
    )
    # 讨论第三方（其他角色、NPC、家人、老师）时，"她/他"的先行词往往在更早的
    # 轮次里。此前 prompt 只给了"用户 / 角色"两个主体，模型面对第三方代词只能
    # 猜成这两人之一（造假事实）或原样留代词（下一轮读卡时无从消解）。这里显式
    # 引入第三方概念，并给出"（对象不明）"这个安全出口。
    third_party_rule = (
        f"- 本轮可能在讨论第三方（其他角色、NPC、家人、老师、店长等既不是{peer}也不是{char}的人）。\n"
        f"- 第三方必须实名化：用【前文】和本轮内容确定“她/他”指谁，写出那个人的名字，"
        f"例如“{peer}问起{_third_party_example()}的状态，{char}说{_third_party_example()}最近很忙”。\n"
        f"- 第三方事实不得被压成{peer}或{char}的事实；“劝她休息”若指第三方，"
        f"就不能写成“劝{peer}休息”。\n"
        f"- 如果结合【前文】仍无法确定代词指谁，写“（对象不明）”并降低 confidence；"
        f"禁止猜成{peer}或{char}。"
    )
    room_context = _room_fact_context(assistant_entry)
    if room_context:
        input_block = room_context + "\n\n" + input_block
        actor_rule = (
            f"- 这是{_dev_name()}与多个角色的谈话，不是私聊。每条事实明确保留发言人、动作主体和回应对象的姓名。\n"
            "- 其他成员的话属于其他成员；本角色引用、听到或回应某人的话，不代表本角色亲历或承诺。\n"
            f"- ‘你/她/你们/我们’按可见发言消解；对象不明就保留不确定性，不猜成{_dev_name()}或当前角色。\n"
            "- 地理共在只服从多人语境；LINE/mixed 的文字可见不等于身体共在，不能虚构触碰或递送。"
        )
        turn_rule = (
            f"- 只提取本轮{char}实际确认、纠正、回答或承诺的新进展；保留{char}回应谁及回应什么。"
            f"{_dev_name()}一句话可能分别对不同成员说，不能把整句话中的每项要求都分派给本角色。\n"
            "- 前序成员发言仅作本次抽取的依据，不复制成当前角色的新行为；沉默不是同意，建议不是承诺。"
        )
    developer_state_rule = (
        f"- developer_state_updates 只记录{_dev_name()}本人明确说出的、跨轮持续有效的当前状态；"
        "例如放暑假、已经开学、考试仍在进行/已经结束、旅行中、生病中。\n"
        f"- 每条更新必须给出 user 原文中逐字存在的 evidence_span；不能从角色回复推断{_dev_name()}状态。\n"
        "- 已经生效写 temporal_status=active；尚未发生的计划写 planned。"
        if peer == _dev_name() and not proactive
        else "- developer_state_updates 必须输出空数组。"
    )
    if proactive:
        character_action_rule = (
            "- 这是主动消息，character_action_updates 必须输出空数组；"
            "主动生成的台词不能反向改写权威日程或空间状态。"
        )
    elif assistant_text:
        character_action_rule = (
            f"- character_action_updates 只记录{char}在本轮回复中亲口确认的现实动作迁移；"
            "complete 表示已经做完，start 表示刚开始，cancel 表示明确取消。\n"
            f"- evidence_span 必须逐字来自{char}回复；event_text 只写动作核心，例如“买面包”。\n"
            "- start 只用于此刻明确正在发生或刚开始的动作；"
            "“明天/之后/到时候/届时/打算/准备/会做/承诺做”属于未来计划或承诺，禁止写成 start。"
        )
    else:
        character_action_rule = "- character_action_updates 必须输出空数组。"
    return f"""你是 {collective_label()} 对话系统的事实压缩器。把这一轮聊天提炼成一张事实卡，供下一轮对话理解上下文。

硬规则：
- 只输出 JSON，不要解释，不要 Markdown。
- 不要逐字照抄角色旧回复；要抽象成事实、状态变化、承诺或纠正。
- 不要编造本轮没有发生的内容。
- 【口语与证据】区分否定、反问、假设、转述、犹豫和确定决定；“也许/好像/再想想/没定”不得写成已确认事实或承诺。保留“谁说/谁猜测”的来源与不确定性。自我纠正以最后明确表述为准，纠正应写明具体对象和新值，不得把不同人的不同事项拼成一件事。
- 【上下文不足】前文不是完整历史，摘要不是逐字证据；优先使用本轮明确表述及旧原文，旧事实仅补足本轮确实引用的对象。多个先行词同样合理或只有“那个/她/行吧”时，不按最近的人名猜答案；记录询问、犹豫或“（对象不明）”，canonical_key 留空，不生成依赖猜测的状态更新或承诺。输入若标有中段省略，不得把两端直接拼成完整原话。
- 【主体完整性】fact 必须保留事件当事人，不能只写“双方讨论了某物”。例如前文“美咲丢了红色雨伞”，本轮“她找到了吗/还没”，应写“角色表示美咲尚未找到红色雨伞”，而非“角色还在找雨伞”。若只有“她那件事”且无先行词，写“{_dev_name()}仍在考虑某件事（对象不明）”，不得写成“角色提出的事”。口语中的“他/她/对方”也不能仅因出现在“{_dev_name()}说……”之后就归给{_dev_name()}。
- 【引用语义】用户若用引号、Markdown 引用（>）、“你说过：……”或转发卡片复述旧句，引用内容只是本轮正在讨论的证据，不等于{_dev_name()}当前重新确认、承诺或持有该立场；不要仅因被引用就把旧句写成新 fact/state_delta/commitment。只有用户在引用之外明确确认、否定、追问、改写或要求执行时，才记录相应的当前轮推进；若角色回应的是被引用内容，near_summary 可说明“{_dev_name()}引用并询问/纠正某说法”，但保留原说话主体和时间归属。
- 用户消息里的具体地点、对象、活动名、时间范围不要泛化丢失；例如”冲绳/向日葵/七月上旬到八月下旬”必须原样保留。
- 如果有纠正，以 corrections 记录。commitments 只保存角色明确将要履行的承诺；已经回答或确认的普通结论放 confirmed_outcomes，不能占据未完成约定的位置。重要情感约定和明确记忆请求不得因“不像任务”被丢弃。
- 提醒、建议、安慰和”别硬扛/可以求助”不等于角色承诺；只有角色明确说自己之后会做什么，才能写”角色承诺”。
- 【时间表达】保留事实发生时间与消息来源时间的区别。fact/state_delta/near_summary 必须保留原文明示的“今天/明天/昨天”等时间归属，不得猜绝对日期。event_time 仅逐字摘录时间短语；未知留空。未来计划不能写成已发生，过去经历不能写成此刻状态。
- 【偏好铁律】单次提到某食物/活动/地点/人物不等于稳定偏好；不得把一次性行为描述成”喜欢/总是/爱吃/经常”之类偏好断言，除非用户在本轮明确说”我喜欢”或已有多次记录。
- topic_labels 只能从以下受控词表选取（2-5 个）：{_topic_labels_vocab_str()}；不在词表里的词不要用，宁可选近义词也不自造新词。
{turn_rule}
{actor_rule}
{third_party_rule}
{developer_state_rule}
{character_action_rule}

输入：
{input_block}

输出 JSON schema：
{{
  "near_summary": "近端轻摘要，保留用户意图、角色立场和情绪变化，80字以内",
  "fact": "一条稳定事实，80字以内",
  "state_delta": "本轮让当前讨论状态发生了什么变化，80字以内",
  "open_threads": ["仍未解决或下一步要讨论的问题"],
  "commitments": ["角色明确将履行的约定，保留重要情感约定"],
  "confirmed_outcomes": ["已回答或确认的结论，不等于待履行承诺"],
  "corrections": ["用户或角色纠正过的旧事实"],
  "canonical_key": "可变 claim 的稳定身份，格式为 user:实体:属性 / character:实体:属性 / shared:实体:属性；不含当前值、日期或状态；无法可靠判断时为空字符串",
  "event_time": "原文中的事件时间短语，未知留空；不等于消息时间",
  "topic_labels": ["话题标签2-5个，只能从硬规则词表选取"],
  "developer_state_updates": [{{
    "predicate": "academic.phase|exam.status|health.current|travel.status|workload.current|availability.current|location.current",
    "value": "规范化后的简短状态值",
    "operation": "set|clear",
    "temporal_status": "active|planned",
    "valid_from": "YYYY-MM-DD 或空字符串",
    "valid_until": "YYYY-MM-DD 或空字符串",
    "evidence_span": "必须逐字来自{_dev_name()}本轮原文",
    "confidence": 0.0
  }}],
  "character_action_updates": [{{
    "transition": "start|complete|cancel",
    "event_text": "动作核心，例如买面包",
    "evidence_span": "必须逐字来自角色本轮回复",
    "confidence": 0.0
  }}],
  "salience": 0.0,
  "confidence": 0.0
}}
salience 六档参考（不确定时往下取）：
  0.0=纯日常寒暄；0.2=普通事件（问作业/吃什么）；0.4=有情感色彩但不特别（吐槽、小愿望）；
  0.6=值得日后回忆（重要活动、初次提及的困扰）；0.8=高情感强度（重大消息、争吵、告白）；
  1.0=极罕见（死亡/分离/人生转折）"""


def _extract_json(raw: str) -> dict:
    s = str(raw or "").strip()
    if not s:
        return {}
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1).strip())
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    obj = re.search(r"\{[\s\S]*\}", s)
    if obj:
        try:
            parsed = json.loads(obj.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass
    return {}


_MONTH_RANGE_RE = re.compile(
    r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)\s*(?:到|至|~|～|-|—|－|开到)\s*"
    r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)"
)


def _salient_user_detail_hints(user_text: str, existing_text: str) -> list[str]:
    text = re.sub(r"https?://\S+", "", str(user_text or ""))
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    existing = str(existing_text or "")
    details: list[str] = []

    def add(value: str) -> None:
        value = _text(value, 60)
        if not value or len(value) < 2:
            return
        range_match = re.match(
            r"([一-龥A-Za-z0-9]{2,12})从"
            r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)到"
            r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)",
            value,
        )
        if range_match:
            meaningful = [range_match.group(1), range_match.group(2), range_match.group(3)]
        else:
            meaningful = [
                re.sub(r"^(到|至|开到)", "", p)
                for p in re.findall(r"[一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?|[一-龥A-Za-z0-9]+", value)
                if len(re.sub(r"^(到|至|开到)", "", p)) >= 2
            ]
        if meaningful and all(p in existing for p in meaningful):
            return
        if value in existing:
            return
        if any(value in item or item in value for item in details):
            return
        if value not in details:
            details.append(value)

    for match in re.finditer(
        r"(?:看到|看见|刷到|查到|提到|说到|听说)?([一-龥A-Za-z0-9]{2,10})(?:那边|当地|附近)",
        text,
    ):
        location = re.sub(r"^(看到|看见|刷到|查到|提到|说到|听说)", "", match.group(1))
        if location not in {"你", "我", "对方", "用户", _dev_name(), "访客"}:
            add(location)
    for match in re.finditer(
        r"([一-龥A-Za-z0-9·]{2,12})从"
        r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)"
        r"(?:开始)?(?:开|开放)?到"
        r"([一二三四五六七八九十\d]{1,3}月(?:上旬|中旬|下旬)?)",
        text,
    ):
        name, start, end = match.groups()
        add(f"{name}从{start}到{end}")
    for match in _MONTH_RANGE_RE.finditer(text):
        add(f"{match.group(1)}到{match.group(2)}")
    return details[:4]


def _preserve_salient_user_details(raw: dict, user_entry: dict) -> dict:
    user_text = _entry_text(user_entry, 1200)
    if not user_text:
        return raw
    existing = " ".join(
        [
            str(raw.get("near_summary", "") or ""),
            str(raw.get("fact", "") or ""),
            str(raw.get("state_delta", "") or ""),
            " ".join(str(x or "") for x in (raw.get("open_threads", []) or []) if isinstance(raw.get("open_threads", []), list)),
        ]
    )
    details = _salient_user_detail_hints(user_text, existing)
    if not details:
        return raw
    data = dict(raw)
    hint = "具体线索：" + "、".join(details)
    fact = str(data.get("fact", "") or "").strip()
    if fact:
        data["fact"] = _text(f"{fact}（{hint}）", _TEXT_LIMITS["fact"])
    else:
        data["fact"] = hint
    near = str(data.get("near_summary", "") or "").strip()
    if near and not any(d in near for d in details):
        data["near_summary"] = _text(f"{near}（{hint}）", _TEXT_LIMITS["near_summary"])
    return data


def _normalize_actor_references(
    value: Any,
    character: str,
    *,
    limit: int,
    interlocutor_label: str = "",
    third_party_present: bool | None = None,
) -> str:
    text = _text(_canonicalize_character_aliases(value), limit)
    if not text:
        return ""
    char = _text(character, 20) or "当前角色"
    peer = _text(interlocutor_label, 20) or _dev_name()
    text = text.replace("用户", peer)

    if peer != _dev_name():
        return _text(text, limit)

    # 本轮提到第三方时，"她/他"的先行词可能是那个第三方，不是开发者。这里改写
    # 会造出「劝用户休息」这类假事实并流入 FACT_WORKSPACE / semantic memory，
    # 所以整段跳过代词修复：歧义代词只是难读，假事实会污染下游权威。
    # third_party_present 由调用方按整轮（原始对话 + 全部卡字段）预先判定，
    # 单条字段自身看不到先行词——它常在兄弟字段或用户原文里。
    if third_party_present if third_party_present is not None else _text_mentions_third_party(text, char):
        return _text(text, limit)

    text = text.replace("对方", peer)

    # Qwen occasionally writes the male developer as "她" in compressed facts.
    # Keep this conservative: only normalize common support/pressure phrases or
    # clauses that already name 用户 but not the current character.
    dev_phrase_rewrites = {
        "提醒她": f"提醒{_dev_name()}",
        "劝她": f"劝{_dev_name()}",
        "帮她": f"帮{_dev_name()}",
        "帮助她": f"帮助{_dev_name()}",
        "支持她": f"支持{_dev_name()}",
        "照顾她": f"照顾{_dev_name()}",
        "安抚她": f"安抚{_dev_name()}",
        "鼓励她": f"鼓励{_dev_name()}",
        "问她": f"问{_dev_name()}",
        "询问她": f"询问{_dev_name()}",
        "告诉她": f"告诉{_dev_name()}",
        "不会让她": f"不会让{_dev_name()}",
        "不让她": f"不让{_dev_name()}",
        "让她独自": f"让{_dev_name()}独自",
        "在她撑不住时": f"在{_dev_name()}撑不住时",
        "她并非": f"{_dev_name()}并非",
        "她不是": f"{_dev_name()}不是",
        "她不必": f"{_dev_name()}不必",
        "她不会": f"{_dev_name()}不会",
        "她独自": f"{_dev_name()}独自",
        "她自己": f"{_dev_name()}自己",
        "她的电路": f"{_dev_name()}的电路",
        "她的考试": f"{_dev_name()}的考试",
        "她的状态": f"{_dev_name()}的状态",
    }
    for old, new in dev_phrase_rewrites.items():
        text = text.replace(old, new)

    parts = re.split(r"([，。；、！？,.!?;])", text)
    normalized: list[str] = []
    for idx in range(0, len(parts), 2):
        clause = parts[idx]
        punct = parts[idx + 1] if idx + 1 < len(parts) else ""
        if _dev_name() in clause and char not in clause:
            clause = re.sub(r"(?<!其)她(?!们)", _dev_name(), clause)
            clause = re.sub(r"(?<!其)他(?!们|人|方|处)", _dev_name(), clause)
        normalized.append(clause + punct)
    return _text("".join(normalized), limit)


def _normalize_actor_list(
    value: Any,
    character: str,
    *,
    limit: int = 6,
    interlocutor_label: str = "",
    third_party_present: bool | None = None,
) -> list[str]:
    out: list[str] = []
    for item in _coerce_list(value, limit=limit):
        tx = _normalize_actor_references(
            item,
            character,
            limit=_LIST_ITEM_LIMIT,
            interlocutor_label=interlocutor_label,
            third_party_present=third_party_present,
        )
        if tx and tx not in out:
            out.append(tx)
    return out


def _rule_fallback_card(character: str, user_entry: dict, assistant_entry: dict) -> dict:
    char = _text(character, 20) or "当前角色"
    user_text = _extraction_text((user_entry or {}).get("content", ""), 140)
    assistant_text = _extraction_text((assistant_entry or {}).get("content", ""), 120)
    peer = _interlocutor_label(user_entry, assistant_entry)
    near = f"{peer}对{char}说：{user_text}"
    if assistant_text:
        near = f"{near}；{char}回应：{assistant_text}"
    fact = f"{peer}对{char}说：{user_text}" if user_text else f"{peer}和{char}完成了一轮对话。"
    return normalize_turn_fact_card(
        {
            "near_summary": near,
            "fact": fact,
            "state_delta": "",
            "open_threads": [],
            "commitments": [],
            "corrections": [],
            "topic_labels": [],
            "salience": 0.45,
            "confidence": 0.45,
        },
        character,
        user_entry,
        assistant_entry,
        model="rule_fallback",
        fallback_used=True,
    )


def build_rule_fallback_card(character: str, user_entry: dict, assistant_entry: dict) -> dict:
    """Build the deterministic card used when a fast follow-up cannot wait longer."""
    return _rule_fallback_card(character, user_entry, assistant_entry)


def _repair_remote_self_action_drift(
    raw: dict,
    character: str,
    assistant_entry: dict,
) -> dict:
    """Repair model-created remote caregiving when the source says self-use."""
    data = dict(raw) if isinstance(raw, dict) else {}
    assistant_text = _entry_text(assistant_entry, 1200)
    char = _text(character, 20) or "当前角色"
    peer = _interlocutor_label(assistant_entry)
    if not re.search(r"(?:我这边|自己)[^。！？\n]{0,48}(?:炒|煮|做|弄)", assistant_text):
        return data
    if re.search(rf"(?:给|为|替)(?:你|{re.escape(peer)})", assistant_text):
        return data
    action_match = re.search(r"(?:炒|煮|做|弄)(?:个|点|一份)?[一-龥]{1,8}", assistant_text)
    if not action_match:
        return data
    canonical = f"{char}自己准备{action_match.group(0)}"

    def is_drift_clause(value: Any) -> bool:
        text = str(value or "")
        return bool(
            char in text
            and peer in text
            and re.search(r"饭|面|汤|菜|餐|炒|煮|做饭|下厨", text)
            and re.search(r"给|为|替|照顾|做给", text)
        )

    def repair_text(value: Any) -> str:
        parts = re.split(r"([，。；！？,.!?;])", str(value or ""))
        repaired: list[str] = []
        for idx in range(0, len(parts), 2):
            clause = parts[idx]
            punctuation = parts[idx + 1] if idx + 1 < len(parts) else ""
            repaired.append((canonical if is_drift_clause(clause) else clause) + punctuation)
        return "".join(repaired)

    for key in ("near_summary", "fact", "state_delta"):
        data[key] = repair_text(data.get(key, ""))
    for key in ("open_threads", "corrections"):
        value = data.get(key, [])
        if isinstance(value, list):
            data[key] = [repair_text(item) for item in value]
    commitments = data.get("commitments", [])
    if isinstance(commitments, list):
        data["commitments"] = [
            repair_text(item) for item in commitments
            if not is_drift_clause(item)
        ]
    return data


def normalize_turn_fact_card(
    raw: dict,
    character: str,
    user_entry: dict,
    assistant_entry: dict,
    *,
    model: str = "",
    fallback_used: bool = False,
) -> dict:
    data = raw if isinstance(raw, dict) else {}
    data = _preserve_salient_user_details(data, user_entry)
    char = _text(character, 20) or "当前角色"
    room = assistant_entry.get("room_context", {})
    room = room if isinstance(room, dict) else {}
    if not room and not data.get("room_id"):
        data = _repair_remote_self_action_drift(data, char, assistant_entry)
    user_id = _entry_id(user_entry)
    user_ids = _entry_source_user_ids(user_entry)
    assistant_id = _entry_id(assistant_entry)
    # Audience scope is transport metadata, never model-authored content.  It
    # follows the source exchange so a shared session ledger cannot bridge a
    # developer turn into a guest prompt (or the reverse).
    audience_scope = _text(
        (assistant_entry or {}).get("audience_scope", "")
        or (user_entry or {}).get("audience_scope", "")
        or data.get("audience_scope", ""),
        _AUDIENCE_SCOPE_LIMIT,
    )
    peer = "访客" if audience_scope.startswith("guest:") else _dev_name()
    created_at_ms = (
        _entry_ts_ms(assistant_entry)
        or _entry_ts_ms(user_entry)
        or int(now_world().timestamp() * 1000)
    )
    # 第三方在场判定必须是整轮级的：先行词常在原始对话或兄弟字段里，单条字段
    # 自己看不到。逐字段各判会让「用户纠正了关于她的说法」在兄弟项已经点名
    # 角色F的情况下仍被改写成假事实。
    third_party_present = bool(room or data.get("room_id")) or bool(re.search(r"他(?!们|人|方|处)|她|对方", str(user_entry.get("content") or "") + " " + str(assistant_entry.get("content") or ""))) or _text_mentions_third_party(
        _canonicalize_character_aliases(
            " ".join(
                [
                    _entry_text(user_entry, 1200),
                    _entry_text(assistant_entry, 1200),
                    str(data.get("near_summary", "") or ""),
                    str(data.get("fact", "") or ""),
                    str(data.get("state_delta", "") or ""),
                    " ".join(str(x or "") for x in _coerce_list(data.get("open_threads", []), limit=6)),
                    " ".join(str(x or "") for x in _coerce_list(data.get("commitments", []), limit=6)),
                    " ".join(str(x or "") for x in _coerce_list(data.get("corrections", []), limit=6)),
                ]
            )
        ),
        char,
    )
    near_summary = _normalize_actor_references(
        data.get("near_summary", ""),
        char,
        limit=_TEXT_LIMITS["near_summary"],
        interlocutor_label=peer,
        third_party_present=third_party_present,
    )
    fact = _normalize_actor_references(
        data.get("fact", ""),
        char,
        limit=_TEXT_LIMITS["fact"],
        interlocutor_label=peer,
        third_party_present=third_party_present,
    )
    state_delta = _normalize_actor_references(
        data.get("state_delta", ""),
        char,
        limit=_TEXT_LIMITS["state_delta"],
        interlocutor_label=peer,
        third_party_present=third_party_present,
    )
    if not fact:
        fact = near_summary or state_delta
    if not near_summary:
        near_summary = fact
    turn_id = _text(data.get("turn_id", ""), 80) or _turn_id(char, user_id, assistant_id, created_at_ms, fact)
    try:
        from tsuzuri.interlocutor_state_ledger import (
            normalize_developer_state_updates,
            sanitize_persisted_state_updates,
        )

        if str((user_entry or {}).get("content", "") or "").strip():
            developer_state_updates = normalize_developer_state_updates(
                _restore_update_evidence_aliases(
                    data.get("developer_state_updates", []), user_entry
                ),
                user_entry,
            )
        else:
            developer_state_updates = sanitize_persisted_state_updates(
                data.get("developer_state_updates", [])
            )
    except Exception:
        developer_state_updates = []
    try:
        from tsuzuri.character_action_ledger import (
            normalize_character_action_updates,
            sanitize_persisted_action_updates,
        )

        if str((assistant_entry or {}).get("content", "") or "").strip():
            character_action_updates = normalize_character_action_updates(
                _restore_update_evidence_aliases(
                    data.get("character_action_updates", []), assistant_entry
                ),
                assistant_entry,
            )
        else:
            character_action_updates = sanitize_persisted_action_updates(
                data.get("character_action_updates", [])
            )
    except Exception:
        character_action_updates = []
    _raw_labels = data.get("topic_labels", [])
    topic_labels: list[str] = []
    if isinstance(_raw_labels, list):
        for _lbl in _raw_labels:
            _s = str(_lbl or "").strip().lower().replace(" ", "_")
            if _s and len(_s) <= 40 and re.match(r"^[a-z][a-z0-9_]{0,39}$", _s) and _s not in topic_labels:
                topic_labels.append(_s)
            if len(topic_labels) >= 6:
                break
    raw_canonical_key = " ".join(str(data.get("canonical_key", "") or "").split()).strip()
    canonical_key = ""
    if (
        len(raw_canonical_key) <= 120
        and re.match(r"^(?:user|character|shared|other|unknown):[^\s:]+:[^\s:]+$", raw_canonical_key)
    ):
        canonical_key = raw_canonical_key
    # Current-turn adjudication was made before the reply, from user evidence.
    # Never let the character's generated answer invent a lifecycle closure.
    validity = (user_entry or {}).get('_fact_validity', {})
    if not isinstance(validity, dict) or validity.get('user_id') != user_id:
        validity = {}
    event_time = _text(data.get('event_time', ''), 80)
    time_sources = str((user_entry or {}).get('content', '') or '') + '\n' + str((assistant_entry or {}).get('content', '') or '')
    if time_sources.strip() and event_time not in time_sources:
        event_time = ''
    return {
        "turn_id": turn_id,
        "user_id": user_id,
        "user_ids": user_ids,
        "assistant_id": assistant_id,
        "audience_scope": audience_scope,
        "character": char,
        **({"room_id": _text(room.get("room_id") or data.get("room_id"), 120),
            "room_channel": _text(room.get("channel") or data.get("room_channel"), 20)}
           if room.get("room_id") or data.get("room_id") else {}),
        "canonical_key": canonical_key,
        "event_time": event_time,
        "_background_facts": _coerce_list(data.get('_background_facts', []), limit=8)
        if not str((user_entry or {}).get('content', '') or '').strip() else [],
        "fact_validity_actions": list(validity.get('actions', (
            data.get('fact_validity_actions', []) if not str((user_entry or {}).get('content', '') or '').strip() else []
        )) or []) if peer == _dev_name() else [],
        "created_at_ms": created_at_ms,
        "near_summary": near_summary,
        "fact": fact,
        "state_delta": state_delta,
        "open_threads": _normalize_actor_list(
            data.get("open_threads", []), char, limit=6, interlocutor_label=peer,
            third_party_present=third_party_present,
        ),
        "commitments": _normalize_actor_list(
            data.get("commitments", []), char, limit=6, interlocutor_label=peer,
            third_party_present=third_party_present,
        ),
        "confirmed_outcomes": _normalize_actor_list(
            data.get('confirmed_outcomes', []), char, limit=6, interlocutor_label=peer,
            third_party_present=third_party_present,
        ),
        "corrections": _normalize_actor_list(
            data.get("corrections", []), char, limit=6, interlocutor_label=peer,
            third_party_present=third_party_present,
        ),
        # Continuing developer state is persisted in a separate authority
        # ledger. The turn card only carries evidence-checked update proposals.
        "developer_state_updates": (
            sanitize_persisted_state_updates(validity.get('developer_state_updates', []))
            or developer_state_updates
        ) if peer == _dev_name() else [],
        "character_action_updates": character_action_updates if peer == _dev_name() else [],
        "topic_labels": topic_labels,
        "salience": _clamp_float(data.get("salience", 0.5), 0.5),
        "confidence": _clamp_float(data.get("confidence", 0.55), 0.55),
        "model": _text(model or data.get("model", ""), 80),
        "fallback_used": bool(fallback_used or data.get("fallback_used", False)),
        "archived": bool(data.get("archived", False)),
    }


def extract_turn_fact_card(
    character: str,
    user_entry: dict,
    assistant_entry: dict,
    *,
    llm_callers: Iterable[tuple[str, Callable[[str], str]]] | None = None,
    prior_cards: Any = None,
    prior_history: Any = None,
) -> dict:
    prompt = build_fact_extraction_prompt(
        character,
        user_entry,
        assistant_entry,
        prior_cards=prior_cards,
        prior_history=prior_history,
    )
    source = _canonicalize_character_aliases(
        str(user_entry.get("content") or "") + " " + str(assistant_entry.get("content") or "")
    )
    # With neither a named referent nor historical evidence, even fluent JSON
    # can invent the subject. Preserve attributed source text, not a guessed claim.
    # 主动轮次（auto_greet 等，无用户发言）不存在可误归到开发者头上的代词，
    # guard 只会短路掉为它们准备的 _background_facts 通道，因此只守卫真实对话轮。
    if (str(user_entry.get("content") or "").strip()
            and re.search(r"(?<!其)他(?!们|人|方|处)|她|对方", source)
            and not _text_mentions_third_party(source, character)):
        from tsuzuri.turn_context_workspace import _topic_tokens

        context = build_extraction_context(character, user_entry, assistant_entry,
                                           prior_cards=prior_cards, prior_history=prior_history)
        context += "\n" + _room_fact_context(assistant_entry)
        shared = _extraction_anchor_tokens(character, user_entry) & _topic_tokens(context)
        anchored = len(shared) >= 2 or any(len(token) >= 3 for token in shared)
        if not _text_mentions_third_party(context, character) and not anchored:
            return _rule_fallback_card(character, user_entry, assistant_entry)
    callers = list(llm_callers or [])
    for idx, (model, caller) in enumerate(callers):
        if not callable(caller):
            continue
        try:
            parsed = _extract_json(caller(prompt))
        except Exception:
            parsed = {}
        if not parsed:
            continue
        card = normalize_turn_fact_card(
            parsed,
            character,
            user_entry,
            assistant_entry,
            model=model,
            fallback_used=idx > 0,
        )
        if card.get("fact") or card.get("near_summary") or card.get("state_delta"):
            return card
    return _rule_fallback_card(character, user_entry, assistant_entry)


def sanitize_fact_cards(cards: Any, *, limit: int = 80) -> list[dict]:
    if not isinstance(cards, list):
        return []
    out: list[dict] = []
    dropped_invalid = 0
    dropped_empty = 0
    dropped_over_limit = 0
    for item in cards:
        if not isinstance(item, dict):
            dropped_invalid += 1
            continue
        user_stub = {
            "id": item.get("user_id", ""),
            "source_user_ids": item.get("user_ids", []),
            "timestamp": item.get("created_at_ms", 0),
        }
        assistant_stub = {"id": item.get("assistant_id", ""), "timestamp": item.get("created_at_ms", 0)}
        user_stub["audience_scope"] = item.get("audience_scope", "")
        assistant_stub["audience_scope"] = item.get("audience_scope", "")
        normalized = normalize_turn_fact_card(
            item,
            str(item.get("character", "") or ""),
            user_stub,
            assistant_stub,
            model=str(item.get("model", "") or ""),
            fallback_used=bool(item.get("fallback_used", False)),
        )
        normalized["archived"] = bool(item.get("archived", False))
        if not any(normalized.get(key) for key in ('fact', 'near_summary', 'state_delta',
                                                   'commitments', 'open_threads', 'corrections',
                                                   '_background_facts', 'fact_validity_actions')):
            dropped_empty += 1
            continue
        if len(out) >= limit:
            dropped_over_limit += 1
            continue
        out.append(normalized)
    _record_evictions(
        "sanitize_fact_cards",
        sanitize_invalid_row=dropped_invalid,
        sanitize_empty_card=dropped_empty,
        sanitize_over_limit=dropped_over_limit,
    )
    return out


def lifecycle_card_window(cards: Any, *, recent: int = 80, reserve: int = 24) -> list[dict]:
    """Recent sliding window plus a reserve of older lifecycle-bearing cards.

    The reserve keeps cards carrying corrections/commitments/open_threads
    visible to ranking after they age out of the recent window; otherwise the
    81st card could silently erase a still-open promise. Order is preserved
    (reserve rows are older than recent rows). Every eviction is counted (see
    _record_evictions) so a shrinking window is observable instead of silent.
    """
    raw = cards if isinstance(cards, list) else []
    rows = [row for row in raw if isinstance(row, dict)]
    recent_rows = rows[-recent:]
    older_rows = rows[:-recent]
    reserve_candidates = [
        row
        for row in older_rows
        if any(row.get(key) for key in ("corrections", "commitments", "open_threads"))
    ]
    reserve_rows = reserve_candidates[-reserve:]
    _record_evictions(
        "lifecycle_card_window",
        window_invalid_row=len(raw) - len(rows),
        window_aged_out=len(older_rows) - len(reserve_candidates),
        window_reserve_overflow=len(reserve_candidates) - len(reserve_rows),
    )
    return [*reserve_rows, *recent_rows]


def remove_fact_cards_for_message_ids(cards: Any, message_ids: Iterable[str]) -> list[dict]:
    ids = {str(x or "").strip() for x in message_ids if str(x or "").strip()}
    if not isinstance(cards, list):
        return []
    if not ids:
        return [dict(x) for x in cards if isinstance(x, dict)]
    out = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        card = dict(item)
        if ids.intersection(_card_user_ids(card)):
            continue
        if str(card.get("assistant_id", "") or "") in ids:
            continue
        out.append(card)
    return lifecycle_card_window(out, recent=80, reserve=24)


def build_fact_state_summary(cards: Any, *, limit: int = 900) -> str:
    # The runtime list is already windowed to 80 recent + 24 lifecycle reserve
    # (newest last). Sanitizing with the default cap of 80 would keep the
    # reserve plus the OLDEST recent rows and silently drop the newest ones
    # from this summary, so use the full window size here.
    clean = [c for c in sanitize_fact_cards(cards, limit=104) if not c.get("archived")]
    corrections: list[tuple[str, str]] = []
    commitments: list[tuple[str, str]] = []
    states: list[tuple[str, str]] = []
    for card in clean[-20:]:
        source = _source_prefix(_card_source_period(card))
        corrections.extend((source, item) for item in _coerce_list(card.get("corrections", []), limit=4))
        commitments.extend((source, item) for item in _coerce_list(card.get("commitments", []), limit=4))
        state = _text(card.get("state_delta", ""), 120)
        if state:
            states.append((source, state))
    lines: list[str] = []
    for label, items in (("纠正", corrections[-5:]), ("已确认/承诺", commitments[-5:]), ("状态推进", states[-6:])):
        for source, item in items:
            line = f"- {source}{label}：{item}"
            if line not in lines:
                lines.append(line)
    text = "\n".join(lines)
    return text[:limit]


def _ordered_history(history: Any) -> list[dict]:
    rows = [dict(x) for x in history if isinstance(x, dict)] if isinstance(history, list) else []

    def key(item: dict) -> float:
        ts = _entry_ts_ms(item)
        return float(ts) if ts > 0 else float(rows.index(item))

    try:
        return sorted(rows, key=key)
    except Exception:
        return rows


def _last_current_user(history: list[dict]) -> dict:
    for item in reversed(history):
        if str(item.get("role", "") or "").lower() == "user" and _entry_text(item, 1000):
            return item
    return {}


def _last_completed_exchange(
    history: list[dict],
    current_user_id: str,
    *,
    skip_proactive: bool = True,
) -> tuple[dict, dict]:
    assistant: dict = {}
    assistant_index = -1
    for idx in range(len(history) - 1, -1, -1):
        item = history[idx]
        if _entry_id(item) == current_user_id:
            continue
        if str(item.get("role", "") or "").lower() == "assistant" and _entry_text(item, 1000):
            # Idle proactive rows (auto_greet / idle_followup) are not a real
            # exchange: gap detection and mid-group priming must anchor on the
            # last real reply instead.
            if skip_proactive and bool(item.get("proactive")):
                continue
            assistant = item
            assistant_index = idx
            break
    if not assistant:
        return {}, {}
    for idx in range(assistant_index - 1, -1, -1):
        item = history[idx]
        if str(item.get("role", "") or "").lower() == "user" and _entry_text(item, 1000):
            return item, assistant
    return {}, assistant


def _recent_dialogue_limits() -> tuple[int, int]:
    try:
        max_chars = int(os.environ.get("TURN_RECENT_DIALOGUE_MAX_CHARS", str(RECENT_DIALOGUE_DEFAULT_MAX_CHARS)) or 0)
    except (TypeError, ValueError):
        max_chars = RECENT_DIALOGUE_DEFAULT_MAX_CHARS
    try:
        max_messages = int(os.environ.get("TURN_RECENT_DIALOGUE_MAX_MESSAGES", str(RECENT_DIALOGUE_DEFAULT_MAX_MESSAGES)) or 0)
    except (TypeError, ValueError):
        max_messages = RECENT_DIALOGUE_DEFAULT_MAX_MESSAGES
    return max(1200, min(12000, max_chars)), max(3, min(30, max_messages))


def _recent_raw_entries(
    ordered: list[dict],
    current_user: dict,
    *,
    max_chars: int | None = None,
    max_messages: int | None = None,
) -> list[dict]:
    """Return a bounded verbatim tail from the current continuous time segment."""
    default_chars, default_messages = _recent_dialogue_limits()
    char_budget = max(1200, int(max_chars or default_chars))
    message_cap = max(3, int(max_messages or default_messages))
    current_id = _entry_id(current_user)
    current_index = next(
        (
            idx for idx in range(len(ordered) - 1, -1, -1)
            if ordered[idx] is current_user or (current_id and _entry_id(ordered[idx]) == current_id)
        ),
        len(ordered) - 1,
    )
    dialog = [
        row for row in ordered[:current_index]
        if str(row.get("role", "") or "").lower() in ("user", "assistant")
        and _entry_text(row, 20)
    ]
    if not dialog:
        return []
    selected: list[dict] = []
    used_chars = 0
    newer = current_user
    for row in reversed(dialog):
        newer_ts = _entry_ts_ms(newer)
        row_ts = _entry_ts_ms(row)
        if newer_ts > 0 and row_ts > 0:
            gap_min = max(0.0, (newer_ts - row_ts) / 60000.0)
            if gap_min >= CONTINUITY_RAW_MAX_GAP_MIN:
                break
        vision_meta = row.get("vision_meta", {}) if isinstance(row.get("vision_meta", {}), dict) else {}
        # A prior image remains part of the recent conversation, but its full
        # vision prose is evidence, not dialogue. Carry only the compact visual
        # proposition unless the image is the current user message.
        raw_text = str(vision_meta.get("compact", "") or row.get("content", "") or "").strip()
        if not raw_text:
            newer = row
            continue
        remaining = char_budget - used_chars
        if selected and remaining < min(240, len(raw_text)):
            break
        if vision_meta.get("compact"):
            rendered = _text(raw_text, min(1800, max(240, remaining)))
        else:
            rendered = _entry_text(row, min(1800, max(240, remaining)))
        if not rendered:
            newer = row
            continue
        copied = dict(row)
        copied["content"] = rendered
        selected.append(copied)
        used_chars += len(rendered)
        newer = row
        if len(selected) >= message_cap or used_chars >= char_budget:
            break
    return list(reversed(selected))


def recent_dialogue_raw_ids(history: list, *, group_mode: bool = False) -> set[str]:
    ordered = _ordered_history(history)
    if group_mode:
        return _render_group_dialogue_window(ordered).get("raw_ids", set())
    current_user = _last_current_user(ordered)
    if not current_user:
        return set()
    rows = _recent_raw_entries(ordered, current_user)
    return {
        value for row in [*rows, current_user]
        for value in [_entry_id(row)]
        if value
    }


def _render_group_dialogue_window(ordered: list[dict], current_character: str = "") -> dict:
    """multi-room 群聊窗口（2026-08-08）：整段按时间序渲染对话尾部。

    与 1v1 窗口的差异：turn 内前序 speaker 的回复在触发用户消息之后，仍是当前
    对话尾部——1v1 窗口以「最后一条用户消息」为锚、只取其前文，会丢掉这些回复
    （dump 审查实测：第 2 个 speaker 看不到第 1 个的发言）。这里以最新一条消息
    为锚整段保留：用户行渲染 [用户]（对全场、无定向），其他成员发言转 user role
    + [名字]：前缀（归属），当前角色自己的发言保持 assistant 无前缀。
    packet 由 assemble 拼到最末 user-role 行，语义 = 「接在最末发言后继续」。
    """
    anchor = ordered[-1] if ordered else {}
    if not anchor or not _entry_text(anchor, 1000):
        return {"guard": "", "messages": [], "raw_ids": set(), "raw_chars": 0, "raw_messages": 0}
    recent = _recent_raw_entries(ordered, anchor)
    rows = [*recent, anchor]
    messages: list[dict] = []
    for row in rows:
        role = str(row.get("role", "") or "").lower()
        if role == "user":
            content = _render_user_content(
                _entry_text(row, 1800),
                "",
                interlocutor_label=_interlocutor_label(row),
            )
        else:
            _speaker = _text(row.get("character"), 40)
            if _speaker and current_character and _speaker != str(current_character):
                role = "user"
                content = f"[{_speaker}]：{_entry_text(row, 1800)}"
            else:
                content = _entry_text(row, 1800)
        if content:
            messages.append({"role": role, "content": content})
    raw_ids = {
        value for row in rows
        for value in [_entry_id(row)]
        if value
    }
    guard = ""
    if recent:
        guard = (
            '<recent-dialogue-window fidelity="verbatim">\n'
            "以下消息是当前连续时间段内的近期原文，按时间顺序保留。"
            "用它承接角色实际说法、语气和已经推进的动作；不要复述旧句，也不要把已完成动作退回未发生。\n"
            "</recent-dialogue-window>"
        )
    return {
        "guard": guard,
        "messages": messages,
        "raw_ids": raw_ids,
        "raw_chars": sum(len(str(row.get("content", "") or "")) for row in rows),
        "raw_messages": len(rows),
    }


def render_recent_dialogue_window(
    history: list,
    *,
    current_character: str = "",
    group_mode: bool = False,
) -> dict:
    ordered = _ordered_history(history)
    if group_mode:
        return _render_group_dialogue_window(ordered, current_character)
    current_user = _last_current_user(ordered)
    if not current_user:
        return {"guard": "", "messages": [], "raw_ids": set(), "raw_chars": 0, "raw_messages": 0}
    recent = _recent_raw_entries(ordered, current_user)
    messages: list[dict] = []
    for row in recent:
        role = str(row.get("role", "") or "").lower()
        if role == "user":
            content = _render_user_content(
                _entry_text(row, 1800),
                current_character,
                interlocutor_label=_interlocutor_label(row),
            )
        else:
            content = _entry_text(row, 1800)
        if content:
            messages.append({"role": role, "content": content})
    current_text = _entry_text(current_user, 2000)
    messages.append({
        "role": "user",
        "content": _render_user_content(
            current_text,
            str(current_user.get("target", "") or ""),
            interlocutor_label=_interlocutor_label(current_user),
        ),
    })
    raw_ids = {
        value for row in [*recent, current_user]
        for value in [_entry_id(row)]
        if value
    }
    guard = ""
    if recent:
        guard = (
            '<recent-dialogue-window fidelity="verbatim">\n'
            "以下消息是当前连续时间段内的近期原文，按时间顺序保留。"
            "用它承接角色实际说法、语气和已经推进的动作；不要复述旧句，也不要把已完成动作退回未发生。\n"
            "</recent-dialogue-window>"
        )
    return {
        "guard": guard,
        "messages": messages,
        "raw_ids": raw_ids,
        "raw_chars": sum(len(str(row.get("content", "") or "")) for row in recent),
        "raw_messages": len(recent),
    }


def _gap_phrase(gap_min: int) -> str:
    if gap_min < 60:
        return "约半小时"
    if gap_min < 120:
        return "约一小时"
    if gap_min < 240:
        return "好几个小时"
    if gap_min < 1440:
        return f"{gap_min // 60} 个小时左右"
    return f"差不多 {max(1, gap_min // 1440)} 天"


def _derive_current_gap_hint(current_user: dict, prev_assistant: dict) -> tuple[str, int]:
    prev_ts = _entry_ts_sec(prev_assistant)
    curr_ts = _entry_ts_sec(current_user)
    if prev_ts <= 0 or curr_ts <= 0 or curr_ts <= prev_ts:
        return "", 0
    gap_min = int(max(0.0, (curr_ts - prev_ts) / 60.0))
    if gap_min < CONTINUITY_RAW_MAX_GAP_MIN:
        return "", gap_min
    phrase = _gap_phrase(gap_min)
    return (
        f"当前用户消息和上一轮角色回复之间隔了{phrase}，不要把当前轮当成紧跟上一轮的承接；"
        "按当前时间、当前话题和最新用户意图回应，不要延续上一时间段的瞬时动作或场景。",
        gap_min,
    )


def _clean_hint(text: Any, limit: int = 260) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    s = s.strip("()（） \t\r\n")
    return _text(s, limit)


def _looks_like_state_hint(text: str) -> bool:
    s = str(text or "")
    if not s.strip():
        return False
    markers = (
        "系统提示",
        "当前轮次优先",
        "用户刚切换",
        "切换了一个新话题",
        "不要把下面的对话当成紧跟上一条",
        "不是连着说的",
        "不要承接上一个时间段",
        "不要继续上一轮",
        "延迟回复",
        "未回",
        "不方便回",
        "状态提示",
    )
    return any(m in s for m in markers)


def _inline_state_hints(history: list[dict], *, limit: int = 6) -> list[str]:
    # These markers guard one segment only. Walk backward from the current user
    # and stop at the previous assistant so old gap/topic notices do not become
    # permanent state merely because raw history retains them for audit.
    hints_reversed: list[str] = []
    current_user_seen = False
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "") or "").lower()
        if not current_user_seen:
            if role == "user":
                current_user_seen = True
            continue
        if role == "assistant":
            break
        if role != "system":
            continue
        content = str(item.get("content", "") or "")
        if not _looks_like_state_hint(content):
            continue
        hint = _clean_hint(content)
        if hint and hint not in hints_reversed:
            hints_reversed.append(hint)
        if len(hints_reversed) >= limit:
            break
    return list(reversed(hints_reversed))


def _topic_key(text: str) -> set[str]:
    s = re.sub(r"\s+", "", str(text or "").lower())
    chars = [c for c in s if "\u4e00" <= c <= "\u9fff" or c.isalnum()]
    if len(chars) <= 1:
        return set(chars)
    grams = {("".join(chars[i:i + 2])) for i in range(len(chars) - 1)}
    grams.update(chars)
    return grams


def _topic_similarity(a: str, b: str) -> float:
    ka = _topic_key(a)
    kb = _topic_key(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / max(1, len(ka | kb))


def _continuity_overlap(current: str, previous: list[str]) -> bool:
    current_key = _topic_key(current)
    if not current_key:
        return False
    for prev in previous[:2]:
        prev_key = _topic_key(prev)
        if not prev_key:
            continue
        shared = current_key & prev_key
        if len(shared) >= 2:
            return True
    return False


def _derive_topic_shift_hint(history: list[dict], current_user: dict) -> str:
    current = _entry_text(current_user, 120)
    if len(current) < 8 or _TOPIC_FOLLOWUP_RE.search(current):
        return ""
    previous: list[str] = []
    previous_near: list[str] = []
    current_id = _entry_id(current_user)
    for item in reversed(history):
        role = str(item.get("role", "") or "").lower()
        if role not in ("user", "assistant"):
            continue
        if role == "user" and _entry_id(item) == current_id:
            continue
        text = _entry_text(item, 80)
        if not text:
            continue
        previous_near.append(text)
        if role == "user":
            previous.append(text)
        if len(previous) >= 3 and len(previous_near) >= 4:
            break
    if not previous:
        return ""
    if _continuity_overlap(current, previous_near):
        return ""
    sim = max((_topic_similarity(current, p) for p in previous), default=0.0)
    if sim >= 0.18:
        return ""
    return "用户当前轮看起来切换了新话题；优先直接回答最后一条用户消息，除非用户明确要求继续旧话题。"


def _derive_topic_reopen_hint(topic_reopen: dict) -> str:
    """Frame a detected old-topic reopen so it is not read as a topic switch."""
    if not isinstance(topic_reopen, dict) or not topic_reopen.get("reopen"):
        return ""
    gap_min = int(float(topic_reopen.get("gap_minutes", 0.0) or 0.0))
    phrase = _gap_phrase(gap_min) if gap_min > 0 else "一段时间"
    source_label = str(topic_reopen.get("earliest_source_time", "") or "").strip()
    source_part = f"（来源约 {source_label}）" if source_label else ""
    return (
        f"用户在隔了{phrase}后重新提起今天早些时候的话题{source_part}；"
        "本轮相关旧事实已按来源时间标注，可以自然承接这个话题本身，"
        "但当时的瞬时动作、场景或状态不要当成现在仍在进行。"
    )


def _is_image_entry(entry: dict | None) -> bool:
    if not isinstance(entry, dict) or str(entry.get("role", "") or "").lower() != "user":
        return False
    content = str(entry.get("content", "") or "")
    if content.startswith("[对方发了一张图片"):
        return True
    return bool(content.startswith("[") and "看到一张图片" in content[:40])


def _compact_image_fact(entry: dict) -> str:
    meta = entry.get("vision_meta") if isinstance(entry.get("vision_meta"), dict) else {}
    compact = _clean_hint((meta or {}).get("compact", ""), 220)
    if compact:
        return compact
    content = str(entry.get("content", "") or "")
    if "早些时候发过一张图" in content:
        return _clean_hint(content, 220)
    tail = ""
    if "]\n" in content:
        tail = content.split("]\n", 1)[1].strip()
    return _clean_hint("[早些时候发过一张图（已浏览）]" + (f" {tail}" if tail else ""), 220)


def _visual_fact_records(
    history: list[dict],
    *,
    raw_ids: set[str],
    covered_card_user_ids: set[str],
    limit: int = 4,
) -> list[dict]:
    records: list[dict] = []
    for item in reversed(history):
        if not _is_image_entry(item):
            continue
        item_id = _entry_id(item)
        if item_id in raw_ids or item_id in covered_card_user_ids:
            continue
        fact = _compact_image_fact(item)
        if not fact:
            continue
        ts_ms = _entry_ts_ms(item)
        records.append({
            "text": fact,
            "kind": "fact",
            "source_channel": "visual_history",
            "source_ref_id": item_id,
            "source_character": _interlocutor_label(item),
            "created_at_ms": ts_ms,
            "source_time_period": _format_world_source_period(ts_ms),
            "salience": 0.7,
            "confidence": 0.85,
        })
        if len(records) >= limit:
            break
    return records


def _visual_fact_hints(
    history: list[dict],
    *,
    raw_ids: set[str],
    covered_card_user_ids: set[str],
    limit: int = 4,
) -> list[str]:
    return [
        _text(f"{_source_prefix(row.get('source_time_period', ''))}{row.get('text', '')}", 260)
        for row in _visual_fact_records(
            history,
            raw_ids=raw_ids,
            covered_card_user_ids=covered_card_user_ids,
            limit=limit,
        )
    ]


def _assistant_salient_fact_records(
    history: list[dict],
    *,
    raw_ids: set[str],
    covered_card_assistant_ids: set[str],
    current_character: str = "",
    limit: int = 4,
) -> list[dict]:
    records: list[dict] = []
    fallback_char = _text(current_character, 20) or "角色"
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "") or "").lower() != "assistant":
            continue
        item_id = _entry_id(item)
        if item_id in raw_ids or item_id in covered_card_assistant_ids:
            continue
        # Multi-room history keeps the actual speaker on each assistant row;
        # falling back to the current 1v1 character would reverse provenance.
        char = _text(item.get("character", ""), 20) or fallback_char
        content = _entry_text(item, 600)
        details = _salient_user_detail_hints(content, "")
        if not details:
            continue
        if "花展" in content and not any("花展" in d for d in details):
            for idx, detail in enumerate(details):
                if "向日葵" in detail:
                    details[idx] = "夏季花展：" + detail
                    break
        ts_ms = _entry_ts_ms(item)
        records.append({
            "text": _text(f"{char}早些时候提到" + "、".join(details), 220),
            "kind": "fact",
            "source_channel": "assistant_history",
            "source_ref_id": item_id,
            "source_character": char,
            "created_at_ms": ts_ms,
            "source_time_period": _format_world_source_period(ts_ms),
            "salience": 0.75,
            "confidence": 0.9,
        })
        if len(records) >= limit:
            break
    return records


def _assistant_salient_fact_hints(
    history: list[dict],
    *,
    raw_ids: set[str],
    covered_card_assistant_ids: set[str],
    current_character: str = "",
    limit: int = 4,
) -> list[str]:
    return [
        _text(f"{_source_prefix(row.get('source_time_period', ''))}{row.get('text', '')}", 260)
        for row in _assistant_salient_fact_records(
            history,
            raw_ids=raw_ids,
            covered_card_assistant_ids=covered_card_assistant_ids,
            current_character=current_character,
            limit=limit,
        )
    ]


def collect_uncovered_history_fact_records(
    history: list,
    fact_cards: list,
    *,
    current_character: str = "",
    group_mode: bool = False,
) -> list[dict]:
    """Expose compact facts from raw entries that have no TurnFactCard owner."""
    ordered = _ordered_history(history)
    current_user = _last_current_user(ordered)
    if not current_user:
        return []
    raw_ids = recent_dialogue_raw_ids(ordered, group_mode=group_mode)
    cards = [c for c in sanitize_fact_cards(fact_cards, limit=104) if not c.get("archived")]
    covered_user_ids = {
        user_id
        for card in cards
        for user_id in _card_user_ids(card)
    }
    covered_assistant_ids = {
        str(card.get("assistant_id", "") or "") for card in cards
        if str(card.get("assistant_id", "") or "")
    }
    return [
        *_visual_fact_records(
            ordered,
            raw_ids=raw_ids,
            covered_card_user_ids=covered_user_ids,
        ),
        *_assistant_salient_fact_records(
            ordered,
            raw_ids=raw_ids,
            covered_card_assistant_ids=covered_assistant_ids,
            current_character=current_character,
        ),
    ]


def select_fact_cards_for_workspace(history: list, fact_cards: list, *, group_mode: bool = False) -> list[dict]:
    """Exclude every card already carried by the bounded verbatim window."""
    raw_ids = recent_dialogue_raw_ids(history, group_mode=group_mode)
    if not raw_ids:
        return []
    return [
        card for card in sanitize_fact_cards(fact_cards, limit=104)
        if not card.get("archived")
        and (bool(card.get("room_transition")) or group_mode or (
            not raw_ids.intersection(_card_user_ids(card))
            and str(card.get("assistant_id", "") or "") not in raw_ids
        ))
    ]


def _delayed_reply_state_hints(pending_user_messages: Any, delay_sneak_prompt: str = "") -> list[str]:
    pending = [p for p in (pending_user_messages or []) if isinstance(p, dict)] if isinstance(pending_user_messages, list) else []
    hints: list[str] = []
    if pending:
        hints.append(
            "延迟回复状态：下面这些用户消息之前进入延迟回复队列，角色当时没有即时回应；"
            "本轮需要和当前用户消息自然融合，不要逐条复述，也不要假装没看到。"
        )
        for row in pending[:5]:
            txt = _clean_hint(row.get("content", ""), 160)
            if not txt:
                continue
            hints.append(f"未即时回应：{txt}")
    elif delay_sneak_prompt:
        hint = _clean_hint(delay_sneak_prompt, 320)
        if hint:
            hints.append(f"延迟回复状态：{hint}")
    return hints


def collect_conversation_state_hints(
    history: list,
    *,
    state_hints: list[str] | None = None,
    pending_user_messages: list[dict] | None = None,
    delay_sneak_prompt: str = "",
    infer_current_user_transition: bool = True,
    topic_reopen: dict | None = None,
) -> list[str]:
    """Collect every transient conversation guard through one shared channel.

    The raw history remains the audit source, but inline system markers, elapsed
    time, topic switches, and delayed-reply queue state are rendered once in the
    current state frame instead of surviving as scattered history messages.
    """
    ordered = _ordered_history(history)
    current_user = _last_current_user(ordered)
    _prev_user, prev_assistant = _last_completed_exchange(ordered, _entry_id(current_user))
    candidates: list[str] = []
    if infer_current_user_transition:
        gap_hint, _gap_min = _derive_current_gap_hint(current_user, prev_assistant)
        if gap_hint:
            candidates.append(gap_hint)
        # A genuine old-topic reopen must not be framed as a topic switch;
        # the reopen hint carries the continuity framing instead.
        _reopen_hint = _derive_topic_reopen_hint(topic_reopen)
        if _reopen_hint:
            candidates.append(_reopen_hint)
            topic_hint = ""
        else:
            topic_hint = _derive_topic_shift_hint(ordered, current_user)
        if topic_hint:
            candidates.append(topic_hint)
    candidates.extend(_inline_state_hints(ordered))
    candidates.extend(_delayed_reply_state_hints(pending_user_messages, delay_sneak_prompt))
    for item in state_hints or []:
        hint = _clean_hint(item)
        if hint:
            candidates.append(hint)
    out: list[str] = []
    for hint in candidates:
        clean = _clean_hint(hint, 320)
        if clean and clean not in out:
            out.append(clean)
    return out[:12]


def _render_user_content(text: str, target: str = "", *, interlocutor_label: str = "") -> str:
    peer = _text(interlocutor_label, 20) or _dev_name()
    if target:
        return f"[{peer} 对 {target} 说]：{text}"
    return f"[{peer}]：{text}"


def render_fact_ledger_messages(
    history: list,
    fact_cards: list,
    *,
    current_character: str = "",
    state_summary: str = "",
    state_hints: list[str] | None = None,
    pending_user_messages: list[dict] | None = None,
    delay_sneak_prompt: str = "",
    max_cards: int = 16,
    near_summary_cards: int = 2,
) -> list[dict]:
    ordered = _ordered_history(history)
    current_user = _last_current_user(ordered)
    if not current_user:
        return []
    current_user_id = _entry_id(current_user)
    _prev_user, prev_assistant = _last_completed_exchange(ordered, current_user_id)
    gap_hint, _gap_min = _derive_current_gap_hint(current_user, prev_assistant)
    recent_window = render_recent_dialogue_window(
        ordered,
        current_character=current_character,
    )
    raw_ids = set(recent_window.get("raw_ids", set()) or set())
    history_ts_by_id = {
        _entry_id(item): _entry_ts_ms(item)
        for item in ordered
        if _entry_id(item) and _entry_ts_ms(item) > 0
    }
    cards = [
        c for c in sanitize_fact_cards(fact_cards, limit=80)
        if not c.get("archived")
        and c.get("assistant_id") not in raw_ids
        and not raw_ids.intersection(_card_user_ids(c))
    ]
    cards = sorted(cards, key=lambda c: int(c.get("created_at_ms", 0) or 0))[-max_cards:]
    all_cards = [c for c in sanitize_fact_cards(fact_cards, limit=80) if not c.get("archived")]
    covered_card_user_ids = {
        user_id
        for card in all_cards
        for user_id in _card_user_ids(card)
    }
    covered_card_assistant_ids = {str(c.get("assistant_id", "") or "") for c in all_cards if str(c.get("assistant_id", "") or "")}

    lines = [
        "【正在进行的对话上下文 · 渐进式事实流】",
        "以下是旧对话的事实化记录，用来理解背景；不要复述、改写或模仿任何旧回复。",
        "越靠近当前轮的信息越完整；越早的信息只保留事实、纠正、承诺和未完成话题。",
        "每条事实前的“来源”表示该事实来自哪一段历史对话；旧来源时间不代表当前时间或当前状态。",
    ]
    summary = _text(state_summary, 900)
    if cards and "来源：" not in summary:
        summary = build_fact_state_summary(cards)
    if not summary:
        summary = build_fact_state_summary(cards)
    if summary:
        lines.append("【当前会话状态】")
        lines.append(summary)
    deduped_hints = collect_conversation_state_hints(
        ordered,
        state_hints=state_hints,
        pending_user_messages=pending_user_messages,
        delay_sneak_prompt=delay_sneak_prompt,
    )
    if deduped_hints:
        lines.append("【对话状态提示】")
        for hint in deduped_hints[:8]:
            lines.append(f"- {hint}")
    visual_hints = _visual_fact_hints(
        ordered,
        raw_ids=raw_ids,
        covered_card_user_ids=covered_card_user_ids,
    )
    if visual_hints:
        lines.append("【视觉事实】")
        for hint in visual_hints:
            lines.append(f"- {hint}")
    assistant_hints = _assistant_salient_fact_hints(
        ordered,
        raw_ids=raw_ids,
        covered_card_assistant_ids=covered_card_assistant_ids,
        current_character=current_character,
    )
    if assistant_hints:
        lines.append("【角色先前提到的具体线索】")
        for hint in assistant_hints:
            lines.append(f"- {hint}")
    older = cards[: max(0, len(cards) - near_summary_cards)]
    near = cards[max(0, len(cards) - near_summary_cards):]
    if older:
        lines.append("【中远端事实】")
        for card in older:
            source = _source_prefix(_card_source_period(card, history_ts_by_id))
            corrections = _coerce_list(card.get("corrections", []), limit=3)
            commitments = _coerce_list(card.get("commitments", []), limit=3)
            fact = _text(card.get("fact", "") or card.get("state_delta", ""), 180)
            if fact:
                lines.append(f"- {source}{fact}")
            for item in corrections:
                lines.append(f"- {source}纠正优先：{item}")
            for item in commitments:
                lines.append(f"- {source}已确认/承诺：{item}")
    if near:
        lines.append("【近端轻摘要】")
        for card in near:
            source = _source_prefix(_card_source_period(card, history_ts_by_id))
            near_text = _text(card.get("near_summary", "") or card.get("fact", ""), 180)
            if near_text:
                lines.append(f"- {source}{near_text}")

    messages = [{"role": "system", "content": "\n".join(lines)}]
    messages.extend(list(recent_window.get("messages", []) or []))
    return messages
