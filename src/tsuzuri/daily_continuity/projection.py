"""Selective FactWorkspace projection for unresolved continuity."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from .resolver import resolve_continuity_items


_KIND_PRIORITY = {
    "correction": 4.0,
    "commitment": 3.0,
    "open_thread": 2.2,
    "user_fact": 2.0,
    "relation_delta": 1.8,
    "event": 1.5,
}
_STOP_TOKENS = {
    "这个", "那个", "现在", "然后", "还是", "已经",
    "今天", "今日", "昨天", "昨日", "前天", "大前天",
    "明天", "后天", "大后天", "那天", "上次", "之前", "前几天",
    "还记得", "记得", "回想", "回忆", "答应", "承诺", "说过",
    "发生", "事情", "什么", "哪里", "去哪", "怎么", "如何",
    "整理", "开始", "结束", "完成", "取消", "需要", "帮忙",
    "这件事", "那件事", "件事",
    "你们", "我们", "他们", "她们", "你", "我", "他", "她", "吗",
}
_RECALL_CUE_RE = re.compile(r"昨天|前天|那天|之前|上次|前几天|昨日|还记得|回想|回忆")
_SCHEDULE_CUE_RE = re.compile(
    r"日程|待办|排满|忙不忙|"
    r"接下来.{0,8}(?:做什么|干什么)|"
    r"(?:今天|明天|后天|周[一二三四五六日天]|星期[一二三四五六日天])"
    r".{0,10}(?:计划|安排|打算|做什么)"
)
from tsuzuri.time_source import WORLD_TZ as _WORLD_TZ, world_tz_label  # world-clock time zone


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                raw = method()
            except TypeError:
                continue
            if isinstance(raw, Mapping):
                return dict(raw)
    return {}


def _tokens(value: Any) -> set[str]:
    text = re.sub(r"https?://\S+", " ", str(value or "").lower())
    for noise in sorted(_STOP_TOKENS, key=len, reverse=True):
        text = text.replace(noise, " ")
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text):
        if word in _STOP_TOKENS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            if len(word) <= 4:
                out.add(word)
            for size in (2, 3):
                for index in range(max(0, len(word) - size + 1)):
                    gram = word[index:index + size]
                    if gram not in _STOP_TOKENS:
                        out.add(gram)
        else:
            out.add(word)
    return out


def _schedule_actor(value: Any) -> str:
    text = str(value or "")
    if re.search(r"我们|咱们", text):
        return "shared"
    if re.search(r"你(?:的|明天|今天|后天|接下来)|你的", text):
        return "character"
    if re.search(r"我(?:的|明天|今天|后天|接下来)|我的", text):
        return "user"
    return ""


def _age_days(item: Mapping[str, Any], as_of_ms: int) -> int:
    raw_date = str(item.get("source_time_period", "") or "").strip()[:10]
    if not raw_date or as_of_ms <= 0:
        return 0
    try:
        source_date = date.fromisoformat(raw_date)
    except ValueError:
        return 0
    as_of_date = datetime.fromtimestamp(as_of_ms / 1000, tz=_WORLD_TZ).date()
    return max(0, (as_of_date - source_date).days)


def _date_from_ms(value: Any) -> date | None:
    try:
        timestamp_ms = int(value or 0)
    except (TypeError, ValueError):
        return None
    if timestamp_ms <= 0:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=_WORLD_TZ).date()


def _source_date(item: Mapping[str, Any]) -> date | None:
    raw = str(item.get("source_time_period", "") or "").strip()[:10]
    try:
        return date.fromisoformat(raw) if raw else None
    except ValueError:
        return None


def _minute_from_ms(value: Any) -> int | None:
    try:
        timestamp_ms = int(value or 0)
    except (TypeError, ValueError):
        return None
    return timestamp_ms // 60_000 if timestamp_ms > 0 else None


def _chinese_clock_number(value: str) -> int | None:
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    text = str(value or "")
    if not text:
        return None
    if "十" not in text:
        return digits.get(text) if len(text) == 1 else None
    tens, ones = text.split("十", 1)
    if tens not in {"", *digits.keys()} or ones not in {"", *digits.keys()}:
        return None
    return (digits.get(tens, 1) * 10) + digits.get(ones, 0)


def _clock_minutes(value: Any) -> set[int]:
    text = str(value or "")
    out: set[int] = set()
    for match in re.finditer(r"(?<!\d)([01]?\d|2[0-3])[:：]([0-5]\d)(?!\d)", text):
        out.add(int(match.group(1)) * 60 + int(match.group(2)))
    for match in re.finditer(r"(?<!\d)([01]?\d|2[0-3])点(?:(半)|([0-5]?\d)分?)?", text):
        minute = 30 if match.group(2) else int(match.group(3) or 0)
        out.add(int(match.group(1)) * 60 + minute)
    chinese_number = r"[零〇一二两三四五六七八九十]{1,3}"
    for match in re.finditer(
        rf"({chinese_number})[点时](?:(半)|({chinese_number})分?)?",
        text,
    ):
        hour = _chinese_clock_number(match.group(1))
        minute = 30 if match.group(2) else _chinese_clock_number(match.group(3) or "零")
        if hour is not None and minute is not None and 0 <= hour <= 23 and 0 <= minute <= 59:
            out.add(hour * 60 + minute)
    return out


def _date_minutes(dates: set[date], clock_minutes: set[int]) -> set[int]:
    return {
        int(
            datetime(
                target.year,
                target.month,
                target.day,
                minute // 60,
                minute % 60,
                tzinfo=_WORLD_TZ,
            ).timestamp()
            * 1000
        )
        // 60_000
        for target in dates
        for minute in clock_minutes
    }


def _query_temporal_targets(
    current_user_text: Any,
    as_of_ms: int,
) -> tuple[set[date], set[date], set[int], set[int]]:
    """Resolve source-day and due-day cues without treating time words as topics."""
    text = str(current_user_text or "")
    as_of_date = _date_from_ms(as_of_ms)
    source_dates: set[date] = set()
    due_dates: set[date] = set()

    for match in re.finditer(
        r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?",
        text,
    ):
        try:
            explicit = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
        source_dates.add(explicit)
        due_dates.add(explicit)

    if as_of_date is None:
        return source_dates, due_dates, set(), set()

    for match in re.finditer(r"(?<!\d)(\d{1,2})月(\d{1,2})日", text):
        try:
            explicit = date(as_of_date.year, int(match.group(1)), int(match.group(2)))
            if explicit > as_of_date:
                explicit = explicit.replace(year=as_of_date.year - 1)
        except ValueError:
            continue
        source_dates.add(explicit)
        due_dates.add(explicit)

    relative_sources = (
        ("大前天", -3),
        ("前天", -2),
        ("昨天", -1),
        ("昨日", -1),
        ("今天", 0),
        ("今日", 0),
    )
    relative_due = (
        ("大后天", 3),
        ("后天", 2),
        ("明天", 1),
        ("今天", 0),
        ("今日", 0),
    )
    for cue, offset in relative_sources:
        if cue in text:
            source_dates.add(as_of_date + timedelta(days=offset))
    for cue, offset in relative_due:
        if cue in text:
            due_dates.add(as_of_date + timedelta(days=offset))

    week_match = re.search(r"上周([一二三四五六日天])?", text)
    if week_match:
        previous_monday = as_of_date - timedelta(days=as_of_date.weekday() + 7)
        weekday_labels = "一二三四五六日"
        label = "日" if week_match.group(1) == "天" else week_match.group(1)
        if label:
            source_dates.add(previous_monday + timedelta(days=weekday_labels.index(label)))
        else:
            source_dates.update(previous_monday + timedelta(days=offset) for offset in range(7))
    clock_minutes = _clock_minutes(text)
    if not clock_minutes:
        return source_dates, due_dates, set(), set()
    source_minute_dates = set(source_dates)
    due_minute_dates = set(due_dates)
    if not source_minute_dates and not due_dates:
        source_minute_dates.add(as_of_date)
    if not due_minute_dates and not source_dates:
        due_minute_dates.add(as_of_date)
    return (
        source_dates,
        due_dates,
        _date_minutes(source_minute_dates, clock_minutes),
        _date_minutes(due_minute_dates, clock_minutes),
    )


def _temporal_match(
    item: Mapping[str, Any],
    source_dates: set[date],
    due_dates: set[date],
    source_minutes: set[int],
    due_minutes: set[int],
) -> bool:
    item_source_date = _source_date(item)
    item_due_date = _date_from_ms(item.get("due_at_ms", 0))
    item_source_minute = _minute_from_ms(item.get("created_at_ms", 0))
    item_due_minute = _minute_from_ms(item.get("due_at_ms", 0))
    return bool(
        (
            item_source_minute is not None
            and item_source_minute in source_minutes
        )
        or (
            not source_minutes
            and item_source_date is not None
            and item_source_date in source_dates
        )
        or (item_due_minute is not None and item_due_minute in due_minutes)
        or (
            not due_minutes
            and item_due_date is not None
            and item_due_date in due_dates
        )
    )


def _overlap(query_tokens: set[str], item: Mapping[str, Any]) -> tuple[float, list[str]]:
    item_text = " ".join([
        str(item.get("text", "") or ""),
        str(item.get("canonical_key", "") or ""),
        " ".join(str(value or "") for value in item.get("subjects", []) or []),
        " ".join(str(value or "") for value in item.get("topics", []) or []),
    ])
    shared = sorted(query_tokens & _tokens(item_text))
    if not shared:
        return 0.0, []
    score = sum(min(4, len(token)) for token in shared) / 2.0 + len(shared) * 0.65
    return min(12.0, score), shared[:12]


def _record_id(item_id: str, text: str) -> str:
    raw = f"daily_continuity|{item_id}|{text}".encode("utf-8", errors="ignore")
    return "fw-dc-" + hashlib.sha1(raw).hexdigest()[:12]


def _fact_record(
    item: Mapping[str, Any],
    *,
    score: float,
    matched_tokens: list[str],
    pinned: bool,
    as_of_date: date | None,
) -> dict[str, Any]:
    item_id = str(item.get("item_id", "") or "")
    text = str(item.get("text", "") or "").strip()[:360]
    source_time_period = str(item.get("source_time_period", "") or "")[:80]
    source_at_ms = int(item.get("created_at_ms", 0) or 0)
    source_date = _date_from_ms(source_at_ms)
    if source_at_ms > 0 and as_of_date is not None and source_date == as_of_date:
        source_time_period = datetime.fromtimestamp(
            source_at_ms / 1000,
            tz=_WORLD_TZ,
        ).strftime("%Y-%m-%d %H:%M ") + world_tz_label()
    return {
        "fact_id": _record_id(item_id, text),
        "text": text,
        "kind": str(item.get("kind", "") or "fact"),
        "source_channel": "tsuzuri.daily_continuity",
        "created_at_ms": int(item.get("created_at_ms", 0) or 0),
        "source_time_period": source_time_period,
        "salience": float(item.get("salience", 0.5) or 0.5),
        "confidence": float(item.get("confidence", 0.6) or 0.6),
        "source_ref_id": item_id,
        "source_character": str(item.get("source_character", "") or "")[:40],
        "canonical_key": str(item.get("canonical_key", "") or "")[:120],
        "actor": str(item.get("actor", "unknown") or "unknown")[:20],
        "due_at_ms": int(item.get("due_at_ms", 0) or 0),
        "due_text": str(item.get("due_text", "") or "")[:80],
        "due_precision": str(item.get("due_precision", "unknown") or "unknown")[:20],
        "due_state": str(item.get("due_state", "unknown") or "unknown")[:40],
        "continuity_status": str(item.get("status", "") or "active")[:30],
        "pinned": bool(pinned),
        "selection_score": round(score, 4),
        "matched_tokens": list(matched_tokens)[:12],
    }


def project_continuity_items(
    current_user_text: str,
    resolution_or_items: Any,
    *,
    max_records: int = 6,
    max_pins: int = 2,
) -> dict[str, Any]:
    """Select relevant capsule facts plus at most two unresolved pins."""
    query_tokens = _tokens(current_user_text)
    recall_requested = bool(_RECALL_CUE_RE.search(str(current_user_text or "")))
    schedule_requested = bool(_SCHEDULE_CUE_RE.search(str(current_user_text or "")))
    schedule_actor = _schedule_actor(current_user_text) if schedule_requested else ""
    record_limit = max(0, min(20, int(max_records or 0)))
    pin_limit = min(2, max(0, int(max_pins or 0)))
    resolved = _mapping(resolution_or_items)
    if isinstance(resolved.get("items"), list):
        items = [dict(row) for row in resolved["items"] if isinstance(row, Mapping)]
    else:
        items = resolve_continuity_items(resolution_or_items)["items"]
    as_of_ms = max(0, int(resolved.get("as_of_ms", 0) or 0))
    as_of_date = _date_from_ms(as_of_ms)
    (
        source_target_dates,
        due_target_dates,
        source_target_minutes,
        due_target_minutes,
    ) = _query_temporal_targets(
        current_user_text,
        as_of_ms,
    )
    has_temporal_target = bool(
        source_target_dates
        or due_target_dates
        or source_target_minutes
        or due_target_minutes
    )
    if (
        not query_tokens
        and not recall_requested
        and not schedule_requested
        and not has_temporal_target
    ) or (record_limit <= 0 and pin_limit <= 0):
        return {
            "contract_version": "daily_continuity_projection_v2",
            "records": [],
            "relevant_records": [],
            "unresolved_pins": [],
            "selected_item_ids": [],
            "rejected": [],
        }
    overlap_rows = [_overlap(query_tokens, item) for item in items]
    topic_match_available = any(
        overlap > 0
        for item, (overlap, _) in zip(items, overlap_rows)
        if str(item.get("status", "") or "active").lower()
        not in {"cancelled", "superseded", "archived"}
    )

    ranked: list[tuple[float, dict[str, Any], list[str]]] = []
    rejected: list[dict[str, str]] = []
    for item, (overlap, shared) in zip(items, overlap_rows):
        status = str(item.get("status", "") or "active").lower()
        if status in {"cancelled", "superseded", "archived"}:
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": status})
            continue
        kind = str(item.get("kind", "") or "").lower()
        age_days = _age_days(item, as_of_ms)
        item_source_date = _source_date(item)
        item_source_minute = _minute_from_ms(item.get("created_at_ms", 0))
        as_of_minute = _minute_from_ms(as_of_ms)
        source_is_today = bool(
            item_source_date is not None
            and as_of_date is not None
            and item_source_date == as_of_date
        )
        temporal_match = _temporal_match(
            item,
            source_target_dates,
            due_target_dates,
            source_target_minutes,
            due_target_minutes,
        )
        if (
            item_source_minute is not None
            and as_of_minute is not None
            and item_source_minute > as_of_minute
        ):
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "future_source_minute"})
            continue
        if age_days > 14 and (
            kind in {"event", "relation_delta"}
            or status in {"fulfilled", "resolved"}
        ):
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "stale"})
            continue
        schedule_like = bool(
            kind in {"commitment", "open_thread", "event"}
            and (
                int(item.get("due_at_ms", 0) or 0) > 0
                or str(item.get("temporal_status", "") or "").lower() == "planned"
            )
        )
        if (
            due_target_dates
            and int(item.get("due_at_ms", 0) or 0) > 0
            and not temporal_match
        ):
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "temporal_mismatch"})
            continue
        if (
            (source_target_minutes or due_target_minutes)
            and not temporal_match
        ):
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "temporal_mismatch"})
            continue
        if (
            schedule_requested
            and schedule_actor
            and schedule_like
            and str(item.get("actor", "") or "").lower() != schedule_actor
        ):
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "actor_mismatch"})
            continue
        schedule_candidate = bool(
            schedule_requested
            and status not in {"fulfilled", "cancelled", "superseded", "archived"}
            and schedule_like
            and (
                str(item.get("actor", "") or "").lower() == schedule_actor
                if schedule_actor
                else overlap > 0
            )
            and (
                overlap > 0
                or temporal_match
                or source_is_today
                or (item_source_date is None and not has_temporal_target)
            )
        )
        broad_recall = recall_requested and not topic_match_available and source_is_today
        if overlap <= 0 and not temporal_match and not broad_recall and not schedule_candidate:
            rejected.append({"item_id": str(item.get("item_id", "")), "reason": "irrelevant"})
            continue
        due_bonus = 0.0
        due_at_ms = int(item.get("due_at_ms", 0) or 0)
        if schedule_requested and due_at_ms > 0 and as_of_ms > 0:
            delta_ms = due_at_ms - as_of_ms
            if delta_ms < 0:
                due_bonus = 1.2
            elif delta_ms <= 24 * 60 * 60 * 1000:
                due_bonus = 2.4
            elif delta_ms <= 3 * 24 * 60 * 60 * 1000:
                due_bonus = 1.5
            elif delta_ms <= 7 * 24 * 60 * 60 * 1000:
                due_bonus = 0.8
        score = (
            _KIND_PRIORITY.get(kind, 1.0)
            + overlap
            + due_bonus
            + float(item.get("salience", 0.5) or 0.5)
            + 0.5 * float(item.get("confidence", 0.6) or 0.6)
            + (1.5 if temporal_match else 0.0)
            + (0.35 if broad_recall and overlap <= 0 else 0.0)
            + (0.25 if schedule_candidate and overlap <= 0 else 0.0)
        )
        ranked.append((score, item, shared))
    ranked.sort(
        key=lambda entry: (
            entry[0],
            -int(
                entry[1].get("due_at_ms", 0) or 2**63 - 1
                if schedule_requested
                else 2**63 - 1
            ),
            int(entry[1].get("created_at_ms", 0) or 0),
            str(entry[1].get("item_id", "")),
        ),
        reverse=True,
    )

    relevant_records = [
        _fact_record(
            item,
            score=score,
            matched_tokens=shared,
            pinned=False,
            as_of_date=as_of_date,
        )
        for score, item, shared in ranked[:record_limit]
    ]
    unresolved_ranked = [
        entry for entry in ranked
        if bool(entry[1].get("is_unresolved", False))
    ]
    unresolved_pins = [
        _fact_record(
            item,
            score=score,
            matched_tokens=shared,
            pinned=True,
            as_of_date=as_of_date,
        )
        for score, item, shared in unresolved_ranked[:pin_limit]
    ]
    pinned_ids = {record["source_ref_id"] for record in unresolved_pins}
    records = list(unresolved_pins)
    records.extend(
        record for record in relevant_records
        if record["source_ref_id"] not in pinned_ids
    )
    selected_ids = [row["source_ref_id"] for row in records]
    selected_id_set = set(selected_ids)
    for _, item, _ in ranked:
        item_id = str(item.get("item_id", ""))
        if item_id not in selected_id_set:
            rejected.append({"item_id": item_id, "reason": "record_limit"})
    for _, item, _ in unresolved_ranked[pin_limit:]:
        rejected.append({"item_id": str(item.get("item_id", "")), "reason": "pin_limit"})
    return {
        "contract_version": "daily_continuity_projection_v2",
        "records": records,
        "relevant_records": relevant_records,
        "unresolved_pins": unresolved_pins,
        "selected_item_ids": selected_ids,
        "rejected": rejected,
    }


def project_unresolved_continuity(
    current_user_text: str,
    resolution_or_items: Any,
    *,
    max_items: int = 2,
) -> dict[str, Any]:
    """Compatibility entry point; ``max_items`` controls only pin count."""
    return project_continuity_items(
        current_user_text,
        resolution_or_items,
        max_records=6,
        max_pins=max_items,
    )


def project_to_fact_workspace(
    current_user_text: str,
    resolution_or_items: Any,
    *,
    max_records: int = 6,
    max_pins: int = 2,
) -> list[dict[str, Any]]:
    """Convenience adapter returning only structured FactWorkspace records."""
    return project_continuity_items(
        current_user_text,
        resolution_or_items,
        max_records=max_records,
        max_pins=max_pins,
    )["records"]


__all__ = [
    "project_continuity_items",
    "project_to_fact_workspace",
    "project_unresolved_continuity",
]
