"""Deterministic WORLD_TZ parsing for cited continuity due expressions.

2026-09-11：补齐「无钟点时段词」解析。此前 `_resolve_time` 只认「N点/时」，
`due_text` 里的「今晚」「晚上」「放学」一律解析失败 → 大量未结 commitment 永远
`due_at_ms = 0`，既标不出 overdue、也永远占着 FactWorkspace 的 pin 席位
（审计见 `_offline_smoke_out/due_period_audit_20260911.json`：130 条承诺里 100 条无 due）。
时段词按固定默认时点落成**日期级** due（precision=date），不引入新枚举：
  凌晨 00:30 / 早上·早晨 08:00 / 上午 10:00 / 中午 12:00 / 下午 15:00 /
  傍晚 18:00 / 晚上 20:00 / 放学 16:00
日期锚点为「今天/今晚/今夜/今早」（当日）与「明天/明晚/明早」（次日）；无日期词时
仍锚定 `source_at_ms` 当天（prompt 侧只允许逐字复制原文表达，禁止推算）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from zoneinfo import ZoneInfo


from tsuzuri.time_source import WORLD_TZ  # world-clock time zone
_WEEKDAYS = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}
_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
# 时段词 → 默认时点（长词在前，避免「明早」被「早」类短词提前吃掉）。
_PERIOD_DEFAULT_TIMES: tuple[tuple[str, int, int], ...] = (
    ("凌晨", 0, 30),
    ("早晨", 8, 0),
    ("早上", 8, 0),
    ("今早", 8, 0),
    ("明早", 8, 0),
    ("上午", 10, 0),
    ("中午", 12, 0),
    ("下午", 15, 0),
    ("傍晚", 18, 0),
    ("晚上", 20, 0),
    ("今晚", 20, 0),
    ("今夜", 20, 0),
    ("明晚", 20, 0),
    ("放学", 16, 0),
)
# 含时段词的日锚点：token → 相对天数（长词在前）。
_RELATIVE_DATE_ANCHORS: tuple[tuple[str, int], ...] = (
    ("大后天", 3),
    ("后天", 2),
    ("明晚", 1),
    ("明早", 1),
    ("明天", 1),
    ("今晚", 0),
    ("今夜", 0),
    ("今早", 0),
    ("今天", 0),
)
_PERIOD_DEFAULT_HOUR = 9


@dataclass(frozen=True)
class DueResolution:
    due_at_ms: int = 0
    precision: str = "unknown"


def _small_number(value: str) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, right = text.split("十", 1)
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(char in _CN_DIGITS for char in text):
        number = 0
        for char in text:
            number = number * 10 + _CN_DIGITS[char]
        return number
    return None


def _resolve_date(text: str, base_date: date) -> tuple[date, bool]:
    relative = next(
        (
            (token, offset)
            for token, offset in _RELATIVE_DATE_ANCHORS
            if token in text
        ),
        None,
    )
    if relative is not None:
        return base_date + timedelta(days=relative[1]), True

    absolute = re.search(r"(?<!\d)(\d{1,2})月(\d{1,2})(?:日|号)?", text)
    if absolute:
        month = int(absolute.group(1))
        day = int(absolute.group(2))
        year = base_date.year
        try:
            candidate = date(year, month, day)
            if candidate < base_date - timedelta(days=31):
                candidate = date(year + 1, month, day)
            return candidate, True
        except ValueError:
            return base_date, False

    weekday = re.search(r"(下周|下星期|周|星期)([一二三四五六日天])", text)
    if weekday:
        target = _WEEKDAYS[weekday.group(2)]
        if weekday.group(1) in {"下周", "下星期"}:
            delta = (7 - base_date.weekday()) + target
        else:
            delta = (target - base_date.weekday()) % 7 or 7
        return base_date + timedelta(days=delta), True
    return base_date, False


def _resolve_time(text: str) -> tuple[int, int, bool]:
    match = re.search(
        r"(凌晨|早上|上午|中午|下午|傍晚|晚上)?\s*"
        r"([零〇一二两三四五六七八九十\d]{1,3})\s*(?:点|时)"
        r"(?:(半)|([零〇一二两三四五六七八九十\d]{1,3})分?)?",
        text,
    )
    if not match:
        return 9, 0, False
    hour = _small_number(match.group(2))
    minute = 30 if match.group(3) else (_small_number(match.group(4)) or 0)
    if hour is None or not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return 9, 0, False
    period = match.group(1) or ""
    if period in {"下午", "傍晚", "晚上"} and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    elif period == "凌晨" and hour == 12:
        hour = 0
    return hour, minute, True


def _resolve_period(text: str) -> tuple[int, int] | None:
    """无钟点时段词 → 默认时点；找不到返回 None。"""
    for token, hour, minute in _PERIOD_DEFAULT_TIMES:
        if token in text:
            return hour, minute
    return None


def derive_due_at_ms(
    due_text: str,
    *,
    date_world: str,
    source_at_ms: int = 0,
) -> DueResolution:
    text = " ".join(str(due_text or "").split()).strip()
    if not text:
        return DueResolution()
    if source_at_ms > 0:
        base_date = datetime.fromtimestamp(source_at_ms / 1000, tz=WORLD_TZ).date()
    else:
        try:
            base_date = date.fromisoformat(str(date_world or ""))
        except ValueError:
            return DueResolution()
    target_date, has_date = _resolve_date(text, base_date)
    hour, minute, has_clock = _resolve_time(text)
    period = None if has_clock else _resolve_period(text)
    if not has_date and not has_clock and period is None:
        return DueResolution()
    if not has_clock:
        # 无钟点时用时段默认时点；连时段词都没有（如「明天」）沿用既有 09:00 约定。
        hour, minute = period if period is not None else (_PERIOD_DEFAULT_HOUR, 0)
    target = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=WORLD_TZ,
    )
    # 只有钟点（N点/时）算 datetime；日期或时段词都按日期级落库（2026-09-11 决定）。
    precision = "datetime" if has_clock else "date"
    return DueResolution(int(target.timestamp() * 1000), precision)


__all__ = ["DueResolution", "derive_due_at_ms"]
