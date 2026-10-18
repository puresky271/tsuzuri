"""WORLD_TZ / mock clock SSOT.

Every runtime module must read "now" through this module instead of calling
``datetime.now()`` / ``date.today()`` / ``time.localtime()`` / ``time.strftime()``
directly (``test_time_source_guard.py`` enforces this). Rationale and the
"mock-blind clock silently bypasses tests" trap are documented in
``docs/brain/cognitive/time_as_core_input.md``.

Three kinds of "now", pick deliberately:

- ``now_world()``       — character-world time: UTC, ``DEBUG_MOCK_DATE``-aware.
                        Anything that drives schedules, date keys, grounding tables,
                        or is compared against WORLD_TZ-stamped values belongs here.
- ``utc_now_naive()`` — memory-core storage convention: real wall clock in UTC,
                        tz-naive, intentionally mock-invisible (memory ages in real
                        time; smoke tests clear memory instead of mocking).
- ``local_now()``     — server-local naive real clock; diagnostic metadata only
                        (dump filenames, ``generated_at`` fields), never used for
                        character-world decisions.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

# The world clock's time zone. ``WORLD_TZ`` keeps its historical name; it is whatever
# ``WORLD_TIMEZONE`` names (IANA key), defaulting to the deployment's zone.
WORLD_TIMEZONE_ENV = "WORLD_TIMEZONE"
WORLD_TIMEZONE_LABEL_ENV = "WORLD_TIMEZONE_LABEL"
_DEFAULT_WORLD_TIMEZONE = "UTC"
_DEFAULT_WORLD_TIMEZONE_LABEL = "UTC"
_FIXED_OFFSET_FALLBACKS = {"UTC": 9, "UTC": 0, "Etc/UTC": 0}


def _load_world_timezone():
    name = str(os.getenv(WORLD_TIMEZONE_ENV, "") or "").strip() or _DEFAULT_WORLD_TIMEZONE
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(name)
    except Exception:  # pragma: no cover - tzdata unavailable
        return timezone(timedelta(hours=_FIXED_OFFSET_FALLBACKS.get(name, 0)))


WORLD_TZ = _load_world_timezone()
WORLD_TZ = WORLD_TZ


def world_tz_label() -> str:
    """Short label printed next to world-clock times in prompts (e.g. "WORLD_TZ", "UTC")."""
    label = str(os.getenv(WORLD_TIMEZONE_LABEL_ENV, "") or "").strip()
    if label:
        return label
    name = str(os.getenv(WORLD_TIMEZONE_ENV, "") or "").strip() or _DEFAULT_WORLD_TIMEZONE
    return _DEFAULT_WORLD_TIMEZONE_LABEL if name == _DEFAULT_WORLD_TIMEZONE else name


def world_utc_offset(at: datetime | None = None) -> timedelta:
    """UTC offset of the world clock at ``at`` (naive values are world wall times)."""
    probe = at if at is not None else datetime.now(timezone.utc)
    if probe.tzinfo is None:
        probe = probe.replace(tzinfo=WORLD_TZ)
    return probe.astimezone(WORLD_TZ).utcoffset() or timedelta(0)

MOCK_DATE_ENV = "DEBUG_MOCK_DATE"

_MOCK_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")


def now_world_detailed() -> tuple[datetime, bool, str, str | None]:
    """Return ``(now_world, mock_applied, raw_env, parse_error)``.

    The diagnostics let health endpoints distinguish "mock really applied" from
    "env set but failed to parse". A naive mock value is interpreted as a WORLD_TZ
    wall time; an aware one is converted to WORLD_TZ.
    """
    raw_env = os.getenv(MOCK_DATE_ENV, "")
    mock = raw_env.strip().strip('"').strip("'")  # tolerate quoted values
    if mock:
        last_err = None
        for parser in (
            datetime.fromisoformat,
            lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M:%S"),
            lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M"),
            lambda value: datetime.strptime(value, "%Y-%m-%d"),
        ):
            try:
                mocked = parser(mock)
            except Exception as exc:  # try the next format
                last_err = repr(exc)
                continue
            if mocked.tzinfo is None:
                mocked = mocked.replace(tzinfo=WORLD_TZ)
            return mocked.astimezone(WORLD_TZ), True, raw_env, None
        parse_error = f"all parsers failed; last={last_err}"
        return datetime.now(WORLD_TZ), False, raw_env, parse_error
    return datetime.now(WORLD_TZ), False, raw_env, None


def now_world() -> datetime:
    """Current character-world time: WORLD_TZ, honoring ``DEBUG_MOCK_DATE``."""
    dt, _applied, _raw, _err = now_world_detailed()
    return dt


def ensure_world(value: datetime | None) -> datetime:
    """Normalize to an aware WORLD_TZ datetime.

    ``None`` resolves via :func:`now_world` (mock-aware). A naive value is a WORLD_TZ
    wall time (the schedule/pilot convention); an aware value is converted.
    """
    if value is None:
        return now_world()
    if value.tzinfo is None:
        return value.replace(tzinfo=WORLD_TZ)
    return value.astimezone(WORLD_TZ)


def world_today() -> str:
    """Current WORLD_TZ date as ``YYYY-MM-DD`` (mock-aware)."""
    return now_world().strftime("%Y-%m-%d")


def local_now() -> datetime:
    """Server-local naive real clock — diagnostic metadata only."""
    return datetime.now()


def utc_now() -> datetime:
    """UTC real clock, tz-aware — for ``...Z``/RFC-style diagnostic stamps."""
    return datetime.now(timezone.utc)


def utc_now_naive() -> datetime:
    """UTC real clock, tz-naive — the memory-core storage convention."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
