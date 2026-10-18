from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from tsuzuri.sqlite_reliability import connect_sqlite
from tsuzuri.time_source import now_world as _ts_now_world


from tsuzuri.time_source import WORLD_TZ as _WORLD_TZ, world_tz_label  # world-clock time zone
_DEVELOPER_SCOPE = "developer"
_TRANSITIONS = {"start", "complete", "cancel"}
_ACTION_TTL_MS = 24 * 60 * 60 * 1000
_ACTION_RECENT_GRACE_MS = 2 * 60 * 60 * 1000
_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = threading.Lock()
_STOP_TOKENS = {
    "我", "你", "了", "的", "然后", "之后", "现在", "已经", "刚刚", "刚才",
    "准备", "打算", "顺路", "一下", "一个", "一点", "继续", "事情", "东西",
}


def _world_now_ms() -> int:
    return int(_ts_now_world().timestamp() * 1000)


_START_FUTURE_MARKERS = (
    "明天", "之后", "以后", "到时候", "届时", "改天", "下次", "将来",
    "打算", "计划", "准备", "承诺", "会去", "会做", "会买", "会吃", "会发", "会拍",
)
_START_PRESENT_PATTERNS = (
    re.compile(r"(?:现在|正在|刚开始|刚在|这会儿|此刻|目前|先|开动|开始)"),
    re.compile(r"(?:^|[，。！？])(?:我|这边)?[^，。！？]{0,4}在[^，。！？]{1,24}"),
)


def _default_db_path() -> str:
    try:
        from tsuzuri.project_paths import MEMORY_CORE_DB_PATH

        return str(MEMORY_CORE_DB_PATH)
    except Exception:
        return str(Path(__file__).resolve().with_name("memory_core.db"))


def _one_line(value: Any, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit > 0 and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _clamp(value: Any, default: float = 0.75) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _timestamp_ms(entry: dict | None) -> int:
    raw = (entry or {}).get("timestamp", 0) or (entry or {}).get("ts", 0) or 0
    try:
        value = float(raw)
        if value <= 0:
            return _world_now_ms()
        if value < 1e11:
            value *= 1000.0
        return int(value)
    except (TypeError, ValueError):
        return _world_now_ms()


def _format_world(ts_ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).astimezone(_WORLD_TZ).strftime(
            "%Y-%m-%d %H:%M "
        ) + world_tz_label()
    except Exception:
        return ""


def _tokens(value: Any) -> set[str]:
    text = _one_line(value, 500)
    tokens: set[str] = set()
    for chunk in re.findall(r"[A-Za-z0-9_-]{2,}|[一-龥]{2,}", text):
        if chunk in _STOP_TOKENS:
            continue
        if re.fullmatch(r"[一-龥]+", chunk):
            for size in (2, 3):
                for idx in range(max(0, len(chunk) - size + 1)):
                    gram = chunk[idx:idx + size]
                    if gram not in _STOP_TOKENS:
                        tokens.add(gram)
        else:
            tokens.add(chunk.lower())
    return tokens


def _match_score(event_text: str, candidate_text: str) -> float:
    left = _tokens(event_text)
    right = _tokens(candidate_text)
    if not left or not right:
        return 0.0
    shared = left & right
    if not shared:
        return 0.0
    return len(shared) / max(1, min(len(left), len(right)))


def _canonical_event_text(value: Any) -> str:
    text = _one_line(value, 120)
    text = re.sub(r"^(?:已经|刚刚|刚才|刚|顺路|准备|打算|去|要)", "", text)
    text = re.sub(r"(?:完了|好了|结束了|完成了|了)$", "", text)
    replacements = {
        "买完": "买",
        "买好": "买",
        "吃完": "吃",
        "做完": "做",
        "写完": "写",
        "考完": "考试",
        "练完": "练习",
        "收拾完": "收拾",
    }
    for old, new in replacements.items():
        if text.startswith(old):
            text = new + text[len(old):]
            break
    return _one_line(text.strip("，。！？、 "), 100)


def _deterministic_updates(assistant_text: str) -> list[dict]:
    text = _one_line(assistant_text, 1600)
    if not text:
        return []
    out: list[dict] = []
    patterns = (
        re.compile(r"(?:我)?(?:已经|刚刚|刚才|刚)?(?P<evidence>(?:买完|买好|买到|吃完|做完|写完|考完|练完|收拾完)[^，。！？]{1,24}(?:了)?)"),
        re.compile(r"(?:我)?(?P<evidence>刚[^，。！？]{0,6}买了(?:个|点|些)?[^，。！？]{1,18})"),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        evidence = match.group("evidence")
        event_text = _canonical_event_text(evidence)
        if len(event_text) >= 2:
            out.append({
                "transition": "complete",
                "event_text": event_text,
                "evidence_span": evidence,
                "confidence": 1.0,
                "source": "deterministic",
            })
            break
    return out


def normalize_character_action_updates(raw_updates: Any, assistant_entry: dict) -> list[dict]:
    assistant_text = str((assistant_entry or {}).get("content", "") or "")
    # Proactive text is generated from the state frame and cannot become a new
    # authority over that frame. Otherwise one hallucinated auto-greet creates
    # a self-reinforcing location/action record on the next turn.
    if bool((assistant_entry or {}).get("proactive")) or str(
        (assistant_entry or {}).get("proactive_kind", "") or ""
    ).strip():
        return []
    source_at_ms = _timestamp_ms(assistant_entry)
    rows = raw_updates if isinstance(raw_updates, list) else []
    candidates = [*rows, *_deterministic_updates(assistant_text)]
    by_event: dict[tuple[str, str], dict] = {}
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        transition = _one_line(raw.get("transition"), 20).lower()
        evidence = _one_line(raw.get("evidence_span"), 240)
        if transition not in _TRANSITIONS or not evidence or evidence not in assistant_text:
            continue
        if transition == "start" and (
            any(marker in evidence for marker in _START_FUTURE_MARKERS)
            or not any(pattern.search(evidence) for pattern in _START_PRESENT_PATTERNS)
        ):
            continue
        event_text = _canonical_event_text(raw.get("event_text") or evidence)
        if len(event_text) < 2:
            continue
        if not action_has_world_support(event_text, evidence, assistant_entry.get('_action_authority_snapshot')):
            continue
        item = {
            "transition": transition,
            "event_text": event_text,
            "evidence_span": evidence,
            "source_at_ms": source_at_ms,
            "valid_until_ms": source_at_ms + _ACTION_TTL_MS,
            "confidence": _clamp(raw.get("confidence"), 0.78),
            "source": _one_line(raw.get("source"), 40) or "turn_fact_llm",
        }
        key = (transition, event_text)
        current = by_event.get(key)
        if current is None or float(item["confidence"]) >= float(current["confidence"]):
            by_event[key] = item
    return list(by_event.values())[:6]


def action_has_world_support(event_text: str, evidence: str, snapshot: dict | None) -> bool:
    """A retrospective claim is a report, not new authority over world history.

    Local present actions retain the existing interaction contract. Past-trip
    claims require same-day world support; generated speech alone cannot make
    a remembered visit into a new completed activity.
    """
    if re.search(r'昨天|前天|上次|以前|最近(?:去|到|看)|去过', evidence):
        return False
    retrospective = bool(re.search(r'今天去|去了|今天.*(?:逛完|到过|参观)', evidence))
    if not retrospective:
        return True
    if not isinstance(snapshot, dict):
        return False
    schedule = snapshot.get('schedule_state', {}) or {}
    world = snapshot.get('world_context', {}) or {}
    rows = [row for key in ('past_steps', 'current_steps') for row in schedule.get(key, [])
            if isinstance(row, dict)]
    candidates = [_row_text(row) + ' ' + str(row.get('location', '') or '') for row in rows]
    candidates.append(' '.join(str(world.get(key, '') or '') for key in ('activity', 'location')))
    # Similar venue types do not identify the same place (墨田 vs 阳光城).
    venues = re.findall(r'[一-龥A-Za-z]{2,16}(?:水族馆|水族館|博物馆|动物园|公园)', event_text)
    for candidate in candidates:
        if venues and not all(venue in candidate for venue in venues):
            continue
        if _match_score(event_text, candidate) >= .34:
            return True
    return False


def sanitize_persisted_action_updates(raw_updates: Any) -> list[dict]:
    rows = raw_updates if isinstance(raw_updates, list) else []
    out: list[dict] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        transition = _one_line(raw.get("transition"), 20).lower()
        event_text = _canonical_event_text(raw.get("event_text"))
        evidence = _one_line(raw.get("evidence_span"), 240)
        if transition not in _TRANSITIONS or len(event_text) < 2 or not evidence:
            continue
        out.append({
            "transition": transition,
            "event_text": event_text,
            "evidence_span": evidence,
            "source_at_ms": int(raw.get("source_at_ms", 0) or 0),
            "valid_until_ms": int(raw.get("valid_until_ms", 0) or 0),
            "confidence": _clamp(raw.get("confidence"), 0.78),
            "source": _one_line(raw.get("source"), 40) or "turn_fact_llm",
        })
    return out[:6]


def ensure_schema(db_path: str | None = None) -> None:
    path = str(db_path or _default_db_path())
    with _SCHEMA_LOCK:
        if path in _SCHEMA_READY and Path(path).exists():
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(connect_sqlite(path, timeout=8)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS character_action_events (
                    action_id TEXT PRIMARY KEY,
                    character TEXT NOT NULL,
                    transition TEXT NOT NULL,
                    event_text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_session_id TEXT NOT NULL DEFAULT '',
                    source_user_id TEXT NOT NULL DEFAULT '',
                    source_assistant_id TEXT NOT NULL DEFAULT '',
                    source_turn_id TEXT NOT NULL DEFAULT '',
                    source_at_ms INTEGER NOT NULL,
                    valid_until_ms INTEGER NOT NULL,
                    evidence_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.75,
                    supersedes_id TEXT NOT NULL DEFAULT '',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(character_action_events)").fetchall()}
            if "supersedes_id" not in columns:
                conn.execute(
                    "ALTER TABLE character_action_events ADD COLUMN supersedes_id TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_character_action_active "
                "ON character_action_events(character, status, valid_until_ms, source_at_ms)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_character_action_source "
                "ON character_action_events(source_user_id, source_assistant_id)"
            )
            conn.commit()
        _SCHEMA_READY.add(path)


def _action_id(character: str, assistant_id: str, transition: str, event_text: str) -> str:
    raw = f"{character}|{assistant_id}|{transition}|{event_text}".encode("utf-8", errors="ignore")
    return "cae-" + hashlib.sha1(raw).hexdigest()[:20]


def commit_turn_action_updates(
    character: str,
    session_id: str,
    card: dict,
    *,
    db_path: str | None = None,
) -> list[dict]:
    if str((card or {}).get("audience_scope", "") or "") not in ("", _DEVELOPER_SCOPE):
        return []
    updates = card.get("character_action_updates", []) if isinstance(card, dict) else []
    if not isinstance(updates, list) or not updates:
        return []
    path = str(db_path or _default_db_path())
    ensure_schema(path)
    char = _one_line(character or card.get("character"), 40)
    assistant_id = _one_line(card.get("assistant_id"), 120)
    user_id = _one_line(card.get("user_id"), 120)
    turn_id = _one_line(card.get("turn_id"), 120)
    now_ms = _world_now_ms()
    committed: list[dict] = []
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        for update in updates[:6]:
            if not isinstance(update, dict):
                continue
            transition = _one_line(update.get("transition"), 20).lower()
            event_text = _canonical_event_text(update.get("event_text"))
            if transition not in _TRANSITIONS or len(event_text) < 2:
                continue
            action_id = _action_id(char, assistant_id, transition, event_text)
            if conn.execute(
                "SELECT 1 FROM character_action_events WHERE action_id=?",
                (action_id,),
            ).fetchone():
                continue
            previous_rows = conn.execute(
                """
                SELECT action_id, transition, event_text FROM character_action_events
                WHERE character=? AND status='active'
                ORDER BY source_at_ms DESC
                """,
                (char,),
            ).fetchall()
            predecessor = next(
                (
                    row for row in previous_rows
                    if _match_score(event_text, str(row[2] or "")) >= 0.6
                ),
                None,
            )
            if predecessor is not None and str(predecessor[1] or "") == "complete" and transition == "start":
                # Action progress is monotonic inside its 24-hour authority
                # window. A later planning-like line cannot reopen completion.
                continue
            supersedes_id = str(predecessor[0] or "") if predecessor is not None else ""
            if supersedes_id:
                conn.execute(
                    "UPDATE character_action_events SET status='superseded', updated_at_ms=? WHERE action_id=?",
                    (now_ms, supersedes_id),
                )
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO character_action_events(
                    action_id, character, transition, event_text, status,
                    source_session_id, source_user_id, source_assistant_id, source_turn_id,
                    source_at_ms, valid_until_ms, evidence_text, confidence, supersedes_id,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    char,
                    transition,
                    event_text,
                    _one_line(session_id, 180),
                    user_id,
                    assistant_id,
                    turn_id,
                    int(update.get("source_at_ms", 0) or card.get("created_at_ms", 0) or now_ms),
                    int(update.get("valid_until_ms", 0) or now_ms + _ACTION_TTL_MS),
                    _one_line(update.get("evidence_span"), 240),
                    _clamp(update.get("confidence"), 0.78),
                    supersedes_id,
                    now_ms,
                    now_ms,
                ),
            )
            if cursor.rowcount:
                committed.append({"action_id": action_id, **dict(update)})
        conn.commit()
    return committed


def load_active_actions(
    character: str,
    *,
    now_ms: int | None = None,
    db_path: str | None = None,
) -> list[dict]:
    path = str(db_path or _default_db_path())
    if not Path(path).exists():
        return []
    ensure_schema(path)
    now = int(now_ms or _world_now_ms())
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute(
            "UPDATE character_action_events SET status='expired', updated_at_ms=? "
            "WHERE character=? AND status='active' AND valid_until_ms < ?",
            (now, _one_line(character, 40), now),
        )
        rows = conn.execute(
            """
            SELECT * FROM character_action_events
            WHERE character=? AND status='active' AND valid_until_ms >= ?
            ORDER BY source_at_ms DESC, confidence DESC
            LIMIT 16
            """,
            (_one_line(character, 40), now),
        ).fetchall()
        conn.commit()
    return [dict(row) for row in rows]


def get_completed_events(
    character: str,
    *,
    window_h: float = 8,
    db_path: str | None = None,
) -> list[str]:
    """Return event_text list for 'complete' action events within window_h hours.

    Used by plan-gen to exclude already-confirmed-done steps from re-planning.
    Only returns records that are still within their TTL (status='active') so
    expired or archived completions do not pollute the plan-gen prompt.
    """
    path = str(db_path or _default_db_path())
    if not Path(path).exists():
        return []
    ensure_schema(path)
    now_ms = _world_now_ms()
    since_ms = now_ms - int(window_h * 3600 * 1000)
    try:
        with closing(connect_sqlite(path, timeout=5)) as conn:
            rows = conn.execute(
                """
                SELECT event_text FROM character_action_events
                WHERE character = ?
                  AND status = 'active'
                  AND transition = 'complete'
                  AND valid_until_ms >= ?
                  AND source_at_ms >= ?
                ORDER BY source_at_ms DESC
                LIMIT 8
                """,
                (_one_line(character, 40), now_ms, since_ms),
            ).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []


def archive_actions_for_source_message_ids(
    message_ids: Iterable[str],
    *,
    db_path: str | None = None,
) -> int:
    ids = [_one_line(value, 120) for value in message_ids if _one_line(value, 120)]
    path = str(db_path or _default_db_path())
    if not ids or not Path(path).exists():
        return 0
    ensure_schema(path)
    placeholders = ",".join("?" for _ in ids)
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        now_ms = _world_now_ms()
        conn.execute("BEGIN IMMEDIATE")
        removed = conn.execute(
            f"SELECT action_id, character, supersedes_id FROM character_action_events "
            f"WHERE (source_user_id IN ({placeholders}) OR source_assistant_id IN ({placeholders})) "
            "AND status='active'",
            (*ids, *ids),
        ).fetchall()
        cursor = conn.execute(
            f"UPDATE character_action_events SET status='archived', updated_at_ms=? "
            f"WHERE (source_user_id IN ({placeholders}) OR source_assistant_id IN ({placeholders})) "
            "AND status='active'",
            (now_ms, *ids, *ids),
        )
        for row in removed:
            predecessor_id = str(row["supersedes_id"] or "")
            if not predecessor_id:
                continue
            conn.execute(
                """
                UPDATE character_action_events SET status='active', updated_at_ms=?
                WHERE action_id=? AND status='superseded' AND valid_until_ms>=?
                  AND NOT EXISTS (
                      SELECT 1 FROM character_action_events newer
                      WHERE newer.character=? AND newer.status='active' AND newer.action_id<>?
                        AND newer.event_text=(SELECT event_text FROM character_action_events WHERE action_id=?)
                  )
                """,
                (now_ms, predecessor_id, now_ms, row["character"], row["action_id"], predecessor_id),
            )
        conn.commit()
        return int(cursor.rowcount or 0)


def _row_text(row: dict) -> str:
    return " ".join(
        _one_line(row.get(key), 180)
        for key in ("title", "activity", "topic", "text")
        if _one_line(row.get(key), 180)
    )


def _advance_activity_text(value: Any, event_text: str) -> str:
    text = _one_line(value, 220)
    if not text or _match_score(event_text, text) < 0.34:
        return text
    for separator in ("然后", "之后", "接着", "再"):
        parts = text.split(separator, 1)
        if len(parts) == 2 and _match_score(event_text, parts[0]) >= 0.34:
            return _one_line(parts[1], 220)
    return f"已完成{event_text}，准备继续后续安排"


def _future_step_can_match_action(row: dict, action_source_ms: int) -> bool:
    """Prevent a recent completion from consuming tomorrow's repeated event."""
    raw_start = _one_line(row.get("start_world") or row.get("start"), 60)
    if not raw_start or action_source_ms <= 0:
        return False
    try:
        start = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
    except ValueError:
        return False
    if start.tzinfo is None:
        start = start.replace(tzinfo=_WORLD_TZ)
    action_dt = datetime.fromtimestamp(action_source_ms / 1000.0, tz=timezone.utc).astimezone(_WORLD_TZ)
    return start.astimezone(_WORLD_TZ).date() == action_dt.date()


def _overlay_now(world: dict, schedule: dict) -> datetime:
    raw = world.get("dt") or schedule.get("asof_world")
    if isinstance(raw, datetime):
        dt = raw
    else:
        try:
            dt = datetime.fromisoformat(str(raw or "").replace("Z", "+00:00"))
        except (TypeError, ValueError):
            dt = _ts_now_world()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_WORLD_TZ)
    return dt.astimezone(_WORLD_TZ)


def _action_record_applies_now(action: dict, world: dict, schedule: dict, now_dt: datetime) -> bool:
    """Keep dialogue actions bounded to their real temporal segment."""
    source_ms = int(action.get("source_at_ms", 0) or 0)
    if source_ms <= 0:
        return False
    source_dt = datetime.fromtimestamp(source_ms / 1000.0, tz=timezone.utc).astimezone(_WORLD_TZ)
    age_ms = int((now_dt - source_dt).total_seconds() * 1000)
    recent = -5 * 60 * 1000 <= age_ms <= _ACTION_RECENT_GRACE_MS
    if source_dt.date() != now_dt.date() and not recent:
        return False

    transition = _one_line(action.get("transition"), 20)
    if transition != "start" or recent:
        return True

    # A long-running start survives only while the current plan still describes
    # the same action. This prevents a morning location claim from replacing an
    # unrelated lunch/evening step merely because its storage TTL is 24 hours.
    current_rows = schedule.get("current_steps", []) if isinstance(schedule.get("current_steps", []), list) else []
    current_blob = " ".join(
        value for value in (
            _one_line(world.get("activity"), 220),
            _one_line(world.get("location"), 220),
            _one_line(schedule.get("expected_activity"), 220),
            _one_line(schedule.get("expected_location"), 220),
            *(_row_text(row) for row in current_rows if isinstance(row, dict)),
        ) if value
    )
    return _match_score(_canonical_event_text(action.get("event_text")), current_blob) >= 0.34


def apply_action_overlays(
    world_context: dict | None,
    schedule_state: dict | None,
    action_records: list[dict] | None,
) -> tuple[dict, dict, list[str]]:
    """Project dialogue-confirmed action transitions over the static schedule."""
    world = dict(world_context or {}) if isinstance(world_context, dict) else {}
    schedule = dict(schedule_state or {}) if isinstance(schedule_state, dict) else {}
    records = [dict(row) for row in (action_records or []) if isinstance(row, dict)]
    if not records:
        return world, schedule, []

    now_dt = _overlay_now(world, schedule)
    records = [
        row for row in records
        if _action_record_applies_now(row, world, schedule, now_dt)
    ]
    if not records:
        return world, schedule, []

    applied_lines: list[str] = []
    past_rows = [dict(row) for row in schedule.get("past_steps", []) if isinstance(row, dict)]
    current_rows = [dict(row) for row in schedule.get("current_steps", []) if isinstance(row, dict)]
    future_rows = [dict(row) for row in schedule.get("future_steps", []) if isinstance(row, dict)]
    for action in reversed(records):
        transition = _one_line(action.get("transition"), 20)
        event_text = _canonical_event_text(action.get("event_text"))
        if transition not in _TRANSITIONS or not event_text:
            continue
        matched = False
        matched_schedule = False
        if transition in ("complete", "cancel"):
            for source_kind, source_rows in (("current", current_rows), ("future", future_rows)):
                for row in list(source_rows):
                    if _match_score(event_text, _row_text(row)) < 0.34:
                        continue
                    if source_kind == "future" and not _future_step_can_match_action(
                        row,
                        int(action.get("source_at_ms", 0) or 0),
                    ):
                        continue
                    source_rows.remove(row)
                    matched = True
                    matched_schedule = True
                    if transition == "complete":
                        completed = dict(row)
                        completed["conversation_status"] = "completed"
                        completed["completed_at_ms"] = int(action.get("source_at_ms", 0) or 0)
                        past_rows.append(completed)
                    break
                if matched:
                    break
        if transition == "complete":
            for key in ("expected_activity", "next_activity"):
                old = _one_line(schedule.get(key), 220)
                new = _advance_activity_text(old, event_text)
                if new != old:
                    schedule[key] = new
                    matched = True
            world_activity = _one_line(world.get("activity"), 220)
            advanced_world = _advance_activity_text(world_activity, event_text)
            if advanced_world != world_activity:
                world["activity"] = advanced_world
                matched = True
            temporal = dict(world.get("temporal_facts", {}) or {})
            present = dict(temporal.get("present_fact", {}) or {})
            present_activity = _one_line(present.get("activity"), 220)
            advanced_present = _advance_activity_text(present_activity, event_text)
            if advanced_present != present_activity:
                present["activity"] = advanced_present
                temporal["present_fact"] = present
                world["temporal_facts"] = temporal
                matched = True
            future_facts = list(temporal.get("future_facts", []) or [])
            kept_future = []
            for future in future_facts:
                future_text = _row_text(future) if isinstance(future, dict) else _one_line(future, 220)
                if _match_score(event_text, future_text) >= 0.34:
                    matched = True
                    continue
                kept_future.append(future)
            past_facts = list(temporal.get("past_facts", []) or [])
            if not matched_schedule:
                completed_fact = {"text": f"已完成{event_text}"}
                if not any(
                    _match_score(event_text, _row_text(item) if isinstance(item, dict) else str(item)) >= 0.6
                    for item in past_facts
                ):
                    past_facts.append(completed_fact)
            temporal["future_facts"] = kept_future
            temporal["past_facts"] = past_facts[-12:]
            world["temporal_facts"] = temporal
            applied_lines.append(
                f"已完成：{event_text}（{_format_world(int(action.get('source_at_ms', 0) or 0)) or '近期对话确认'}）"
            )
        elif transition == "cancel":
            applied_lines.append(f"已取消：{event_text}")
        elif transition == "start":
            world["activity"] = event_text
            schedule["expected_activity"] = event_text
            applied_lines.append(f"正在进行：{event_text}")
        if not matched and transition == "complete":
            # Unmatched actions remain a bounded current-state fact. They do
            # not rewrite the source schedule but still prevent local rollback.
            continue

    # 倒推全量（与 turn_context_workspace 时间态分区同策略，2026-09-05）；正推窗口保留。
    schedule["past_steps"] = past_rows
    schedule["current_steps"] = current_rows[:2]
    schedule["future_steps"] = future_rows[:4]
    schedule["conversation_action_overlays"] = applied_lines[-8:]
    return world, schedule, applied_lines[-8:]


__all__ = [
    "apply_action_overlays",
    "archive_actions_for_source_message_ids",
    "commit_turn_action_updates",
    "ensure_schema",
    "get_completed_events",
    "load_active_actions",
    "normalize_character_action_updates",
    "sanitize_persisted_action_updates",
]
