from __future__ import annotations

import hashlib
import json
import re
from tsuzuri.persona_registry import user_name
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from tsuzuri.sqlite_reliability import connect_sqlite
from tsuzuri.time_source import now_world


from tsuzuri.time_source import WORLD_TZ as _WORLD_TZ, world_tz_label  # world-clock time zone
_SUBJECT_DEVELOPER = "developer"
_DEVELOPER_SCOPE = "developer"
_MAX_ACTIVE_ROWS = 16
_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = threading.Lock()

# This ledger owns continuing interlocutor state, not episodic recollection.
# Expiry is intentionally conservative: explicit corrections supersede rows
# immediately, while a forgotten temporary state cannot remain current forever.
_PREDICATES = {
    "academic.phase": {"label": "学业阶段", "ttl_days": 120},
    "exam.status": {"label": "考试状态", "ttl_days": 30},
    "health.current": {"label": "身体状态", "ttl_days": 14},
    "travel.status": {"label": "出行状态", "ttl_days": 45},
    "workload.current": {"label": "近期负担", "ttl_days": 14},
    "availability.current": {"label": "近期时间状态", "ttl_days": 7},
    "location.current": {"label": "当前所在地", "ttl_hours": 12},
}

_VALUE_LABELS = {
    "summer_vacation": "正在放暑假",
    "winter_vacation": "正在放寒假",
    "spring_vacation": "正在放春假",
    "in_term": "已经开学，处于在校学期",
    "exam_ongoing": "考试仍在进行",
    "exam_finished": "考试已经结束",
}

_VALUE_ALIASES = {
    "暑假": "summer_vacation",
    "放暑假": "summer_vacation",
    "summer vacation": "summer_vacation",
    "寒假": "winter_vacation",
    "放寒假": "winter_vacation",
    "winter vacation": "winter_vacation",
    "春假": "spring_vacation",
    "放春假": "spring_vacation",
    "spring vacation": "spring_vacation",
    "开学": "in_term",
    "已开学": "in_term",
    "在校学期": "in_term",
    "考试中": "exam_ongoing",
    "考试仍在进行": "exam_ongoing",
    "考试结束": "exam_finished",
    "考试结束了": "exam_finished",
    "考完了": "exam_finished",
}

_FUTURE_MARKERS_RE = re.compile(r"(?:快要|快|准备|打算|计划|之后|以后|明天|后天|下周|下个月|还没|尚未)")
_FIRST_PERSON_RE = re.compile(r"(?:^|[，。！？、\s])(?:我|我们|本人)(?:$|[，。！？、\s]|已经|现在|目前|今天|这边|在|要|快|刚)")
_CJK_RE = re.compile(r"[一-龥]")

# Closed-value predicates: the LLM extractor sometimes files unrelated personal
# facts under a predicate (e.g. academic.phase="刚满20岁"). Only canonical values
# may pass; free-text predicates must at least be written in CJK, otherwise lazy
# English placeholders ("recovered", "planned_trip") would render verbatim.
_ENUM_PREDICATE_VALUES = {
    "academic.phase": {"summer_vacation", "winter_vacation", "spring_vacation", "in_term"},
    "exam.status": {"exam_ongoing", "exam_finished"},
}


def _value_allowed(predicate: str, value: str) -> bool:
    allowed = _ENUM_PREDICATE_VALUES.get(predicate)
    if allowed is not None:
        return value in allowed
    return bool(_CJK_RE.search(value))


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


def _clamp(value: Any, default: float = 0.7) -> float:
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
            return int(time.time() * 1000)
        if value < 1e11:
            value *= 1000.0
        return int(value)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


def _format_world(ts_ms: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).astimezone(_WORLD_TZ).strftime(
            "%Y-%m-%d %H:%M "
        ) + world_tz_label()
    except Exception:
        return ""


def _parse_date_ms(value: Any, *, end_of_day: bool = False) -> int:
    text = _one_line(value, 40)
    if not text:
        return 0
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_WORLD_TZ)
    if end_of_day and len(text) <= 10:
        dt = dt.replace(hour=23, minute=59, second=59)
    return int(dt.timestamp() * 1000)


def _canonical_value(predicate: str, value: Any) -> str:
    text = _one_line(value, 120)
    alias = _VALUE_ALIASES.get(text.lower()) or _VALUE_ALIASES.get(text)
    if alias:
        return alias
    if predicate == "academic.phase":
        if "暑假" in text:
            return "summer_vacation"
        if "寒假" in text:
            return "winter_vacation"
        if "春假" in text:
            return "spring_vacation"
        if "开学" in text:
            return "in_term"
    if predicate == "exam.status":
        if re.search(r"(?:结束|考完|完事)", text):
            return "exam_finished"
        if re.search(r"(?:考试中|还在考|仍在考|考试周)", text):
            return "exam_ongoing"
    return text


def _default_valid_until_ms(predicate: str, valid_from_ms: int) -> int:
    spec = _PREDICATES.get(predicate, {})
    if spec.get("ttl_hours"):
        return valid_from_ms + int(spec["ttl_hours"]) * 60 * 60 * 1000
    if spec.get("ttl_days"):
        return valid_from_ms + int(spec["ttl_days"]) * 24 * 60 * 60 * 1000
    return 0


def _relative_valid_from_ms(evidence: str, source_at_ms: int) -> int:
    if "前天" in evidence:
        return source_at_ms - 2 * 24 * 60 * 60 * 1000
    if "昨天" in evidence:
        return source_at_ms - 24 * 60 * 60 * 1000
    return source_at_ms


def _deterministic_updates(user_text: str) -> list[dict]:
    """Catch explicit high-value state transitions when the fact LLM misses them."""
    from tsuzuri.fact_validity import direct_text
    text = _one_line(direct_text(user_text), 1600)
    if not text:
        return []
    updates: list[dict] = []

    vacation_patterns = (
        ("summer_vacation", re.compile(r"(?:我|我们)[^。！？]{0,12}(?:已经|正式|现在|今天|刚)?放暑假(?:了|啦|咯)?")),
        ("winter_vacation", re.compile(r"(?:我|我们)[^。！？]{0,12}(?:已经|正式|现在|今天|刚)?放寒假(?:了|啦|咯)?")),
        ("spring_vacation", re.compile(r"(?:我|我们)[^。！？]{0,12}(?:已经|正式|现在|今天|刚)?放春假(?:了|啦|咯)?")),
        ("in_term", re.compile(r"(?:我|我们)[^。！？]{0,12}(?:已经|今天|刚|正式)?开学(?:了|啦)?")),
    )
    for value, pattern in vacation_patterns:
        match = pattern.search(text)
        if not match:
            continue
        evidence = match.group(0)
        if re.search(r'不想|不愿|不希望|可不想', evidence):
            continue
        phase_future = bool(re.search(r'(?:下周|明天|后天|快要|准备|打算|还没)[^，,。]{0,8}(?:开学|放.{0,1}假)', evidence))
        updates.append({
            "predicate": "academic.phase",
            "value": value,
            "operation": "set",
            "temporal_status": "planned" if phase_future else "active",
            "evidence_span": evidence,
            "confidence": 1.0,
            "source": "deterministic",
        })
        break

    exam_patterns = (
        ("exam_finished", re.compile(r"(?:我|我们)[^。！？]{0,18}(?:考试[^。！？]{0,8}(?:结束|考完)|(?:考完|结束)[^。！？]{0,8}考试)")),
        ("exam_ongoing", re.compile(r"(?:我|我们)[^。！？]{0,18}(?:还在考试|考试还没结束|正在考试|处在考试周)")),
    )
    for value, pattern in exam_patterns:
        match = pattern.search(text)
        if match:
            evidence = match.group(0)
            updates.append({
                "predicate": "exam.status",
                "value": value,
                "operation": "set",
                "temporal_status": "active",
                "evidence_span": evidence,
                "confidence": 1.0,
                "source": "deterministic",
            })
            break
    # Only explicit self reports. Third-party, questions and quoted examples
    # cannot advance the developer's trip, even if the LLM misses the turn.
    for clause in re.split(r'[，,。；;！!\n]', text):
        clause = clause.strip()
        if re.fullmatch(r'(?:我)?(?:已经|刚刚|刚|现在)?(?:回到家|到家|回家)(?:了|啦|了呢)', clause):
            for predicate, value in (('travel.status', '已回到家'), ('location.current', '在家')):
                updates.append({'predicate': predicate, 'value': value, 'operation': 'set',
                                'temporal_status': 'active', 'evidence_span': clause,
                                'confidence': 1.0, 'source': 'deterministic'})
    return updates


def normalize_developer_state_updates(raw_updates: Any, user_entry: dict) -> list[dict]:
    from tsuzuri.fact_validity import evidence_is_direct, state_window
    user_text = str((user_entry or {}).get("content", "") or "")
    source_at_ms = _timestamp_ms(user_entry)
    candidates = raw_updates if isinstance(raw_updates, list) else []
    candidates = [*candidates, *_deterministic_updates(user_text)]
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        predicate = _one_line(raw.get("predicate"), 60)
        if predicate not in _PREDICATES:
            continue
        operation = _one_line(raw.get("operation"), 20).lower() or "set"
        if operation not in ("set", "clear"):
            continue
        evidence = _one_line(raw.get("evidence_span"), 240)
        # Model-authored state may only commit when its exact evidence is in
        # the developer's own message. Assistant inference never crosses this gate.
        if not evidence_is_direct(evidence, user_text) or re.search(r'[?？]', evidence):
            continue
        if re.search(r'我(?:朋友|同学|室友|妈妈|爸爸|妹妹|姐姐|弟弟|哥哥)', evidence):
            continue
        if predicate in ('workload.current', 'availability.current') and re.search(r'发完|做完|完成了|结束了', evidence):
            continue
        explicit_return = bool(re.fullmatch(r'(?:我)?(?:已经|刚刚|刚|现在)?(?:回到家|到家|回家)(?:了|啦|了呢)', evidence))
        if not (_FIRST_PERSON_RE.search(evidence) or evidence.startswith(("我", "我们", "本人")) or explicit_return):
            continue
        if predicate == 'travel.status' and operation == 'set' and not re.search(
                r'出发|出门|旅行|旅途|在路上|飞机|到家|外出|出远门|旅游|车上|到了|回到|出差|回家', evidence):
            continue
        value = _canonical_value(predicate, raw.get("value"))
        if operation == "set" and not value:
            continue
        if operation == "set" and not _value_allowed(predicate, value):
            continue
        temporal_status = _one_line(raw.get("temporal_status"), 20).lower() or "active"
        if temporal_status not in ("active", "planned"):
            temporal_status = "active"
        if predicate == 'academic.phase' and re.search(r'(?:刚|已经|正式)(?:开学|放.{0,1}假)', evidence):
            temporal_status = 'active'
        elif _FUTURE_MARKERS_RE.search(evidence) and not re.search(r"(?:已经|现在|目前|正在|刚刚|刚才)", evidence):
            temporal_status = "planned"
        valid_from_ms = _parse_date_ms(raw.get("valid_from")) or _relative_valid_from_ms(
            evidence,
            source_at_ms,
        )
        valid_until_ms = _parse_date_ms(raw.get("valid_until"), end_of_day=True)
        if not valid_until_ms:
            valid_until_ms = _default_valid_until_ms(predicate, valid_from_ms)
        semantic_start, semantic_end, future = state_window(predicate, evidence, source_at_ms)
        if semantic_start:
            valid_from_ms = semantic_start
        if semantic_end:
            valid_until_ms = min(valid_until_ms, semantic_end) if valid_until_ms else semantic_end
        if future:
            temporal_status = 'planned'
        key = (predicate, value, operation)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "predicate": predicate,
            "value": value,
            "operation": operation,
            "temporal_status": temporal_status,
            "valid_from_ms": valid_from_ms,
            "valid_until_ms": valid_until_ms,
            "evidence_span": evidence,
            "confidence": _clamp(raw.get("confidence"), 0.78),
            "source": _one_line(raw.get("source"), 40) or "turn_fact_llm",
        })
    # Deterministic candidates are appended last and should win dedupe when
    # the model used a less canonical value for the same predicate.
    by_predicate: dict[str, dict] = {}
    for item in out:
        current = by_predicate.get(item["predicate"])
        if current is None or float(item["confidence"]) >= float(current["confidence"]):
            by_predicate[item["predicate"]] = item
    return list(by_predicate.values())[:8]


def sanitize_persisted_state_updates(raw_updates: Any) -> list[dict]:
    """Validate already-evidenced updates while hydrating a persisted fact card."""
    rows = raw_updates if isinstance(raw_updates, list) else []
    out: list[dict] = []
    seen: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        predicate = _one_line(raw.get("predicate"), 60)
        operation = _one_line(raw.get("operation"), 20).lower() or "set"
        value = _canonical_value(predicate, raw.get("value"))
        evidence = _one_line(raw.get("evidence_span"), 240)
        if (
            predicate not in _PREDICATES
            or predicate in seen
            or operation not in ("set", "clear")
            or (operation == "set" and not value)
            or not evidence
        ):
            continue
        seen.add(predicate)
        out.append({
            "predicate": predicate,
            "value": value,
            "operation": operation,
            "temporal_status": (
                _one_line(raw.get("temporal_status"), 20)
                if _one_line(raw.get("temporal_status"), 20) in ("active", "planned")
                else "active"
            ),
            "valid_from_ms": int(raw.get("valid_from_ms", 0) or 0),
            "valid_until_ms": int(raw.get("valid_until_ms", 0) or 0),
            "evidence_span": evidence,
            "confidence": _clamp(raw.get("confidence"), 0.78),
            "source": _one_line(raw.get("source"), 40) or "turn_fact_llm",
        })
    return out[:8]


def ensure_schema(db_path: str | None = None) -> None:
    path = str(db_path or _default_db_path())
    with _SCHEMA_LOCK:
        if path in _SCHEMA_READY and Path(path).exists():
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with closing(connect_sqlite(path, timeout=8)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interlocutor_state_facts (
                    state_id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    knowledge_scope TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    value_text TEXT NOT NULL,
                    temporal_status TEXT NOT NULL DEFAULT 'active',
                    status TEXT NOT NULL DEFAULT 'active',
                    valid_from_ms INTEGER NOT NULL,
                    valid_until_ms INTEGER NOT NULL DEFAULT 0,
                    source_character TEXT NOT NULL DEFAULT '',
                    source_session_id TEXT NOT NULL DEFAULT '',
                    source_user_id TEXT NOT NULL DEFAULT '',
                    source_turn_id TEXT NOT NULL DEFAULT '',
                    source_at_ms INTEGER NOT NULL,
                    evidence_text TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.7,
                    authority TEXT NOT NULL DEFAULT 'explicit_user',
                    supersedes_id TEXT NOT NULL DEFAULT '',
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interlocutor_state_active "
                "ON interlocutor_state_facts(subject_id, knowledge_scope, status, predicate, valid_until_ms)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_interlocutor_state_source "
                "ON interlocutor_state_facts(source_user_id, source_turn_id)"
            )
            conn.commit()
        _SCHEMA_READY.add(path)


def _state_id(scope: str, source_user_id: str, predicate: str, value: str, operation: str) -> str:
    raw = f"{scope}|{source_user_id}|{predicate}|{value}|{operation}".encode("utf-8", errors="ignore")
    return "isf-" + hashlib.sha1(raw).hexdigest()[:20]


def commit_turn_state_updates(
    character: str,
    session_id: str,
    card: dict,
    *,
    db_path: str | None = None,
) -> list[dict]:
    if str((card or {}).get("audience_scope", "") or "") not in ("", _DEVELOPER_SCOPE):
        return []
    updates = card.get("developer_state_updates", []) if isinstance(card, dict) else []
    if not isinstance(updates, list) or not updates:
        return []
    path = str(db_path or _default_db_path())
    ensure_schema(path)
    char = _one_line(character or card.get("character"), 40) or "当前角色"
    knowledge_scope = f"character:{char}"
    source_user_id = _one_line(card.get("user_id"), 120)
    source_turn_id = _one_line(card.get("turn_id"), 120)
    source_at_ms = int(card.get("created_at_ms", 0) or int(time.time() * 1000))
    now_ms = int(time.time() * 1000)
    committed: list[dict] = []
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        for update in updates[:8]:
            if not isinstance(update, dict):
                continue
            predicate = _one_line(update.get("predicate"), 60)
            value = _canonical_value(predicate, update.get("value"))
            operation = _one_line(update.get("operation"), 20).lower() or "set"
            if predicate not in _PREDICATES or operation not in ("set", "clear"):
                continue
            if operation == "set" and not _value_allowed(predicate, value):
                continue
            temporal_status = _one_line(update.get("temporal_status"), 20) or "active"
            state_id = _state_id(knowledge_scope, source_user_id, predicate, value, operation)
            existing_same = conn.execute(
                "SELECT state_id FROM interlocutor_state_facts WHERE state_id=?",
                (state_id,),
            ).fetchone()
            if existing_same:
                continue
            if operation == "clear" or temporal_status == "active":
                target_statuses = ("active", "planned")
            else:
                target_statuses = ("planned",)
            status_placeholders = ",".join("?" for _ in target_statuses)
            # Out-of-order commits (cross-session or slow extraction task) must
            # not let an older claim supersede a newer one; equality keeps the
            # provisional-card -> LLM-card replacement of the same exchange working.
            active_rows = conn.execute(
                f"""
                SELECT state_id FROM interlocutor_state_facts
                WHERE subject_id=? AND knowledge_scope=? AND predicate=?
                  AND status IN ({status_placeholders})
                  AND source_at_ms <= ?
                  AND (status <> 'planned' OR valid_from_ms <= ? OR ? = 'clear' OR ? = 'planned')
                ORDER BY source_at_ms DESC
                """,
                (_SUBJECT_DEVELOPER, knowledge_scope, predicate, *target_statuses, source_at_ms,
                 source_at_ms, operation, temporal_status),
            ).fetchall()
            supersedes_id = str(active_rows[0]["state_id"] or "") if active_rows else ""
            if active_rows:
                conn.executemany(
                    "UPDATE interlocutor_state_facts SET status='superseded', updated_at_ms=? WHERE state_id=?",
                    [(now_ms, str(row["state_id"])) for row in active_rows],
                )
            if operation == "clear":
                conn.execute(
                    """
                    INSERT INTO interlocutor_state_facts(
                        state_id, subject_id, knowledge_scope, predicate, value_text,
                        temporal_status, status, valid_from_ms, valid_until_ms,
                        source_character, source_session_id, source_user_id, source_turn_id,
                        source_at_ms, evidence_text, confidence, authority, supersedes_id,
                        created_at_ms, updated_at_ms
                    ) VALUES (?, ?, ?, ?, '', 'active', 'cleared', ?, 0, ?, ?, ?, ?, ?, ?, ?, 'explicit_user', ?, ?, ?)
                    """,
                    (
                        state_id,
                        _SUBJECT_DEVELOPER,
                        knowledge_scope,
                        predicate,
                        source_at_ms,
                        char,
                        _one_line(session_id, 180),
                        source_user_id,
                        source_turn_id,
                        source_at_ms,
                        _one_line(update.get("evidence_span"), 240),
                        _clamp(update.get("confidence"), 0.78),
                        supersedes_id,
                        now_ms,
                        now_ms,
                    ),
                )
                committed.append({"state_id": state_id, "predicate": predicate, "operation": "clear", "supersedes_id": supersedes_id})
                continue
            valid_from_ms = int(update.get("valid_from_ms", 0) or source_at_ms)
            valid_until_ms = int(update.get("valid_until_ms", 0) or _default_valid_until_ms(predicate, valid_from_ms))
            row_status = "planned" if temporal_status == "planned" else "active"
            conn.execute(
                """
                INSERT INTO interlocutor_state_facts(
                    state_id, subject_id, knowledge_scope, predicate, value_text,
                    temporal_status, status, valid_from_ms, valid_until_ms,
                    source_character, source_session_id, source_user_id, source_turn_id,
                    source_at_ms, evidence_text, confidence, authority, supersedes_id,
                    created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state_id,
                    _SUBJECT_DEVELOPER,
                    knowledge_scope,
                    predicate,
                    value,
                    temporal_status,
                    row_status,
                    valid_from_ms,
                    valid_until_ms,
                    char,
                    _one_line(session_id, 180),
                    source_user_id,
                    source_turn_id,
                    source_at_ms,
                    _one_line(update.get("evidence_span"), 240),
                    _clamp(update.get("confidence"), 0.78),
                    "explicit_user",
                    supersedes_id,
                    now_ms,
                    now_ms,
                ),
            )
            committed.append({"state_id": state_id, **dict(update)})
        conn.commit()
    return committed


def _materialize_expired(conn: sqlite3.Connection, scope: str, now: int) -> None:
    conn.execute(
        """
        UPDATE interlocutor_state_facts
        SET status='expired', updated_at_ms=?
        WHERE subject_id=? AND knowledge_scope=? AND status IN ('active', 'planned')
          AND valid_until_ms > 0 AND valid_until_ms < ?
        """,
        (now, _SUBJECT_DEVELOPER, scope, now),
    )


def _promote_due_planned(conn: sqlite3.Connection, scope: str, now: int) -> None:
    # A planned row becomes current only when its explicit effective time has
    # arrived. Rows whose valid_from simply defaulted to the source message
    # time stay planned; promoting them would turn "我下周放暑假" into today's
    # state.
    conn.execute(
        """
        UPDATE interlocutor_state_facts
        SET status='active', updated_at_ms=?
        WHERE subject_id=? AND knowledge_scope=? AND status='planned'
          AND predicate IN ('academic.phase', 'exam.status')
          AND valid_from_ms <> source_at_ms AND valid_from_ms <= ?
          AND (valid_until_ms=0 OR valid_until_ms>=?)
        """,
        (now, _SUBJECT_DEVELOPER, scope, now, now),
    )


def _collapse_actives_to_latest_claim(conn: sqlite3.Connection, scope: str, now: int) -> None:
    # At most one active row per predicate: the claim with the newest source
    # wins regardless of commit order. Read-path backstop for cross-session or
    # out-of-order extraction commits.
    rows = conn.execute(
        """
        SELECT state_id, predicate FROM interlocutor_state_facts
        WHERE subject_id=? AND knowledge_scope=? AND status='active'
        ORDER BY source_at_ms DESC, confidence DESC
        """,
        (_SUBJECT_DEVELOPER, scope),
    ).fetchall()
    seen: set[str] = set()
    losers: list[str] = []
    for row in rows:
        predicate = str(row["predicate"] or "")
        if predicate in seen:
            losers.append(str(row["state_id"]))
        else:
            seen.add(predicate)
    if losers:
        conn.executemany(
            "UPDATE interlocutor_state_facts SET status='superseded', updated_at_ms=? WHERE state_id=?",
            [(now, state_id) for state_id in losers],
        )


def load_active_states(
    character: str,
    *,
    now_ms: int | None = None,
    db_path: str | None = None,
) -> list[dict]:
    path = str(db_path or _default_db_path())
    if not Path(path).exists():
        return []
    ensure_schema(path)
    now = int(now_ms or now_world().timestamp() * 1000)
    scope = f"character:{_one_line(character, 40) or '当前角色'}"
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        _materialize_expired(conn, scope, now)
        _promote_due_planned(conn, scope, now)
        _collapse_actives_to_latest_claim(conn, scope, now)
        rows = conn.execute(
            """
            SELECT * FROM interlocutor_state_facts
            WHERE subject_id=? AND knowledge_scope=? AND status='active'
              AND valid_from_ms <= ? AND (valid_until_ms=0 OR valid_until_ms >= ?)
            ORDER BY source_at_ms DESC, confidence DESC
            LIMIT ?
            """,
            (_SUBJECT_DEVELOPER, scope, now, now, _MAX_ACTIVE_ROWS),
        ).fetchall()
        conn.commit()
    from tsuzuri.fact_validity import project_states
    return project_states([dict(row) for row in rows], character, now)['active']


def load_planned_states(
    character: str,
    *,
    now_ms: int | None = None,
    db_path: str | None = None,
    limit: int = 4,
) -> list[dict]:
    """Confirmed-but-not-yet-effective states for the frame's planned section."""
    path = str(db_path or _default_db_path())
    if not Path(path).exists():
        return []
    ensure_schema(path)
    now = int(now_ms or now_world().timestamp() * 1000)
    scope = f"character:{_one_line(character, 40) or '当前角色'}"
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        _materialize_expired(conn, scope, now)
        _promote_due_planned(conn, scope, now)
        rows = conn.execute(
            """
            SELECT * FROM interlocutor_state_facts
            WHERE subject_id=? AND knowledge_scope=?
              AND (status='planned' OR (status='active' AND temporal_status='planned'))
              AND (valid_until_ms=0 OR valid_until_ms>=?)
            ORDER BY valid_from_ms ASC, source_at_ms DESC
            LIMIT ?
            """,
            (_SUBJECT_DEVELOPER, scope, now, _MAX_ACTIVE_ROWS),
        ).fetchall()
        conn.commit()
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        predicate = str(row["predicate"] or "")
        if predicate in seen:
            continue
        seen.add(predicate)
        out.append(dict(row))
        if len(out) >= max(1, int(limit or 4)):
            break
    from tsuzuri.fact_validity import project_states
    return project_states({'planned': out}, character, now)['planned']


def archive_states_for_source_message_ids(
    message_ids: Iterable[str],
    *,
    db_path: str | None = None,
) -> int:
    ids = [_one_line(value, 120) for value in message_ids if _one_line(value, 120)]
    path = str(db_path or _default_db_path())
    if not ids or not Path(path).exists():
        return 0
    ensure_schema(path)
    now_ms = int(time.time() * 1000)
    placeholders = ",".join("?" for _ in ids)
    with closing(connect_sqlite(path, timeout=8)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        removed = conn.execute(
            f"SELECT state_id, subject_id, knowledge_scope, predicate, supersedes_id "
            f"FROM interlocutor_state_facts WHERE source_user_id IN ({placeholders}) "
            "AND status IN ('active', 'planned', 'cleared')",
            ids,
        ).fetchall()
        cursor = conn.execute(
            f"UPDATE interlocutor_state_facts SET status='archived', updated_at_ms=? "
            f"WHERE source_user_id IN ({placeholders}) AND status IN ('active', 'planned', 'cleared')",
            (now_ms, *ids),
        )
        for row in removed:
            predecessor_id = str(row["supersedes_id"] or "")
            if not predecessor_id:
                continue
            has_newer = conn.execute(
                """
                SELECT 1 FROM interlocutor_state_facts
                WHERE subject_id=? AND knowledge_scope=? AND predicate=?
                  AND status IN ('active', 'planned') AND state_id<>?
                LIMIT 1
                """,
                (row["subject_id"], row["knowledge_scope"], row["predicate"], row["state_id"]),
            ).fetchone()
            if has_newer:
                continue
            conn.execute(
                """
                UPDATE interlocutor_state_facts
                SET status=CASE WHEN temporal_status='planned' AND valid_from_ms>? THEN 'planned' ELSE 'active' END,
                    updated_at_ms=?
                WHERE state_id=? AND status='superseded'
                  AND (valid_until_ms=0 OR valid_until_ms>=?)
                """,
                (now_ms, now_ms, predecessor_id, now_ms),
            )
        conn.commit()
        return int(cursor.rowcount or 0)


def compile_interlocutor_state_frame(
    character: str,
    records: list[dict] | dict | None,
    *,
    interlocutor_label: str = "",
) -> dict:
    peer = _one_line(interlocutor_label, 40) or user_name()
    if isinstance(records, dict):
        active_records = records.get("active", [])
        planned_records = records.get("planned", [])
    else:
        active_records = records
        planned_records = []
    rows = [dict(row) for row in (active_records or []) if isinstance(row, dict)]
    planned_rows = [dict(row) for row in (planned_records or []) if isinstance(row, dict)]
    lines = [
        '<interlocutor-state-frame subject="developer" mandatory="true">',
        f"【{peer}当前持续状态】这些是仍然生效的本人状态，不是需要关键词命中的旧回忆。",
    ]
    digest: list[str] = []
    seen_predicates: set[str] = set()
    for row in rows:
        predicate = _one_line(row.get("predicate"), 60)
        if predicate in seen_predicates or predicate not in _PREDICATES:
            continue
        seen_predicates.add(predicate)
        value = _canonical_value(predicate, row.get("value_text") or row.get("value"))
        rendered = _VALUE_LABELS.get(value, value)
        label = str(_PREDICATES[predicate]["label"])
        source_char = _one_line(row.get("source_character"), 40) or character
        source_time = _format_world(int(row.get("source_at_ms", 0) or 0))
        valid_from_time = _format_world(int(row.get("valid_from_ms", 0) or 0))
        source = f"{peer}于{source_time}直接告诉{source_char}" if source_time else f"{peer}直接告诉{source_char}"
        effective = (
            f"；生效时间：{valid_from_time}"
            if valid_from_time and valid_from_time[:10] != source_time[:10]
            else ""
        )
        lines.append(f"- {label}：{rendered}（来源：{source}{effective}）")
        digest.append(f"{peer}{rendered}")
    if not seen_predicates:
        lines.append("- 当前没有已确认且仍生效的持续状态；不要据此猜测对方正在上课、放假、考试或出行。")
    planned_count = 0
    if planned_rows:
        planned_section: list[str] = []
        seen_planned: set[str] = set()
        for row in planned_rows:
            predicate = _one_line(row.get("predicate"), 60)
            if predicate in seen_planned or predicate not in _PREDICATES:
                continue
            seen_planned.add(predicate)
            value = _canonical_value(predicate, row.get("value_text") or row.get("value"))
            rendered = _VALUE_LABELS.get(value, value)
            label = str(_PREDICATES[predicate]["label"])
            source_char = _one_line(row.get("source_character"), 40) or character
            source_time = _format_world(int(row.get("source_at_ms", 0) or 0))
            source_at_ms = int(row.get("source_at_ms", 0) or 0)
            valid_from_ms = int(row.get("valid_from_ms", 0) or 0)
            source = f"{peer}于{source_time}直接告诉{source_char}" if source_time else f"{peer}直接告诉{source_char}"
            eta = ""
            if valid_from_ms > source_at_ms + 3_600_000:
                eta_time = _format_world(valid_from_ms)
                if eta_time:
                    eta = f"；预计生效：{eta_time}"
            planned_section.append(f"- {label}：{rendered}（来源：{source}{eta}）")
            planned_count += 1
            if planned_count >= 4:
                break
        if planned_section:
            lines.append(
                f"【已确认但尚未生效的计划】{peer}亲口提过、尚未到生效时间；"
                "只能当作将来的计划，不得说成当前状态。"
            )
            lines.extend(planned_section)
    lines.extend(
        [
            "【使用边界】持续状态优先于按需旧记忆；本轮用户的新纠正优先于这里的旧投影。",
            "不要向对方复述数据库式字段或来源说明，只把这些状态当作自然认识对方的一部分。",
            "</interlocutor-state-frame>",
        ]
    )
    return {
        "text": "\n".join(lines),
        "state_digest": digest[:6],
        "record_count": len(seen_predicates),
        "planned_count": planned_count,
    }


__all__ = [
    "archive_states_for_source_message_ids",
    "commit_turn_state_updates",
    "compile_interlocutor_state_frame",
    "ensure_schema",
    "load_active_states",
    "load_planned_states",
    "normalize_developer_state_updates",
    "sanitize_persisted_state_updates",
]
