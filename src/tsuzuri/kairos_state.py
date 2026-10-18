from __future__ import annotations

from tsuzuri.time_source import world_tz_label

import builtins
import copy
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _now_ts() -> float:
    return float(time.time())


def _coerce_optional_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in ("1", "true", "yes", "on", "enabled", "enable"):
        return True
    if text in ("0", "false", "no", "off", "disabled", "disable"):
        return False
    return None


def is_kairos_default_enabled() -> bool:
    env_value = _coerce_optional_bool(os.getenv("TSUZURI_KAIROS_ENABLE"))
    if env_value is None:
        return True
    return bool(env_value)


def resolve_kairos_switches(*sources) -> dict[str, bool]:
    kairos_enabled = bool(is_kairos_default_enabled())
    brief_enabled = bool(kairos_enabled)
    brief_explicit = False
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        kairos_value = None
        for key in ("kairos_enabled", "enable_kairos", "mode_kairos_enabled"):
            kairos_value = _coerce_optional_bool(source.get(key))
            if kairos_value is not None:
                break
        brief_value = None
        for key in ("brief_enabled", "kairos_brief_enabled", "enable_kairos_brief"):
            brief_value = _coerce_optional_bool(source.get(key))
            if brief_value is not None:
                break
        if kairos_value is not None:
            kairos_enabled = bool(kairos_value)
            if not brief_explicit:
                brief_enabled = bool(kairos_value)
        if brief_value is not None:
            brief_enabled = bool(brief_value)
            brief_explicit = True
    if not kairos_enabled:
        brief_enabled = False
    return {
        "kairos_enabled": bool(kairos_enabled),
        "brief_enabled": bool(brief_enabled),
    }


@dataclass
class KairosBriefSnapshot:
    brief_text: str = ""
    summary: dict[str, object] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    source_usage: dict[str, object] = field(default_factory=dict)
    created_at: float = 0.0


@dataclass
class KairosDreamRecord:
    ok: bool = False
    summary: dict[str, object] = field(default_factory=dict)
    domain_writes: list[str] = field(default_factory=list)
    meta_writes: list[str] = field(default_factory=list)
    refreshed_brief: dict[str, object] = field(default_factory=dict)
    created_at: float = 0.0
    turn_count: int = 0


@dataclass
class KairosSessionState:
    session_id: str
    channel_kind: str = ""
    character: str = ""
    kairos_enabled: bool = False
    brief_enabled: bool = False
    last_brief: KairosBriefSnapshot | None = None
    last_brief_summary: dict[str, object] = field(default_factory=dict)
    last_dream_record: KairosDreamRecord | None = None
    turn_count: int = 0
    last_user_turn_at: float = 0.0
    last_brief_at: float = 0.0
    last_dream_at: float = 0.0
    hydrated_from_persistent: bool = False


class KairosStateStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, KairosSessionState] = {}
        self._active_dream_sessions: set[str] = set()
        self._scheduler_state: dict[str, object] = {
            "alive": False,
            "interval_sec": 600,
            "timezone": world_tz_label(),
            "window_start": "01:00",
            "window_end": "05:00",
            "last_tick_at": 0.0,
            "last_run_at": 0.0,
            "last_error": "",
        }

    def ensure_session(
        self,
        session_id: str,
        *,
        channel_kind: str = "",
        character: str = "",
        kairos_enabled: bool | None = None,
        brief_enabled: bool | None = None,
        bump_turn: bool = False,
    ) -> KairosSessionState:
        session_key = str(session_id or "default").strip() or "default"
        with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                state = KairosSessionState(session_id=session_key)
                self._sessions[session_key] = state
            if channel_kind:
                state.channel_kind = str(channel_kind or "").strip()
            if character:
                state.character = str(character or "").strip()
            if kairos_enabled is not None:
                state.kairos_enabled = bool(kairos_enabled)
            if brief_enabled is not None:
                state.brief_enabled = bool(brief_enabled)
            if bump_turn:
                state.turn_count = int(state.turn_count or 0) + 1
                state.last_user_turn_at = _now_ts()
            return state

    def get_session(self, session_id: str) -> KairosSessionState | None:
        session_key = str(session_id or "").strip()
        if not session_key:
            return None
        with self._lock:
            return self._sessions.get(session_key)

    def record_brief(self, session_id: str, snapshot: KairosBriefSnapshot | dict | None = None) -> KairosBriefSnapshot:
        state = self.ensure_session(session_id)
        payload = _coerce_brief_snapshot(snapshot)
        if payload.created_at <= 0:
            payload.created_at = _now_ts()
        with self._lock:
            state.last_brief = copy.deepcopy(payload)
            state.last_brief_summary = dict(payload.summary)
            state.last_brief_at = float(payload.created_at or _now_ts())
        return payload

    def record_dream(self, session_id: str, record: KairosDreamRecord | dict | None = None) -> KairosDreamRecord:
        state = self.ensure_session(session_id)
        payload = _coerce_dream_record(record)
        if payload.created_at <= 0:
            payload.created_at = _now_ts()
        with self._lock:
            state.last_dream_record = copy.deepcopy(payload)
            state.last_dream_at = float(payload.created_at or _now_ts())
            refreshed = payload.refreshed_brief if isinstance(payload.refreshed_brief, dict) else {}
            if refreshed:
                brief_snapshot = _coerce_brief_snapshot(refreshed)
                if brief_snapshot.created_at <= 0:
                    brief_snapshot.created_at = state.last_dream_at
                state.last_brief = copy.deepcopy(brief_snapshot)
                state.last_brief_summary = dict(brief_snapshot.summary)
                state.last_brief_at = float(brief_snapshot.created_at or state.last_dream_at)
        return payload

    def set_scheduler_state(self, **fields) -> dict[str, object]:
        payload = {str(k or "").strip(): copy.deepcopy(v) for k, v in fields.items() if str(k or "").strip()}
        with self._lock:
            self._scheduler_state.update(payload)
            return dict(copy.deepcopy(self._scheduler_state))

    def describe_scheduler_state(self) -> dict[str, object]:
        with self._lock:
            return dict(copy.deepcopy(self._scheduler_state))

    def mark_dream_active(self, session_id: str) -> bool:
        session_key = str(session_id or "").strip()
        if not session_key:
            return False
        with self._lock:
            if session_key in self._active_dream_sessions:
                return False
            self._active_dream_sessions.add(session_key)
        return True

    def mark_dream_finished(self, session_id: str) -> None:
        session_key = str(session_id or "").strip()
        if not session_key:
            return
        with self._lock:
            self._active_dream_sessions.discard(session_key)

    def is_dream_active(self, session_id: str) -> bool:
        session_key = str(session_id or "").strip()
        if not session_key:
            return False
        with self._lock:
            return session_key in self._active_dream_sessions

    def list_session_snapshots(self) -> list[dict[str, object]]:
        with self._lock:
            return [asdict(copy.deepcopy(self._sessions[key])) for key in sorted(self._sessions.keys())]

    def list_recent_dreams(self, limit: int = 10) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for snapshot in self.list_session_snapshots():
            record = snapshot.get("last_dream_record", {})
            if not isinstance(record, dict) or not record:
                continue
            rows.append(
                {
                    "session_id": str(snapshot.get("session_id", "") or ""),
                    "channel_kind": str(snapshot.get("channel_kind", "") or ""),
                    "character": str(snapshot.get("character", "") or ""),
                    "created_at": float(record.get("created_at", 0.0) or 0.0),
                    "ok": bool(record.get("ok", False)),
                    "summary": dict(record.get("summary", {}) if isinstance(record.get("summary", {}), dict) else {}),
                }
            )
        rows.sort(key=lambda row: float(row.get("created_at", 0.0) or 0.0), reverse=True)
        return rows[: max(1, int(limit or 10))]

    def hydrate_from_persistent(self, core=None, *, max_sessions: int = 80) -> dict[str, object]:
        """Rebuild lightweight display state from durable KAIROS DB rows."""
        payload = _load_persistent_kairos_payload(core)
        sessions = payload.get("sessions", {}) if isinstance(payload.get("sessions", {}), dict) else {}
        hydrated = 0
        with self._lock:
            for session_id, data in list(sessions.items())[: max(1, int(max_sessions or 80))]:
                if not isinstance(data, dict):
                    continue
                sid = str(data.get("session_id", "") or session_id or "").strip()
                if not sid:
                    continue
                state = self._sessions.get(sid)
                if state is None:
                    state = KairosSessionState(session_id=sid)
                    self._sessions[sid] = state
                channel_kind = str(data.get("channel_kind", "") or "").strip()
                character = str(data.get("character", "") or "").strip()
                if channel_kind:
                    state.channel_kind = channel_kind
                if character:
                    state.character = character
                state.kairos_enabled = bool(data.get("kairos_enabled", state.kairos_enabled or is_kairos_default_enabled()))
                state.brief_enabled = bool(data.get("brief_enabled", state.brief_enabled or state.kairos_enabled))
                state.turn_count = max(int(state.turn_count or 0), int(data.get("turn_count", 0) or 0))
                dream_ts = _coerce_ts(data.get("last_dream_at", 0.0))
                if dream_ts > float(state.last_dream_at or 0.0):
                    state.last_dream_at = dream_ts
                    state.last_dream_record = KairosDreamRecord(
                        ok=True,
                        summary=dict(data.get("dream_summary", {}) if isinstance(data.get("dream_summary", {}), dict) else {"reason_code": "hydrated_from_persistent"}),
                        created_at=dream_ts,
                        turn_count=state.turn_count,
                    )
                brief_value = data.get("durable_brief", {}) if isinstance(data.get("durable_brief", {}), dict) else {}
                brief_snapshot = _coerce_brief_snapshot(brief_value) if brief_value else None
                if brief_snapshot is not None:
                    if brief_snapshot.created_at <= 0:
                        brief_snapshot.created_at = _coerce_ts(data.get("updated_at", 0.0)) or dream_ts
                    if brief_snapshot.created_at >= float(state.last_brief_at or 0.0):
                        state.last_brief = copy.deepcopy(brief_snapshot)
                        state.last_brief_summary = dict(brief_snapshot.summary)
                        state.last_brief_at = float(brief_snapshot.created_at or 0.0)
                state.hydrated_from_persistent = True
                hydrated += 1
        return {"ok": True, "hydrated_sessions": hydrated, "persistent_sessions": len(sessions)}

    def snapshot(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
        with self._lock:
            return asdict(copy.deepcopy(state))

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._active_dream_sessions.clear()
            self._scheduler_state = {
                "alive": False,
                "interval_sec": 600,
                "timezone": world_tz_label(),
                "window_start": "01:00",
                "window_end": "05:00",
                "last_tick_at": 0.0,
                "last_run_at": 0.0,
                "last_error": "",
            }


def _coerce_brief_snapshot(value) -> KairosBriefSnapshot:
    if isinstance(value, KairosBriefSnapshot):
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return KairosBriefSnapshot(
            brief_text=str(value.get("brief_text", "") or ""),
            summary=dict(value.get("summary", {}) if isinstance(value.get("summary", {}), dict) else {}),
            sections={str(k): str(v or "") for k, v in (value.get("sections", {}) if isinstance(value.get("sections", {}), dict) else {}).items()},
            source_usage=dict(value.get("source_usage", {}) if isinstance(value.get("source_usage", {}), dict) else {}),
            created_at=float(value.get("created_at", 0.0) or 0.0),
        )
    return KairosBriefSnapshot(created_at=_now_ts())


def _coerce_dream_record(value) -> KairosDreamRecord:
    if isinstance(value, KairosDreamRecord):
        return copy.deepcopy(value)
    if isinstance(value, dict):
        return KairosDreamRecord(
            ok=bool(value.get("ok", False)),
            summary=dict(value.get("summary", {}) if isinstance(value.get("summary", {}), dict) else {}),
            domain_writes=[str(x or "") for x in (value.get("domain_writes", []) if isinstance(value.get("domain_writes", []), list) else []) if str(x or "").strip()],
            meta_writes=[str(x or "") for x in (value.get("meta_writes", []) if isinstance(value.get("meta_writes", []), list) else []) if str(x or "").strip()],
            refreshed_brief=dict(value.get("refreshed_brief", {}) if isinstance(value.get("refreshed_brief", {}), dict) else {}),
            created_at=float(value.get("created_at", 0.0) or 0.0),
            turn_count=int(value.get("turn_count", 0) or 0),
        )
    return KairosDreamRecord(created_at=_now_ts())


def _coerce_ts(value) -> float:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return 0.0
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        pass
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(dt.timestamp())
    except Exception:
        return 0.0


def _load_persistent_kairos_payload(core=None) -> dict[str, object]:
    target = core
    if target is None:
        try:
            from tsuzuri.memory_core import get_memory_core
            from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH
            target = get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)
        except Exception:
            target = None
    if target is None:
        return {"sessions": {}}
    rows: dict[str, dict[str, object]] = {}
    conn = None
    try:
        conn = target._connect()
        conn.row_factory = getattr(conn, "row_factory", None) or None
        try:
            domain_rows = conn.execute(
                "SELECT domain_key, domain_json FROM memory_domains WHERE domain_key LIKE 'kairos.session.%';"
            ).fetchall()
        except Exception:
            domain_rows = []
        try:
            meta_rows = conn.execute(
                "SELECT state_key, value_json FROM meta_state WHERE state_key LIKE 'kairos.session.%';"
            ).fetchall()
        except Exception:
            meta_rows = []
    except Exception:
        domain_rows = []
        meta_rows = []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _row_get(row, key_or_idx):
        try:
            return row[key_or_idx]
        except Exception:
            try:
                return row[int(key_or_idx)]
            except Exception:
                return ""

    for row in domain_rows:
        key = str(_row_get(row, "domain_key") or _row_get(row, 0) or "").strip()
        session_id = key.split("kairos.session.", 1)[-1].strip()
        if not session_id:
            continue
        data = _json_loads_local(str(_row_get(row, "domain_json") or _row_get(row, 1) or "{}"), {})
        if not isinstance(data, dict):
            data = {}
        item = rows.setdefault(session_id, {"session_id": session_id})
        item.update(data)
        item["session_id"] = str(data.get("session_id", "") or session_id)
        if isinstance(data.get("durable_brief", {}), dict):
            item["durable_brief"] = dict(data.get("durable_brief", {}))
    for row in meta_rows:
        key = str(_row_get(row, "state_key") or _row_get(row, 0) or "").strip()
        session_id = key.split("kairos.session.", 1)[-1].strip()
        if not session_id:
            continue
        data = _json_loads_local(str(_row_get(row, "value_json") or _row_get(row, 1) or "{}"), {})
        if not isinstance(data, dict):
            data = {}
        item = rows.setdefault(session_id, {"session_id": session_id})
        for field in ("channel_kind", "character", "last_dream_at", "turn_count"):
            if field in data and data.get(field) not in ("", None):
                item[field] = data.get(field)
        item.setdefault("kairos_enabled", True)
        item.setdefault("brief_enabled", True)
        if isinstance(data.get("signal_summary", {}), dict):
            item["dream_summary"] = {
                "reason_code": "hydrated_from_persistent",
                "signal_summary": dict(data.get("signal_summary", {})),
            }
    return {"sessions": rows}


def _json_loads_local(raw: str, default):
    try:
        return __import__("json").loads(str(raw or ""))
    except Exception:
        return default


def get_kairos_state_store() -> KairosStateStore:
    store = getattr(builtins, "_tsuzuri_kairos_state_store", None)
    if not isinstance(store, KairosStateStore):
        store = KairosStateStore()
        setattr(builtins, "_tsuzuri_kairos_state_store", store)
    return store


def ensure_kairos_session(
    session_id: str,
    *,
    channel_kind: str = "",
    character: str = "",
    kairos_enabled: bool | None = None,
    brief_enabled: bool | None = None,
    bump_turn: bool = False,
) -> KairosSessionState:
    return get_kairos_state_store().ensure_session(
        session_id,
        channel_kind=channel_kind,
        character=character,
        kairos_enabled=kairos_enabled,
        brief_enabled=brief_enabled,
        bump_turn=bump_turn,
    )


def record_kairos_brief(session_id: str, snapshot: KairosBriefSnapshot | dict | None = None) -> KairosBriefSnapshot:
    return get_kairos_state_store().record_brief(session_id, snapshot)


def record_kairos_dream(session_id: str, record: KairosDreamRecord | dict | None = None) -> KairosDreamRecord:
    return get_kairos_state_store().record_dream(session_id, record)


def snapshot_kairos_session(session_id: str) -> dict[str, object]:
    return get_kairos_state_store().snapshot(session_id)


def list_kairos_sessions() -> list[dict[str, object]]:
    return get_kairos_state_store().list_session_snapshots()


def list_recent_kairos_dreams(limit: int = 10) -> list[dict[str, object]]:
    return get_kairos_state_store().list_recent_dreams(limit=limit)


def hydrate_kairos_state_from_persistent(core=None) -> dict[str, object]:
    return get_kairos_state_store().hydrate_from_persistent(core)


def describe_kairos_scheduler_state() -> dict[str, object]:
    return get_kairos_state_store().describe_scheduler_state()


__all__ = [
    "KairosBriefSnapshot",
    "KairosDreamRecord",
    "KairosSessionState",
    "KairosStateStore",
    "describe_kairos_scheduler_state",
    "ensure_kairos_session",
    "get_kairos_state_store",
    "hydrate_kairos_state_from_persistent",
    "is_kairos_default_enabled",
    "list_kairos_sessions",
    "list_recent_kairos_dreams",
    "record_kairos_brief",
    "record_kairos_dream",
    "resolve_kairos_switches",
    "snapshot_kairos_session",
]
