from __future__ import annotations

import builtins
import copy
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class ToolRunRecord:
    tool_name: str
    status: str = "pending"
    read_only: bool = False
    started_at: float = 0.0
    completed_at: float = 0.0
    meta: dict[str, object] = field(default_factory=dict)
    result_excerpt: str = ""
    reason_code: str = ""


@dataclass
class TurnRuntimeTrace:
    turn_id: str = ""
    channel_kind: str = ""
    character: str = ""
    prompt_summary: dict[str, object] = field(default_factory=dict)
    projection_summary: dict[str, object] = field(default_factory=dict)
    tool_plan: dict[str, object] = field(default_factory=dict)
    delivery_plan: dict[str, object] = field(default_factory=dict)
    telemetry: dict[str, object] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0


@dataclass
class RuntimeSessionState:
    session_id: str
    channel_kind: str = ""
    character: str = ""
    mode_flags: dict[str, object] = field(default_factory=dict)
    pending_tool_outputs: list[dict[str, object]] = field(default_factory=list)
    active_tool_runs: list[ToolRunRecord] = field(default_factory=list)
    last_prompt_summary: dict[str, object] = field(default_factory=dict)
    last_projection_summary: dict[str, object] = field(default_factory=dict)
    last_agent_summary: dict[str, object] = field(default_factory=dict)
    last_turn_trace: TurnRuntimeTrace | None = None


class RuntimeStateStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._legacy_globals: dict[str, object] = {}
        self._sessions: dict[str, RuntimeSessionState] = {}

    @property
    def legacy_globals(self) -> dict[str, object]:
        return self._legacy_globals

    def adopt_legacy_globals(self, values: dict[str, object] | None = None) -> None:
        if not isinstance(values, dict):
            return
        with self._lock:
            for key, value in values.items():
                if key not in self._legacy_globals:
                    self._legacy_globals[key] = value

    def get_legacy_value(self, name: str, default=None):
        key = str(name or "").strip()
        with self._lock:
            if key not in self._legacy_globals:
                self._legacy_globals[key] = default
            return self._legacy_globals.get(key)

    def set_legacy_value(self, name: str, value):
        key = str(name or "").strip()
        with self._lock:
            self._legacy_globals[key] = value
            return value

    def ensure_session(
        self,
        session_id: str,
        *,
        channel_kind: str = "",
        character: str = "",
        mode_flags: dict[str, object] | None = None,
    ) -> RuntimeSessionState:
        session_key = str(session_id or "default").strip() or "default"
        with self._lock:
            state = self._sessions.get(session_key)
            if state is None:
                state = RuntimeSessionState(session_id=session_key)
                self._sessions[session_key] = state
            if channel_kind:
                state.channel_kind = str(channel_kind or "").strip()
            if character:
                state.character = str(character or "").strip()
            if isinstance(mode_flags, dict) and mode_flags:
                merged = dict(state.mode_flags)
                merged.update(copy.deepcopy(mode_flags))
                state.mode_flags = merged
            return state

    def get_session(self, session_id: str) -> RuntimeSessionState | None:
        session_key = str(session_id or "").strip()
        if not session_key:
            return None
        with self._lock:
            return self._sessions.get(session_key)

    def list_sessions(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._sessions.keys()))

    def set_prompt_summary(self, session_id: str, summary: dict | None = None) -> dict[str, object]:
        state = self.ensure_session(session_id)
        payload = dict(copy.deepcopy(summary) if isinstance(summary, dict) else {})
        with self._lock:
            state.last_prompt_summary = payload
        return payload

    def set_projection_summary(self, session_id: str, summary: dict | None = None) -> dict[str, object]:
        state = self.ensure_session(session_id)
        payload = dict(copy.deepcopy(summary) if isinstance(summary, dict) else {})
        with self._lock:
            state.last_projection_summary = payload
        return payload

    def set_agent_summary(self, session_id: str, summary: dict | None = None) -> dict[str, object]:
        state = self.ensure_session(session_id)
        payload = dict(copy.deepcopy(summary) if isinstance(summary, dict) else {})
        with self._lock:
            state.last_agent_summary = payload
        return payload

    def append_pending_tool_outputs(
        self,
        session_id: str,
        outputs: list[dict[str, object]] | tuple[dict[str, object], ...] | None = None,
    ) -> list[dict[str, object]]:
        state = self.ensure_session(session_id)
        rows = [copy.deepcopy(row) for row in (outputs or []) if isinstance(row, dict)]
        if not rows:
            return []
        with self._lock:
            state.pending_tool_outputs.extend(rows)
        return rows

    def consume_pending_tool_outputs(self, session_id: str) -> list[dict[str, object]]:
        state = self.ensure_session(session_id)
        with self._lock:
            rows = [copy.deepcopy(row) for row in state.pending_tool_outputs if isinstance(row, dict)]
            state.pending_tool_outputs = []
        return rows

    def peek_pending_tool_outputs(self, session_id: str) -> list[dict[str, object]]:
        state = self.ensure_session(session_id)
        with self._lock:
            return [copy.deepcopy(row) for row in state.pending_tool_outputs if isinstance(row, dict)]

    def start_tool_run(
        self,
        session_id: str,
        tool_name: str,
        *,
        read_only: bool = False,
        meta: dict[str, object] | None = None,
    ) -> ToolRunRecord:
        state = self.ensure_session(session_id)
        record = ToolRunRecord(
            tool_name=str(tool_name or "").strip(),
            status="in_progress",
            read_only=bool(read_only),
            started_at=time.time(),
            meta=dict(copy.deepcopy(meta) if isinstance(meta, dict) else {}),
        )
        with self._lock:
            state.active_tool_runs.append(record)
        return record

    def finish_tool_run(
        self,
        session_id: str,
        tool_name: str,
        *,
        ok: bool,
        result_text: str = "",
        meta: dict[str, object] | None = None,
        reason_code: str = "",
    ) -> ToolRunRecord:
        state = self.ensure_session(session_id)
        payload = dict(copy.deepcopy(meta) if isinstance(meta, dict) else {})
        key = str(tool_name or "").strip()
        with self._lock:
            target: ToolRunRecord | None = None
            for record in reversed(state.active_tool_runs):
                if record.tool_name == key and record.completed_at <= 0:
                    target = record
                    break
            if target is None:
                target = ToolRunRecord(tool_name=key, started_at=time.time())
                state.active_tool_runs.append(target)
            target.completed_at = time.time()
            target.result_excerpt = str(result_text or "").strip()[:200]
            target.reason_code = str(reason_code or "").strip()
            if payload:
                merged = dict(target.meta)
                merged.update(payload)
                target.meta = merged
            if ok:
                target.status = "completed"
            elif target.reason_code:
                target.status = "denied"
            else:
                target.status = "failed"
            return target

    def record_turn_trace(self, session_id: str, trace: TurnRuntimeTrace | dict | None = None) -> TurnRuntimeTrace:
        state = self.ensure_session(session_id)
        if isinstance(trace, TurnRuntimeTrace):
            payload = copy.deepcopy(trace)
        elif isinstance(trace, dict):
            payload = TurnRuntimeTrace(
                turn_id=str(trace.get("turn_id", "") or ""),
                channel_kind=str(trace.get("channel_kind", "") or ""),
                character=str(trace.get("character", "") or ""),
                prompt_summary=dict(trace.get("prompt_summary", {}) if isinstance(trace.get("prompt_summary", {}), dict) else {}),
                projection_summary=dict(trace.get("projection_summary", {}) if isinstance(trace.get("projection_summary", {}), dict) else {}),
                tool_plan=dict(trace.get("tool_plan", {}) if isinstance(trace.get("tool_plan", {}), dict) else {}),
                delivery_plan=dict(trace.get("delivery_plan", {}) if isinstance(trace.get("delivery_plan", {}), dict) else {}),
                telemetry=dict(trace.get("telemetry", {}) if isinstance(trace.get("telemetry", {}), dict) else {}),
                started_at=float(trace.get("started_at", 0.0) or 0.0),
                completed_at=float(trace.get("completed_at", 0.0) or 0.0),
            )
        else:
            payload = TurnRuntimeTrace()
        with self._lock:
            state.last_turn_trace = payload
        return payload

    def snapshot(self, session_id: str) -> dict[str, object]:
        state = self.ensure_session(session_id)
        with self._lock:
            return asdict(copy.deepcopy(state))

    def clear(self) -> None:
        with self._lock:
            self._legacy_globals.clear()
            self._sessions.clear()


def get_runtime_state_store() -> RuntimeStateStore:
    store = getattr(builtins, "_tsuzuri_runtime_state_store", None)
    legacy_seed = getattr(builtins, "_tsuzuri_runtime_state", None)
    if not isinstance(store, RuntimeStateStore):
        store = RuntimeStateStore()
        if isinstance(legacy_seed, dict):
            store.adopt_legacy_globals(legacy_seed)
        setattr(builtins, "_tsuzuri_runtime_state_store", store)
    elif isinstance(legacy_seed, dict):
        store.adopt_legacy_globals(legacy_seed)
    setattr(builtins, "_tsuzuri_runtime_state", store.legacy_globals)
    return store


__all__ = [
    "RuntimeSessionState",
    "RuntimeStateStore",
    "ToolRunRecord",
    "TurnRuntimeTrace",
    "get_runtime_state_store",
]
