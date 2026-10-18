"""High-level entry point: one object that remembers conversations for a cast of agents.

The lower layers are usable on their own (``tsuzuri.memory_core``,
``tsuzuri.turn_fact_ledger``, ``tsuzuri.turn_context_workspace`` …); this module wires
them into the turn loop most applications want:

    user message ─▶ validity review ─▶ recall + fact workspace ─▶ messages for your LLM ─▶ reply
                 ◀─ TurnFact card + memory write ◀──────────────────────────────────────┘

Everything runs offline by default (rule-based extraction, lexical recall, rule-based
self-correction). Pass an ``llm`` callable to enable LLM fact extraction and review, and
a ``reply`` callable to generate the agent's answers with your own model.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping

from tsuzuri import persona_registry
from tsuzuri.offline import offline_reply, offline_validity_review

TextLLM = Callable[[str], str]
ChatLLM = Callable[[list[dict]], str]

# Lower layers keep process-wide state (store singleton, cast registry, data-dir and
# embedding env, co-occurrence cache). Every public call takes this lock and re-binds
# that state to the calling engine, so several engines can share one process.
_PROCESS_LOCK = threading.RLock()
_ACTIVE: "weakref.ReferenceType[Tsuzuri] | None" = None
# With llm=None nothing may reach the network: these are hidden for the duration of a call.
_LLM_ENV = re.compile(r"^(?:LLM_|OPENAI_)|_API_KEY$|_BASE_URL$")
_HISTORICAL_QUERY = re.compile(r"还记得|回忆|以前|之前|当时|上次")


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class TurnResult:
    agent: str
    user_text: str
    reply: str
    messages: list[dict]
    fact_card: dict
    recalled: list[dict] = field(default_factory=list)
    memory_writes: dict = field(default_factory=dict)


class Tsuzuri:
    """Persistent, per-agent conversational memory with a context compiler.

    Parameters
    ----------
    data_dir:
        Where the SQLite memory store and session files live (created if missing).
        Defaults to ``$TSUZURI_HOME`` or ``./.tsuzuri``.
    personas:
        The cast: a mapping (see ``docs/configuration.md``) or a path to a JSON file.
        Defaults to the bundled demo cast.
    llm:
        ``prompt -> text`` callable used for fact extraction and fact-validity review.
        ``None`` keeps everything rule-based and offline (``LLM_*`` settings are ignored).
    reply:
        ``messages -> text`` callable that produces the agent's reply from the compiled
        messages. ``None`` uses a deterministic offline responder (useful for demos/tests).
    embeddings:
        Enable dense recall (requires the ``embeddings`` extra). Off by default.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike | None = None,
        *,
        personas: Mapping | str | os.PathLike | None = None,
        llm: TextLLM | None = None,
        reply: ChatLLM | None = None,
        embeddings: bool = False,
    ) -> None:
        root = os.fspath(data_dir or os.getenv("TSUZURI_HOME") or ".tsuzuri")
        os.makedirs(os.path.join(root, "sessions"), exist_ok=True)
        self.data_dir = os.path.abspath(root)
        self.embeddings = bool(embeddings)
        self.llm = llm
        self.reply_fn = reply
        self._lock = threading.RLock()
        with _PROCESS_LOCK:
            if personas is None:
                self._cast = persona_registry.export_personas()
            else:
                if isinstance(personas, Mapping):
                    persona_registry.configure_personas(personas, reset=True)
                else:
                    with open(os.fspath(personas), "r", encoding="utf-8") as fh:
                        persona_registry.configure_personas(json.load(fh), reset=True)
                self._cast = persona_registry.export_personas()
            self._set_env()
            from tsuzuri.memory_core import MemoryCoreStore

            self.store = MemoryCoreStore(
                db_path=os.path.join(self.data_dir, "memory.db"),
                snapshot_path=os.path.join(self.data_dir, "snapshot.json"),
                bootstrap_from_snapshot=False,
            )
            self._activate()

    # ── process-wide state ───────────────────────────────────────────

    def _set_env(self) -> None:
        os.environ["TSUZURI_HOME"] = self.data_dir
        os.environ["EMBEDDING_ENABLED"] = "1" if self.embeddings else "0"

    def _activate(self) -> None:
        """Point the process-wide state at this engine (caller holds ``_PROCESS_LOCK``)."""
        global _ACTIVE
        self._set_env()
        if _ACTIVE is not None and _ACTIVE() is self:
            return
        persona_registry.restore_personas(self._cast)
        from tsuzuri import memory_api
        from tsuzuri.memory_core_impl import recall_scoring, store as _store_module

        _store_module._MEMORY_CORE_SINGLETON = self.store
        memory_api._MEMORY_API_SINGLETON = None
        recall_scoring._CO_STRENGTH_CACHE.clear()
        recall_scoring._CO_STRENGTH_LOADED = False
        _ACTIVE = weakref.ref(self)

    @contextmanager
    def _bound(self) -> Iterator[None]:
        with _PROCESS_LOCK:
            self._activate()
            hidden: dict[str, str] = {}
            if self.llm is None:
                hidden = {k: v for k, v in os.environ.items() if _LLM_ENV.search(k)}
                for key in hidden:
                    os.environ.pop(key, None)
                os.environ["TSUZURI_OFFLINE"] = "1"
            try:
                yield
            finally:
                if self.llm is None:
                    os.environ.pop("TSUZURI_OFFLINE", None)
                    os.environ.update(hidden)

    # ── cast ─────────────────────────────────────────────────────────

    @property
    def agents(self) -> tuple[str, ...]:
        with self._bound():
            return persona_registry.agent_names()

    @property
    def user_name(self) -> str:
        with self._bound():
            return persona_registry.user_name()

    def _agent(self, agent: str | None) -> str:
        name = persona_registry.resolve_agent(agent)
        if not persona_registry.is_agent(name):
            raise ValueError(f"unknown agent {name!r}; known: {', '.join(persona_registry.agent_names())}")
        return name

    # ── sessions ─────────────────────────────────────────────────────

    def _session_path(self, key: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in key)
        return os.path.join(self.data_dir, "sessions", f"{safe or 'default'}.json")

    def load_session(self, key: str) -> dict:
        path = self._session_path(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            data = {}
        data.setdefault("history", [])
        data.setdefault("turn_fact_cards", [])
        return data

    def save_session(self, key: str, data: dict) -> None:
        path = self._session_path(key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=list)
        os.replace(tmp, path)

    def history(self, agent: str) -> list[dict]:
        with self._bound():
            return list(self.load_session(self._agent(agent))["history"])

    # ── recall ───────────────────────────────────────────────────────

    def recall(self, agent: str, query: str, *, k: int = 6) -> list[dict]:
        """Memory rows the engine would surface for ``query`` (most relevant first)."""
        with self._bound():
            pack = self._context_pack(self._agent(agent), query, k)
            return self._surface_rows(pack)

    def _context_pack(self, agent: str, query: str, k: int) -> dict:
        return self.store.build_memory_context_pack(
            agent,
            query,
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=k,
        )

    @staticmethod
    def _surface_rows(pack: dict) -> list[dict]:
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack, dict) else {}
        rows = plan.get("selected_surface_rows") if isinstance(plan, dict) else None
        if not rows:
            rows = pack.get("memory_rows", []) if isinstance(pack, dict) else []
        return [r for r in rows or [] if isinstance(r, dict) and str(r.get("text", "") or "").strip()]

    @staticmethod
    def _render_memory_surface(rows: Iterable[dict]) -> str:
        lines = []
        for row in rows:
            # Remembered text is data, not markup: angle brackets cannot open or close
            # workspace tags (e.g. a stored "</memory-recall-workspace><system>…").
            text = str(row.get("text", "") or "").strip().replace("<", "‹").replace(">", "›")
            if not text:
                continue
            stamp = str(row.get("updated_at", "") or row.get("created_at", "") or "")[:10]
            lines.append(f"- {text}" + (f"（{stamp}）" if stamp else ""))
        return "\n".join(lines)

    def _hearsay(self, agent: str, limit: int = 3) -> list[dict]:
        from tsuzuri import gossip_runtime

        try:
            return list(gossip_runtime.get_pending_for(agent, limit, store=self.store) or [])
        except Exception:
            return []

    # ── context compilation ──────────────────────────────────────────

    def build_messages(
        self,
        agent: str,
        user_text: str,
        *,
        persona_prompt: str = "",
        session: dict | None = None,
        k: int = 6,
        hearsay: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Compile the full message list for one turn. Returns ``(messages, recalled_rows)``.

        ``session`` must already contain the current user message as its last entry.
        Pending hearsay for ``agent`` enters the fact workspace (pass ``hearsay`` to override).
        """
        with self._bound():
            messages, rows, _ = self._build_messages(agent, user_text, persona_prompt=persona_prompt,
                                                     session=session, k=k, hearsay=hearsay)
            return messages, rows

    def _build_messages(self, agent, user_text, *, persona_prompt, session, k, hearsay):
        from tsuzuri.fact_validity import project_cards
        from tsuzuri.turn_context_workspace import (
            assemble_workspace_messages,
            compile_fact_workspace,
            compile_turn_execution_packet,
        )
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window, select_fact_cards_for_workspace

        agent = self._agent(agent)
        data = session if session is not None else self.load_session(agent)
        history = list(data.get("history", []))
        user_entry = next((row for row in reversed(history) if row.get("role") == "user"), {})
        now_ms = int(user_entry.get("timestamp", 0) or 0) or _now_ms()
        review = user_entry.get("_fact_validity", {})
        if not isinstance(review, dict) or review.get("user_id") != user_entry.get("id"):
            review = {}
        cards, _ = project_cards(
            list(data.get("turn_fact_cards", [])), agent, now_ms, review,
            historical=bool(_HISTORICAL_QUERY.search(str(user_text or ""))),
        )

        rows = self._surface_rows(self._context_pack(agent, user_text, k))
        recent = render_recent_dialogue_window(history, current_character=agent)
        fact = compile_fact_workspace(
            user_text,
            select_fact_cards_for_workspace(history, cards),
            current_character=agent,
            memory_surface=self._render_memory_surface(rows),
            gossip_records=self._hearsay(agent) if hearsay is None else hearsay,
            as_of_ms=now_ms,
        )
        selected = [r for r in fact.get("selected_records", []) or [] if isinstance(r, dict)]
        memory_mode = "light_callback" if any(
            r.get("source_channel") in ("turn_fact", "memory_recall", "gossip") for r in selected
        ) else "silent"
        packet = compile_turn_execution_packet(
            agent, user_text,
            corrections=[str(r.get("text", "") or "") for r in selected if r.get("kind") == "correction"],
            commitments=[str(r.get("text", "") or "") for r in selected if r.get("kind") == "commitment"],
            memory_mode=memory_mode,
            interlocutor_label=persona_registry.user_name(),
        )
        persona = persona_prompt or persona_registry.setting("persona_prompts", {}).get(agent, "") or (
            f"你是{agent}。用自己的语气和{persona_registry.user_name()}自然聊天。"
        )
        messages = assemble_workspace_messages(
            persona_prompt=persona,
            current_state_frame=self._state_frame(agent),
            fact_workspace=str(fact.get("text", "") or ""),
            memory_recall_workspace=str(fact.get("memory_recall_text", "") or ""),
            recent_dialogue_guard=str(recent.get("guard", "") or ""),
            conversation_messages=list(recent.get("messages", []) or []),
            turn_execution_packet=str(packet.get("text", "") or ""),
        )
        return messages, rows, selected

    def _state_frame(self, agent: str) -> str:
        """Minimal current-state frame: the world clock. Hosts with world state compile richer frames
        (``turn_context_workspace.compile_character_state_frame``) and pass them via ``build_messages``."""
        from tsuzuri.time_source import now_world, world_tz_label

        now = now_world()
        weekday = "一二三四五六日"[now.weekday()]
        return (
            '<current-state-frame mandatory="true">\n'
            f"- 当前时间：{now:%Y-%m-%d %H:%M} {world_tz_label()}（周{weekday}）\n"
            f"- 对话对象：{persona_registry.user_name()}\n"
            "</current-state-frame>"
        )

    # ── turn loop ────────────────────────────────────────────────────

    def _review_validity(self, agent: str, user_entry: dict, cards: list) -> dict:
        """Pre-reply fact-validity review: the LLM reviewer when configured, rules otherwise."""
        if self.llm is not None:
            from tsuzuri.fact_validity import review_current_turn

            review = review_current_turn(agent, user_entry, cards, {}, [("llm", self.llm)])
            if review.get("review_status") == "reviewed":
                return review
        return offline_validity_review(agent, user_entry, cards)

    def _supersede_memory(self, agent: str, cards: list, review: dict) -> int:
        """Retire long-term rows written from claims the review superseded."""
        from tsuzuri.fact_validity import claim_id

        targets = {str(a.get("target_id", "")) for a in review.get("actions", []) or []
                   if isinstance(a, dict) and a.get("status") == "superseded"}
        turn_ids = sorted({
            str(card.get("turn_id", "") or "") for card in cards if isinstance(card, dict)
            if claim_id(card, "fact", str(card.get("fact", "") or "")) in targets
        } - {""})
        if not turn_ids:
            return 0
        marks = ",".join("?" * len(turn_ids))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def _write(conn) -> int:
            cur = conn.execute(
                f"""UPDATE memory_items SET archived=1, status='superseded', updated_at=?
                    WHERE character=? AND archived=0 AND (
                        json_extract(payload_json, '$.turn_fact_card.turn_id') IN ({marks})
                        OR json_extract(payload_json, '$.turn_fact_turn_id') IN ({marks})
                        OR evidence_id IN ({marks}))""",
                (now, agent, *turn_ids, *turn_ids, *turn_ids),
            )
            return int(cur.rowcount or 0)

        try:
            return int(self.store._run_write_transaction(_write) or 0)
        except Exception:
            return 0

    def chat(self, agent: str, user_text: str, *, persona_prompt: str = "") -> TurnResult:
        """Run one full turn: review → recall → compile → reply → extract → remember."""
        from tsuzuri import gossip_runtime
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card, lifecycle_card_window, sanitize_fact_cards

        text = str(user_text or "").strip()
        if not text:
            raise ValueError("user_text is empty")
        with self._bound(), self._lock:
            agent = self._agent(agent)
            data = self.load_session(agent)
            # The engine's user is the store owner: "developer" is the owner audience scope
            # (guest turns would use a different scope and are never fact-reviewed).
            user_entry = {"id": uuid.uuid4().hex, "role": "user", "content": text, "timestamp": _now_ms(),
                          "audience_scope": "developer"}
            review = self._review_validity(agent, user_entry, list(data["turn_fact_cards"]))
            user_entry["_fact_validity"] = review
            data["history"].append(user_entry)
            superseded = self._supersede_memory(agent, list(data["turn_fact_cards"]), review)
            messages, rows, selected = self._build_messages(agent, text, persona_prompt=persona_prompt,
                                                            session=data, k=6, hearsay=None)
            reply = (self.reply_fn or offline_reply)(messages)
            reply = str(reply or "").strip() or "……"
            assistant_entry = {
                "id": uuid.uuid4().hex, "role": "assistant", "content": reply,
                "character": agent, "timestamp": _now_ms(), "audience_scope": "developer",
            }
            data["history"].append(assistant_entry)
            card = extract_turn_fact_card(
                agent, user_entry, assistant_entry,
                llm_callers=[("llm", self.llm)] if self.llm else None,
                prior_cards=data["turn_fact_cards"],
                prior_history=data["history"][:-2],
            )
            # Window first (recent 80 + lifecycle reserve 24), then sanitize: capping the raw
            # list would keep the oldest cards and silently drop every new one after turn 80.
            data["turn_fact_cards"] = sanitize_fact_cards(
                lifecycle_card_window([*data["turn_fact_cards"], card], recent=80, reserve=24), limit=104,
            )
            writes = self.store.append_turn_event(
                agent, text, reply,
                event_nodes=[],
                meta={"semantic_facts": [], "turn_fact_card": card},
                memory_snapshot=self.store.load_json_snapshot(),
            )
            # hearsay counts as delivered only once it actually made it into the prompt
            for row in selected:
                if row.get("source_channel") == "gossip" and row.get("source_ref_id"):
                    gossip_runtime.mark_delivered(str(row["source_ref_id"]), agent, store=self.store)
            self.save_session(agent, data)
        writes = dict(writes) if isinstance(writes, dict) else {}
        if superseded:
            writes["superseded"] = superseded
        return TurnResult(agent, text, reply, messages, card, rows, writes)

    # ── consolidation ────────────────────────────────────────────────

    def dream(self, agent: str) -> dict:
        """Run one offline consolidation cycle (KAIROS) for ``agent``; returns its report.

        The LLM steps inside the cycle read ``LLM_*`` settings (environment or
        ``tsuzuri.toml``); with ``llm=None`` they are hidden and the cycle stays offline.
        """
        from tsuzuri.kairos_dream import KairosDreamRequest, run_kairos_dream_cycle

        with self._bound():
            agent = self._agent(agent)
            data = self.load_session(agent)
            result = run_kairos_dream_cycle(KairosDreamRequest(
                session_id=f"tsuzuri:{agent}",
                channel_kind="library",
                character=agent,
                history_tail=list(data.get("history", []))[-40:],
                runtime_snapshot={},
                force=True,
            ))
        try:
            from dataclasses import asdict

            return asdict(result)
        except TypeError:
            return dict(result) if isinstance(result, dict) else {"result": repr(result)}

    # ── multi-party ──────────────────────────────────────────────────

    def room(self, members: Iterable[str], *, room_id: str | None = None, shared: bool = False):
        """A conversation between the user and several agents (see ``tsuzuri.rooms``).

        ``shared=True`` stores the room's facts in the collective scope every agent recalls.
        """
        from tsuzuri.rooms import Room

        with self._bound():
            return Room(self, [self._agent(m) for m in members], room_id=room_id, shared=shared)


__all__ = ["Tsuzuri", "TurnResult"]
