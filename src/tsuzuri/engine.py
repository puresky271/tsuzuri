"""High-level entry point: one object that remembers conversations for a cast of agents.

The lower layers are usable on their own (``tsuzuri.memory_core``,
``tsuzuri.turn_fact_ledger``, ``tsuzuri.turn_context_workspace`` …); this module wires
them into the turn loop most applications want:

    user message ─▶ recall + fact workspace ─▶ messages for your LLM ─▶ reply
                 ◀─ TurnFact card + memory write ◀───────────────────────┘

Everything runs offline by default (rule-based extraction, lexical recall). Pass an
``llm`` callable to enable LLM fact extraction, semantic rethink and consolidation,
and a ``reply`` callable to generate the agent's answers with your own model.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from tsuzuri import persona_registry
from tsuzuri.offline import offline_reply

TextLLM = Callable[[str], str]
ChatLLM = Callable[[list[dict]], str]


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
        ``prompt -> text`` callable used for fact extraction, rethink and dreaming.
        ``None`` keeps everything rule-based and offline.
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
        os.environ["TSUZURI_HOME"] = self.data_dir
        if not embeddings:
            os.environ["EMBEDDING_ENABLED"] = "0"
        if personas is not None:
            if isinstance(personas, Mapping):
                persona_registry.configure_personas(personas, reset=True)
            else:
                with open(os.fspath(personas), "r", encoding="utf-8") as fh:
                    persona_registry.configure_personas(json.load(fh), reset=True)
        self.llm = llm
        self.reply_fn = reply
        self._lock = threading.RLock()

        from tsuzuri.memory_core import MemoryCoreStore

        self.store = MemoryCoreStore(
            db_path=os.path.join(self.data_dir, "memory.db"),
            snapshot_path=os.path.join(self.data_dir, "snapshot.json"),
            bootstrap_from_snapshot=False,
        )
        # Internal modules (consolidation, memory API, gossip) reach the store through the
        # process-wide singleton; point it at this engine's store. One engine per process.
        from tsuzuri.memory_core_impl import store as _store_module

        _store_module._MEMORY_CORE_SINGLETON = self.store

    # ── cast ─────────────────────────────────────────────────────────

    @property
    def agents(self) -> tuple[str, ...]:
        return persona_registry.agent_names()

    @property
    def user_name(self) -> str:
        return persona_registry.user_name()

    def _agent(self, agent: str | None) -> str:
        name = persona_registry.resolve_agent(agent)
        if not persona_registry.is_agent(name):
            raise ValueError(f"unknown agent {name!r}; known: {', '.join(self.agents)}")
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
        return list(self.load_session(self._agent(agent))["history"])

    # ── recall ───────────────────────────────────────────────────────

    def recall(self, agent: str, query: str, *, k: int = 6) -> list[dict]:
        """Memory rows the engine would surface for ``query`` (most relevant first)."""
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
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            stamp = str(row.get("updated_at", "") or row.get("created_at", "") or "")[:10]
            lines.append(f"- {text}" + (f"（{stamp}）" if stamp else ""))
        return "\n".join(lines)

    # ── context compilation ──────────────────────────────────────────

    def build_messages(
        self,
        agent: str,
        user_text: str,
        *,
        persona_prompt: str = "",
        session: dict | None = None,
        k: int = 6,
    ) -> tuple[list[dict], list[dict]]:
        """Compile the full message list for one turn. Returns ``(messages, recalled_rows)``.

        ``session`` must already contain the current user message as its last entry.
        """
        from tsuzuri.turn_context_workspace import (
            assemble_workspace_messages,
            compile_fact_workspace,
            compile_turn_execution_packet,
        )
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window, select_fact_cards_for_workspace

        agent = self._agent(agent)
        data = session if session is not None else self.load_session(agent)
        history = list(data.get("history", []))
        cards = list(data.get("turn_fact_cards", []))

        rows = self._surface_rows(self._context_pack(agent, user_text, k))
        recent = render_recent_dialogue_window(history, current_character=agent)
        fact = compile_fact_workspace(
            user_text,
            select_fact_cards_for_workspace(history, cards),
            current_character=agent,
            memory_surface=self._render_memory_surface(rows),
            as_of_ms=_now_ms(),
        )
        state = {"text": self._state_frame(agent)}
        packet = compile_turn_execution_packet(
            agent, user_text, interlocutor_label=self.user_name, reply_target=user_text,
        )
        persona = persona_prompt or persona_registry.setting("persona_prompts", {}).get(agent, "") or (
            f"你是{agent}。用自己的语气和{self.user_name}自然聊天。"
        )
        messages = assemble_workspace_messages(
            persona_prompt=persona,
            current_state_frame=str(state.get("text", "") or ""),
            fact_workspace=str(fact.get("text", "") or ""),
            memory_recall_workspace=str(fact.get("memory_recall_text", "") or ""),
            recent_dialogue_guard=str(recent.get("guard", "") or ""),
            conversation_messages=list(recent.get("messages", []) or []),
            turn_execution_packet=str(packet.get("text", "") or ""),
        )
        return messages, rows

    def _state_frame(self, agent: str) -> str:
        """Minimal current-state frame: the world clock. Hosts with world state compile richer frames
        (``turn_context_workspace.compile_character_state_frame``) and pass them via ``build_messages``."""
        from tsuzuri.time_source import now_world, world_tz_label

        now = now_world()
        weekday = "一二三四五六日"[now.weekday()]
        return (
            '<current-state-frame mandatory="true">\n'
            f"- 当前时间：{now:%Y-%m-%d %H:%M} {world_tz_label()}（周{weekday}）\n"
            f"- 对话对象：{self.user_name}\n"
            "</current-state-frame>"
        )

    # ── turn loop ────────────────────────────────────────────────────

    def chat(self, agent: str, user_text: str, *, persona_prompt: str = "") -> TurnResult:
        """Run one full turn: recall → compile → reply → extract → remember."""
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card, sanitize_fact_cards

        agent = self._agent(agent)
        text = str(user_text or "").strip()
        if not text:
            raise ValueError("user_text is empty")
        with self._lock:
            data = self.load_session(agent)
            user_entry = {"id": uuid.uuid4().hex, "role": "user", "content": text, "timestamp": _now_ms()}
            data["history"].append(user_entry)
            messages, rows = self.build_messages(agent, text, persona_prompt=persona_prompt, session=data)
            reply = (self.reply_fn or offline_reply)(messages)
            reply = str(reply or "").strip() or "……"
            assistant_entry = {
                "id": uuid.uuid4().hex, "role": "assistant", "content": reply,
                "character": agent, "timestamp": _now_ms(),
            }
            data["history"].append(assistant_entry)
            card = extract_turn_fact_card(
                agent, user_entry, assistant_entry,
                llm_callers=[("llm", self.llm)] if self.llm else None,
                prior_cards=data["turn_fact_cards"],
                prior_history=data["history"][:-2],
            )
            data["turn_fact_cards"] = sanitize_fact_cards([*data["turn_fact_cards"], card])
            writes = self.store.append_turn_event(
                agent, text, reply,
                event_nodes=[],
                meta={"semantic_facts": [], "turn_fact_card": card},
                memory_snapshot=self.store.load_json_snapshot(),
            )
            self.save_session(agent, data)
        return TurnResult(agent, text, reply, messages, card, rows, writes if isinstance(writes, dict) else {})

    # ── consolidation ────────────────────────────────────────────────

    def dream(self, agent: str) -> dict:
        """Run one offline consolidation cycle (KAIROS) for ``agent``; returns its report."""
        from tsuzuri.kairos_dream import KairosDreamRequest, run_kairos_dream_cycle

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

        return Room(self, [self._agent(m) for m in members], room_id=room_id, shared=shared)


__all__ = ["Tsuzuri", "TurnResult"]
