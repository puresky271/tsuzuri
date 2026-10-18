"""Multi-party conversations: the user and several agents in one room.

Each agent reads the shared history from its own point of view (other agents'
lines are attributed, never mistaken for its own), each reply produces a TurnFact
card attributed to its speaker, and facts persist into the speaker's memory — or,
for a room flagged ``shared``, into the collective scope every member can recall.

Agents also pass things on: :meth:`Room.share` records a piece of news one agent
heard; other agents receive it as attributed hearsay in their next prompt (room or 1:1)
and it is marked delivered once they have seen it (``tsuzuri.gossip_runtime``).

Without an ``llm`` the rule-based fact cards are only echoes of the raw line, so room
facts are not persisted to long-term memory offline (``room_memory.room_fact_is_writable``);
the room's own history still carries the conversation.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from tsuzuri import persona_registry
from tsuzuri.offline import offline_reply

if TYPE_CHECKING:  # pragma: no cover
    from tsuzuri.engine import Tsuzuri


@dataclass
class RoomTurn:
    user_text: str
    replies: list[tuple[str, str]] = field(default_factory=list)  # (agent, reply)
    fact_cards: list[dict] = field(default_factory=list)


class Room:
    def __init__(self, engine: "Tsuzuri", members: list[str], *, room_id: str | None = None, shared: bool = False):
        if len(members) < 2:
            raise ValueError("a room needs at least two agents")
        self.engine = engine
        self.members = list(dict.fromkeys(members))
        self.room_id = room_id or ("room-" + "-".join(self.members))
        self.shared = shared
        self._key = f"_room_{self.room_id}"

    # persistence ------------------------------------------------------
    def _load(self) -> dict:
        data = self.engine.load_session(self._key)
        data.setdefault("members", self.members)
        return data

    def _save(self, data: dict) -> None:
        self.engine.save_session(self._key, data)

    @property
    def history(self) -> list[dict]:
        return list(self._load()["history"])

    # turn loop --------------------------------------------------------
    def messages_for(self, agent: str, user_text: str, *, data: dict | None = None,
                     hearsay: list[dict] | None = None) -> list[dict]:
        """Compiled messages for ``agent``'s reply in this room (pending hearsay included)."""
        with self.engine._bound():
            return self._messages_for(persona_registry.resolve_agent(agent), user_text, data=data, hearsay=hearsay)

    def _messages_for(self, agent: str, user_text: str, *, data: dict | None, hearsay: list[dict] | None,
                      with_selected: bool = False):
        from tsuzuri.room_memory import build_group_context, room_history_for_workspace
        from tsuzuri.turn_context_workspace import compile_fact_workspace
        from tsuzuri.turn_fact_ledger import select_fact_cards_for_workspace

        data = data if data is not None else self._load()
        history = room_history_for_workspace(data["history"], agent, room_id=self.room_id)
        rows = self.engine.recall(agent, user_text)
        fact = compile_fact_workspace(
            user_text,
            select_fact_cards_for_workspace(history, data.get("turn_fact_cards", []), group_mode=True),
            current_character=agent,
            memory_surface=self.engine._render_memory_surface(rows),
            gossip_records=self.engine._hearsay(agent) if hearsay is None else hearsay,
        )
        others = "、".join(m for m in self.members if m != agent)
        persona = persona_registry.setting("persona_prompts", {}).get(agent, "") or f"你是{agent}。"
        system = (
            f"{persona}\n现在是多人对话：{self.engine.user_name}、你和{others}。"
            "只代表你自己发言；[名字]：开头的是其他人说的话。"
        )
        messages = [{"role": "system", "content": system}]
        if fact.get("text"):
            messages.append({"role": "system", "content": fact["text"]})
        messages.extend(build_group_context(data["history"], 40, current_character=agent))
        if with_selected:
            return messages, [r for r in fact.get("selected_records", []) or [] if isinstance(r, dict)]
        return messages

    def say(self, user_text: str, *, speakers: Iterable[str] | None = None) -> RoomTurn:
        """The user speaks; each speaker (default: every member, in order) replies once."""
        text = str(user_text or "").strip()
        if not text:
            raise ValueError("user_text is empty")
        with self.engine._bound():
            return self._say(text, speakers)

    def _say(self, text: str, speakers: Iterable[str] | None) -> RoomTurn:
        from tsuzuri import gossip_runtime
        from tsuzuri.room_memory import merge_room_fact_card, room_history_for_workspace, write_room_fact_memory
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        order = [persona_registry.resolve_agent(s) for s in (speakers or self.members)]
        data = self._load()
        user_entry = {"id": uuid.uuid4().hex, "role": "user", "content": text, "timestamp": int(time.time() * 1000)}
        data["history"].append(user_entry)
        turn = RoomTurn(text)
        for agent in order:
            if agent not in self.members:
                raise ValueError(f"{agent!r} is not in this room")
            messages, selected = self._messages_for(agent, text, data=data, hearsay=None, with_selected=True)
            reply = str((self.engine.reply_fn or offline_reply)(messages) or "").strip() or "……"
            for row in selected:
                if row.get("source_channel") == "gossip" and row.get("source_ref_id"):
                    gossip_runtime.mark_delivered(str(row["source_ref_id"]), agent, store=self.engine.store)
            entry = {
                "id": uuid.uuid4().hex, "role": "assistant", "content": reply, "character": agent,
                "timestamp": int(time.time() * 1000),
                "room_context": {"participants": list(self.members), "channel": "line", "members": []},
            }
            data["history"].append(entry)
            history = room_history_for_workspace(data["history"], agent, room_id=self.room_id)
            card = extract_turn_fact_card(
                agent, user_entry, entry,
                llm_callers=[("llm", self.engine.llm)] if self.engine.llm else None,
                prior_cards=data.get("turn_fact_cards", []), prior_history=history[:-1],
            )
            data["turn_fact_cards"], data["turn_fact_state_summary"] = merge_room_fact_card(
                data.get("turn_fact_cards", []), card,
            )
            write_room_fact_memory(self.engine.store, self.room_id, agent, user_entry, entry, card,
                                   shared_room=self.shared, source="library_room")
            turn.replies.append((agent, reply))
            turn.fact_cards.append(card)
        self._save(data)
        return turn

    # hearsay ------------------------------------------------------------
    def share(self, source_agent: str, text: str, *, trust: float = 0.8) -> list[str]:
        """``source_agent`` passes ``text`` on to the other room members; returns the hearsay ids.

        Delivery is decided here (every other member hears it), so the spread roll is forced;
        confidence still decays with ``trust`` and the source agent's distortion profile.
        """
        from tsuzuri import gossip_runtime

        with self.engine._bound():
            source = persona_registry.resolve_agent(source_agent)
            graph = {
                a: {b: {"trust": trust, "info_sharing": 1.0 if b in self.members else 0.0}
                    for b in persona_registry.agent_names()}
                for a in self.members
            }
            return gossip_runtime.record_fanout(source, text, graph=graph, store=self.engine.store,
                                                rng=lambda: 1e-9)  # any non-zero edge spreads; info_sharing=0 never does

    def hearsay_for(self, agent: str, limit: int = 3) -> list[dict]:
        """Pending second-hand news ``agent`` has heard and not yet mentioned."""
        from tsuzuri import gossip_runtime

        with self.engine._bound():
            return gossip_runtime.get_pending_for(persona_registry.resolve_agent(agent), limit, store=self.engine.store)


__all__ = ["Room", "RoomTurn"]
