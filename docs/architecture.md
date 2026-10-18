# Architecture

tsuzuri has three rings. Each can be used without the ones outside it.

| Ring | Modules | Responsibility |
|---|---|---|
| **Memory engine** | `memory_core`, `memory_core_impl/`, `memory_provider*`, `memory_api`, `memory_entities`, `cluster_memory*`, `memory_rethink_engine`, `embedding_engine`, `reranker_engine` | Persistent, per-agent memory rows: writes, recall, evolution, rethink, clustering, management API |
| **Chat workspace** | `turn_fact_ledger`, `turn_context_workspace`, `daily_continuity/`, `chat_compact`, `conversation_arc`, `fact_validity`, `interlocutor_state_ledger`, `character_action_ledger`, `memory_tools`, `room_memory`, `gossip_runtime` | Turning conversations into fact cards and compiling what the model sees each turn, one-on-one and in rooms |
| **Orchestration** | `engine`, `rooms`, `kairos_dream`, `kairos_state`, `kairos_signal_hub`, `developer_profile`, `cli`, `evaluation` | The turn loop, the offline consolidation cycle, long-term impressions, tooling |

Two seams keep the rings independent of any application: `persona_registry` (who the system is about) and `memory_host` (optional hooks into the host).

## One turn

```
user message
  │
  ├─ recall ─────────────── MemoryCoreStore.build_memory_context_pack(agent, text)
  │                           hybrid candidates (recent, FTS, dense/sparse, structured slots)
  │                           → unified score (relevance, recency, ACT-R activation, importance)
  │                           → correction resolution → rerank → entity boost → MMR
  │                           → surface plan (explicit recall / silent grounding / abstain)
  │
  ├─ fact cards ─────────── select_fact_cards_for_workspace(history, cards)
  ├─ recent dialogue ────── render_recent_dialogue_window(history)
  ├─ conversation arc ───── compile_conversation_arc(...)            (reopen / correct / fulfil links)
  ├─ fact workspace ─────── compile_fact_workspace(text, cards, memory_surface=..., ...)
  │                           eligibility gate, lexical + optional semantic scoring,
  │                           supersession by corrections, de-duplication, char budget
  ├─ state frame ────────── compile_character_state_frame(agent, ...)
  ├─ execution packet ───── compile_turn_execution_packet(agent, text, ...)
  └─ assemble ───────────── assemble_workspace_messages(...)          → messages for the model

reply
  ├─ extract_turn_fact_card(agent, user_entry, assistant_entry, llm_callers=...)
  │     LLM extraction with name canonicalization and evidence checks; rule fallback offline
  └─ MemoryCoreStore.append_turn_event(agent, user_text, reply, meta=...)
        plan_turn_memory_writes → working / episodic / semantic / collective rows
```

`Tsuzuri.chat()` runs exactly this; `Tsuzuri.build_messages()` stops before the model call.

### Message order

`assemble_workspace_messages` keeps a fixed order: persona prompt → historical recap → current state → interlocutor state → proactive continuity → fact workspace → recalled memory → cluster memory → conversation arc → on-demand evidence → cognitive workspace → recent-dialogue guard → conversation → current user message with the execution packet appended. Stable material comes first; the packet sits where the model attends most.

## Storage

One SQLite file (`memory.db`) holds:

- `memory_items` (+ `memory_items_fts`): every memory row with `character` (agent or collective scope), `layer`, typed payload, cues, lifecycle, salience/confidence/importance, optional embeddings.
- `memory_links`, `co_occurrence_strength`: association edges and Hebbian co-occurrence.
- `memory_entities`, `memory_entity_links`: a derived entity projection (rebuildable).
- `meta_state`, `memory_domains`: snapshots and structured domains (user profile per agent, relationships).
- `shared_info`, `gossip_audit_log`: hearsay and its propagation audit.
- `interlocutor_state_facts`, `character_action_events`: the user's ongoing states and agents' confirmed actions.
- audit tables for retrieval, evolution, rethink and the management API.

Session history and fact cards live in JSON files under `sessions/`.

**No hard deletes.** Archiving flips status; forgetting replaces text with a tombstone and keeps provenance. An LLM never writes a row directly: models propose, local code plans, validates and writes.

## Layers

| Layer | Holds | Promoted by |
|---|---|---|
| `working` | compressed recent exchanges | — |
| `episodic` | events, observations, digests passed between agents | turn writes, cross-agent digests |
| `semantic` | stable facts, preferences, key events | explicit "remember this", corrections, consolidation |
| `procedural` | interaction strategies (never surfaced verbatim) | evolution |
| `collective` | facts shared by the whole cast (`collective_scope`) | multi-member context, shared rooms |

## Consolidation (`kairos_dream`)

`Tsuzuri.dream(agent)` runs one cycle: freeze evidence → rule-based evolution (promote, merge, decay) → semantic rethink (rewrite / merge / archive / no-change against new evidence, with OOC checks) → evidence maintenance → cross-agent digest → weekly relationship reflection → event/topic clustering with a faithfulness audit → core summary → entity rebuild → forgetting scan. Steps that need an LLM run only when `llm=` is configured; destructive steps default to dry-run (see [configuration](configuration.md)).

## Multi-party memory

- `room_memory.build_group_context` renders a shared history from one agent's point of view; other agents' lines become `[name]：` user turns so the model never mistakes them for its own.
- Room fact cards de-duplicate per `(user message, speaker)`; facts persist into the speaker's memory, or into the collective scope for a shared room.
- `gossip_runtime` propagates hearsay: first hop confidence depends on the edge's trust and the source's distortion profile, each further hop decays it, and delivery is audited.

## Time

Memory ages in real UTC time (`utc_now_naive`). Dates that a character reasons about ("today", "next Wednesday", day boundaries for continuity) use the *world clock* (`time_source.now_world()`), whose time zone is `WORLD_TIMEZONE` (default UTC). `DEBUG_MOCK_DATE` moves the world clock for tests.
