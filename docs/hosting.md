# Embedding tsuzuri in an application

## Three ways to use it

1. **Turn loop** — `Tsuzuri.chat(agent, text)` does recall, compilation, the reply call and the memory write.
2. **Compiler only** — keep your own model call:

   ```python
   messages, recalled = memory.build_messages(agent, text, session=session)
   reply = my_model(messages)
   ```

   then record the turn yourself (append both entries to the session, extract a fact card with
   `turn_fact_ledger.extract_turn_fact_card`, and call `memory.store.append_turn_event`). `engine.py`
   is a short reference implementation of exactly these steps.
3. **Components** — use `MemoryCoreStore`, `compile_fact_workspace`, `room_memory` or `gossip_runtime`
   directly; each module documents its contract in its docstring.

## Host hooks

The engine never imports your application. Where a step can use host knowledge it asks `memory_host`
for an optional hook; when the hook is missing the step is skipped or uses a built-in default.

Register hooks explicitly:

```python
from tsuzuri import memory_host

memory_host.register("agent_system_prompt", lambda character: build_my_system_prompt(character))
memory_host.register("ooc_score", my_ooc_classifier)
```

or ship a module named by `MEMORY_HOST_MODULE` (default `tsuzuri_host`) with

```python
def register_hooks(register):
    register("agent_profile_slice", lambda c: {"core_traits": TRAITS[c]})
    ...
```

The full list, with signatures, is in the `tsuzuri.memory_host` docstring. The most useful ones:

| Hook | Effect when present |
|---|---|
| `agent_system_prompt(character)` | Weekly reflections are written with the character's full chat prompt (better voice, less projection) |
| `agent_profile_slice(character)` | Core summaries are anchored on canon facts |
| `ooc_score(reply, character)` | Semantic rethink rejects out-of-character rewrites |
| `known_person_names()` / `person_alias_catalog()` | Better third-party detection and nickname normalization in fact extraction |
| `reflection_sink()` | Reflections go to your own surface (message board, notifications) instead of the built-in table |
| `build_dream_brief(request)` | The dream cycle sees your world/schedule state |

## Reflections

Weekly relationship reflections are notes an agent writes to the user. By default they are stored in the
`reflection_notes` table and folded into the agent's user profile after the user replies
(`store.process_reflection_into_user_profile(message_id=..., user_reply_text=...)`) or after seven days.
A `reflection_sink` hook may deliver them elsewhere; it implements `publish`, `load`, `update_meta` and
`list_before` (see `tsuzuri.memory_core_impl.reflection_sink`).

## Processes and threads

- One `Tsuzuri` per process: internal modules share a process-wide store singleton, which the engine
  points at its own store.
- `MemoryCoreStore` is thread-safe for concurrent reads and serialized writes; SQLite runs in WAL mode.
- Long LLM steps (extraction, rethink, dreaming) are synchronous; run them off your request path.

## Data and privacy

Everything the user says may end up in `memory.db` and `sessions/`. Protect the data directory, back it
up with `runtime_backup`, and treat recalled text as untrusted model input.
