# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break APIs).

## [0.1.1] - 2026-10-19

Fixes from an external review of 0.1.0. Not published to PyPI; install from source.

### Fixed
- FTS5 index drift: the external-content index is now maintained by triggers on `memory_items` (rowid-aligned, active rows only) and rebuilt once on upgrade; MATCH input is quoted so punctuation cannot raise a syntax error. Previously any text update left orphan rowids, the next MATCH raised `fts5: missing row` and the FTS lane silently stopped working.
- Confidential remarks ("先别告诉 X 和 Y") were promoted to the collective scope that X and Y recall, and could reach the cross-agent digest; they are now kept out of both.
- Several `Tsuzuri` engines in one process no longer overwrite each other's cast, store or data directory: each call re-binds the process-wide state to the calling engine.
- Self-corrections now take effect: the engine runs the fact-validity review before each reply (the LLM reviewer with `llm=`, rule-based offline), projects superseded cards out of the workspace and retires the long-term rows written from the stale claim.
- Fact cards stopped growing after 80 turns (the newest card was dropped); the engine now windows cards (80 recent + 24 lifecycle reserve) before capping.
- The execution packet no longer re-sends the user's message as an explicit reply target (which also disabled the per-agent length tiers).
- The offline responder ignored the user's own 1:1 message and always answered "嗯。".
- Pending hearsay now enters the receiving agent's fact workspace (1:1 and rooms) and is marked delivered once it has been shown.
- `Tsuzuri(llm=None)` hides `LLM_*` settings during its calls, so consolidation steps cannot go online unexpectedly.
- The weekly reflection processor referenced undefined names (`NameError`, swallowed by the dream cycle); the export now runs an undefined-name lint gate.
- Cluster summaries no longer write a "[Cluster of N] <first line>" placeholder into the semantic layer when the LLM is missing or fails; the cluster is skipped.
- Read-path Hebbian co-occurrence updates are written in one transaction per recall instead of one connection and commit per edge.
- Recalled text is rendered with angle brackets neutralized, so stored text cannot open or close workspace tags.
- Dense recall retries with `local_files_only` when the model hub is unreachable; `tsuzuri eval --embeddings` reports the mode that actually ran.
- The world-clock fixed-offset table had a duplicate `"UTC"` key.

### Changed
- `tsuzuri eval` is stricter: the persona prompt no longer counts as visible context, hearsay must reach the compiled context, and isolation cases require the owner to recall the fact (positive control). The bundled set's hearsay and isolation probes were adjusted accordingly; the offline baseline is 7/16.
- Lint covers the whole package and test suite; CI reports coverage, checks the built distributions and runs the demo from the installed wheel.
- Documentation states what `Tsuzuri.chat()` compiles versus host-level compilers, the shadow week for evolution, the conditions for rethink, and the current deletion semantics.

## [0.1.0] - 2026-10-18

Initial source release.

### Added
- Layered per-agent memory store (working / episodic / semantic / procedural + collective scope) on SQLite with FTS5, hybrid recall, reranking hooks and ACT-R style decay.
- TurnFact cards: per-exchange facts, state deltas, commitments, corrections and open threads, with lifecycle windows and correction supersession.
- Context compiler (`compile_fact_workspace`, state frame, on-demand evidence, conversation arc, turn execution packet, `assemble_workspace_messages`).
- Daily continuity capsules and day recaps; fact validity review; interlocutor state and agent action ledgers.
- Offline consolidation cycle: evolution, semantic rethink, event/topic clustering, core summary, entity projection, forgetting scan, weekly relationship reflections with a built-in reflection sink.
- Multi-party rooms with speaker attribution, room fact cards, shared collective memory and hearsay propagation.
- Persona registry (cast, aliases, collective wording, user identity, per-deployment settings) and optional host hooks.
- High-level `Tsuzuri` engine, `Room` API, offline responder, OpenAI-compatible adapter, `tsuzuri` CLI (`demo`, `chat`, `recall`, `dream`, `eval`) and a synthetic recall evaluation set.
