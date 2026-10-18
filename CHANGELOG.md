# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break APIs).

## [0.1.0] - Unreleased

### Added
- Layered per-agent memory store (working / episodic / semantic / procedural + collective scope) on SQLite with FTS5, hybrid recall, reranking hooks and ACT-R style decay.
- TurnFact cards: per-exchange facts, state deltas, commitments, corrections and open threads, with lifecycle windows and correction supersession.
- Context compiler (`compile_fact_workspace`, state frame, on-demand evidence, conversation arc, turn execution packet, `assemble_workspace_messages`).
- Daily continuity capsules and day recaps; fact validity review; interlocutor state and agent action ledgers.
- Offline consolidation cycle: evolution, semantic rethink, event/topic clustering, core summary, entity projection, forgetting scan, weekly relationship reflections with a built-in reflection sink.
- Multi-party rooms with speaker attribution, room fact cards, shared collective memory and hearsay propagation.
- Persona registry (cast, aliases, collective wording, user identity, per-deployment settings) and optional host hooks.
- High-level `Tsuzuri` engine, `Room` API, offline responder, OpenAI-compatible adapter, `tsuzuri` CLI (`demo`, `chat`, `recall`, `dream`, `eval`) and a synthetic recall evaluation set.
