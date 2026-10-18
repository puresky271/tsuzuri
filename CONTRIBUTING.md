# Contributing to tsuzuri

Thanks for helping. This document covers how to set up, what we expect in a change, and how decisions are made.

## Setup

```bash
git clone https://github.com/<you>/tsuzuri.git   # your fork
cd tsuzuri
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Tests run offline (no API keys, no model downloads). `EMBEDDING_ENABLED=0` is the default in tests.

## What a good change looks like

- **One concern per pull request.** Bug fix, feature, or refactor — not all three.
- **Tests.** New behaviour comes with a test; bug fixes come with a test that fails before the fix. Engine-level suites live in `tests/core/` and run against the fixed test cast in `tests/fixtures/test_cast.json`; API-level tests in `tests/` use the demo cast.
- **No cast in library code.** Agent names, the user's name, group wording and domain vocabulary belong in configuration (`persona_registry`, `settings`), never as literals in `src/`. Use `persona_registry.resolve_agent()`, `agent_names()`, `user_name()`, `collective_scope()`, `persona_regex()` and `setting()`.
- **No application imports.** The engine must not import a host application. If a step needs host knowledge, add an optional hook in `memory_host` and keep the hook-absent path working.
- **Prompt changes are behaviour changes.** If you change text that reaches a model, say so in the PR and include before/after examples of the rendered prompt.
- **Storage changes are migrations.** Schema changes must be backward compatible (`CREATE TABLE IF NOT EXISTS`, guarded `ALTER TABLE`); never assume a fresh database.
- **Keep the LLM out of the write path.** Models propose; local code plans, validates and writes.

## Style

- Python 3.11+, type hints where they help, standard library first.
- `ruff check .` must pass. Match the surrounding code; avoid drive-by reformatting.
- Comments explain *why*. Don't paste real conversations or personal data into code, tests or docs — use invented examples.

## Commit and PR

- Write commit messages in the imperative ("Add …", "Fix …"), with a body that explains the reason.
- Fill in the PR template; link the issue.
- A maintainer reviews within a reasonable time; larger design changes should start as an issue so we can agree on the approach first.

## Reporting bugs and proposing features

Use the issue templates. For bugs, include the version, Python version, a minimal reproduction and what you expected. For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

By contributing you agree that your contributions are licensed under the Apache License 2.0.
