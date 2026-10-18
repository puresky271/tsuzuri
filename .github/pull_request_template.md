## What and why

<!-- One concern per PR. Link the issue: Fixes #123 -->

## Checklist

- [ ] Tests added or updated (`pytest` passes locally)
- [ ] `ruff check .` passes
- [ ] No cast names, user names or domain vocabulary hard-coded in `src/` (use `persona_registry` / `setting()`)
- [ ] No application imports added to the engine (use an optional `memory_host` hook)
- [ ] Prompt text changed? Before/after rendered examples included below
- [ ] Schema changed? Migration is backward compatible
- [ ] No real personal conversations or credentials in code, tests or docs

## Prompt / storage impact

<!-- "none", or describe and show before/after -->
