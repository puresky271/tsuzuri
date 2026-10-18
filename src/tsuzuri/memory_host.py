"""Optional integration points between the memory system and the application embedding it.

The engine never imports application modules. Where it can use something only the
host knows (a character's full system prompt, canon facts, a message board, an
OOC classifier, …) it asks for a named hook here; a missing hook means the step
is skipped, exactly as when the corresponding feature is unavailable.

Hosts register hooks either explicitly::

    from tsuzuri import memory_host as memory_host
    memory_host.register("agent_system_prompt", lambda character: ...)

or by providing a module that exposes ``register_hooks(register)``. That module is
imported lazily on first lookup; its name comes from env ``MEMORY_HOST_MODULE``
(default ``tsuzuri_host``). Hooks a host does not register stay absent.

Known hook names (all optional):

- ``agent_profile_slice(character) -> dict``: short canon facts for core summaries.
- ``agent_motivation_core(character) -> list``: seed motivations.
- ``agent_system_prompt(character) -> str``: full chat-time system prompt (weekly reflection).
- ``agent_voice_manifest(character) -> str``: voice/style notes (long-term impression editor).
- ``person_alias_catalog() -> dict``: agent -> nicknames, for TurnFact name normalization.
- ``known_person_names() -> iterable``: every known person name (agents, full names, other characters).
- ``ooc_score(reply, character) -> obj``: out-of-character classifier (``.confidence``, ``.features_hit``).
- ``prompt_block_budget(block_id) -> int``: character budget of a host prompt block.
- ``compose_prompt_blocks(profile=, dynamic_blocks=) -> dict``: compose a host persona prompt (``{"prompt": ...}``).
- ``apply_canon_penalty(snapshot, character, audit, penalty) -> dict`` / ``save_memory_snapshot(snapshot)``.
- ``decay_shared_topics(relationship, now)`` / ``promote_topic_candidates(relationship, now)``.
- ``relation_tier(conversation_count, relationship) -> tuple``.
- ``build_dream_brief(request) -> obj``: world/schedule brief for the dream cycle.
- ``reflection_sink() -> obj``: where weekly reflections are delivered (see ``memory_core_impl.reflection_sink``).
"""
from __future__ import annotations

import importlib
import os
import threading
from typing import Any, Callable

_DEFAULT_HOST_MODULE = "tsuzuri_host"

_lock = threading.RLock()
_hooks: dict[str, Callable[..., Any]] = {}
_autoload_done = False


def register(name: str, fn: Callable[..., Any] | None) -> None:
    """Register (or with ``fn=None`` remove) a hook."""
    key = str(name or "").strip()
    if not key:
        return
    with _lock:
        if fn is None:
            _hooks.pop(key, None)
        else:
            _hooks[key] = fn


def _autoload() -> None:
    global _autoload_done
    if _autoload_done:
        return
    with _lock:
        if _autoload_done:
            return
        _autoload_done = True
        module_name = str(os.getenv("MEMORY_HOST_MODULE", _DEFAULT_HOST_MODULE) or "").strip()
        if not module_name or module_name.lower() in {"0", "none", "off"}:
            return
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return
        installer = getattr(module, "register_hooks", None)
        if callable(installer):
            installer(register)


def get(name: str) -> Callable[..., Any] | None:
    _autoload()
    with _lock:
        return _hooks.get(str(name or "").strip())


def has(name: str) -> bool:
    return get(name) is not None


def call(name: str, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Invoke hook ``name``; return ``default`` when it is not registered. Hook errors propagate."""
    fn = get(name)
    if fn is None:
        return default
    return fn(*args, **kwargs)


def reset(*, autoload: bool = True) -> None:
    """Drop all hooks (tests). ``autoload=False`` keeps the host module from loading again."""
    global _autoload_done
    with _lock:
        _hooks.clear()
        _autoload_done = not autoload


__all__ = ["register", "get", "has", "call", "reset"]
