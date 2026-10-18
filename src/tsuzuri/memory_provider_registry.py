"""Memory provider registry — Phase C.4 (2026-05-01).

灵感来自 hermes-agent agent/memory_manager.py L192-250 但简化：
- 群组 只跑单 provider（不像 hermes 的 builtin + external 双 provider 编排）
- 所有 provider 共享同一份 dev_memory.json + memory_core.db、provider 只换召回
  策略不换底层数据
- 默认 provider = "hybrid_v2"；biomimetic 保留为显式回滚路径
- env TSUZURI_MEMORY_PROVIDER=<name> 切换、用于 A/B 对照

Public API:
    register_provider(provider) — 由模块加载时自动注册自己
    set_active_provider(name) — 切换 (例如对照实验)
    get_active_memory_provider() -> MemoryProvider — chat reply 路径调
    list_providers() -> list[str] — debug
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from tsuzuri.memory_provider import MemoryProvider


_REGISTRY: dict[str, MemoryProvider] = {}
_REGISTRY_LOCK = threading.RLock()
_ACTIVE_NAME: Optional[str] = None


def register_provider(provider: MemoryProvider) -> None:
    """Register a provider by its `name` property. Idempotent — re-register
    overwrites the existing entry with same name (useful for hot reload)."""
    if not isinstance(provider, MemoryProvider):
        raise TypeError(f"expected MemoryProvider, got {type(provider).__name__}")
    name = str(provider.name).strip()
    if not name:
        raise ValueError("provider.name must be non-empty")
    with _REGISTRY_LOCK:
        _REGISTRY[name] = provider


def list_providers() -> list[str]:
    """Return registered provider names (sorted, for stable debug output)."""
    with _REGISTRY_LOCK:
        return sorted(_REGISTRY.keys())


def set_active_provider(name: str) -> None:
    """Switch active provider. Raises if not registered."""
    global _ACTIVE_NAME
    with _REGISTRY_LOCK:
        if name not in _REGISTRY:
            raise KeyError(f"provider {name!r} not registered (have: {sorted(_REGISTRY.keys())})")
        _ACTIVE_NAME = name


def _resolve_default_name() -> str:
    """Default = env override, else 'hybrid_v2'."""
    env = str(os.environ.get("TSUZURI_MEMORY_PROVIDER", "") or "").strip()
    return env or "hybrid_v2"


def get_active_memory_provider() -> MemoryProvider:
    """Get the currently active provider. Lazy-initializes default on first call.

    Raises if no provider is registered (caller should ensure
    register_default_providers() ran at module load).
    """
    global _ACTIVE_NAME
    with _REGISTRY_LOCK:
        if _ACTIVE_NAME is None:
            _ACTIVE_NAME = _resolve_default_name()
        provider = _REGISTRY.get(_ACTIVE_NAME)
        if provider is None:
            if _ACTIVE_NAME in ("hybrid_v2", "biomimetic"):
                _autoregister_default_providers()
                provider = _REGISTRY.get(_ACTIVE_NAME)
            if provider is None and _ACTIVE_NAME == "hybrid_v2" and not str(os.environ.get("TSUZURI_MEMORY_PROVIDER", "") or "").strip():
                provider = _REGISTRY.get("biomimetic")
                if provider is not None:
                    _ACTIVE_NAME = "biomimetic"
            if provider is None:
                raise RuntimeError(
                    f"active provider {_ACTIVE_NAME!r} not registered "
                    f"(have: {sorted(_REGISTRY.keys())})"
                )
        return provider


def _autoregister_default_providers() -> None:
    """Auto-register built-ins without touching the database."""
    try:
        from tsuzuri.memory_provider_hybrid import HybridIndexedMemoryProvider
        register_provider(HybridIndexedMemoryProvider())
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "HybridIndexedMemoryProvider auto-register failed", exc_info=True,
        )
    try:
        from tsuzuri.memory_provider_biomimetic import BiomimeticMemoryProvider
        register_provider(BiomimeticMemoryProvider())
    except Exception:
        # Defensive — never crash registry init at chat time
        import logging
        logging.getLogger(__name__).warning(
            "BiomimeticMemoryProvider auto-register failed", exc_info=True,
        )


# Eager attempt at import time. Failures are silent (registry stays empty
# and get_active_memory_provider will raise meaningful error).
try:
    _autoregister_default_providers()
except Exception:
    pass


__all__ = [
    "register_provider",
    "set_active_provider",
    "get_active_memory_provider",
    "list_providers",
]
