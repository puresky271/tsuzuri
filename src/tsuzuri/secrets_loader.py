"""Configuration lookup: environment first, then an optional ``tsuzuri.toml``.

``tsuzuri.toml`` is read from the data directory (``$TSUZURI_HOME``); it holds the
same top-level keys as the environment (``LLM_API_KEY``, ``MEM_*`` flags, …).
Never commit it.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


# Keys that sync_env_from_secrets copies from tsuzuri.toml into the environment.
_RUNTIME_ENV_KEYS: tuple[str, ...] = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "MEM_ASSOC_HOP1_SHADOW_ENABLE",
    "MEM_CLUSTER_MEMORY_AUDIT_ENABLE",
    "MEM_CLUSTER_MEMORY_CREATE_ENABLE",
    "MEM_CLUSTER_MEMORY_ENABLE",
    "MEM_CLUSTER_MEMORY_MAX_CALLS",
    "MEM_CLUSTER_MEMORY_MODEL",
    "MEM_CLUSTER_MEMORY_RECALL_ENABLE",
    "MEM_CONTEXT_DUAL_READ_ENABLE",
    "MEM_CONTEXT_PRIMARY_ENABLE",
    "MEM_CORE_SUMMARY_ENABLE",
    "MEM_CORE_SUMMARY_MODEL",
    "MEM_CORE_SUMMARY_RENDER",
    "MEM_ENTITY_BOOST_ENABLE",
    "MEM_ENTITY_PROACTIVE_ENABLE",
    "MEM_ENTITY_REBUILD_ENABLE",
    "MEM_EVIDENCE_MAINTENANCE_APPLY",
    "MEM_EVOLVE_MIN_SHADOW_DAYS",
    "MEM_EVOLVE_SHADOW_ONLY",
    "MEM_FORGETTING_APPLY",
    "MEM_FORGETTING_ENABLED",
    "MEM_FORGETTING_FROZEN_DAYS",
    "MEM_FORGETTING_TOMBSTONE_DAYS",
    "MEM_JSON_BOOTSTRAP_FALLBACK_ENABLE",
    "MEM_JSON_ONLINE_EXPORT_ENABLE",
    "MEM_LLM_SELECTOR_ENABLE",
    "MEM_POLICY_LLM_ENABLE",
    "MEM_RETHINK_ALLOW_ARCHIVE",
    "MEM_RETHINK_ENABLED",
    "MEM_RETHINK_K",
    "MEM_RETHINK_MAX_ACTIONS",
    "MEM_RETHINK_MAX_ITERATIONS",
    "MEM_RETHINK_MIN_SHADOW_DAYS",
    "MEM_RETHINK_OOC_THRESHOLD",
    "MEM_RETHINK_SHADOW_ONLY",
    "MEM_SNAPSHOT_BACKUP_ENABLE",
    "MEM_SNAPSHOT_BACKUP_INTERVAL_HOURS",
    "MEM_SNAPSHOT_BACKUP_KEEP",
    "MEM_SQLITE_ENABLE",
    "MEM_SUBJECTIVE_POSTURE_ENABLE",
    "MEM_TURN_TOOLS_ENABLE",
    "MEM_TURN_TOOLS_MAX_CALLS",
    "MEM_TURN_TOOLS_MODEL",
    "MEM_TURN_TOOLS_WRITE_ENABLE",
    "TURN_FACT_SEMANTIC_ENABLE",
)


def _config_path(project_root: str | os.PathLike[str] | None = None) -> Path:
    if project_root:
        return Path(project_root) / "tsuzuri.toml"
    from tsuzuri.project_paths import data_root

    return Path(data_root()) / "tsuzuri.toml"


@lru_cache(maxsize=4)
def load_project_secrets(project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = _config_path(project_root)
    if not path.is_file():
        return {}
    import tomllib

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return payload if isinstance(payload, dict) else {}


_LLM_SETTING = re.compile(r"^(?:LLM_|OPENAI_)|_API_KEY$|_BASE_URL$")


def _offline() -> bool:
    """``Tsuzuri(llm=None)`` sets TSUZURI_OFFLINE for each call: LLM settings stay unread."""
    return os.environ.get("TSUZURI_OFFLINE", "") == "1"


def get_secret(name: str, default: Any = "", *, env_first: bool = True) -> Any:
    """Read one top-level setting, with environment variables taking priority."""
    key = str(name or "").strip()
    if not key:
        return default
    if _offline() and _LLM_SETTING.search(key):
        return default
    if env_first and key in os.environ:
        return os.environ[key]
    return load_project_secrets().get(key, default)


def env_bool(name: str, default: bool = False) -> bool:
    value = get_secret(name, None)
    raw = "" if value is None else str(value).strip().lower()
    if not raw:
        return bool(default)
    return raw in ("1", "true", "yes", "on")


def get_secret_section(name: str) -> dict[str, Any]:
    value = load_project_secrets().get(str(name or "").strip(), {})
    return dict(value) if isinstance(value, dict) else {}


def sync_env_from_secrets(project_root: str | os.PathLike[str] | None = None, *, overwrite: bool = False) -> int:
    """Copy top-level string/number settings from ``tsuzuri.toml`` into ``os.environ``."""
    synced = 0
    for key, raw in load_project_secrets(project_root).items():
        if isinstance(raw, (dict, list)) or raw is None:
            continue
        if key not in _RUNTIME_ENV_KEYS and not key.startswith(("MEM_", "LLM_", "TSUZURI_")):
            continue
        if _offline() and _LLM_SETTING.search(key):
            continue
        value = str(raw).strip()
        if value and (overwrite or key not in os.environ):
            os.environ[key] = value
            synced += 1
    return synced


def clear_secret_cache() -> None:
    load_project_secrets.cache_clear()


__all__ = ["load_project_secrets", "get_secret", "env_bool", "get_secret_section", "sync_env_from_secrets", "clear_secret_cache"]
