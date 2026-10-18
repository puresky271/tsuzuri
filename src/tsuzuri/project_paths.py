"""Where tsuzuri keeps its data.

Everything lives under one data directory: ``$TSUZURI_HOME`` if set, otherwise
``./.tsuzuri`` relative to the current working directory. Path constants are
resolved on access, so setting ``TSUZURI_HOME`` (or constructing ``Tsuzuri(data_dir=...)``)
before the first memory operation is enough.
"""
from __future__ import annotations

import os

_FILES = {
    "MEMORY_CORE_DB_PATH": "memory.db",
    "DEV_MEMORY_PATH": "snapshot.json",
    "WORLD_LEDGER_PATH": "world_ledger.json",
    "CHARACTER_DAILY_STATE_PATH": "agent_daily_state.json",
}


def data_root() -> str:
    root = str(os.getenv("TSUZURI_HOME", "") or "").strip() or os.path.join(os.getcwd(), ".tsuzuri")
    return os.path.abspath(root)


def resolve_project_path(*parts) -> str:
    cleaned = [str(part) for part in parts if str(part or "").strip()]
    return os.path.join(data_root(), *cleaned)


def describe_runtime_paths() -> dict[str, str]:
    return {name.lower(): resolve_project_path(filename) for name, filename in _FILES.items()}


def __getattr__(name: str) -> str:
    if name == "PROJECT_ROOT":
        return data_root()
    if name in _FILES:
        return resolve_project_path(_FILES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# PROJECT_ROOT and the path constants are served by __getattr__ (resolved on access).
__all__ = ["data_root", "resolve_project_path", "describe_runtime_paths", "PROJECT_ROOT", *_FILES]  # noqa: F822
