"""WAL-safe SQLite backup and rotating runtime snapshot helpers."""

from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import closing
from tsuzuri.time_source import utc_now
from pathlib import Path


def _readonly_uri(path: Path) -> str:
    return f"file:{path.resolve().as_posix()}?mode=ro"


def sqlite_quick_check(path: str | os.PathLike[str], *, timeout: float = 5.0) -> list[str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(sqlite3.connect(_readonly_uri(source), uri=True, timeout=timeout)) as conn:
        return [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]


def backup_sqlite(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
) -> Path:
    """Create and verify a consistent SQLite backup, including live WAL pages."""
    src_path = Path(source)
    dst_path = Path(destination)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst_path.with_name(f".{dst_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with closing(sqlite3.connect(_readonly_uri(src_path), uri=True, timeout=timeout)) as src:
            with closing(sqlite3.connect(temp_path, timeout=timeout)) as dst:
                src.backup(dst)
                quick = [str(row[0]) for row in dst.execute("PRAGMA quick_check").fetchall()]
                if quick != ["ok"]:
                    raise RuntimeError(f"backup quick_check failed: {quick}")
        os.replace(temp_path, dst_path)
        return dst_path
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def restore_sqlite(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
) -> Path:
    """Restore a verified SQLite image through the SQLite backup API."""
    src_path = Path(source)
    dst_path = Path(destination)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    if sqlite_quick_check(src_path, timeout=timeout) != ["ok"]:
        raise RuntimeError(f"restore source quick_check failed: {src_path}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    destination_usable = not dst_path.exists()
    if dst_path.is_file():
        try:
            destination_usable = sqlite_quick_check(dst_path, timeout=timeout) == ["ok"]
        except Exception:
            destination_usable = False
    if not destination_usable:
        temp_path = dst_path.with_name(f".{dst_path.name}.{uuid.uuid4().hex}.restore.tmp")
        try:
            with closing(sqlite3.connect(_readonly_uri(src_path), uri=True, timeout=timeout)) as src:
                with closing(sqlite3.connect(temp_path, timeout=timeout)) as dst:
                    src.backup(dst)
                    quick = [str(row[0]) for row in dst.execute("PRAGMA quick_check").fetchall()]
                    if quick != ["ok"]:
                        raise RuntimeError(f"restored database quick_check failed: {quick}")
            Path(f"{dst_path}-wal").unlink(missing_ok=True)
            Path(f"{dst_path}-shm").unlink(missing_ok=True)
            os.replace(temp_path, dst_path)
            return dst_path
        finally:
            temp_path.unlink(missing_ok=True)
    with closing(sqlite3.connect(_readonly_uri(src_path), uri=True, timeout=timeout)) as src:
        with closing(sqlite3.connect(dst_path, timeout=timeout)) as dst:
            src.backup(dst)
            quick = [str(row[0]) for row in dst.execute("PRAGMA quick_check").fetchall()]
            if quick != ["ok"]:
                raise RuntimeError(f"restored database quick_check failed: {quick}")
    return dst_path


def prune_rotating_files(
    directory: str | os.PathLike[str],
    pattern: str,
    *,
    keep: int,
) -> list[Path]:
    root = Path(directory)
    files = sorted(
        (path for path in root.glob(pattern) if path.is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    removed: list[Path] = []
    for path in files[: max(0, len(files) - max(1, int(keep)))]:
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def create_rotating_sqlite_backup(
    source: str | os.PathLike[str],
    directory: str | os.PathLike[str],
    *,
    prefix: str = "memory_core_backup",
    keep: int = 7,
    reason: str = "scheduled",
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
    safe_reason = "".join(ch for ch in str(reason or "scheduled") if ch.isalnum() or ch in "_-")[:32]
    destination = root / f"{prefix}_{timestamp}_{safe_reason or 'scheduled'}.sqlite3"
    backup_sqlite(source, destination)
    prune_rotating_files(root, f"{prefix}_*.sqlite3", keep=keep)
    return destination


__all__ = [
    "backup_sqlite",
    "create_rotating_sqlite_backup",
    "prune_rotating_files",
    "restore_sqlite",
    "sqlite_quick_check",
]
