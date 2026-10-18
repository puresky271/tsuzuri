"""Crash-safe file persistence primitives shared by runtime state writers."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from copy import deepcopy
from contextlib import contextmanager
from tsuzuri.time_source import utc_now
from pathlib import Path
from typing import Any, Iterator, Mapping


def _timestamp() -> str:
    return utc_now().strftime("%Y%m%dT%H%M%S_%fZ")


@contextmanager
def interprocess_file_lock(
    target: str | os.PathLike[str],
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Lock a one-byte sidecar file across processes on Windows and POSIX."""
    path = Path(target)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if handle.seek(0, os.SEEK_END) == 0:
        handle.write(b"\0")
        handle.flush()
    deadline = time.monotonic() + max(0.0, float(timeout))
    acquired = False
    try:
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"file lock timeout: {lock_path}")
                time.sleep(max(0.01, float(poll_interval)))
        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def atomic_write_bytes(path: str | os.PathLike[str], payload: bytes) -> Path:
    """Write bytes through a unique fsynced sibling and atomically replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temp_path, destination)
                break
            except PermissionError:
                if attempt >= 4:
                    raise
                time.sleep(0.05 * (2**attempt))
        return destination
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
) -> Path:
    if not isinstance(payload, Mapping):
        raise TypeError("JSON payload must be a mapping")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
        separators=None if indent is not None else (",", ":"),
    ).encode("utf-8")
    return atomic_write_bytes(path, encoded)


def preserve_corrupt_file(
    path: str | os.PathLike[str],
    *,
    label: str = "corrupt",
) -> Path | None:
    """Move an unreadable file aside without overwriting an earlier artifact."""
    source = Path(path)
    if not source.is_file():
        return None
    destination = source.with_name(f"{source.name}.{label}_{_timestamp()}.bak")
    try:
        os.replace(source, destination)
    except OSError:
        shutil.copy2(source, destination)
        source.unlink()
    return destination


def load_json_with_corrupt_backup(
    path: str | os.PathLike[str],
    *,
    default: Mapping[str, Any] | None = None,
    require_mapping: bool = True,
    label: str = "corrupt",
) -> tuple[Any, Path | None]:
    """Load JSON and preserve malformed input before returning a fresh default."""
    source = Path(path)
    if not source.exists():
        return dict(default or {}), None
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if require_mapping and not isinstance(payload, dict):
            raise ValueError("top-level JSON value must be an object")
        return payload, None
    except Exception:
        backup = preserve_corrupt_file(source, label=label)
        return dict(default or {}), backup


def merge_mapping_changes(
    base: Mapping[str, Any],
    local: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply local changes since base onto the latest mapping from disk."""
    baseline = dict(base or {})
    local_map = dict(local or {})
    merged = deepcopy(dict(current or {}))
    for key in set(baseline) | set(local_map):
        in_base = key in baseline
        in_local = key in local_map
        if not in_local:
            if in_base and key in merged and merged.get(key) == baseline.get(key):
                merged.pop(key, None)
            continue
        local_value = local_map[key]
        if in_base and local_value == baseline[key]:
            continue
        current_value = merged.get(key)
        if (
            in_base
            and isinstance(baseline[key], Mapping)
            and isinstance(local_value, Mapping)
            and isinstance(current_value, Mapping)
        ):
            merged[key] = merge_mapping_changes(baseline[key], local_value, current_value)
        else:
            merged[key] = deepcopy(local_value)
    return merged


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "interprocess_file_lock",
    "load_json_with_corrupt_backup",
    "merge_mapping_changes",
    "preserve_corrupt_file",
]
