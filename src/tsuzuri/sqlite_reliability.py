"""Shared SQLite connection with bounded BUSY/LOCKED retries."""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Callable


def is_sqlite_busy_error(exc: BaseException) -> bool:
    message = str(exc or "").lower()
    return isinstance(exc, sqlite3.OperationalError) and (
        "locked" in message or "busy" in message
    )


class RetryingSQLiteConnection(sqlite3.Connection):
    def _retry(self, operation: Callable[[], Any]):
        try:
            attempts = max(1, int(os.getenv("MEM_SQLITE_WRITE_ATTEMPTS", "4")))
        except Exception:
            attempts = 4
        try:
            base_delay = max(0.01, float(os.getenv("MEM_SQLITE_WRITE_BACKOFF_SEC", "0.1")))
        except Exception:
            base_delay = 0.1
        for attempt in range(attempts):
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not is_sqlite_busy_error(exc) or attempt >= attempts - 1:
                    raise
                time.sleep(base_delay * (2**attempt))
        raise RuntimeError("SQLite retry loop did not execute")

    def execute(self, *args, **kwargs):
        return self._retry(lambda: super(RetryingSQLiteConnection, self).execute(*args, **kwargs))

    def executemany(self, *args, **kwargs):
        return self._retry(lambda: super(RetryingSQLiteConnection, self).executemany(*args, **kwargs))

    def executescript(self, *args, **kwargs):
        return self._retry(lambda: super(RetryingSQLiteConnection, self).executescript(*args, **kwargs))

    def commit(self):
        return self._retry(lambda: super(RetryingSQLiteConnection, self).commit())


def connect_sqlite(
    path: str | os.PathLike[str],
    *,
    timeout: float = 30.0,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    return sqlite3.connect(
        str(path),
        timeout=float(timeout),
        check_same_thread=bool(check_same_thread),
        factory=RetryingSQLiteConnection,
    )


__all__ = ["RetryingSQLiteConnection", "connect_sqlite", "is_sqlite_busy_error"]
