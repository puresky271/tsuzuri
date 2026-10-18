"""Where weekly relationship reflections are delivered and tracked.

A reflection is a note an agent writes to the user; it is folded into the agent's
user profile once the user replies (or after a timeout). Hosts that already have a
message surface register ``memory_host`` hook ``reflection_sink`` returning an
object with the methods below; otherwise reflections live in the memory database's
own ``reflection_notes`` table.

Sink protocol:

- ``publish(*, author, text, source, meta, created_at, expires_at) -> str`` (message id)
- ``load(message_id) -> dict | None`` with keys ``message_id, author, text, source, meta, created_at``
- ``update_meta(message_id, meta) -> None``
- ``list_before(*, author, source, created_before) -> list[dict]`` (same row shape, oldest first)
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

from tsuzuri import memory_host as memory_host
from tsuzuri.sqlite_reliability import connect_sqlite

REFLECTION_SOURCE = "kairos_relationship_reflection"
REFLECTION_TTL_SEC = 24 * 3600 * 14


class SqliteReflectionSink:
    """Built-in sink backed by ``reflection_notes`` in the memory database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reflection_notes (
                message_id TEXT PRIMARY KEY,
                author TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reflection_notes_author ON reflection_notes(author, source, created_at);"
        )
        return conn

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        try:
            meta = json.loads(str(row["meta_json"] or "{}"))
        except ValueError:
            meta = {}
        return {
            "message_id": str(row["message_id"] or ""),
            "author": str(row["author"] or ""),
            "text": str(row["text"] or ""),
            "source": str(row["source"] or ""),
            "meta": meta if isinstance(meta, dict) else {},
            "created_at": float(row["created_at"] or 0.0),
        }

    def publish(self, *, author: str, text: str, source: str, meta: dict,
                created_at: float | None = None, expires_at: float | None = None) -> str:
        now = float(created_at if created_at is not None else time.time())
        message_id = uuid.uuid4().hex
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO reflection_notes(message_id, author, text, source, meta_json, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?);",
                (message_id, author, text, source, json.dumps(meta or {}, ensure_ascii=False), now,
                 float(expires_at if expires_at is not None else now + REFLECTION_TTL_SEC)),
            )
            conn.commit()
        finally:
            conn.close()
        return message_id

    def load(self, message_id: str) -> dict | None:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM reflection_notes WHERE message_id=?;", (message_id,)).fetchone()
        finally:
            conn.close()
        return self._row(row) if row is not None else None

    def update_meta(self, message_id: str, meta: dict) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE reflection_notes SET meta_json=? WHERE message_id=?;",
                (json.dumps(meta or {}, ensure_ascii=False), message_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_before(self, *, author: str, source: str, created_before: float) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM reflection_notes WHERE source=? AND author=? AND created_at<? ORDER BY created_at ASC;",
                (source, author, float(created_before)),
            ).fetchall()
        finally:
            conn.close()
        return [self._row(r) for r in rows]


def get_reflection_sink(db_path: str):
    factory = memory_host.get("reflection_sink")
    if factory is not None:
        sink = factory()
        if sink is not None:
            return sink
    return SqliteReflectionSink(db_path)


__all__ = ["REFLECTION_SOURCE", "REFLECTION_TTL_SEC", "SqliteReflectionSink", "get_reflection_sink"]
