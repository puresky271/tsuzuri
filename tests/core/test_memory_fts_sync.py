"""memory_items_fts 由 trigger 按 rowid 同步（2026-10-19 外部审计 ARCH-001/002）。"""
import os
import sqlite3
import tempfile
import unittest

from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_core_impl.store_retrieval import _fts5_quoted_query


def _row(item_id: str, text: str, *, archived: int = 0, status: str = "active") -> dict:
    ts = "2026-10-19T00:00:00Z"
    return {
        "item_id": item_id,
        "character": "张三",
        "layer": "episodic",
        "memory_role": "episodic",
        "engram_type": "event",
        "item_type": "event",
        "text": text,
        "payload_json": "{}",
        "cue_json": "{}",
        "lifecycle_json": "{}",
        "salience": 0.6,
        "confidence": 0.6,
        "novelty": 0.3,
        "importance": 0.6,
        "status": status,
        "source": "test",
        "evidence_id": "",
        "created_at": ts,
        "updated_at": ts,
        "last_accessed_at": "",
        "access_count": 0,
        "archived": archived,
    }


class MemoryFtsSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)
        conn = sqlite3.connect(self.db_path)
        try:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_items_fts'"
            ).fetchone():
                self.skipTest("SQLite build 无 FTS5")
        finally:
            conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _match(self, term: str) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            return [
                str(r[0])
                for r in conn.execute(
                    "SELECT item_id FROM memory_items_fts WHERE text MATCH ?", (_fts5_quoted_query(term),)
                ).fetchall()
            ]
        finally:
            conn.close()

    def _integrity_ok(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT INTO memory_items_fts(memory_items_fts) VALUES('integrity-check')")
        finally:
            conn.close()

    def test_upsert_rewrite_moves_index_to_new_text(self):
        self.core._upsert_item(_row("fts-a", "practice guitar tonight"))
        self.assertEqual(self._match("guitar"), ["fts-a"])
        self.core._upsert_item(_row("fts-a", "practice drums tonight"))
        self.assertEqual(self._match("drums"), ["fts-a"])
        self.assertEqual(self._match("guitar"), [])
        self._integrity_ok()

    def test_archive_and_delete_leave_no_orphan_rowid(self):
        self.core._upsert_item(_row("fts-b", "lunch at the station"))
        self.core._upsert_item(_row("fts-c", "lunch at school"))
        self.core._upsert_item(_row("fts-b", "lunch at the station", archived=1, status="archived"))
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM memory_items WHERE item_id='fts-c'")
            conn.commit()
        finally:
            conn.close()
        # 孤儿 rowid 曾让 MATCH 抛 "fts5: missing row"，整条 FTS 通道被吞掉
        self.assertEqual(self._match("lunch"), [])
        self._integrity_ok()

    def test_reopen_rebuilds_legacy_drifted_index(self):
        self.core._upsert_item(_row("fts-d", "session moved to friday"))
        conn = sqlite3.connect(self.db_path)
        try:
            for name in ("memory_items_fts_ai", "memory_items_fts_ad", "memory_items_fts_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            # 旧版手写同步：不带 rowid 插入 → 索引里出现指向不存在内容行的 rowid
            conn.execute("INSERT INTO memory_items_fts(item_id, text) VALUES ('ghost', 'session ghost')")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(sqlite3.DatabaseError):
            self._match("session")
        MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)
        self.assertEqual(self._match("session"), ["fts-d"])
        self._integrity_ok()

    def test_query_quoting_survives_fts_syntax_characters(self):
        self.core._upsert_item(_row("fts-e", "learning C++ and studio schedules"))
        for raw in ('C++', 'studio"', 'a-b', 'col: studio'):
            self._match(raw)  # 不得抛 OperationalError
        self.assertEqual(self._match("studio schedules"), ["fts-e"])


if __name__ == "__main__":
    unittest.main()
