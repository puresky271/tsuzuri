import os
import sqlite3
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

from tsuzuri.memory_core import MemoryCoreStore, utc_now


def _iso(dt) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RetrievalLogRetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        # 构造期自动 prune 对空表无副作用；每个用例自行安排数据后再调。
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _insert_log(self, *, created_at: str, run_id: str = "ret_test", candidate_item_id: str = "item-1") -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO retrieval_logs
                (run_id, character, query_text, query_bundle_json, candidate_item_id,
                 candidate_text, source, score, selected, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (run_id, "张三", "测试查询", "{}", candidate_item_id, "候选文本", "unit", 0.5, 0, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _count_logs(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM retrieval_logs;").fetchone()[0])
        finally:
            conn.close()

    def test_prune_deletes_only_rows_beyond_retention(self):
        now = utc_now()
        self._insert_log(created_at=_iso(now - timedelta(days=9)), candidate_item_id="old")
        self._insert_log(created_at=_iso(now - timedelta(days=6, hours=23)), candidate_item_id="recent")
        self._insert_log(created_at=_iso(now), candidate_item_id="fresh")

        deleted = self.core.prune_retrieval_logs()

        self.assertEqual(deleted, 1)
        self.assertEqual(self._count_logs(), 2)
        conn = sqlite3.connect(self.db_path)
        try:
            kept = {row[0] for row in conn.execute("SELECT candidate_item_id FROM retrieval_logs;")}
        finally:
            conn.close()
        self.assertEqual(kept, {"recent", "fresh"})

    def test_prune_respects_max_age_override_and_env_retention(self):
        now = utc_now()
        self._insert_log(created_at=_iso(now - timedelta(days=10)), candidate_item_id="ten-days")
        self._insert_log(created_at=_iso(now - timedelta(days=3)), candidate_item_id="three-days")

        with patch.dict(os.environ, {"MEMORY_RETRIEVAL_LOG_RETENTION_DAYS": "5"}, clear=False):
            self.assertEqual(self.core.retrieval_log_retention_days(), 5)
            self.assertEqual(self.core.prune_retrieval_logs(), 1)
        self.assertEqual(self._count_logs(), 1)

        self.assertEqual(self.core.prune_retrieval_logs(max_age_days=2), 1)
        self.assertEqual(self._count_logs(), 0)

    def test_prune_records_meta_state_for_throttle(self):
        self._insert_log(created_at=_iso(utc_now() - timedelta(days=60)))
        self.core.prune_retrieval_logs()
        meta = self.core.get_meta_state_json(MemoryCoreStore.RETRIEVAL_LOG_PRUNE_STATE_KEY, {})
        self.assertEqual(meta.get("deleted"), 1)
        self.assertEqual(meta.get("retention_days"), 7)
        self.assertTrue(str(meta.get("at", "")).endswith("Z"))

    def test_maybe_prune_throttles_within_interval(self):
        self._insert_log(created_at=_iso(utc_now() - timedelta(days=60)))
        # 第一次 _maybe_ 清理（构造时已对空表跑过一次并写入节流戳）。
        self.core._maybe_prune_retrieval_logs()
        self.assertEqual(self._count_logs(), 1)  # 节流窗口内不再清

        # 把节流戳拨到窗口之外后才会再清。
        stale = _iso(utc_now() - timedelta(hours=25))
        self.core.set_meta_state_json(
            MemoryCoreStore.RETRIEVAL_LOG_PRUNE_STATE_KEY, {"at": stale, "deleted": 0}
        )
        self.core._maybe_prune_retrieval_logs()
        self.assertEqual(self._count_logs(), 0)

    def test_maybe_prune_disabled_with_zero_interval(self):
        self._insert_log(created_at=_iso(utc_now() - timedelta(days=60)))
        with patch.dict(os.environ, {"MEMORY_RETRIEVAL_LOG_PRUNE_INTERVAL_HOURS": "0"}, clear=False):
            self.core._maybe_prune_retrieval_logs()
        self.assertEqual(self._count_logs(), 1)

    def test_prune_with_vacuum_keeps_data_intact(self):
        now = utc_now()
        self._insert_log(created_at=_iso(now - timedelta(days=9)), candidate_item_id="old")
        self._insert_log(created_at=_iso(now), candidate_item_id="fresh")

        deleted = self.core.prune_retrieval_logs(vacuum=True)

        self.assertEqual(deleted, 1)
        self.assertEqual(self._count_logs(), 1)


if __name__ == "__main__":
    unittest.main()
