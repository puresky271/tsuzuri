"""扩散激活的 Hebbian 落库按批进行（2026-10-19 外部审计 ARCH-007）。"""
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_core_impl import recall_scoring as recall_scoring


class HebbianBatchPersistTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.core = MemoryCoreStore(
            db_path=self.db_path, snapshot_path=os.path.join(self.tmpdir.name, "dev_memory_test.json")
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_spreading_activation_writes_all_edges_in_one_connection(self):
        cue = json.dumps({"entities": ["水族馆"]}, ensure_ascii=False)
        candidates = [
            {"item_id": f"m{i}", "score": 1.0 - i * 0.1, "cue_json": cue, "created_at": "2026-10-01T10:00:00Z"}
            for i in range(6)
        ]
        real_connect = self.core._connect
        calls = []

        def _counting_connect():
            calls.append(1)
            return real_connect()

        with patch.object(recall_scoring, "_get_co_store", return_value=self.core), \
                patch.object(recall_scoring, "_CO_STRENGTH_LOADED", True), \
                patch.dict(recall_scoring._CO_STRENGTH_CACHE, {}, clear=True), \
                patch.object(self.core, "_connect", side_effect=_counting_connect):
            recall_scoring._apply_spreading_activation(candidates)
            boosted = [c for c in candidates if c.get("_spread_activated")]
        self.assertEqual(len(boosted), 3)  # top3 之外的 3 条都与 top3 共现
        self.assertEqual(len(calls), 1)
        conn = sqlite3.connect(self.db_path)
        try:
            edges = conn.execute("SELECT COUNT(*) FROM co_occurrence_strength").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(edges, 9)  # 3 条候选 × 3 条 top


if __name__ == "__main__":
    unittest.main()
