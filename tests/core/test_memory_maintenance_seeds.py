import json
import os
import tempfile
import unittest
from unittest.mock import patch

from tsuzuri.memory_core_impl import maintenance_seeds
from tsuzuri.memory_core_impl.memory_rows import (
    _memory_explicit_archive_matches_query,
    _memory_query_allows_system_origin,
    _memory_row_blocked_by_meta_topic,
)


class MaintenanceSeedsTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, "seeds.json")
        self._env = patch.dict(os.environ, {"MEM_MAINTENANCE_SEEDS_PATH": self.path}, clear=False)
        self._env.start()

    def tearDown(self):
        self._env.stop()
        self.tmpdir.cleanup()

    def _write(self, payload):
        with open(self.path, "w", encoding="utf-8") as fh:
            if isinstance(payload, str):
                fh.write(payload)
            else:
                json.dump(payload, fh, ensure_ascii=False)
        # mtime 分辨率不足时强制刷新缓存键
        st = os.stat(self.path)
        os.utime(self.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    def test_missing_file_means_no_seeds_and_generic_gates(self):
        seeds = maintenance_seeds.load_maintenance_seeds()
        self.assertEqual(seeds["consolidated_items"], [])
        self.assertEqual(seeds["exclude_source_ids"], {})
        self.assertTrue(_memory_query_allows_system_origin("系统最开始怎么来的"))
        self.assertFalse(_memory_query_allows_system_origin("课程项目那会儿"))
        row = {"item_id": "seed_archive_x_v1", "text": "2026-04-01 某件事", "character": "李四"}
        self.assertTrue(_memory_explicit_archive_matches_query(row, "随便问问"))
        self.assertFalse(_memory_row_blocked_by_meta_topic(row))

    def test_malformed_json_and_bad_regex_fail_soft(self):
        self._write("{not json")
        self.assertEqual(maintenance_seeds.load_maintenance_seeds()["key_events"], [])
        self._write({
            "recall_gates": {
                "explicit_archives": [{"item_id_marker": "seed_archive_x", "anchor": "(unclosed"}],
            }
        })
        row = {"item_id": "seed_archive_x_v1", "text": "旧事", "character": "*"}
        # 锚点无效时该归档不对任何问法放行，而不是抛异常
        self.assertFalse(_memory_explicit_archive_matches_query(row, "旧事"))
        self.assertTrue(_memory_row_blocked_by_meta_topic(row))

    def test_seed_terms_extend_generic_gates(self):
        self._write({
            "recall_gates": {
                "system_origin_anchor_terms": ["课程项目"],
                "meta_topic_profile_terms": ["画界面"],
                "explicit_archives": [
                    {"item_id_marker": "seed_archive_account", "text_marker": "2026-04-01", "anchor": "(账号|停用)"}
                ],
            }
        })
        self.assertTrue(_memory_query_allows_system_origin("课程项目那会儿"))
        profile = {"item_id": "p1", "text": "用户喜欢画界面", "character": "*", "item_type": "profile_fact"}
        self.assertTrue(_memory_row_blocked_by_meta_topic(profile))
        archive = {"item_id": "seed_archive_account_v1", "text": "2026-04-01 旧账号停用", "character": "*"}
        self.assertTrue(_memory_explicit_archive_matches_query(archive, "账号那次"))
        self.assertFalse(_memory_explicit_archive_matches_query(archive, "今天吃什么"))

    def test_loaded_copy_does_not_leak_mutations_into_cache(self):
        self._write({"key_events": [{"event": "e1", "source_item_ids": ["a"]}]})
        first = maintenance_seeds.load_maintenance_seeds()
        first["key_events"].append({"event": "mutated"})
        first["key_events"][0]["source_item_ids"].append("b")
        again = maintenance_seeds.load_maintenance_seeds()
        self.assertEqual(again["key_events"], [{"event": "e1", "source_item_ids": ["a"]}])


if __name__ == "__main__":
    unittest.main()
