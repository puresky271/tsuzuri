from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tsuzuri.memory_api import MemoryApi
from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.turn_context_workspace import compile_character_state_frame


class MemoryCoreSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.core = MemoryCoreStore(
            db_path=self.db_path,
            snapshot_path=os.path.join(self.tmpdir.name, "snapshot.json"),
        )
        self.api = MemoryApi(self.core)

    def test_fixed_id_upsert_replaces_single_row(self):
        first = self.core.upsert_core_summary("张三", "第一版跨日背景。")
        second = self.core.upsert_core_summary("张三", "第二版跨日背景。")
        self.assertEqual(first["item_id"], "core_summary:张三")
        self.assertEqual(second["item_id"], first["item_id"])
        with closing(sqlite3.connect(self.db_path)) as conn:
            count, text = conn.execute(
                "SELECT COUNT(*), MAX(text) FROM memory_items WHERE item_type='core_summary' AND character='张三'"
            ).fetchone()
        self.assertEqual(count, 1)
        self.assertEqual(text, "第二版跨日背景。")

    def test_rethink_candidates_exclude_core_summary(self):
        self.core.upsert_core_summary("张三", "阿澈长期在做角色 AI 项目。")
        self.core._upsert_item({
            "item_id": "normal-semantic",
            "character": "张三",
            "layer": "semantic",
            "memory_role": "semantic",
            "engram_type": "fact",
            "item_type": "fact",
            "text": "阿澈喜欢柠檬茶。",
            "importance": 0.58,
            "confidence": 0.6,
            "access_count": 5,
            "embedding_text": "0.1",
        })
        ids = {item["item_id"] for item in self.core.select_rethink_candidates("张三")}
        self.assertIn("normal-semantic", ids)
        self.assertNotIn("core_summary:张三", ids)

    def test_generic_recall_candidates_exclude_core_summary(self):
        self.core.upsert_core_summary("张三", "阿澈长期在做角色 AI 项目。")
        rows = self.core._query_items(
            "张三",
            query_text="你还记得角色 AI 项目吗？",
        )
        self.assertNotIn("core_summary:张三", {row["item_id"] for row in rows})

    def test_recall_api_rejects_core_summary_from_alternate_candidates(self):
        injected = {
            "item_id": "core_summary:张三",
            "character": "张三",
            "layer": "semantic",
            "item_type": "core_summary",
            "text": "阿澈长期在做角色 AI 项目。",
        }
        with patch.object(self.core, "_query_items", return_value=[injected]):
            recalled = self.core.retrieve_memory_context(
                "张三",
                "你还记得角色 AI 项目吗？",
                max_items=2,
                touch_selected=False,
                log_retrieval=False,
            )
        self.assertEqual(recalled["rows"], [])

    def test_memory_read_plan_rejects_cached_core_summary(self):
        injected = {
            "item_id": "core_summary:张三",
            "character": "张三",
            "layer": "semantic",
            "item_type": "core_summary",
            "text": "阿澈长期在做角色 AI 项目。",
            "score": 1.0,
        }
        with patch.object(
            self.core,
            "retrieve_memory_context",
            return_value={"run_id": "cached", "rows": [injected], "collective_rows": []},
        ):
            planned = self.core.plan_memory_read(
                "张三",
                "你还记得角色 AI 项目吗？",
                max_items=2,
            )
        self.assertEqual(planned["retrieved_rows"], [])
        self.assertEqual(planned["memory_surface_plan"]["selected_surface_rows"], [])

    def test_default_empty_render_is_byte_identical_and_nonempty_uses_generator_budget(self):
        baseline = compile_character_state_frame("张三", use_structured_sources=True)["text"]
        empty = compile_character_state_frame(
            "张三",
            use_structured_sources=True,
            core_summary="",
        )["text"]
        full_summary = "甲" * 268
        rendered_full = compile_character_state_frame(
            "张三",
            use_structured_sources=True,
            core_summary=full_summary,
        )["text"]
        rendered_bounded = compile_character_state_frame(
            "张三",
            use_structured_sources=True,
            core_summary="乙" * 500,
        )["text"]
        self.assertEqual(empty, baseline)
        self.assertIn("【跨日核心背景（常驻）】", rendered_full)
        full_line = next(line for line in rendered_full.splitlines() if line.startswith("- 甲"))
        bounded_line = next(line for line in rendered_bounded.splitlines() if line.startswith("- 乙"))
        self.assertEqual(full_line[2:], full_summary)
        self.assertLessEqual(len(bounded_line[2:]), 360)

    def test_memory_api_item_type_filter_and_update(self):
        self.core.upsert_core_summary("张三", "旧摘要。")
        listed = self.api.list_memories("张三", item_type="core_summary")
        self.assertEqual([item["item_id"] for item in listed["items"]], ["core_summary:张三"])
        updated = self.api.update_core_summary("张三", "开发者手工修正后的摘要。", reason="纠错")
        self.assertTrue(updated["ok"])
        self.assertEqual(self.core.get_core_summary("张三")["text"], "开发者手工修正后的摘要。")

    def test_delete_all_exempts_core_summary(self):
        self.core.upsert_core_summary("张三", "保留摘要。")
        self.core._upsert_item({
            "item_id": "ordinary",
            "character": "张三",
            "layer": "semantic",
            "item_type": "fact",
            "text": "普通事实",
            "embedding_text": "0.1",
        })
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            result = self.api.delete_all_memories("张三", confirm=True)
        self.assertTrue(result["ok"])
        self.assertIsNotNone(self.core.get_core_summary("张三"))
        with closing(sqlite3.connect(self.db_path)) as conn:
            archived = conn.execute("SELECT archived FROM memory_items WHERE item_id='ordinary'").fetchone()[0]
        self.assertEqual(archived, 1)

    pass  # upstream-only test removed

    def test_kairos_rebuild_is_idempotent_and_uses_fixed_row(self):
        from tsuzuri import kairos_dream as kairos_dream

        capsules = [{"items": [{"item_id": "dc-1", "text": "阿澈长期在开发角色 AI。"}]}]
        resolution = {
            "items": [{
                "item_id": "dc-1",
                "text": "阿澈长期在开发角色 AI。",
                "kind": "fact",
                "status": "active",
            }]
        }
        with patch("tsuzuri.chat_compact.load_daily_continuity_capsules_for_character", return_value=capsules), \
             patch("tsuzuri.daily_continuity.resolver.resolve_continuity_items", return_value=resolution):
            first = kairos_dream._run_core_summary_rebuild(
                self.core,
                "张三",
                llm_caller=lambda _prompt: "阿澈长期在开发角色 AI，并持续维护角色记忆系统。",
            )
            second = kairos_dream._run_core_summary_rebuild(
                self.core,
                "张三",
                llm_caller=lambda _prompt: self.fail("idempotent rebuild must not call LLM"),
            )
        self.assertEqual(first["status"], "rebuilt")
        self.assertEqual(second["status"], "idempotent")
        self.assertEqual(first["item_id"], "core_summary:张三")

    def test_schema_does_not_create_test_agent_tables_in_memory_core(self):
        with closing(self.core._connect()) as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        self.assertNotIn("test_runs", tables)
        self.assertNotIn("test_question_scores", tables)

    def test_cluster_summarize_write_uses_shared_write_transaction(self):
        import inspect

        source = inspect.getsource(MemoryCoreStore.cluster_and_summarize_memories)
        self.assertIn("self._run_write_transaction(_write_cluster_parent)", source)
        self.assertNotIn("conn.commit()", source)

    def test_kairos_rebuild_consumes_cluster_capsules(self):
        """Phase 4:高显著 cluster 的 capsule 作为证据进入 core_summary rebuild prompt。"""
        from tsuzuri import kairos_dream as kairos_dream
        from tsuzuri.cluster_memory import write_cluster_memory

        now_iso = datetime.now(timezone.utc).isoformat()
        # 2026-08-30: 日期改相对今日（固定日期会随真实时间滚出 14 天 lookback）
        recent_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
        for i in range(3):
            self.core._upsert_item({
                "item_id": f"cm-{i}",
                "character": "张三",
                "layer": "episodic",
                "memory_role": "episodic",
                "engram_type": "event",
                "item_type": "event",
                "text": f"江之岛片段{i}",
                "salience": 0.7,
                "importance": 0.6,
                "confidence": 0.6,
                "created_at": now_iso,
                "updated_at": now_iso,
                "topic_labels_json": '["travel"]',
                "embedding_text": "0.1",
            })
        write_cluster_memory(self.core, "张三", {
            "cluster_kind": "event",
            "item_type": "event_cluster",
            "title": "江之岛一日游",
            "text": "阿澈和张三在江之岛看海，沿着海岸走了很久，傍晚一起吃了海鲜盖饭。" * 2,
            "capsule": "阿澈和张三的江之岛一日游",
            "timeline": [],
            "people": ["阿澈"],
            "places": ["江之岛"],
            "start_at": recent_date,
            "end_at": recent_date,
            "topic_labels": ["travel"],
            "significance": 7,
            "confidence": 0.9,
            "salience": 0.8,
            "importance": 0.7,
            "member_ids": ["cm-0", "cm-1", "cm-2"],
            "evidence_ids": [],
            "canonical_key": "clu_test_capsule_1",
        })

        captured = {}

        def _caller(prompt):
            captured["prompt"] = prompt
            return "阿澈与张三在江之岛一日游后关系更近。"

        with patch("tsuzuri.chat_compact.load_daily_continuity_capsules_for_character", return_value=[]), \
             patch("tsuzuri.daily_continuity.resolver.resolve_continuity_items", return_value={"items": []}), \
             patch.dict(os.environ, {"MEM_CLUSTER_MEMORY_ENABLE": "1"}):
            result = kairos_dream._run_core_summary_rebuild(self.core, "张三", llm_caller=_caller)
        self.assertEqual(result["status"], "rebuilt")
        # 无 continuity 证据时 cluster capsule 单独驱动 rebuild;prompt 含 capsule 行
        self.assertIn("江之岛一日游", captured["prompt"])
        self.assertIn("event_cluster", captured["prompt"])

    def test_kairos_rebuild_skips_cluster_capsules_when_flag_off(self):
        """flag 关:cluster 行存在也不进 rebuild(无 continuity 证据 → no_input)。"""
        from tsuzuri import kairos_dream as kairos_dream

        with patch("tsuzuri.chat_compact.load_daily_continuity_capsules_for_character", return_value=[]), \
             patch("tsuzuri.daily_continuity.resolver.resolve_continuity_items", return_value={"items": []}), \
             patch.dict(os.environ, {"MEM_CLUSTER_MEMORY_ENABLE": "0"}):
            result = kairos_dream._run_core_summary_rebuild(
                self.core, "张三", llm_caller=lambda _p: self.fail("flag off 不应调 LLM")
            )
        self.assertEqual(result["status"], "no_input")


if __name__ == "__main__":
    unittest.main()
