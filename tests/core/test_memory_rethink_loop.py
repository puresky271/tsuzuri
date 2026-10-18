from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timezone
from unittest.mock import patch

from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_rethink_engine import propose_memory_rethink_loop
from tsuzuri.prompts.memory_rethink_prompt import build_rethink_prompt


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class MemoryRethinkLoopTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.core = MemoryCoreStore(
            db_path=self.db_path,
            snapshot_path=os.path.join(self.tmpdir.name, "snapshot.json"),
        )
        self.core.build_character_canon_profile = lambda _char: {
            "anchors": {"identity": ["张三是 测试组 的水彩手。"]}
        }

    def _seed(
        self,
        item_id: str,
        *,
        layer: str,
        text: str,
        salience: float = 0.72,
        access_count: int = 3,
        engram_type: str = "fact",
        payload: dict | None = None,
    ) -> None:
        now = _now()
        self.core._upsert_item({
            "item_id": item_id,
            "character": "张三",
            "layer": layer,
            "memory_role": layer,
            "engram_type": engram_type,
            "item_type": engram_type,
            "text": text,
            "payload_json": json.dumps(payload or {}, ensure_ascii=False),
            "salience": salience,
            "confidence": 0.75,
            "importance": 0.58,
            "access_count": access_count,
            "source": "turn_append" if layer in {"working", "episodic"} else "test",
            "created_at": now,
            "updated_at": now,
            "embedding_text": "0.1,0.2",
        })

    def _text(self, item_id: str) -> str:
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute("SELECT text FROM memory_items WHERE item_id=?", (item_id,)).fetchone()
        return str(row[0]) if row else ""

    def test_loop_continues_without_intermediate_finish_and_reuses_run(self):
        self._seed("sem_old", layer="semantic", text="阿澈目前住在上海。")
        self._seed(
            "epi_fix",
            layer="episodic",
            text="阿澈明确纠正：现在住在杭州。",
            engram_type="fact_observation",
            payload={"fact": "阿澈明确纠正：现在住在杭州。"},
        )
        responses = iter([
            {
                "summary": "先纠正住址",
                "finish_edits": False,
                "actions": [{
                    "action_type": "rewrite",
                    "target_item_ids": ["sem_old"],
                    "resolution_type": "explicit_correction",
                    "evidence_item_ids": ["epi_fix"],
                    "new_text": "阿澈目前住在杭州，之前住在上海。",
                    "reason": "用户明确纠正当前住址",
                }],
            },
            {"summary": "已完成", "finish_edits": True, "actions": []},
        ])

        evidence = self.core.select_rethink_evidence("张三")
        result = propose_memory_rethink_loop(
            self.core,
            "张三",
            shadow=False,
            max_iterations=3,
            evidence=evidence,
            llm_caller=lambda _prompt: json.dumps(next(responses), ensure_ascii=False),
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["iterations"], 2)
        self.assertTrue(result["finish_confirmed"])
        self.assertEqual(self._text("sem_old"), "阿澈目前住在杭州，之前住在上海。")
        with closing(sqlite3.connect(self.db_path)) as conn:
            run_count = conn.execute("SELECT COUNT(*) FROM memory_rethink_runs").fetchone()[0]
            pass_rows = conn.execute(
                "SELECT DISTINCT loop_pass FROM memory_rethink_actions WHERE run_id=?",
                (result["run_id"],),
            ).fetchall()
        self.assertEqual(run_count, 1)
        self.assertEqual(pass_rows, [(0,)])
        expected_ts = max(str(item.get("updated_at", "") or "") for item in evidence)
        self.assertEqual(result["cursor_after"]["last_turn_ts"], expected_ts)

    def test_loop_duplicate_target_fails_and_rolls_back_prior_pass(self):
        self._seed("sem_old", layer="semantic", text="阿澈目前住在上海。")
        self._seed(
            "epi_fix",
            layer="episodic",
            text="阿澈明确纠正：现在住在杭州。",
            engram_type="fact_observation",
            payload={"fact": "阿澈明确纠正：现在住在杭州。"},
        )
        actions = [{
            "action_type": "rewrite",
            "target_item_ids": ["sem_old"],
            "resolution_type": "explicit_correction",
            "evidence_item_ids": ["epi_fix"],
            "new_text": "阿澈目前住在杭州，之前住在上海。",
            "reason": "用户明确纠正当前住址",
        }]
        responses = iter([
            {"summary": "pass one", "finish_edits": False, "actions": actions},
            {"summary": "repeat", "finish_edits": True, "actions": actions},
        ])

        result = propose_memory_rethink_loop(
            self.core,
            "张三",
            shadow=False,
            max_iterations=3,
            evidence=self.core.select_rethink_evidence("张三"),
            llm_caller=lambda _prompt: json.dumps(next(responses), ensure_ascii=False),
        )

        self.assertFalse(result["ok"])
        self.assertTrue(result["rolled_back"])
        self.assertEqual(self._text("sem_old"), "阿澈目前住在上海。")
        self.assertEqual(result["cursor_after"], result["cursor_before"])

    def test_insert_live_writes_semantic_lineage(self):
        self._seed("sem_anchor", layer="semantic", text="阿澈喜欢记录课程安排。")
        self._seed(
            "epi_course",
            layer="episodic",
            text="阿澈接下来一个月每周三晚上有产品课。",
            salience=0.82,
            access_count=0,
            engram_type="fact_observation",
            payload={"fact": "阿澈接下来一个月每周三晚上有产品课。"},
        )
        response = {
            "summary": "补一条稳定日程",
            "finish_edits": True,
            "actions": [{
                "action_type": "insert",
                "target_item_ids": [],
                "resolution_type": "evidence_distillation",
                "evidence_item_ids": ["epi_course"],
                "new_text": "阿澈接下来一个月每周三晚上有产品课。",
                "reason": "高显著度明确事实尚未进入 semantic 层",
            }],
        }
        result = propose_memory_rethink_loop(
            self.core,
            "张三",
            shadow=False,
            max_iterations=1,
            evidence=self.core.select_rethink_evidence("张三"),
            llm_caller=lambda _prompt: json.dumps(response, ensure_ascii=False),
        )
        self.assertTrue(result["ok"], result)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memory_items WHERE source='memory_rethink_insert'"
            ).fetchone()
            link = conn.execute(
                "SELECT 1 FROM memory_links WHERE src_item_id='epi_course' AND dst_item_id=?",
                (row["item_id"],),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["layer"], "semantic")
        self.assertGreaterEqual(float(row["salience"]), 0.60)
        self.assertIsNotNone(link)

    def test_insert_evidence_cas_detects_concurrent_change(self):
        self._seed("epi_course", layer="episodic", text="阿澈每周三有产品课。", salience=0.8)
        run_id = self.core._record_rethink_run_open("张三", [], shadow=False)
        action_pk = self.core._record_rethink_action(
            run_id,
            0,
            "insert",
            target_item_ids=[],
            old_texts=[],
            resolution_type="evidence_distillation",
            evidence_item_ids=["epi_course"],
            new_text="阿澈每周三晚上有产品课。",
            reason="蒸馏稳定事实",
            validation_status="ok",
        )
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("UPDATE memory_items SET text='并发改写后的证据' WHERE item_id='epi_course'")
            conn.commit()
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertFalse(result["ok"])
        self.assertIn("evidence_before_hash_mismatch", result["error"])

    def test_cursor_does_not_advance_when_final_finish_is_missing(self):
        self._seed("sem_anchor", layer="semantic", text="阿澈喜欢记录课程安排。")
        self._seed("epi_course", layer="episodic", text="阿澈每周三有产品课。")
        before = self.core.get_rethink_cursor("张三")
        result = propose_memory_rethink_loop(
            self.core,
            "张三",
            shadow=False,
            max_iterations=1,
            evidence=self.core.select_rethink_evidence("张三"),
            llm_caller=lambda _prompt: json.dumps({"summary": "未结束", "actions": []}),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed_incomplete")
        self.assertEqual(self.core.get_rethink_cursor("张三"), before)

    def test_loop_stops_after_unfinished_pass_without_progress(self):
        self._seed("sem_anchor", layer="semantic", text="阿澈喜欢记录课程安排。")
        calls = 0

        def _reply(_prompt: str) -> str:
            nonlocal calls
            calls += 1
            return json.dumps(
                {"summary": "没有可安全执行的编辑", "finish_edits": False, "actions": []},
                ensure_ascii=False,
            )

        result = propose_memory_rethink_loop(
            self.core,
            "张三",
            shadow=False,
            max_iterations=3,
            evidence=[],
            llm_caller=_reply,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed_incomplete")
        self.assertEqual(result["fallback_reason"], "loop_stalled_without_finish")
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(calls, 1)

    def test_loop_prompt_extensions_do_not_change_single_batch_prompt(self):
        kwargs = {
            "character": "张三",
            "candidates": [],
            "canon_profile": None,
            "evidence": [],
            "max_actions": 4,
        }
        single_system, single_user = build_rethink_prompt(**kwargs)
        explicit_defaults = build_rethink_prompt(
            **kwargs,
            enable_insert=False,
            loop_diff_context="",
        )
        loop_system, loop_user = build_rethink_prompt(
            **kwargs,
            enable_insert=True,
            loop_diff_context="pass=0 rewrite targets=[mem_a]",
        )
        self.assertEqual((single_system, single_user), explicit_defaults)
        self.assertNotIn("evidence_distillation", single_system + single_user)
        self.assertIn("evidence_distillation", loop_system + loop_user)
        self.assertIn("上一轮已完成的编辑", loop_system)

    def test_cursor_since_timestamp_is_strictly_newer(self):
        self._seed("epi_same", layer="episodic", text="阿澈说今天要开会。")
        evidence = self.core.select_rethink_evidence("张三")
        self.assertEqual(len(evidence), 1)
        self.core.advance_rethink_cursor(
            "张三",
            last_turn_ts=str(evidence[0]["updated_at"]),
        )
        self.assertEqual(
            self.core.select_rethink_evidence(
                "张三",
                since_ts=self.core.get_rethink_cursor("张三")["last_turn_ts"],
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
