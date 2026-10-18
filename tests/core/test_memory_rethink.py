"""Unit tests for Phase R.1 LLM-driven memory rethink (2026-05-16).

覆盖：
  · memory_core 端的 schema + select_candidates + record + apply + revert
  · memory_rethink_engine 端的 parse + validate + ooc score + e2e propose

不依赖真 LLM（用 injected llm_caller）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tsuzuri.memory_core import (
    MemoryCoreStore,
    mem_rethink_k,
    mem_rethink_max_actions,
    mem_rethink_min_shadow_days,
    mem_rethink_shadow_only,
)
from tsuzuri.memory_rethink_engine import (
    ParsedAction,
    _call_rethink_llm,
    parse_rethink_batch,
    parse_rethink_response,
    propose_memory_rethink,
    score_action_ooc,
    validate_action,
)
from tsuzuri.prompts.memory_rethink_prompt import (
    build_rethink_prompt,
    render_candidates_block,
    render_canon_summary,
)


def _ts_iso(offset_days: float = 0.0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=offset_days)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


class RethinkLlmCallTests(unittest.TestCase):
    @staticmethod
    def _response(content: str, total_tokens: int) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=total_tokens),
        )

    def test_env_llm_retries_once_after_empty_content(self):
        create = MagicMock(side_effect=[
            self._response("", 7),
            self._response('{"finish_edits": true, "actions": []}', 11),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.invalid/v1",
            "LLM_MODEL": "test-model",
        }, clear=False), patch("openai.OpenAI", return_value=client):
            raw, meta = _call_rethink_llm("system", "user")

        self.assertEqual(raw, '{"finish_edits": true, "actions": []}')
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[0], create.call_args_list[1])
        self.assertEqual(meta["model"], "test-model")
        self.assertEqual(meta["attempts"], 2)
        self.assertEqual(meta["total_tokens"], 18)
        self.assertEqual(meta["fallback_reason"], "")

    def test_env_llm_reports_empty_after_single_retry(self):
        create = MagicMock(side_effect=[
            self._response("", 5),
            self._response("   ", 6),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with patch.dict(os.environ, {
            "LLM_API_KEY": "test-key",
            "LLM_BASE_URL": "https://llm.invalid/v1",
            "LLM_MODEL": "test-model",
        }, clear=False), patch("openai.OpenAI", return_value=client):
            raw, meta = _call_rethink_llm("system", "user")

        self.assertEqual(raw, "")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(meta["attempts"], 2)
        self.assertEqual(meta["total_tokens"], 11)
        self.assertEqual(meta["fallback_reason"], "env_llm_empty_response_after_retry")


class MemoryRethinkSchemaTests(unittest.TestCase):
    """schema migration + 基础 record helpers"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_schema_migration_creates_tables(self):
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('memory_rethink_runs', 'memory_rethink_actions');"
            ).fetchall()
            names = {r[0] for r in rows}
            self.assertIn("memory_rethink_runs", names)
            self.assertIn("memory_rethink_actions", names)
        finally:
            conn.close()

    def test_record_run_open_returns_run_id(self):
        candidates = [
            {"item_id": "mem_1", "text": "阿澈 喜欢黑咖啡", "importance": 0.62, "confidence": 0.65, "access_count": 5, "cluster_member_ids": ["mem_1"], "cluster_size": 1, "score_breakdown": {"final_score": 0.5}},
        ]
        run_id = self.core._record_rethink_run_open("张三", candidates, shadow=True, ooc_threshold=0.45)
        self.assertTrue(run_id.startswith("reth_"))
        # row 存在
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM memory_rethink_runs WHERE run_id=?", (run_id,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["character"], "张三")
            self.assertEqual(int(row["shadow"]), 1)
            self.assertEqual(row["status"], "shadow_planned")
            self.assertEqual(int(row["candidates_count"]), 1)
        finally:
            conn.close()

    def test_record_action_creates_audit_row(self):
        run_id = self.core._record_rethink_run_open("张三", [], shadow=True)
        action_pk = self.core._record_rethink_action(
            run_id, 0, "rewrite",
            target_item_ids=["mem_x"],
            old_texts=["原文本"],
            new_text="改写后的文本",
            reason="更准确",
            ooc_score=0.1,
            ooc_verdict="ok",
            ooc_features=[],
            validation_status="ok",
        )
        self.assertGreater(action_pk, 0)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM memory_rethink_actions WHERE action_pk=?", (action_pk,)).fetchone()
            self.assertEqual(row["action_type"], "rewrite")
            self.assertEqual(row["new_text"], "改写后的文本")
            self.assertEqual(row["validation_status"], "ok")
            self.assertEqual(int(row["applied"]), 0)
        finally:
            conn.close()


def _insert_memory_item(
    db_path: str,
    *,
    item_id: str,
    character: str = "张三",
    layer: str = "semantic",
    text: str,
    importance: float = 0.62,
    confidence: float = 0.65,
    salience: float = 0.55,
    access_count: int = 3,
    updated_at: str | None = None,
) -> None:
    ts = updated_at or _ts_iso()
    role = "working" if layer == "working" else ("episodic" if layer == "episodic" else "semantic")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_items
            (item_id, character, layer, memory_role, engram_type, item_type, text,
             payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance,
             status, source, evidence_id, created_at, updated_at, last_accessed_at,
             access_count, archived, embedding_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, '{}', '{}', '{}', ?, ?, 0.35, ?, 'active', 'test', '', ?, ?, '', ?, 0, '');
            """,
            (item_id, character, layer, role, "fact", "fact", text,
             salience, confidence, importance, ts, ts, access_count),
        )
        try:
            conn.execute("INSERT OR REPLACE INTO memory_items_fts(item_id, text) VALUES (?, ?);", (item_id, text))
        except Exception:
            pass
        conn.commit()
    finally:
        conn.close()


class SelectRethinkCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_empty_character_returns_empty(self):
        out = self.core.select_rethink_candidates("张三", k=10)
        self.assertEqual(out, [])

    def test_mid_importance_high_access_gets_picked(self):
        # 中区 importance + 高 access → 应被选
        _insert_memory_item(self.db_path, item_id="mem_target", text="阿澈喜欢黑咖啡",
                            importance=0.58, access_count=8)
        # 高 importance（已稳定）不选
        _insert_memory_item(self.db_path, item_id="mem_high_imp", text="阿澈的小名是阿青",
                            importance=0.92, access_count=15)
        # 低 importance（让规则衰减处理）不选
        _insert_memory_item(self.db_path, item_id="mem_low_imp", text="阿澈说过晚安",
                            importance=0.22, access_count=1)

        out = self.core.select_rethink_candidates("张三", k=5)
        self.assertGreater(len(out), 0)
        # target 应该在 top
        ids = [c["item_id"] for c in out]
        self.assertIn("mem_target", ids)
        # decision_score: target > low_imp
        target_score = next(c["score_breakdown"]["decision_score"] for c in out if c["item_id"] == "mem_target")
        self.assertGreater(target_score, 0.0)

    def test_redundant_cluster_detected(self):
        # 3 条几乎同文记忆 → 应聚成 cluster。
        # 注意：_lexical_similarity 由 bigram + jieba word_jaccard 组成、
        # 生产环境 jieba 在、相似句能到 0.6+；这里测试 env 没 jieba、
        # 单靠 bigram 极端相似才能跨 0.35。所以测试用真重复文本验证 cluster 算法。
        _insert_memory_item(self.db_path, item_id="mem_a", text="阿澈喜欢喝黑咖啡",
                            importance=0.60, access_count=4)
        _insert_memory_item(self.db_path, item_id="mem_b", text="阿澈喜欢喝黑咖啡",
                            importance=0.58, access_count=3)
        _insert_memory_item(self.db_path, item_id="mem_c", text="阿澈喜欢喝黑咖啡",
                            importance=0.55, access_count=2)
        # 一条无关 — 不进 cluster
        _insert_memory_item(self.db_path, item_id="mem_z", text="阿澈在大阪上班开会",
                            importance=0.60, access_count=3)

        out = self.core.select_rethink_candidates("张三", k=10, redundancy_threshold=0.35)
        # 至少 1 条候选有 cluster_size >= 2
        cluster_heads = [c for c in out if c["cluster_size"] >= 2]
        self.assertGreater(len(cluster_heads), 0)
        head = cluster_heads[0]
        # cluster head 的 score_breakdown.cluster_score > 0
        self.assertGreater(head["score_breakdown"]["cluster_score"], 0.0)
        self.assertGreaterEqual(len(head["cluster_member_ids"]), 2)

    def test_non_semantic_excluded(self):
        # 同样 mid_importance + acc 高、但 layer=episodic → 不选
        _insert_memory_item(self.db_path, item_id="mem_ep", text="阿澈昨天说他累",
                            layer="episodic", importance=0.58, access_count=8)
        out = self.core.select_rethink_candidates("张三", k=10)
        ids = [c["item_id"] for c in out]
        self.assertNotIn("mem_ep", ids)

    def test_other_character_excluded(self):
        _insert_memory_item(self.db_path, item_id="mem_wangwu", character="王五",
                            text="阿澈喜欢黑咖啡", importance=0.58, access_count=8)
        out = self.core.select_rethink_candidates("张三", k=10)
        ids = [c["item_id"] for c in out]
        self.assertNotIn("mem_wangwu", ids)


class SelectRethinkEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_evidence_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_rethink_defaults_to_live_apply_without_forced_shadow_window(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(mem_rethink_shadow_only())
            self.assertEqual(mem_rethink_min_shadow_days(), 0)
            shadow_only, meta = self.core._effective_rethink_shadow_mode()
        self.assertFalse(shadow_only)
        self.assertFalse(meta["base_shadow"])
        self.assertFalse(meta["forced_shadow"])
        self.assertEqual(meta["min_days"], 0)

    def test_guest_and_other_character_rows_are_excluded(self):
        rows = (
            ("normal_evidence", "张三", {"fact": "阿澈最近在准备考试。", "memory_meta": {"mode_scope": "normal"}}),
            ("guest_mode", "张三", {"fact": "访客的私密事实。", "memory_meta": {"mode_scope": "guest"}}),
            ("guest_audience", "张三", {"turn_fact_card": {"fact": "另一位访客的私密事实。", "audience_scope": "guest:session-1"}}),
            ("other_character", "李四", {"fact": "只属于李四的证据。", "memory_meta": {"mode_scope": "normal"}}),
        )
        for item_id, character, payload in rows:
            _insert_memory_item(
                self.db_path,
                item_id=item_id,
                character=character,
                layer="episodic",
                text=str(payload.get("fact", "evidence")),
                updated_at=_ts_iso(),
            )
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE memory_items SET engram_type='fact_observation', payload_json=? WHERE item_id=?",
                    (json.dumps(payload, ensure_ascii=False), item_id),
                )
                conn.commit()
            finally:
                conn.close()

        evidence = self.core.select_rethink_evidence("张三", k=10)
        ids = {str(item.get("item_id", "")) for item in evidence}
        self.assertEqual(ids, {"normal_evidence"})


class ApplyRethinkActionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_validated_action(self, action_type: str, target_ids: list[str], new_text: str = "", reason: str = "") -> int:
        """直接写一条 action audit、validation_status='ok'。返回 action_pk。"""
        run_id = self.core._record_rethink_run_open("张三", [], shadow=True)
        conn = sqlite3.connect(self.db_path)
        try:
            old_texts = [
                str(conn.execute("SELECT text FROM memory_items WHERE item_id=?", (item_id,)).fetchone()[0])
                for item_id in target_ids
            ]
        finally:
            conn.close()
        return self.core._record_rethink_action(
            run_id, 0, action_type,
            target_item_ids=target_ids,
            old_texts=old_texts,
            new_text=new_text,
            reason=reason or "test",
            ooc_score=0.05,
            ooc_verdict="ok",
            ooc_features=[],
            validation_status="ok",
        )

    def test_rewrite_shadow_keeps_text_unchanged(self):
        _insert_memory_item(self.db_path, item_id="mem_rw", text="原文本旧版")
        action_pk = self._make_validated_action("rewrite", ["mem_rw"], new_text="改写后的文本")
        result = self.core.apply_rethink_action(action_pk, shadow=True)
        self.assertTrue(result["ok"])
        # 真实 memory_items.text 没变
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT text FROM memory_items WHERE item_id='mem_rw'").fetchone()
            self.assertEqual(row[0], "原文本旧版")
        finally:
            conn.close()
        # audit 行有 before/after
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT applied, apply_status, before_json, after_json FROM memory_rethink_actions WHERE action_pk=?", (action_pk,)).fetchone()
            self.assertEqual(int(row["applied"]), 0)
            self.assertEqual(row["apply_status"], "simulated")
            before = json.loads(row["before_json"])
            after = json.loads(row["after_json"])
            self.assertIn("mem_rw", before)
            self.assertIn("mem_rw", after)
            # after.row.text 是新文本
            self.assertEqual(after["mem_rw"]["row"]["text"], "改写后的文本")
            self.assertTrue(after["mem_rw"].get("shadow_simulated"))
        finally:
            conn.close()

    def test_rewrite_real_mutates_memory_items(self):
        _insert_memory_item(self.db_path, item_id="mem_rw2", text="原文本旧版")
        action_pk = self._make_validated_action("rewrite", ["mem_rw2"], new_text="改写后的真实文本")
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertTrue(result["ok"])
        # 真实 memory_items.text 变了
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT text, embedding_text FROM memory_items WHERE item_id='mem_rw2'").fetchone()
            self.assertEqual(row[0], "改写后的真实文本")
            # embedding 被清掉、待 backfill
            self.assertEqual(row[1], "")
        finally:
            conn.close()

    def test_merge_real_archives_originals_and_creates_new(self):
        _insert_memory_item(self.db_path, item_id="mem_m1", text="阿澈喜欢黑咖啡", access_count=3)
        _insert_memory_item(self.db_path, item_id="mem_m2", text="阿澈不爱拿铁", access_count=2)
        _insert_memory_item(self.db_path, item_id="mem_m3", text="阿澈咖啡不加奶", access_count=1)
        action_pk = self._make_validated_action(
            "merge", ["mem_m1", "mem_m2", "mem_m3"],
            new_text="阿澈咖啡偏好：黑咖啡、不加奶、不喝拿铁。",
            reason="3 条合并",
        )
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertTrue(result["ok"])
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 原 3 条 archived=1 + status='merged_by_rethink'
            for iid in ["mem_m1", "mem_m2", "mem_m3"]:
                row = conn.execute("SELECT archived, status FROM memory_items WHERE item_id=?", (iid,)).fetchone()
                self.assertEqual(int(row["archived"]), 1)
                self.assertEqual(row["status"], "merged_by_rethink")
            # 有一条新 item 承载合并后文本
            new_rows = conn.execute(
                "SELECT item_id, text FROM memory_items WHERE source='memory_rethink_merge' AND character='张三'"
            ).fetchall()
            self.assertEqual(len(new_rows), 1)
            self.assertEqual(new_rows[0]["text"], "阿澈咖啡偏好：黑咖啡、不加奶、不喝拿铁。")
            # merge_into link 也写了
            links = conn.execute(
                "SELECT src_item_id, dst_item_id, link_type FROM memory_links WHERE link_type='rethink_merge_into'"
            ).fetchall()
            self.assertEqual(len(links), 3)
        finally:
            conn.close()

    def test_archive_real_sets_archived(self):
        _insert_memory_item(self.db_path, item_id="mem_arch", text="阿澈 3 月要考研")
        action_pk = self._make_validated_action("archive", ["mem_arch"], reason="3 月已结束、过期")
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertTrue(result["ok"])
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT archived, status FROM memory_items WHERE item_id='mem_arch'").fetchone()
            self.assertEqual(int(row["archived"]), 1)
            self.assertEqual(row["status"], "archived_by_rethink")
        finally:
            conn.close()

    def test_non_semantic_target_rejected(self):
        _insert_memory_item(self.db_path, item_id="mem_ep", text="昨天聊到加班", layer="episodic")
        action_pk = self._make_validated_action("rewrite", ["mem_ep"], new_text="改写")
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertFalse(result["ok"])
        self.assertIn("non_semantic_target", str(result.get("error", "")))

    def test_already_applied_rejected(self):
        _insert_memory_item(self.db_path, item_id="mem_dup", text="原文本")
        action_pk = self._make_validated_action("rewrite", ["mem_dup"], new_text="新")
        self.core.apply_rethink_action(action_pk, shadow=True)
        # 第二次 apply → 错误
        result = self.core.apply_rethink_action(action_pk, shadow=True)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "already_applied")

    def test_stale_old_text_rejected_without_mutation(self):
        _insert_memory_item(self.db_path, item_id="mem_stale", text="提案时文本")
        action_pk = self._make_validated_action("rewrite", ["mem_stale"], new_text="模型建议文本")
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE memory_items SET text='提案后合法更新' WHERE item_id='mem_stale'")
            conn.commit()
        finally:
            conn.close()
        result = self.core.apply_rethink_action(action_pk, shadow=False)
        self.assertFalse(result["ok"])
        self.assertIn("stale_conflict", result["error"])
        conn = sqlite3.connect(self.db_path)
        try:
            text = conn.execute("SELECT text FROM memory_items WHERE item_id='mem_stale'").fetchone()[0]
            self.assertEqual(text, "提案后合法更新")
        finally:
            conn.close()

    def test_atomic_batch_rolls_back_first_action_when_second_is_stale(self):
        _insert_memory_item(self.db_path, item_id="mem_batch_a", text="A old")
        _insert_memory_item(self.db_path, item_id="mem_batch_b", text="B old")
        run_id = self.core._record_rethink_run_open("张三", [], shadow=False)
        action_a = self.core._record_rethink_action(
            run_id,
            0,
            "rewrite",
            target_item_ids=["mem_batch_a"],
            old_texts=["A old"],
            new_text="A new",
            reason="unit",
            validation_status="ok",
        )
        action_b = self.core._record_rethink_action(
            run_id,
            1,
            "rewrite",
            target_item_ids=["mem_batch_b"],
            old_texts=["B old"],
            new_text="B new",
            reason="unit",
            validation_status="ok",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE memory_items SET text='B legitimate update' WHERE item_id='mem_batch_b'")
            conn.commit()
        finally:
            conn.close()

        result = self.core.apply_rethink_actions([action_a, action_b], shadow=False)
        self.assertFalse(result["ok"])
        self.assertIn("stale_conflict", result["error"])
        conn = sqlite3.connect(self.db_path)
        try:
            rows = dict(conn.execute(
                "SELECT item_id, text FROM memory_items WHERE item_id IN ('mem_batch_a','mem_batch_b')"
            ).fetchall())
            statuses = dict(conn.execute(
                "SELECT action_pk, apply_status FROM memory_rethink_actions WHERE action_pk IN (?, ?)",
                (action_a, action_b),
            ).fetchall())
        finally:
            conn.close()
        self.assertEqual(rows["mem_batch_a"], "A old")
        self.assertEqual(rows["mem_batch_b"], "B legitimate update")
        self.assertEqual(statuses[action_a], "batch_rolled_back")
        self.assertEqual(statuses[action_b], "stale_conflict")


class RevertRethinkRunTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_revert_rewrite_restores_original_text(self):
        _insert_memory_item(self.db_path, item_id="mem_rev", text="原文本")
        run_id = self.core._record_rethink_run_open("张三", [], shadow=False)
        action_pk = self.core._record_rethink_action(
            run_id, 0, "rewrite",
            target_item_ids=["mem_rev"],
            old_texts=["原文本"],
            new_text="改写后",
            reason="test",
            validation_status="ok",
        )
        self.core.apply_rethink_action(action_pk, shadow=False)
        # 验证已改
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT text FROM memory_items WHERE item_id='mem_rev'").fetchone()
            self.assertEqual(row[0], "改写后")
        finally:
            conn.close()
        # revert
        result = self.core.revert_rethink_run(run_id, reason="unit_test")
        self.assertTrue(result["ok"])
        self.assertEqual(result["reverted"], 1)
        # 验证已还原
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT text FROM memory_items WHERE item_id='mem_rev'").fetchone()
            self.assertEqual(row[0], "原文本")
        finally:
            conn.close()

    def test_revert_merge_archives_new_item(self):
        _insert_memory_item(self.db_path, item_id="mem_mg1", text="A")
        _insert_memory_item(self.db_path, item_id="mem_mg2", text="B")
        run_id = self.core._record_rethink_run_open("张三", [], shadow=False)
        action_pk = self.core._record_rethink_action(
            run_id, 0, "merge",
            target_item_ids=["mem_mg1", "mem_mg2"],
            old_texts=["A", "B"],
            new_text="A 和 B 的合并",
            reason="cluster",
            validation_status="ok",
        )
        self.core.apply_rethink_action(action_pk, shadow=False)
        # 找到新建的 merge item
        conn = sqlite3.connect(self.db_path)
        try:
            new_iid_row = conn.execute(
                "SELECT item_id FROM memory_items WHERE source='memory_rethink_merge' LIMIT 1"
            ).fetchone()
            self.assertIsNotNone(new_iid_row)
            new_iid = new_iid_row[0]
        finally:
            conn.close()
        # revert
        result = self.core.revert_rethink_run(run_id, reason="unit_test")
        self.assertTrue(result["ok"], result)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 原 2 条已还原
            for iid in ["mem_mg1", "mem_mg2"]:
                row = conn.execute("SELECT archived, status FROM memory_items WHERE item_id=?", (iid,)).fetchone()
                self.assertEqual(int(row["archived"]), 0)
                self.assertEqual(row["status"], "active")
            # 新 merge item 已 archive
            row = conn.execute("SELECT archived, status FROM memory_items WHERE item_id=?", (new_iid,)).fetchone()
            self.assertEqual(int(row["archived"]), 1)
            self.assertEqual(row["status"], "reverted_rethink_merge")
        finally:
            conn.close()

    def test_revert_refuses_to_overwrite_later_legitimate_update(self):
        _insert_memory_item(self.db_path, item_id="mem_rev_cas", text="before rethink")
        run_id = self.core._record_rethink_run_open("张三", [], shadow=False)
        action_pk = self.core._record_rethink_action(
            run_id,
            0,
            "rewrite",
            target_item_ids=["mem_rev_cas"],
            old_texts=["before rethink"],
            new_text="after rethink",
            reason="unit",
            validation_status="ok",
        )
        self.core.apply_rethink_action(action_pk, shadow=False)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET text='later legitimate update', updated_at=? WHERE item_id='mem_rev_cas'",
                (_ts_iso(),),
            )
            conn.commit()
        finally:
            conn.close()

        result = self.core.revert_rethink_run(run_id, reason="unit_conflict")
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "revert_conflict")
        conn = sqlite3.connect(self.db_path)
        try:
            text = conn.execute(
                "SELECT text FROM memory_items WHERE item_id='mem_rev_cas'"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(text, "later legitimate update")


class ParseRethinkResponseTests(unittest.TestCase):
    def test_valid_json(self):
        raw = json.dumps({
            "summary": "今晚整理了 1 条",
            "actions": [
                {"action_type": "rewrite", "target_item_ids": ["mem_1"], "new_text": "新", "reason": "更准"},
            ],
        }, ensure_ascii=False)
        actions, summary, err = parse_rethink_response(raw)
        self.assertEqual(err, "")
        self.assertEqual(summary, "今晚整理了 1 条")
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action_type, "rewrite")
        self.assertEqual(actions[0].target_item_ids, ["mem_1"])

    def test_markdown_fence_stripped(self):
        raw = "下面是输出：\n```json\n" + json.dumps({"summary": "ok", "actions": []}, ensure_ascii=False) + "\n```\n以上"
        actions, summary, err = parse_rethink_response(raw)
        self.assertEqual(err, "")
        self.assertEqual(summary, "ok")
        self.assertEqual(actions, [])

    def test_invalid_json_returns_error(self):
        raw = "not json at all"
        actions, summary, err = parse_rethink_response(raw)
        self.assertNotEqual(err, "")
        self.assertEqual(actions, [])

    def test_unknown_action_type_keeps_error(self):
        raw = json.dumps({
            "summary": "ok",
            "actions": [{"action_type": "frobnicate", "target_item_ids": ["x"]}],
        })
        actions, summary, err = parse_rethink_response(raw)
        self.assertEqual(err, "")  # top-level OK
        self.assertEqual(len(actions), 1)
        self.assertTrue(actions[0].parse_error.startswith("unknown_action_type"))

    def test_batch_records_explicit_finish(self):
        batch = parse_rethink_batch(json.dumps({"summary": "done", "finish_edits": True, "actions": []}))
        self.assertTrue(batch.finish_edits)
        self.assertEqual(batch.parse_error, "")


class ValidateActionTests(unittest.TestCase):
    def setUp(self):
        self.cids = {"mem_a", "mem_b", "mem_c"}
        self.eids = {"ev_a"}

    def test_rewrite_ok(self):
        a = ParsedAction(
            action_type="rewrite", target_item_ids=["mem_a"], new_text="新文本", reason="r",
            resolution_type="explicit_correction", evidence_item_ids=["ev_a"],
        )
        status, _ = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "ok")

    def test_target_outside_pool_rejected(self):
        a = ParsedAction(
            action_type="rewrite", target_item_ids=["mem_outside"], new_text="新文本", reason="r",
            resolution_type="explicit_correction", evidence_item_ids=["ev_a"],
        )
        status, reason = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "scope_invalid")
        self.assertIn("outside_candidate_pool", reason)

    def test_merge_needs_2_targets(self):
        a = ParsedAction(action_type="merge", target_item_ids=["mem_a"], new_text="新文本", reason="r", resolution_type="compatible_merge")
        status, _ = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "format_invalid")

    def test_archive_requires_reason(self):
        a = ParsedAction(action_type="archive", target_item_ids=["mem_a"], reason="", resolution_type="obsolete_duplicate")
        status, _ = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "format_invalid")

    def test_rewrite_requires_new_text(self):
        a = ParsedAction(action_type="rewrite", target_item_ids=["mem_a"], new_text="", reason="r", resolution_type="explicit_correction", evidence_item_ids=["ev_a"])
        status, _ = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "format_invalid")

    def test_new_text_too_short(self):
        # 1-2 字符（"啊"/"ok"）算太短、3 字（"搬家了"）可接受
        a = ParsedAction(action_type="rewrite", target_item_ids=["mem_a"], new_text="ok", reason="r", resolution_type="explicit_correction", evidence_item_ids=["ev_a"])
        status, _ = validate_action(a, candidate_ids=self.cids, evidence_ids=self.eids)
        self.assertEqual(status, "format_invalid")

    def test_no_change_ok(self):
        a = ParsedAction(action_type="no_change", target_item_ids=["mem_a"], reason="保留", resolution_type="no_change")
        status, _ = validate_action(a, candidate_ids=self.cids)
        self.assertEqual(status, "ok")


class ScoreActionOocTests(unittest.TestCase):
    def test_clean_text_low_score(self):
        # 普通陈述句、应该不触发 OOC feature
        score, verdict, feats = score_action_ooc("阿澈目前住在临川。", "张三", threshold=0.45)
        self.assertLessEqual(score, 0.45)
        self.assertIn(verdict, ("ok", "warn"))

    def test_empty_text_returns_ok(self):
        score, verdict, _ = score_action_ooc("", "张三", threshold=0.45)
        self.assertEqual(score, 0.0)
        self.assertEqual(verdict, "ok")


class PromptBuildTests(unittest.TestCase):
    def test_render_canon_summary_empty(self):
        self.assertIn("无 canon", render_canon_summary(None))
        self.assertIn("无 canon", render_canon_summary({}))

    def test_render_canon_summary_with_anchors(self):
        prof = {"anchors": {
            "identity": ["一中高一", "测试组 水彩手"],
            "speech_style": ["语气词诶", "句尾~"],
        }}
        out = render_canon_summary(prof)
        self.assertIn("一中高一", out)
        self.assertIn("identity", out)

    def test_render_candidates_block_with_cluster(self):
        cands = [{
            "item_id": "mem_a", "text": "阿澈喜欢黑咖啡",
            "importance": 0.62, "confidence": 0.65, "access_count": 5,
            "created_at": "2026-04-01T00:00:00Z",
            "cluster_size": 2, "cluster_member_ids": ["mem_a", "mem_b"],
            "cluster_members": [{"item_id": "mem_b", "text": "阿澈不爱拿铁", "access_count": 3}],
        }]
        out = render_candidates_block(cands)
        self.assertIn("mem_a", out)
        self.assertIn("阿澈喜欢黑咖啡", out)
        self.assertIn("cluster_size=2", out)
        self.assertIn("mem_b", out)

    def test_build_prompt_returns_two_messages(self):
        sys_msg, user_msg = build_rethink_prompt(
            character="张三",
            candidates=[],
            canon_profile=None,
            max_actions=8,
        )
        self.assertIsInstance(sys_msg, str)
        self.assertIsInstance(user_msg, str)
        self.assertIn("张三", sys_msg)
        self.assertIn("JSON", sys_msg)


class ProposeMemoryRethinkE2ETests(unittest.TestCase):
    """E2E：注入 LLM caller、跑完整 pipeline。"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "rethink_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_no_candidates_returns_skipped(self):
        # 空候选 — 不开 run
        result = propose_memory_rethink(self.core, "张三", llm_caller=lambda _: "{}")
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates_count"], 0)
        self.assertEqual(result["fallback_reason"], "no_candidates")
        self.assertEqual(result["run_id"], "")

    def test_e2e_shadow_rewrite_flow(self):
        # 准备 2 条 mid-importance + high-access 候选
        _insert_memory_item(self.db_path, item_id="mem_rw_e2e",
                            text="阿澈 3 月去大阪出差", importance=0.60, access_count=8)
        # 注入 LLM caller — 返回一条 rewrite action
        def fake_llm(_combined_prompt: str) -> str:
            return json.dumps({
                "summary": "整理过期出差信息",
                "finish_edits": True,
                "actions": [{
                    "action_type": "rewrite",
                    "target_item_ids": ["mem_rw_e2e"],
                    "resolution_type": "temporal_update",
                    "evidence_item_ids": ["ev_trip_finished"],
                    "new_text": "阿澈过去 3 月去大阪出差过（已结束）",
                    "reason": "时间已过、改成过去时表达",
                }],
            }, ensure_ascii=False)

        result = propose_memory_rethink(
            self.core, "张三",
            shadow=True, llm_caller=fake_llm, k=5,
            evidence=[{
                "item_id": "ev_trip_finished", "text": "阿澈明确说大阪出差已经结束。",
                "layer": "episodic", "source": "turn_append", "confidence": 0.95,
                "created_at": _ts_iso(), "evidence_role": "state_update",
            }],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates_count"], 1)
        self.assertEqual(result["actions_proposed_count"], 1)
        self.assertEqual(result["actions_applied_count"], 1)
        self.assertTrue(result["run_id"].startswith("reth_"))
        # shadow 模式下 memory_items.text 没变
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT text FROM memory_items WHERE item_id='mem_rw_e2e'").fetchone()
            self.assertEqual(row[0], "阿澈 3 月去大阪出差")
        finally:
            conn.close()
        # 但 audit 记录了 action
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT validation_status, applied, apply_status FROM memory_rethink_actions WHERE run_id=?",
                (result["run_id"],),
            ).fetchone()
            self.assertEqual(row["validation_status"], "ok")
            self.assertEqual(int(row["applied"]), 0)
            self.assertEqual(row["apply_status"], "simulated")
        finally:
            conn.close()

    def test_e2e_scope_invalid_rejected(self):
        _insert_memory_item(self.db_path, item_id="mem_real", text="阿澈喜欢咖啡", importance=0.60, access_count=8)
        # LLM 编造一个不在候选池里的 id
        def evil_llm(_p: str) -> str:
            return json.dumps({
                "summary": "test",
                "finish_edits": True,
                "actions": [{
                    "action_type": "rewrite",
                    "target_item_ids": ["mem_HALLUCINATED"],
                    "resolution_type": "explicit_correction",
                    "evidence_item_ids": ["ev_scope"],
                    "new_text": "篡改的内容",
                    "reason": "evil",
                }],
            }, ensure_ascii=False)

        result = propose_memory_rethink(
            self.core, "张三", shadow=True, llm_caller=evil_llm, k=5,
            evidence=[{
                "item_id": "ev_scope", "text": "阿澈纠正了旧说法。", "layer": "episodic",
                "source": "turn_append", "confidence": 0.9, "created_at": _ts_iso(),
                "evidence_role": "correction",
            }],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["actions_applied_count"], 0)
        self.assertEqual(result["actions_rejected_count"], 1)
        # action audit 留下 scope_invalid 记录
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT validation_status, validation_reason FROM memory_rethink_actions WHERE run_id=?",
                (result["run_id"],),
            ).fetchone()
            self.assertEqual(row["validation_status"], "scope_invalid")
            self.assertIn("outside_candidate_pool", row["validation_reason"])
        finally:
            conn.close()

    def test_e2e_live_invalid_action_blocks_valid_action_atomically(self):
        _insert_memory_item(
            self.db_path,
            item_id="mem_live_valid",
            text="阿澈喜欢黑咖啡",
            importance=0.60,
            access_count=8,
        )

        def mixed_llm(_prompt: str) -> str:
            return json.dumps({
                "summary": "one valid and one invalid",
                "finish_edits": True,
                "actions": [
                    {
                        "action_type": "rewrite",
                        "target_item_ids": ["mem_live_valid"],
                        "resolution_type": "explicit_correction",
                        "evidence_item_ids": ["ev_live"],
                        "new_text": "阿澈现在不喝咖啡。",
                        "reason": "用户明确纠正",
                    },
                    {
                        "action_type": "rewrite",
                        "target_item_ids": ["mem_missing"],
                        "resolution_type": "explicit_correction",
                        "evidence_item_ids": ["ev_live"],
                        "new_text": "越界内容",
                        "reason": "invalid target",
                    },
                ],
            }, ensure_ascii=False)

        with patch.dict(os.environ, {"MEM_RETHINK_MIN_SHADOW_DAYS": "0"}, clear=False), patch(
            "tsuzuri.memory_rethink_engine.score_action_ooc",
            return_value=(0.05, "ok", []),
        ), patch.object(
            self.core,
            "build_character_canon_profile",
            return_value={"anchors": {"identity": ["张三是 测试组 成员"]}},
        ):
            result = propose_memory_rethink(
                self.core,
                "张三",
                shadow=False,
                llm_caller=mixed_llm,
                k=5,
                evidence=[{
                    "item_id": "ev_live",
                    "text": "阿澈明确说自己现在不喝咖啡。",
                    "layer": "episodic",
                    "source": "turn_append",
                    "confidence": 0.95,
                    "created_at": _ts_iso(),
                    "evidence_role": "correction",
                }],
            )

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["status"], "failed_atomic")
        self.assertEqual(result["actions_applied_count"], 0)
        self.assertEqual(result["actions_rejected_count"], 2)
        conn = sqlite3.connect(self.db_path)
        try:
            text = conn.execute(
                "SELECT text FROM memory_items WHERE item_id='mem_live_valid'"
            ).fetchone()[0]
            statuses = conn.execute(
                "SELECT validation_status, apply_status FROM memory_rethink_actions WHERE run_id=? ORDER BY action_index",
                (result["run_id"],),
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(text, "阿澈喜欢黑咖啡")
        self.assertEqual(statuses[0], ("ok", "batch_blocked"))
        self.assertEqual(statuses[1][0], "scope_invalid")

    def test_e2e_llm_returns_empty(self):
        _insert_memory_item(self.db_path, item_id="mem_x", text="阿澈喜欢咖啡", importance=0.60, access_count=8)
        # 隔离 env LLM fallback：import testgroup/chat_server 会把真实 LLM_* 写进 os.environ，
        # 不隔离的话空 caller 会回落到真实 API——既翻转断言，又在测试里产生真实计费调用。
        with patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""}, clear=False):
            result = propose_memory_rethink(self.core, "张三", shadow=True, llm_caller=lambda _: "", k=5)
        self.assertFalse(result["ok"])
        self.assertIn("llm", result["fallback_reason"])
        # run 状态 = failed
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT status FROM memory_rethink_runs WHERE run_id=?", (result["run_id"],)).fetchone()
            self.assertEqual(row["status"], "failed")
        finally:
            conn.close()

    def test_e2e_llm_returns_garbage(self):
        _insert_memory_item(self.db_path, item_id="mem_y", text="阿澈喜欢咖啡", importance=0.60, access_count=8)
        result = propose_memory_rethink(self.core, "张三", shadow=True, llm_caller=lambda _: "完全不是 JSON 的输出 啊嘛呢", k=5)
        self.assertFalse(result["ok"])
        self.assertIn("parse_error", result["fallback_reason"])

    def test_e2e_missing_finish_fails_without_actions(self):
        _insert_memory_item(self.db_path, item_id="mem_unfinished", text="阿澈喜欢咖啡", importance=0.60, access_count=8)
        result = propose_memory_rethink(
            self.core,
            "张三",
            shadow=True,
            evidence=[],
            llm_caller=lambda _: json.dumps({"summary": "not finished", "actions": []}),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed_incomplete")


class EnvFlagDefaultsTests(unittest.TestCase):
    def test_default_k(self):
        self.assertEqual(mem_rethink_k(), 15)

    def test_default_max_actions(self):
        self.assertEqual(mem_rethink_max_actions(), 8)


if __name__ == "__main__":
    unittest.main()
