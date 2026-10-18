"""test_memory_api.py — P0 显式记忆 API façade（memory_api.py）测试。

夹具照 test_memory_core.py：TemporaryDirectory + 直实例化 MemoryCoreStore（不走单例）。
种子数据用 store._upsert_item（自带 setdefault + FTS 同步）；embedding_text 预填假值，
避免测试触发本地 BGE-M3 模型加载（_upsert_item 仅在 embedding_text 为空且可用时编码）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_api import MemoryApi

_WORLD_TZ = timezone(timedelta(hours=9))


def _meta_payload(mode_scope: str = "normal") -> str:
    return json.dumps({"memory_meta": {"mode_scope": mode_scope}}, ensure_ascii=False)


class _MemoryApiTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "memory_core.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)
        self.api = MemoryApi(self.core)

    def _seed(self, item_id: str, *, character: str = "张三", layer: str = "semantic",
              text: str = "", status: str = "active", archived: int = 0,
              mode_scope: str = "normal", created_at: str = "") -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        self.core._upsert_item({
            "item_id": item_id,
            "character": character,
            "layer": layer,
            "memory_role": layer,
            "engram_type": "note",
            "text": text or f"{item_id} 的文本",
            "payload_json": _meta_payload(mode_scope),
            "salience": 0.6,
            "confidence": 0.7,
            "importance": 0.6,
            "status": status,
            "archived": archived,
            "created_at": created_at or now_iso,
            "updated_at": now_iso,
            "embedding_text": "0.1,0.2,0.3",  # 假 embedding，跳过模型编码
        })

    def _raw_row(self, item_id: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM memory_items WHERE item_id=?", (item_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _fts_match(self, phrase: str) -> bool:
        """召回路径同款 MATCH 查询（:8137）。

        memory_items_fts 由 memory_items 上的 trigger 按 rowid 同步（2026-10-19），
        这里直接查索引、不再先 'rebuild'：测的就是更新路径本身有没有把新文本同步进去。
        LIKE 不可用：对 external-content 表全表 LIKE 直接读内容列，验证不到索引。
        """
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_items_fts'"
            ).fetchone()
            if not row:
                self.skipTest("SQLite build 无 FTS5")
            hits = conn.execute(
                'SELECT item_id FROM memory_items_fts WHERE text MATCH ?', (f'"{phrase}"',)
            ).fetchall()
            return bool(hits)
        finally:
            conn.close()


class MemoryApiListGetTests(_MemoryApiTestBase):
    def setUp(self):
        super().setUp()
        self._seed("a-sem-1", character="张三", layer="semantic", text="张三记得阿澈喜欢柠檬茶拿铁")
        self._seed("a-epi-1", character="张三", layer="episodic", text="张三和阿澈去了 活动中心")
        self._seed("b-sem-1", character="王五", layer="semantic", text="王五的鼓槌放在活动室")
        self._seed("shared-1", character="", layer="semantic", text="共享事实：活动中心 在北町")
        self._seed("star-1", character="*", layer="semantic", text="全局事实：测试组 有五个人")
        self._seed("testgroup-1", character="测试组", layer="collective", text="小组记忆：首次 live 的日期")
        self._seed("guest-1", character="张三", layer="semantic", text="guest  scope 的记忆", mode_scope="guest")
        self._seed("old-1", character="张三", layer="semantic", text="已归档的旧记忆",
                   status="archived", archived=1)

    def test_list_per_char_isolation(self):
        res = self.api.list_memories("张三")
        ids = {it["item_id"] for it in res["items"]}
        self.assertIn("a-sem-1", ids)
        self.assertIn("a-epi-1", ids)
        self.assertIn("shared-1", ids)  # '' 共享行可见
        self.assertIn("star-1", ids)    # '*' 全局行可见
        self.assertNotIn("b-sem-1", ids)   # 越角色不可见
        self.assertNotIn("testgroup-1", ids)    # collective 默认不混入
        self.assertNotIn("guest-1", ids)   # guest scope 剔除
        self.assertNotIn("old-1", ids)     # archived 默认剔除

    def test_list_include_collective(self):
        res = self.api.list_memories("张三", include_collective=True)
        ids = {it["item_id"] for it in res["items"]}
        self.assertIn("testgroup-1", ids)

    def test_list_layer_filter(self):
        res = self.api.list_memories("张三", layer="episodic")
        ids = {it["item_id"] for it in res["items"]}
        self.assertEqual(ids, {"a-epi-1"})

    def test_list_query_substring(self):
        res = self.api.list_memories("张三", query="柠檬茶")
        ids = {it["item_id"] for it in res["items"]}
        self.assertEqual(ids, {"a-sem-1"})
        # 越角色的文本即使命中也不返回
        res2 = self.api.list_memories("张三", query="鼓槌")
        self.assertEqual(res2["items"], [])

    def test_list_include_archived(self):
        res = self.api.list_memories("张三", include_archived=True)
        ids = {it["item_id"] for it in res["items"]}
        self.assertIn("old-1", ids)
        self.assertIn("a-sem-1", ids)

    def test_get_memory_scope(self):
        self.assertIsNotNone(self.api.get_memory("张三", "a-sem-1"))
        self.assertIsNone(self.api.get_memory("张三", "b-sem-1"))   # 越角色 → 404 语义
        self.assertIsNotNone(self.api.get_memory("王五", "shared-1"))  # 共享行各角色可读
        self.assertIsNone(self.api.get_memory("张三", "guest-1"))   # guest 行不可见
        self.assertIsNone(self.api.get_memory("张三", "testgroup-1"))    # collective 不进 per-char get
        self.assertIsNotNone(self.api.get_memory("测试组", "testgroup-1"))

    def test_get_memory_detail_fields(self):
        item = self.api.get_memory("张三", "a-sem-1")
        self.assertEqual(item["memory_meta"]["mode_scope"], "normal")
        self.assertTrue(item["has_embedding"])
        self.assertIn("payload", item)
        self.assertIn("cue", item)
        self.assertIn("lifecycle", item)


class MemoryApiClusterMemberTests(_MemoryApiTestBase):
    """cluster member 展开回读(Phase 4):get_cluster_members + parent/member_count 链接。"""

    def _seed_cluster(self) -> str:
        for i in range(3):
            self._seed(
                f"m{i}", layer="episodic", text=f"片段{i}:阿澈和张三在江之岛看海",
                created_at=f"2026-08-10T0{i}:00:00+00:00",
            )
        from tsuzuri.cluster_memory import write_cluster_memory

        draft = {
            "cluster_kind": "event",
            "item_type": "event_cluster",
            "title": "江之岛一日游",
            "text": "阿澈和张三在江之岛看海，沿着海岸走了很久，傍晚一起吃了海鲜盖饭。" * 2,
            "capsule": "阿澈和张三的江之岛一日游",
            "timeline": [],
            "people": ["阿澈"],
            "places": ["江之岛"],
            "start_at": "2026-08-10",
            "end_at": "2026-08-10",
            "topic_labels": ["travel"],
            "significance": 7,
            "confidence": 0.9,
            "salience": 0.8,
            "importance": 0.7,
            "member_ids": ["m0", "m1", "m2"],
            "evidence_ids": [],
            "canonical_key": "clu_test_members_1",
        }
        return write_cluster_memory(self.core, "张三", draft)

    def test_expand_returns_members_sorted(self):
        cluster_id = self._seed_cluster()
        result = self.api.get_cluster_members("张三", cluster_id)
        self.assertTrue(result["ok"])
        self.assertEqual(result["member_count"], 3)
        self.assertEqual(result["members_returned"], 3)
        self.assertEqual([m["item_id"] for m in result["members"]], ["m0", "m1", "m2"])
        for m in result["members"]:
            self.assertEqual(m["parent_id"], cluster_id)
        self.assertEqual(result["cluster"]["payload"]["title"], "江之岛一日游")
        self.assertEqual(result["cluster"]["member_ids"], ["m0", "m1", "m2"])

    def test_member_limit(self):
        cluster_id = self._seed_cluster()
        result = self.api.get_cluster_members("张三", cluster_id, limit=2)
        self.assertEqual(result["members_returned"], 2)
        self.assertEqual(result["member_count"], 3)

    def test_non_cluster_rejected(self):
        self._seed("plain-1", text="普通记忆")
        result = self.api.get_cluster_members("张三", "plain-1")
        self.assertIsNotNone(result)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "not_a_cluster")

    pass  # upstream-only test removed

    def test_wrong_char_none(self):
        cluster_id = self._seed_cluster()
        self.assertIsNone(self.api.get_cluster_members("李四", cluster_id))

    def test_expand_skips_archived_member(self):
        cluster_id = self._seed_cluster()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET archived=1, status='archived' WHERE item_id='m1'"
            )
            conn.commit()
        finally:
            conn.close()
        result = self.api.get_cluster_members("张三", cluster_id)
        self.assertEqual(result["member_count"], 3)
        self.assertEqual(result["members_returned"], 2)
        self.assertNotIn("m1", [m["item_id"] for m in result["members"]])

    def test_expand_skips_other_character_member(self):
        from tsuzuri.cluster_memory import write_cluster_memory

        self._seed("m0", layer="episodic", text="片段0:阿澈和张三在江之岛看海")
        self._seed("m1", character="李四", layer="episodic", text="片段1:李四单独看到的风景")
        draft = {
            "cluster_kind": "event",
            "item_type": "event_cluster",
            "title": "江之岛一日游",
            "text": "阿澈和张三在江之岛看海，沿着海岸走了很久。",
            "capsule": "阿澈和张三的江之岛一日游",
            "timeline": [],
            "people": ["阿澈"],
            "places": ["江之岛"],
            "start_at": "2026-08-10",
            "end_at": "2026-08-10",
            "topic_labels": ["travel"],
            "significance": 7,
            "confidence": 0.9,
            "salience": 0.8,
            "importance": 0.7,
            "member_ids": ["m0", "m1"],
            "evidence_ids": [],
            "canonical_key": "clu_test_members_scope",
        }
        cluster_id = write_cluster_memory(self.core, "张三", draft)
        result = self.api.get_cluster_members("张三", cluster_id)
        self.assertEqual(result["member_count"], 2)
        self.assertEqual(result["members_returned"], 1)
        self.assertEqual([m["item_id"] for m in result["members"]], ["m0"])

    def test_normalize_row_linkage_fields(self):
        cluster_id = self._seed_cluster()
        listed = self.api.list_memories("张三", item_type="event_cluster")
        self.assertEqual(listed["items"][0]["member_count"], 3)
        detail = self.api.get_memory("张三", cluster_id)
        self.assertEqual(detail["member_ids"], ["m0", "m1", "m2"])
        self.assertEqual(detail["cluster_meta"]["summarizer_version"], "cluster_memory_v2_typed")
        member = self.api.get_memory("张三", "m0")
        self.assertEqual(member["parent_id"], cluster_id)
        self.assertEqual(member["cluster_memory_parent_id"], cluster_id)

    def test_members_surface_traceability_fields(self):
        """cluster Phase4-B2: member 展开透出 location_anchor(cue)/turn_fact_turn_id(payload);
        旧行无信号时为空串而非缺键。"""
        now_iso = datetime.now(timezone.utc).isoformat()
        for i in range(2):
            self.core._upsert_item({
                "item_id": f"tm{i}",
                "character": "张三",
                "layer": "episodic",
                "memory_role": "episodic",
                "engram_type": "event",
                "item_type": "event",
                "text": f"片段{i}:阿澈和张三在江之岛看海",
                "payload_json": json.dumps(
                    {"memory_meta": {"mode_scope": "normal"}, "turn_fact_turn_id": f"t-00{i}"},
                    ensure_ascii=False,
                ),
                "cue_json": json.dumps(
                    {"location_anchor": "江之岛", "entities": ["江之岛"]}, ensure_ascii=False
                ),
                "salience": 0.6,
                "confidence": 0.7,
                "importance": 0.6,
                "status": "active",
                "archived": 0,
                "created_at": f"2026-08-10T0{i}:00:00+00:00",
                "updated_at": now_iso,
                "embedding_text": "0.1,0.2,0.3",  # 假 embedding,跳过模型编码
            })
        from tsuzuri.cluster_memory import write_cluster_memory

        draft = {
            "cluster_kind": "event",
            "item_type": "event_cluster",
            "title": "江之岛看海",
            "text": "阿澈和张三在江之岛看海,傍晚一起吃了海鲜盖饭。" * 2,
            "capsule": "江之岛看海",
            "timeline": [],
            "people": ["阿澈"],
            "places": ["江之岛"],
            "start_at": "2026-08-10",
            "end_at": "2026-08-10",
            "topic_labels": ["travel"],
            "significance": 7,
            "confidence": 0.9,
            "salience": 0.8,
            "importance": 0.7,
            "member_ids": ["tm0", "tm1"],
            "evidence_ids": [],
            "canonical_key": "clu_test_traceability_1",
        }
        cluster_id = write_cluster_memory(self.core, "张三", draft)
        result = self.api.get_cluster_members("张三", cluster_id)
        self.assertTrue(result["ok"])
        by_id = {m["item_id"]: m for m in result["members"]}
        self.assertEqual(by_id["tm0"]["turn_fact_turn_id"], "t-000")
        self.assertEqual(by_id["tm0"]["location_anchor"], "江之岛")
        self.assertEqual(by_id["tm1"]["turn_fact_turn_id"], "t-001")

        # 旧种子行(无 payload turn_id / cue location):键在、值为空串
        legacy_cluster = self._seed_cluster()
        legacy = self.api.get_cluster_members("张三", legacy_cluster)
        for m in legacy["members"]:
            self.assertIn("turn_fact_turn_id", m)
            self.assertIn("location_anchor", m)
            self.assertEqual(m["turn_fact_turn_id"], "")
            self.assertEqual(m["location_anchor"], "")


class MemoryApiUpdateTests(_MemoryApiTestBase):
    def setUp(self):
        super().setUp()
        self._seed("u-1", character="张三", layer="semantic", text="阿澈喜欢红茶")
        self._seed("u-arch", character="张三", layer="semantic", text="旧归档",
                   status="archived", archived=1)
        self._seed("u-guest", character="张三", layer="semantic", text="guest 行", mode_scope="guest")
        self._seed("u-other", character="王五", layer="semantic", text="王五的事实")

    def test_update_happy_path(self):
        res = self.api.update_memory("张三", "u-1", "阿澈喜欢焙茶", reason="纠错")
        self.assertTrue(res["ok"])
        row = self._raw_row("u-1")
        self.assertEqual(row["text"], "阿澈喜欢焙茶")
        self.assertEqual(row["embedding_text"], "")        # 清 embedding 等 backfill
        self.assertEqual(row["embedding_sparse_text"], "")
        self.assertEqual(float(row["confidence"]), 0.7)    # dev 纠错不静默改评分
        self.assertTrue(self._fts_match("阿澈喜欢焙茶"))    # FTS 索引同步：新文本可 MATCH 到
        self.assertFalse(self._fts_match("阿澈喜欢红茶"))   # 旧文本索引已清
        # 审计行
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            audit = conn.execute(
                "SELECT * FROM memory_api_action_log WHERE item_id=? AND action='update'", ("u-1",)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(audit)
        before = json.loads(audit["before_json"])
        self.assertEqual(before["row"]["text"], "阿澈喜欢红茶")

    def test_update_rejects_stale_updated_at_without_overwrite(self):
        original = self.api.get_memory("张三", "u-1")
        token = original["updated_at"]
        first = self.api.update_memory(
            "张三",
            "u-1",
            "第一次纠错",
            expected_updated_at=token,
        )
        self.assertTrue(first["ok"])
        stale = self.api.update_memory(
            "张三",
            "u-1",
            "不应覆盖",
            expected_updated_at=token,
        )
        self.assertFalse(stale["ok"])
        self.assertEqual("stale_conflict", stale["error"])
        self.assertEqual(first["updated_at"], stale["actual_updated_at"])
        self.assertEqual("第一次纠错", self._raw_row("u-1")["text"])

    def test_update_rejects(self):
        self.assertEqual(self.api.update_memory("张三", "u-1", "  ")["error"], "text_required")
        self.assertEqual(self.api.update_memory("张三", "u-1", "x" * 2001)["error"], "text_too_long")
        self.assertEqual(self.api.update_memory("张三", "no-such", "t")["error"], "not_found")
        self.assertEqual(self.api.update_memory("张三", "u-other", "t")["error"], "not_found")  # 越角色
        self.assertEqual(self.api.update_memory("张三", "u-guest", "t")["error"], "not_found")  # guest 行
        self.assertEqual(self.api.update_memory("张三", "u-arch", "t")["error"], "not_active")  # 已归档


class MemoryApiDeleteTests(_MemoryApiTestBase):
    def setUp(self):
        super().setUp()
        self._seed("d-1", character="张三", layer="semantic", text="要删的语义记忆")
        old_iso = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        self._seed("d-work-old", character="张三", layer="working", text="超宽限的 working",
                   created_at=old_iso)
        self._seed("d-work-new", character="张三", layer="working", text="宽限期内的 working")
        self._seed("d-other", character="王五", layer="semantic", text="王五的记忆")
        self._seed("d-shared", character="", layer="semantic", text="共享行")
        self._seed("d-arch", character="张三", layer="semantic", text="已归档",
                   status="archived", archived=1)

    def test_delete_gate_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEM_RETHINK_ALLOW_ARCHIVE", None)
            res = self.api.delete_memory("张三", "d-1")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "archive_disabled")
        row = self._raw_row("d-1")
        self.assertEqual(int(row["archived"]), 0)  # 未被动过

    def test_delete_soft_semantics(self):
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            res = self.api.delete_memory("张三", "d-1", reason="测试删除")
        self.assertTrue(res["ok"])
        row = self._raw_row("d-1")
        self.assertIsNotNone(row)                    # 物理行仍在（无硬删除）
        self.assertEqual(int(row["archived"]), 1)
        self.assertEqual(row["status"], "archived")
        # 软删后内容行仍在，但 trigger 已把它移出 FTS 索引；召回侧另由
        # archived=0 AND status='active' 过滤（下方列表断言覆盖该契约）。
        self.assertFalse(self._fts_match("要删的语义记忆"))
        ids = {it["item_id"] for it in self.api.list_memories("张三")["items"]}
        self.assertNotIn("d-1", ids)
        ids_arch = {it["item_id"] for it in self.api.list_memories("张三", include_archived=True)["items"]}
        self.assertIn("d-1", ids_arch)

    def test_delete_within_grace_audit(self):
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            self.api.delete_memory("张三", "d-work-new")
            self.api.delete_memory("张三", "d-work-old")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = {
                r["item_id"]: json.loads(r["meta_json"])
                for r in conn.execute(
                    "SELECT item_id, meta_json FROM memory_api_action_log WHERE action='delete'"
                ).fetchall()
            }
        finally:
            conn.close()
        self.assertTrue(rows["d-work-new"]["within_grace"])    # working <168h
        self.assertFalse(rows["d-work-old"]["within_grace"])   # working 200 天前

    def test_delete_already_archived_and_cross_char(self):
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            self.assertEqual(self.api.delete_memory("张三", "d-arch")["error"], "already_archived")
            self.assertEqual(self.api.delete_memory("张三", "d-other")["error"], "not_found")

    def test_delete_all_requires_confirm(self):
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            res = self.api.delete_all_memories("张三")
        self.assertEqual(res["error"], "confirm_required")

    def test_delete_all_scope(self):
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            res = self.api.delete_all_memories("张三", confirm=True)
        self.assertTrue(res["ok"])
        # 只清 character='张三' 的 active 行（d-1 / d-work-old / d-work-new）
        self.assertEqual(res["archived_count"], 3)
        self.assertEqual(int(self._raw_row("d-shared")["archived"]), 0)  # 共享行不动
        self.assertEqual(int(self._raw_row("d-other")["archived"]), 0)   # 他角色不动
        self.assertEqual(int(self._raw_row("d-1")["archived"]), 1)


class MemoryApiBudgetTests(_MemoryApiTestBase):
    def test_record_and_get_today(self):
        res = self.api.record_budget_event("张三", "turn_time_llm_calls", tokens=520)
        self.assertTrue(res["ok"])
        self.api.record_budget_event("张三", "rethink_actions", amount=3)
        budget = self.api.get_budget("张三")
        self.assertEqual(budget["today"]["turn_time_llm_calls"], 1)
        self.assertEqual(budget["today"]["rethink_actions"], 3)
        self.assertEqual(budget["today"]["approx_tokens"], 520)
        # per-char 隔离
        other = self.api.get_budget("王五")
        self.assertEqual(other["today"]["turn_time_llm_calls"], 0)

    def test_bad_kind_rejected(self):
        res = self.api.record_budget_event("张三", "not_a_kind")
        self.assertFalse(res["ok"])

    def test_per_day_isolation(self):
        yesterday = (datetime.now(_WORLD_TZ) - timedelta(days=1)).strftime("%Y-%m-%d")
        self.api.record_budget_event("张三", "turn_time_llm_calls", day=yesterday)
        budget = self.api.get_budget("张三", days=7)
        self.assertEqual(budget["today"]["turn_time_llm_calls"], 0)
        days = {h["day"] for h in budget["history"]}
        self.assertIn(yesterday, days)


class MemoryApiHistoryTests(_MemoryApiTestBase):
    def test_history_merges_api_and_evolution(self):
        self._seed("h-1", character="张三", layer="semantic", text="历史目标")
        # 手工种一行 evolution_action_log
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO evolution_action_log(run_id, action_index, op, item_id, before_json, after_json, status, created_at) "
                "VALUES ('run-evo-1', 0, 'rewrite', 'h-1', ?, ?, 'applied', '2026-08-01T00:00:00+00:00')",
                (json.dumps({"exists": True, "row": {"text": "演化前文本"}}),
                 json.dumps({"exists": True, "row": {"text": "演化后文本"}})),
            )
            conn.commit()
        finally:
            conn.close()
        with patch.dict(os.environ, {"MEM_RETHINK_ALLOW_ARCHIVE": "1"}):
            self.api.update_memory("张三", "h-1", "API 改写后")
            self.api.delete_memory("张三", "h-1")
        hist = self.api.get_history("张三", "h-1")
        self.assertIsNotNone(hist)
        sources = {e["source"] for e in hist["entries"]}
        self.assertIn("evolution", sources)
        self.assertIn("api", sources)
        api_actions = [e["action"] for e in hist["entries"] if e["source"] == "api"]
        self.assertEqual(set(api_actions), {"update", "delete"})
        evo = next(e for e in hist["entries"] if e["source"] == "evolution")
        self.assertEqual(evo["before_text"], "演化前文本")

    def test_history_cross_char_blocked(self):
        self._seed("h-2", character="王五", layer="semantic", text="王五的历史")
        self.assertIsNone(self.api.get_history("张三", "h-2"))


_CHAT_SERVER = None


pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
