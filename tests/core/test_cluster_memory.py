"""test_cluster_memory.py — 事件/话题聚类记忆(Phase 1-2)单测。

运行: py -X utf8 -m unittest test_cluster_memory -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tsuzuri import cluster_memory as cluster_memory
from tsuzuri.cluster_memory import (
    _extract_json_object,
    _normalize_draft,
    _split_by_time_gap,
    build_cluster_plan,
    create_cluster_memories,
    fold_member_prefixes,
    run_kairos_cluster_memory,
    select_cluster_memory_candidates,
    select_cluster_rethink_candidates,
    write_cluster_memory,
)
from tsuzuri.memory_core import MemoryCoreStore

_UTC = timezone.utc
_T0 = datetime(2026, 8, 10, 12, 0, 0, tzinfo=_UTC)  # WORLD_TZ 2026-08-10 21:00


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _valid_raw(**overrides):
    raw = {
        "coherent": True,
        "cluster_kind": "event",
        "title": "江之岛一日游",
        "narrative": "阿澈和张三在江之岛看海，沿着海岸走了很久，傍晚一起吃了海鲜盖饭。" * 2,
        "timeline": [{"date": "2026-08-10", "detail": "海边散步", "significance": "第一次一起看海"}],
        "people": ["阿澈"],
        "places": ["江之岛"],
        "start_at": "2026-08-10",
        "end_at": "2026-08-10",
        "capsule": "阿澈和张三的江之岛一日游",
        "significance": 7,
        "confidence": 0.8,
    }
    raw.update(overrides)
    return raw


class ClusterMemoryTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _insert_episodic(
        self,
        item_id: str,
        *,
        character: str = "张三",
        text: str = "测试记忆片段",
        created_at: datetime,
        labels: list[str] | None = None,
        salience: float = 0.5,
        layer: str = "episodic",
        source: str = "runtime",
        parent_id: str = "",
        cluster_memory_parent_id: str = "",
    ) -> None:
        self.core._upsert_item(
            {
                "item_id": item_id,
                "character": character,
                "layer": layer,
                "memory_role": layer,
                "item_type": "event",
                "text": text,
                "created_at": _iso(created_at),
                "updated_at": _iso(created_at),
                "salience": salience,
                "importance": 0.6,
                "source": source,
                "topic_labels_json": json.dumps(labels or [], ensure_ascii=False),
                "embedding_text": "fake-embedding",
                "embedding_model_id": "fake",
                "embedding_dim": 3,
            }
        )
        if parent_id:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("UPDATE memory_items SET parent_id=? WHERE item_id=?", (parent_id, item_id))
                conn.commit()
            finally:
                conn.close()

        if cluster_memory_parent_id:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE memory_items SET cluster_memory_parent_id=? WHERE item_id=?",
                    (cluster_memory_parent_id, item_id),
                )
                conn.commit()
            finally:
                conn.close()

    def _member_rows(self, n: int, *, start: datetime = _T0, step_hours: int = 5, label: str = "travel",
                     labels: list[str] | None = None):
        rows = []
        for i in range(n):
            rows.append(
                {
                    "item_id": f"m{i}",
                    "text": f"片段{i}",
                    "created_at": start + timedelta(hours=step_hours * i),
                    "salience": 0.6,
                    "importance": 0.6,
                    "confidence": 0.6,
                    "topic_labels": list(labels) if labels else [label],
                    "evidence_id": f"ev{i}",
                }
            )
        return rows


class GenericSurfaceIsolationTest(ClusterMemoryTestBase):
    @__import__("unittest").skipUnless(__import__("os").getenv("EMBEDDING_ENABLED") == "1", "needs dense embeddings (embeddings extra)")
    def test_cluster_rows_do_not_enter_generic_memory_surface(self):
        self._insert_episodic("ordinary", text="张三和阿澈在江之岛看海", created_at=_T0, labels=["travel"], salience=0.9)
        self.core._upsert_item({
            "item_id": "cluster-1", "character": "张三", "layer": "semantic",
            "memory_role": "semantic", "item_type": "event_cluster",
            "text": "江之岛一日游的经历概括",
            "payload_json": json.dumps({"title": "江之岛一日游", "places": ["江之岛"]}, ensure_ascii=False),
            "created_at": _iso(_T0), "updated_at": _iso(_T0), "source": "cluster_memory",
            "topic_labels_json": json.dumps(["travel"], ensure_ascii=False),
            "salience": 1.0, "confidence": 0.9, "importance": 0.9,
        })
        pack = self.core.build_memory_context_pack("张三", "你还记得江之岛旅行吗", query_bundle={"explicit_recall_intent": True}, max_items=6)
        rows = pack.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        self.assertTrue(rows)
        self.assertNotIn("event_cluster", {str(row.get("item_type", "")) for row in rows})

    def test_cluster_rethink_candidates_target_parent_only(self):
        parent_id = write_cluster_memory(self.core, "张三", {
            "title": "江之岛一日游", "text": "张三和阿澈在江之岛看海。",
            "capsule": "江之岛看海", "timeline": [], "people": ["阿澈"],
            "places": ["江之岛"], "start_at": "2026-08-10", "end_at": "2026-08-10",
            "cluster_kind": "event", "canonical_key": "test-cluster-key",
            "member_ids": [], "item_type": "event_cluster", "topic_labels": ["travel"],
            "significance": 6, "confidence": 0.8, "salience": 0.8, "importance": 0.8,
        })
        candidates = select_cluster_rethink_candidates(self.core, "张三")
        self.assertEqual([row["item_id"] for row in candidates], [parent_id])
        self.assertEqual(candidates[0]["editable_target"], "parent_only")


class CandidateFilterTest(ClusterMemoryTestBase):
    def test_topic_or_salience_gate(self):
        ins = self._insert_episodic
        ins("a", text="有标签低显著", created_at=_T0, labels=["travel"], salience=0.3)
        ins("b", text="无标签高显著", created_at=_T0, labels=[], salience=0.7)
        ins("c", text="无标签低显著", created_at=_T0, labels=[], salience=0.3)
        ins("d", text="太旧", created_at=_T0 - timedelta(days=30), labels=["travel"], salience=0.3)
        ins("e", text="旧聚类成员", created_at=_T0, labels=["travel"], parent_id="legacy-parent")
        ins("h", text="新聚类成员", created_at=_T0, labels=["travel"], cluster_memory_parent_id="narrative-parent")
        ins("f", text="cross digest", created_at=_T0, labels=["travel"], source="cross_char_daily_digest_x")
        ins("g", text="working 层", created_at=_T0, labels=["travel"], layer="working")
        rows = cluster_memory._load_candidates(self.core, "张三", now=_T0 + timedelta(days=1))
        ids = {r["item_id"] for r in rows}
        self.assertEqual(ids, {"a", "b", "e"})

    def test_legacy_narrative_membership_migrates_to_dedicated_edge(self):
        self._insert_episodic("member", created_at=_T0, labels=["travel"])
        self.core._upsert_item(
            {
                "item_id": "narrative-parent",
                "character": "张三",
                "layer": "semantic",
                "memory_role": "semantic",
                "item_type": "event_cluster",
                "text": "旧叙事聚类",
                "created_at": _iso(_T0),
                "updated_at": _iso(_T0),
                "source": "cluster_memory",
            }
        )
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET parent_id=? WHERE item_id='member'",
                ("narrative-parent",),
            )
            conn.commit()
        finally:
            conn.close()

        MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT parent_id, cluster_memory_parent_id FROM memory_items WHERE item_id='member'"
            ).fetchone()
        finally:
            conn.close()
        self.assertEqual(row, ("", "narrative-parent"))

    def test_time_gap_split(self):
        rows = self._member_rows(2, start=_T0, step_hours=5)
        rows += self._member_rows(2, start=_T0 + timedelta(hours=40), step_hours=5)
        buckets = _split_by_time_gap(rows, 30.0)
        self.assertEqual([len(b) for b in buckets], [2, 2])
        # gap 刚好等于阈值不 split
        rows2 = self._member_rows(2, start=_T0, step_hours=30)
        self.assertEqual(len(_split_by_time_gap(rows2, 30.0)), 1)


class BuildPlanTest(ClusterMemoryTestBase):
    def test_min_size_and_run_cap(self):
        # 3 个 label 各 3 条 → max_clusters=2 → 2 passed + 1 run_cap_reached
        for label_idx, label in enumerate(("travel", "outing", "romance")):
            base = _T0 + timedelta(days=label_idx * 2)
            for i in range(3):
                self._insert_episodic(
                    f"{label}{i}", text=f"{label}片段{i}",
                    created_at=base + timedelta(hours=5 * i), labels=[label],
                )
        # 不足 min_cluster_size 的 label 不进 groups
        self._insert_episodic("lonely0", text="单独一条", created_at=_T0, labels=["health"])
        plan = build_cluster_plan(self.core, "张三", max_clusters=2, now=_T0 + timedelta(days=6))
        statuses = [g["status"] for g in plan["groups"]]
        self.assertEqual(statuses.count("passed"), 2)
        self.assertEqual(plan["groups"][2]["reason"], "run_cap_reached")
        self.assertEqual(len(plan["groups"]), 3)

    def test_members_consumed_dedup(self):
        # 同一批行带两个 label → 两个桶出同组,后到组 members_consumed
        for i in range(3):
            self._insert_episodic(
                f"x{i}", text=f"片段{i}", created_at=_T0 + timedelta(hours=5 * i),
                labels=["outing", "travel"],
            )
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=1))
        self.assertEqual(len(plan["groups"]), 2)
        statuses = sorted(g["status"] for g in plan["groups"])
        self.assertEqual(statuses, ["passed", "skipped"])
        skipped = [g for g in plan["groups"] if g["status"] == "skipped"][0]
        self.assertEqual(skipped["reason"], "members_consumed")

    def test_embedding_split_when_large_bucket(self):
        # 6 行同 label 同时间窗、embedding 向量分两簇 → 拆成 2 组
        import numpy as np

        vecs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.0], [0.0, 0.9, 0.1], [0.1, 1.0, 0.0]]
        for i in range(6):
            self._insert_episodic(
                f"v{i}", text=f"向量片段{i}", created_at=_T0 + timedelta(hours=3 * i),
                labels=["travel"],
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE memory_items SET embedding_text=? WHERE item_id=?",
                (json.dumps(vecs[i]), f"v{i}"),
            )
            conn.commit()
            conn.close()
        with patch.object(cluster_memory, "embedding_row_matches_current", lambda r: True), patch.object(
            cluster_memory, "_emb_decode", lambda s: np.array(json.loads(s))
        ):
            plan = build_cluster_plan(
                self.core, "张三", now=_T0 + timedelta(days=1), similarity_threshold=0.5
            )
        self.assertEqual(len(plan["groups"]), 2)
        for g in plan["groups"]:
            self.assertEqual(g["member_count"], 3)

    def test_distinct_text_gate(self):
        # live 数据存在同文重复行:3 行仅 2 种文本 → 不成组
        for i, txt in enumerate(("同一事实", "同一事实", "另一事实")):
            self._insert_episodic(
                f"dup{i}", text=txt, created_at=_T0 + timedelta(hours=5 * i), labels=["travel"],
            )
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=1))
        self.assertEqual(plan["groups"], [])
        # 补一条第三种文本 → 3 种文本成组,重复行仍是 member(provenance 保留)
        self._insert_episodic("dup3", text="第三个事实", created_at=_T0 + timedelta(hours=20), labels=["travel"])
        plan2 = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=1))
        self.assertEqual(len(plan2["groups"]), 1)
        self.assertEqual(plan2["groups"][0]["member_count"], 4)


class NormalizeDraftTest(ClusterMemoryTestBase):
    def test_valid_raw(self):
        members = self._member_rows(3)
        draft = _normalize_draft(_valid_raw(), members, "张三", label="travel")
        self.assertIsNotNone(draft)
        self.assertEqual(draft["item_type"], "event_cluster")
        self.assertEqual(draft["topic_labels"], ["travel"])
        self.assertEqual(draft["member_ids"], ["m0", "m1", "m2"])
        self.assertEqual(draft["evidence_ids"], ["ev0", "ev1", "ev2"])
        self.assertAlmostEqual(draft["salience"], 0.7, places=3)
        self.assertTrue(draft["canonical_key"].startswith("clu_"))

    def test_incoherent_rejected(self):
        members = self._member_rows(3)
        self.assertIsNone(_normalize_draft(_valid_raw(coherent=False), members, "张三", label="travel"))

    def test_short_narrative_rejected(self):
        members = self._member_rows(3)
        self.assertIsNone(_normalize_draft(_valid_raw(narrative="太短"), members, "张三", label="travel"))

    def test_date_fallback_and_swap(self):
        members = self._member_rows(2, start=_T0, step_hours=30)  # WORLD_TZ 跨 2026-08-10/12
        draft = _normalize_draft(
            _valid_raw(start_at="不是日期", end_at="2026-08-01"), members, "张三", label="travel"
        )
        self.assertEqual(draft["start_at"], "2026-08-01")  # swap 后 start<=end
        self.assertEqual(draft["end_at"], "2026-08-10")    # member 首条 WORLD_TZ 日期(fallback)

    def test_invalid_kind_is_not_guessed_from_span(self):
        members = self._member_rows(3)
        draft = _normalize_draft(
            _valid_raw(cluster_kind="乱写", start_at="2026-08-01", end_at="2026-08-02"),
            members, "张三", label="travel",
        )
        self.assertIsNone(draft)
        draft2 = _normalize_draft(
            _valid_raw(cluster_kind="乱写", start_at="2026-08-01", end_at="2026-08-20"),
            members, "张三", label="travel",
        )
        self.assertIsNone(draft2)

    def test_extract_json_object(self):
        self.assertEqual(_extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(_extract_json_object('前缀 {"a": 2} 后缀'), {"a": 2})
        self.assertIsNone(_extract_json_object("完全不是 json"))
        self.assertIsNone(_extract_json_object("[1, 2]"))

    def test_oversized_narrative_rejected_without_losing_tail(self):
        members = self._member_rows(3)
        unit = "这是一个完整的长句子，用来凑够字数。"
        long_text = unit * (cluster_memory.NARRATIVE_MAX_CHARS // len(unit) + 2)  # 确定性超上界
        self.assertGreater(len(long_text), cluster_memory.NARRATIVE_MAX_CHARS)
        draft = _normalize_draft(_valid_raw(narrative=long_text), members, "张三", label="travel")
        self.assertIsNone(draft)

    def test_relaxed_narrative_budget_accepts_former_oversize(self):
        members = self._member_rows(3)
        text = "阿澈和张三在江之岛沿海岸散步，聊了小组近期的安排，傍晚一起吃了海鲜盖饭。" * 11
        self.assertGreater(len(text), 300)  # 旧 300 上界会拒的区间
        self.assertLessEqual(len(text), cluster_memory.NARRATIVE_MAX_CHARS)
        draft = _normalize_draft(_valid_raw(narrative=text), members, "张三", label="travel")
        self.assertIsNotNone(draft)
        self.assertEqual(draft["text"], text)

    def test_narrative_prompt_budget_follows_constant(self):
        prompt = cluster_memory.build_narrative_prompt([], "张三", "travel")
        self.assertIn(f"120-{cluster_memory.NARRATIVE_MAX_CHARS} 字", prompt)
        self.assertIn(f"120-{cluster_memory.NARRATIVE_MAX_CHARS}字", prompt)
        self.assertNotIn("300字", prompt)
        self.assertIn(str(cluster_memory.NARRATIVE_MAX_CHARS), cluster_memory._narrative_retry_note())

    def test_narrative_violations_report_specifics(self):
        good = {"coherent": True, "cluster_kind": "topic", "narrative": "正文" * 10}
        self.assertEqual(cluster_memory._narrative_violations(good, "time_window"), [])
        over = {
            "coherent": True,
            "cluster_kind": "topic",
            "narrative": "长" * (cluster_memory.NARRATIVE_MAX_CHARS + 1),
        }
        violations = cluster_memory._narrative_violations(over, "time_window")
        self.assertEqual(len(violations), 1)
        self.assertIn(str(len(over["narrative"])), violations[0])
        self.assertIn(str(cluster_memory.NARRATIVE_MAX_CHARS), violations[0])
        bad_kind = {"coherent": True, "cluster_kind": "story", "narrative": "正文" * 10}
        self.assertIn("cluster_kind", cluster_memory._narrative_violations(bad_kind, "time_window")[0])
        scope_mismatch = {"coherent": True, "cluster_kind": "event", "narrative": "正文" * 10}
        self.assertIn("topic", cluster_memory._narrative_violations(scope_mismatch, "recurring_topic")[0])

    def test_retry_notes_render_dynamic_feedback(self):
        note = cluster_memory._narrative_retry_note(["叙事 487 字超过 400 上限"])
        self.assertIn("487", note)
        self.assertIn(f"120-{cluster_memory.NARRATIVE_MAX_CHARS}", note)
        generic = cluster_memory._narrative_retry_note()
        self.assertNotIn("487", generic)
        self.assertIn(f"{cluster_memory.NARRATIVE_MAX_CHARS} 字内", cluster_memory._narrative_truncation_note())


class CreateFlowTest(ClusterMemoryTestBase):
    def _seed_travel_rows(self):
        for i in range(3):
            self._insert_episodic(
                f"t{i}", text=f"江之岛片段{i}", created_at=_T0 + timedelta(hours=5 * i),
                labels=["travel"], salience=0.7,
            )

    def test_dry_run_no_write(self):
        self._seed_travel_rows()
        result = create_cluster_memories(
            self.core, "张三", dry_run=True,
            llm_narrator=lambda members, char, label: _valid_raw(),
            now=_T0 + timedelta(days=1),
        )
        self.assertEqual(len(result["drafts"]), 1)
        self.assertEqual(result["created"], 0)
        conn = sqlite3.connect(self.db_path)
        n = conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE item_type IN ('event_cluster','topic_cluster')"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(n, 0)

    def test_low_significance_dropped(self):
        self._seed_travel_rows()
        result = create_cluster_memories(
            self.core, "张三", dry_run=True,
            llm_narrator=lambda m, c, l: _valid_raw(significance=2),
            now=_T0 + timedelta(days=1),
        )
        self.assertEqual(result["drafts"][0].get("dropped"), "low_significance")

    def test_narrator_none_marks_group(self):
        self._seed_travel_rows()
        result = create_cluster_memories(
            self.core, "张三", dry_run=True,
            llm_narrator=lambda m, c, l: None,
            now=_T0 + timedelta(days=1),
        )
        self.assertEqual(result["plan"]["groups"][0]["reason"], "narrator_rejected")

    def test_write_path_and_canonical_dedup(self):
        self._seed_travel_rows()
        kwargs = dict(
            dry_run=False,
            llm_narrator=lambda m, c, l: _valid_raw(),
            llm_auditor=lambda narrative, samples, char: {"verdict": "faithful"},
            now=_T0 + timedelta(days=1),
        )
        r1 = create_cluster_memories(self.core, "张三", **kwargs)
        self.assertEqual(r1["created"], 1)
        parent_id = r1["parents_created"][0]["item_id"]
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM memory_items WHERE item_id=?", (parent_id,)).fetchone()
        self.assertEqual(row["item_type"], "event_cluster")
        self.assertEqual(row["layer"], "semantic")
        self.assertEqual(row["source"], "cluster_memory")
        self.assertEqual(json.loads(row["topic_labels_json"]), ["travel"])
        self.assertEqual(len(json.loads(row["member_ids_json"])), 3)
        meta = json.loads(row["cluster_meta_json"])
        self.assertEqual(meta["summarizer_version"], "cluster_memory_v2_typed")
        payload = json.loads(row["payload_json"])
        self.assertEqual(payload["start_at"], "2026-08-10")
        children = conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE cluster_memory_parent_id=?", (parent_id,)
        ).fetchone()[0]
        legacy_children = conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE parent_id=?", (parent_id,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(children, 3)
        self.assertEqual(legacy_children, 0)
        # 同批 member 已标新管线专属边 → 离开新管线候选池,不会重复聚类
        r2 = create_cluster_memories(self.core, "张三", **kwargs)
        self.assertEqual(r2["created"], 0)
        self.assertEqual(len(r2["drafts"]), 0)
        # 同一事件来了新 member(同 label/日期 → 同 canonical_key)→ duplicate 拦截
        for i in range(3, 6):
            self._insert_episodic(
                f"t{i}", text=f"江之岛片段{i}", created_at=_T0 + timedelta(hours=5 * i),
                labels=["travel"], salience=0.7,
            )
        r3 = create_cluster_memories(self.core, "张三", **kwargs)
        self.assertEqual(r3["created"], 0)
        self.assertEqual(r3["quarantined"][0]["reason"], "duplicate_canonical_key")

    def test_audit_fabricated_quarantined(self):
        self._seed_travel_rows()
        result = create_cluster_memories(
            self.core, "张三", dry_run=False,
            llm_narrator=lambda m, c, l: _valid_raw(),
            llm_auditor=lambda narrative, samples, char: {"verdict": "fabricated", "reason": "编造地点"},
            now=_T0 + timedelta(days=1),
        )
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["quarantined"][0]["reason"], "编造地点")

    def test_audit_distorted_halves_confidence(self):
        self._seed_travel_rows()
        result = create_cluster_memories(
            self.core, "张三", dry_run=False,
            llm_narrator=lambda m, c, l: _valid_raw(confidence=0.8),
            llm_auditor=lambda narrative, samples, char: {"verdict": "distorted"},
            now=_T0 + timedelta(days=1),
        )
        self.assertEqual(result["created"], 1)
        conn = sqlite3.connect(self.db_path)
        conf = conn.execute(
            "SELECT confidence FROM memory_items WHERE item_type='event_cluster'"
        ).fetchone()[0]
        conn.close()
        self.assertAlmostEqual(conf, 0.4, places=3)


class KairosRunnerTest(ClusterMemoryTestBase):
    def test_idempotent_second_run(self):
        for i in range(3):
            self._insert_episodic(
                f"t{i}", text=f"片段{i}", created_at=_T0 + timedelta(hours=5 * i),
                labels=["travel"], salience=0.7,
            )
        narrator = lambda m, c, l: _valid_raw()  # noqa: E731
        with patch.object(cluster_memory, "_default_llm_auditor", lambda n, s, c: {"verdict": "faithful"}):
            r1 = run_kairos_cluster_memory(self.core, "张三", llm_narrator=narrator, now=_T0 + timedelta(days=6))
            self.assertEqual(r1["status"], "created")
            self.assertEqual(r1["created"], 1)
            r2 = run_kairos_cluster_memory(self.core, "张三", llm_narrator=narrator, now=_T0 + timedelta(days=6))
            self.assertEqual(r2["status"], "idempotent")
            self.assertEqual(r2["created"], 0)
        meta = self.core.get_meta_state_json(r1["meta_key"], {})
        self.assertTrue(meta.get("terminal"))

    def test_narrator_failure_not_terminal(self):
        # narrator 全部失败 → 不消费幂等键,修复后同日重跑可成功
        for i in range(3):
            self._insert_episodic(
                f"t{i}", text=f"片段{i}", created_at=_T0 + timedelta(hours=5 * i),
                labels=["travel"], salience=0.7,
            )
        r1 = run_kairos_cluster_memory(
            self.core, "张三", llm_narrator=lambda m, c, l: None, now=_T0 + timedelta(days=6)
        )
        self.assertFalse(r1["ok"])
        self.assertEqual(r1["status"], "failed")
        meta = self.core.get_meta_state_json(r1["meta_key"], {})
        self.assertFalse(meta.get("terminal", False))
        narrator = lambda m, c, l: _valid_raw()  # noqa: E731
        with patch.object(cluster_memory, "_default_llm_auditor", lambda n, s, c: {"verdict": "faithful"}):
            r2 = run_kairos_cluster_memory(self.core, "张三", llm_narrator=narrator, now=_T0 + timedelta(days=6))
        self.assertEqual(r2["status"], "created")
        self.assertEqual(r2["created"], 1)


class SelectorTest(ClusterMemoryTestBase):
    def _seed_cluster(self, labels: list[str] | None = None):
        members = self._member_rows(3, label="travel", labels=labels)
        draft = _normalize_draft(_valid_raw(), members, "张三", label="travel")
        return write_cluster_memory(self.core, "张三", draft)

    def test_two_label_overlap_triggers(self):
        self._seed_cluster()
        # query 命中 travel(旅游) + outing? 单行只有 travel → 用 explicit 路径
        hits = select_cluster_memory_candidates(
            self.core, "张三", "还记得我们那次去江之岛旅游吗", explicit_recall=True
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["trigger_reason"], "topic_overlap+explicit")
        self.assertEqual(hits[0]["member_count"], 3)

    def test_two_label_overlap_pure_triggers(self):
        self._seed_cluster(labels=["travel", "outing"])
        # 无显式回忆、无 entity 命中,仅 query 推断出 ≥2 个 label → 纯 topic 路径
        hits = select_cluster_memory_candidates(self.core, "张三", "旅游时散步的路线")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["trigger_reason"], "topic_overlap>=2")

    def test_single_label_no_signal_no_trigger(self):
        self._seed_cluster()
        hits = select_cluster_memory_candidates(self.core, "张三", "旅游保险怎么买")
        self.assertEqual(hits, [])

    def test_entity_hit_triggers(self):
        self._seed_cluster()
        # "旅游" 命中 travel label + "江之岛" 命中 places,无显式回忆 → topic+entity
        hits = select_cluster_memory_candidates(self.core, "张三", "江之岛的旅游安排")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["trigger_reason"], "topic_overlap+entity")

    def test_explicit_plus_entity_triggers(self):
        self._seed_cluster()
        hits = select_cluster_memory_candidates(
            self.core, "张三", "还记得江之岛吗", explicit_recall=True
        )
        self.assertEqual(len(hits), 1)
        # "江之岛" 在 places,query 无 topic 关键词 → explicit+entity
        self.assertIn(hits[0]["trigger_reason"], ("explicit_recall+entity", "topic_overlap+explicit"))

    def test_no_signal_empty(self):
        self._seed_cluster()
        self.assertEqual(select_cluster_memory_candidates(self.core, "张三", "今天天气怎么样"), [])

    def test_explicit_topic_does_not_return_unrelated_same_people_cluster(self):
        self._seed_cluster(labels=["travel"])
        members = self._member_rows(3, label="romance", labels=["romance"])
        raw = _valid_raw(title="告白之夜", capsule="阿澈和张三的告白对话",
                         narrative="阿澈和张三在告白相关的话题上进行了重要对话，确认了彼此的感受。" * 2,
                         places=[], cluster_kind="event")
        second = write_cluster_memory(self.core, "张三", _normalize_draft(raw, members, "张三", label="romance"))
        hits = select_cluster_memory_candidates(self.core, "张三", "还记得阿澈压力很大、后来向你告白的那次吗", explicit_recall=True)
        self.assertEqual([row["item_id"] for row in hits], [second])
        self.assertTrue(hits[0]["query_lexical_hit"])


class PassiveChannelExclusionTest(unittest.TestCase):
    """决策B:被动召回通道排除 cluster 行(只走专属 CLUSTER_MEMORY_FRAME)。"""

    def test_exclude_helper(self):
        from tsuzuri.cluster_memory import exclude_cluster_item_rows

        rows = [
            {"item_type": "event_cluster", "text": "概括"},
            {"item_type": "topic_cluster", "text": "话题"},
            {"item_type": "cluster_summary", "text": "旧短语 parent"},
            {"item_type": "event", "text": "原文"},
            {"text": "无类型"},
            "not-a-dict",
        ]
        out = exclude_cluster_item_rows(rows)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["item_type"], "event")
        self.assertEqual(out[2], "not-a-dict")
        self.assertEqual(exclude_cluster_item_rows(None), [])
        self.assertEqual(exclude_cluster_item_rows([]), [])

    pass  # upstream-only test removed


class CascadeFoldTest(ClusterMemoryTestBase):
    """Phase 4 级联折叠:选中 cluster 的 member 原文行从 FACT/MEMORY 抑制,最近豁免。"""

    def _seed_realistic_cluster(self, *, with_recent: bool = False) -> None:
        texts = [
            "阿澈和张三在江之岛看海，沿着海岸走了很久。",
            "阿澈和张三傍晚在江之岛吃了海鲜盖饭。",
            "阿澈和张三回程前在弁天桥上看了日落。",
        ]
        members = []
        for i, txt in enumerate(texts):
            created = _T0 + timedelta(hours=5 * i)
            self._insert_episodic(f"m{i}", text=txt, created_at=created, labels=["travel"], salience=0.7)
            members.append(
                {
                    "item_id": f"m{i}",
                    "text": txt,
                    "created_at": created,
                    "salience": 0.7,
                    "importance": 0.6,
                    "confidence": 0.6,
                    "topic_labels": ["travel"],
                    "evidence_id": f"ev{i}",
                }
            )
        if with_recent:
            recent_txt = "阿澈和张三约定下周再去一次江之岛。"
            recent_at = datetime.now(_UTC) - timedelta(hours=1)
            self._insert_episodic("m3", text=recent_txt, created_at=recent_at, labels=["travel"], salience=0.7)
            members.append(
                {
                    "item_id": "m3",
                    "text": recent_txt,
                    "created_at": recent_at,
                    "salience": 0.7,
                    "importance": 0.6,
                    "confidence": 0.6,
                    "topic_labels": ["travel"],
                    "evidence_id": "ev3",
                }
            )
        draft = _normalize_draft(_valid_raw(), members, "张三", label="travel")
        write_cluster_memory(self.core, "张三", draft)

    def test_fold_prefixes_and_recency_exemption(self):
        self._seed_realistic_cluster(with_recent=True)
        records = select_cluster_memory_candidates(
            self.core, "张三", "还记得我们那次去江之岛旅游吗", explicit_recall=True
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["member_ids"]), 4)
        prefixes = fold_member_prefixes(self.core, records)
        self.assertEqual(len(prefixes), 3)  # m3 在 48h 内,豁免
        # 归一化(去标点)后整条约 24 字,不足 36 字截断 → 前缀即整条归一化文本
        self.assertIn("阿澈和张三在江之岛看海沿着海岸走了很久", prefixes)

    def test_fact_workspace_folds_member_lines(self):
        self._seed_realistic_cluster(with_recent=True)
        records = select_cluster_memory_candidates(
            self.core, "张三", "还记得我们那次去江之岛旅游吗", explicit_recall=True
        )
        prefixes = fold_member_prefixes(self.core, records)

        from tsuzuri.turn_context_workspace import compile_fact_workspace

        surface = (
            "- 阿澈和张三在江之岛看海，沿着海岸走了很久。\n"
            "- 阿澈和张三约定下周再去一次江之岛。"
        )
        query = "还记得我们那次去江之岛旅游吗"
        control = compile_fact_workspace(query, [], current_character="张三", memory_surface=surface)
        control_text = control["text"] + control["memory_recall_text"]
        self.assertIn("沿着海岸走了很久", control_text)  # 无 fold 时 member 行可选中

        folded = compile_fact_workspace(
            query, [], current_character="张三",
            memory_surface=surface, cluster_fold_prefixes=prefixes,
        )
        self.assertEqual(folded["cluster_folded_count"], 1)
        folded_text = folded["text"] + folded["memory_recall_text"]
        self.assertNotIn("沿着海岸走了很久", folded_text)  # 老 member 被折叠
        self.assertIn("下周再去一次", folded_text)         # 最近 member 豁免保留

    def test_fold_skips_archived_member(self):
        self._seed_realistic_cluster()
        conn = sqlite3.connect(self.core.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET archived=1, status='archived' WHERE item_id='m0'"
            )
            conn.commit()
        finally:
            conn.close()
        records = select_cluster_memory_candidates(
            self.core, "张三", "还记得我们那次去江之岛旅游吗", explicit_recall=True
        )
        prefixes = fold_member_prefixes(self.core, records)
        self.assertNotIn("阿澈和张三在江之岛看海沿着海岸走了很久", prefixes)


class CapsuleEvidenceTest(ClusterMemoryTestBase):
    """Phase 4:向上胶囊证据(core_summary rebuild 输入源)。"""

    # 2026-08-30: 窗口断言改相对今日（固定日期会随真实时间滚出 14 天 lookback）
    @staticmethod
    def _recent_date(days_ago: int = 1) -> str:
        return (datetime.now(_UTC) - timedelta(days=days_ago)).date().isoformat()

    def _seed_with_sig(self, significance: int, *, start_at: str | None = None, end_at: str | None = None) -> None:
        members = self._member_rows(3, label="travel")
        raw = _valid_raw(
            significance=significance,
            start_at=start_at or self._recent_date(1),
            end_at=end_at or self._recent_date(1),
        )
        draft = _normalize_draft(raw, members, "张三", label="travel")
        write_cluster_memory(self.core, "张三", draft)

    def test_capsule_shape_and_significance_gate(self):
        from tsuzuri.cluster_memory import load_cluster_capsule_evidence

        self._seed_with_sig(7)
        self._seed_with_sig(4)  # 低于 min_significance=5,不向上消费
        rows = load_cluster_capsule_evidence(self.core, "张三")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "event_cluster")
        self.assertEqual(rows[0]["status"], "active")
        self.assertIn("江之岛一日游", rows[0]["text"])
        self.assertIn("阿澈和张三的江之岛一日游", rows[0]["text"])
        self.assertIn(f"({self._recent_date(1)})", rows[0]["text"])
        self.assertTrue(rows[0]["item_id"])

    def test_capsule_window_filter(self):
        from tsuzuri.cluster_memory import load_cluster_capsule_evidence

        old = self._recent_date(30)  # 超出 14 天窗口
        self._seed_with_sig(7, start_at=old, end_at=old)
        self.assertEqual(load_cluster_capsule_evidence(self.core, "张三"), [])


class FrameCompileTest(unittest.TestCase):
    def _rec(self, **kw):
        rec = {
            "item_id": "clu_test",
            "item_type": "event_cluster",
            "text": "阿澈和张三在江之岛看海，沿着海岸走了很久，傍晚一起吃了海鲜盖饭。",
            "title": "江之岛一日游",
            "start_at": "2026-08-10",
            "end_at": "2026-08-10",
            "member_count": 5,
            "topic_labels": ["travel"],
        }
        rec.update(kw)
        return rec

    def test_empty_records(self):
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        self.assertEqual(compile_cluster_memory_frame([]), "")
        self.assertEqual(compile_cluster_memory_frame(None), "")
        self.assertEqual(compile_cluster_memory_frame([{"text": ""}]), "")

    def test_render_basic(self):
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        frame = compile_cluster_memory_frame([self._rec()])
        self.assertIn("【相关经历概括】", frame)
        self.assertIn("【2026-08-10 · 事件 · 共5条记忆】江之岛一日游:", frame)
        self.assertIn("江之岛看海", frame)
        # XML 包裹与 fact-workspace / conversation-arc-frame 等兄弟平面一致
        self.assertTrue(frame.startswith('<cluster-memory-frame selective="true" summary="true">'))
        self.assertTrue(frame.endswith("</cluster-memory-frame>"))

    def test_topic_kind_and_date_span(self):
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        frame = compile_cluster_memory_frame(
            [self._rec(item_type="topic_cluster", start_at="2026-08-07", end_at="2026-08-12")]
        )
        self.assertIn("【2026-08-07~2026-08-12 · 话题 · 共5条记忆】", frame)

    def test_budget_sentence_boundary(self):
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        long_text = "这是一个完整的长句子，用来凑够字数。" * 30
        frame = compile_cluster_memory_frame([self._rec(text=long_text)], max_chars=240)
        self.assertLessEqual(len(frame), 240)
        # wrapper 计入预算;内容在句边界收尾,closing tag 在最后
        body = frame.rstrip().removesuffix("</cluster-memory-frame>").rstrip()
        self.assertTrue(body.endswith("。"))

    def test_second_record_skipped_when_no_budget(self):
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        long_text = "这是一个完整的长句子，用来凑够字数。" * 30
        frame = compile_cluster_memory_frame(
            [self._rec(text=long_text), self._rec(item_id="clu_2", title="第二条")],
            max_chars=200,
        )
        self.assertNotIn("第二条", frame)

    def test_tail_exclusion_by_position(self):
        # frame 只作为独立 system 平面出现;assemble 不把它并入户 tail(由顺序测试锁定)
        from tsuzuri.turn_context_workspace import compile_cluster_memory_frame

        frame = compile_cluster_memory_frame([self._rec()])
        self.assertNotIn("【回复目标】", frame)


pass  # upstream-only test removed


class FlagTest(unittest.TestCase):
    def test_flags_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("MEM_CLUSTER_MEMORY_ENABLE", "MEM_CLUSTER_MEMORY_CREATE_ENABLE",
                        "MEM_CLUSTER_MEMORY_RECALL_ENABLE"):
                os.environ.pop(var, None)
            self.assertFalse(cluster_memory.mem_cluster_memory_enabled())
            self.assertFalse(cluster_memory.mem_cluster_memory_create_enabled())
            self.assertFalse(cluster_memory.mem_cluster_memory_recall_enabled())

    def test_flag_on(self):
        with patch.dict(os.environ, {"MEM_CLUSTER_MEMORY_ENABLE": "1"}):
            self.assertTrue(cluster_memory.mem_cluster_memory_enabled())

    def test_master_switch_gates_sub_flags(self):
        # 子开关单独置 1 但无总开关 → create/recall 均不生效
        with patch.dict(os.environ, {
            "MEM_CLUSTER_MEMORY_CREATE_ENABLE": "1",
            "MEM_CLUSTER_MEMORY_RECALL_ENABLE": "1",
        }, clear=False):
            os.environ.pop("MEM_CLUSTER_MEMORY_ENABLE", None)
            self.assertFalse(cluster_memory.mem_cluster_memory_create_enabled())
            self.assertFalse(cluster_memory.mem_cluster_memory_recall_enabled())
        # 总开关 + 子开关 → 只开对应路径
        with patch.dict(os.environ, {
            "MEM_CLUSTER_MEMORY_ENABLE": "1",
            "MEM_CLUSTER_MEMORY_CREATE_ENABLE": "1",
        }, clear=False):
            os.environ.pop("MEM_CLUSTER_MEMORY_RECALL_ENABLE", None)
            self.assertTrue(cluster_memory.mem_cluster_memory_create_enabled())
            self.assertFalse(cluster_memory.mem_cluster_memory_recall_enabled())


if __name__ == "__main__":
    unittest.main()
