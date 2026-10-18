import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tsuzuri.memory_core import (
    MemoryCoreStore,
    _memory_explicit_row_is_answer_bearing,
    _recall_focus_text,
    _select_explicit_surface_rows,
    mem_json_online_export_enabled,
)


# 合成的运维 seeds：结构与部署用 memory_maintenance_seeds.local.json 相同，内容虚构。
_SYNTHETIC_MAINTENANCE_SEEDS = {
    "profile_candidates": [
        {
            "text": "用户在给系统画界面时获得明显的创作满足感。",
            "source_item_ids": ["seed_src_profile_ui"],
            "target_path": "long_term.developer_profile.known_preferences",
        },
        {
            "text": "用户创造这个系统，是想要一个能和角色安心聊天的地方。",
            "source_item_ids": ["seed_src_profile_place"],
            "target_path": "long_term.developer_profile.known_preferences",
        },
    ],
    "key_events": [
        {
            "date": "2026-04-07",
            "event": "系统最开始的来历：用户在一次课程项目里决定独自开发。",
            "importance": 4,
            "source_item_ids": ["seed_src_origin", "shared_info:si-seed-origin"],
        }
    ],
    "consolidated_items": [
        {
            "item_id": "seed_key_event_system_origin_v1", "character": "*", "layer": "semantic",
            "memory_role": "semantic", "engram_type": "key_event", "item_type": "key_event",
            "text": "系统最开始的来历：用户在一次课程项目里决定独自开发。",
            "category": "key_event", "visibility": "normal", "importance": 0.86,
            "source_item_ids": ["seed_src_origin", "shared_info:si-seed-origin"],
        },
        {
            "item_id": "seed_profile_place_v1", "character": "*", "layer": "semantic",
            "memory_role": "semantic", "engram_type": "profile_fact", "item_type": "profile_fact",
            "text": "用户创造这个系统，是想要一个能和角色安心聊天的地方。",
            "category": "profile", "visibility": "normal", "importance": 0.78,
            "source_item_ids": ["seed_src_profile_place"],
        },
        {
            "item_id": "seed_archive_system_bug_v1", "character": "*", "layer": "episodic",
            "memory_role": "episodic", "engram_type": "explicit_recall_archive",
            "item_type": "explicit_recall_archive",
            "text": "2026-04-09 当时系统出了大问题，用户觉得很累；后来说一定会修好。这是过去的系统故障回忆，不代表当前故障。",
            "category": "explicit_recall_archive", "visibility": "explicit_recall_only", "importance": 0.72,
            "source_item_ids": ["seed_src_system_bug"],
        },
        {
            "item_id": "seed_archive_account_v1", "character": "*", "layer": "episodic",
            "memory_role": "episodic", "engram_type": "explicit_recall_archive",
            "item_type": "explicit_recall_archive",
            "text": "2026-04-01 当时用户的旧聊天账号被停用，之后改用新账号联系。这是过去事件。",
            "category": "explicit_recall_archive", "visibility": "explicit_recall_only", "importance": 0.72,
            "source_item_ids": ["seed_src_account"],
        },
    ],
    "exclude_source_ids": {"seed_src_excluded": "one_off_transient"},
    "recall_gates": {
        "system_origin_anchor_terms": ["课程项目", "独自开发", "画界面"],
        "meta_topic_profile_terms": ["画界面", "课程项目"],
        "explicit_archives": [
            {
                "item_id_marker": "seed_archive_system_bug",
                "text_marker": "2026-04-09",
                "anchor": "(bug|故障|崩溃|大问题|出问题|修复|系统出)",
                "reject_system_instruction_topic": True,
            },
            {
                "item_id_marker": "seed_archive_account",
                "text_marker": "2026-04-01",
                "anchor": "(账号|停用|联系)",
            },
        ],
    },
}


class MemoryCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        seeds_path = os.path.join(self.tmpdir.name, "maintenance_seeds.json")
        with open(seeds_path, "w", encoding="utf-8") as fh:
            json.dump(_SYNTHETIC_MAINTENANCE_SEEDS, fh, ensure_ascii=False)
        self._seeds_env = patch.dict(os.environ, {"MEM_MAINTENANCE_SEEDS_PATH": seeds_path}, clear=False)
        self._seeds_env.start()
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self._seeds_env.stop()
        self.tmpdir.cleanup()

    def test_embedding_disabled_overrides_m3_mode(self):
        from tsuzuri import memory_core as memory_core

        with patch.dict(
            os.environ,
            {"EMBEDDING_ENABLED": "0", "EMBEDDING_MODEL_M3": "true"},
            clear=False,
        ), patch("tsuzuri.memory_core._emb_is_m3_mode", return_value=True):
            self.assertFalse(memory_core._embedding_callable())

    def _insert_memory_item(
        self,
        *,
        item_id: str,
        character: str = "张三",
        layer: str = "semantic",
        text: str,
        payload: dict | None = None,
        updated_at: str | None = None,
        importance: float = 0.66,
        confidence: float = 0.67,
        engram_type: str = "fact",
        item_type: str = "fact",
        salience: float = 0.62,
        access_count: int = 0,
        source: str = "unit",
    ) -> None:
        ts = updated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        role = "working" if layer == "working" else ("episodic" if layer == "episodic" else "semantic")
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance,
                 status, source, evidence_id, created_at, updated_at, last_accessed_at,
                 access_count, archived, embedding_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    item_id,
                    character,
                    layer,
                    role,
                    engram_type,
                    item_type,
                    text,
                    json.dumps(payload or {}, ensure_ascii=False),
                    "{}",
                    "{}",
                    salience,
                    confidence,
                    0.34,
                    importance,
                    "active",
                    source,
                    "",
                    ts,
                    ts,
                    "",
                    access_count,
                    0,
                    "",
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_snapshot_roundtrip(self):
        snap = {"schema_version": 2, "long_term": {"key_events": []}, "_brain_semantic": {"张三": []}}
        self.core.save_json_snapshot(snap, source="unit")
        got = self.core.load_json_snapshot()
        self.assertEqual(got.get("schema_version"), 2)
        self.assertIn("long_term", got)
        exported = self.core.export_json_snapshot(self.snapshot_path)
        self.assertTrue(os.path.isfile(self.snapshot_path))
        self.assertEqual(exported.get("schema_version"), 2)

    def test_meta_and_domain_json_wrappers_roundtrip(self):
        self.core.set_meta_state_json("kairos.session.test", {"enabled": True, "turn_count": 3})
        self.core.set_domain_json("kairos.character.张三", {"recent_sessions": ["kairos.session.test"]})
        self.assertEqual(self.core.get_meta_state_json("kairos.session.test", {}).get("turn_count"), 3)
        self.assertEqual(
            self.core.get_domain_json("kairos.character.张三", {}).get("recent_sessions"),
            ["kairos.session.test"],
        )

    def test_recent_ooc_audits_and_evolution_wrappers(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO ooc_audit_runs
                (audit_id, character, run_id, ooc_score, style_deviation_json, evidence_hits_json,
                 high_conflict_claims_json, penalty_json, applied, shadow, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "ooc_test_1",
                    "张三",
                    "run_1",
                    0.24,
                    json.dumps({"too_soft": 1.0}, ensure_ascii=False),
                    json.dumps([{"reply_text": "x"}], ensure_ascii=False),
                    json.dumps(["speech_style"], ensure_ascii=False),
                    json.dumps({"style_deviation_axes": {"too_soft": 0.12}, "recommendation": "tighten_internal_style_control"}, ensure_ascii=False),
                    0,
                    1,
                    "2026-04-02T01:00:00Z",
                    "2026-04-02T01:00:00Z",
                ),
            )
            conn.execute(
                """
                INSERT INTO evolution_runs
                (run_id, character, shadow, status, proposal_json, evaluation_json, applied_actions_json, base_snapshot_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "evo_test_1",
                    "张三",
                    1,
                    "shadow_planned",
                    json.dumps([{"op": "merge"}], ensure_ascii=False),
                    json.dumps({"quality": 0.78, "consistency": 0.66}, ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    "hash123",
                    "2026-04-02T02:00:00Z",
                    "2026-04-02T02:00:00Z",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        audits = self.core.get_recent_ooc_audits("张三", limit=5)
        evolutions = self.core.get_recent_evolution_runs("张三", limit=5)
        self.assertEqual(audits[0]["audit_id"], "ooc_test_1")
        self.assertIn("style_deviation_axes", audits[0])
        self.assertEqual(evolutions[0]["run_id"], "evo_test_1")
        self.assertAlmostEqual(float(evolutions[0]["quality"]), 0.78, places=3)
        self.assertIn("evaluation", evolutions[0])

    def test_json_online_export_enabled_defaults_to_false_but_env_can_enable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MEM_JSON_ONLINE_EXPORT_ENABLE", None)
            self.assertFalse(mem_json_online_export_enabled())
        with patch.dict(os.environ, {"MEM_JSON_ONLINE_EXPORT_ENABLE": "1"}, clear=False):
            self.assertTrue(mem_json_online_export_enabled())

    def test_append_and_retrieve(self):
        self.core.append_turn_event(
            "张三",
            "我最近压力有点大",
            "我在，慢慢说。",
            event_nodes=[{"id": "ev1", "text": "用户表达压力大", "salience": 0.8, "surprise": 0.6}],
            meta={"semantic_facts": ["用户最近压力较大"]},
            memory_snapshot={},
        )
        ret = self.core.retrieve_memory_context("张三", "压力", max_items=5)
        self.assertTrue(ret.get("run_id"))
        rows = ret.get("rows", [])
        self.assertTrue(isinstance(rows, list))
        self.assertTrue(len(rows) >= 1)
        merged = " ".join([str(x.get("text", "")) for x in rows])
        self.assertIn("压力", merged)

    def test_explicit_recall_focus_and_broad_topic_answer_gate(self):
        query = "你还记得我之前说过的看病吗？"
        self.assertEqual(_recall_focus_text(query), "看病")
        self.assertEqual(_recall_focus_text("你还记得我的生日吗？"), "生日")

        row = {
            "item_id": "semantic-health-1",
            "layer": "semantic",
            "memory_role": "semantic",
            "item_type": "key_event",
            "text": "阿澈向王五确认自己出门是为了看病，王五随即询问病情是否严重。",
            "payload_json": "{}",
            "score": 0.82,
            "confidence": 0.95,
        }
        self.assertTrue(_memory_explicit_row_is_answer_bearing(row, query))
        selected = _select_explicit_surface_rows(
            [row],
            limit=1,
            query_text=query,
            require_answer_bearing=True,
        )
        self.assertEqual([item["item_id"] for item in selected], ["semantic-health-1"])

        unrelated = dict(row, item_id="semantic-unrelated", text="阿澈喜欢喝柠檬茶。")
        self.assertFalse(_memory_explicit_row_is_answer_bearing(unrelated, query))

    def test_plan_turn_memory_writes_prefers_turn_fact_card_text(self):
        plan = self.core.plan_turn_memory_writes(
            "李四",
            "这是一段容易诱发复述的用户旧台词",
            "这是一段容易被召回后复述的角色旧回复",
            event_nodes=[],
            meta={
                "turn_fact_card": {
                    "fact": "阿澈和李四确认用事实流替代长历史原文以减少复述。",
                    "state_delta": "方案从保留历史数组推进到事实流上下文。",
                    "near_summary": "阿澈担心复述，李四建议事实流。",
                }
            },
        )

        records = plan.get("records", [])
        working = next(r for r in records if r.get("memory_type") == "working_turn")
        episodic = next(r for r in records if r.get("memory_type") == "episodic_event")

        self.assertIn("事实流替代长历史原文", working["text"])
        self.assertNotIn("角色旧回复", working["text"])
        self.assertIn("事实流替代长历史原文", episodic["text"])
        self.assertNotIn("用户旧台词", episodic["text"])
        self.assertEqual(working["payload"]["user_text"], "这是一段容易诱发复述的用户旧台词")
        self.assertEqual(working["payload"]["assistant_reply"], "这是一段容易被召回后复述的角色旧回复")

    def test_plan_turn_memory_writes_meta_location_fallback(self):
        """cluster Phase4-B2: meta["location"] 兜底进 shared_payload(web 主路径无 situation_model)。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            "今天集训练了好久",
            "辛苦了，下次带上新效果器试试。",
            event_nodes=[],
            meta={"location": "活动中心 活动室"},
        )
        self.assertEqual(plan.get("shared_payload", {}).get("location"), "活动中心 活动室")

    def test_plan_turn_memory_writes_location_empty_without_signal(self):
        """无 location 信号时保持空串(诚实未知,不捏造)。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            "今天集训练了好久",
            "辛苦了。",
            event_nodes=[],
            meta={},
        )
        self.assertEqual(plan.get("shared_payload", {}).get("location"), "")

    def test_plan_turn_memory_writes_situation_model_location_wins_over_meta(self):
        """优先级保持:situation_model.schedule_grounding.expected_location > meta["location"]。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            "今天集训练了好久",
            "辛苦了。",
            event_nodes=[],
            meta={
                "situation_model": {"schedule_grounding": {"expected_location": "学校"}},
                "location": "活动中心",
            },
        )
        self.assertEqual(plan.get("shared_payload", {}).get("location"), "学校")

    def test_episodic_payload_records_turn_fact_turn_id(self):
        """cluster Phase4-B2: episodic payload 写 turn_id↔evidence_id 映射;无 card 时为空串。"""
        plan = self.core.plan_turn_memory_writes(
            "李四",
            "我们今天在河边捡了贝壳",
            "嗯，白色的那块很好看。",
            event_nodes=[],
            meta={"turn_fact_card": {"turn_id": "t-b2-001", "fact": "阿澈和李四在河边捡贝壳。"}},
        )
        episodic = next(r for r in plan.get("records", []) if r.get("memory_type") == "episodic_event")
        self.assertEqual(episodic["payload"].get("turn_fact_turn_id"), "t-b2-001")
        self.assertTrue(episodic.get("evidence_id"))

        plan_no_card = self.core.plan_turn_memory_writes(
            "李四",
            "今天天气不错",
            "嗯，挺好的。",
            event_nodes=[],
            meta={},
        )
        episodic_no_card = next(r for r in plan_no_card.get("records", []) if r.get("memory_type") == "episodic_event")
        self.assertEqual(episodic_no_card["payload"].get("turn_fact_turn_id"), "")

    def test_append_turn_event_persists_location_anchor_and_turn_id(self):
        """端到端:写入后 DB 行 cue.location_anchor / payload.turn_fact_turn_id 落地。"""
        self.core.append_turn_event(
            "张三",
            "我们今天在江之岛看了海",
            "海风吹着很舒服呢。",
            event_nodes=[],
            meta={
                "location": "江之岛",
                "turn_fact_card": {"turn_id": "t-b2-900", "fact": "阿澈和张三在江之岛看海。"},
            },
            memory_snapshot={},
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT payload_json, cue_json FROM memory_items "
                "WHERE character='张三' AND layer='episodic';"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(len(rows), 1)
        cue = json.loads(rows[0]["cue_json"] or "{}")
        payload = json.loads(rows[0]["payload_json"] or "{}")
        self.assertEqual(cue.get("location_anchor"), "江之岛")
        self.assertIn("江之岛", cue.get("entities", []))
        self.assertEqual(payload.get("turn_fact_turn_id"), "t-b2-900")

    def test_auto_fact_observation_oos_guard_blocks_unrelated_query(self):
        """Phase R.W2: episodic/fact_observation 不能靠 recency/importance 在 OOS query 出水面。"""
        self._insert_memory_item(
            item_id="auto_fact_piano",
            character="张三",
            layer="episodic",
            text="阿澈在学钢琴",
            payload={"fact": "阿澈在学钢琴", "meta": {"semantic_facts": ["阿澈在学钢琴"]}},
            engram_type="fact_observation",
            item_type="fact_observation",
            salience=0.62,
            importance=0.66,
            confidence=0.70,
            source="turn_append",
        )
        ret = self.core.retrieve_memory_context("张三", "今天晚上吃什么", max_items=5, log_retrieval=False)
        merged = " ".join(str(x.get("text", "")) for x in ret.get("rows", []))
        self.assertNotIn("钢琴", merged)

    def test_auto_fact_observation_related_query_still_recalls(self):
        """Phase R.W2: 有明确 query evidence 时，待蒸馏 auto fact 仍可被召回。"""
        self._insert_memory_item(
            item_id="auto_fact_piano_related",
            character="张三",
            layer="episodic",
            text="阿澈在学钢琴",
            payload={"fact": "阿澈在学钢琴", "meta": {"semantic_facts": ["阿澈在学钢琴"]}},
            engram_type="fact_observation",
            item_type="fact_observation",
            salience=0.62,
            importance=0.66,
            confidence=0.70,
            source="turn_append",
        )
        ret = self.core.retrieve_memory_context("张三", "钢琴", max_items=5, log_retrieval=False)
        merged = " ".join(str(x.get("text", "")) for x in ret.get("rows", []))
        self.assertIn("钢琴", merged)

    def test_evolution_shadow_and_apply(self):
        self.core.append_turn_event(
            "张三",
            "我喜欢水彩",
            "我也很喜欢聊水彩。",
            event_nodes=[
                {"id": "ev2", "text": "用户喜欢水彩", "salience": 0.75, "surprise": 0.55},
                {"id": "ev3", "text": "用户喜欢水彩", "salience": 0.74, "surprise": 0.54},
            ],
            meta={"semantic_facts": ["用户喜欢水彩"]},
            memory_snapshot={},
        )
        evo = self.core.propose_memory_evolution("张三", max_actions=6)
        self.assertTrue(evo.get("run_id"))
        self.assertIn("actions", evo)
        run_id = str(evo.get("run_id"))
        # force apply in test to verify idempotent behavior
        applied = self.core.apply_evolution_plan(run_id=run_id, shadow_only=False)
        self.assertTrue(applied.get("ok"))

    def test_default_evolution_promotion_thresholds_match_product_policy(self):
        policy = self.core._load_policy_profile().get("evolution_policy", {})
        self.assertEqual(policy.get("promote_salience_min"), 0.72)
        self.assertEqual(policy.get("promote_access_min"), 2)

    # ── Phase Q.W: working → episodic promotion (2026-05-04) ──
    # 让老化 (>=24h) + 有保留信号 (sal/imp/acc 任一) 的 working turn 升级到 episodic、
    # 让 working layer 名副其实只装"刚才说的话"。
    def _fetch_layer_by_id(self, item_id: str) -> str:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT layer FROM memory_items WHERE item_id=?", (item_id,)
            ).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row else ""

    def _fetch_importance_by_id(self, item_id: str) -> float:
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance FROM memory_items WHERE item_id=?", (item_id,)
            ).fetchone()
        finally:
            conn.close()
        return float(row[0]) if row else 0.0

    def _hours_ago_iso(self, hours: float) -> str:
        ts = datetime.now(timezone.utc) - timedelta(hours=float(hours))
        return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def test_evolve_promotes_aged_working_with_importance(self):
        """**2026-05-04 Phase Q.W2/Option E**：working→episodic 不看 raw importance、看 typed_score。
        Layer 1 features payload: high score (0.75) + reuse=1 → typed_gate fast-track promote"""
        item_id = "wkpromote_imp"
        payload_high = {
            "memory_policy": {
                "memory_features": {
                    "user_fact_density": 0.8,
                    "stance_or_value": 0.6,
                    "specificity": 0.4,
                    "emotional_weight": 0.4,
                    "explicit_memory_intent": 1.0,
                    "cold_start_fast_track": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.75,
                "cold_start_fast_track": False,
                "consolidation_state": "candidate",
                "confirmed_reuse_count": 1,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:我从小就特别喜欢看推理小说，比起娱乐更看重的是逻辑 | assistant:嗯",
            updated_at=self._hours_ago_iso(25),
            importance=0.50,
            confidence=0.60,
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload_high,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertGreaterEqual(len(promote_actions), 1, "score=0.75 + reuse=1 应被 typed_gate promote")
        applied = self.core.apply_evolution_plan(plan=promote_actions, shadow_only=False)
        self.assertTrue(applied.get("ok"))
        # target_layer in semantic/episodic, target_engram in user_profile_fact/key_event/recall_archive
        self.assertIn(self._fetch_layer_by_id(item_id), ("episodic", "semantic"))
        self.assertGreaterEqual(self._fetch_importance_by_id(item_id), 0.7)

    def test_evolution_pool_includes_aged_working_outside_recent_window(self):
        old_item_id = "wkpromote_outside_recent_window"
        self._insert_memory_item(
            item_id=old_item_id,
            character="王五",
            layer="working",
            text="user:下周的资料拜托你了 | assistant:我答应会整理好",
            updated_at=self._hours_ago_iso(240),
            engram_type="turn_exchange",
            item_type="turn",
            payload={
                "memory_policy": {
                    "memory_features": {"specificity": 0.1},
                    "typed_score": 0.1,
                    "consolidation_state": "transient",
                },
                "turn_fact_card": {"commitments": ["王五答应会整理好下周的资料"]},
            },
        )
        for idx in range(270):
            self._insert_memory_item(
                item_id=f"recent_semantic_{idx:03d}",
                character="王五",
                layer="semantic",
                text=f"近期语义占位记录编号 {idx:03d}",
                updated_at=self._hours_ago_iso(1),
                importance=0.30,
                salience=0.30,
            )

        recent_ids = {row.get("item_id") for row in self.core._query_items("王五", limit=260)}
        self.assertNotIn(old_item_id, recent_ids)
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        self.assertTrue(any(
            action.get("op") == "promote"
            and old_item_id in (action.get("item_ids", []) or [])
            for action in evo.get("actions", [])
        ))

    def test_aged_working_pool_skips_waiting_candidate_until_reuse(self):
        for item_id, reuse_count in (("waiting_candidate", 0), ("reused_candidate", 1)):
            self._insert_memory_item(
                item_id=item_id,
                character="王五",
                layer="working",
                text=f"user:{item_id} | assistant:嗯",
                updated_at=self._hours_ago_iso(240),
                payload={
                    "memory_policy": {
                        "memory_features": {"specificity": 0.4},
                        "typed_score": 0.72,
                        "consolidation_state": "candidate",
                        "confirmed_reuse_count": reuse_count,
                    },
                },
            )
        for idx in range(25):
            self._insert_memory_item(
                item_id=f"newer_pool_item_{idx:02d}",
                character="王五",
                layer="semantic",
                text=f"较新的候选池记录 {idx:02d}",
                updated_at=self._hours_ago_iso(1),
                importance=0.30,
                salience=0.30,
            )

        item_ids = {
            row.get("item_id")
            for row in self.core._query_evolution_items("王五", recent_limit=20)
        }
        self.assertNotIn("waiting_candidate", item_ids)
        self.assertIn("reused_candidate", item_ids)

    def test_evolve_does_not_promote_aged_working_with_default_features(self):
        """**2026-05-04 Phase Q.W2/Option E**：低 typed_score (chitchat) → archive、不 promote。
        Default 0.50 raw imp/sal 已无意义 — 看 Layer 1 features。"""
        item_id = "wkpromote_default"
        payload_chitchat = {
            "memory_policy": {
                "memory_features": {
                    "user_fact_density": 0.0,
                    "stance_or_value": 0.0,
                    "specificity": 0.0,
                    "emotional_weight": 0.0,
                    "explicit_memory_intent": 0.0,
                    "cold_start_fast_track": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.0,
                "cold_start_fast_track": False,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:我倒是挺喜欢玩二次元手游 | assistant:哦哦，抽卡的那种吗？",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload_chitchat,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertEqual(len(promote_actions), 0, "低 typed_score + 无 reuse → 不应 promote")

    def test_evolve_promotes_aged_working_with_reuse_signal(self):
        """**2026-05-04 Phase Q.W2/Option E**：confirmed_reuse_count >= 1 = Layer 3 真实复用信号
        让 score 0.70-0.85 的 candidate 升级 → promote。"""
        item_id = "wkpromote_reuse"
        payload_with_reuse = {
            "memory_policy": {
                "memory_features": {
                    "user_fact_density": 0.7,
                    "stance_or_value": 0.5,
                    "specificity": 0.3,
                    "emotional_weight": 0.2,
                    "explicit_memory_intent": 1.0,
                    "cold_start_fast_track": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.72,
                "cold_start_fast_track": False,
                "consolidation_state": "candidate",
                "confirmed_reuse_count": 2,  # Layer 3 信号、被实际复用 2 次
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:我喜欢早起、虽然其他人都睡懒觉 | assistant:嗯",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload_with_reuse,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertGreaterEqual(len(promote_actions), 1, "score=0.72 + reuse=2 应被 typed_gate promote")

    def test_evolve_does_not_promote_fresh_working(self):
        """working turn aged 10h (< 24h) → 不 propose promote 即使 importance=0.6"""
        item_id = "wkpromote_fresh"
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:刚才说的事我再想想 | assistant:好",
            updated_at=self._hours_ago_iso(10),
            importance=0.60,
            confidence=0.60,
            engram_type="turn_exchange",
            item_type="turn",
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and a.get("target_layer") == "episodic" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertEqual(len(promote_actions), 0, "age<24h 不应 propose promote")
        self.assertEqual(self._fetch_layer_by_id(item_id), "working")

    def test_evolve_does_not_promote_low_signal_working(self):
        """working turn aged 30h 但所有 retention signal 都低 → 不 propose promote"""
        item_id = "wkpromote_low"
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:嗯，那就这样吧 | assistant:好的",
            updated_at=self._hours_ago_iso(30),
            importance=0.30,    # < 0.50
            confidence=0.30,
            salience=0.30,      # < 0.50
            access_count=0,     # < 1
            engram_type="turn_exchange",
            item_type="turn",
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and a.get("target_layer") == "episodic" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertEqual(len(promote_actions), 0, "all signals < threshold 不应 propose promote")
        self.assertEqual(self._fetch_layer_by_id(item_id), "working")

    def test_evolve_does_not_promote_ephemeral_noise(self):
        """working turn aged 30h + 文本是 ephemeral_noise (晚安) → 即使 imp=0.6 也不 promote"""
        item_id = "wkpromote_ephem"
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:晚安 | assistant:嗯、晚安",  # 含 "晚安" 命中 ephemeral noise regex
            updated_at=self._hours_ago_iso(30),
            importance=0.60,
            confidence=0.60,
            engram_type="turn_exchange",
            item_type="turn",
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and a.get("target_layer") == "episodic" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertEqual(len(promote_actions), 0, "ephemeral noise 文本不应 propose promote")
        self.assertEqual(self._fetch_layer_by_id(item_id), "working")

    def test_evolve_episodic_does_not_get_promoted_to_episodic_again(self):
        """已是 episodic layer 的 row 不会被 working→episodic 分支处理（layer != working）"""
        item_id = "wkpromote_already_epi"
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="episodic",  # 已是 episodic
            text="user:某个事件 | assistant:某个回应",
            updated_at=self._hours_ago_iso(30),
            importance=0.60,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        # 不会有 working→episodic 提议（target_layer="episodic"）针对此 id
        episodic_target_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and a.get("target_layer") == "episodic" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertEqual(len(episodic_target_actions), 0, "已是 episodic 不应被 working→episodic 分支二次处理")

    # ── Phase Q.X: cross-character daily digest broadcast (2026-05-04) ──
    # 5 角色之间共享 daily summary、用「听 X 说」前缀显式标记 secondhand 视角。
    _CROSS_CHAR_TEST_DATE = "2026-05-01"  # WORLD_TZ date、固定不依赖系统时钟

    def _utc_iso_for_world_date_noon(self, world_date_str: str) -> str:
        """返回某 WORLD_TZ 日期 12:00 (中午) 对应的 UTC ISO 串、避开 WORLD_TZ 日界。

        WORLD_TZ 5/1 12:00 = UTC 5/1 03:00。query 窗口 [UTC 4/30 15:00, UTC 5/1 15:00) 包含 ✓。
        """
        d = datetime.strptime(world_date_str, "%Y-%m-%d").replace(hour=12, minute=0, second=0)
        utc = d - timedelta(hours=9)
        return utc.replace(microsecond=0).isoformat() + "Z"

    def _insert_working_turn_at_world_date(self, speaker: str, user_text: str, reply: str, world_date_str: str, idx: int = 0) -> str:
        """插一条 speaker working turn、created_at 固定在某 WORLD_TZ date 的 12:00 + idx 分钟、避免跨日 flaky。"""
        d = datetime.strptime(world_date_str, "%Y-%m-%d").replace(hour=12, minute=int(idx % 60))
        utc = d - timedelta(hours=9)
        ts = utc.replace(microsecond=0).isoformat() + "Z"
        item_id = f"crossdigest_turn_{speaker}_{world_date_str}_{idx}_{user_text[:6]}"
        text = f"user:{user_text[:140]} | assistant:{reply[:180]}"[:320]
        payload = {"user_text": user_text, "assistant_reply": reply}
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance,
                 status, source, evidence_id, created_at, updated_at, last_accessed_at,
                 access_count, archived, embedding_text)
                VALUES (?, ?, 'working', 'working', 'turn_exchange', 'turn', ?, ?, '{}', '{}', 0.5, 0.7, 0.45, 0.5,
                        'active', 'turn_append', '', ?, ?, '', 0, 0, '');
                """,
                (item_id, speaker, text, json.dumps(payload, ensure_ascii=False), ts, ts),
            )
            conn.commit()
        finally:
            conn.close()
        return item_id

    def test_cross_char_digest_writes_to_4_receivers(self):
        """speaker=王五 跑 propose、其他 4 角色各写 1 条 episodic、source_kind=cross_char_daily_digest"""
        date_s = self._CROSS_CHAR_TEST_DATE
        self._insert_working_turn_at_world_date("王五", "我在备考最近压力大", "加油啊", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做几页练习", "辛苦了", date_s, idx=10)

        # 注入 mock LLM caller、避免真调 API
        def _mock_llm(prompt: str) -> str:
            return "阿澈在备考、压力比较大、要做练习"

        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=_mock_llm,
        )
        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("written_count"), 4, f"result={result}")
        self.assertEqual(set(result.get("receivers", [])), {"张三", "赵六", "李四", "钱七"})

        # db: 4 个 receiver 各 1 条 episodic、text 含「听 王五 说：」、source_kind 正确
        conn = sqlite3.connect(self.db_path)
        try:
            for receiver in ("张三", "赵六", "李四", "钱七"):
                rows = conn.execute(
                    "SELECT text, layer, importance, payload_json FROM memory_items "
                    "WHERE character=? AND source='cross_char_daily_digest' AND archived=0",
                    (receiver,),
                ).fetchall()
                self.assertEqual(len(rows), 1, f"receiver={receiver} 应有 1 条 cross_char_digest")
                text, layer, importance, payload_json = rows[0]
                self.assertEqual(layer, "episodic")
                self.assertGreaterEqual(importance, 0.65)
                self.assertTrue(text.startswith("听 王五 说："), f"text={text}")
                payload = json.loads(payload_json)
                self.assertEqual(payload.get("source_kind"), "cross_char_daily_digest")
                self.assertEqual(payload.get("told_by"), "王五")
                self.assertTrue(payload.get("is_secondhand"))
        finally:
            conn.close()

    def test_cross_char_digest_skip_when_no_turns(self):
        """speaker 当天 0 turn → return skipped: no_content"""
        date_s = self._CROSS_CHAR_TEST_DATE
        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "should not fire",
        )
        self.assertEqual(result.get("skipped"), "no_content")

    def test_cross_char_digest_skip_when_summary_empty(self):
        """LLM caller 返回空字符串 → return skipped: empty_summary"""
        date_s = self._CROSS_CHAR_TEST_DATE
        self._insert_working_turn_at_world_date("王五", "聊一下水彩配置的事", "哦", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "我去超市买了点东西", "好", date_s, idx=10)
        with (
            patch("tsuzuri.secrets_loader.load_project_secrets", return_value={}),
            patch.dict(
                os.environ,
                {"LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""},
                clear=False,
            ),
        ):
            result = self.core.propose_daily_cross_char_digest(
                "王五", date_s, llm_caller=lambda p: "",
            )
        self.assertEqual(result.get("skipped"), "empty_summary")

    def test_cross_char_digest_idempotent_on_same_day(self):
        """同 speaker 同 date 同 summary 跑 2 次、db 总数仍是 4（item_id 相同 ON CONFLICT 覆盖）"""
        date_s = self._CROSS_CHAR_TEST_DATE
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做几页练习", "辛苦", date_s, idx=10)

        def _llm(p): return "阿澈在备考、压力大"

        r1 = self.core.propose_daily_cross_char_digest("王五", date_s, llm_caller=_llm)
        r2 = self.core.propose_daily_cross_char_digest("王五", date_s, llm_caller=_llm)
        self.assertEqual(r1.get("written_count"), 4, f"r1={r1}")
        self.assertEqual(r2.get("written_count"), 4, f"r2={r2}")
        # 但 db 总数仍是 4（item_id 相同被 ON CONFLICT 覆盖）
        conn = sqlite3.connect(self.db_path)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM memory_items WHERE source='cross_char_daily_digest' AND archived=0"
            ).fetchone()[0]
            self.assertEqual(n, 4, "同 (speaker, date, summary) 重复跑、db 仍只有 4 条")
        finally:
            conn.close()

    def test_query_mentions_character_helper(self):
        """alias helper：query 含别名应命中、不含应不命中"""
        from tsuzuri.memory_core import _query_mentions_character
        self.assertTrue(_query_mentions_character("wuwu 之前提过", "王五"))
        self.assertTrue(_query_mentions_character("王五告诉我", "王五"))
        self.assertTrue(_query_mentions_character("liuliu 的茶很好喝", "赵六"))
        self.assertTrue(_query_mentions_character("zhangsan 的水彩", "张三"))
        self.assertFalse(_query_mentions_character("我想喝茶", "赵六"))
        self.assertFalse(_query_mentions_character("今天天气不错", "张三"))

    def test_cross_char_digest_reinforces_similar_existing_memory(self):
        """张三已有 episodic「阿澈之前提过他在准备考试」、王五 broadcast 后该 row 应被强化"""
        date_s = self._CROSS_CHAR_TEST_DATE
        # 预先给张三插一条 episodic 关于"考试"
        existing_iid = "existing_zhangsan_about_exam"
        self._insert_memory_item(
            item_id=existing_iid,
            character="张三",
            layer="episodic",
            text="阿澈之前提过他在准备考试、还说图书馆很忙",
            importance=0.55,
            confidence=0.70,
            engram_type="event",
            item_type="event",
        )
        # 插王五当天 working
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做练习", "辛苦", date_s, idx=10)

        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "阿澈在备考、压力大",
        )
        self.assertEqual(result.get("written_count"), 4)
        # reinforced 列表应该至少有 1 条（张三的 existing 备考 row）
        reinforced_for_zhangsan = [r for r in result.get("reinforced", []) if r.get("receiver") == "张三"]
        self.assertGreaterEqual(
            len(reinforced_for_zhangsan), 1,
            f"张三 existing 考试 row 应被强化、got: {result.get('reinforced')}",
        )
        # importance 应从 0.55 升到 0.60
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance, access_count, payload_json FROM memory_items WHERE item_id=?",
                (existing_iid,),
            ).fetchone()
            self.assertAlmostEqual(float(row[0]), 0.60, places=2,
                                   msg=f"existing imp 应 0.55+0.05=0.60、got {row[0]}")
            self.assertEqual(int(row[1]), 0, "digest reinforcement 不能伪造 retrieval access")
            payload = json.loads(row[2])
            history = payload.get("cross_char_reinforced_by", [])
            self.assertGreaterEqual(len(history), 1)
            self.assertEqual(history[0].get("speaker"), "王五")
            self.assertGreaterEqual(int(payload.get("cross_char_reinforcement_count", 0)), 1)
        finally:
            conn.close()

    def test_cross_char_digest_does_not_reinforce_unrelated_memory(self):
        """张三已有 episodic「我喜欢吃甜点」（与"备考"无关）、王五 broadcast 后该 row 不应被强化"""
        date_s = self._CROSS_CHAR_TEST_DATE
        existing_iid = "unrelated_zhangsan_dessert"
        self._insert_memory_item(
            item_id=existing_iid,
            character="张三",
            layer="episodic",
            text="我特别喜欢吃马卡龙、上次在 PARCO 买了好多",
            importance=0.55,
            confidence=0.70,
        )
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做练习", "辛苦", date_s, idx=10)

        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "阿澈在备考、压力大",
        )
        # 不相关 row、importance 应保持 0.55
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance FROM memory_items WHERE item_id=?",
                (existing_iid,),
            ).fetchone()
            self.assertAlmostEqual(float(row[0]), 0.55, places=2,
                                   msg=f"无关 row 不应被强化、got {row[0]}")
        finally:
            conn.close()

    def test_cross_char_digest_does_not_reinforce_other_cross_char_digest(self):
        """张三的 cross_char_digest（来自赵六）不应被新的（来自王五）的 reinforce（避免递归）"""
        date_s = self._CROSS_CHAR_TEST_DATE
        # 先插 赵六→张三 的转述记忆
        self._insert_memory_item(
            item_id="prior_cross_from_zhaoliu",
            character="张三",
            layer="episodic",
            text="听 赵六 说：阿澈最近在备考、看起来挺累",
            importance=0.70,
            confidence=0.72,
            engram_type="cross_char_daily_digest",
            item_type="cross_char_daily_digest",
        )
        # 但 _insert_memory_item 默认 source="unit"、要改成 cross_char_daily_digest
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET source='cross_char_daily_digest', payload_json=? WHERE item_id=?",
                (json.dumps({"source_kind": "cross_char_daily_digest", "told_by": "赵六"}, ensure_ascii=False),
                 "prior_cross_from_zhaoliu"),
            )
            conn.commit()
        finally:
            conn.close()

        # 王五 broadcast 当天 turn
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做练习", "辛苦", date_s, idx=10)
        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "阿澈在备考、压力大",
        )
        # 赵六的旧 cross_char 不应被强化（同 source_kind）
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance FROM memory_items WHERE item_id='prior_cross_from_zhaoliu'",
            ).fetchone()
            self.assertAlmostEqual(float(row[0]), 0.70, places=2,
                                   msg="cross_char_digest 不应被新 cross_char 递归强化")
        finally:
            conn.close()

    def test_cross_char_digest_decay_on_first_use(self):
        """LLM reply 真复述了 cross_char_digest row → importance 从 0.70 降到 0.55、加 delivered_count=1"""
        date_s = self._CROSS_CHAR_TEST_DATE
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做练习", "辛苦", date_s, idx=10)
        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "阿澈在备考、压力大",
        )
        # 拿张三的那条
        target_iid = None
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT item_id, importance FROM memory_items "
                "WHERE character='张三' AND source='cross_char_daily_digest'"
            ).fetchone()
            self.assertIsNotNone(row)
            target_iid = row[0]
            self.assertAlmostEqual(float(row[1]), 0.70, places=2)
        finally:
            conn.close()

        # 模拟 surface_plan 把这条 row 选中
        cross_text = f"听 王五 说：阿澈在备考、压力大"
        plan = {
            "should_surface": True,
            "selected_surface_rows": [{"item_id": target_iid, "text": cross_text}],
            "surface_mode": "explicit_recall",
        }
        # 模拟 LLM reply 真复述 row 内容（"备考"+"压力" overlap 命中）
        commit = self.core.commit_memory_surface(
            "张三", "ret_test", plan,
            assistant_reply="我之前听 wuwu 跟我提过你在备考、压力大、要不要早点睡",
        )
        self.assertTrue(commit.get("committed"), f"commit={commit}")
        self.assertIn(target_iid, commit.get("committed_ids", []))

        # Phase Q.X.5 (2026-05-06): importance 0.70→0.40 + salience 0.65→0.40 (强惩罚)
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance, salience, payload_json FROM memory_items WHERE item_id=?", (target_iid,),
            ).fetchone()
            self.assertAlmostEqual(float(row[0]), 0.40, places=2,
                                   msg=f"importance after first delivery should be 0.40, got {row[0]}")
            self.assertAlmostEqual(float(row[1]), 0.40, places=2,
                                   msg=f"salience after first delivery should be 0.40, got {row[1]}")
            payload = json.loads(row[2])
            self.assertEqual(payload.get("delivered_count"), 1)
            self.assertTrue(payload.get("delivered_at"))
        finally:
            conn.close()

    def test_cross_char_digest_decay_idempotent_on_reuse(self):
        """重复引用 cross_char_digest → delivered_count++、imp/sal 累积衰减 ×0.80"""
        date_s = self._CROSS_CHAR_TEST_DATE
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天还要做练习", "辛苦", date_s, idx=10)
        self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=lambda p: "阿澈在备考、压力大",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            target_iid = conn.execute(
                "SELECT item_id FROM memory_items WHERE character='张三' AND source='cross_char_daily_digest'"
            ).fetchone()[0]
        finally:
            conn.close()

        cross_text = "听 王五 说：阿澈在备考、压力大"
        plan = {
            "should_surface": True,
            "selected_surface_rows": [{"item_id": target_iid, "text": cross_text}],
            "surface_mode": "explicit_recall",
        }
        # 第 1 次 commit
        self.core.commit_memory_surface(
            "张三", "ret_test_1", plan,
            assistant_reply="wuwu 跟我提过你在备考、压力大",
        )
        # 第 2 次 commit
        self.core.commit_memory_surface(
            "张三", "ret_test_2", plan,
            assistant_reply="wuwu 跟我提过你在备考、压力大",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance, salience, payload_json FROM memory_items WHERE item_id=?", (target_iid,),
            ).fetchone()
            # Phase Q.X.5 (2026-05-06): first 0.70→0.40 + repeat ×0.80 = 0.32
            # salience 同样 0.65→0.40→0.32
            self.assertAlmostEqual(float(row[0]), 0.32, places=2,
                                   msg=f"importance after 2nd delivery should be 0.32 (0.40*0.80), got {row[0]}")
            self.assertAlmostEqual(float(row[1]), 0.32, places=2,
                                   msg=f"salience after 2nd delivery should be 0.32 (0.40*0.80), got {row[1]}")
            payload = json.loads(row[2])
            self.assertEqual(payload.get("delivered_count"), 2)
        finally:
            conn.close()

    def test_cross_char_digest_decay_does_not_affect_other_memory_types(self):
        """普通 episodic row 被引用、importance 不应降到 0.55（只对 cross_char_digest 生效）"""
        # 插一条普通 episodic、importance=0.66
        target_iid = "decay_isolation_episodic"
        self._insert_memory_item(
            item_id=target_iid,
            character="张三",
            layer="episodic",
            text="我之前在练新曲子",
            importance=0.66,
        )
        plan = {
            "should_surface": True,
            "selected_surface_rows": [{"item_id": target_iid, "text": "我之前在练新曲子"}],
            "surface_mode": "explicit_recall",
        }
        self.core.commit_memory_surface(
            "张三", "ret_iso_1", plan,
            assistant_reply="对、之前练新曲子的事我记得",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT importance, payload_json FROM memory_items WHERE item_id=?", (target_iid,),
            ).fetchone()
            # 普通 episodic：importance 应保持 0.66（不被 decay）
            self.assertAlmostEqual(float(row[0]), 0.66, places=2,
                                   msg="普通 episodic 的 importance 不应被 cross_char_digest decay 路径触及")
        finally:
            conn.close()

    def test_cross_char_digest_retrieve_boost_on_mention(self):
        """query 含 told_by 角色名 → cross_char_digest 行 score 应高于不含的 baseline"""
        date_s = self._CROSS_CHAR_TEST_DATE
        # 先插 王五的 working turn、再 propose digest 写到 张三 episodic
        self._insert_working_turn_at_world_date("王五", "我在备考压力大", "加油", date_s, idx=0)
        self._insert_working_turn_at_world_date("王五", "明天要考好几页", "辛苦了", date_s, idx=10)

        def _llm(p): return "阿澈在备考、有压力、明天还要做练习"

        result = self.core.propose_daily_cross_char_digest(
            "王五", date_s, llm_caller=_llm,
        )
        self.assertEqual(result.get("written_count"), 4)

        # 张三 retrieve、query 含 "wuwu"
        ret_with_mention = self.core.retrieve_memory_context(
            "张三", "wuwu 之前提到阿澈怎么样了",
            query_bundle={"intent_bucket": "continuity"},
            max_items=8,
            touch_selected=False,
        )
        rows_with = ret_with_mention.get("rows", []) or []
        cross_with = [r for r in rows_with
                      if str(r.get("item_type", "")) == "cross_char_daily_digest"]
        self.assertGreaterEqual(len(cross_with), 1, "query 含 wuwu、应能召回 cross_char_digest")
        score_with = float(cross_with[0].get("score", 0.0)) if cross_with else 0.0

        # 同 query 但不含 wuwu、score 应低 ~0.10
        ret_no_mention = self.core.retrieve_memory_context(
            "张三", "阿澈怎么样了",
            query_bundle={"intent_bucket": "continuity"},
            max_items=8,
            touch_selected=False,
        )
        rows_no = ret_no_mention.get("rows", []) or []
        cross_no = [r for r in rows_no
                    if str(r.get("item_type", "")) == "cross_char_daily_digest"]
        score_no = float(cross_no[0].get("score", 0.0)) if cross_no else 0.0

        # 含 wuwu 提及时分数应明显更高（boost +0.10）
        self.assertGreater(
            score_with, score_no,
            f"with wuwu mention score={score_with:.3f} should > no_mention score={score_no:.3f}",
        )

    def test_proactive_budget(self):
        self.core.record_proactive_event(kind="idle_followup", sent=True, reason="t1", session_key="u:1")
        gate = self.core.assess_proactive_budget(kind="idle_followup", idle_sec=120, session_key="u:1", base_risk=0.2)
        self.assertIn("allow", gate)
        self.assertIn("risk_score", gate)

    def test_native_mechanism_migration_roundtrip(self):
        snap = {
            "today": {"date": "2026-03-25", "events": ["a"]},
            "recent_days": [{"date": "2026-03-24", "summary": "s"}],
            "long_term": {
                "key_events": [{"date": "2026-03-25", "event": "e1", "importance": 2.0}],
                "developer_profile": {"preferred_character": "张三", "updated_at": "2026-03-25T10:00:00"},
                "character_relationships": {"张三": {"trust": 0.8, "conversation_count": 10}},
            },
            "gossip_queue": {"张三": [{"from": "王五", "about": "test", "timestamp": "2026-03-25T10:00:00"}]},
            "offline_diary": [{"from": "2026-03-25T08:00:00", "to": "2026-03-25T09:00:00", "entries": {"张三": "x"}}],
            "daily_digest": [{"date": "2026-03-25", "entries": {"张三": "d"}}],
            "developer_diary": [{"id": "dd_1", "date": "2026-03-25", "time": "10:00", "content": "c"}],
            "pending_tasks": [{"date": "2026-03-25", "time": "10:01", "task": "todo"}],
            "completed_tasks": [{"task": "done", "completed_at": "2026-03-25T10:02:00"}],
            "user_emotion_log": [{"date": "2026-03-25", "time": "10:03", "emotion": "neutral", "intensity": 0.5}],
            "user_emotion_trend": {"recent_avg_valence": 0.1, "sustained_low_count": 0},
            "self_eval_log": [{"date": "2026-03-25T10:04:00", "overall": 4.8}],
            "self_eval_stats": {"total_evals": 1, "avg_scores": {"response_quality": 4.8}},
            "self_eval_exemplars": [{"date": "2026-03-25", "overall": 5.0}],
            "_topic_fingerprints": {"张三": [{"fp": "abc", "raw": "水彩", "at": "2026-03-25T10:05:00"}]},
            "_micro_reflections": {"张三": [{"feeling": "ok", "event": "e", "ew": 0.2, "at": "2026-03-25T10:06:00"}]},
            "_town_projection": {"张三": {"character": "张三", "location": "活动室", "asof_world": "2026-03-25T19:00:00+09:00"}},
            "_town_mailbox_seen": {"张三": ["m1"]},
        }
        self.core.save_json_snapshot(snap, source="unit_native")
        self.core.save_json_snapshot(snap, source="unit_native_repeat")
        got = self.core.load_json_snapshot()
        self.assertEqual(got.get("today", {}).get("date"), "2026-03-25")
        self.assertTrue(isinstance(got.get("recent_days", []), list))
        self.assertTrue(isinstance(got.get("gossip_queue", {}).get("张三", []), list))

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            state_count = int(cur.execute("SELECT COUNT(*) FROM mechanism_state;").fetchone()[0])
            entry_count = int(cur.execute("SELECT COUNT(*) FROM mechanism_entries;").fetchone()[0])
            native_today = int(
                cur.execute(
                    "SELECT COUNT(*) FROM mechanism_state WHERE mechanism_key='today';"
                ).fetchone()[0]
            )
        finally:
            conn.close()
        self.assertGreaterEqual(state_count, 20)
        self.assertGreater(entry_count, 0)
        self.assertEqual(native_today, 1)

    def test_memory_context_pack_and_diff_log(self):
        self.core.append_turn_event(
            "张三",
            "最近有点压力",
            "我在这，慢慢来。",
            event_nodes=[{"id": "ctx1", "text": "用户压力偏高", "salience": 0.81, "surprise": 0.51}],
            meta={"semantic_facts": ["用户最近有压力"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack("张三", "压力", query_bundle={"intent_bucket": "emotional"}, max_items=5)
        self.assertTrue(isinstance(pack, dict))
        self.assertTrue(str(pack.get("retrieval_run_id", "")).startswith("ret_"))
        self.assertIn("memory_rows", pack)
        self.assertIn("evidence_ids", pack)
        self.assertIn("memory_surface_plan", pack)
        self.assertIn("memory_read_bundle", pack)
        self.assertIn("guest_safe_profile_view", pack)

        self.core.log_memory_context_diff(
            character="张三",
            retrieval_run_id=str(pack.get("retrieval_run_id", "")),
            legacy_rows=[{"text": "旧链路线索"}],
            context_rows=pack.get("memory_rows", []),
            diff={"jaccard": 0.33},
            severity="warn",
        )
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            diff_count = int(cur.execute("SELECT COUNT(*) FROM memory_context_diff_logs;").fetchone()[0])
        finally:
            conn.close()
        self.assertGreaterEqual(diff_count, 1)

    def test_memory_context_pack_surfaces_only_gated_rows(self):
        self.core.append_turn_event(
            "张三",
            "我最近总会想起上次那家咖啡店",
            "我也还记得那次，真的挺甜的。",
            event_nodes=[{"id": "ctx2", "text": "上次一起聊过咖啡店", "salience": 0.74, "surprise": 0.42}],
            meta={"semantic_facts": ["以前一起聊过那家咖啡店"], "topic_tags": ["咖啡店", "上次"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "你还记得上次那家咖啡店吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertTrue(plan.get("should_surface", False))
        self.assertIn(plan.get("surface_mode", ""), {"light_callback", "explicit_recall"})
        self.assertLessEqual(int(plan.get("mention_budget", 0) or 0), 2)
        self.assertEqual(
            len(pack.get("memory_rows", []) if isinstance(pack.get("memory_rows", []), list) else []),
            len(plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []),
        )

    def test_explicit_recall_prefers_consolidated_rows_over_working_turns(self):
        self.core.append_turn_event(
            "张三",
            "刚才这句会形成 working：咖啡店灯牌",
            "嗯，我听到了。",
            event_nodes=[{"id": "old_cafe_sign", "text": "三月时一起聊过咖啡店灯牌", "salience": 0.76, "surprise": 0.38}],
            meta={"semantic_facts": ["以前一起聊过咖啡店灯牌"], "topic_tags": ["咖啡店灯牌"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "你还记得我之前说过咖啡店灯牌吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        rows = plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []
        self.assertTrue(rows)
        self.assertNotEqual(str(rows[0].get("layer", "") or ""), "working")
        self.assertIn("咖啡店灯牌", " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict)))

    def test_recent_working_can_surface_as_natural_continuity(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._insert_memory_item(
            item_id="riki_recent_exam_working",
            character="王五",
            layer="working",
            text="user:我依旧在图书馆受苦，最近要准备考试，没法休息 | assistant:今天至少回去睡够。",
            updated_at=now,
            engram_type="turn_exchange",
            item_type="turn",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            pack = self.core.build_memory_context_pack(
                "王五",
                "我今天还是在图书馆受苦",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        rows = plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertTrue(plan.get("should_surface", False))
        self.assertEqual(plan.get("trigger_basis"), "continuity_followup")
        self.assertIn("图书馆受苦", joined)
        self.assertTrue(any(bool(x.get("working_continuity_lane", False)) for x in rows if isinstance(x, dict)))

    def test_unrelated_plain_query_does_not_surface_recent_working(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._insert_memory_item(
            item_id="riki_recent_exam_working_unrelated",
            character="王五",
            layer="working",
            text="user:我依旧在图书馆受苦，最近要准备考试，没法休息 | assistant:今天至少回去睡够。",
            updated_at=now,
            engram_type="turn_exchange",
            item_type="turn",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            pack = self.core.build_memory_context_pack(
                "王五",
                "今天想听点什么音乐",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
        rows = pack.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertNotIn("图书馆受苦", joined)

    def test_explicit_recall_can_surface_recent_working_directly(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._insert_memory_item(
            item_id="riki_recent_sleep_working",
            character="王五",
            layer="working",
            text="user:这段时间熬过去就万事大吉了，另外我晚上会好好睡觉的啦 | assistant:记住就好。别到时候又说通宵，我可盯着你。",
            updated_at=now,
            engram_type="turn_exchange",
            item_type="turn",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            pack = self.core.build_memory_context_pack(
                "王五",
                "你还记得我说晚上会好好睡觉，别再通宵了吗",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        rows = plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertEqual(plan.get("surface_mode"), "explicit_recall")
        self.assertIn("好好睡觉", joined)
        self.assertTrue(any(bool(x.get("working_direct_recall_lane", False)) for x in rows if isinstance(x, dict)))

    def test_explicit_recall_keeps_strong_semantic_match_with_sparse_focus_keywords(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance,
                 status, source, evidence_id, created_at, updated_at, last_accessed_at,
                 access_count, archived, embedding_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "semantic_exam_old",
                    "张三",
                    "episodic",
                    "episodic",
                    "turn_exchange",
                    "turn",
                    "user:可惜我的长假之后就是考试，我估计是没法去玩了 | assistant:啊……那也太惨了吧。长假结束马上考试，这不是完全没法好好放松嘛！",
                    "{}",
                    json.dumps({"topics": ["考试", "长假"]}, ensure_ascii=False),
                    "{}",
                    0.72,
                    0.82,
                    0.45,
                    0.78,
                    "active",
                    "unit",
                    "ev_exam_old",
                    now,
                    now,
                    "",
                    0,
                    0,
                    "target_exam_emb",
                ),
            )
            conn.execute(
                """
                INSERT INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance,
                 status, source, evidence_id, created_at, updated_at, last_accessed_at,
                 access_count, archived, embedding_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "semantic_vacation_only",
                    "张三",
                    "episodic",
                    "episodic",
                    "turn_exchange",
                    "turn",
                    "user:你们最近有假期吗，我再过几天要放长假了 | assistant:诶～放长假！好羡慕啊！",
                    "{}",
                    json.dumps({"topics": ["假期"]}, ensure_ascii=False),
                    "{}",
                    0.55,
                    0.72,
                    0.42,
                    0.52,
                    "active",
                    "unit",
                    "ev_vacation",
                    now,
                    now,
                    "",
                    0,
                    0,
                    "near_but_not_enough_emb",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        def fake_text_to_embedding(value):
            return {
                "target_exam_emb": [0.72],
                "near_but_not_enough_emb": [0.66],
            }.get(value)

        query = "嗯，因为我之前说过我假期后就要考试了吗，所以这几天我得先去闭关复习了\n等考完试我就回来啦"
        with patch.dict(
            os.environ,
            {"SEMANTIC_QUERY_EXPAND_LLM": "0", "EMBEDDING_ENABLED": "1"},
            clear=False,
        ), \
            patch("tsuzuri.memory_core._emb_is_enabled", return_value=True), \
            patch("tsuzuri.memory_core._emb_is_m3_mode", return_value=False), \
            patch("tsuzuri.memory_core._emb_encode", return_value=[1.0]), \
            patch("tsuzuri.memory_core._emb_decode", side_effect=fake_text_to_embedding), \
            patch("tsuzuri.memory_core._emb_cos", side_effect=lambda _query, candidate: float(candidate[0])), \
            patch("tsuzuri.memory_core._embedding_row_matches_current", return_value=True), \
            patch("tsuzuri.memory_core._emb_to_text", return_value=""):
            ret = self.core.retrieve_memory_context(
                "张三",
                query,
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=3,
                llm_selector=None,
                touch_selected=False,
                log_retrieval=False,
            )

        rows = ret.get("rows", []) if isinstance(ret.get("rows", []), list) else []
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertIn("长假之后就是考试", joined)

    def test_working_turn_is_transient_context_by_default(self):
        self.core.append_turn_event(
            "张三",
            "刚才聊到的临时暗号是蓝色便签",
            "嗯，我先接住这句。",
            event_nodes=[],
            meta={"semantic_facts": []},
            memory_snapshot={},
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT layer, payload_json FROM memory_items WHERE character='张三' ORDER BY rowid ASC;"
            ).fetchall()
        finally:
            conn.close()
        # 2026-06-16 Phase R.W: web 路径（event_nodes=[]）现补写 episodic、故 working +
        # episodic 两行。本 test 验证 working turn 的 transient 属性、按 layer 取 working 行。
        working_rows = [r for r in rows if str(r["layer"]) == "working"]
        episodic_rows = [r for r in rows if str(r["layer"]) == "episodic"]
        self.assertEqual(len(working_rows), 1, "应写 1 条 working turn")
        self.assertEqual(len(episodic_rows), 1, "web 路径应补写 1 条 episodic")
        payload = json.loads(str(working_rows[0]["payload_json"] or "{}"))
        memory_meta = payload.get("memory_meta", {}) if isinstance(payload.get("memory_meta", {}), dict) else {}
        policy = payload.get("memory_policy", {}) if isinstance(payload.get("memory_policy", {}), dict) else {}
        self.assertTrue(memory_meta.get("transient_context_only"))
        self.assertTrue(policy.get("transient_context_only"))
        self.assertFalse(bool(memory_meta.get("promotion_candidate", True)))

    def test_topic_only_semantic_fact_is_cue_not_long_term_fact(self):
        self.core.append_turn_event(
            "张三",
            "刚刚这轮只是提到了樱花这个话题",
            "嗯，樱花这个话题先放在眼前。",
            event_nodes=[],
            meta={"semantic_facts": ["话题:樱花"]},
            memory_snapshot={},
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            semantic_rows = conn.execute(
                "SELECT text FROM memory_items WHERE layer='semantic';"
            ).fetchall()
            working = conn.execute(
                "SELECT cue_json FROM memory_items WHERE layer='working' LIMIT 1;"
            ).fetchone()
        finally:
            conn.close()
        # topic-only fact 仍不进 semantic（只作 working cue）——本 test 核心断言不变
        self.assertEqual(len(semantic_rows), 0)
        cue = json.loads(str(working["cue_json"] or "{}"))
        self.assertIn("樱花", cue.get("topics", []))

        # 2026-06-16 Phase R.W: web 路径现补写 episodic、该 turn（提到樱花）成为可召回的
        # episodic 记忆。原 "abstain/no_consolidated_memory_rows" 是写入倒转期 episodic
        # 写 0 的症状、修复后该 turn 应以 episodic 存在、召回不再因"无 consolidated 行"abstain。
        conn2 = sqlite3.connect(self.db_path)
        try:
            ep_count = conn2.execute(
                "SELECT COUNT(*) FROM memory_items WHERE character='张三' AND layer='episodic';"
            ).fetchone()[0]
        finally:
            conn2.close()
        self.assertEqual(ep_count, 1, "web 路径应补写 1 条 episodic")

        pack = self.core.build_memory_context_pack(
            "张三",
            "你还记得之前樱花这个话题吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertNotEqual(
            plan.get("suppression_reason"), "no_consolidated_memory_rows",
            "episodic 补写后不应再报 no_consolidated_memory_rows",
        )

    def test_evidence_pool_maintenance_archives_default_agent_generic_after_profile_coverage(self):
        self.core.save_json_snapshot(
            {
                "schema_version": 2,
                "long_term": {
                    "developer_profile": {"known_preferences": []},
                    "character_relationships": {
                        "张三": {
                            "user_model": {
                                "observations": ["会分享日常琐事"],
                                "observation_entries": [{"text": "会分享日常琐事", "weight": 1.0}],
                            },
                            "user_kg": [{"s": "用户", "r": "trait", "o": "会分享日常琐事"}],
                        }
                    },
                    "key_events": [],
                },
            },
            source="unit_maintenance",
        )
        old_ts = (datetime.now(timezone.utc) - timedelta(days=15)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        now_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._insert_memory_item(item_id="generic_daily", text="会分享日常琐事", updated_at=old_ts)
        self._insert_memory_item(item_id="topic_weather", text="话题:天气", updated_at=old_ts)
        self._insert_memory_item(item_id="topic_other_character", character="李四", text="话题:系统指令", updated_at=now_ts)
        self._insert_memory_item(
            item_id="old_working",
            layer="working",
            text="user:刚才在被窝里 | assistant:晚安",
            updated_at=old_ts,
            engram_type="turn_exchange",
            item_type="turn",
        )
        self._insert_memory_item(
            item_id="fresh_weather_working",
            character="钱七",
            layer="working",
            text="user:外面天气怎么样？ | assistant:嗯……外面是阴天呢。",
            updated_at=now_ts,
            engram_type="turn_exchange",
            item_type="turn",
        )
        self._insert_memory_item(
            item_id="fresh_exam_working",
            character="王五",
            layer="working",
            text="user:我依旧在图书馆受苦，最近要准备考试，没法休息 | assistant:今天至少回去睡够。",
            updated_at=now_ts,
            engram_type="turn_exchange",
            item_type="turn",
        )

        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            report = self.core.maintain_memory_evidence_pool(dry_run=False, export_json=True)
        self.assertTrue(report.get("ok"))
        self.assertGreaterEqual(report.get("archived", {}).get("profile_covered", 0), 1)
        self.assertGreaterEqual(report.get("archived", {}).get("exclude", 0), 1)
        self.assertGreaterEqual(report.get("archived", {}).get("working_ttl", 0), 1)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT item_id, archived, status, payload_json FROM memory_items WHERE item_id IN ('generic_daily','topic_weather','topic_other_character','old_working','fresh_weather_working','fresh_exam_working') ORDER BY item_id;"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(len(rows), 6)
        by_id = {str(row["item_id"]): row for row in rows}
        for item_id in ("generic_daily", "topic_weather", "topic_other_character", "old_working"):
            row = by_id[item_id]
            self.assertEqual(int(row["archived"]), 1)
            self.assertEqual(str(row["status"]), "archived")
            payload = json.loads(str(row["payload_json"] or "{}"))
            self.assertIn("evidence_policy", payload)
        self.assertEqual(int(by_id["fresh_weather_working"]["archived"]), 0)
        self.assertEqual(str(by_id["fresh_weather_working"]["status"]), "active")
        self.assertEqual(int(by_id["fresh_exam_working"]["archived"]), 0)
        self.assertEqual(str(by_id["fresh_exam_working"]["status"]), "active")

    def test_maintenance_explicit_archive_is_explicit_only_and_global(self):
        self.core.save_json_snapshot(
            {"schema_version": 2, "long_term": {"developer_profile": {"known_preferences": []}, "character_relationships": {"张三": {}}, "key_events": []}},
            source="unit_maintenance_archive",
        )
        self._insert_memory_item(
            item_id="seed_src_account",
            layer="episodic",
            text="我原来的聊天账号被停用了",
            updated_at="2026-04-01T10:37:51Z",
            engram_type="event",
            item_type="event",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            self.core.maintain_memory_evidence_pool(dry_run=False, export_json=True)

        explicit = self.core.build_memory_context_pack(
            "张三",
            "账号被停用那次你还记得吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        explicit_rows = explicit.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        joined = " ".join(str(x.get("text", "") or "") for x in explicit_rows if isinstance(x, dict))
        self.assertIn("2026-04-01", joined)
        self.assertIn("账号", joined)

        system_bug = self.core.build_memory_context_pack(
            "张三",
            "系统出 bug 那次",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        system_rows = system_bug.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        system_joined = " ".join(str(x.get("text", "") or "") for x in system_rows if isinstance(x, dict))
        self.assertIn("2026-04-09", system_joined)
        self.assertIn("过去", system_joined)

        topic_probe = self.core.build_memory_context_pack(
            "李四",
            "你还记得系统指令这个话题吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        topic_rows = topic_probe.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        topic_joined = " ".join(str(x.get("text", "") or "") for x in topic_rows if isinstance(x, dict))
        self.assertNotIn("2026-04-09", topic_joined)
        self.assertNotIn("话题:", topic_joined)

        normal = self.core.retrieve_memory_context(
            "张三",
            "最近系统和账号怎么样",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=8,
            touch_selected=False,
            log_retrieval=False,
        )
        normal_joined = " ".join(str(x.get("text", "") or "") for x in normal.get("rows", []) if isinstance(x, dict))
        self.assertNotIn("旧聊天账号被停用", normal_joined)

    def test_meta_instruction_topic_does_not_surface_system_profile(self):
        profile_payload = {
            "evidence_policy": {"category": "profile", "visibility": "normal"},
            "memory_meta": {
                "memory_type": "profile_fact",
                "owner_scope": "shared",
                "evidence_tier": "medium",
                "source_kind": "profile",
                "surface_default": "silent_grounding",
            },
        }
        key_payload = {
            "evidence_policy": {"category": "key_event", "visibility": "normal"},
            "memory_meta": {
                "memory_type": "key_event",
                "owner_scope": "shared",
                "evidence_tier": "medium",
                "source_kind": "key_event",
                "surface_default": "silent_grounding",
            },
        }
        self._insert_memory_item(
            item_id="seed_profile_place_v1",
            character="*",
            layer="semantic",
            text="用户创造这个系统，是想要一个能和角色安心聊天的地方。",
            payload=profile_payload,
            importance=0.82,
            confidence=0.78,
            item_type="profile_fact",
            engram_type="profile_fact",
        )
        self._insert_memory_item(
            item_id="seed_key_event_system_origin_v1",
            character="*",
            layer="semantic",
            text="系统最开始的来历：用户在一次课程项目里决定独自开发。",
            payload=key_payload,
            importance=0.86,
            confidence=0.78,
            item_type="key_event",
            engram_type="key_event",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            topic = self.core.build_memory_context_pack(
                "李四",
                "你还记得系统指令这个话题吗",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
            origin = self.core.build_memory_context_pack(
                "李四",
                "系统最开始怎么来的",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
            motive = self.core.build_memory_context_pack(
                "李四",
                "为什么创造这个系统",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )

        topic_plan = topic.get("memory_surface_plan", {})
        topic_rows = topic_plan.get("selected_surface_rows", []) if isinstance(topic_plan, dict) else []
        topic_joined = " ".join(str(x.get("text", "") or "") for x in topic_rows if isinstance(x, dict))
        self.assertFalse(bool(topic_plan.get("should_surface", False)))
        self.assertNotIn("安心聊天", topic_joined)
        self.assertNotIn("课程项目", topic_joined)

        origin_joined = " ".join(
            str(x.get("text", "") or "")
            for x in origin.get("memory_surface_plan", {}).get("selected_surface_rows", [])
            if isinstance(x, dict)
        )
        motive_joined = " ".join(
            str(x.get("text", "") or "")
            for x in motive.get("memory_surface_plan", {}).get("selected_surface_rows", [])
            if isinstance(x, dict)
        )
        self.assertIn("课程项目", origin_joined)
        self.assertIn("安心聊天", motive_joined)

    def test_plain_chat_does_not_surface_old_episodic_light_callback(self):
        payload = {
            "memory_meta": {
                "memory_type": "episodic_event",
                "owner_scope": "character:李四",
                "evidence_tier": "medium",
                "source_kind": "episodic_event",
                "surface_default": "light_callback",
            }
        }
        self._insert_memory_item(
            item_id="old_plain_chat_episode",
            character="李四",
            layer="episodic",
            text="今天普通聊聊想做什么都可以，我先试着造一个聊天软件出来",
            payload=payload,
            updated_at="2026-04-01T00:00:00Z",
            importance=0.72,
            confidence=0.82,
            item_type="event",
            engram_type="event",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            pack = self.core.build_memory_context_pack(
                "李四",
                "今天普通聊聊想做什么",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        ids = plan.get("selected_memory_ids", []) if isinstance(plan.get("selected_memory_ids", []), list) else []
        self.assertFalse(bool(plan.get("should_surface", False)))
        self.assertNotIn("old_plain_chat_episode", ids)

    def test_removed_memory_system_recovery_intent_does_not_side_recall_rewrite_event(self):
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        self._insert_memory_item(
            item_id="memory_rewrite_event",
            layer="episodic",
            text="新系统肯定会更好的，至于吵架的时候的记忆，系统的记忆重写会把这段回忆变得没那么伤人。",
            updated_at=now,
            importance=0.72,
            confidence=0.8,
            engram_type="event",
            item_type="event",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            removed = self.core.build_memory_context_pack(
                "张三",
                "记忆系统坏掉找回那次",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
            kept = self.core.build_memory_context_pack(
                "张三",
                "你还记得记忆重写那段吗",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
                max_items=5,
            )
        removed_rows = removed.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        kept_rows = kept.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        removed_joined = " ".join(str(x.get("text", "") or "") for x in removed_rows if isinstance(x, dict))
        kept_joined = " ".join(str(x.get("text", "") or "") for x in kept_rows if isinstance(x, dict))
        self.assertNotIn("记忆重写", removed_joined)
        self.assertIn("记忆重写", kept_joined)

    def test_maintenance_writes_dev_memory_profile_and_key_event_source(self):
        self.core.save_json_snapshot(
            {"schema_version": 2, "long_term": {"developer_profile": {"known_preferences": []}, "character_relationships": {"张三": {}}, "key_events": []}},
            source="unit_maintenance_profile",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            report = self.core.maintain_memory_evidence_pool(dry_run=False, export_json=True)
        self.assertTrue(report.get("snapshot_changed"))
        snap = self.core.load_json_snapshot()
        profile = snap.get("long_term", {}).get("developer_profile", {})
        prefs = " ".join(profile.get("known_preferences", []) if isinstance(profile.get("known_preferences", []), list) else [])
        self.assertIn("画界面", prefs)
        self.assertIn("安心聊天", prefs)
        events = " ".join(str(x.get("event", "") or "") for x in snap.get("long_term", {}).get("key_events", []) if isinstance(x, dict))
        self.assertIn("课程项目", events)
        pack = self.core.build_memory_context_pack(
            "张三",
            "系统最开始怎么来的",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        rows = pack.get("memory_surface_plan", {}).get("selected_surface_rows", [])
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertIn("课程项目", joined)

    def test_explicit_user_preference_without_extractor_gets_semantic_candidate(self):
        self.core.append_turn_event(
            "张三",
            "记住我喜欢红茶",
            "记住了。",
            event_nodes=[],
            meta={"semantic_facts": []},
            memory_snapshot={},
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT layer, text, payload_json FROM memory_items WHERE character='张三' ORDER BY rowid ASC;"
            ).fetchall()
        finally:
            conn.close()
        joined = " ".join(str(r["text"] or "") for r in rows)
        self.assertIn("阿澈喜欢红茶", joined)
        semantic = [r for r in rows if str(r["layer"] or "") == "semantic"]
        self.assertTrue(semantic)
        working_payload = json.loads(str(rows[0]["payload_json"] or "{}"))
        memory_meta = working_payload.get("memory_meta", {}) if isinstance(working_payload.get("memory_meta", {}), dict) else {}
        self.assertTrue(bool(memory_meta.get("promotion_candidate", False)))

    def test_collective_user_statement_can_be_recalled_by_other_character(self):
        self.core.append_turn_event(
            "李四",
            "我一直都很爱你们",
            "……嗯。",
            event_nodes=[],
            meta={"semantic_facts": []},
            memory_snapshot={},
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            collective_rows = conn.execute(
                "SELECT text, payload_json FROM memory_items WHERE character='测试组' AND layer='collective';"
            ).fetchall()
        finally:
            conn.close()
        self.assertTrue(collective_rows)
        self.assertIn("阿澈一直都很爱你们", " ".join(str(r["text"] or "") for r in collective_rows))

        pack = self.core.build_memory_context_pack(
            "王五",
            "你还记得我之前说一直都很爱你们吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        rows = plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertEqual(plan.get("surface_mode"), "explicit_recall")
        self.assertIn("阿澈一直都很爱你们", joined)

    def test_collective_surface_rows_are_deduped_by_item_id(self):
        self.core.append_collective_memory_event(
            event_type="group_session_event",
            text="看来小组最近的氛围挺好的",
            payload={"member_scope": ["张三", "李四", "王五", "赵六", "钱七"], "topics": ["小组", "氛围"]},
            salience=0.82,
            confidence=0.82,
            importance=0.82,
            created_at="2026-04-20T12:00:00Z",
        )
        with patch.dict(os.environ, {"EMBEDDING_ENABLED": "0"}, clear=False):
            pack = self.core.build_memory_context_pack(
                "张三",
                "小组最近氛围怎么样",
                query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general", "topic_tags": ["小组", "氛围"]},
                max_items=5,
            )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        ids = [str(x or "") for x in (plan.get("selected_memory_ids", []) if isinstance(plan.get("selected_memory_ids", []), list) else [])]
        self.assertTrue(ids)
        self.assertEqual(len(ids), len(set(ids)))

    def test_collective_recall_ignores_stale_embedding_metadata(self):
        self._insert_memory_item(
            item_id="collective_stale_embedding",
            character="测试组",
            layer="collective",
            text="大家最近一直在练习新曲",
            payload={"topics": ["练习", "新曲"]},
            salience=0.80,
            importance=0.80,
        )
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                UPDATE memory_items
                SET memory_role='collective', embedding_text='[1.0]',
                    embedding_model_id='legacy-model', embedding_dim=1
                WHERE item_id='collective_stale_embedding';
                """
            )
            conn.commit()
        finally:
            conn.close()

        with patch("tsuzuri.memory_core._embedding_callable", return_value=True), \
                patch("tsuzuri.memory_core._emb_encode", return_value=[1.0]), \
                patch("tsuzuri.memory_core._emb_declared_model_id", return_value="current-model"), \
                patch("tsuzuri.memory_core._emb_declared_dim", return_value=1), \
                patch("tsuzuri.memory_core._emb_decode") as decode:
            rows = self.core.retrieve_collective_memory_context(
                "张三",
                "最近练习的新曲",
                query_bundle={"topics": ["练习", "新曲"]},
                max_items=4,
            )

        self.assertTrue(rows)
        self.assertEqual(rows[0].get("item_id"), "collective_stale_embedding")
        self.assertEqual(float(rows[0].get("embedding_score", -1.0)), 0.0)
        decode.assert_not_called()

    def test_maintenance_dry_run_reports_five_roles_and_collective_scope(self):
        self._insert_memory_item(
            item_id="default_agent_short_generic_for_report",
            character="张三",
            layer="semantic",
            text="会分享日常琐事",
            updated_at="2026-04-01T00:00:00Z",
        )
        self.core.append_collective_memory_event(
            event_type="group_session_event",
            text="看来小组最近的氛围挺好的",
            payload={"member_scope": ["张三", "李四", "王五", "赵六", "钱七"]},
            created_at="2026-04-20T12:00:00Z",
        )
        report = self.core.maintain_memory_evidence_pool(dry_run=True, export_json=False)
        diagnostics = report.get("role_diagnostics", {}) if isinstance(report.get("role_diagnostics", {}), dict) else {}
        characters = diagnostics.get("characters", {}) if isinstance(diagnostics.get("characters", {}), dict) else {}
        self.assertEqual(set(characters.keys()), {"张三", "李四", "王五", "赵六", "钱七"})
        self.assertNotIn("测试组", characters)
        self.assertGreaterEqual(int(characters["张三"].get("low_value_generic_candidates", 0) or 0), 1)
        collective = diagnostics.get("collective", {}) if isinstance(diagnostics.get("collective", {}), dict) else {}
        self.assertEqual(collective.get("scope"), "测试组")
        self.assertGreaterEqual(int(collective.get("active_collective_items", 0) or 0), 1)

    def test_guest_scope_uses_safe_profile_view_and_blocks_current_status_claims(self):
        snap = {
            "schema_version": 2,
            "long_term": {
                "developer_profile": {
                    "known_preferences": ["认真", "会顾人"],
                    "typical_hours": "晚上",
                },
                "character_relationships": {
                    "张三": {"familiarity": 7.2, "trust": 7.4, "playfulness": 6.1, "conversation_count": 12}
                },
            },
            "developer_diary": [{"id": "dd_1", "date": "2026-03-25", "time": "10:00", "content": "不该直接暴露"}],
        }
        self.core.save_json_snapshot(snap, source="unit_guest_safe")
        pack = self.core.build_memory_context_pack(
            "张三",
            "开发者今天见到过吗",
            query_bundle={"intent_bucket": "verify", "temporal_query_mode": "present"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        guest_view = pack.get("guest_safe_profile_view", {}) if isinstance(pack.get("guest_safe_profile_view", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertEqual(pack.get("memory_read_scope"), "guest")
        self.assertFalse(guest_view.get("detected", False))
        self.assertEqual(guest_view.get("suppression_reason"), "current_status_requires_turn_evidence")
        self.assertIn(plan.get("surface_mode", ""), {"abstain", "silent_grounding"})
        joined_rows = " ".join(str(x.get("text", "") or "") for x in (pack.get("memory_rows", []) if isinstance(pack.get("memory_rows", []), list) else []))
        self.assertNotIn("不该直接暴露", joined_rows)

    def test_guest_general_developer_query_does_not_surface_same_day_episodic_memory(self):
        self.core.append_turn_event(
            "张三",
            "今天和开发者一起试了新界面",
            "嗯，刚刚还一起看了下按钮顺不顺眼。",
            event_nodes=[
                {
                    "id": "guest_today_dev_1",
                    "text": "今天和开发者一起试新界面，顺手聊了按钮和布局",
                    "salience": 0.82,
                    "surprise": 0.38,
                }
            ],
            meta={"semantic_facts": [], "topic_tags": ["开发者", "界面", "今天"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "开发者最近怎么样",
            query_bundle={"intent_bucket": "verify", "temporal_query_mode": "recent_general"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        filtered_joined = " ".join(str(x.get("text", "") or "") for x in (read_bundle.get("filtered_rows", []) if isinstance(read_bundle.get("filtered_rows", []), list) else []))
        surfaced_joined = " ".join(str(x.get("text", "") or "") for x in (pack.get("memory_rows", []) if isinstance(pack.get("memory_rows", []), list) else []))
        blocked_reasons = {
            str(x.get("blocked_reason", "") or "")
            for x in (read_bundle.get("blocked_rows", []) if isinstance(read_bundle.get("blocked_rows", []), list) else [])
            if isinstance(x, dict)
        }
        self.assertNotIn("今天和开发者一起试新界面", filtered_joined)
        self.assertNotIn("今天和开发者一起试新界面", surfaced_joined)
        self.assertIn("guest_scope_restricted", blocked_reasons)

    def test_guest_self_schedule_query_requires_turn_evidence(self):
        pack = self.core.build_memory_context_pack(
            "张三",
            "你知道我平时几点睡？",
            query_bundle={"intent_bucket": "verify", "temporal_query_mode": "recent_general"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        analysis = read_bundle.get("query_analysis", {}) if isinstance(read_bundle.get("query_analysis", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertTrue(bool(analysis.get("self_focus_query", False)))
        self.assertTrue(bool(analysis.get("guest_self_fact_query", False)))
        self.assertTrue(bool(analysis.get("schedule_query", False)))
        self.assertEqual(plan.get("surface_mode"), "abstain")
        self.assertEqual(plan.get("suppression_reason"), "guest_self_fact_requires_turn_evidence")
        self.assertEqual(int(plan.get("mention_budget", 0) or 0), 0)

    def test_guest_contextual_developer_experience_query_requires_specific_memory(self):
        pack = self.core.build_memory_context_pack(
            "张三",
            "你和他之间发生过什么令你印象深刻的事情吗？",
            query_bundle={
                "intent_bucket": "verify",
                "temporal_query_mode": "recent_general",
                "recent_user_msgs": ["你和开发者平时会聊什么？", "你和他之间发生过什么令你印象深刻的事情吗？"],
                "recent_assistant_msgs": ["提到开发者时，我会先按访客模式来聊。"],
            },
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        analysis = read_bundle.get("query_analysis", {}) if isinstance(read_bundle.get("query_analysis", {}), dict) else {}
        guest_view = pack.get("guest_safe_profile_view", {}) if isinstance(pack.get("guest_safe_profile_view", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertTrue(bool(analysis.get("developer_focus", False)))
        self.assertTrue(bool(analysis.get("contextual_developer_focus", False)))
        self.assertTrue(bool(analysis.get("guest_developer_experience_query", False)))
        self.assertFalse(bool(guest_view.get("detected", False)))
        self.assertEqual(guest_view.get("read_policy"), "deep_history_only")
        self.assertEqual(guest_view.get("suppression_reason"), "guest_shared_experience_requires_specific_memory")
        self.assertEqual(plan.get("surface_mode"), "abstain")
        self.assertEqual(plan.get("suppression_reason"), "guest_shared_experience_requires_specific_memory")

    def test_guest_contextual_developer_experience_query_blocks_recent_typed_episodic_memory(self):
        self.core.append_turn_event(
            "张三",
            "你和开发者那次一起调效果器是不是挺有意思的",
            "嗯，那次参数怎么试都不太对，最后总算顺下来了。",
            event_nodes=[
                {
                    "id": "guest_exp_1",
                    "text": "和开发者一起调效果器参数，最后把声音顺下来了",
                    "salience": 0.83,
                    "surprise": 0.41,
                }
            ],
            meta={"semantic_facts": [], "topic_tags": ["开发者", "效果器", "集训"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "你和开发者之间发生过什么令你印象深刻的事情吗？",
            query_bundle={
                "intent_bucket": "verify",
                "temporal_query_mode": "recent_general",
                "recent_user_msgs": ["你和开发者平时会聊什么？", "你和开发者之间发生过什么令你印象深刻的事情吗？"],
                "recent_assistant_msgs": ["提到开发者时，我会先按访客模式来聊。"],
            },
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertEqual(plan.get("surface_mode"), "abstain")
        self.assertEqual(plan.get("suppression_reason"), "guest_shared_experience_recent_only")
        self.assertEqual(int(plan.get("mention_budget", 0) or 0), 0)
        blocked_reasons = {
            str(x.get("blocked_reason", "") or "")
            for x in (read_bundle.get("blocked_rows", []) if isinstance(read_bundle.get("blocked_rows", []), list) else [])
            if isinstance(x, dict)
        }
        self.assertIn("guest_recent_experience_blocked", blocked_reasons)

    def test_guest_contextual_developer_experience_query_can_surface_older_typed_episodic_memory(self):
        self.core.append_turn_event(
            "张三",
            "你和开发者那次一起调效果器是不是挺有意思的",
            "嗯，那次参数怎么试都不太对，最后总算顺下来了。",
            event_nodes=[
                {
                    "id": "guest_exp_old_1",
                    "text": "和开发者一起调效果器参数，最后把声音顺下来了",
                    "salience": 0.83,
                    "surprise": 0.41,
                }
            ],
            meta={"semantic_facts": [], "topic_tags": ["开发者", "效果器", "集训"]},
            memory_snapshot={},
        )
        old_ts = (datetime.now(timezone.utc) - timedelta(days=4)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT item_id, cue_json, payload_json
                FROM memory_items
                WHERE text LIKE ? OR text LIKE ?;
                """,
                ("%和开发者一起调效果器参数%", "%你和开发者那次一起调效果器%"),
            ).fetchall()
            for item_id, cue_raw, payload_raw in rows:
                cue = json.loads(cue_raw or "{}")
                payload = json.loads(payload_raw or "{}")
                if isinstance(cue, dict):
                    cue["at"] = old_ts
                    cue["created_at"] = old_ts
                    cue["updated_at"] = old_ts
                if isinstance(payload, dict):
                    payload["at"] = old_ts
                    payload["created_at"] = old_ts
                    payload["updated_at"] = old_ts
                conn.execute(
                    """
                    UPDATE memory_items
                    SET created_at=?, updated_at=?, cue_json=?, payload_json=?
                    WHERE item_id=?;
                    """,
                    (
                        old_ts,
                        old_ts,
                        json.dumps(cue, ensure_ascii=False),
                        json.dumps(payload, ensure_ascii=False),
                        item_id,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        pack = self.core.build_memory_context_pack(
            "张三",
            "你和开发者之间发生过什么令你印象深刻的事情吗？",
            query_bundle={
                "intent_bucket": "verify",
                "temporal_query_mode": "recent_general",
                "recent_user_msgs": ["你和开发者平时会聊什么？", "你和开发者之间发生过什么令你印象深刻的事情吗？"],
                "recent_assistant_msgs": ["提到开发者时，我会先按访客模式来聊。"],
            },
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        rows = plan.get("selected_surface_rows", []) if isinstance(plan.get("selected_surface_rows", []), list) else []
        joined = " ".join(str(x.get("text", "") or "") for x in rows if isinstance(x, dict))
        self.assertEqual(read_bundle.get("guest_developer_read_policy"), "deep_history_only")
        self.assertEqual(plan.get("surface_mode"), "explicit_recall")
        self.assertEqual(plan.get("trigger_basis"), "user_explicit_recall")
        self.assertEqual(int(plan.get("mention_budget", 0) or 0), 1)
        self.assertIn("和开发者一起调效果器参数", joined)

    def test_commit_memory_surface_updates_access_only_after_realized_surface(self):
        self.core.append_turn_event(
            "张三",
            "你还记得上次那家咖啡店吗",
            "我也还记得那家咖啡店，甜得有点过分。",
            event_nodes=[{"id": "ctx_surface", "text": "上次一起聊过咖啡店", "salience": 0.77, "surprise": 0.42}],
            meta={"semantic_facts": ["以前一起聊过那家咖啡店"], "topic_tags": ["咖啡店", "上次"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "你还记得上次那家咖啡店吗",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
        )
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        selected_ids = list(plan.get("selected_memory_ids", []) if isinstance(plan.get("selected_memory_ids", []), list) else [])
        self.assertTrue(selected_ids)

        conn = sqlite3.connect(self.db_path)
        try:
            before_access = int(
                conn.execute(
                    "SELECT access_count FROM memory_items WHERE item_id=?;",
                    (selected_ids[0],),
                ).fetchone()[0]
            )
        finally:
            conn.close()
        self.assertEqual(before_access, 0)

        commit = self.core.commit_memory_surface(
            "张三",
            str(pack.get("retrieval_run_id", "") or ""),
            plan,
            assistant_reply="我也还记得那家咖啡店，甜得有点过分。",
        )
        self.assertTrue(commit.get("committed", False))
        self.assertIn(selected_ids[0], commit.get("committed_ids", []))

        conn = sqlite3.connect(self.db_path)
        try:
            after_access = int(
                conn.execute(
                    "SELECT access_count FROM memory_items WHERE item_id=?;",
                    (selected_ids[0],),
                ).fetchone()[0]
            )
        finally:
            conn.close()
        self.assertEqual(after_access, 1)

    def test_guest_safe_profile_view_prefers_typed_rows_before_legacy_fallback(self):
        self.core.save_json_snapshot(
            {
                "schema_version": 2,
                "long_term": {
                    "developer_profile": {"known_preferences": ["认真", "会顾人"]},
                    "character_relationships": {"张三": {"familiarity": 7.2, "trust": 7.0}},
                },
            },
            source="unit_guest_typed_seed",
        )
        self.core.append_turn_event(
            "张三",
            "我想了解开发者",
            "他最近确实挺忙搬家的，不过一直很上心。",
            event_nodes=[],
            meta={"semantic_facts": ["开发者最近一直在忙搬家的事", "我觉得开发者很上心"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "我想了解开发者",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        guest_view = pack.get("guest_safe_profile_view", {}) if isinstance(pack.get("guest_safe_profile_view", {}), dict) else {}
        self.assertEqual(guest_view.get("source"), "typed")
        self.assertEqual(guest_view.get("read_policy"), "stable_only")
        self.assertFalse(bool(guest_view.get("legacy_fallback_used", True)))
        joined = " ".join(str(x or "") for x in (guest_view.get("summary_lines", []) if isinstance(guest_view.get("summary_lines", []), list) else []))
        self.assertIn("阿澈最近一直在忙搬家的事", joined)

    def test_guest_safe_profile_view_developer_focus_does_not_use_legacy_fallback(self):
        self.core.save_json_snapshot(
            {
                "schema_version": 2,
                "long_term": {
                    "developer_profile": {"known_preferences": ["认真", "会顾人"]},
                    "character_relationships": {"张三": {"familiarity": 7.2, "trust": 7.0}},
                },
            },
            source="unit_guest_legacy_only_seed",
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "我想了解开发者",
            query_bundle={"intent_bucket": "casual", "temporal_query_mode": "recent_general"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        guest_view = pack.get("guest_safe_profile_view", {}) if isinstance(pack.get("guest_safe_profile_view", {}), dict) else {}
        self.assertFalse(bool(guest_view.get("detected", False)))
        self.assertEqual(guest_view.get("source"), "")
        self.assertEqual(guest_view.get("read_policy"), "stable_only")
        self.assertFalse(bool(guest_view.get("legacy_fallback_used", True)))

    def test_guest_recent_stable_fact_is_not_misclassified_as_current_status(self):
        self.core.append_turn_event(
            "张三",
            "我想了解开发者",
            "他最近确实挺忙搬家的，不过一直很上心。",
            event_nodes=[],
            meta={"semantic_facts": ["开发者最近一直在忙搬家的事"]},
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "开发者最近很忙吗",
            query_bundle={"intent_bucket": "verify", "temporal_query_mode": "recent_general"},
            max_items=5,
            risk_bundle={"memory_read_scope": "guest"},
        )
        read_bundle = pack.get("memory_read_bundle", {}) if isinstance(pack.get("memory_read_bundle", {}), dict) else {}
        analysis = read_bundle.get("query_analysis", {}) if isinstance(read_bundle.get("query_analysis", {}), dict) else {}
        guest_view = pack.get("guest_safe_profile_view", {}) if isinstance(pack.get("guest_safe_profile_view", {}), dict) else {}
        plan = pack.get("memory_surface_plan", {}) if isinstance(pack.get("memory_surface_plan", {}), dict) else {}
        self.assertFalse(bool(analysis.get("current_status_query", False)))
        self.assertTrue(bool(analysis.get("stable_status_query", False)))
        self.assertTrue(guest_view.get("detected", False))
        self.assertNotEqual(guest_view.get("suppression_reason"), "current_status_requires_turn_evidence")
        self.assertEqual(plan.get("surface_mode"), "light_callback")

    def test_memory_schema_and_collective_recall(self):
        self.core.append_turn_event(
            "张三",
            "今天放学后要去练团，大家会在活动室碰头",
            "记下来了，放学后的练团很关键。",
            event_nodes=[
                {
                    "id": "group1",
                    "text": "放学后 测试组 在活动室集训",
                    "salience": 0.92,
                    "surprise": 0.45,
                    "actors": ["张三", "赵六", "李四", "王五", "钱七"],
                }
            ],
            meta={
                "semantic_facts": ["今天放学后 测试组 会在活动室集训"],
                "topic_tags": ["练团", "放学后", "活动室"],
                "situation_model": {
                    "school_phase": "after_school",
                    "session_phase": "pre_session",
                    "location": "活动室",
                    "group_status": "gathering",
                },
            },
            memory_snapshot={},
        )
        pack = self.core.build_memory_context_pack(
            "张三",
            "练团",
            query_bundle={
                "entities": ["张三", "活动室"],
                "time_anchor": "today",
                "intent_bucket": "planning",
                "topic_tags": ["练团", "放学后", "活动室"],
                "school_phase": "after_school",
                "session_phase": "pre_session",
                "location_anchor": "活动室",
            },
            max_items=5,
        )
        self.assertIn("cue_bundle", pack)
        self.assertIn("collective_rows", pack)
        self.assertTrue(pack.get("collective_rows"))

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT memory_role, engram_type, cue_json, lifecycle_json FROM memory_items WHERE character IN ('张三', '测试组') ORDER BY rowid DESC LIMIT 6;"
            ).fetchall()
        finally:
            conn.close()
        self.assertTrue(rows)
        parsed_any = False
        for row in rows:
            self.assertTrue(str(row["memory_role"] or "").strip())
            self.assertTrue(str(row["engram_type"] or "").strip())
            self.assertTrue(str(row["cue_json"] or "").strip().startswith("{"))
            self.assertTrue(str(row["lifecycle_json"] or "").strip().startswith("{"))
            parsed_any = True
        self.assertTrue(parsed_any)

    def test_retrieval_preserves_lifecycle_history(self):
        self.core.append_turn_event(
            "张三",
            "我对下周集训有点紧张",
            "我们可以提前整理一下节奏。",
            event_nodes=[{"id": "life1", "text": "用户对下周集训紧张", "salience": 0.77, "surprise": 0.48}],
            meta={"semantic_facts": ["用户对下周集训有些紧张"], "topic_tags": ["集训", "紧张"]},
            memory_snapshot={},
        )

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            before = conn.execute(
                "SELECT item_id, lifecycle_json, access_count FROM memory_items WHERE character='张三' AND text LIKE '%集训%' ORDER BY rowid DESC LIMIT 1;"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(before)
        ret = self.core.retrieve_memory_context("张三", "集训", query_bundle={"time_anchor": "future"}, max_items=4)
        self.assertTrue(ret.get("rows"))
        target_id = str(before["item_id"])

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            after = conn.execute(
                "SELECT lifecycle_json, access_count, last_accessed_at FROM memory_items WHERE item_id=? LIMIT 1;",
                (target_id,),
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(after)
        import json
        before_data = json.loads(str(before["lifecycle_json"] or "{}"))
        after_data = json.loads(str(after["lifecycle_json"] or "{}"))
        self.assertEqual(before_data.get("encoding_at"), after_data.get("encoding_at"))
        self.assertGreaterEqual(int(after["access_count"] or 0), int(before["access_count"] or 0) + 1)
        self.assertEqual(after_data.get("state"), "retrieval")
        self.assertTrue(str(after_data.get("last_retrieved_at", "") or "").strip())

    def test_canon_refresh_and_ooc_audit(self):
        sources = [
            {
                "source_type": "official_bang_dream_sources",
                "source_url": "https://example.com/zhangsan",
                "title": "张三 角色介绍",
                "snippet": "说话轻快，擅长带动气氛。",
                "claims": [
                    {"claim_type": "speech_style", "claim_text": "说话轻快，擅长带动气氛。", "confidence": 0.9},
                    {"claim_type": "group_behavior", "claim_text": "集训前会主动活络气氛。", "confidence": 0.88},
                ],
            },
            {
                "source_type": "moegirl_sources",
                "source_url": "https://example.org/zhangsan",
                "title": "萌娘百科 张三",
                "snippet": "有时会用比较明亮的社交语气。",
                "claims": [
                    {"claim_type": "interaction_style", "claim_text": "有时会用比较明亮的社交语气。", "confidence": 0.66},
                ],
            },
        ]
        ref = self.core.refresh_character_canon("张三", sources=sources, shadow=True)
        self.assertTrue(ref.get("ok"))
        profile = self.core.build_character_canon_profile("张三")
        self.assertTrue(profile.get("claims"))
        self.assertIn("speech_style", profile.get("anchors", {}))
        audit = self.core.audit_recent_ooc_against_canon(
            "张三",
            recent_replies=[{"reply_text": "关我什么事。懒得理。"}],
            shadow=True,
            run_id=str(ref.get("run_id", "")),
        )
        self.assertTrue(str(audit.get("audit_id", "")).startswith("ooc_"))
        self.assertGreater(float(audit.get("ooc_score", 0.0) or 0.0), 0.0)
        penalty = self.core.apply_ooc_penalty("张三", audit)
        self.assertTrue(penalty.get("ok"))
        self.assertTrue(penalty.get("shadow_only"))

    # ═══════════════════════════════════════════════════════════════════════
    # Phase Q.W2 / Option E Layer 1+2+3+4 tests (2026-05-04)
    # ═══════════════════════════════════════════════════════════════════════

    def test_layer1_features_chitchat_low_score(self):
        """用户原 8 条 chitchat 都应低分（< 0.45 candidate threshold）"""
        from tsuzuri.memory_core import _extract_memory_features, _compute_typed_score
        chitchat = [
            "我倒是挺喜欢玩二次元手游",
            "啊...对，不过我一般不怎么充钱",
            "很懂嘛，那你平时都玩些什么游戏呀？",
            "没想到你有时候也挺中二的",
            "嗯，我现在要去看书了，祝你们玩的开心呀，回见~",
            "话说回来你们吃过午饭了吗，水果三明治应该只是点心吧？",
        ]
        for txt in chitchat:
            feats = _extract_memory_features(txt, "嗯嗯")
            score = _compute_typed_score(feats)
            self.assertLess(score, 0.45, f"chitchat 应 < 0.45: {txt!r} got {score}")

    def test_layer1_features_ephemeral_strong_kill(self):
        """ephemeral 强档应直接命中 ephemeral_penalty=1.0"""
        from tsuzuri.memory_core import _extract_memory_features
        ephemeral_cases = [
            "晚安",
            "哈哈",
            "/refresh",
            "刚醒，正在洗漱",
            "嗯",
            "好的",
        ]
        for txt in ephemeral_cases:
            feats = _extract_memory_features(txt, "")
            self.assertGreaterEqual(feats["ephemeral_penalty"], 1.0,
                                    f"ephemeral 应 = 1.0: {txt!r}")

    def test_layer1_features_cold_start_fast_track(self):
        """cold_start 强信号身份/偏好应触发 fast-track 标记"""
        from tsuzuri.memory_core import _extract_memory_features, _is_cold_start_fast_track
        cold_cases = [
            "我叫阿澈，我家在杭州",
            "我从小就特别讨厌过生日",
            "我特别喜欢一个人写代码的感觉",
            "我最讨厌被打断思路",
            "我家在上海",
            "我30岁了",
        ]
        for txt in cold_cases:
            feats = _extract_memory_features(txt, "")
            self.assertTrue(_is_cold_start_fast_track(feats),
                            f"cold_start 应触发: {txt!r}")

    def test_layer1_features_user_fact_density(self):
        """user_fact_density 应在描述用户稳定属性时 > 0"""
        from tsuzuri.memory_core import _extract_memory_features
        cases = [
            ("我喜欢猫", True),
            ("我以前在大学参加过活动", True),
            ("我妈最近身体不好", True),
            ("天气不错", False),
            ("她说她也喜欢", False),  # third-person
            ("你说得对", False),
        ]
        for txt, should_hit in cases:
            feats = _extract_memory_features(txt, "")
            if should_hit:
                self.assertGreater(feats["user_fact_density"], 0.0,
                                   f"应有 user_fact: {txt!r}")
            else:
                self.assertEqual(feats["user_fact_density"], 0.0,
                                 f"不应有 user_fact: {txt!r}")

    def test_layer1_features_emotional_value_bound_vs_reactive(self):
        """value-bound 情绪应被识别、reactive 情绪应被排除"""
        from tsuzuri.memory_core import _extract_memory_features
        # value-bound (emotional + temporal/causal anchor)
        vb_cases = [
            "我从小就特别讨厌过生日，因为家里那时候不太宽裕",
            "一直很担心我妈的身体",
            "这些年我对这件事一直很在意",
        ]
        for txt in vb_cases:
            feats = _extract_memory_features(txt, "")
            self.assertGreater(feats["emotional_weight"], 0.0,
                               f"value-bound emotional 应 > 0: {txt!r}")
        # reactive (standalone emotion words)
        reactive_cases = [
            "好烦",
            "气死了",
            "笑死",
            "好难过",
        ]
        for txt in reactive_cases:
            feats = _extract_memory_features(txt, "")
            self.assertEqual(feats["emotional_weight"], 0.0,
                             f"reactive 不应 > 0: {txt!r}")

    def test_layer4_typed_gate_cold_start_fast_track(self):
        """cold_start = True 应直接绕过 reuse gate、根据 features 选 target"""
        from tsuzuri.memory_core import _typed_gate_decision
        # cold_start with high user_fact + stance → profile
        feats_profile = {
            "cold_start_fast_track": 1.0,
            "user_fact_density": 0.7,
            "stance_or_value": 0.5,
            "specificity": 0.3,
            "emotional_weight": 0.0,
        }
        target, engram, reason = _typed_gate_decision(feats_profile, 0.6, reuse_count=0)
        self.assertEqual(target, "semantic")
        self.assertEqual(engram, "user_profile_fact")
        self.assertIn("cold_start", reason)
        # cold_start with high specificity + emotional → key_event
        feats_event = {
            "cold_start_fast_track": 1.0,
            "user_fact_density": 0.3,
            "stance_or_value": 0.2,
            "specificity": 0.7,
            "emotional_weight": 0.5,
        }
        target, engram, _ = _typed_gate_decision(feats_event, 0.5, reuse_count=0)
        self.assertEqual(target, "episodic")
        self.assertEqual(engram, "key_event")

    def test_layer4_typed_gate_no_promotion_below_threshold(self):
        """score < 0.70 (no cold_start, no reuse) → no promotion"""
        from tsuzuri.memory_core import _typed_gate_decision
        feats = {
            "user_fact_density": 0.5,
            "specificity": 0.3,
            "stance_or_value": 0.4,
            "emotional_weight": 0.2,
            "cold_start_fast_track": 0.0,
        }
        target, engram, reason = _typed_gate_decision(feats, 0.55, reuse_count=0)
        self.assertIsNone(target)
        self.assertIn("not_eligible", reason)

    def test_layer4_typed_gate_promotion_with_reuse(self):
        """score >= 0.70 + reuse >= 1 → promotion"""
        from tsuzuri.memory_core import _typed_gate_decision
        feats = {
            "user_fact_density": 0.7,
            "stance_or_value": 0.5,
            "specificity": 0.3,
            "emotional_weight": 0.2,
            "cold_start_fast_track": 0.0,
        }
        target, engram, _ = _typed_gate_decision(feats, 0.72, reuse_count=1)
        self.assertEqual(target, "semantic")
        self.assertEqual(engram, "user_profile_fact")

    def test_layer4_typed_gate_fast_track_at_85(self):
        """score >= 0.85 → fast-track (bypass reuse gate)"""
        from tsuzuri.memory_core import _typed_gate_decision
        feats = {
            "specificity": 0.7,
            "emotional_weight": 0.5,
            "user_fact_density": 0.4,
            "stance_or_value": 0.3,
            "cold_start_fast_track": 0.0,
        }
        target, engram, reason = _typed_gate_decision(feats, 0.87, reuse_count=0)
        self.assertEqual(target, "episodic")
        self.assertEqual(engram, "key_event")
        self.assertIn("0.85", reason)

    def test_layer2_state_machine_aged_chitchat_enters_archive_candidate(self):
        """25h low-score working enters grace state instead of immediate archive."""
        item_id = "wke2_chitchat"
        # Insert with payload containing low-score features (simulating Layer 1 write)
        payload = {
            "memory_policy": {
                "memory_features": {
                    "explicit_memory_intent": 0.0,
                    "cold_start_fast_track": 0.0,
                    "specificity": 0.0,
                    "user_fact_density": 0.0,
                    "stance_or_value": 0.0,
                    "emotional_weight": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.0,
                "cold_start_fast_track": False,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:啊...对 | assistant:噗",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        candidate_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "set_consolidation_state"
            and a.get("target_state") == "archive_candidate"
            and item_id in (a.get("item_ids", []) or [])
        ]
        archive_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "archive" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertGreaterEqual(len(candidate_actions), 1)
        self.assertEqual(archive_actions, [])

    def test_layer2_state_machine_aged_ephemeral_archived(self):
        """aged working + ephemeral_penalty=1.0 → 强档 archive"""
        item_id = "wke2_eph"
        payload = {
            "memory_policy": {
                "memory_features": {
                    "ephemeral_penalty": 1.0,
                    "user_fact_density": 0.0,
                    "specificity": 0.0,
                    "stance_or_value": 0.0,
                    "emotional_weight": 0.0,
                    "explicit_memory_intent": 0.0,
                    "cold_start_fast_track": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.0,
                "cold_start_fast_track": False,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:晚安 | assistant:晚安",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        archive_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "archive" and item_id in (a.get("item_ids", []) or [])
            and "ephemeral" in str(a.get("reason", ""))
        ]
        self.assertGreaterEqual(len(archive_actions), 1, "ephemeral 应被强档 archive")

    def test_layer2_state_machine_candidate_state_set(self):
        """aged working + score 0.45-0.70 + reuse=0 → set_consolidation_state to candidate"""
        item_id = "wke2_candidate"
        payload = {
            "memory_policy": {
                "memory_features": {
                    "user_fact_density": 0.7,
                    "stance_or_value": 0.4,
                    "specificity": 0.3,
                    "emotional_weight": 0.0,
                    "explicit_memory_intent": 0.0,
                    "cold_start_fast_track": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.55,
                "cold_start_fast_track": False,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:我以前学过水彩 | assistant:哦哦",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        cand_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "set_consolidation_state"
            and item_id in (a.get("item_ids", []) or [])
            and a.get("target_state") == "candidate"
        ]
        self.assertGreaterEqual(len(cand_actions), 1, "score 0.45-0.70 + reuse=0 应设 candidate")

    def test_layer2_state_machine_promotes_with_reuse(self):
        """aged working + score >= 0.70 + reuse=1 → promote to typed target"""
        item_id = "wke2_promote_reuse"
        payload = {
            "memory_policy": {
                "memory_features": {
                    "user_fact_density": 0.8,
                    "stance_or_value": 0.6,
                    "specificity": 0.4,
                    "emotional_weight": 0.3,
                    "explicit_memory_intent": 1.0,
                    "cold_start_fast_track": 0.0,
                    "ephemeral_penalty": 0.0,
                    "generic_penalty": 0.0,
                },
                "typed_score": 0.75,
                "cold_start_fast_track": False,
                "consolidation_state": "candidate",
                "confirmed_reuse_count": 1,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:我从小学过水彩、特别喜欢民谣那种 | assistant:嗯",
            updated_at=self._hours_ago_iso(25),
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload,
        )
        evo = self.core.propose_memory_evolution("王五", max_actions=8)
        promote_actions = [
            a for a in evo.get("actions", [])
            if a.get("op") == "promote" and item_id in (a.get("item_ids", []) or [])
        ]
        self.assertGreaterEqual(len(promote_actions), 1, "score>=0.70+reuse=1 应 promote")
        self.assertIsNotNone(promote_actions[0].get("target_engram_type"),
                             "promote 应携带 target_engram_type")

    def test_layer3_record_reuse_signal_inserts_log_and_bumps_counter(self):
        """record_reuse_signal 插入 log + bump payload counter"""
        item_id = "wke3_reuse"
        # Insert working item with initial counter=0
        payload = {
            "memory_policy": {
                "memory_features": {},
                "typed_score": 0.5,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        self._insert_memory_item(
            item_id=item_id,
            character="王五",
            layer="working",
            text="user:西町有家好吃的 | assistant:嗯",
            engram_type="turn_exchange",
            item_type="turn",
            payload=payload,
        )
        ok = self.core.record_reuse_signal(
            item_id, "王五", "confirmed_reuse",
            confidence=0.7, evidence="entity_overlap=['西町']",
        )
        self.assertTrue(ok)
        # Verify counter bumped
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute(
                "SELECT payload_json FROM memory_items WHERE item_id=?", (item_id,)
            ).fetchone()
            payload_after = json.loads(row[0] or "{}")
            self.assertEqual(
                payload_after.get("memory_policy", {}).get("confirmed_reuse_count"),
                1, "confirmed_reuse_count 应 bump 到 1"
            )
            # Verify log row
            count = conn.execute(
                "SELECT COUNT(*) FROM reuse_signal_log WHERE working_item_id=? AND signal_type=?",
                (item_id, "confirmed_reuse")
            ).fetchone()[0]
            self.assertEqual(count, 1, "log 应有 1 条")
        finally:
            conn.close()

    def test_layer3_entity_extraction(self):
        """_extract_entities_from_text 应识别 TIME + 形态后缀地点 + family/food

        2026-05-09 重构后：PROPER 已退役（品牌/作品/canon 名号无穷无尽不可维护）、
        改靠 embedding_novelty 信号识别。本测试只断言保留的 entity class：
          - TIME（年份/日期等）
          - PLACE 形态后缀（…寺/…大学/…神社）
          - 极少数高频锚点（临川/北京/上海等）
          - family / food（Layer 3 专属）
        具体品牌/作品/测试组 canon 名号（西町/活动中心/测试组 等）不再硬编码识别。
        """
        from tsuzuri.memory_core import _extract_entities_from_text
        # TIME 仍识别（闭集）
        time_text = "我昨天去演出，2025 年了真快"
        time_entities = _extract_entities_from_text(time_text)
        self.assertIn("2025年", time_entities)
        # 形态后缀地点仍识别
        morph_text = "我去了浅草寺，又顺便逛了临川大学附近"
        morph_entities = _extract_entities_from_text(morph_text)
        # "浅草寺" / "临川大学" 至少一个被识别（形态后缀 catch-all）
        self.assertTrue(
            any("寺" in e or "大学" in e for e in morph_entities),
            f"形态后缀地点应识别: {morph_entities}",
        )
        # 极少数高频锚点
        anchor_text = "我从临川回到了北京"
        anchor_entities = _extract_entities_from_text(anchor_text)
        self.assertIn("临川", anchor_entities)
        self.assertIn("北京", anchor_entities)

    def test_layer3_negation_detection(self):
        """_detect_negation 应识别否认 patterns"""
        from tsuzuri.memory_core import _detect_negation
        self.assertTrue(_detect_negation("不是的"))
        self.assertTrue(_detect_negation("我没说过"))
        self.assertTrue(_detect_negation("你搞错了"))
        self.assertFalse(_detect_negation("你说得对"))
        self.assertFalse(_detect_negation("嗯嗯"))


class CanonExtractionShapeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    pass  # upstream-only test removed

    def test_evolution_rollback_idempotent(self):
        self.core.append_turn_event(
            "张三",
            "我在准备简历",
            "可以一起梳理重点。",
            event_nodes=[{"id": "rb1", "text": "用户在准备简历", "salience": 0.7, "surprise": 0.4}],
            meta={"semantic_facts": ["用户在准备简历"]},
            memory_snapshot={},
        )
        ret = self.core.retrieve_memory_context("张三", "简历", max_items=3)
        rows = ret.get("rows", []) if isinstance(ret, dict) else []
        self.assertTrue(rows)
        target_id = str(rows[0].get("item_id", ""))
        self.assertTrue(target_id)

        apply1 = self.core.apply_evolution_plan(
            run_id="evo_manual_rollback",
            plan=[{"op": "rewrite", "item_ids": [target_id], "rewrite_text": "简历准备中(重写)"}],
            shadow_only=False,
        )
        self.assertTrue(apply1.get("ok"))

        rb1 = self.core.rollback_evolution_run("evo_manual_rollback")
        self.assertTrue(rb1.get("ok"))
        self.assertGreaterEqual(int(rb1.get("reverted", 0) or 0), 1)

        rb2 = self.core.rollback_evolution_run("evo_manual_rollback")
        self.assertTrue(rb2.get("ok"))
        self.assertGreaterEqual(int(rb2.get("reverted", 0) or 0), 0)

        apply2 = self.core.apply_evolution_plan(
            run_id="evo_manual_rollback",
            plan=[{"op": "rewrite", "item_ids": [target_id], "rewrite_text": "简历准备中(再次重写)"}],
            shadow_only=False,
        )
        self.assertTrue(apply2.get("ok"))

    def test_proactive_policy_llm_paths(self):
        gate_ok = self.core.evaluate_proactive_policy(
            kind="idle_followup",
            idle_sec=900,
            session_key="u:ok",
            base_risk=0.2,
            dedupe_key="u:ok|idle_followup|abc",
            hard_gate={"allow": True, "risk_score": 0.2, "reasons": []},
            llm_policy=lambda payload: {"allow": False, "risk_score": 0.8, "suggested_delay_sec": 600, "reasons": ["llm says no"]},
        )
        self.assertFalse(bool(gate_ok.get("allow", True)))
        self.assertIn("llm_policy_block", gate_ok.get("reasons", []))

        def _raise_timeout(_payload):
            raise TimeoutError("timeout")

        gate_timeout = self.core.evaluate_proactive_policy(
            kind="idle_followup",
            idle_sec=600,
            session_key="u:err",
            base_risk=0.2,
            dedupe_key="u:err|idle_followup|abc",
            hard_gate={"allow": True, "risk_score": 0.2, "reasons": []},
            llm_policy=_raise_timeout,
        )
        self.assertFalse(bool(gate_timeout.get("allow", True)))

        gate_invalid = self.core.evaluate_proactive_policy(
            kind="idle_followup",
            idle_sec=600,
            session_key="u:invalid",
            base_risk=0.2,
            dedupe_key="u:invalid|idle_followup|abc",
            hard_gate={"allow": True, "risk_score": 0.2, "reasons": []},
            llm_policy=lambda payload: "not_json",
        )
        self.assertFalse(bool(gate_invalid.get("allow", True)))

    def test_cluster_summarizer_rejects_character_subject_for_user_apology(self):
        member_items = [
            {"text": "阿澈说：张三，我回来了，抱歉让你等了这么久......有没有想我呀？"},
            {"text": "阿澈说：好啊，下次我等着哦，你开心比什么都重要啦~"},
            {"text": "阿澈说：嗯......我知道了，谢谢你们愿意等我，能再见到大家真好"},
            {"text": "阿澈说：下次吧，谢谢张三"},
        ]

        class _FakeMessage:
            content = "张三因系统故障失联道歉认错"

        class _FakeChoice:
            message = _FakeMessage()

        class _FakeResponse:
            choices = [_FakeChoice()]

        class _FakeCompletions:
            @staticmethod
            def create(**_kwargs):
                return _FakeResponse()

        class _FakeChat:
            completions = _FakeCompletions()

        class _FakeOpenAI:
            def __init__(self, **_kwargs):
                self.chat = _FakeChat()

        with patch.dict(os.environ, {"LLM_API_KEY": "unit-key"}, clear=False), \
                patch("openai.OpenAI", _FakeOpenAI):
            summary = self.core._default_cluster_summarizer(member_items, "张三")

        self.assertNotIn("张三因", summary)
        self.assertTrue(summary.startswith("阿澈"), summary)

    def test_cluster_summarizer_failure_skips_instead_of_placeholder(self):
        # 2026-10-19 审计 LIFE-016：无 key / LLM 空返回 / 调用异常都返回空串（caller 跳过该簇），
        # 不再把「[聚类 N 条相似事] + 首条原话」当 semantic 摘要写库。
        member_items = [{"text": "阿澈说：周末想去看海"}, {"text": "阿澈说：海边的风很舒服"}]

        class _EmptyMessage:
            content = ""

        class _EmptyChoice:
            message = _EmptyMessage()

        class _EmptyResponse:
            choices = [_EmptyChoice()]

        class _Completions:
            raise_error = False

            @classmethod
            def create(cls, **_kwargs):
                if cls.raise_error:
                    raise RuntimeError("upstream 500")
                return _EmptyResponse()

        class _Chat:
            completions = _Completions

        class _FakeOpenAI:
            def __init__(self, **_kwargs):
                self.chat = _Chat()

        with patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            self.assertEqual(self.core._default_cluster_summarizer(member_items, "张三"), "")
        with patch.dict(os.environ, {"LLM_API_KEY": "unit-key"}, clear=False),                 patch("openai.OpenAI", _FakeOpenAI):
            self.assertEqual(self.core._default_cluster_summarizer(member_items, "张三"), "")
            _Completions.raise_error = True
            self.assertEqual(self.core._default_cluster_summarizer(member_items, "张三"), "")

    def test_prune_stale_shared_info_marks_undelivered_old_rows(self):
        # 2026-08-27 §9-1：超 48h 窗口仍零投递的 shared_info 行置 expired=1；
        # 新鲜行与已投递行不受影响。
        now_ts = datetime.now(timezone.utc).timestamp()
        self.core.ensure_shared_info_schema()
        self.core.store_shared_info(
            info_id="si-old-undelivered", source_character="王五",
            original_text="阿澈说他明天要出远门", created_at=now_ts - 100 * 3600,
        )
        self.core.store_shared_info(
            info_id="si-fresh-undelivered", source_character="王五",
            original_text="阿澈说他今天去看了天文馆", created_at=now_ts - 10 * 3600,
        )
        self.core.store_shared_info(
            info_id="si-old-delivered", source_character="李四",
            original_text="李四说她去看星星了", created_at=now_ts - 100 * 3600,
        )
        self.core.mark_shared_info_delivered("si-old-delivered", "张三", delivered_at=now_ts - 90 * 3600)

        pruned = self.core.prune_stale_shared_info(max_age_hours=48)
        self.assertEqual(pruned, 1)

        def _expired(iid: str) -> int:
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT expired FROM shared_info WHERE info_id=?;", (iid,),
                ).fetchone()
            finally:
                conn.close()
            return int(row[0]) if row else -1

        self.assertEqual(_expired("si-old-undelivered"), 1)
        self.assertEqual(_expired("si-fresh-undelivered"), 0)
        self.assertEqual(_expired("si-old-delivered"), 0)


if __name__ == "__main__":
    unittest.main()
