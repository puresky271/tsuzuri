from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest import mock

import numpy as np

from tsuzuri.memory_core import (
    _apply_memory_correction_dominance,
    _memory_explicit_row_is_answer_bearing,
    _explicit_recall_answer_budget,
    _memory_row_can_surface_nonexplicit,
    _memory_row_is_working,
    _memory_should_run_expensive_reranker,
)
from tsuzuri.memory_provider_hybrid import HybridIndexedMemoryProvider, HybridIndexedMemoryStore


def _embedding_text(values: list[float]) -> str:
    return base64.b64encode(np.asarray(values, dtype=np.float32).tobytes()).decode("ascii")


class HybridMemoryProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db_path = str(root / "memory.db")
        self.snapshot_path = str(root / "snapshot.json")
        self.store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=self._dense_encoder,
            sparse_query_encoder=lambda text: None,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_adaptive_explicit_recall_budget(self) -> None:
        self.assertEqual(1, _explicit_recall_answer_budget("你还记得我的生日是几月几号吗？"))
        self.assertEqual(3, _explicit_recall_answer_budget("你还记得我周二开过什么玩笑吗？"))
        self.assertEqual(4, _explicit_recall_answer_budget("你还记得我之前买过哪些盆栽吗？"))

    def test_open_ended_bounded_slot_accepts_aligned_specific_evidence(self) -> None:
        row = {
            "text": "阿澈和张三确认了周二阿澈开过的两个玩笑：假装失忆让张三重新自我介绍，以及装迷路小孩被张三识破。",
            "payload_json": "{}",
            "score": 0.43,
            "focus_relevance": 0.12,
            "focus_keyword_score": 0.50,
            "embedding_score": 0.74,
        }
        self.assertTrue(_memory_explicit_row_is_answer_bearing(row, "你还记得我周二开过什么玩笑吗？"))

    def test_recall_focus_strips_conversational_fillers_and_character_name(self) -> None:
        from tsuzuri.memory_core import _recall_focus_text

        self.assertEqual("周二开过 玩笑", _recall_focus_text("对了，张三你还记得我周二开过什么玩笑嘛？"))

    @staticmethod
    def _dense_encoder(text: str):
        query = str(text or "")
        if "老家" in query:
            return np.asarray([1.0, 0.0], dtype=np.float32)
        if "生日" in query:
            return np.asarray([0.8, 0.2], dtype=np.float32)
        return np.asarray([0.9, 0.1], dtype=np.float32)

    def _insert(
        self,
        item_id: str,
        text: str,
        *,
        character: str = "王五",
        layer: str = "episodic",
        payload: dict | None = None,
        updated_at: str = "2026-07-24T12:00:00Z",
        embedding: list[float] | None = None,
        sparse: dict | None = None,
        archived: int = 0,
        status: str = "active",
        confidence: float = 0.82,
        importance: float = 0.8,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty,
                 importance, status, source, evidence_id, created_at, updated_at,
                 last_accessed_at, access_count, archived, embedding_text,
                 embedding_sparse_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '{}', '{}', 0.7, ?, 0.3, ?,
                        ?, 'test', ?, ?, ?, '', 0, ?, ?, ?)
                """,
                (
                    item_id,
                    character,
                    layer,
                    layer,
                    "fact" if layer == "semantic" else "event",
                    "fact" if layer == "semantic" else "event",
                    text,
                    json.dumps(payload or {}, ensure_ascii=False),
                    confidence,
                    importance,
                    status,
                    f"evidence:{item_id}",
                    updated_at,
                    updated_at,
                    archived,
                    _embedding_text(embedding) if embedding is not None else "",
                    json.dumps(sparse or {}, separators=(",", ":")),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def test_full_store_structured_lane_recovers_fact_outside_recent_window(self) -> None:
        self._insert(
            "birthday",
            "阿澈的重要日期是3月14日。",
            layer="semantic",
            payload={"about": "developer", "memory_policy": {"manual_promote": True}},
            updated_at="2026-01-01T00:00:00Z",
            embedding=[0.8, 0.2],
        )
        conn = sqlite3.connect(self.db_path)
        try:
            rows = []
            for index in range(805):
                ts = f"2026-07-{1 + (index // 40):02d}T{index % 24:02d}:00:00Z"
                rows.append(
                    (
                        f"recent-{index}", "王五", "episodic", "episodic", "event", "event",
                        f"近期普通对话记录第{index}条。", "{}", "{}", "{}", 0.6, 0.7,
                        0.3, 0.6, "active", "test", f"ev:{index}", ts, ts, "", 0, 0, "", "",
                    )
                )
            conn.executemany(
                """INSERT INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty,
                 importance, status, source, evidence_id, created_at, updated_at,
                 last_accessed_at, access_count, archived, embedding_text, embedding_sparse_text)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

        rows = self.store._query_items("王五", limit=800, query_text="你还记得我是几月几号出生的吗？")
        birthday = next(row for row in rows if row["item_id"] == "birthday")
        self.assertTrue(birthday.get("_hybrid_structured_hit"))
        self.assertTrue(birthday.get("_skip_embedding_backfill"))
        self.assertGreater(len(rows), 800)

    def test_sparse_lane_requires_threshold_and_preserves_character_scope(self) -> None:
        sparse_store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=lambda text: None,
            sparse_query_encoder=lambda text: {7: 1.0},
        )
        self._insert("sparse-hit", "旧的目标事实", sparse={7: 1.3}, updated_at="2026-01-01T00:00:00Z")
        self._insert("sparse-weak", "弱稀疏噪声", sparse={7: 1.1}, updated_at="2026-01-01T00:00:00Z")
        self._insert("other-character", "跨角色秘密", character="赵六", sparse={7: 9.0})
        self._insert("archived", "已归档秘密", sparse={7: 9.0}, archived=1)

        rows = sparse_store._query_items("王五", limit=20, query_text="目标事实是什么")
        by_id = {row["item_id"]: row for row in rows}
        self.assertTrue(by_id["sparse-hit"].get("_hybrid_sparse_hit"))
        self.assertFalse(by_id["sparse-weak"].get("_hybrid_sparse_hit", False))
        self.assertNotIn("other-character", by_id)
        self.assertNotIn("archived", by_id)

    def test_generic_explicit_query_does_not_force_unrelated_structured_answer(self) -> None:
        store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=lambda text: None,
            sparse_query_encoder=lambda text: None,
        )
        self._insert(
            "unrelated-birthday",
            "阿澈的生日是3月14日。",
            layer="semantic",
            payload={"about": "developer", "memory_policy": {"manual_promote": True}},
        )
        rows = store._query_items("王五", limit=20, query_text="你还记得上次那件事吗？")
        birthday = next(row for row in rows if row["item_id"] == "unrelated-birthday")
        self.assertFalse(birthday.get("_hybrid_structured_hit", False))

    def test_recent_base_row_keeps_embedding_backfill_contract(self) -> None:
        self._insert("recent-missing-embedding", "最近提过的目标事实。", embedding=None)
        rows = self.store._query_items("王五", limit=20, query_text="目标事实")
        target = next(row for row in rows if row["item_id"] == "recent-missing-embedding")
        self.assertFalse(target.get("_skip_embedding_backfill", False))

    def test_failed_query_encoding_is_not_negative_cached(self) -> None:
        calls = 0

        def encoder(text: str):
            nonlocal calls
            calls += 1
            return None if calls == 1 else np.asarray([1.0, 0.0], dtype=np.float32)

        store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=encoder,
            sparse_query_encoder=lambda text: None,
        )
        first, _ = store._encode_memory_query_features("同一个查询")
        second, _ = store._encode_memory_query_features("同一个查询")
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(2, calls)

    def test_low_importance_correction_reaches_structured_lane(self) -> None:
        store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=lambda text: None,
            sparse_query_encoder=lambda text: None,
        )
        self._insert(
            "low-importance-correction",
            "阿澈明确纠正：之前关于谈恋爱的说法是乌龙，并非真实意图。",
            payload={"canonical_key": "user:relationship:intent", "evidence_role": "correction"},
            importance=0.10,
        )
        rows = store._query_items("王五", limit=20, query_text="我之前到底有没有打算谈恋爱？")
        correction = next(row for row in rows if row["item_id"] == "low-importance-correction")
        self.assertTrue(correction.get("_hybrid_structured_hit", False))

    def test_legacy_correction_suppresses_only_older_premise(self) -> None:
        premise = {
            "item_id": "premise",
            "text": "王五询问阿澈之前想谈恋爱一事是否有进展。",
            "payload_json": json.dumps({"ts": "2026-07-24T14:32:40Z"}),
        }
        correction = {
            "item_id": "correction",
            "text": "阿澈向王五澄清，之前关于谈恋爱的说法是乌龙，并非真实意图。",
            "payload_json": json.dumps({"ts": "2026-07-24T14:37:51Z"}),
        }
        later_fact = {
            "item_id": "later",
            "text": "阿澈后来表示现在有谈恋爱的打算。",
            "payload_json": json.dumps({"ts": "2026-07-25T10:00:00Z"}),
        }
        resolved = _apply_memory_correction_dominance([premise, correction, later_fact])
        self.assertEqual(["correction", "later"], [row["item_id"] for row in resolved])

    def test_correction_dominance_compares_mixed_timezone_instants(self) -> None:
        correction = {
            "item_id": "correction",
            "text": "阿澈澄清：谈恋爱的说法是乌龙，并非真实意图。",
            "payload_json": json.dumps({"ts": "2026-07-24T15:30:00Z"}),
        }
        actually_older = {
            "item_id": "older-plus-nine",
            "text": "阿澈之前提到想谈恋爱。",
            "payload_json": json.dumps({"ts": "2026-07-25T00:20:00+09:00"}),
        }
        actually_newer = {
            "item_id": "newer-plus-nine",
            "text": "阿澈后来表示现在有谈恋爱的打算。",
            "payload_json": json.dumps({"ts": "2026-07-25T00:40:00+09:00"}),
        }
        resolved = _apply_memory_correction_dominance([actually_older, correction, actually_newer])
        self.assertEqual(["correction", "newer-plus-nine"], [row["item_id"] for row in resolved])

    def test_explicit_answer_gate_accepts_verified_raw_user_hometown(self) -> None:
        target = {
            "text": "user: 我老家在海边，小时候最喜欢海。",
            "payload_json": json.dumps(
                {"user_text": "我老家在海边，小时候最喜欢海。", "source_trust": {"user_text": "verified"}},
                ensure_ascii=False,
            ),
            "embedding_score": 0.8,
        }
        distractor = {
            "text": "阿澈暑假回老家吃饭和休息。",
            "payload_json": json.dumps({"about": "developer"}),
            "embedding_score": 0.8,
        }
        query = "你还记得我小时候最喜欢老家的什么地方吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(distractor, query))

    def test_birthday_gate_rejects_another_person_date(self) -> None:
        target = {"text": "阿澈的生日是3月14日。", "payload_json": "{}", "embedding_score": 0.8}
        distractor = {"text": "阿澈提过赵六的生日是5月27日。", "payload_json": "{}", "embedding_score": 0.8}
        query = "你还记得我的生日是几月几号吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(distractor, query))

    def test_first_person_fact_gate_rejects_character_profile_high_embedding(self) -> None:
        # 2026-09-14 郁金香失败案例：rolequestion 问"我最喜欢的花"时，
        # 角色自己的档案事实靠 0.72-0.76 的 embedding 通过了 open-ended 路径、
        # 挤掉了真正 developer-grounded 的答案。
        target = {
            "text": "阿澈最喜欢粉色的郁金香。",
            "payload_json": "{}",
            "score": 0.76,
            "embedding_score": 0.76,
        }
        profile_row = {
            "text": "张三学习成绩不错。",
            "payload_json": "{}",
            "score": 0.76,
            "embedding_score": 0.72,
        }
        birthday_row = {
            "text": "张三的生日是9月8日。",
            "payload_json": "{}",
            "score": 0.76,
            "embedding_score": 0.72,
        }
        query = "张三，你还记得我最喜欢的花是哪一种吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(profile_row, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(birthday_row, query))

    def test_expression_gate_accepts_classifier_variant(self) -> None:
        row = {
            "text": "阿澈说王五平时一直板着张脸。",
            "payload_json": "{}",
            "_hybrid_structured_hit": True,
        }
        self.assertTrue(
            _memory_explicit_row_is_answer_bearing(row, "我之前总说你平时是什么表情？")
        )

    def test_expression_gate_accepts_profile_promoted_raw_user_answer(self) -> None:
        row = {
            "layer": "semantic",
            "source": "turn_append",
            "text": (
                "user:你还记得我说过，你大部分时间都板着张脸吗？ | "
                "assistant:记得。我表情本来就这样。"
            ),
            "payload_json": json.dumps(
                {"memory_policy": {"consolidation_state": "profile_promoted"}},
                ensure_ascii=False,
            ),
            "_hybrid_structured_hit": True,
        }
        self.assertTrue(
            _memory_explicit_row_is_answer_bearing(row, "我之前总说你平时是什么表情？")
        )

    def test_expression_gate_rejects_unpromoted_probe_turn(self) -> None:
        row = {
            "layer": "working",
            "text": "user:你还记得我说过，你大部分时间都板着张脸吗？ | assistant:记得。",
            "payload_json": "{}",
            "working_direct_recall_lane": True,
        }
        self.assertFalse(
            _memory_explicit_row_is_answer_bearing(row, "我之前总说你平时是什么表情？")
        )

    def test_unknown_explicit_slot_fails_closed(self) -> None:
        row = {
            "text": "阿澈那好吧，我喜欢你。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        self.assertFalse(
            _memory_explicit_row_is_answer_bearing(row, "你还记得上次那件事具体是什么吗？")
        )

    def test_flavor_slot_requires_the_asked_item(self) -> None:
        target = {
            "text": "阿澈喜欢巧克力口味的夹心冰棍。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        distractor = {
            "text": "阿澈那好吧，我喜欢你。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        query = "你还记得我喜欢什么口味的夹心冰棍吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(distractor, query))

    def test_compound_drink_slot_requires_positive_and_negative_preferences(self) -> None:
        target = {
            "text": "阿澈喝奶茶喜欢加芋圆和椰果，不爱珍珠。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        incomplete = {
            "text": "阿澈喝奶茶喜欢加芋圆。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        query = "你还记得我喝奶茶喜欢加什么、不喜欢什么吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(incomplete, query))

    def test_compound_drink_slot_accepts_digest_ellipsis(self) -> None:
        target = {
            "text": (
                "我和赵六、李四、王五等提到：阿澈说暑假回老家吃好吃的、"
                "喜欢芋圆椰果不爱珍珠；我推荐西町黑糖珍珠奶茶。"
            ),
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        missing_positive_additive = {
            "text": "阿澈喜欢奶茶，但不爱珍珠。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        query = "你还记得我喝奶茶喜欢加什么、不喜欢什么吗？"
        self.assertTrue(_memory_explicit_row_is_answer_bearing(target, query))
        self.assertFalse(_memory_explicit_row_is_answer_bearing(missing_positive_additive, query))

    def test_reranker_skips_single_plausible_candidate(self) -> None:
        rows = [
            {"heuristic_score": 0.6, "embedding_score": 0.8},
            {"heuristic_score": 0.4, "embedding_score": 0.2},
            {"heuristic_score": 0.3, "embedding_score": 0.1},
            {"heuristic_score": 0.2, "embedding_score": 0.1},
        ]
        self.assertFalse(
            _memory_should_run_expensive_reranker(rows, {"explicit_recall_intent": True, "fact_like_query": True})
        )

    def test_reranker_runs_for_two_close_plausible_candidates(self) -> None:
        rows = [
            {
                "text": "阿澈的生日是3月14日。",
                "payload_json": json.dumps({"about": "developer"}),
                "heuristic_score": 0.61,
                "embedding_score": 0.70,
            },
            {
                "text": "阿澈的生日是3月15日。",
                "payload_json": json.dumps({"about": "developer"}),
                "heuristic_score": 0.56,
                "embedding_score": 0.68,
            },
            {"text": "生日礼物。", "heuristic_score": 0.31, "embedding_score": 0.20},
            {"text": "生日祝福。", "heuristic_score": 0.25, "embedding_score": 0.18},
        ]
        self.assertTrue(
            _memory_should_run_expensive_reranker(
                rows,
                {
                    "query_text": "你还记得我的生日是几月几号吗？",
                    "explicit_recall_intent": True,
                    "fact_like_query": True,
                },
            )
        )

    def test_promoted_semantic_marker_overrides_stale_working_metadata(self) -> None:
        row = {
            "layer": "semantic",
            "memory_role": "working",
            "_hybrid_promoted_layer": True,
            "payload_json": json.dumps(
                {
                    "memory_meta": {
                        "memory_type": "working_turn",
                        "source_kind": "turn_exchange",
                        "owner_scope": "character:张三",
                        "mode_scope": "guest",
                        "evidence_tier": "high",
                        "temporal_validity": "transient",
                        "surface_default": "silent_grounding",
                    }
                }
            ),
        }
        self.assertFalse(_memory_row_is_working(row))
        decorated = self.store._decorate_memory_row(row)
        self.assertEqual("shared_fact", decorated["memory_type"])
        self.assertEqual("character:张三", decorated["owner_scope"])
        self.assertEqual("guest", decorated["mode_scope"])
        self.assertEqual("high", decorated["evidence_tier"])

    def test_structured_answer_skips_ambiguous_reranker(self) -> None:
        rows = [
            {
                "text": "阿澈的生日是3月14日。",
                "payload_json": json.dumps({"about": "developer"}),
                "heuristic_score": 0.62,
                "embedding_score": 0.8,
                "_hybrid_structured_hit": True,
            },
            {"text": "赵六的生日是5月27日。", "heuristic_score": 0.6, "embedding_score": 0.78},
            {"text": "生日祝福", "heuristic_score": 0.4, "embedding_score": 0.6},
            {"text": "生日礼物", "heuristic_score": 0.3, "embedding_score": 0.55},
        ]
        analysis = {
            "query_text": "你还记得我的生日是几月几号吗？",
            "explicit_recall_intent": True,
            "fact_like_query": True,
        }
        self.assertFalse(_memory_should_run_expensive_reranker(rows, analysis))

    def test_single_answer_bearing_candidate_skips_reranker(self) -> None:
        rows = [
            {
                "text": "阿澈在乎你们的感受，希望你们过好自己的日子。",
                "payload_json": json.dumps({"about": "developer"}),
                "heuristic_score": 0.64,
                "embedding_score": 0.82,
                "_hybrid_structured_hit": True,
            },
            {
                "text": "阿澈希望张三成为大明星后给自己留LIVE门票。",
                "payload_json": json.dumps({"about": "developer"}),
                "heuristic_score": 0.60,
                "embedding_score": 0.79,
                "_hybrid_structured_hit": True,
            },
            {"text": "阿澈喜欢音乐。", "heuristic_score": 0.31, "embedding_score": 0.56},
            {"text": "阿澈最近很忙。", "heuristic_score": 0.28, "embedding_score": 0.53},
        ]
        analysis = {
            "query_text": "你成为大明星以后，我希望你给我留什么？",
            "explicit_recall_intent": False,
            "fact_like_query": True,
        }
        self.assertFalse(_memory_should_run_expensive_reranker(rows, analysis))

    def test_nonexplicit_fact_surface_keeps_only_answer_bearing_rows(self) -> None:
        analysis = {
            "query_text": "你成为大明星以后，我希望你给我留什么？",
            "explicit_recall_intent": False,
            "fact_like_query": True,
        }
        target = {
            "text": "阿澈希望张三成为大明星后给自己留LIVE门票。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        distractor = {
            "text": "阿澈在乎你们的感受，希望你们过好自己的日子。",
            "payload_json": json.dumps({"about": "developer"}),
            "_hybrid_structured_hit": True,
        }
        self.assertTrue(_memory_row_can_surface_nonexplicit(target, analysis))
        self.assertFalse(_memory_row_can_surface_nonexplicit(distractor, analysis))

    def test_retrieval_invokes_reranker_when_gate_is_eligible(self) -> None:
        for index, text in enumerate(
            (
                "阿澈在乎你们的感受，希望你们过好自己的日子。",
                "阿澈希望张三成为大明星后给自己留LIVE门票。",
                "阿澈喜欢音乐。",
                "阿澈最近很忙。",
            )
        ):
            self._insert(f"rerank-{index}", text, layer="semantic")
        with (
            mock.patch.dict(os.environ, {"USE_CROSS_ENCODER_RERANK": "1", "MEM_LLM_SELECTOR_ENABLE": "1"}),
            mock.patch("tsuzuri.memory_core._memory_should_run_expensive_reranker", return_value=True),
            mock.patch("tsuzuri.memory_core._embedding_callable", return_value=False),
            mock.patch("tsuzuri.reranker_engine.is_reranker_enabled", return_value=True),
            mock.patch("tsuzuri.reranker_engine.is_reranker_ready", return_value=True),
            mock.patch("tsuzuri.reranker_engine.get_top_k", return_value=8),
            mock.patch(
                "tsuzuri.reranker_engine.rerank_pairs",
                side_effect=lambda query, docs: [0.9] * len(docs),
            ) as rerank_pairs,
        ):
            result = self.store.retrieve_memory_context(
                "王五",
                "你成为大明星以后，我希望你给我留什么？",
                query_bundle={"intent_bucket": "verify"},
                max_items=4,
                touch_selected=False,
                log_retrieval=False,
            )
        self.assertEqual(1, rerank_pairs.call_count)
        self.assertTrue(result.get("reranker_used"))

    def test_retrieval_never_cold_loads_reranker_on_reply_thread(self) -> None:
        for index, text in enumerate(("候选事实甲。", "候选事实乙。", "候选事实丙。", "候选事实丁。")):
            self._insert(f"cold-rerank-{index}", text, layer="semantic")
        with (
            mock.patch.dict(os.environ, {"USE_CROSS_ENCODER_RERANK": "1", "MEM_LLM_SELECTOR_ENABLE": "0"}),
            mock.patch("tsuzuri.memory_core._memory_should_run_expensive_reranker", return_value=True),
            mock.patch("tsuzuri.memory_core._embedding_callable", return_value=False),
            mock.patch("tsuzuri.reranker_engine.is_reranker_enabled", return_value=True) as is_enabled,
            mock.patch("tsuzuri.reranker_engine.is_reranker_ready", return_value=False),
            mock.patch("tsuzuri.reranker_engine.rerank_pairs") as rerank_pairs,
        ):
            result = self.store.retrieve_memory_context(
                "王五",
                "两个候选事实哪一个更准确？",
                query_bundle={"intent_bucket": "verify"},
                max_items=4,
                touch_selected=False,
                log_retrieval=False,
            )
        rerank_pairs.assert_not_called()
        is_enabled.assert_not_called()
        self.assertFalse(result.get("reranker_used"))

    def test_promoted_semantic_preserves_existing_scope_metadata(self) -> None:
        row = {
            "layer": "semantic",
            "memory_role": "working",
            "_hybrid_promoted_layer": True,
            "payload_json": json.dumps(
                {
                    "memory_meta": {
                        "memory_type": "working_turn",
                        "source_kind": "turn_exchange",
                        "owner_scope": "character:张三",
                        "mode_scope": "guest",
                        "evidence_tier": "high",
                    }
                },
                ensure_ascii=False,
            ),
        }
        decorated = self.store._decorate_memory_row(row)
        self.assertEqual("character:张三", decorated["owner_scope"])
        self.assertEqual("guest", decorated["mode_scope"])
        self.assertEqual("high", decorated["evidence_tier"])

    def test_base_window_row_keeps_embedding_self_heal_enabled(self) -> None:
        self._insert("base-row", "阿澈希望张三给自己留LIVE门票。", layer="semantic")
        with mock.patch("tsuzuri.memory_core._embedding_callable", return_value=False):
            rows = self.store._query_items("王五", limit=20, query_text="LIVE门票")
        row = next(item for item in rows if item["item_id"] == "base-row")
        self.assertFalse(row.get("_skip_embedding_backfill", False))

    def test_failed_query_encoding_is_not_negatively_cached(self) -> None:
        calls = 0

        def flaky_encoder(text: str):
            nonlocal calls
            calls += 1
            return None if calls == 1 else np.asarray([0.9, 0.1], dtype=np.float32)

        store = HybridIndexedMemoryStore(
            db_path=self.db_path,
            snapshot_path=self.snapshot_path,
            dense_query_encoder=flaky_encoder,
            sparse_query_encoder=lambda text: None,
        )
        first, _ = store._encode_memory_query_features("相同查询")
        second, _ = store._encode_memory_query_features("相同查询")
        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(2, calls)

    def test_provider_is_lazy_and_registry_default_is_reversible(self) -> None:
        provider = HybridIndexedMemoryProvider(db_path=self.db_path, snapshot_path=self.snapshot_path)
        self.assertIsNone(provider._store)
        from tsuzuri import memory_provider_registry as registry

        self.assertIn("hybrid_v2", registry.list_providers())
        self.assertIn("biomimetic", registry.list_providers())
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual("hybrid_v2", registry._resolve_default_name())
        with mock.patch.dict(os.environ, {"TSUZURI_MEMORY_PROVIDER": "biomimetic"}, clear=True):
            self.assertEqual("biomimetic", registry._resolve_default_name())


if __name__ == "__main__":
    unittest.main()
