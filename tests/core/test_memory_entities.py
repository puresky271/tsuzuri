from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tsuzuri.memory_api import MemoryApi
from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_entities import (
    entity_links,
    evaluate_association_hop1_shadow,
    list_entities,
    rebuild_entities,
)
from tsuzuri.memory_provider_hybrid import HybridIndexedMemoryStore
from tsuzuri.memory_tools import propose_proactive_memory_record


class MemoryEntityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.store = MemoryCoreStore(
            db_path=str(root / "memory.db"),
            snapshot_path=str(root / "snapshot.json"),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _put(
        self,
        item_id: str,
        *,
        character: str,
        text: str,
        entities: list[str],
        topic_labels: list[str] | None = None,
        guest: bool = False,
    ) -> None:
        self.store._upsert_item({
            "item_id": item_id,
            "character": character,
            "layer": "collective" if character == "测试组" else "semantic",
            "memory_role": "collective" if character == "测试组" else "semantic",
            "engram_type": "fact",
            "item_type": "fact",
            "text": text,
            "payload_json": json.dumps(
                {"memory_meta": {"mode_scope": "guest:test" if guest else "normal"}},
                ensure_ascii=False,
            ),
            "cue_json": json.dumps({"entities": entities}, ensure_ascii=False),
            "topic_labels_json": json.dumps(topic_labels or [], ensure_ascii=False),
            "salience": 0.75,
            "confidence": 0.85,
            "importance": 0.75,
            "source": "entity_test",
            "created_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-01T00:00:00Z",
        })

    def test_rebuild_is_stable_and_scoped(self) -> None:
        self._put(
            "zhangsan-item",
            character="张三",
            text="李四在活动中心集训。",
            entities=["张三", "李四", "活动中心"],
            topic_labels=["session", "activity_center"],
        )
        self._put(
            "guest-item",
            character="张三",
            text="不应派生的访客事实。",
            entities=["访客秘密"],
            guest=True,
        )
        self._put("zhaoliu-item", character="赵六", text="李四在活动中心集训。", entities=["李四", "活动中心"])
        self._put("collective-item", character="测试组", text="全员在活动中心集训。", entities=["活动中心"])

        first = rebuild_entities(self.store, "张三")
        first_rows = list_entities(self.store, "张三", limit=100)
        second = rebuild_entities(self.store, "张三")
        second_rows = list_entities(self.store, "张三", limit=100)
        rebuild_entities(self.store, "赵六")
        rebuild_entities(self.store, "测试组")

        self.assertEqual(first["entity_count"], second["entity_count"])
        self.assertEqual(
            {(row["entity_id"], row["name"], row["count"]) for row in first_rows},
            {(row["entity_id"], row["name"], row["count"]) for row in second_rows},
        )
        self.assertNotIn("张三", {row["name"] for row in first_rows})
        self.assertNotIn("访客秘密", {row["name"] for row in first_rows})
        zhangsan_ring = next(row for row in first_rows if row["name"] == "活动中心")
        zhaoliu_ring = next(row for row in list_entities(self.store, "赵六") if row["name"] == "活动中心")
        collective_ring = next(row for row in list_entities(self.store, "测试组") if row["name"] == "活动中心")
        self.assertEqual("张三", zhangsan_ring["char_scope"])
        self.assertNotEqual(zhangsan_ring["entity_id"], zhaoliu_ring["entity_id"])
        self.assertNotEqual(zhangsan_ring["entity_id"], collective_ring["entity_id"])

    def test_rebuild_uses_cooccurrence_and_memory_links(self) -> None:
        self._put("item-a", character="张三", text="活动中心的第一次集训。", entities=["活动中心"])
        self._put("item-b", character="张三", text="活动中心的第二次集训。", entities=["活动中心"])
        with self.store._lock:
            conn = self.store._connect()
            try:
                conn.execute(
                    "INSERT INTO co_occurrence_strength(item_a,item_b,strength,last_updated,update_count) "
                    "VALUES ('item-a','item-b',0.6,'2026-07-02T00:00:00Z',1);"
                )
                conn.execute(
                    "INSERT INTO memory_links(src_item_id,dst_item_id,link_type,weight,meta_json,created_at) "
                    "VALUES ('item-a','item-b','merge_into',0.8,'{}','2026-07-02T00:00:00Z');"
                )
                conn.commit()
            finally:
                conn.close()

        rebuild_entities(self.store, "张三")
        ring = next(row for row in list_entities(self.store, "张三") if row["name"] == "活动中心")
        self.assertEqual(2, ring["count"])
        self.assertAlmostEqual(0.6, ring["cooc_strength"])
        linked = entity_links(self.store, ring["entity_id"], char_scope="张三")
        self.assertEqual({"item-a", "item-b"}, {row["memory_item_id"] for row in linked})

    def test_hybrid_boost_is_default_off_and_marks_entity_overlap(self) -> None:
        root = Path(self.tmp.name)
        self.store = HybridIndexedMemoryStore(
            db_path=str(root / "hybrid.db"),
            snapshot_path=str(root / "hybrid.json"),
            dense_query_encoder=lambda _text: None,
            sparse_query_encoder=lambda _text: None,
        )
        self._put(
            "entity-hit",
            character="张三",
            text="李四在活动中心集训时写下诗句。",
            entities=["李四", "活动中心"],
            topic_labels=["session", "activity_center"],
        )
        rebuild_entities(self.store, "张三")

        with mock.patch.dict(os.environ, {"MEM_ENTITY_BOOST_ENABLE": "0"}), \
             mock.patch("tsuzuri.memory_entities.annotate_entity_boosts") as annotate:
            off_rows = self.store._query_items("张三", limit=20, query_text="李四在活动中心集训的事")
            off_context = self.store.retrieve_memory_context(
                "张三",
                "李四在活动中心集训的事",
                query_bundle={},
                max_items=6,
                touch_selected=False,
                log_retrieval=False,
            )
        annotate.assert_not_called()
        self.assertTrue(all(float(row.get("_entity_boost", 0.0) or 0.0) == 0.0 for row in off_rows))

        self.store._recall_inflight_cache.clear()
        with mock.patch.dict(os.environ, {"MEM_ENTITY_BOOST_ENABLE": "1"}):
            on_rows = self.store._query_items("张三", limit=20, query_text="李四在活动中心集训的事")
            on_context = self.store.retrieve_memory_context(
                "张三",
                "李四在活动中心集训的事",
                query_bundle={},
                max_items=6,
                touch_selected=False,
                log_retrieval=False,
            )
        hit = next(row for row in on_rows if row["item_id"] == "entity-hit")
        self.assertEqual(1.35, hit["_entity_boost"])
        off_score = next(row["score"] for row in off_context["rows"] if row["item_id"] == "entity-hit")
        on_score = next(row["score"] for row in on_context["rows"] if row["item_id"] == "entity-hit")
        self.assertGreater(on_score, off_score)

    def test_hop1_shadow_finds_exact_scope_neighbor_without_mutating_rows(self) -> None:
        self._put("seed", character="张三", text="活动中心で歌詞を書いた。", entities=["活动中心"])
        self._put("neighbor", character="张三", text="ライブの帰りに雨が降った。", entities=["活动中心"])
        self._put("cross-scope", character="赵六", text="赵六だけが知る話。", entities=["活动中心"])
        rebuild_entities(self.store, "张三")
        seed_rows = [{"item_id": "seed", "character": "张三", "score": 0.82}]
        before = json.loads(json.dumps(seed_rows, ensure_ascii=False))

        report = evaluate_association_hop1_shadow(
            self.store,
            "张三",
            "活动中心で書いた歌詞",
            seed_rows,
            retrieval_run_id="fixture-hop1",
            persist=True,
        )

        self.assertEqual("ok", report["status"])
        self.assertEqual(before, seed_rows)
        self.assertEqual(["neighbor"], [row["candidate_item_id"] for row in report["candidates"]])
        self.assertNotIn("cross-scope", json.dumps(report, ensure_ascii=False))
        with self.store._lock:
            conn = self.store._connect()
            try:
                persisted = conn.execute(
                    "SELECT retrieval_run_id, char_scope FROM memory_association_shadow_runs;"
                ).fetchall()
            finally:
                conn.close()
        self.assertEqual([("fixture-hop1", "张三")], [tuple(row) for row in persisted])

    def test_hop1_shadow_blocks_collective_guest_and_topic_only_paths(self) -> None:
        self._put(
            "topic-seed",
            character="张三",
            text="最初の話。",
            entities=["固有A"],
            topic_labels=["shared_topic"],
        )
        self._put(
            "topic-neighbor",
            character="张三",
            text="別の話。",
            entities=["固有B"],
            topic_labels=["shared_topic"],
        )
        rebuild_entities(self.store, "张三")
        report = evaluate_association_hop1_shadow(
            self.store,
            "张三",
            "最初の話",
            [{"item_id": "topic-seed", "character": "张三", "score": 0.9}],
        )
        self.assertEqual("empty", report["status"])
        self.assertGreater(report["metrics"]["excluded_topic_paths"], 0)
        self.assertEqual(
            "collective_scope_blocked",
            evaluate_association_hop1_shadow(self.store, "测试组", "集训", [],)["status"],
        )
        self.assertEqual(
            "invalid_scope",
            evaluate_association_hop1_shadow(self.store, "guest:test", "集训", [],)["status"],
        )

    def test_hop1_shadow_excludes_generic_entities(self) -> None:
        self._put("gseed", character="张三", text="阿澈说了一个故事。", entities=["user", "阿澈"])
        self._put("gneighbor", character="张三", text="另一个故事。", entities=["user", "阿澈"])
        rebuild_entities(self.store, "张三")
        report = evaluate_association_hop1_shadow(
            self.store,
            "张三",
            "阿澈的故事",
            [{"item_id": "gseed", "character": "张三", "score": 0.9}],
        )
        self.assertEqual("empty", report["status"])
        self.assertGreater(report["metrics"]["excluded_generic_paths"], 0)

    def test_hop1_shadow_downweights_groupmate_hub_below_floor(self) -> None:
        self._put("hseed", character="张三", text="王五今天练习。", entities=["王五"])
        self._put("hn1", character="张三", text="王五昨天练习。", entities=["王五"])
        self._put("hn2", character="张三", text="王五上周练习。", entities=["王五"])
        rebuild_entities(self.store, "张三")
        report = evaluate_association_hop1_shadow(
            self.store,
            "张三",
            "王五练习",
            [{"item_id": "hseed", "character": "张三", "score": 0.9}],
        )
        self.assertGreater(report["metrics"]["downweighted_person_hub_paths"], 0)
        # 降权 + idf 后分数低于地板，不应把泛化的 groupmate 边当"有效联想"
        self.assertEqual("empty", report["status"])

    def test_hop1_shadow_uses_pair_cooccurrence_as_edge_weight(self) -> None:
        self._put("pseed", character="张三", text="活动中心集训。", entities=["活动中心"])
        self._put("pn1", character="张三", text="活动中心演出。", entities=["活动中心"])
        rebuild_entities(self.store, "张三")
        with self.store._lock:
            conn = self.store._connect()
            try:
                conn.execute(
                    "INSERT INTO co_occurrence_strength(item_a, item_b, strength, last_updated, update_count) "
                    "VALUES (?, ?, ?, ?, 1);",
                    (min("pseed", "pn1"), max("pseed", "pn1"), 0.5, "2026-07-01T00:00:00Z"),
                )
                conn.commit()
            finally:
                conn.close()
        report = evaluate_association_hop1_shadow(
            self.store,
            "张三",
            "活动中心",
            [{"item_id": "pseed", "character": "张三", "score": 0.8}],
        )
        self.assertEqual("ok", report["status"])
        self.assertGreater(report["candidates"][0]["shadow_score"], 0.2)

    def test_hybrid_provider_shadow_is_default_off_and_does_not_change_memory_rows(self) -> None:
        root = Path(self.tmp.name)
        provider = __import__("tsuzuri.memory_provider_hybrid", fromlist=["_"]).HybridIndexedMemoryProvider(
            db_path=str(root / "provider.db"),
            snapshot_path=str(root / "provider.json"),
            dense_query_encoder=lambda _text: None,
            sparse_query_encoder=lambda _text: None,
        )
        store = provider._get_store()
        self.store = store
        self._put("provider-seed", character="张三", text="李四在活动中心集训。", entities=["李四", "活动中心"])
        rebuild_entities(store, "张三")
        with mock.patch.dict(os.environ, {"MEM_ASSOC_HOP1_SHADOW_ENABLE": "0"}), \
             mock.patch("tsuzuri.memory_entities.evaluate_association_hop1_shadow") as evaluate:
            off = provider.build_memory_context_pack("张三", "李四在活动中心集训", max_items=6)
        evaluate.assert_not_called()
        self.assertNotIn("association_hop1_shadow", off)

        with mock.patch.dict(os.environ, {"MEM_ASSOC_HOP1_SHADOW_ENABLE": "1"}), \
             mock.patch(
                 "tsuzuri.memory_entities.evaluate_association_hop1_shadow",
                 return_value={"status": "empty", "candidates": [], "metrics": {}},
             ) as evaluate:
            on = provider.build_memory_context_pack("张三", "李四在活动中心集训", max_items=6)
        evaluate.assert_called_once()
        self.assertEqual(off.get("memory_rows", []), on.get("memory_rows", []))
        self.assertEqual("empty", on["association_hop1_shadow"]["status"])

    pass  # upstream-only test removed

    def test_kairos_entity_rebuild_is_idempotent(self) -> None:
        from tsuzuri import kairos_dream as kairos_dream

        self._put("kairos-item", character="张三", text="李四在活动中心。", entities=["李四", "活动中心"])
        self._put("kairos-collective", character="测试组", text="全员在活动中心。", entities=["活动中心"])
        with mock.patch.object(kairos_dream, "_kairos_logical_day_world", return_value="2026-08-03"):
            first = kairos_dream._run_entity_rebuild(self.store, "张三")
            with mock.patch("tsuzuri.memory_entities.rebuild_entities", side_effect=AssertionError("must not rebuild")):
                second = kairos_dream._run_entity_rebuild(self.store, "张三")
        self.assertEqual("rebuilt", first["status"])
        self.assertEqual("idempotent", second["status"])
        meta = self.store.get_meta_state_json("kairos.entity_rebuilt.张三.2026-08-03", {})
        self.assertTrue(meta["terminal"])

    def test_proactive_record_gate_evidence_and_collective_policy(self) -> None:
        card = {
            "fact": "阿澈和李四、王五在活动中心一起集训。",
            "salience": 0.9,
            "confidence": 0.9,
            "topic_labels": ["session", "activity_center", "group_dynamics"],
            "audience_scope": "developer",
        }
        context = {"turn_fact_cards": [card], "recent_user_msgs": ["这不是事实卡原文"]}
        with mock.patch.dict(os.environ, {"MEM_ENTITY_PROACTIVE_ENABLE": "0"}):
            disabled = propose_proactive_memory_record(
                "张三", trigger_kind="chat_followup", context=context, store=self.store,
            )
        self.assertEqual("disabled", disabled["status"])

        with mock.patch.dict(os.environ, {"MEM_ENTITY_PROACTIVE_ENABLE": "1"}), \
             mock.patch("tsuzuri.memory_core._rewrite_fact_via_llm", side_effect=lambda text, **_kwargs: text):
            rejected_trigger = propose_proactive_memory_record(
                "张三", trigger_kind="auto_greet", context=context, store=self.store,
            )
            low_signal = propose_proactive_memory_record(
                "张三",
                trigger_kind="chat_followup",
                context={"turn_fact_cards": [{**card, "salience": 0.4}]},
                store=self.store,
            )
            accepted = propose_proactive_memory_record(
                "张三", trigger_kind="chat_followup", context=context, store=self.store,
            )
            repeated = propose_proactive_memory_record(
                "张三", trigger_kind="delayed_followup", context=context, store=self.store,
            )
        self.assertEqual("ineligible_trigger", rejected_trigger["status"])
        self.assertEqual("low_signal", low_signal["status"])
        self.assertEqual("accepted", accepted["status"])
        self.assertEqual("idempotent", repeated["status"])
        with self.store._lock:
            conn = self.store._connect()
            try:
                rows = conn.execute(
                    "SELECT character, layer, source, evidence_id FROM memory_items "
                    "WHERE source='proactive_record' ORDER BY character;"
                ).fetchall()
            finally:
                conn.close()
        self.assertEqual({"张三", "测试组"}, {str(row["character"]) for row in rows})
        self.assertTrue(all(str(row["evidence_id"]).startswith("proactive:") for row in rows))


if __name__ == "__main__":
    unittest.main()
