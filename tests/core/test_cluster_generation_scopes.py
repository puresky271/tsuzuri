"""Separate recurring-topic eligibility from contiguous event windows."""

import unittest
from datetime import timedelta
from unittest.mock import patch

from test_cluster_memory import ClusterMemoryTestBase, _T0, _valid_raw
from tsuzuri.cluster_memory import build_cluster_plan, create_cluster_memories, _normalize_draft


class GenerationScopeTests(ClusterMemoryTestBase):
    def setUp(self):
        super().setUp()
        self.encoder = patch("tsuzuri.memory_core._embedding_callable", return_value=False)
        self.encoder.start()
        self.addCleanup(self.encoder.stop)

    def seed(self, prefix, days, labels=None):
        for index, day in enumerate(days):
            self._insert_episodic(f"{prefix}{index}", text=f"{prefix}具体讨论内容{index}",
                                  created_at=_T0 + timedelta(days=day), labels=labels or ["travel"])

    def test_sparse_cross_day_discussion_gets_topic_candidate(self):
        self.seed("topic", [0, 2, 5])
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=6))
        self.assertEqual(len(plan["groups"]), 1)
        row = plan["groups"][0]
        self.assertEqual(row["generation_scope"], "recurring_topic")
        self.assertEqual(set(row["member_ids"]), {"topic0", "topic1", "topic2"})

    def test_two_complete_events_are_not_recombined_as_topic(self):
        self.seed("event", [0, 0.1, 0.2, 4, 4.1, 4.2])
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=6))
        self.assertEqual(len(plan["groups"]), 2)
        self.assertTrue(all(g["generation_scope"] == "time_window" for g in plan["groups"]))
        self.assertFalse(set(plan["groups"][0]["member_ids"]) & set(plan["groups"][1]["member_ids"]))

    def test_event_and_sparse_topic_share_cap_not_evidence(self):
        self.seed("event", [0, 0.1, 0.2])
        self.seed("topic", [3, 5, 7])
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=8), max_clusters=1)
        self.assertEqual([g["status"] for g in plan["groups"]], ["passed", "skipped"])
        self.assertEqual(plan["groups"][1]["reason"], "run_cap_reached")
        self.assertFalse(set(plan["groups"][0]["member_ids"]) & set(plan["groups"][1]["member_ids"]))

    def test_repeated_identical_fragments_do_not_form_topic(self):
        for i in range(3):
            self._insert_episodic(f"repeat{i}", text="同一段话没有新增事实", created_at=_T0 + timedelta(days=2*i), labels=["travel"])
        self.assertEqual(build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=5))["groups"], [])

    def test_recurring_topic_cannot_be_written_as_event(self):
        self.seed("topic", [0, 2, 5])
        observed = []
        def narrator(members, char, label):
            observed.append(members[0]["_generation_scope"])
            return _valid_raw(cluster_kind="event")
        result = create_cluster_memories(self.core, "张三", dry_run=False, audit=False,
                                          llm_narrator=narrator, now=_T0 + timedelta(days=6))
        self.assertEqual(observed, ["recurring_topic"])
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["narrator_failed"], 1)

    def test_window_may_describe_topic_without_guessing_by_days(self):
        members = self._member_rows(3)
        draft = _normalize_draft(_valid_raw(cluster_kind="topic"), members, "张三", "travel")
        self.assertEqual(draft["item_type"], "topic_cluster")
        self.assertEqual(draft["generation_scope"], "time_window")

    def test_location_and_coactor_context_split_same_day_label(self):
        rows = [
            ("a", "张三和李四在活动中心集训前奏", "活动中心"),
            ("b", "张三和李四在活动中心调整音量", "活动中心"),
            ("c", "张三和李四在活动中心结束集训", "活动中心"),
            ("d", "张三与赵六在北町活动室集训结尾", "北町活动室"),
            ("e", "张三与赵六在北町活动室调整音量", "北町活动室"),
            ("f", "张三与赵六在北町活动室结束练习", "北町活动室"),
        ]
        for iid, text, location in rows:
            self.core._upsert_item({"item_id": iid, "character": "张三", "layer": "episodic", "item_type": "event",
                                    "text": text, "created_at": _T0, "updated_at": _T0, "source": "turn_fact",
                                    "topic_labels_json": '["travel"]', "payload_json": '{"location":"' + location + '"}',
                                    "cue_json": '{"location_anchor":"' + location + '"}', "salience": 0.8,
                                    "importance": 0.8, "confidence": 0.8, "embedding_text": "fake", "embedding_model_id": "fake", "embedding_dim": 3})
        plan = build_cluster_plan(self.core, "张三", now=_T0 + timedelta(days=1))
        self.assertEqual(len(plan["groups"]), 2)
        self.assertEqual({g["member_count"] for g in plan["groups"]}, {3})


if __name__ == "__main__":
    unittest.main()
