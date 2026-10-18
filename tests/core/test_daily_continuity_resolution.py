from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import unittest

from tsuzuri.daily_continuity.coverage import build_shadow_memory_write_plan
from tsuzuri.daily_continuity.projection import project_continuity_items, project_unresolved_continuity
from tsuzuri.daily_continuity.resolver import resolve_continuity_items


class DailyContinuityResolutionTests(unittest.TestCase):
    def test_correction_supersedes_older_claim_with_same_canonical_key(self):
        result = resolve_continuity_items([
            {
                "item_id": "old-location",
                "canonical_key": "event:flower-show:location",
                "actor": "user",
                "kind": "commitment",
                "text": "花展地点是关东。",
                "created_at_ms": 1000,
            },
            {
                "item_id": "new-location",
                "canonical_key": "event:flower-show:location",
                "actor": "user",
                "kind": "correction",
                "text": "花展地点改成冲绳了。",
                "created_at_ms": 2000,
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["old-location"]["status"], "superseded")
        self.assertEqual(rows["old-location"]["resolution_reason"], "corrected_by:new-location")
        self.assertEqual([row["item_id"] for row in result["unresolved_items"]], ["new-location"])

    def test_cross_capsule_correction_same_actor_and_claim_key_supersedes_old_value(self):
        result = resolve_continuity_items([
            {
                "item_id": "old-intent",
                "canonical_key": "user:阿澈:relationship_intent",
                "actor": "user",
                "kind": "user_fact",
                "text": "阿澈打算谈恋爱。",
                "created_at_ms": 1000,
            },
            {
                "item_id": "new-intent",
                "canonical_key": "user:阿澈:relationship_intent",
                "actor": "user",
                "kind": "correction",
                "text": "阿澈实际上没打算谈恋爱，之前的说法是乌龙。",
                "created_at_ms": 2000,
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["old-intent"]["status"], "superseded")
        self.assertEqual(rows["new-intent"]["status"], "active")

    def test_same_claim_key_different_actor_does_not_supersede(self):
        result = resolve_continuity_items([
            {
                "item_id": "user-plan",
                "canonical_key": "schedule:ring:arrival",
                "actor": "user",
                "kind": "commitment",
                "text": "阿澈明天去 活动中心。",
                "created_at_ms": 1000,
            },
            {
                "item_id": "character-change",
                "canonical_key": "schedule:ring:arrival",
                "actor": "character",
                "kind": "correction",
                "text": "王五明天不去 活动中心。",
                "created_at_ms": 2000,
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["user-plan"]["status"], "active")

    def test_ambiguous_correction_without_key_or_source_ref_fails_closed(self):
        result = resolve_continuity_items([
            {
                "item_id": "old-plan",
                "actor": "user",
                "kind": "commitment",
                "text": "阿澈准备参加花展。",
                "created_at_ms": 1000,
            },
            {
                "item_id": "vague-change",
                "actor": "user",
                "kind": "correction",
                "text": "花展的安排有变化。",
                "created_at_ms": 2000,
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["old-plan"]["status"], "active")

    def test_ledger_event_fulfils_commitment_and_removes_it_from_unresolved(self):
        result = resolve_continuity_items(
            [{
                "item_id": "bring-score",
                "canonical_key": "commitment:score",
                "kind": "commitment",
                "text": "王五明天会把谱子带到 活动中心。",
            }],
            ledger_state={
                "events": [{
                    "target_id": "bring-score",
                    "status": "fulfilled",
                    "reason": "action_ledger_completed",
                }]
            },
        )

        self.assertEqual(result["unresolved_count"], 0)
        self.assertEqual(result["items"][0]["status"], "fulfilled")

    def test_canonical_ledger_status_is_scoped_by_actor(self):
        result = resolve_continuity_items(
            [
                {
                    "item_id": "user-ring",
                    "canonical_key": "schedule:ring:arrival",
                    "actor": "user",
                    "kind": "commitment",
                    "text": "阿澈会去 活动中心。",
                },
                {
                    "item_id": "character-ring",
                    "canonical_key": "schedule:ring:arrival",
                    "actor": "character",
                    "kind": "commitment",
                    "text": "王五会去 活动中心。",
                },
            ],
            ledger_state={
                "events": [
                    {
                        "canonical_key": "schedule:ring:arrival",
                        "actor": "character",
                        "status": "fulfilled",
                    }
                ]
            },
        )

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["user-ring"]["status"], "active")
        self.assertEqual(rows["character-ring"]["status"], "fulfilled")

    def test_turn_fact_ledger_correction_supersedes_capsule_claim_by_text(self):
        result = resolve_continuity_items(
            [{
                "item_id": "old-flower-location",
                "kind": "event",
                "actor": "user",
                "text": "花展地点是在关东。",
                "source_message_ids": ["capsule-old"],
                "source_at_ms": 1000,
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "ledger-correction",
                    "created_at_ms": 2000,
                    "corrections": ["花展地点改成冲绳了。"],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "superseded")
        self.assertTrue(result["items"][0]["resolution_reason"].startswith("ledger_correction:"))

    def test_turn_fact_explicit_source_cannot_supersede_different_actor(self):
        result = resolve_continuity_items(
            [{
                "item_id": "user-plan",
                "kind": "commitment",
                "actor": "user",
                "text": "阿澈明天去 活动中心。",
                "source_message_ids": ["capsule-user-plan"],
                "source_at_ms": 1000,
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "character-correction",
                    "created_at_ms": 2000,
                    "corrections": [{
                        "actor": "character",
                        "text": "王五明天不去 活动中心。",
                        "supersedes_source_message_ids": ["capsule-user-plan"],
                    }],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "active")

    def test_turn_fact_state_delta_fulfils_matching_capsule_commitment(self):
        result = resolve_continuity_items(
            [{
                "item_id": "bring-score",
                "kind": "commitment",
                "actor": "character",
                "text": "王五答应把谱子带到 活动中心。",
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "ledger-complete",
                    "created_at_ms": 2000,
                    "state_delta": "王五已经把谱子带到 活动中心，约定已完成。",
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "fulfilled")
        self.assertEqual(result["unresolved_items"], [])

    def test_turn_fact_explicit_status_cancels_matching_open_thread(self):
        result = resolve_continuity_items(
            [{
                "item_id": "session-time",
                "canonical_key": "session:saturday:time",
                "actor": "shared",
                "kind": "open_thread",
                "text": "还要确认周六集训时间。",
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "ledger-cancel",
                    "created_at_ms": 2000,
                    "open_threads": [{
                        "text": "周六集训时间不再需要确认。",
                        "canonical_key": "session:saturday:time",
                        "status": "cancelled",
                    }],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "cancelled")
        self.assertEqual(result["unresolved_items"], [])

    def test_character_action_cancel_closes_matching_capsule_commitment(self):
        result = resolve_continuity_items(
            [{
                "item_id": "ask-magazine",
                "kind": "commitment",
                "actor": "character",
                "text": "张三答应询问liuliu有没有杂志推荐。",
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "cancel-magazine",
                    "created_at_ms": 2000,
                    "character": "张三",
                    "character_action_updates": [{
                        "transition": "cancel",
                        "event_text": "询问liuliu杂志意见",
                    }],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "cancelled")
        self.assertEqual(result["unresolved_items"], [])
        self.assertIn("ledger_character_action", result["items"][0]["resolution_reason"])

    def test_character_action_cancel_matches_alias_and_recommendation_wording(self):
        result = resolve_continuity_items(
            [{
                "item_id": "old-magazine-plan",
                "kind": "commitment",
                "actor": "character",
                "text": "张三计划回头找赵六推荐几本杂志。",
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "cancel-magazine",
                    "created_at_ms": 2_000,
                    "character": "张三",
                    "character_action_updates": [{
                        "transition": "cancel",
                        "event_text": "询问liuliu杂志意见",
                    }],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "cancelled")

    def test_cross_clause_time_correction_supersedes_old_commitment(self):
        result = resolve_continuity_items(
            [{
                "item_id": "old-arrival-time",
                "kind": "commitment",
                "actor": "user",
                "text": "阿澈和张三确认看展约定：阿澈十八点半到西町。",
                "status": "active",
                "created_at_ms": 1_000,
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "correct-arrival-time",
                    "created_at_ms": 2_000,
                    "corrections": ["阿澈更正原先的十八点半，改为十九点十分。"],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "superseded")

    def test_location_slip_correction_supersedes_old_commitment(self):
        result = resolve_continuity_items(
            [{
                "item_id": "old-location-task",
                "kind": "commitment",
                "actor": "character",
                "text": "张三承诺帮阿澈查看北町附近顺路可逛的店。",
                "status": "active",
                "created_at_ms": 1_000,
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "correct-location",
                    "created_at_ms": 2_000,
                    "corrections": ["阿澈先前说的北町是口误，正确地点为西町。"],
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "superseded")

    def test_ambiguous_already_state_does_not_fulfil_commitment(self):
        result = resolve_continuity_items(
            [{
                "item_id": "bring-score-later",
                "kind": "commitment",
                "text": "王五答应之后把谱子带到 活动中心。",
                "status": "active",
            }],
            ledger_state={
                "fact_cards": [{
                    "turn_id": "ledger-delayed",
                    "state_delta": "王五已经把带谱子的计划延期了。",
                }]
            },
        )

        self.assertEqual(result["items"][0]["status"], "active")
        self.assertEqual(result["unresolved_count"], 1)

    def test_turn_fact_card_adapter_extracts_only_lifecycle_items(self):
        result = resolve_continuity_items({
            "turn_fact_cards": [{
                "turn_id": "turn-1",
                "character": "王五",
                "created_at_ms": 1000,
                "fact": "阿澈提到了花展。",
                "corrections": ["花展地点不是关东。"],
                "commitments": ["王五答应核对花展日期。"],
                "open_threads": ["还没有确认花展的入场时间。"],
                "confidence": 0.8,
            }]
        })

        self.assertEqual(result["unresolved_count"], 3)
        self.assertEqual(
            {row["kind"] for row in result["unresolved_items"]},
            {"correction", "commitment", "open_thread"},
        )

    def test_capsule_source_message_supersession_resolves_prior_item(self):
        result = resolve_continuity_items([
            {
                "item_id": "old-plan",
                "actor": "user",
                "kind": "commitment",
                "text": "演出原计划在关东举行。",
                "source_message_ids": ["msg-old-plan"],
                "source_at_ms": 1000,
                "temporal_status": "planned",
                "status": "active",
            },
            {
                "item_id": "corrected-plan",
                "actor": "user",
                "kind": "correction",
                "text": "演出地点改到冲绳。",
                "source_message_ids": ["msg-correction"],
                "supersedes_source_message_ids": ["msg-old-plan"],
                "source_at_ms": 2000,
                "temporal_status": "planned",
                "status": "active",
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["old-plan"]["status"], "superseded")
        self.assertEqual(
            rows["old-plan"]["resolution_reason"],
            "source_corrected_by:corrected-plan",
        )
        self.assertEqual(rows["corrected-plan"]["created_at_ms"], 2000)

    def test_explicit_supersession_cannot_cross_actor(self):
        result = resolve_continuity_items([
            {
                "item_id": "user-plan",
                "actor": "user",
                "kind": "commitment",
                "text": "阿澈明天去 活动中心。",
                "source_message_ids": ["msg-user-plan"],
                "source_at_ms": 1000,
            },
            {
                "item_id": "character-correction",
                "actor": "character",
                "kind": "correction",
                "text": "王五明天不去 活动中心。",
                "source_message_ids": ["msg-character-correction"],
                "supersedes": ["user-plan"],
                "supersedes_source_message_ids": ["msg-user-plan"],
                "source_at_ms": 2000,
            },
        ])

        rows = {row["item_id"]: row for row in result["items"]}
        self.assertEqual(rows["user-plan"]["status"], "active")

    def test_dataclass_input_is_accepted_without_contract_dependency(self):
        @dataclass
        class Item:
            item_id: str
            canonical_key: str
            kind: str
            text: str
            status: str = "active"

        result = resolve_continuity_items([
            Item("thread-1", "topic:bread", "open_thread", "还没决定要烤哪种面包。")
        ])

        self.assertEqual(result["unresolved_items"][0]["item_id"], "thread-1")

    def test_projection_selects_only_relevant_unresolved_items(self):
        resolution = resolve_continuity_items([
            {
                "item_id": "flower-location",
                "kind": "correction",
                "canonical_key": "flower:location",
                "text": "花展地点改成冲绳了。",
                "topics": ["花展", "冲绳"],
            },
            {
                "item_id": "bread-thread",
                "kind": "open_thread",
                "canonical_key": "bread:choice",
                "text": "还没决定要烤哪种面包。",
                "topics": ["烘焙", "面包"],
            },
        ])

        projection = project_unresolved_continuity("冲绳的花展后来怎么样了？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["flower-location"])
        self.assertTrue(projection["records"][0]["pinned"])
        self.assertEqual(projection["records"][0]["source_channel"], "tsuzuri.daily_continuity")

    def test_projection_never_pins_more_than_two_items(self):
        resolution = resolve_continuity_items([
            {
                "item_id": f"project-{index}",
                "kind": "commitment",
                "canonical_key": f"project:{index}",
                "text": f"项目收尾事项 {index} 还需要确认。",
                "topics": ["项目", "收尾"],
                "salience": 0.6 + index * 0.05,
            }
            for index in range(4)
        ])

        projection = project_continuity_items(
            "项目收尾还有哪些事项？",
            resolution,
            max_records=10,
            max_pins=10,
        )

        self.assertEqual(len(projection["unresolved_pins"]), 2)
        self.assertEqual(len(projection["records"]), 4)
        self.assertEqual(
            len([row for row in projection["rejected"] if row["reason"] == "pin_limit"]),
            2,
        )

    def test_schedule_overload_projection_is_relevant_bounded_and_due_ordered(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-07-25T12:00:00+09:00").timestamp() * 1000
        )
        tomorrow_start_ms = int(
            datetime.fromisoformat("2026-07-26T08:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items(
            [
                {
                    "item_id": item_id,
                    "canonical_key": f"schedule:wuwu:{item_id}",
                    "actor": "character",
                    "kind": "commitment",
                    "text": text,
                    "temporal_status": "planned",
                    "due_at_ms": due_at_ms,
                    "status": "active",
                }
                for item_id, text, due_at_ms in (
                    ("ring-shift", "王五明天先去 活动中心 上早班。", tomorrow_start_ms),
                    ("practice", "王五明天下午继续练习。", tomorrow_start_ms + 6 * 60 * 60 * 1000),
                    ("fireworks", "王五明天晚上去花火大会。", tomorrow_start_ms + 11 * 60 * 60 * 1000),
                )
            ],
            as_of_ms=as_of_ms,
        )

        projection = project_continuity_items(
            "你明天的日程是不是排满了？",
            resolution,
            max_records=2,
            max_pins=2,
        )
        unrelated = project_continuity_items("今天的天气怎么样？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["ring-shift", "practice"])
        self.assertLessEqual(len(projection["records"]), 2)
        self.assertLessEqual(len(projection["unresolved_pins"]), 2)
        self.assertEqual(unrelated["records"], [])

    def test_relationship_intent_negation_does_not_open_schedule_recall(self):
        resolution = resolve_continuity_items([
            {
                "item_id": "ring-plan",
                "canonical_key": "character:王五:ring_shift",
                "actor": "character",
                "kind": "commitment",
                "text": "王五明天去 活动中心 上早班。",
                "temporal_status": "planned",
                "due_at_ms": 3_000,
            }
        ], as_of_ms=2_000)

        projection = project_continuity_items("我其实没打算谈恋爱。", resolution)

        self.assertEqual(projection["records"], [])

    def test_due_urgency_does_not_rank_non_schedule_topic(self):
        resolution = resolve_continuity_items(
            [
                {
                    "item_id": "urgent-shift",
                    "actor": "character",
                    "kind": "commitment",
                    "text": "王五要去 活动中心 值班。",
                    "due_at_ms": 3_000,
                    "salience": 0.1,
                },
                {
                    "item_id": "salient-ring-fact",
                    "actor": "character",
                    "kind": "commitment",
                    "text": "王五很重视 活动中心 的工作交接。",
                    "salience": 0.9,
                },
            ],
            as_of_ms=2_000,
        )

        projection = project_continuity_items(
            "活动中心 的工作交接怎么样？",
            resolution,
            max_records=1,
            max_pins=0,
        )

        self.assertEqual(projection["selected_item_ids"], ["salient-ring-fact"])

    def test_schedule_query_selects_only_requested_actor(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-07-25T12:00:00+09:00").timestamp() * 1000
        )
        tomorrow_ms = int(
            datetime.fromisoformat("2026-07-26T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items(
            [
                {
                    "item_id": "user-exam",
                    "canonical_key": "user:阿澈:exam",
                    "actor": "user",
                    "kind": "commitment",
                    "text": "阿澈明天复习考试。",
                    "temporal_status": "planned",
                    "due_at_ms": tomorrow_ms,
                },
                {
                    "item_id": "character-shift",
                    "canonical_key": "character:王五:ring_shift",
                    "actor": "character",
                    "kind": "commitment",
                    "text": "王五明天去 活动中心 上早班。",
                    "temporal_status": "planned",
                    "due_at_ms": tomorrow_ms + 60 * 60 * 1000,
                },
            ],
            as_of_ms=as_of_ms,
        )

        mine = project_continuity_items("我的日程是什么？", resolution)
        yours = project_continuity_items("你明天打算做什么？", resolution)

        self.assertEqual(mine["selected_item_ids"], ["user-exam"])
        self.assertEqual(yours["selected_item_ids"], ["character-shift"])

    def test_projection_does_not_pin_irrelevant_item_without_query_overlap(self):
        resolution = resolve_continuity_items([{
            "item_id": "schedule-thread",
            "kind": "open_thread",
            "canonical_key": "schedule:session",
            "text": "还没确认周六集训时间。",
            "topics": ["集训"],
        }])

        projection = project_unresolved_continuity("今天的天气怎么样？", resolution)

        self.assertEqual(projection["records"], [])

    def test_relevant_event_and_user_fact_enter_without_using_pin_budget(self):
        resolution = resolve_continuity_items([
            {
                "item_id": "ring-session",
                "kind": "event",
                "text": "王五昨天在 活动中心 完成了集训。",
                "subjects": ["王五", "活动中心"],
                "status": "active",
            },
            {
                "item_id": "user-city",
                "kind": "user_fact",
                "text": "阿澈现在住在杭州。",
                "subjects": ["阿澈", "杭州"],
                "status": "active",
            },
        ])

        projection = project_continuity_items("你还记得我住在杭州吗？", resolution)

        self.assertIn("user-city", projection["selected_item_ids"])
        self.assertEqual(projection["unresolved_pins"], [])
        self.assertTrue(all(not row["pinned"] for row in projection["records"]))

    def test_explicit_cross_day_recall_can_select_recent_event_without_entity_overlap(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-07-25T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "recent-event",
                "kind": "event",
                "text": "王五在 活动中心 完成了集训。",
                "source_time_period": "2026-07-24",
                "status": "active",
            },
            {
                "item_id": "older-event",
                "kind": "event",
                "text": "张三整理了演出服。",
                "source_time_period": "2026-07-23",
                "status": "active",
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items("昨天发生了什么？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["recent-event"])
        self.assertEqual(projection["unresolved_pins"], [])

    def test_explicit_calendar_date_selects_only_matching_capsule_day(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "august-third",
                "kind": "event",
                "text": "张三在录音棚拍了合照。",
                "source_time_period": "2026-08-03",
            },
            {
                "item_id": "august-second",
                "kind": "event",
                "text": "李四整理了诗句本。",
                "source_time_period": "2026-08-02",
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items("8月3日发生了什么？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["august-third"])

    def test_old_time_word_in_item_text_does_not_create_false_topic_overlap(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T12:00:00+09:00").timestamp() * 1000
        )
        old_due_ms = int(
            datetime.fromisoformat("2026-07-26T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "stale-tomorrow-promise",
            "kind": "commitment",
            "actor": "character",
            "text": "张三承诺明天发送粉色衣服的照片。",
            "source_time_period": "2026-07-25",
            "due_at_ms": old_due_ms,
            "temporal_status": "planned",
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items(
            "你还记得我明天去哪、你答应了什么吗？",
            resolution,
        )

        self.assertEqual(projection["selected_item_ids"], [])
        self.assertIn(
            {"item_id": "stale-tomorrow-promise", "reason": "temporal_mismatch"},
            projection["rejected"],
        )

    def test_generic_recall_and_commitment_words_are_not_topic_overlap(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "old-photo-promise",
                "kind": "commitment",
                "text": "张三答应给阿澈看粉色裙子的照片。",
                "source_time_period": "2026-07-30",
            },
            {
                "item_id": "old-ice-cream-fact",
                "kind": "user_fact",
                "text": "张三记得阿澈喜欢巧克力夹心冰棍。",
                "source_time_period": "2026-07-25",
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items(
            "你还记得我明天去哪、你答应了什么吗？",
            resolution,
        )

        self.assertEqual(projection["selected_item_ids"], [])

    def test_generic_recall_can_still_open_current_day_continuity(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "today-thread",
            "kind": "open_thread",
            "text": "张三正在等阿澈确认晚餐选哪家店。",
            "source_time_period": "2026-08-05",
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items("你还记得刚才的事吗？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["today-thread"])

    def test_generic_action_word_does_not_activate_unrelated_old_continuity(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "old-camp-list",
            "kind": "commitment",
            "actor": "character",
            "text": "张三承诺整理合宿地点清单。",
            "source_time_period": "2026-08-02",
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items("路线整理到哪一步了？", resolution)

        self.assertEqual(projection["selected_item_ids"], [])

    def test_generic_thing_reference_does_not_activate_unrelated_old_fact(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "old-nightmare",
            "kind": "event",
            "text": "阿澈昨晚做了一个噩梦，并保证这件事是真的。",
            "source_time_period": "2026-07-24",
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items(
            "你还记得杂志那件事还需要你做吗？",
            resolution,
        )

        self.assertEqual(projection["selected_item_ids"], [])

    def test_today_exact_minute_selects_only_matching_source_and_renders_minute(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:45+09:00").timestamp() * 1000
        )
        source_1942 = int(
            datetime.fromisoformat("2026-08-05T19:42:51+09:00").timestamp() * 1000
        )
        source_1943 = int(
            datetime.fromisoformat("2026-08-05T19:43:02+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "at-1942",
                "kind": "event",
                "text": "张三开始整理会议照片。",
                "source_time_period": "2026-08-05",
                "source_at_ms": source_1942,
            },
            {
                "item_id": "at-1943",
                "kind": "event",
                "text": "张三把乐谱放到桌上。",
                "source_time_period": "2026-08-05",
                "source_at_ms": source_1943,
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items("今天19点42分发生了什么？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["at-1942"])
        self.assertEqual(
            projection["records"][0]["source_time_period"],
            "2026-08-05 19:42 WORLD_TZ",
        )
        self.assertIn(
            {"item_id": "at-1943", "reason": "temporal_mismatch"},
            projection["rejected"],
        )

    def test_today_chinese_exact_minute_selects_only_matching_source(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:45+09:00").timestamp() * 1000
        )
        source_1910 = int(
            datetime.fromisoformat("2026-08-05T19:10:51+09:00").timestamp() * 1000
        )
        source_1930 = int(
            datetime.fromisoformat("2026-08-05T19:30:02+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "at-1910",
                "kind": "event",
                "text": "张三开始整理路线。",
                "source_time_period": "2026-08-05",
                "source_at_ms": source_1910,
            },
            {
                "item_id": "at-1930",
                "kind": "event",
                "text": "张三结束整理路线。",
                "source_time_period": "2026-08-05",
                "source_at_ms": source_1930,
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items("今天十九点十分发生了什么？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["at-1910"])
        self.assertIn(
            {"item_id": "at-1930", "reason": "temporal_mismatch"},
            projection["rejected"],
        )

    def test_today_future_source_minute_is_not_visible(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:45+09:00").timestamp() * 1000
        )
        future_ms = int(
            datetime.fromisoformat("2026-08-05T20:01:01+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "future-today",
            "kind": "event",
            "text": "张三结束了会议。",
            "source_time_period": "2026-08-05",
            "source_at_ms": future_ms,
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items("你还记得刚才的事吗？", resolution)

        self.assertEqual(projection["selected_item_ids"], [])
        self.assertIn(
            {"item_id": "future-today", "reason": "future_source_minute"},
            projection["rejected"],
        )

    def test_today_same_minute_source_is_visible_at_minute_precision(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:01+09:00").timestamp() * 1000
        )
        same_minute_ms = int(
            datetime.fromisoformat("2026-08-05T20:00:59+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([{
            "item_id": "same-minute",
            "kind": "event",
            "text": "张三打开了会议资料。",
            "source_time_period": "2026-08-05",
            "source_at_ms": same_minute_ms,
        }], as_of_ms=as_of_ms)

        projection = project_continuity_items("你还记得刚才的事吗？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["same-minute"])
        self.assertEqual(
            projection["records"][0]["source_time_period"],
            "2026-08-05 20:00 WORLD_TZ",
        )

    def test_future_time_cue_activates_matching_due_date_only(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-08-05T12:00:00+09:00").timestamp() * 1000
        )
        tomorrow_ms = int(
            datetime.fromisoformat("2026-08-06T18:00:00+09:00").timestamp() * 1000
        )
        old_due_ms = int(
            datetime.fromisoformat("2026-07-26T18:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items([
            {
                "item_id": "tomorrow-plan",
                "kind": "commitment",
                "actor": "user",
                "text": "阿澈要去西町看展。",
                "source_time_period": "2026-08-04",
                "due_at_ms": tomorrow_ms,
                "temporal_status": "planned",
            },
            {
                "item_id": "expired-plan",
                "kind": "commitment",
                "actor": "user",
                "text": "阿澈要去北町买东西。",
                "source_time_period": "2026-07-25",
                "due_at_ms": old_due_ms,
                "temporal_status": "planned",
            },
        ], as_of_ms=as_of_ms)

        projection = project_continuity_items("我明天要去哪里？", resolution)

        self.assertEqual(projection["selected_item_ids"], ["tomorrow-plan"])

    def test_topic_specific_cross_day_recall_excludes_unrelated_items(self):
        resolution = resolve_continuity_items([
            {
                "item_id": "conflict-correction",
                "kind": "correction",
                "actor": "user",
                "text": "阿澈是和家里闹矛盾后向王五抱怨，并非和王五吵架。",
                "status": "active",
            },
            {
                "item_id": "dessert-promise",
                "kind": "commitment",
                "actor": "user",
                "text": "阿澈承诺以后带张三去甜品店。",
                "status": "active",
            },
        ])

        projection = project_continuity_items(
            "昨天你是不是以为我和王五吵架了？", resolution
        )

        self.assertEqual(projection["selected_item_ids"], ["conflict-correction"])
        self.assertIn(
            {"item_id": "dessert-promise", "reason": "irrelevant"},
            projection["rejected"],
        )

    def test_generic_recall_does_not_force_unrelated_old_correction(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-07-25T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items(
            [
                {
                    "item_id": "old-event",
                    "kind": "event",
                    "actor": "character",
                    "text": "王五去过 活动中心。",
                    "source_time_period": "2026-06-20",
                },
                {
                    "item_id": "old-correction",
                    "kind": "correction",
                    "actor": "user",
                    "canonical_key": "user:阿澈:relationship_intent",
                    "text": "阿澈没有谈恋爱的打算，之前是乌龙。",
                    "source_time_period": "2026-06-20",
                },
            ],
            as_of_ms=as_of_ms,
        )

        projection = project_continuity_items("你还记得之前的事吗？", resolution)

        self.assertNotIn("old-event", projection["selected_item_ids"])
        self.assertNotIn("old-correction", projection["selected_item_ids"])
        relevant = project_continuity_items("你还记得谈恋爱的乌龙吗？", resolution)
        self.assertIn("old-correction", relevant["selected_item_ids"])

    def test_old_resolved_lifecycle_item_expires(self):
        as_of_ms = int(
            datetime.fromisoformat("2026-07-25T12:00:00+09:00").timestamp() * 1000
        )
        resolution = resolve_continuity_items(
            [{
                "item_id": "resolved-plan",
                "kind": "commitment",
                "actor": "character",
                "text": "王五已经确认了集训时间。",
                "status": "resolved",
                "source_time_period": "2026-06-20",
            }],
            as_of_ms=as_of_ms,
        )

        projection = project_continuity_items("你还记得之前的事吗？", resolution)

        self.assertEqual(projection["selected_item_ids"], [])
        self.assertEqual(projection["rejected"][0]["reason"], "stale")

    def test_coverage_suppresses_item_already_present_in_memory(self):
        resolution = resolve_continuity_items([{
            "item_id": "home-location",
            "kind": "correction",
            "canonical_key": "user:home-city",
            "actor": "user",
            "text": "阿澈现在住在杭州。",
            "confidence": 0.92,
        }])

        plan = build_shadow_memory_write_plan(
            resolution,
            [{
                "item_id": "mem-home-city",
                "canonical_key": "user:home-city",
                "actor": "user",
                "text": "阿澈目前住在杭州。",
            }],
            character="张三",
        )

        self.assertEqual(plan["records"], [])
        self.assertEqual(plan["coverage_audit"]["covered_count"], 1)
        self.assertEqual(plan["coverage_audit"]["durable_writes_performed"], 0)

    def test_coverage_same_key_different_actor_is_not_treated_as_covered(self):
        plan = build_shadow_memory_write_plan(
            [
                {
                    "item_id": "user-ring-plan",
                    "kind": "commitment",
                    "canonical_key": "schedule:ring:arrival",
                    "actor": "user",
                    "text": "阿澈明天去 活动中心。",
                    "confidence": 0.9,
                }
            ],
            [
                {
                    "item_id": "character-ring-plan",
                    "canonical_key": "schedule:ring:arrival",
                    "actor": "character",
                    "text": "王五明天去 活动中心。",
                }
            ],
            character="王五",
        )

        self.assertEqual(plan["coverage_audit"]["covered_count"], 0)
        self.assertEqual([row["evidence_id"] for row in plan["records"]], ["user-ring-plan"])

    def test_coverage_same_text_different_actor_is_not_treated_as_covered(self):
        plan = build_shadow_memory_write_plan(
            [{
                "item_id": "user-ring-plan",
                "kind": "commitment",
                "actor": "user",
                "text": "明天去 活动中心 确认设备。",
                "confidence": 0.9,
            }],
            [{
                "item_id": "character-ring-plan",
                "actor": "character",
                "text": "明天去 活动中心 确认设备。",
            }],
            character="王五",
        )

        self.assertEqual(plan["coverage_audit"]["covered_count"], 0)
        self.assertEqual([row["evidence_id"] for row in plan["records"]], ["user-ring-plan"])

    def test_coverage_emits_shadow_candidate_but_never_apply_instruction(self):
        resolution = resolve_continuity_items([{
            "item_id": "new-correction",
            "kind": "correction",
            "canonical_key": "event:venue",
            "text": "演出场地改到下北泽。",
            "confidence": 0.88,
            "salience": 0.84,
        }])

        plan = build_shadow_memory_write_plan(resolution, [], character="王五")

        self.assertTrue(plan["shadow_only"])
        self.assertFalse(plan["apply"])
        self.assertEqual(plan["coverage_audit"]["durable_writes_performed"], 0)
        self.assertEqual(len(plan["records"]), 1)
        self.assertTrue(plan["records"][0]["payload"]["shadow_only"])

    def test_open_thread_is_audited_but_not_promoted_to_memory_candidate(self):
        resolution = resolve_continuity_items([{
            "item_id": "open-time",
            "kind": "open_thread",
            "canonical_key": "event:start-time",
            "text": "还没确认演出开始时间。",
            "confidence": 0.9,
        }])

        plan = build_shadow_memory_write_plan(resolution, [], character="王五")

        self.assertEqual(plan["records"], [])
        self.assertEqual(plan["suppressed"][0]["reason"], "transient_open_thread")

    def test_v2_event_and_user_fact_are_included_in_shadow_coverage(self):
        plan = build_shadow_memory_write_plan([
            {
                "item_id": "event-1",
                "kind": "event",
                "text": "王五在 活动中心 完成了集训。",
                "status": "active",
                "confidence": 0.82,
            },
            {
                "item_id": "user-fact-1",
                "kind": "user_fact",
                "text": "阿澈现在住在杭州。",
                "status": "active",
                "confidence": 0.91,
            },
        ], [], character="王五")

        self.assertEqual(
            {record["item_type"] for record in plan["records"]},
            {"event", "user_fact"},
        )
        self.assertFalse(plan["apply"])


if __name__ == "__main__":
    unittest.main()
