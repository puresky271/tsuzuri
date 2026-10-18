import os
import tempfile
import time
import unittest
from unittest import mock


class InterlocutorStateLedgerTests(unittest.TestCase):
    def _entry(self, role, content, entry_id, timestamp, *, scope="developer"):
        return {
            "role": role,
            "content": content,
            "id": entry_id,
            "timestamp": timestamp,
            "audience_scope": scope,
        }

    def test_explicit_vacation_is_captured_without_relying_on_llm_output(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        user = self._entry("user", "我前天就已经放暑假了，这几天不用去学校。", "u-1", 1_721_000_000_000)
        assistant = self._entry("assistant", "那你总算能缓一口气了。", "a-1", 1_721_000_001_000)

        card = normalize_turn_fact_card(
            {"fact": "阿澈告诉王五自己已经放暑假。"},
            "王五",
            user,
            assistant,
            model="fake",
        )

        self.assertEqual(len(card["developer_state_updates"]), 1)
        update = card["developer_state_updates"][0]
        self.assertEqual(update["predicate"], "academic.phase")
        self.assertEqual(update["value"], "summer_vacation")
        self.assertEqual(update["temporal_status"], "active")
        self.assertIn("放暑假", update["evidence_span"])
        self.assertEqual(update["valid_from_ms"], user["timestamp"] - 2 * 24 * 60 * 60 * 1000)

    def test_vacation_persists_across_days_and_opening_school_supersedes_it(self):
        from tsuzuri.interlocutor_state_ledger import (
            commit_turn_state_updates,
            compile_interlocutor_state_frame,
            load_active_states,
        )
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        base = 1_721_000_000_000
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            vacation = normalize_turn_fact_card(
                {"fact": "阿澈已经放暑假。"},
                "王五",
                self._entry("user", "我已经放暑假了。", "u-vac", base),
                self._entry("assistant", "知道了。", "a-vac", base + 1_000),
            )
            committed = commit_turn_state_updates("王五", "sid", vacation, db_path=db_path)
            self.assertEqual(len(committed), 1)

            two_days_later = load_active_states(
                "王五",
                now_ms=base + 2 * 24 * 60 * 60 * 1000,
                db_path=db_path,
            )
            frame = compile_interlocutor_state_frame("王五", two_days_later)
            self.assertIn("正在放暑假", frame["text"])
            self.assertIn("直接告诉王五", frame["text"])

            school = normalize_turn_fact_card(
                {"fact": "阿澈已经开学。"},
                "王五",
                self._entry("user", "我今天已经开学了。", "u-school", base + 3_000),
                self._entry("assistant", "那作息也得跟着收回来了。", "a-school", base + 4_000),
            )
            commit_turn_state_updates("王五", "sid", school, db_path=db_path)
            active = load_active_states("王五", now_ms=base + 5_000, db_path=db_path)
            self.assertEqual([row["value_text"] for row in active], ["in_term"])

    def test_character_action_completion_advances_matching_schedule(self):
        from tsuzuri.character_action_ledger import apply_action_overlays, normalize_character_action_updates

        assistant = self._entry(
            "assistant",
            "哈？想什么想。买完面包了，你等一下别乱跑。",
            "a-bread",
            1_721_000_001_000,
        )
        updates = normalize_character_action_updates([], assistant)
        self.assertEqual(updates[0]["transition"], "complete")
        self.assertEqual(updates[0]["event_text"], "买面包")

        action = {
            **updates[0],
            "source_at_ms": assistant["timestamp"],
        }
        world, schedule, lines = apply_action_overlays(
            {
                "dt": __import__("datetime").datetime.fromtimestamp(
                    assistant["timestamp"] / 1000.0,
                    tz=__import__("datetime").timezone.utc,
                ),
                "activity": "顺路买面包然后回家",
                "temporal_facts": {
                    "present_fact": {"activity": "顺路买面包然后回家"},
                    "future_facts": [{"text": "顺路买面包"}, {"text": "回家收拾东西"}],
                },
            },
            {
                "expected_activity": "顺路买面包然后回家",
                "current_steps": [{"id": "bread", "title": "顺路买面包"}],
                "future_steps": [{"id": "home", "title": "回家收拾东西"}],
                "past_steps": [],
            },
            [action],
        )

        self.assertEqual(world["activity"], "回家")
        self.assertEqual(world["temporal_facts"]["present_fact"]["activity"], "回家")
        self.assertNotIn("顺路买面包", str(world["temporal_facts"]["future_facts"]))
        self.assertEqual(schedule["current_steps"], [])
        self.assertEqual(schedule["past_steps"][-1]["conversation_status"], "completed")
        self.assertIn("已完成：买面包", lines[0])

    def test_character_action_start_rejects_future_intent_and_proactive_authority(self):
        from tsuzuri.character_action_ledger import normalize_character_action_updates

        future = self._entry(
            "assistant",
            "花火大会那天我会拍一张最好看的给你。",
            "a-future",
            1_721_000_001_000,
        )
        future_updates = normalize_character_action_updates(
            [{
                "transition": "start",
                "event_text": "拍浴衣照片",
                "evidence_span": "我会拍一张最好看的给你",
                "confidence": 1.0,
            }],
            future,
        )
        self.assertEqual(future_updates, [])

        proactive = {
            **self._entry(
                "assistant",
                "我现在在南北町公园草坪休息。",
                "a-proactive",
                1_721_000_002_000,
            ),
            "proactive": True,
            "proactive_kind": "auto_greet",
        }
        proactive_updates = normalize_character_action_updates(
            [{
                "transition": "start",
                "event_text": "在南北町公园草坪休息",
                "evidence_span": "我现在在南北町公园草坪休息",
                "confidence": 1.0,
            }],
            proactive,
        )
        self.assertEqual(proactive_updates, [])

    def test_character_action_overlay_drops_stale_cross_day_start(self):
        from datetime import datetime, timedelta, timezone

        from tsuzuri.character_action_ledger import apply_action_overlays

        world = timezone(timedelta(hours=9))
        source = datetime(2026, 7, 22, 14, 15, tzinfo=world)
        now = datetime(2026, 7, 23, 12, 5, tzinfo=world)
        world, schedule, lines = apply_action_overlays(
            {
                "dt": now,
                "activity": "独自在家吃简单的午饭",
                "location": "自宅客厅",
            },
            {
                "asof_world": now.isoformat(),
                "expected_activity": "独自在家吃简单的午饭",
                "expected_location": "自宅客厅",
                "current_steps": [{"title": "独自在家吃简单的午饭", "location": "自宅客厅"}],
            },
            [{
                "transition": "start",
                "event_text": "在南北町公园草坪休息",
                "source_at_ms": int(source.timestamp() * 1000),
            }],
        )

        self.assertEqual(world["activity"], "独自在家吃简单的午饭")
        self.assertEqual(schedule["expected_activity"], "独自在家吃简单的午饭")
        self.assertEqual(lines, [])

    def test_undoing_new_developer_state_restores_superseded_state(self):
        from tsuzuri.interlocutor_state_ledger import (
            archive_states_for_source_message_ids,
            commit_turn_state_updates,
            load_active_states,
        )

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            for user_id, turn_id, value, offset in (
                ("u-vac", "t-vac", "summer_vacation", 0),
                ("u-school", "t-school", "in_term", 1_000),
            ):
                card = {
                    "audience_scope": "developer",
                    "character": "王五",
                    "user_id": user_id,
                    "turn_id": turn_id,
                    "created_at_ms": now + offset,
                    "developer_state_updates": [{
                        "predicate": "academic.phase",
                        "value": value,
                        "operation": "set",
                        "temporal_status": "active",
                        "valid_from_ms": now + offset,
                        "valid_until_ms": now + 30 * 24 * 60 * 60 * 1000,
                        "evidence_span": "我明确说了当前学业阶段",
                        "confidence": 1.0,
                    }],
                }
                commit_turn_state_updates("王五", "sid", card, db_path=db_path)

            archive_states_for_source_message_ids(["u-school"], db_path=db_path)
            active = load_active_states("王五", now_ms=now + 2_000, db_path=db_path)

            self.assertEqual([row["value_text"] for row in active], ["summer_vacation"])

    def test_planned_vacation_does_not_replace_current_academic_phase(self):
        from tsuzuri.interlocutor_state_ledger import commit_turn_state_updates, load_active_states

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            for user_id, value, temporal_status in (
                ("u-term", "in_term", "active"),
                ("u-plan", "summer_vacation", "planned"),
            ):
                commit_turn_state_updates(
                    "王五",
                    "sid",
                    {
                        "audience_scope": "developer",
                        "character": "王五",
                        "user_id": user_id,
                        "turn_id": "t-" + user_id,
                        "created_at_ms": now,
                        "developer_state_updates": [{
                            "predicate": "academic.phase",
                            "value": value,
                            "operation": "set",
                            "temporal_status": temporal_status,
                            "valid_from_ms": now,
                            "valid_until_ms": now + 30 * 24 * 60 * 60 * 1000,
                            "evidence_span": "明确学业阶段",
                            "confidence": 1.0,
                        }],
                    },
                    db_path=db_path,
                )

            active = load_active_states("王五", now_ms=now + 1_000, db_path=db_path)
            self.assertEqual([row["value_text"] for row in active], ["in_term"])

    def test_character_action_completion_is_monotonic_and_undo_restores_start(self):
        from tsuzuri.character_action_ledger import (
            archive_actions_for_source_message_ids,
            commit_turn_action_updates,
            load_active_actions,
        )

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "actions.db")

            def card(assistant_id, transition, offset):
                return {
                    "audience_scope": "developer",
                    "character": "王五",
                    "user_id": f"u-{assistant_id}",
                    "assistant_id": assistant_id,
                    "turn_id": f"t-{assistant_id}",
                    "created_at_ms": now + offset,
                    "character_action_updates": [{
                        "transition": transition,
                        "event_text": "买面包",
                        "evidence_span": f"{transition} 买面包",
                        "source_at_ms": now + offset,
                        "valid_until_ms": now + 24 * 60 * 60 * 1000,
                        "confidence": 1.0,
                    }],
                }

            commit_turn_action_updates("王五", "sid", card("a-start", "start", 0), db_path=db_path)
            commit_turn_action_updates("王五", "sid", card("a-complete", "complete", 1_000), db_path=db_path)
            rejected = commit_turn_action_updates(
                "王五", "sid", card("a-reopen", "start", 2_000), db_path=db_path
            )
            self.assertEqual(rejected, [])
            self.assertEqual(load_active_actions("王五", now_ms=now + 3_000, db_path=db_path)[0]["transition"], "complete")

            archive_actions_for_source_message_ids(["a-complete"], db_path=db_path)
            restored = load_active_actions("王五", now_ms=now + 4_000, db_path=db_path)
            self.assertEqual(restored[0]["transition"], "start")

    def test_recent_dialogue_keeps_bread_completion_through_seven_short_turns(self):
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window

        history = []
        base = 1_721_000_000_000
        turns = [
            ("想你了会怼我吗", "买完面包了，你等一下别乱跑。"),
            ("我在房间能跑哪去", "刚多买了个面包，要不要。"),
            ("隔着屏幕吃不到", "知道，就随口一问。"),
            ("老家餐馆很多", "那还挺方便。"),
            ("长高一点更好", "打球又不看身高。"),
            ("长高更好看嘛", "我现在这样就行。"),
        ]
        for idx, (user, assistant) in enumerate(turns):
            history.append(self._entry("user", user, f"u-{idx}", base + idx * 2_000))
            history.append(self._entry("assistant", assistant, f"a-{idx}", base + idx * 2_000 + 1_000))
        history.append(self._entry("user", "那你现在打算干嘛", "u-now", base + 13_000))

        rendered = render_recent_dialogue_window(history, current_character="王五")
        merged = "\n".join(row["content"] for row in rendered["messages"])

        self.assertIn("买完面包了", merged)
        self.assertIn("我现在这样就行", merged)
        self.assertEqual(rendered["messages"][-1]["role"], "user")
        self.assertLessEqual(rendered["raw_chars"], 6000)

    pass  # upstream-only test removed


    def test_planned_travel_does_not_promote_without_departure_evidence(self):
        from tsuzuri.interlocutor_state_ledger import commit_turn_state_updates, load_active_states

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            card = {
                "audience_scope": "developer",
                "character": "王五",
                "user_id": "u-plan",
                "turn_id": "t-plan",
                "created_at_ms": now,
                "developer_state_updates": [{
                    "predicate": "travel.status",
                    "value": "去大阪看演唱会",
                    "operation": "set",
                    "temporal_status": "planned",
                    "valid_from_ms": now + 2_000,
                    "valid_until_ms": now + 30 * 24 * 60 * 60 * 1000,
                    "evidence_span": "我下周去大阪看演唱会",
                    "confidence": 1.0,
                }],
            }
            commit_turn_state_updates("王五", "sid", card, db_path=db_path)
            before = load_active_states("王五", now_ms=now + 1_000, db_path=db_path)
            self.assertEqual(before, [])
            after = load_active_states("王五", now_ms=now + 3_000, db_path=db_path)
            self.assertEqual(after, [])

    def test_planned_state_without_effective_date_stays_planned_but_is_visible(self):
        from tsuzuri.interlocutor_state_ledger import (
            commit_turn_state_updates,
            compile_interlocutor_state_frame,
            load_active_states,
            load_planned_states,
        )

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")
            card = {
                "audience_scope": "developer",
                "character": "王五",
                "user_id": "u-plan",
                "turn_id": "t-plan",
                "created_at_ms": now,
                "developer_state_updates": [{
                    "predicate": "travel.status",
                    "value": "去大阪看演唱会",
                    "operation": "set",
                    "temporal_status": "planned",
                    "valid_from_ms": now,
                    "valid_until_ms": now + 30 * 24 * 60 * 60 * 1000,
                    "evidence_span": "我下周去大阪看演唱会",
                    "confidence": 1.0,
                }],
            }
            commit_turn_state_updates("王五", "sid", card, db_path=db_path)
            active = load_active_states("王五", now_ms=now + 1_000, db_path=db_path)
            self.assertEqual(active, [])
            planned = load_planned_states("王五", now_ms=now + 1_000, db_path=db_path)
            self.assertEqual([row["value_text"] for row in planned], ["去大阪看演唱会"])
            frame = compile_interlocutor_state_frame("王五", {"active": active, "planned": planned})
            self.assertIn("尚未生效", frame["text"])
            self.assertIn("去大阪看演唱会", frame["text"])
            self.assertEqual(frame["record_count"], 0)
            self.assertEqual(frame["planned_count"], 1)

    def test_non_canonical_academic_value_is_rejected_at_normalize(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        user = self._entry("user", "我刚满20岁了，生日是上个月。", "u-age", 1_721_000_000_000)
        assistant = self._entry("assistant", "生日快乐。", "a-age", 1_721_000_001_000)
        card = normalize_turn_fact_card(
            {
                "fact": "阿澈刚满20岁。",
                "developer_state_updates": [{
                    "predicate": "academic.phase",
                    "value": "刚满20岁",
                    "operation": "set",
                    "temporal_status": "active",
                    "evidence_span": "我刚满20岁了",
                    "confidence": 0.9,
                }],
            },
            "王五",
            user,
            assistant,
            model="fake",
        )
        self.assertEqual(card["developer_state_updates"], [])

    def test_latin_only_free_text_value_is_rejected_at_normalize(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        user = self._entry("user", "我已经 recovered 了，不用担心。", "u-health", 1_721_000_000_000)
        assistant = self._entry("assistant", "那就好。", "a-health", 1_721_000_001_000)
        card = normalize_turn_fact_card(
            {
                "fact": "阿澈身体恢复了。",
                "developer_state_updates": [{
                    "predicate": "health.current",
                    "value": "recovered",
                    "operation": "set",
                    "temporal_status": "active",
                    "evidence_span": "我已经 recovered 了",
                    "confidence": 0.9,
                }],
            },
            "王五",
            user,
            assistant,
            model="fake",
        )
        self.assertEqual(card["developer_state_updates"], [])

    def test_older_commit_does_not_supersede_newer_state(self):
        from tsuzuri.interlocutor_state_ledger import commit_turn_state_updates, load_active_states

        now = int(time.time() * 1000)
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, "state.db")

            def card(user_id, value, source_ms):
                return {
                    "audience_scope": "developer",
                    "character": "王五",
                    "user_id": user_id,
                    "turn_id": "t-" + user_id,
                    "created_at_ms": source_ms,
                    "developer_state_updates": [{
                        "predicate": "academic.phase",
                        "value": value,
                        "operation": "set",
                        "temporal_status": "active",
                        "valid_from_ms": source_ms,
                        "valid_until_ms": source_ms + 30 * 24 * 60 * 60 * 1000,
                        "evidence_span": "明确学业阶段",
                        "confidence": 1.0,
                    }],
                }

            commit_turn_state_updates("王五", "sid", card("u-new", "in_term", now + 5_000), db_path=db_path)
            commit_turn_state_updates("王五", "sid", card("u-old", "summer_vacation", now), db_path=db_path)
            active = load_active_states("王五", now_ms=now + 6_000, db_path=db_path)
            self.assertEqual([row["value_text"] for row in active], ["in_term"])


if __name__ == "__main__":
    unittest.main()
