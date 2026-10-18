from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tsuzuri.daily_continuity.compactor import (
    CompactorConfig,
    DailyContinuityCompactor,
    build_extract_prompt,
    render_summary,
)
from tsuzuri.daily_continuity.contracts import DailyContinuityCapsule, SchemaValidationError
from tsuzuri.daily_continuity.segmenter import normalize_archive_history, segment_messages
from tsuzuri.daily_continuity.storage import (
    ContinuityStorage,
    compute_archive_sha256,
)
from tsuzuri.daily_continuity.temporal import derive_due_at_ms


def _message(role: str, content: str, msg_id: str, timestamp: int) -> dict:
    return {
        "role": role,
        "content": content,
        "id": msg_id,
        "timestamp": timestamp,
        "audience_scope": "developer",
    }


def _archive(messages: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "session_id": "sid-daily-v2",
        "character": "王五",
        "date_world": "2026-07-23",
        "history": messages,
    }


def _model_item(
    source_id: str,
    timestamp: int,
    *,
    kind: str = "user_fact",
    text: str = "阿澈在复习电路。",
    canonical_key: str = "user:阿澈:study_topic",
    actor: str = "user",
    due_text: str = "",
) -> dict:
    return {
        "kind": kind,
        "text": text,
        "subjects": ["阿澈"],
        "source_message_ids": [source_id],
        "source_at_ms": timestamp,
        "temporal_status": "active",
        "status": "active",
        "supersedes_source_message_ids": [],
        "canonical_key": canonical_key,
        "actor": actor,
        "due_text": due_text,
        "confidence": 0.91,
    }


class _SchemaAwareCaller:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if prompt.startswith("TASK=DAILY_CONTINUITY_V2_REDUCE"):
            candidates = re.findall(r'"candidate_id":"([^"]+)"', prompt)
            limit_match = re.search(r"最多选择 (\d+) 条", prompt)
            limit = int(limit_match.group(1)) if limit_match else 24
            return json.dumps({"selected_candidate_ids": candidates[:limit]})
        match = re.search(
            r"source_id=([^ ]+) timestamp_ms=(\d+)[^\]]*\]\s*([^：]+)：",
            prompt,
        )
        if not match:
            return json.dumps({"items": []})
        source_id, timestamp, speaker = match.group(1), int(match.group(2)), match.group(3)
        actor = "user" if speaker == "阿澈" else "character"
        return json.dumps(
            {
                "items": [
                    _model_item(
                        source_id,
                        timestamp,
                        kind="user_fact" if actor == "user" else "event",
                        text="阿澈在复习电路。" if actor == "user" else "王五回应了复习话题。",
                        canonical_key=f"{actor}:dialogue:study_topic",
                        actor=actor,
                    )
                ]
            },
            ensure_ascii=False,
        )


class ArchiveHashTests(unittest.TestCase):
    def test_archive_hash_is_canonical_and_content_sensitive(self):
        first = _archive([_message("user", "甲", "u-1", 1000)])
        reordered = {
            "history": first["history"],
            "date_world": first["date_world"],
            "character": first["character"],
            "session_id": first["session_id"],
            "schema_version": first["schema_version"],
        }
        changed = _archive([_message("user", "乙", "u-1", 1000)])
        self.assertEqual(compute_archive_sha256(first), compute_archive_sha256(reordered))
        self.assertNotEqual(compute_archive_sha256(first), compute_archive_sha256(changed))


class TemporalResolutionTests(unittest.TestCase):
    def test_late_night_logical_day_tomorrow_uses_source_calendar_date(self):
        source_at_ms = int(
            datetime.fromisoformat("2026-07-25T01:00:00+09:00").timestamp() * 1000
        )
        result = derive_due_at_ms(
            "明天十点",
            date_world="2026-07-24",
            source_at_ms=source_at_ms,
        )
        expected = int(
            datetime.fromisoformat("2026-07-26T10:00:00+09:00").timestamp() * 1000
        )
        self.assertEqual(result.due_at_ms, expected)

    def test_next_week_monday_from_friday_is_three_days_later(self):
        source_at_ms = int(
            datetime.fromisoformat("2026-07-24T12:00:00+09:00").timestamp() * 1000
        )
        result = derive_due_at_ms(
            "下周一",
            date_world="2026-07-24",
            source_at_ms=source_at_ms,
        )
        expected = int(
            datetime.fromisoformat("2026-07-27T09:00:00+09:00").timestamp() * 1000
        )
        self.assertEqual(result.due_at_ms, expected)

    def test_period_words_resolve_to_default_times_on_source_day(self):
        """2026-09-11：无钟点时段词按默认时点落成日期级 due（旧行为一律返 0）。"""
        source_at_ms = int(
            datetime.fromisoformat("2026-08-02T15:00:00+09:00").timestamp() * 1000
        )
        cases = {
            "今晚": "2026-08-02T20:00:00+09:00",
            "今夜": "2026-08-02T20:00:00+09:00",
            "晚上": "2026-08-02T20:00:00+09:00",
            "傍晚": "2026-08-02T18:00:00+09:00",
            "下午": "2026-08-02T15:00:00+09:00",
            "中午": "2026-08-02T12:00:00+09:00",
            "上午": "2026-08-02T10:00:00+09:00",
            "早晨": "2026-08-02T08:00:00+09:00",
            "早上": "2026-08-02T08:00:00+09:00",
            "凌晨": "2026-08-02T00:30:00+09:00",
            "放学": "2026-08-02T16:00:00+09:00",
        }
        for due_text, expected_iso in cases.items():
            with self.subTest(due_text=due_text):
                result = derive_due_at_ms(
                    due_text,
                    date_world="2026-08-02",
                    source_at_ms=source_at_ms,
                )
                self.assertEqual(
                    result.due_at_ms,
                    int(datetime.fromisoformat(expected_iso).timestamp() * 1000),
                )
                self.assertEqual(result.precision, "date")

    def test_anchored_period_words_shift_day_and_keep_default_time(self):
        source_at_ms = int(
            datetime.fromisoformat("2026-08-02T15:00:00+09:00").timestamp() * 1000
        )
        cases = {
            "明晚": "2026-08-03T20:00:00+09:00",
            "明早": "2026-08-03T08:00:00+09:00",
            "明天早上": "2026-08-03T08:00:00+09:00",
            "明天": "2026-08-03T09:00:00+09:00",
        }
        for due_text, expected_iso in cases.items():
            with self.subTest(due_text=due_text):
                result = derive_due_at_ms(
                    due_text,
                    date_world="2026-08-02",
                    source_at_ms=source_at_ms,
                )
                self.assertEqual(
                    result.due_at_ms,
                    int(datetime.fromisoformat(expected_iso).timestamp() * 1000),
                )
                self.assertEqual(result.precision, "date")

    def test_clock_expression_keeps_datetime_precision(self):
        source_at_ms = int(
            datetime.fromisoformat("2026-08-02T15:00:00+09:00").timestamp() * 1000
        )
        result = derive_due_at_ms(
            "明天十点半",
            date_world="2026-08-02",
            source_at_ms=source_at_ms,
        )
        self.assertEqual(
            result.due_at_ms,
            int(datetime.fromisoformat("2026-08-03T10:30:00+09:00").timestamp() * 1000),
        )
        self.assertEqual(result.precision, "datetime")

    def test_anchorless_expressions_stay_unresolved(self):
        """无锚点说法不得被推算成具体日期/时刻。"""
        source_at_ms = int(
            datetime.fromisoformat("2026-08-02T15:00:00+09:00").timestamp() * 1000
        )
        for due_text in ("以后", "有空再说", "将来", "改天", ""):
            with self.subTest(due_text=due_text):
                result = derive_due_at_ms(
                    due_text,
                    date_world="2026-08-02",
                    source_at_ms=source_at_ms,
                )
                self.assertEqual(result.due_at_ms, 0)
                self.assertEqual(result.precision, "unknown")


class SegmenterTests(unittest.TestCase):
    def test_long_day_keeps_every_source_id_in_at_least_one_segment(self):
        history = [
            _message("user" if index % 2 == 0 else "assistant", f"第{index}条" + "甲" * 180, f"m-{index}", index * 1000)
            for index in range(180)
        ]
        messages = normalize_archive_history(history)
        segments = segment_messages(messages, max_chars=1000, overlap_messages=1)
        covered = {source_id for segment in segments for source_id in segment.source_message_ids}
        self.assertEqual(covered, {f"m-{index}" for index in range(180)})
        self.assertIn("m-0", covered)
        self.assertIn("m-179", covered)
        self.assertGreater(len(segments), 1)

    def test_oversized_single_message_is_fragmented_without_losing_text(self):
        content = "长" * 3000
        messages = normalize_archive_history([_message("user", content, "u-long", 1000)])
        segments = segment_messages(messages, max_chars=700, overlap_messages=0)
        joined = "".join(fragment.content for segment in segments for fragment in segment.messages)
        self.assertEqual(joined, content)
        self.assertTrue(all(segment.char_count <= 700 for segment in segments))

    def test_guest_and_low_signal_rows_are_excluded(self):
        rows = [
            {**_message("user", "访客秘密", "g-1", 1), "audience_scope": "guest:x"},
            _message("assistant", "嗯。", "a-low", 2),
            _message("user", "开发者事实", "u-dev", 3),
        ]
        messages = normalize_archive_history(rows)
        self.assertEqual([row.source_message_id for row in messages], ["u-dev"])


class CompactorTests(unittest.TestCase):
    def test_extract_prompt_makes_user_fact_actor_boundary_explicit(self):
        messages = normalize_archive_history(
            [_message("user", "我还在复习电路。", "u-1", 1000)]
        )
        segment = segment_messages(messages, max_chars=8000)[0]

        prompt = build_extract_prompt(
            character="王五",
            date_world="2026-07-24",
            segment=segment,
            max_items=24,
            whole_day=True,
        )

        self.assertIn("kind=user_fact 时 actor 必须是 user", prompt)
        self.assertIn("角色自身事实请用 event 或 relation_delta", prompt)
        self.assertIn("未否认/没有反驳", prompt)
        self.assertIn("必须同时引用阿澈与角色的直接证据", prompt)
        self.assertIn("不是 X，是 Y", prompt)
        self.assertIn("即使错误理解来自角色提问或推测", prompt)
        self.assertIn("不能改写成“阿澈知晓该行程”", prompt)

    def test_small_day_uses_exactly_one_model_call(self):
        caller = _SchemaAwareCaller()
        archive = _archive(
            [
                _message("user", "我还在复习电路。", "u-1", 1000),
                _message("assistant", "知道了。", "a-1", 2000),
            ]
        )
        result = DailyContinuityCompactor().compact_archive(archive, llm_caller=caller, now_ms=10)
        self.assertTrue(result.ok)
        self.assertEqual(result.capsule.quality["mode"], "single_pass")
        self.assertEqual(result.capsule.quality["model_call_count"], 1)
        self.assertEqual(len(caller.prompts), 1)
        self.assertEqual(result.capsule.items[0].source_message_ids, ("u-1",))
        self.assertEqual(len(result.capsule.segments), 1)
        self.assertEqual(result.capsule.segments[0].segment_id, "whole-day")
        self.assertEqual(result.capsule.segments[0].map_status, "validated")
        self.assertEqual(
            result.capsule.segments[0].map_item_ids,
            (result.capsule.items[0].item_id,),
        )

    def test_relative_due_is_derived_locally_from_archive_day(self):
        timestamp = int(
            datetime.fromisoformat("2026-07-24T12:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("user", "我明天十点半复习电路。", "u-due", timestamp)]
        )
        archive["date_world"] = "2026-07-24"

        def caller(_prompt: str) -> str:
            return json.dumps(
                {
                    "items": [
                        _model_item(
                            "u-due",
                            timestamp,
                            kind="commitment",
                            text="阿澈计划复习电路。",
                            canonical_key="user:阿澈:study_circuit",
                            due_text="明天十点半",
                        )
                    ]
                },
                ensure_ascii=False,
            )

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        expected = int(
            datetime.fromisoformat("2026-07-25T10:30:00+09:00").timestamp() * 1000
        )
        self.assertEqual(result.capsule.items[0].due_at_ms, expected)
        self.assertEqual(result.capsule.items[0].due_precision, "datetime")

    def test_correction_actor_must_match_source_at_speaker(self):
        timestamp = int(
            datetime.fromisoformat("2026-07-24T12:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("user", "我其实没打算谈恋爱。", "u-correction", timestamp)]
        )

        def caller(_prompt: str) -> str:
            return json.dumps(
                {
                    "items": [
                        _model_item(
                            "u-correction",
                            timestamp,
                            kind="correction",
                            text="阿澈没有谈恋爱的打算，先前说法是乌龙。",
                            canonical_key="character:阿澈:relationship_intent",
                            actor="character",
                        )
                    ]
                },
                ensure_ascii=False,
            )

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        self.assertFalse(result.ok)
        self.assertIn("actor conflicts with source_at speaker role", result.reason)

    def test_single_invalid_item_is_dropped_without_failing_batch(self):
        # 2026-09-06: 一条 canonical_key 域违规的 item 曾让整个 session×day 的
        # capsule 报废并白烧 5 次退避重试。现在应只丢弃坏条目、保留好条目。
        timestamp = int(
            datetime.fromisoformat("2026-07-24T12:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("user", "我明天十点半复习电路。", "u-due", timestamp)]
        )
        archive["date_world"] = "2026-07-24"

        def caller(_prompt: str) -> str:
            good = _model_item("u-due", timestamp)
            bad = _model_item(
                "u-due",
                timestamp,
                text="阿澈提到王五的集训安排。",
                canonical_key="character:王五:session",  # domain 与 actor=user 不匹配
                actor="user",
            )
            return json.dumps({"items": [good, bad]}, ensure_ascii=False)

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(result.capsule.items), 1)
        self.assertEqual(result.capsule.items[0].canonical_key, "user:阿澈:study_topic")

    def test_all_invalid_items_still_fail_closed(self):
        # 全部条目违规 = 模型系统性违约，保持 fail-closed 走重试/隔离，
        # 不写出"看起来正常"的空 capsule。
        timestamp = int(
            datetime.fromisoformat("2026-07-24T12:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("user", "我明天十点半复习电路。", "u-due", timestamp)]
        )
        archive["date_world"] = "2026-07-24"

        def caller(_prompt: str) -> str:
            return json.dumps(
                {
                    "items": [
                        _model_item(
                            "u-due",
                            timestamp,
                            canonical_key="character:王五:session",
                            actor="user",
                        )
                    ]
                },
                ensure_ascii=False,
            )

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        self.assertFalse(result.ok)
        self.assertIn("canonical_key domain must match actor", result.reason)

    def test_shared_plan_requires_direct_evidence_from_both_speakers(self):
        user_at = int(
            datetime.fromisoformat("2026-07-24T23:00:00+09:00").timestamp() * 1000
        )
        assistant_at = user_at + 1000
        archive = _archive(
            [
                _message("user", "祝你明天花火大会玩得开心。", "u-wish", user_at),
                _message("assistant", "嗯，你也早点休息。", "a-reply", assistant_at),
            ]
        )

        def caller(_prompt: str) -> str:
            item = _model_item(
                "u-wish",
                user_at,
                kind="event",
                text="阿澈和王五计划明天一起参加花火大会。",
                canonical_key="shared:花火大会:日程",
                actor="shared",
                due_text="明天",
            )
            item["temporal_status"] = "planned"
            return json.dumps({"items": [item]}, ensure_ascii=False)

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=assistant_at
        )

        self.assertFalse(result.ok)
        self.assertIn("shared actor requires direct evidence from both speakers", result.reason)

    def test_inference_markers_are_rejected_even_when_actor_matches(self):
        timestamp = int(
            datetime.fromisoformat("2026-07-24T23:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("user", "祝你明天花火大会玩得开心。", "u-wish", timestamp)]
        )

        def caller(_prompt: str) -> str:
            item = _model_item(
                "u-wish",
                timestamp,
                kind="event",
                text="阿澈的祝福暗示他知晓王五明天有花火大会行程。",
                canonical_key="user:阿澈:knowledge_of_fireworks",
                actor="user",
                due_text="明天",
            )
            item["temporal_status"] = "planned"
            return json.dumps({"items": [item]}, ensure_ascii=False)

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        self.assertFalse(result.ok)
        self.assertIn("item text relies on unsupported indirect inference", result.reason)

    def test_direct_relationship_hint_is_not_treated_as_schedule_inference(self):
        timestamp = int(
            datetime.fromisoformat("2026-07-24T23:00:00+09:00").timestamp() * 1000
        )
        archive = _archive(
            [_message("assistant", "还有一点点别的，你自己猜吧。", "a-hint", timestamp)]
        )

        def caller(_prompt: str) -> str:
            return json.dumps(
                {
                    "items": [
                        _model_item(
                            "a-hint",
                            timestamp,
                            kind="relation_delta",
                            text="王五暗示还有一点别的感情，让阿澈自己猜。",
                            canonical_key="character:王五:relationship_hint",
                            actor="character",
                        )
                    ]
                },
                ensure_ascii=False,
            )

        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=timestamp
        )

        self.assertTrue(result.ok, result.reason)

    def test_long_day_runs_map_then_selector_reduce(self):
        caller = _SchemaAwareCaller()
        archive = _archive(
            [
                _message("user" if index % 2 == 0 else "assistant", "内容" + str(index) + "甲" * 180, f"m-{index}", 1000 + index)
                for index in range(20)
            ]
        )
        compactor = DailyContinuityCompactor(
            config=CompactorConfig(
                small_day_char_threshold=500,
                segment_char_budget=700,
                overlap_messages=1,
                max_items_per_segment=4,
                max_final_items=24,
                reduce_batch_size=32,
                reduce_batch_target=12,
            )
        )
        result = compactor.compact_archive(archive, llm_caller=caller, now_ms=10)
        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.capsule.quality["mode"], "map_reduce")
        self.assertGreater(result.capsule.quality["segment_count"], 1)
        self.assertTrue(caller.prompts[-1].startswith("TASK=DAILY_CONTINUITY_V2_REDUCE"))
        self.assertGreater(result.capsule.quality["model_call_count"], result.capsule.quality["segment_count"])
        self.assertIn("m-0", result.capsule.source_message_ids)
        self.assertEqual(len(result.capsule.segments), result.capsule.quality["segment_count"])
        covered = {
            source_id
            for segment in result.capsule.segments
            for source_id in segment.source_message_ids
        }
        self.assertEqual(covered, set(result.capsule.source_message_ids))
        self.assertTrue(all(segment.map_status == "validated" for segment in result.capsule.segments))

    def test_complete_turns_keep_first_middle_and_last_provenance_through_compactor(self):
        for turn_count in (151, 300, 1000):
            messages: list[dict] = []
            for index in range(turn_count):
                messages.extend(
                    [
                        _message(
                            "user",
                            f"第{index}轮用户事实" + "甲" * 24,
                            f"u-{index}",
                            1_000 + index * 2,
                        ),
                        _message(
                            "assistant",
                            f"第{index}轮角色回应" + "乙" * 24,
                            f"a-{index}",
                            1_001 + index * 2,
                        ),
                    ]
                )
            caller = _SchemaAwareCaller()
            result = DailyContinuityCompactor(
                config=CompactorConfig(
                    small_day_char_threshold=500,
                    segment_char_budget=4_000,
                    overlap_messages=1,
                    max_items_per_segment=4,
                    max_final_items=24,
                    reduce_batch_size=32,
                    reduce_batch_target=12,
                )
            ).compact_archive(_archive(messages), llm_caller=caller, now_ms=10)

            self.assertTrue(result.ok, result.reason)
            capsule = result.capsule
            self.assertEqual(capsule.source_message_count, turn_count * 2)
            covered = {
                source_id
                for segment in capsule.segments
                for source_id in segment.source_message_ids
            }
            self.assertEqual(covered, set(capsule.source_message_ids))
            self.assertIn("u-0", covered)
            self.assertIn(f"u-{turn_count // 2}", covered)
            self.assertIn(f"a-{turn_count - 1}", covered)
            self.assertTrue(capsule.quality["all_input_processed"])
            self.assertEqual(capsule.quality["mode"], "map_reduce")

    def test_retry_reuses_only_validated_map_segments(self):
        archive = _archive(
            [
                _message(
                    "user" if index % 2 == 0 else "assistant",
                    f"第{index}条" + "甲" * 180,
                    f"m-{index}",
                    1_000 + index,
                )
                for index in range(24)
            ]
        )
        config = CompactorConfig(
            small_day_char_threshold=500,
            segment_char_budget=700,
            overlap_messages=1,
            max_items_per_segment=4,
            max_final_items=24,
            reduce_batch_size=32,
            reduce_batch_target=12,
        )
        first_base = _SchemaAwareCaller()
        extract_calls = 0

        def fail_third_map(prompt: str) -> str:
            nonlocal extract_calls
            if prompt.startswith("TASK=DAILY_CONTINUITY_V2_EXTRACT"):
                extract_calls += 1
                if extract_calls == 3:
                    raise RuntimeError("transient third segment failure")
            return first_base(prompt)

        with tempfile.TemporaryDirectory() as temp_dir:
            capsule_path = Path(temp_dir) / "sid.continuity.v2.json"
            first = DailyContinuityCompactor(config=config).compact_archive(
                archive,
                llm_caller=fail_third_map,
                capsule_path=capsule_path,
                now_ms=1_000,
            )
            checkpoint_path = Path(str(capsule_path) + ".checkpoint.json")
            self.assertEqual(first.status, "retryable_failure")
            self.assertTrue(checkpoint_path.exists())

            second_caller = _SchemaAwareCaller()
            second = DailyContinuityCompactor(config=config).compact_archive(
                archive,
                llm_caller=second_caller,
                capsule_path=capsule_path,
                now_ms=2_000,
            )

            self.assertTrue(second.ok, second.reason)
            self.assertEqual(second.capsule.quality["checkpoint_reused_segment_count"], 2)
            second_extract_calls = sum(
                prompt.startswith("TASK=DAILY_CONTINUITY_V2_EXTRACT")
                for prompt in second_caller.prompts
            )
            self.assertEqual(
                second_extract_calls,
                second.capsule.quality["segment_count"] - 2,
            )
            self.assertFalse(checkpoint_path.exists())

    def test_unknown_provenance_becomes_retryable_failure(self):
        archive = _archive([_message("user", "我在复习。", "u-1", 1000)])

        def bad_caller(_prompt: str) -> str:
            return json.dumps({"items": [_model_item("unknown", 1000)]}, ensure_ascii=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            capsule_path = Path(temp_dir) / "sid.continuity.v2.json"
            result = DailyContinuityCompactor().compact_archive(
                archive, llm_caller=bad_caller, capsule_path=capsule_path, now_ms=1000
            )
            retry_path = Path(str(capsule_path) + ".retry.json")
            self.assertEqual(result.status, "retryable_failure")
            self.assertFalse(capsule_path.exists())
            self.assertTrue(retry_path.exists())
            self.assertEqual(result.retry_state["attempts"], 1)
            self.assertGreater(result.retry_state["next_retry_at_ms"], 1000)

    def test_matching_archive_sha_is_idempotent(self):
        archive = _archive([_message("user", "我在复习。", "u-1", 1000)])
        caller = _SchemaAwareCaller()
        with tempfile.TemporaryDirectory() as temp_dir:
            capsule_path = Path(temp_dir) / "sid.continuity.v2.json"
            compactor = DailyContinuityCompactor()
            first = compactor.compact_archive(
                archive, llm_caller=caller, capsule_path=capsule_path, now_ms=1000
            )
            second = compactor.compact_archive(
                archive,
                llm_caller=lambda _prompt: (_ for _ in ()).throw(AssertionError("must not call")),
                capsule_path=capsule_path,
                now_ms=2000,
            )
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "idempotent")
        self.assertEqual(first.capsule.source_archive_sha256, second.capsule.source_archive_sha256)

    def test_empty_day_completes_without_model_call(self):
        result = DailyContinuityCompactor().compact_archive(
            _archive([]),
            llm_caller=lambda _prompt: (_ for _ in ()).throw(AssertionError("must not call")),
            now_ms=1000,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.capsule.quality["mode"], "empty")
        self.assertEqual(result.capsule.items, ())

    def test_rendered_summary_is_bounded_and_non_authoritative(self):
        archive = _archive([_message("user", "地点改成冲绳了。", "u-1", 1000)])

        def caller(_prompt: str) -> str:
            item = _model_item("u-1", 1000, kind="correction", text="阿澈将花展地点纠正为冲绳。")
            return json.dumps({"items": [item]}, ensure_ascii=False)

        result = DailyContinuityCompactor(
            config=CompactorConfig(rendered_summary_max_chars=120)
        ).compact_archive(archive, llm_caller=caller, now_ms=1000)
        self.assertLessEqual(len(result.capsule.rendered_summary), 120)
        self.assertIn("纠正", result.capsule.rendered_summary)
        self.assertFalse(result.capsule.rendered_summary_authoritative)
        payload = result.capsule.to_dict()
        self.assertIn("segments", payload)
        self.assertEqual(payload["items"][0]["ledger_refs"], [])
        self.assertEqual(payload["items"][0]["memory_refs"], [])
        payload["rendered_summary_authoritative"] = True
        with self.assertRaises(SchemaValidationError):
            DailyContinuityCapsule.from_dict(payload)

    def test_core_1_payload_without_segments_or_coverage_refs_is_read_compatible(self):
        archive = _archive([_message("user", "我在复习。", "u-1", 1000)])
        caller = _SchemaAwareCaller()
        result = DailyContinuityCompactor().compact_archive(
            archive, llm_caller=caller, now_ms=1000
        )
        payload = result.capsule.to_dict()
        payload["generator_version"] = "daily_continuity_v2_core_1"
        payload.pop("segments")
        for item in payload["items"]:
            item.pop("ledger_refs")
            item.pop("memory_refs")

        restored = DailyContinuityCapsule.from_dict(payload)

        self.assertEqual(restored.segments[0].map_status, "legacy_untracked")
        self.assertEqual(restored.segments[0].source_message_ids, ("u-1",))
        self.assertEqual(restored.items[0].ledger_refs, ())
        self.assertEqual(restored.items[0].memory_refs, ())

    def test_model_output_must_match_exact_schema(self):
        archive = _archive([_message("user", "我在复习。", "u-1", 1000)])
        item = _model_item("u-1", 1000)
        item["unexpected"] = True
        result = DailyContinuityCompactor().compact_archive(
            archive,
            llm_caller=lambda _prompt: json.dumps({"items": [item]}, ensure_ascii=False),
            now_ms=1000,
        )
        self.assertEqual(result.status, "retryable_failure")
        self.assertIn("fields mismatch", result.reason)


class RetryStorageTests(unittest.TestCase):
    def test_failures_increment_and_success_clears_due_time(self):
        storage = ContinuityStorage()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "retry.json"
            first = storage.record_retryable_failure(
                path, archive_sha256="a" * 64, error="first", now_ms=1000
            )
            second = storage.record_retryable_failure(
                path, archive_sha256="a" * 64, error="second", now_ms=2000
            )
            self.assertEqual(first["attempts"], 1)
            self.assertEqual(second["attempts"], 2)
            self.assertFalse(storage.retry_is_due(path, now_ms=2001))
            success = storage.record_success(path, archive_sha256="a" * 64, now_ms=3000)
            self.assertEqual(success["status"], "succeeded")
            self.assertEqual(success["next_retry_at_ms"], 0)
            self.assertTrue(storage.retry_is_due(path, now_ms=3001))

    def test_new_archive_sha_bypasses_old_backoff(self):
        storage = ContinuityStorage()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "retry.json"
            storage.record_retryable_failure(
                path,
                archive_sha256="a" * 64,
                error="old archive failed",
                now_ms=1_000,
            )

            self.assertFalse(
                storage.retry_is_due(
                    path,
                    now_ms=1_001,
                    archive_sha256="a" * 64,
                )
            )
            self.assertTrue(
                storage.retry_is_due(
                    path,
                    now_ms=1_001,
                    archive_sha256="b" * 64,
                )
            )

    def test_repeated_schema_failures_quarantine_same_archive(self):
        storage = ContinuityStorage()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "retry.json"
            state = {}
            for attempt in range(5):
                state = storage.record_retryable_failure(
                    path,
                    archive_sha256="a" * 64,
                    error="schema mismatch",
                    now_ms=1_000 + attempt,
                )

            self.assertEqual(state["status"], "quarantined")
            self.assertEqual(state["attempts"], 5)
            self.assertFalse(
                storage.retry_is_due(
                    path,
                    now_ms=9_999_999,
                    archive_sha256="a" * 64,
                )
            )


if __name__ == "__main__":
    unittest.main()
