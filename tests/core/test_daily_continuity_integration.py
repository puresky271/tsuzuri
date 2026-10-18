from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock


def _archive_payload(session_id: str = "sid", date_world: str = "2026-07-24") -> dict:
    return {
        "schema_version": 1,
        "session_id": session_id,
        "character": "王五",
        "date_world": date_world,
        "history": [
            {
                "id": "u-1",
                "role": "user",
                "content": "花展地点改成冲绳了。",
                "timestamp": 1_774_560_000_000,
            },
            {
                "id": "a-1",
                "role": "assistant",
                "content": "知道了，我不会再按关东那条说。",
                "timestamp": 1_774_560_060_000,
                "character": "王五",
            },
        ],
    }


def _valid_model_output() -> str:
    return json.dumps(
        {
            "items": [
                {
                    "kind": "correction",
                    "text": "阿澈把花展地点更正为冲绳。",
                    "subjects": ["阿澈", "王五"],
                    "source_message_ids": ["u-1"],
                    "source_at_ms": 1_774_560_000_000,
                    "temporal_status": "active",
                    "status": "active",
                    "supersedes_source_message_ids": [],
                    "canonical_key": "user:花展:location",
                    "actor": "user",
                    "due_text": "",
                    "confidence": 0.96,
                }
            ]
        },
        ensure_ascii=False,
    )


class DailyContinuityFacadeTests(unittest.TestCase):
    def test_recap_default_today_uses_0230_logical_day(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with mock.patch.object(
            chat_compact,
            "_now_world",
            return_value=datetime.fromisoformat("2026-07-25T01:00:00+09:00"),
        ):
            self.assertEqual(chat_compact._today_world_date_str(), "2026-07-24")

    def test_legacy_recap_truncation_preserves_later_fact_and_commitment_fields(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        summary = "\n".join(
            [
                "1. 重要事件：" + "事件" * 40,
                "2. 关系状态变化：" + "关系" * 40,
                "3. ongoing 话题：" + "待续" * 40,
                "4. 阿澈透露的事实：" + "事实" * 40,
                "5. 王五的情绪 / 承诺：" + "承诺" * 40,
            ]
        )

        truncated = chat_compact._truncate_recap_summary(summary, 80)

        self.assertLessEqual(len(truncated), 80)
        self.assertIn("4.阿澈事实：", truncated)
        self.assertIn("5.情绪/承诺：", truncated)

    def test_default_recap_budgets_expand_recent_week_and_older_fragments(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        summary = "\n".join([
            "1.事件：" + "摄影展路线与当日事件" * 16,
            "2.关系：" + "自然交流没有明确变化" * 12,
            "3.待续：" + "继续确认路线与照片进度" * 14,
            "4.阿澈事实：" + "时间安排兴趣偏好生活状态" * 16,
            "5.情绪/承诺：" + "轻松关心并作出有限承诺" * 16,
        ])

        def load_fixture(_session_id: str, date_world: str, *, character: str = "") -> dict:
            return {"character": character, "date_world": date_world, "summary": summary}

        def section(text: str, start: str, end: str = "") -> str:
            body = text.split(start, 1)[1]
            return body.split(end, 1)[0] if end and end in body else body

        with mock.patch.object(
            chat_compact,
            "_load_recap_source_for_date",
            side_effect=load_fixture,
        ):
            legacy = chat_compact.load_historical_recap(
                "fixture",
                today_date_world="2026-08-15",
                character="张三",
                midrange_budget_per_day=160,
                weekly_budget=160,
            )
            expanded = chat_compact.load_historical_recap(
                "fixture",
                today_date_world="2026-08-15",
                character="张三",
            )

        recent_header = "━━ 最近一周的对话要点 ━━"
        older_header = "━━ 更早的对话碎片 ━━"
        legacy_recent = section(legacy, recent_header, older_header)
        expanded_recent = section(expanded, recent_header, older_header)
        legacy_older = section(legacy, older_header)
        expanded_older = section(expanded, older_header)
        self.assertGreater(len(expanded_recent), len(legacy_recent) + 3000)
        self.assertGreater(len(expanded_older), len(legacy_older) + 2500)
        self.assertIn("5.情绪/承诺：", expanded_recent)
        self.assertIn("[两周前]", expanded_older)

    def test_default_midrange_budget_keeps_long_five_field_summary_tail(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        summary = "\n".join([
            "1.重要事件：" + "事件细节" * 40,
            "2.关系状态变化：" + "关系细节" * 30,
            "3.ongoing 话题：" + "待续细节" * 40,
            "4.阿澈透露的事实：" + "用户事实" * 50,
            "5.张三的情绪/承诺：" + "情绪承诺" * 50 + "尾部完整证据",
        ])

        def load_fixture(_session_id: str, date_world: str, *, character: str = "") -> dict:
            return {"character": character, "date_world": date_world, "summary": summary}

        with mock.patch.object(
            chat_compact,
            "_load_recap_source_for_date",
            side_effect=load_fixture,
        ):
            recap = chat_compact.load_historical_recap(
                "fixture",
                today_date_world="2026-08-15",
                character="张三",
                exclude_dates={
                    "2026-08-14", "2026-08-12", "2026-08-11", "2026-08-10",
                    "2026-08-09", "2026-08-08", "2026-08-07", "2026-08-06",
                    "2026-08-05", "2026-08-04", "2026-08-03", "2026-08-02",
                    "2026-08-01",
                },
            )

        self.assertIn("· 2026-08-13（2 天前）", recap)
        self.assertIn("尾部完整证据", recap)
        self.assertNotIn("…", recap)

    def test_segment_coverage_keeps_first_middle_and_last_at_scale(self) -> None:
        from tsuzuri.daily_continuity.segmenter import normalize_archive_history, segment_messages

        for count in (151, 300, 1000):
            history = [
                {
                    "id": f"m-{index}",
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"第{index}条关键内容" + "甲" * 40,
                    "timestamp": 1_774_560_000_000 + index * 1000,
                }
                for index in range(count)
            ]
            segments = segment_messages(
                normalize_archive_history(history), max_chars=1200, overlap_messages=1
            )
            covered = {
                source_id
                for segment in segments
                for source_id in segment.source_message_ids
            }
            self.assertIn("m-0", covered)
            self.assertIn(f"m-{count // 2}", covered)
            self.assertIn(f"m-{count - 1}", covered)
            self.assertEqual(len(covered), count)

    def test_v2_sidecar_is_idempotent_and_coexists_with_fixed_daily_recap(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            archive = _archive_payload()
            (day_dir / "sid.json").write_text(
                json.dumps(archive, ensure_ascii=False), encoding="utf-8"
            )
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "旧 v1 冲绳摘要"}, ensure_ascii=False),
                encoding="utf-8",
            )
            env = {
                "DAILY_CONTINUITY_V2_ENABLED": "1",
                "DAILY_CONTINUITY_V2_READ_ENABLED": "1",
                "DAILY_CONTINUITY_V2_SHADOW_ONLY": "0",
            }
            caller = mock.Mock(return_value=_valid_model_output())
            with (
                mock.patch.object(chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)),
                mock.patch.dict(os.environ, env, clear=False),
            ):
                first = chat_compact.compact_session_for_date_v2(
                    "sid", "2026-07-24", llm_caller=caller
                )
                retry_path = day_dir / "sid.continuity.v2.json.retry.json"
                retry_path.write_text(
                    json.dumps(
                        {
                            "status": "retryable_failure",
                            "archive_sha256": first["archive_sha256"],
                            "attempts": 2,
                            "next_retry_at_ms": 9_999_999_999_999,
                        }
                    ),
                    encoding="utf-8",
                )
                second = chat_compact.compact_session_for_date_v2(
                    "sid", "2026-07-24", llm_caller=caller
                )
                bundle = chat_compact.load_historical_context_bundle(
                    "sid", today_date_world="2026-07-25"
                )

            self.assertTrue(first["ok"])
            self.assertEqual(first["reason"], "completed")
            self.assertTrue(second["ok"])
            self.assertEqual(second["reason"], "idempotent")
            self.assertEqual(caller.call_count, 1)
            self.assertEqual(
                json.loads(retry_path.read_text(encoding="utf-8"))["status"],
                "succeeded",
            )
            self.assertEqual(bundle["covered_dates"], ["2026-07-24"])
            self.assertEqual(len(bundle["capsules"]), 1)
            self.assertIn("旧 v1 冲绳摘要", bundle["legacy_recap"])

    def test_character_daily_recap_reduces_all_session_maps_and_becomes_fixed_source(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        reduced = """<summary>
1. 重要事件：阿澈更正了花展地点，并和王五确认复习安排。
2. 关系状态变化：无明显变化。
3. ongoing 话题：还要确认花展日期。
4. 阿澈透露的事实：阿澈周六复习电路。
5. 王五的情绪 / 承诺：王五答应周六提醒阿澈。
</summary>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            for sid, summary in (
                ("long", "1. 重要事件：花展地点不是关东，而是冲绳。"),
                ("daily", "5. 王五的情绪 / 承诺：答应周六提醒阿澈复习。"),
            ):
                archive = _archive_payload(session_id=sid)
                (day_dir / f"{sid}.json").write_text(
                    json.dumps(archive, ensure_ascii=False), encoding="utf-8"
                )
                (day_dir / f"{sid}.summary.json").write_text(
                    json.dumps({"summary": summary}, ensure_ascii=False), encoding="utf-8"
                )
            caller = mock.Mock(return_value=reduced)
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                result = chat_compact.compact_character_daily_recap(
                    "王五", "2026-07-24", llm_caller=caller
                )
                recap = chat_compact.load_historical_recap(
                    "long",
                    character="王五",
                    today_date_world="2026-07-25",
                )

            self.assertTrue(result["ok"])
            self.assertEqual(result["reason"], "ok")
            aggregate = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(aggregate["source_session_count"], 2)
            self.assertEqual(aggregate["source_summary_count"], 2)
            prompt = caller.call_args.args[0]
            self.assertIn("花展地点不是关东", prompt)
            self.assertIn("答应周六提醒", prompt)
            self.assertIn("阿澈更正了花展地点", recap)
            self.assertNotIn("花展地点不是关东，而是冲绳", recap)

    def test_character_daily_recap_repairs_interpretation_before_write(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        invalid = """<summary>
1. 重要事件：阿澈向王五说明花展地点。
2. 关系状态变化：双方进入边界模糊状态。
3. ongoing 话题：无。
4. 阿澈透露的事实：阿澈暗示自己会去花展。
5. 王五的情绪 / 承诺：无。
</summary>"""
        repaired = """<summary>
1. 重要事件：阿澈告诉王五花展地点改为冲绳。
2. 关系状态变化：无明确关系表达。
3. ongoing 话题：无。
4. 阿澈透露的事实：阿澈说花展地点改为冲绳。
5. 王五的情绪 / 承诺：无。
</summary>"""
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            (day_dir / "sid.json").write_text(
                json.dumps(_archive_payload(), ensure_ascii=False), encoding="utf-8"
            )
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "1. 重要事件：花展地点改为冲绳。"}, ensure_ascii=False),
                encoding="utf-8",
            )
            caller = mock.Mock(side_effect=[invalid, repaired])
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                result = chat_compact.compact_character_daily_recap(
                    "王五", "2026-07-24", llm_caller=caller
                )
                payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(caller.call_count, 2)
        self.assertEqual(payload["method"], "character_day_session_summary_reduce_repair_v1")
        self.assertEqual(payload["model_call_count"], 2)
        self.assertNotIn("边界模糊", payload["summary"])
        self.assertNotIn("暗示", payload["summary"])

    def test_character_daily_recap_writes_over_budget_summary_without_repair(self) -> None:
        """2026-09-11 决定：900 字是注入侧约束，超预算不再 hard fail。"""
        from tsuzuri import chat_compact as chat_compact

        over_budget = "\n".join(
            [
                "1. 重要事件：" + "事件" * 400,
                "2. 关系状态变化：无明显变化。",
                "3. ongoing 话题：无。",
                "4. 阿澈透露的事实：" + "事实" * 400,
                "5. 王五的情绪 / 承诺：无。",
            ]
        )
        self.assertGreater(len(over_budget), 900)
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            (day_dir / "sid.json").write_text(
                json.dumps(_archive_payload(), ensure_ascii=False), encoding="utf-8"
            )
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "1. 重要事件：花展地点改为冲绳。"}, ensure_ascii=False),
                encoding="utf-8",
            )
            caller = mock.Mock(return_value=f"<summary>\n{over_budget}\n</summary>")
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                result = chat_compact.compact_character_daily_recap(
                    "王五", "2026-07-24", llm_caller=caller
                )
            payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertTrue(result["over_budget"])
        self.assertEqual(caller.call_count, 1)
        self.assertEqual(payload["method"], "character_day_session_summary_reduce_v1")
        self.assertEqual(payload["model_call_count"], 1)
        self.assertTrue(payload["over_budget"])
        self.assertEqual(payload["summary_chars"], len(over_budget))
        # 落盘保留全文，截断交由注入侧 _truncate_recap_summary。
        self.assertEqual(payload["summary"], over_budget)
        self.assertEqual(payload["target_char_budget"], 900)

    def test_character_daily_recap_accepts_over_budget_after_repair(self) -> None:
        """解释性措辞仍需 repair，但 repair 后仍超预算时不再判失败。"""
        from tsuzuri import chat_compact as chat_compact

        invalid = "\n".join(
            [
                "1. 重要事件：" + "事件" * 400,
                "2. 关系状态变化：双方进入边界模糊状态。",
                "3. ongoing 话题：无。",
                "4. 阿澈透露的事实：" + "事实" * 400,
                "5. 王五的情绪 / 承诺：无。",
            ]
        )
        repaired = invalid.replace("双方进入边界模糊状态。", "无明确关系表达。")
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            (day_dir / "sid.json").write_text(
                json.dumps(_archive_payload(), ensure_ascii=False), encoding="utf-8"
            )
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "1. 重要事件：花展地点改为冲绳。"}, ensure_ascii=False),
                encoding="utf-8",
            )
            caller = mock.Mock(side_effect=[invalid, repaired])
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                result = chat_compact.compact_character_daily_recap(
                    "王五", "2026-07-24", llm_caller=caller
                )
            payload = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertTrue(result["over_budget"])
        self.assertEqual(caller.call_count, 2)
        self.assertEqual(payload["method"], "character_day_session_summary_reduce_repair_v1")
        self.assertEqual(payload["model_call_count"], 2)
        self.assertNotIn("边界模糊", payload["summary"])
        self.assertTrue(payload["over_budget"])

    def test_character_daily_recap_interpretive_language_still_fails_closed(self) -> None:
        """解释性措辞在 repair 后仍存在时必须 fail closed（不落盘）。"""
        from tsuzuri import chat_compact as chat_compact

        invalid = "\n".join(
            [
                "1. 重要事件：阿澈向王五说明花展地点。",
                "2. 关系状态变化：双方进入边界模糊状态。",
                "3. ongoing 话题：无。",
                "4. 阿澈透露的事实：阿澈暗示自己会去花展。",
                "5. 王五的情绪 / 承诺：无。",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            (day_dir / "sid.json").write_text(
                json.dumps(_archive_payload(), ensure_ascii=False), encoding="utf-8"
            )
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "1. 重要事件：花展地点改为冲绳。"}, ensure_ascii=False),
                encoding="utf-8",
            )
            caller = mock.Mock(return_value=f"<summary>\n{invalid}\n</summary>")
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                result = chat_compact.compact_character_daily_recap(
                    "王五", "2026-07-24", llm_caller=caller
                )
                recap_written = Path(result["summary_path"]).exists()

        self.assertFalse(recap_written)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "quality_validation_failed")
        self.assertTrue(
            any(
                issue.startswith("interpretive_language:")
                for issue in result.get("quality_issues", [])
            )
        )

    def test_secondary_relationship_labels_are_reserved_for_developer_impression(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        for label in ("未拒绝也未接受", "朋友以上", "边界模糊"):
            summary = f"""<summary>
1. 重要事件：无显著事件。
2. 关系状态变化：双方处于{label}的状态。
3. ongoing 话题：无。
4. 阿澈透露的事实：无。
5. 张三的情绪 / 承诺：无。
</summary>"""
            issues = chat_compact._daily_recap_quality_issues(
                summary,
                target_char_budget=900,
            )
            self.assertTrue(
                any(issue.startswith("interpretive_language:") for issue in issues),
                label,
            )

        self.assertIn(
            "只能归入开发者专属印象，由 KAIROS 每周关系反思维护",
            chat_compact._CHARACTER_DAILY_RECAP_PROMPT,
        )

    def test_character_bundle_reads_valid_capsule_from_another_session(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            archive = _archive_payload(session_id="old-session")
            (day_dir / "old-session.json").write_text(
                json.dumps(archive, ensure_ascii=False), encoding="utf-8"
            )
            env = {
                "DAILY_CONTINUITY_V2_ENABLED": "1",
                "DAILY_CONTINUITY_V2_READ_ENABLED": "1",
                "DAILY_CONTINUITY_V2_SHADOW_ONLY": "0",
            }
            with (
                mock.patch.object(
                    chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
                ),
                mock.patch.dict(os.environ, env, clear=False),
            ):
                generated = chat_compact.compact_session_for_date_v2(
                    "old-session",
                    "2026-07-24",
                    llm_caller=lambda _prompt: _valid_model_output(),
                )
                bundle = chat_compact.load_historical_context_bundle(
                    "new-session",
                    character="王五",
                    today_date_world="2026-07-25",
                )

        self.assertTrue(generated["ok"])
        self.assertEqual(len(bundle["capsules"]), 1)
        self.assertEqual(bundle["capsules"][0]["session_id"], "old-session")

    def test_invalid_or_missing_v2_keeps_v1_fallback(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            (day_dir / "sid.summary.json").write_text(
                json.dumps({"summary": "仅有 v1 的摘要"}, ensure_ascii=False),
                encoding="utf-8",
            )
            with (
                mock.patch.object(chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)),
                mock.patch.dict(
                    os.environ,
                    {
                        "DAILY_CONTINUITY_V2_ENABLED": "1",
                        "DAILY_CONTINUITY_V2_READ_ENABLED": "1",
                        "DAILY_CONTINUITY_V2_SHADOW_ONLY": "0",
                    },
                    clear=False,
                ),
            ):
                bundle = chat_compact.load_historical_context_bundle(
                    "sid", today_date_world="2026-07-25"
                )

            self.assertEqual(bundle["capsules"], [])
            self.assertIn("仅有 v1 的摘要", bundle["legacy_recap"])

    def test_v2_projection_defaults_on_but_keeps_explicit_rollback(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(chat_compact.daily_continuity_v2_enabled())
            self.assertTrue(chat_compact.daily_continuity_v2_projection_enabled())
        with mock.patch.dict(
            os.environ,
            {"DAILY_CONTINUITY_V2_READ_ENABLED": "0"},
            clear=True,
        ):
            self.assertFalse(chat_compact.daily_continuity_v2_projection_enabled())

    def test_retry_scan_ignores_success_state_and_returns_due_failure(self) -> None:
        from tsuzuri import chat_compact as chat_compact

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_root = Path(temp_dir) / "archive"
            day_dir = archive_root / "2026-07-24"
            day_dir.mkdir(parents=True)
            for sid, status in (("done", "succeeded"), ("retry", "retryable_failure")):
                archive = _archive_payload(session_id=sid)
                (day_dir / f"{sid}.json").write_text(
                    json.dumps(archive, ensure_ascii=False), encoding="utf-8"
                )
                (day_dir / f"{sid}.continuity.v2.json.retry.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "next_retry_at_ms": 0,
                            "archive_sha256": sid,
                        }
                    ),
                    encoding="utf-8",
                )
            with mock.patch.object(
                chat_compact, "_CHAT_SESSIONS_ARCHIVE_DIR", str(archive_root)
            ):
                jobs = chat_compact.list_due_daily_continuity_retries(now_ms=10)

        self.assertEqual([job["session_id"] for job in jobs], ["retry"])

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed


class DailyContinuityWorkspaceTests(unittest.TestCase):
    pass  # upstream-only test removed

    def test_two_continuity_pins_survive_dense_turn_fact_competition(self) -> None:
        from tsuzuri.turn_context_workspace import compile_fact_workspace

        cards = [
            {
                "turn_id": f"turn-{index}",
                "created_at_ms": 1_774_600_000_000 + index,
                "corrections": [f"花展候选规则第{index}条已经更正。"],
                "confidence": 0.95,
                "salience": 0.9,
            }
            for index in range(12)
        ]
        continuity = [
            {
                "fact_id": "dc-pin-1",
                "text": "阿澈尚未确认是否去冲绳花展。",
                "kind": "open_thread",
                "source_channel": "tsuzuri.daily_continuity",
                "source_ref_id": "item-pin-1",
                "created_at_ms": 1_774_500_000_000,
                "confidence": 0.9,
                "salience": 0.9,
                "pinned": True,
            },
            {
                "fact_id": "dc-pin-2",
                "text": "王五承诺确认冲绳花展的最终日期。",
                "kind": "commitment",
                "source_channel": "tsuzuri.daily_continuity",
                "source_ref_id": "item-pin-2",
                "created_at_ms": 1_774_500_000_001,
                "confidence": 0.9,
                "salience": 0.9,
                "pinned": True,
            },
        ]
        result = compile_fact_workspace(
            "冲绳花展后来确认了吗？",
            cards,
            history_fact_records=continuity,
            max_records=10,
        )

        selected_refs = {
            row.get("source_ref_id") for row in result["selected_records"]
        }
        self.assertIn("item-pin-1", selected_refs)
        self.assertIn("item-pin-2", selected_refs)
        self.assertLessEqual(
            sum(bool(row.get("pinned")) for row in result["selected_records"]), 2
        )

    pass  # upstream-only test removed

    pass  # upstream-only test removed


class DailyContinuityKairosTests(unittest.TestCase):
    def test_coverage_is_shadow_only_and_uses_existing_memory_write_planner(self) -> None:
        from tsuzuri import kairos_dream as kairos_dream

        capsule = {
            "date_world": "2026-06-20",
            "generator_version": "daily_continuity_v2_core_2",
            "source": {"archive_sha256": "a" * 64},
            "items": [
                {
                    "item_id": "dc2-user-fact",
                    "kind": "user_fact",
                    "text": "阿澈计划在七月底完成电路复习。",
                    "source_message_ids": ["u-old"],
                    "source_at_ms": 1_750_000_000_000,
                    "temporal_status": "planned",
                    "status": "active",
                    "confidence": 0.9,
                }
            ],
        }

        class FakeCore:
            def __init__(self) -> None:
                self.meta = {}
                self.plan_calls = []

            def get_meta_state_json(self, key, default):
                return self.meta.get(key, default)

            def set_meta_state_json(self, key, value):
                self.meta[key] = value

            def _query_items(self, character, limit=500, query_text=""):
                return []

            def plan_turn_memory_writes(self, character, user_text, assistant_reply, **kwargs):
                self.plan_calls.append((character, user_text, kwargs))
                return {"records": [{"layer": "episodic"}]}

        core = FakeCore()
        with mock.patch(
            "tsuzuri.chat_compact.load_daily_continuity_capsules_for_character",
            return_value=[capsule],
        ):
            first = kairos_dream._run_daily_continuity_coverage_shadow(core, "sid", "王五")
            second = kairos_dream._run_daily_continuity_coverage_shadow(core, "sid", "王五")

        self.assertEqual(first["status"], "audited")
        self.assertTrue(first["shadow_only"])
        self.assertFalse(first["apply"])
        self.assertEqual(first["durable_writes_performed"], 0)
        self.assertEqual(first["memory_write_plan_record_count"], 1)
        self.assertEqual(len(core.plan_calls), 1)
        self.assertEqual(second["status"], "idempotent")
        self.assertIn("kairos.daily_continuity_coverage.王五", core.meta)

        changed_capsule = json.loads(json.dumps(capsule, ensure_ascii=False))
        changed_capsule["items"][0]["text"] = "阿澈已经完成了电路复习。"
        with mock.patch(
            "tsuzuri.chat_compact.load_daily_continuity_capsules_for_character",
            return_value=[changed_capsule],
        ):
            changed = kairos_dream._run_daily_continuity_coverage_shadow(
                core, "sid", "王五"
            )
        self.assertEqual(changed["status"], "audited")
        self.assertEqual(len(core.plan_calls), 2)

    def test_coverage_isolated_by_character_when_session_switches_character(self) -> None:
        from tsuzuri import kairos_dream as kairos_dream

        capsules = {
            character: [
                {
                    "character": character,
                    "date_world": "2026-06-20",
                    "generator_version": "daily_continuity_v2_core_2",
                    "source": {"archive_sha256": source_sha * 64},
                    "items": [
                        {
                            "item_id": f"dc2-{character}",
                            "kind": "user_fact",
                            "text": text,
                            "source_message_ids": [f"u-{character}"],
                            "source_at_ms": 1_750_000_000_000,
                            "temporal_status": "current",
                            "status": "active",
                            "confidence": 0.9,
                        }
                    ],
                }
            ]
            for character, source_sha, text in (
                ("王五", "a", "阿澈在复习电路。"),
                ("张三", "b", "阿澈在准备英语考试。"),
            )
        }

        class FakeCore:
            def __init__(self) -> None:
                self.meta = {}
                self.plan_characters = []

            def get_meta_state_json(self, key, default):
                return self.meta.get(key, default)

            def set_meta_state_json(self, key, value):
                self.meta[key] = value

            def _query_items(self, character, limit=500, query_text=""):
                return []

            def plan_turn_memory_writes(self, character, *args, **kwargs):
                self.plan_characters.append(character)
                return {"records": [{"layer": "episodic"}]}

        core = FakeCore()
        with mock.patch(
            "tsuzuri.chat_compact.load_daily_continuity_capsules_for_character",
            side_effect=lambda character, **_: capsules[character],
        ):
            wuwu = kairos_dream._run_daily_continuity_coverage_shadow(core, "sid", "王五")
            zhangsan = kairos_dream._run_daily_continuity_coverage_shadow(core, "sid", "张三")

        self.assertEqual(wuwu["status"], "audited")
        self.assertEqual(zhangsan["status"], "audited")
        self.assertEqual(core.plan_characters, ["王五", "张三"])
        self.assertEqual(
            set(core.meta),
            {
                "kairos.daily_continuity_coverage.王五",
                "kairos.daily_continuity_coverage.张三",
            },
        )


if __name__ == "__main__":
    unittest.main()
