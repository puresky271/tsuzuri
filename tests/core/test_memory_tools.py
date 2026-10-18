from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock


class _FakeProvider:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.calls = []

    def build_memory_context_pack(self, character, query_text, **kwargs):
        self.calls.append((character, query_text, kwargs))
        return {"memory_surface_plan": {"selected_surface_rows": list(self.rows)}}


class MemoryToolActivationTests(unittest.TestCase):
    def test_activation_matrix(self):
        from tsuzuri.memory_tools import should_activate_tool_round

        explicit_pack = {
            "memory_read_bundle": {"query_analysis": {"explicit_recall_intent": True}},
            "memory_surface_plan": {},
        }
        suppressed_pack = {"memory_surface_plan": {"suppression_reason": "no_safe_memory_rows"}}
        quiet_pack = {"memory_surface_plan": {}}

        self.assertTrue(should_activate_tool_round("上次说过什么", explicit_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("继续", suppressed_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertTrue(should_activate_tool_round("上次那件事", suppressed_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("继续", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("晚饭吃什么", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("明天几点集训", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("我以前在上海住过", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertTrue(should_activate_tool_round("后来怎么样了", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("这件事怎么办", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("继续说", quiet_pack, is_guest=False, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("上次说过什么", explicit_pack, is_guest=True, channel_kind="web_1v1")[0])
        self.assertFalse(should_activate_tool_round("上次说过什么", explicit_pack, is_guest=False, channel_kind="group")[0])

    def test_recall_signal_can_activate_without_bounded_slot(self):
        from tsuzuri.memory_tools import should_activate_tool_round

        active, reason = should_activate_tool_round(
            "之前那件事",
            {"memory_surface_plan": {}},
            is_guest=False,
            channel_kind="web_1v1",
        )
        self.assertTrue(active)
        self.assertEqual(reason, "memory_recall_signal")


class MemoryToolParsingTests(unittest.TestCase):
    def test_bad_json_fails_closed(self):
        from tsuzuri.memory_tools import parse_tool_calls

        calls, rejected, error = parse_tool_calls("not-json")
        self.assertEqual(calls, [])
        self.assertEqual(rejected, [])
        self.assertTrue(error.startswith("json_decode_error"))

    def test_unknown_clamp_and_call_limit(self):
        from tsuzuri.memory_tools import parse_tool_calls

        raw = json.dumps({"calls": [
            {"tool": "unknown", "args": {}},
            {"tool": "memory_recall", "args": {"query": "花展", "scope": "char", "top_k": 99}},
            {"tool": "memory_lookup", "args": {"query": "住址", "top_k": 99}},
            {"tool": "memory_lookup", "args": {"query": "生日", "top_k": 1}},
        ]})
        calls, rejected, error = parse_tool_calls(raw, max_calls=2)
        self.assertEqual(error, "")
        self.assertEqual([row["tool"] for row in calls], ["memory_recall", "memory_lookup"])
        self.assertEqual(calls[0]["args"]["top_k"], 3)
        self.assertEqual(calls[1]["args"]["top_k"], 5)
        self.assertTrue(any(row["reason"] == "unknown_tool" for row in rejected))
        self.assertTrue(any(row["reason"] == "max_calls_exceeded" for row in rejected))


class MemoryToolExecutionTests(unittest.TestCase):
    def test_recall_enforces_character_and_collective_scope(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        provider = _FakeProvider([{
            "item_id": "mem-123456789",
            "text": "阿澈后来确认花展地点是冲绳。\n不要保留换行",
            "layer": "semantic",
            "confidence": 0.8,
            "salience": 0.7,
        }])
        raw = json.dumps({"calls": [
            {"tool": "memory_recall", "args": {"query": "花展地点", "scope": "char", "top_k": 3}},
            {"tool": "memory_recall", "args": {"query": "小组集训", "scope": "collective", "top_k": 1}},
        ]})
        result = run_memory_tool_round(
            "张三", "你还记得花展地点吗", {},
            llm_caller=lambda _prompt: raw,
            provider=provider,
            write_enabled=False,
        )
        self.assertEqual([call[0] for call in provider.calls], ["张三", "测试组"])
        self.assertEqual(len(result.records), 2)
        self.assertTrue(all(row["source_channel"] == "memory_recall" for row in result.records))
        self.assertTrue(all(row["source_ref_id"].startswith("memtool:memory_recall:") for row in result.records))
        self.assertTrue(all("\n" not in row["text"] and len(row["text"]) <= 360 for row in result.records))

    def test_caller_exception_fails_closed(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        def fail(_prompt):
            raise RuntimeError("offline")

        result = run_memory_tool_round("张三", "上次呢", {}, llm_caller=fail)
        self.assertEqual(result.records, [])
        self.assertTrue(result.error.startswith("llm_caller_error"))

    def test_lookup_keeps_only_answer_bearing_semantic_rows(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        provider = _FakeProvider([
            {"item_id": "sem-1", "text": "阿澈喜欢柠檬茶。", "layer": "semantic", "confidence": 0.9},
            {"item_id": "epi-1", "text": "阿澈今天喝了柠檬茶。", "layer": "episodic", "confidence": 0.9},
        ])
        raw = json.dumps({"calls": [{
            "tool": "memory_lookup",
            "args": {"query": "柠檬茶偏好", "top_k": 5},
        }]})
        with mock.patch("tsuzuri.memory_core.memory_explicit_row_is_answer_bearing", return_value=True):
            result = run_memory_tool_round(
                "张三", "你还记得我的柠檬茶偏好吗", {},
                llm_caller=lambda _prompt: raw,
                provider=provider,
                write_enabled=False,
            )

        self.assertEqual(len(result.records), 1)
        self.assertIn("阿澈喜欢柠檬茶", result.records[0]["text"])

    def test_write_disabled_returns_rejected_verdict_without_store(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        raw = json.dumps({"calls": [{
            "tool": "memory_propose_write",
            "args": {"kind": "fact", "text": "阿澈喜欢柠檬茶", "evidence": "我喜欢柠檬茶"},
        }]})
        result = run_memory_tool_round(
            "李四", "我喜欢柠檬茶", {}, llm_caller=lambda _prompt: raw, write_enabled=False,
        )
        self.assertEqual(result.calls[0]["status"], "rejected")
        self.assertEqual(result.calls[0]["rejected_reason"], "write_disabled")
        self.assertEqual(result.records, [])

    def test_write_rejects_evidence_not_in_current_user_message(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        raw = json.dumps({"calls": [{
            "tool": "memory_propose_write",
            "args": {"kind": "fact", "text": "阿澈住在杭州", "evidence": "我住在杭州"},
        }]})
        result = run_memory_tool_round(
            "李四", "今天天气不错", {}, llm_caller=lambda _prompt: raw, write_enabled=True,
        )
        self.assertEqual(result.calls[0]["rejected_reason"], "evidence_not_in_user_text")

    def test_write_rejects_dialogue_residue_before_opening_store(self):
        from tsuzuri.memory_tools import run_memory_tool_round

        raw = json.dumps({"calls": [{
            "tool": "memory_propose_write",
            "args": {
                "kind": "fact",
                "text": "嗯，好，那我先去看书了，下次再聊",
                "evidence": "嗯，好，那我先去看书了，下次再聊",
            },
        }]})
        store = mock.Mock()
        result = run_memory_tool_round(
            "李四",
            "嗯，好，那我先去看书了，下次再聊",
            {},
            llm_caller=lambda _prompt: raw,
            write_enabled=True,
            write_store=store,
        )

        self.assertEqual(result.calls[0]["rejected_reason"], "dialogue_residue")
        store.plan_turn_memory_writes.assert_not_called()

    def test_write_rejects_ungrounded_proposal_via_existing_plan(self):
        from tsuzuri.memory_core import MemoryCoreStore
        from tsuzuri.memory_tools import run_memory_tool_round

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryCoreStore(
                db_path=os.path.join(tmp, "memory.db"),
                snapshot_path=os.path.join(tmp, "snapshot.json"),
            )
            raw = json.dumps({"calls": [{
                "tool": "memory_propose_write",
                "args": {
                    "kind": "fact",
                    "text": "阿澈去杭州工作",
                    "evidence": "我喜欢喝柠檬茶",
                },
            }]})
            result = run_memory_tool_round(
                "李四",
                "我喜欢喝柠檬茶",
                {},
                llm_caller=lambda _prompt: raw,
                write_enabled=True,
                write_store=store,
            )

            self.assertEqual(result.calls[0]["rejected_reason"], "ungrounded_hallucination")
            self.assertFalse(store._query_items("李四", limit=20, query_text="杭州"))

    def test_write_accepts_via_existing_plan_and_persists_semantic(self):
        from tsuzuri.memory_core import MemoryCoreStore
        from tsuzuri.memory_tools import run_memory_tool_round

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryCoreStore(
                db_path=os.path.join(tmp, "memory.db"),
                snapshot_path=os.path.join(tmp, "snapshot.json"),
            )
            raw = json.dumps({"calls": [{
                "tool": "memory_propose_write",
                "args": {"kind": "fact", "text": "阿澈喜欢喝柠檬茶", "evidence": "我喜欢喝柠檬茶"},
            }]})
            result = run_memory_tool_round(
                "李四", "我喜欢喝柠檬茶", {},
                llm_caller=lambda _prompt: raw,
                write_enabled=True,
                write_store=store,
            )
            rows = store._query_items("李四", limit=20, query_text="柠檬茶")
            semantic = [row for row in rows if row.get("layer") == "semantic" and "柠檬茶" in str(row.get("text", ""))]
            self.assertEqual(result.calls[0]["status"], "accepted")
            self.assertEqual(result.records, [])
            self.assertTrue(semantic)

    def test_write_does_not_create_collective_memory_from_keyword_only(self):
        from tsuzuri.memory_core import MemoryCoreStore
        from tsuzuri.memory_tools import run_memory_tool_round

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryCoreStore(
                db_path=os.path.join(tmp, "memory.db"),
                snapshot_path=os.path.join(tmp, "snapshot.json"),
            )
            raw = json.dumps({"calls": [{
                "tool": "memory_propose_write",
                "args": {
                    "kind": "fact",
                    "text": "阿澈记得活动中心活动室的位置",
                    "evidence": "我记得活动中心活动室的位置",
                },
            }]})
            result = run_memory_tool_round(
                "王五",
                "我记得活动中心活动室的位置",
                {},
                llm_caller=lambda _prompt: raw,
                write_enabled=True,
                write_store=store,
            )

            self.assertEqual(result.calls[0]["status"], "accepted")
            collective = store._query_items("测试组", limit=20, query_text="活动中心")
            self.assertFalse([row for row in collective if row.get("layer") == "collective"])

    def test_feature_disabled_skips_decision_call(self):
        from tsuzuri.memory_tools import run_memory_tool_round_if_active

        caller = mock.Mock(return_value='{"calls":[]}')
        with mock.patch.dict(os.environ, {"MEM_TURN_TOOLS_ENABLE": "0"}, clear=False):
            result = run_memory_tool_round_if_active(
                "张三", "你还记得上次吗", {},
                llm_caller=caller,
                session_id="s1",
                is_guest=False,
                channel_kind="web_1v1",
            )
        self.assertFalse(result.activated)
        caller.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
