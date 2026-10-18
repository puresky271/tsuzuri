# -*- coding: utf-8 -*-
"""Unit tests for Phase 5.4 collective 写入条件 — LINE channel skip (2026-05-24)。

Phase 5.4 plan：multi-room channel=line 时不写 collective layer
（共在 colocated 自然写、跨地理 LINE = 各自不同位置、不是 group 集体事件、
写 collective 会让 KAIROS dream 误判为"group 一起做某事"、污染集体记忆）。

测试 plan_turn_memory_writes 在不同 meta.channel 下的 records 输出：
  - channel=line + 有 collective trigger → 不写 collective
  - channel=colocated + 有 collective trigger → 写 collective（旧行为）
  - 无 channel meta + 有 collective trigger → 写 collective（向后兼容）
"""
import os
import sqlite3
import tempfile
import unittest

from tsuzuri.memory_core import MemoryCoreStore


class CollectiveChannelSkipTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.snapshot_path = os.path.join(self.tmpdir.name, "dev_memory_test.json")
        self.core = MemoryCoreStore(db_path=self.db_path, snapshot_path=self.snapshot_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _has_collective_record(self, records: list) -> bool:
        """检查 records 中是否含 memory_type=collective_event 的条目。"""
        return any(
            isinstance(r, dict) and r.get("memory_type") == "collective_event"
            for r in records
        )

    # ── 强 collective trigger 文本：含 2+ canon char 名 + group kw ──
    # 这是最强信号、不论长度都应该触发 collective（除非被 channel skip）
    _STRONG_TURN = ("user", "我和赵六、王五在 活动中心 集训时聊了小组的事")

    def test_channel_line_skips_collective(self):
        """Phase 5.4 核心：channel=line → 不写 collective。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            self._STRONG_TURN[1],
            "嗯！",
            event_nodes=[],
            meta={"channel": "line", "room_id": "test", "participants": ["张三", "李四"]},
        )
        records = plan.get("records", [])
        self.assertTrue(records, "至少应有 working_turn record")
        self.assertFalse(
            self._has_collective_record(records),
            "channel=line 时不该写 collective、records=%s" % [r.get("memory_type") for r in records],
        )

    def test_channel_colocated_writes_collective(self):
        """channel=colocated → collective 正常写入（保持旧行为）。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            self._STRONG_TURN[1],
            "嗯！",
            event_nodes=[],
            meta={"channel": "colocated", "room_id": "test", "participants": ["张三", "李四"]},
        )
        records = plan.get("records", [])
        self.assertTrue(
            self._has_collective_record(records),
            "channel=colocated + 强 trigger 应写 collective",
        )

    def test_no_channel_meta_writes_collective(self):
        """无 channel meta（旧 1v1 路径）→ collective 正常写入。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            self._STRONG_TURN[1],
            "嗯！",
            event_nodes=[],
            meta={},  # 无 channel 字段
        )
        records = plan.get("records", [])
        self.assertTrue(
            self._has_collective_record(records),
            "无 channel meta 时保持旧行为（写 collective）",
        )

    def test_channel_mixed_writes_collective(self):
        """channel=mixed → 不命中 line skip、collective 正常写入（mixed 视为 partial colocated）。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            self._STRONG_TURN[1],
            "嗯！",
            event_nodes=[],
            meta={"channel": "mixed"},
        )
        records = plan.get("records", [])
        self.assertTrue(
            self._has_collective_record(records),
            "channel=mixed 应仍写 collective（只 line 才 skip）",
        )

    def test_channel_line_case_insensitive(self):
        """channel meta 大小写敏感测试：LINE / Line / line 都该 skip。"""
        for variant in ("line", "LINE", "Line", "lInE"):
            with self.subTest(channel=variant):
                plan = self.core.plan_turn_memory_writes(
                    "张三",
                    self._STRONG_TURN[1],
                    "嗯！",
                    event_nodes=[],
                    meta={"channel": variant},
                )
                records = plan.get("records", [])
                self.assertFalse(
                    self._has_collective_record(records),
                    f"channel={variant!r} 应识别为 line、跳过 collective",
                )

    def test_no_collective_trigger_unaffected_by_channel(self):
        """无 collective trigger（普通闲聊）→ 不论 channel 都不写 collective（原本就不写）。"""
        plan = self.core.plan_turn_memory_writes(
            "张三",
            "今天天气真好啊",
            "嗯、风也不大。",
            event_nodes=[],
            meta={"channel": "line"},
        )
        records = plan.get("records", [])
        self.assertFalse(self._has_collective_record(records))

    def test_meta_channel_persists_in_working_turn_payload(self):
        """working_turn record 的 payload 里 channel meta 可被下游读到（无强约束、
        但若未来 KAIROS dream 想按 channel 区分召回、字段必须存在）。
        当前 plan_turn_memory_writes 没显式把 channel 写进 record.payload、
        本测试仅记录 baseline 行为：channel 不在 payload、但 meta 已传入函数。
        """
        plan = self.core.plan_turn_memory_writes(
            "张三",
            "李四今天怎么样",
            "我去问问她",
            event_nodes=[],
            meta={"channel": "line", "co_present_chars": ["李四"]},
        )
        # baseline: working_turn 是必有的、第一条
        records = plan.get("records", [])
        self.assertTrue(records and records[0].get("memory_type") == "working_turn")


if __name__ == "__main__":
    unittest.main()
