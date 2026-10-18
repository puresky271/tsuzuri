from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

_CHAT_SERVER = None


pass  # upstream-only test removed


ROOM_PRESENCE = {
    "room_id": "solo_李四",
    "channel": "line",
    "user_present": False,
    "members": [
        {"character": "张三", "co_located": True, "activity": "收拾水彩", "location": "活动中心"},
        {"character": "赵六", "co_located": False},
    ],
}


class RoomPresenceFrameTests(unittest.TestCase):
    """turn_context_workspace 多人在场扩展（2026-08-08，决策 2）。"""

    def _frame(self, room_presence, world=None):
        from tsuzuri.turn_context_workspace import compile_character_state_frame
        return compile_character_state_frame(
            "李四",
            world_context=world or {},
            use_structured_sources=True,
            room_presence_state=room_presence,
        )

    def test_frame_renders_room_presence_section(self):
        result = self._frame(ROOM_PRESENCE)
        text = result["text"]
        self.assertIn("【多人对话在场】", text)
        self.assertIn("这不是你和阿澈的私聊", text)
        self.assertIn("LINE 群聊", text)
        self.assertIn("张三：就在你身边，正在「收拾水彩」（活动中心）", text)
        self.assertIn("赵六：线上成员", text)
        self.assertIn("红线", text)
        self.assertEqual(result["resolved"]["room_presence"]["co_located"], ["张三"])
        self.assertEqual(result["resolved"]["room_presence"]["remote"], ["赵六"])
        self.assertTrue(any("多人对话中" in d for d in result["state_digest"]))

    def test_frame_colocated_channel_user_present(self):
        result = self._frame({**ROOM_PRESENCE, "channel": "colocated", "user_present": True})
        self.assertIn("阿澈：就在你身边", result["text"])
        self.assertIn("面对面", result["text"])

    def test_frame_suppresses_single_peer_boundary_when_room(self):
        indoor_world = {"indoor": {"is_indoor": True, "micro_area": "活动中心"}}
        solo = self._frame(None, world=indoor_world)
        self.assertIn("交互边界：阿澈不在现场", solo["text"])
        multi = self._frame(ROOM_PRESENCE, world=indoor_world)
        self.assertNotIn("交互边界：阿澈不在现场", multi["text"])

    def test_frame_without_room_is_unchanged(self):
        result = self._frame(None)
        self.assertNotIn("【多人对话在场】", result["text"])
        self.assertEqual(result["resolved"]["room_presence"], {})
        self.assertFalse(any("多人对话中" in d for d in result["state_digest"]))

    def test_packet_reminder_distinguishes_co_located_and_remote(self):
        from tsuzuri.turn_context_workspace import compile_turn_execution_packet
        result = compile_turn_execution_packet(
            "李四", "你们刚才练完了吗？",
            room_presence_state=ROOM_PRESENCE,
        )
        text = result["text"]
        self.assertIn("【多人在场提醒】这不是私聊", text)
        self.assertIn("张三（身边·面对面）", text)
        self.assertIn("赵六（线上·LINE）", text)
        # reply target 仍在最末语义位（提醒不得盖过它）
        self.assertTrue(text.rstrip().endswith("</turn-reply-guidance>"))

    def test_packet_without_room_is_unchanged(self):
        from tsuzuri.turn_context_workspace import compile_turn_execution_packet
        result = compile_turn_execution_packet("李四", "hello")
        self.assertNotIn("【多人在场提醒】", result["text"])


class GroupWindowAttributionTests(unittest.TestCase):
    """render_recent_dialogue_window group_mode 说话人归属（2026-08-08）。"""

    HISTORY = [
        {"role": "user", "content": "早上好", "id": "u1", "timestamp": 1700000000000},
        {"role": "assistant", "character": "张三", "content": "早！", "id": "a1", "timestamp": 1700000001000},
        {"role": "assistant", "character": "李四", "content": "早上好……", "id": "a2", "timestamp": 1700000002000},
        {"role": "user", "content": "练完了吗", "id": "u2", "timestamp": 1700000003000},
    ]

    def test_group_mode_prefixes_other_members(self):
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window
        out = render_recent_dialogue_window(self.HISTORY, current_character="李四", group_mode=True)
        msgs = out["messages"]
        # 张三（其他成员）的话转 user role + [名字]：前缀
        self.assertIn({"role": "user", "content": "[张三]：早！"}, msgs)
        # 李四自己的话保持 assistant role 无前缀
        self.assertIn({"role": "assistant", "content": "早上好……"}, msgs)
        # 多人下用户行不带「对 X 说」定向
        user_rows = [m["content"] for m in msgs if m["role"] == "user" and m["content"].startswith("[阿澈")]
        self.assertTrue(user_rows)
        self.assertTrue(all(" 对 " not in row for row in user_rows))

    def test_legacy_mode_keeps_plain_assistant(self):
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window
        out = render_recent_dialogue_window(self.HISTORY, current_character="李四")
        # group_mode=False：不转写（1v1 零漂移）
        self.assertIn({"role": "assistant", "content": "早！"}, out["messages"])
        self.assertNotIn({"role": "user", "content": "[张三]：早！"}, out["messages"])

    def test_group_mode_keeps_full_turn_chain(self):
        """turn 内前序 speaker 的回复在触发用户消息之后，也必须可见（2026-08-08 修复）。"""
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window
        mid_turn = [
            {"role": "user", "content": "练完了吗", "id": "u1", "timestamp": 1700000000000},
            {"role": "assistant", "character": "张三", "content": "刚收工~", "id": "a1", "timestamp": 1700000001000},
            {"role": "assistant", "character": "赵六", "content": "辛苦了", "id": "a2", "timestamp": 1700000002000},
        ]
        out = render_recent_dialogue_window(mid_turn, current_character="李四", group_mode=True)
        msgs = out["messages"]
        # 整段按时间序可见：触发用户消息 → 前序两个 speaker 的回复
        self.assertEqual(
            msgs,
            [
                {"role": "user", "content": "[阿澈]：练完了吗"},
                {"role": "user", "content": "[张三]：刚收工~"},
                {"role": "user", "content": "[赵六]：辛苦了"},
            ],
        )

    def test_group_mode_anchor_is_own_speech(self):
        """当前角色自己是最后一个发言者时，自己的话保持 assistant 且在末尾。"""
        from tsuzuri.turn_fact_ledger import render_recent_dialogue_window
        mid_turn = [
            {"role": "user", "content": "练完了吗", "id": "u1", "timestamp": 1700000000000},
            {"role": "assistant", "character": "张三", "content": "刚收工~", "id": "a1", "timestamp": 1700000001000},
            {"role": "assistant", "character": "李四", "content": "嗯……", "id": "a2", "timestamp": 1700000002000},
        ]
        out = render_recent_dialogue_window(mid_turn, current_character="李四", group_mode=True)
        self.assertEqual(out["messages"][-1], {"role": "assistant", "content": "嗯……"})


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
