"""test_memory_write_rebalance.py — Phase R.W (2026-06-16) 写入倒转纠正回归

验证两件事（用户 2026-06-16 拍板 Option A「两头都纠正」）：
  1. web 主路径（event_nodes=[]）现在补写 episodic（synthetic）、不再 episodic 写 0
  2. turn-time 自动抽取的 fact 落 episodic（靠 KAIROS 蒸馏）、只有显式记忆请求才即时落 semantic

纯调 plan_turn_memory_writes（不跑 LLM / 不需要模型）、temp DB。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from tsuzuri.memory_core import MemoryCoreStore


def _layers(plan):
    return [str(r.get("layer", "")) for r in plan.get("records", [])]


class WebPathEpisodicBackfillTests(unittest.TestCase):
    def _core(self, tmp):
        return MemoryCoreStore(
            db_path=os.path.join(tmp, "t.db"),
            snapshot_path=os.path.join(tmp, "s.json"),
        )

    def test_web_path_synthesizes_episodic(self):
        """event_nodes=[] + 普通对话 → 合成 1 条 episodic（修正 episodic 写 0）。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="我今天在神田川边写了点诗句",
                assistant_reply="听起来很有灵感呢",
                event_nodes=[],
                meta={"latest_user_msg": "我今天在神田川边写了点诗句"},
            )
            self.assertIn("episodic", _layers(plan), "web 路径应补写 episodic")

    def test_suppression_blocks_synthetic(self):
        """meta 带 retrievable_write_block_reason（testgroup present-self-state 抑制）→ 不合成。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="我现在正在洗澡",
                assistant_reply="那你先忙",
                event_nodes=[],
                meta={
                    "latest_user_msg": "我现在正在洗澡",
                    "retrievable_write_block_reason": "present_self_state",
                },
            )
            self.assertNotIn("episodic", _layers(plan), "抑制态不应合成 episodic")

    def test_guest_no_synthetic(self):
        """guest 模式 → 不合成 episodic（尊重 guest 写策略）。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="你好呀我是新来的",
                assistant_reply="你好",
                event_nodes=[],
                meta={
                    "latest_user_msg": "你好呀我是新来的",
                    "memory_plan": {"mode_scope": "guest"},
                },
            )
            self.assertNotIn("episodic", _layers(plan), "guest 不应合成 episodic")

    def test_provided_event_node_no_double_episodic(self):
        """event_nodes=[node]（testgroup 正常路径）→ 只用提供的、不合成双写。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="我今天去了水族馆",
                assistant_reply="海豚可爱吗",
                event_nodes=[{"id": "ev1", "text": "用户去了水族馆", "salience": 0.7, "surprise": 0.5}],
                meta={"latest_user_msg": "我今天去了水族馆"},
            )
            ep = [r for r in plan.get("records", []) if r.get("layer") == "episodic"]
            self.assertEqual(len(ep), 1, "提供 event_node 时 episodic 应恰好 1 条、无 synthetic 双写")


class FactRoutingTests(unittest.TestCase):
    def _core(self, tmp):
        return MemoryCoreStore(
            db_path=os.path.join(tmp, "t.db"),
            snapshot_path=os.path.join(tmp, "s.json"),
        )

    def test_auto_fact_routes_to_episodic(self):
        """非显式自动抽取 fact → 落 episodic（不再即时 semantic）。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="我在学钢琴",
                assistant_reply="加油呀",
                event_nodes=[],
                meta={
                    "latest_user_msg": "我在学钢琴",
                    "semantic_facts": ["阿澈在学钢琴"],
                },
            )
            sem = [r for r in plan.get("records", [])
                   if r.get("layer") == "semantic" and r.get("memory_type") in ("shared_fact", "character_projection")]
            self.assertEqual(len(sem), 0, "自动抽取 fact 不应即时落 semantic")

    def test_explicit_fact_routes_to_semantic(self):
        """显式记忆请求（记住…）→ 仍即时落 semantic。"""
        with tempfile.TemporaryDirectory() as tmp:
            core = self._core(tmp)
            plan = core.plan_turn_memory_writes(
                "李四",
                user_text="记住，我喜欢喝柠檬茶",
                assistant_reply="好的我记住了",
                event_nodes=[],
                meta={"latest_user_msg": "记住，我喜欢喝柠檬茶"},
            )
            sem = [r for r in plan.get("records", []) if r.get("layer") == "semantic"]
            self.assertGreater(len(sem), 0, "显式记忆请求应即时落 semantic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
