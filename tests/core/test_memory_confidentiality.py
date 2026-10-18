"""用户要求保密的原话不进共享层（2026-10-19 外部审计 COMP-004）。"""
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.memory_core_impl.memory_rows import _user_text_requests_confidentiality


class ConfidentialityPatternTests(unittest.TestCase):
    def test_confidential_requests_detected(self):
        for text in (
            "我下个月要搬去大阪了，先别告诉王五和李四",
            "这事你知我知，不要告诉赵六和钱七",
            "别让李四知道我在准备生日礼物",
            "不要跟王五说我今天哭了",
            "这是秘密哦，只告诉你",
            "帮我瞒着她们几天",
        ):
            self.assertTrue(_user_text_requests_confidentiality(text), text)

    def test_ordinary_shared_events_not_flagged(self):
        for text in (
            "周末和王五、李四一起去活动室集训了",
            "我跟赵六说好了明天一起去车站",
            "今天钱七告诉我她找到一只新猫",
        ):
            self.assertFalse(_user_text_requests_confidentiality(text), text)


class ConfidentialCollectiveWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory_core_test.db")
        self.core = MemoryCoreStore(
            db_path=self.db_path, snapshot_path=os.path.join(self.tmpdir.name, "dev_memory_test.json")
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _collective_texts(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        try:
            return [str(r[0]) for r in conn.execute("SELECT text FROM memory_items WHERE memory_role='collective'")]
        finally:
            conn.close()

    def test_secret_naming_two_members_stays_out_of_collective(self):
        self.core.append_turn_event(
            "张三", "我下个月要搬去大阪了，先别告诉王五和李四", "嗯，我不说。", event_nodes=[], meta={}, memory_snapshot={}
        )
        self.assertEqual(self._collective_texts(), [])

    def test_shared_event_still_reaches_collective(self):
        self.core.append_turn_event(
            "张三", "周末和王五、李四一起去活动室集训了", "好耶！", event_nodes=[], meta={}, memory_snapshot={}
        )
        self.assertEqual(self._collective_texts(), ["周末和王五、李四一起去活动室集训了"])

    def test_digest_never_sees_confidential_turns(self):
        def _turn(user_text: str, reply: str) -> dict:
            return {
                "text": f"user:{user_text} | assistant:{reply}",
                "payload_json": json.dumps({"user_text": user_text}, ensure_ascii=False),
            }

        prompts: list[str] = []
        turns = [
            _turn("我下个月要搬去大阪了，先别告诉王五和李四", "嗯，我替你保守秘密。"),
            _turn("我最近在学做蛋糕，已经烤糊两次了", "哈哈哈下次分我一块"),
            _turn("想吃拉面，你有推荐的店吗", "车站那家豚骨就不错"),
        ]
        out = self.core._cross_char_digest_call_llm("张三", turns, lambda p: prompts.append(p) or "阿澈在学做蛋糕")
        self.assertEqual(out, "阿澈在学做蛋糕")
        self.assertTrue(prompts)
        self.assertNotIn("大阪", prompts[0])
        self.assertIn("拉面", prompts[0])
        # 规则 fallback 同样不可见
        with patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_BASE_URL": "", "LLM_MODEL": ""}):
            fallback = self.core._cross_char_digest_call_llm("张三", turns, None)
        self.assertNotIn("大阪", fallback)


if __name__ == "__main__":
    unittest.main()
