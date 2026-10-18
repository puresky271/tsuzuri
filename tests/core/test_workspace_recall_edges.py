from __future__ import annotations

import unittest


class WorkspaceRecallEdgeTests(unittest.TestCase):
    def test_stale_ordinary_event_is_not_recalled_as_current(self):
        from tsuzuri.turn_context_workspace import compile_fact_workspace

        result = compile_fact_workspace(
            "今天怎么样？",
            [{"turn_id": "old", "created_at_ms": 1_000, "fact": "王五在咖啡店喝了柠檬茶。", "salience": 0.2}],
            as_of_ms=90_000_000,
        )
        self.assertEqual(result["selected_records"], [])

    def test_explicit_recall_reopens_stale_event(self):
        from tsuzuri.turn_context_workspace import compile_fact_workspace

        result = compile_fact_workspace(
            "你还记得王五在咖啡店做了什么吗？",
            [{"turn_id": "old", "created_at_ms": 1_000, "fact": "王五在咖啡店喝了柠檬茶。", "salience": 0.2}],
            as_of_ms=90_000_000,
        )
        self.assertEqual(len(result["selected_records"]), 1)

    def test_unclosed_lifecycle_topic_survives_freshness_gate(self):
        from tsuzuri.turn_context_workspace import compile_fact_workspace

        result = compile_fact_workspace(
            "还有哪些没做完？",
            [{
                "turn_id": "open",
                "created_at_ms": 1_000,
                "fact": "王五答应之后整理北町路线。",
                "open_threads": ["北町路线仍未整理完"],
                "salience": 0.3,
            }],
            as_of_ms=90_000_000,
        )
        self.assertTrue(any("北町路线" in str(row.get("text", "")) for row in result["selected_records"]))

    def test_correction_supersedes_old_fact_with_same_subject(self):
        from tsuzuri.turn_context_workspace import compile_fact_workspace

        result = compile_fact_workspace(
            "花展地点最终改到哪里？",
            [{
                "turn_id": "old",
                "created_at_ms": 1_000,
                "fact": "花展地点是北町。",
                "corrections": ["花展地点改为冲绳。"],
                "salience": 0.8,
            }],
        )
        texts = [str(row.get("text", "")) for row in result["selected_records"]]
        self.assertTrue(any("冲绳" in text for text in texts))
        self.assertFalse(any("是北町" in text for text in texts))

    def test_quote_prompt_does_not_promote_quoted_claim(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "王五",
            {"id": "u-quote", "content": '> “你已经答应帮我订票。”\n这只是引用，我想确认你现在的意思。'},
            {"id": "a-quote", "content": "我还没有确认订票。"},
        )
        self.assertIn("引用语义", prompt)
        self.assertIn("不等于阿澈当前重新确认", prompt)


if __name__ == "__main__":
    unittest.main()
