"""Guard: pure-question perception nodes must not persist as episodic event rows."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsuzuri.memory_core import MemoryCoreStore  # noqa: E402


def _node(text, *, intent="qa", node_id="n1"):
    return {
        "id": node_id,
        "ts": "2026-09-11T10:00:00Z",
        "modality": "text",
        "intent": intent,
        "valence": 0.0,
        "surprise": 0.5,
        "salience": 0.5,
        "actors": ["user"],
        "causal_links": [],
        "text": text,
    }


class TestEpisodicQaNodeGuard(unittest.TestCase):
    def _plan(self, nodes):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        core = MemoryCoreStore(
            db_path=os.path.join(tmp.name, "memory.db"),
            snapshot_path=os.path.join(tmp.name, "snapshot.json"),
        )
        return core.plan_turn_memory_writes(
            "李四",
            user_text="你还记得我是谁吗",
            assistant_reply="……当然记得。",
            event_nodes=nodes,
            meta={"latest_user_msg": "你还记得我是谁吗"},
        )

    def test_pure_question_node_is_not_persisted(self):
        plan = self._plan([_node("你还记得我是谁吗")])
        episodic = [r for r in plan.get("records", []) if r.get("item_type") == "event"]
        self.assertEqual(episodic, [])

    def test_non_qa_node_still_persisted(self):
        plan = self._plan([_node("用户说今天学校停课一天", intent="chat")])
        episodic = [r for r in plan.get("records", []) if r.get("item_type") == "event"]
        self.assertEqual(len(episodic), 1)

    def test_long_question_node_still_persisted(self):
        long_q = "我下周三可能要出趟远门大概两个星期回不来，这段时间你会不会想我啊？"
        plan = self._plan([_node(long_q)])
        episodic = [r for r in plan.get("records", []) if r.get("item_type") == "event"]
        self.assertEqual(len(episodic), 1)

    def test_fact_query_intent_not_filtered(self):
        plan = self._plan([_node("我的生日是哪天？?".replace("? ?", "？"), intent="fact_query")])
        episodic = [r for r in plan.get("records", []) if r.get("item_type") == "event"]
        self.assertEqual(len(episodic), 1)


if __name__ == "__main__":
    unittest.main()
