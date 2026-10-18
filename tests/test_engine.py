from __future__ import annotations

import sqlite3

import pytest

from tsuzuri import Tsuzuri


@pytest.fixture()
def memory(tmp_path):
    return Tsuzuri(tmp_path / "store")


def test_chat_writes_memory_and_fact_card(memory):
    agent = memory.agents[0]
    turn = memory.chat(agent, "我家有只橘猫，叫团子")
    assert turn.reply
    assert turn.fact_card.get("fact") or turn.fact_card.get("near_summary")
    with sqlite3.connect(memory.store.db_path) as conn:
        layers = {row[0] for row in conn.execute("SELECT layer FROM memory_items WHERE character=?", (agent,))}
    assert {"working", "episodic"} <= layers
    assert memory.history(agent)[-1]["role"] == "assistant"


def test_recall_surfaces_earlier_fact(memory):
    agent = memory.agents[0]
    memory.chat(agent, "下周三上午我要去面试，有点紧张")
    rows = memory.recall(agent, "你还记得我下周三要去面试吗")
    assert any("面试" in row["text"] for row in rows)


def test_memory_is_private_to_each_agent(memory):
    first, second = memory.agents[:2]
    memory.chat(first, "这件事只跟你说：我在偷偷准备转专业")
    assert not any("转专业" in row["text"] for row in memory.recall(second, "你知道我在准备转专业吗"))


def test_build_messages_orders_system_context_before_conversation(memory):
    agent = memory.agents[0]
    memory.chat(agent, "我最喜欢喝乌龙茶")
    session = memory.load_session(agent)
    session["history"].append({"id": "q", "role": "user", "content": "我喜欢喝什么？", "timestamp": 1})
    messages, _rows = memory.build_messages(agent, "我喜欢喝什么？", session=session)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert "current-state-frame" in messages[1]["content"]
    assert roles.index("user") > max(i for i, r in enumerate(roles) if r == "system")


def test_custom_reply_callable_receives_compiled_messages(tmp_path):
    seen = {}

    def reply(messages):
        seen["messages"] = messages
        return "收到"

    memory = Tsuzuri(tmp_path / "store", reply=reply)
    turn = memory.chat(memory.agents[0], "你好")
    assert turn.reply == "收到"
    assert seen["messages"][-1]["role"] == "user"


def test_llm_callable_is_used_for_fact_extraction(tmp_path):
    calls = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return '{"fact": "用户养了一只叫团子的橘猫。", "near_summary": "用户提到自家的猫。", "topic_labels": ["family"]}'

    memory = Tsuzuri(tmp_path / "store", llm=llm)
    turn = memory.chat(memory.agents[0], "我家有只橘猫，叫团子")
    assert calls, "extraction prompt was not sent to the llm"
    assert "团子" in turn.fact_card.get("fact", "")
    assert turn.fact_card.get("model") == "llm"


def test_unknown_agent_is_rejected(memory):
    with pytest.raises(ValueError):
        memory.chat("nobody", "hi")
    with pytest.raises(ValueError):
        memory.chat(memory.agents[0], "   ")


def test_dream_cycle_runs_offline(memory):
    agent = memory.agents[0]
    memory.chat(agent, "我最近在学做咖喱饭")
    report = memory.dream(agent)
    assert isinstance(report, dict)
