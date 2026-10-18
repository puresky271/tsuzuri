"""Use tsuzuri only as the memory layer: compile messages, call your own model, record the turn.

    python examples/compiler_only.py
"""
import tempfile
import time
import uuid

from tsuzuri import Tsuzuri
from tsuzuri.turn_fact_ledger import extract_turn_fact_card, sanitize_fact_cards


def my_model(messages: list[dict]) -> str:
    """Replace with your own LLM call."""
    return "好的，我记下了。"


memory = Tsuzuri(tempfile.mkdtemp(prefix="tsuzuri-compiler-"))
agent = memory.agents[0]

for text in ["我养了一只叫团子的橘猫", "你还记得我养的橘猫吗？"]:
    session = memory.load_session(agent)
    user_entry = {"id": uuid.uuid4().hex, "role": "user", "content": text, "timestamp": int(time.time() * 1000)}
    session["history"].append(user_entry)

    messages, recalled = memory.build_messages(agent, text, session=session)
    reply = my_model(messages)

    reply_entry = {"id": uuid.uuid4().hex, "role": "assistant", "content": reply, "character": agent,
                   "timestamp": int(time.time() * 1000)}
    session["history"].append(reply_entry)
    card = extract_turn_fact_card(agent, user_entry, reply_entry, prior_cards=session["turn_fact_cards"])
    session["turn_fact_cards"] = sanitize_fact_cards([*session["turn_fact_cards"], card])
    memory.store.append_turn_event(agent, text, reply, event_nodes=[], meta={"turn_fact_card": card},
                                   memory_snapshot=memory.store.load_json_snapshot())
    memory.save_session(agent, session)

    print(f"user: {text}")
    print(f"  {len(messages)} messages compiled; recalled: {[r['text'] for r in recalled]}")
