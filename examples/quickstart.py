"""Offline quick start: remember something in one turn, get it back later.

    python examples/quickstart.py
"""
import tempfile

from tsuzuri import Tsuzuri

memory = Tsuzuri(tempfile.mkdtemp(prefix="tsuzuri-quickstart-"))
agent = memory.agents[0]

for line in [
    "我下周三上午要去面试，有点紧张",
    "对了，我不太能吃辣",
    "最近在学做咖喱饭，周末想再试一次",
]:
    memory.chat(agent, line)

turn = memory.chat(agent, "你还记得我下周三的面试吗？")
print(f"{agent}: {turn.reply}\n")
print("recalled memory:")
for row in turn.recalled:
    print("  -", row["text"])
print("\nfact card of this turn:", {k: turn.fact_card.get(k) for k in ("fact", "state_delta", "commitments")})
