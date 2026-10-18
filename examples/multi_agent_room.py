"""Group conversation with shared memory and hearsay (offline).

    python examples/multi_agent_room.py
"""
import tempfile

from tsuzuri import Tsuzuri

memory = Tsuzuri(tempfile.mkdtemp(prefix="tsuzuri-room-"))
a, b, c = memory.agents[:3]

room = memory.room([a, b, c], shared=True)
turn = room.say("周六我们一起去爬山吧，早上七点在楼下集合")
for agent, reply in turn.replies:
    print(f"{agent}: {reply}")

# b passes a piece of news on; the other members receive it as hearsay
room.share(b, f"{memory.user_name}下个月要去参加一个摄影比赛")
print(f"\n{c} heard:", [h.get("text") or h.get("original_text") for h in room.hearsay_for(c)])

# what c sees when asked about the plan inside the room
messages = room.messages_for(c, "周六我们约了什么？")
print("\nsystem context for", c)
for m in messages:
    if m["role"] == "system":
        print(m["content"][:300], "\n---")
