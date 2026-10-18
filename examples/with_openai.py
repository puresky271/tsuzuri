"""Use a real model through any OpenAI-compatible endpoint.

    pip install -e ".[llm]"
    export LLM_MODEL=... LLM_BASE_URL=... LLM_API_KEY=...
    python examples/with_openai.py
"""
from tsuzuri import Tsuzuri, openai_compatible

llm, chat = openai_compatible()          # reads LLM_MODEL / LLM_BASE_URL / LLM_API_KEY
memory = Tsuzuri("./.tsuzuri", llm=llm, reply=chat)
agent = memory.agents[0]

print(f"chatting with {agent}; empty line to stop")
while True:
    text = input("> ").strip()
    if not text:
        break
    turn = memory.chat(agent, text)
    print(f"{agent}: {turn.reply}")

print("\nconsolidating ...")
report = memory.dream(agent)
print({k: report.get(k) for k in ("status", "domain_writes") if k in report})
