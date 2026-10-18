"""Command line: ``tsuzuri demo | chat | recall | dream | eval``."""
from __future__ import annotations

import argparse
import json
import os
import sys


def _engine(args):
    from tsuzuri import Tsuzuri, openai_compatible

    llm = reply = None
    if getattr(args, "llm", False):
        llm, reply = openai_compatible()
    return Tsuzuri(args.data_dir, personas=args.personas, llm=llm, reply=reply, embeddings=bool(getattr(args, "embeddings", False)))


def cmd_demo(args) -> int:
    """Scripted walkthrough: write facts in one session, recall them later, then dream."""
    import tempfile

    from tsuzuri import Tsuzuri

    data_dir = args.data_dir or tempfile.mkdtemp(prefix="tsuzuri-demo-")
    memory = Tsuzuri(data_dir, personas=args.personas)
    agent = memory.agents[0]
    script = [
        "我下周三要去面试，有点紧张",
        "对了，我最近在学做咖喱饭，周末想再试一次",
        "其实我不太能吃辣，上次放多了辣椒",
        "你还记得我下周三要做什么吗？",
        "我上次说吃辣的事，你还记得吗？",
    ]
    print(f"data dir: {data_dir}\nagent: {agent}   user: {memory.user_name}\n")
    for line in script:
        turn = memory.chat(agent, line)
        print(f"{memory.user_name}: {line}\n{agent}: {turn.reply}")
        if turn.recalled:
            print("   recalled:", " | ".join(r["text"][:40] for r in turn.recalled[:3]))
        print()
    if len(memory.agents) >= 3:
        room = memory.room(memory.agents[:3])
        print("— room —")
        res = room.say("周末大家有空一起去看展吗？")
        for who, text in res.replies:
            print(f"{who}: {text}")
    return 0


def cmd_chat(args) -> int:
    memory = _engine(args)
    agent = args.agent or memory.agents[0]
    print(f"chatting with {agent} (Ctrl-D to quit)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        turn = memory.chat(agent, line)
        print(f"{agent}: {turn.reply}")
    return 0


def cmd_recall(args) -> int:
    memory = _engine(args)
    rows = memory.recall(args.agent or memory.agents[0], args.query, k=args.k)
    for row in rows:
        print(f"[{row.get('layer', '')}] {row.get('text', '')}")
    return 0


def cmd_dream(args) -> int:
    memory = _engine(args)
    report = memory.dream(args.agent or memory.agents[0])
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str)[:4000])
    return 0


def cmd_eval(args) -> int:
    from tsuzuri.evaluation import run_recall_eval

    result = run_recall_eval(args.dataset, data_dir=args.data_dir, personas=args.personas, embeddings=args.embeddings)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    return 0 if result["summary"]["pass_rate"] >= args.min_pass_rate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsuzuri", description=__doc__)
    parser.add_argument("--data-dir", default=os.getenv("TSUZURI_HOME"), help="memory store directory")
    parser.add_argument("--personas", default=os.getenv("PERSONA_CONFIG_PATH"), help="cast JSON file")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="offline scripted walkthrough").set_defaults(func=cmd_demo)

    p = sub.add_parser("chat", help="interactive chat on stdin")
    p.add_argument("--agent")
    p.add_argument("--llm", action="store_true", help="use LLM_MODEL / LLM_BASE_URL / LLM_API_KEY")
    p.add_argument("--embeddings", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("recall", help="show what would be recalled for a query")
    p.add_argument("query")
    p.add_argument("--agent")
    p.add_argument("-k", type=int, default=6)
    p.set_defaults(func=cmd_recall)

    p = sub.add_parser("dream", help="run one consolidation cycle")
    p.add_argument("--agent")
    p.add_argument("--llm", action="store_true")
    p.set_defaults(func=cmd_dream)

    p = sub.add_parser("eval", help="run the recall evaluation set")
    p.add_argument("--dataset", default=None, help="JSONL cases (default: bundled synthetic set)")
    p.add_argument("--report", default=None)
    p.add_argument("--min-pass-rate", type=float, default=0.0)
    p.add_argument("--embeddings", action="store_true", help="enable dense recall (embeddings extra)")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
