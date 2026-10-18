"""Synthetic recall evaluation: does what was said earlier come back when it matters?

Each JSONL case describes a short history and a probe::

    {"id": "pref-tea", "agent": "林栖",
     "turns": ["我最喜欢喝乌龙茶"],                      # earlier user messages (1v1 with agent)
     "room": {"members": ["林栖", "苏芮"], "turns": [...], "shared": true},   # optional
     "share": {"from": "苏芮", "text": "..."},            # optional hearsay
     "probe_agent": "林栖",                               # defaults to agent
     "query": "我喜欢喝什么来着？",
     "expect_any": ["乌龙茶"],                            # pass if any recalled/workspace text contains one
     "expect_none": ["奶茶"]}                             # and none contains these

The score checks both channels a model would see: recalled memory rows and the
compiled fact workspace. Cases run offline (rule-based extraction) in isolated
temporary stores, so results are reproducible.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from importlib import resources
from typing import Any, Iterable


def _bundled_dataset() -> list[dict]:
    text = resources.files("tsuzuri").joinpath("data/eval_recall_v1.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_cases(path: str | None = None) -> list[dict]:
    if not path:
        return _bundled_dataset()
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _visible_texts(memory, agent: str, query: str, *, fresh_session: bool = True) -> list[str]:
    """What the model would see for ``query``. ``fresh_session`` (default) starts a new session —
    no raw transcript and no session fact cards — so only long-term memory can supply the answer."""
    probe = {"id": "probe", "role": "user", "content": query, "timestamp": int(time.time() * 1000)}
    if fresh_session:
        session = {"history": [probe], "turn_fact_cards": []}
    else:
        session = memory.load_session(agent)
        session["history"].append(probe)
    messages, rows = memory.build_messages(agent, query, session=session)
    texts = [str(r.get("text", "") or "") for r in rows]
    texts += [str(m.get("content", "") or "") for m in messages if m.get("role") == "system"]
    return texts


def run_case(case: dict, *, data_dir: str | None = None, personas: Any = None, embeddings: bool = False) -> dict:
    from tsuzuri import Tsuzuri

    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tsuzuri-eval-", dir=data_dir, ignore_cleanup_errors=True) as tmp:
        memory = Tsuzuri(tmp, personas=personas, embeddings=embeddings)
        agent = case.get("agent") or memory.agents[0]
        for line in case.get("turns", []) or []:
            memory.chat(agent, line)
        room_cfg = case.get("room")
        room = None
        if isinstance(room_cfg, dict):
            room = memory.room(room_cfg.get("members", memory.agents[:2]), shared=bool(room_cfg.get("shared")))
            for line in room_cfg.get("turns", []) or []:
                room.say(line)
        share = case.get("share")
        hearsay: list[dict] = []
        if isinstance(share, dict):
            members = list(dict.fromkeys([share.get("from"), case.get("probe_agent") or agent]))
            room = memory.room(members)
            room.share(share.get("from"), share.get("text", ""))
            hearsay = room.hearsay_for(case.get("probe_agent") or agent)
        probe = case.get("probe_agent") or agent
        texts = _visible_texts(memory, probe, case["query"], fresh_session=bool(case.get("fresh_session", True)))
        if room is not None and probe in room.members and not case.get("fresh_session", True):
            # what the agent sees when the question is asked inside the room
            texts += [str(m.get("content", "") or "") for m in room.messages_for(probe, case["query"])]
        texts += [str(h.get("text", "") or h.get("original_text", "") or "") for h in hearsay]
        blob = "\n".join(texts)
        hit = [kw for kw in case.get("expect_any", []) if kw in blob]
        leak = [kw for kw in case.get("expect_none", []) if kw in blob]
        ok = (bool(hit) or not case.get("expect_any")) and not leak
        return {"id": case.get("id"), "ok": ok, "hit": hit, "leak": leak, "probe_agent": probe}


def run_recall_eval(dataset: str | None = None, *, data_dir: str | None = None, personas: Any = None,
                    cases: Iterable[dict] | None = None, embeddings: bool = False) -> dict:
    if embeddings:
        os.environ["EMBEDDING_ENABLED"] = "1"
    rows = [run_case(c, data_dir=data_dir, personas=personas, embeddings=embeddings)
            for c in (cases if cases is not None else load_cases(dataset))]
    passed = sum(1 for r in rows if r["ok"])
    by_tag: dict[str, list[bool]] = {}
    for r in rows:
        tag = str(r["id"]).split("-", 1)[0]
        by_tag.setdefault(tag, []).append(r["ok"])
    return {
        "summary": {
            "mode": "embeddings" if embeddings else "lexical (offline)",
            "cases": len(rows),
            "passed": passed,
            "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
            "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_tag.items())},
        },
        "cases": rows,
    }


__all__ = ["load_cases", "run_case", "run_recall_eval"]
