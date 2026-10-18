"""LoCoMo evidence-retrieval benchmark (offline, no LLM, no answer generation).

LoCoMo (Maharana et al., ACL 2024) gives long two-person conversations and QA pairs whose
``evidence`` lists the dialogue turns that support the answer. This script replays each
conversation into a fresh tsuzuri store through the normal write path (rule-based fact
cards), asks every question, and checks whether an evidence turn is among the memory rows:

- ``surfaced@k``  rows the engine actually puts into the prompt (``Tsuzuri.recall``)
- ``candidates@k`` the store's top-k ranked rows (``retrieve_memory_context``) before the surface plan
- ``bm25@k``       a plain BM25 baseline over the raw turns, for scale

It measures retrieval only. It is **not** the LoCoMo QA score other memory systems report
(LLM-generated answers judged by an LLM), and the numbers are not comparable to those.

Data: https://github.com/snap-research/locomo (``data/locomo10.json``, CC BY-NC 4.0; not
bundled). Usage::

    python benchmarks/locomo_retrieval.py --data locomo10.json --out locomo_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict

CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or but if of to in on at for with from by is are was were be been being do does did "
            "i you he she it we they me him her us them my your his its our their what when where who whom "
            "which why how that this these those there here as so not no yes have has had will would can could "
            "should shall may might must about into over after before up down out than then too very just".split())


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text or "").lower()) if w not in _STOP]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / max(1, len(docs))
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def top(self, query: list[str], k: int) -> list[int]:
        scores = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avg))
                    for t in query if t in tf)
            scores.append((s, i))
        return [i for s, i in sorted(scores, reverse=True)[:k] if s > 0]


def _turn_text(turn: dict) -> str:
    text = str(turn.get("text", "") or "")
    caption = str(turn.get("blip_caption", "") or "").strip()
    if caption:
        text = f"{text} [shares a photo: {caption}]"
    return f"{turn.get('speaker', '')}: {text}"


def _sessions(conv: dict) -> list[list[dict]]:
    keys = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    return [conv[k] for k in keys if isinstance(conv[k], list)]


def _hit(rows: list[str], evidence_texts: list[str]) -> bool:
    probes = [_norm(t)[:80] for t in evidence_texts if _norm(t)]
    return any(p and p in _norm(r) for r in rows for p in probes)


def run_sample(sample: dict, *, k: int, embeddings: bool, limit_questions: int | None) -> list[dict]:
    from tsuzuri import Tsuzuri
    from tsuzuri.turn_fact_ledger import build_rule_fallback_card

    turns = [t for s in _sessions(sample["conversation"]) for t in s]
    by_id = {str(t.get("dia_id")): _turn_text(t) for t in turns}
    bm25 = BM25([_tokens(_turn_text(t)) for t in turns])
    with tempfile.TemporaryDirectory(prefix="tsuzuri-locomo-", ignore_cleanup_errors=True) as tmp:
        memory = Tsuzuri(tmp, embeddings=embeddings)
        agent = memory.agents[0]
        ts = int(time.time() * 1000)
        with memory._bound():
            for i, turn in enumerate(turns):
                text = _turn_text(turn)
                user = {"id": f"u{i}", "role": "user", "content": text, "timestamp": ts + 2 * i,
                        "audience_scope": "developer"}
                reply = {"id": f"a{i}", "role": "assistant", "content": "", "character": agent,
                         "timestamp": ts + 2 * i + 1, "audience_scope": "developer"}
                card = build_rule_fallback_card(agent, user, reply)
                memory.store.append_turn_event(agent, text, "", event_nodes=[],
                                               meta={"semantic_facts": [], "turn_fact_card": card}, memory_snapshot={})
        out = []
        qa = [q for q in sample["qa"] if q.get("category") in CATEGORIES]
        for q in qa[:limit_questions] if limit_questions else qa:
            evidence = [by_id[e] for e in q.get("evidence", []) if e in by_id]
            if not evidence:
                continue
            question = str(q["question"])
            surfaced = [str(r.get("text", "")) for r in memory.recall(agent, question, k=k)]
            with memory._bound():
                ranked = memory.store.retrieve_memory_context(agent, question, max_items=k, log_retrieval=False)
            candidates = [str(r.get("text", "")) for r in (ranked.get("rows") or [])][:k]
            baseline = [by_id_text for by_id_text in (_turn_text(turns[i]) for i in bm25.top(_tokens(question), k))]
            out.append({
                "sample": sample.get("sample_id"), "category": CATEGORIES[q["category"]], "question": question,
                "surfaced": _hit(surfaced, evidence), "candidates": _hit(candidates, evidence),
                "bm25": _hit(baseline, evidence), "n_surfaced": len(surfaced),
            })
        return out


def summarize(rows: list[dict]) -> dict:
    def rate(sel, key):
        return round(sum(1 for r in sel if r[key]) / len(sel), 3) if sel else 0.0

    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    return {
        "questions": len(rows),
        "overall": {key: rate(rows, key) for key in ("surfaced", "candidates", "bm25")},
        "abstained": round(sum(1 for r in rows if r["n_surfaced"] == 0) / len(rows), 3) if rows else 0.0,
        "by_category": {c: {"n": len(v), **{key: rate(v, key) for key in ("surfaced", "candidates", "bm25")}}
                        for c, v in sorted(by_cat.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--data", required=True, help="path to locomo10.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=0, help="only the first N conversations (0 = all)")
    ap.add_argument("--questions", type=int, default=0, help="only the first N questions per conversation")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    with open(args.data, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    samples = data[: args.samples] if args.samples else data
    rows: list[dict] = []
    for i, sample in enumerate(samples, 1):
        t0 = time.time()
        rows += run_sample(sample, k=args.k, embeddings=args.embeddings, limit_questions=args.questions or None)
        print(f"[{i}/{len(samples)}] {sample.get('sample_id')} done in {time.time() - t0:.0f}s", file=sys.stderr)
    report = {"benchmark": "LoCoMo evidence retrieval", "k": args.k, "embeddings": args.embeddings,
              "summary": summarize(rows), "rows": rows}
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
