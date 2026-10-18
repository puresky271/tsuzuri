"""Offline building blocks: a deterministic responder, a rule-based correction reviewer
and an OpenAI-compatible client factory.

``offline_reply`` is not a language model. It answers from the compiled workspace so a
demo or a CI run can exercise the whole memory loop without network access: when the
fact workspace or recall block carries something relevant to the current message it
echoes that back, otherwise it acknowledges the message.

``offline_validity_review`` stands in for the LLM fact-validity reviewer: when the user
explicitly corrects themselves ("说错了，我的猫叫豆包") it supersedes the earlier claim
about the same thing, so the stale value stops winning.
"""
from __future__ import annotations

import os
import re
from typing import Callable

_FACT_LINE = re.compile(r"^\s*[-•]\s*(?:\[[^\]]*\]\s*)*(.+?)\s*$")
# Conversation lines carry a "[speaker]：" (or "[speaker 对 listener 说]：") prefix; in a
# room, a line whose speaker is another agent is not the user's own message.
_SPEAKER_TAG = re.compile(r"^\s*\[([^\]]{1,24})\][:：]\s*")


def _is_other_agent_line(content: str) -> bool:
    m = _SPEAKER_TAG.match(content)
    if not m:
        return False
    from tsuzuri import persona_registry

    speaker = m.group(1).split(" 对 ", 1)[0].strip()
    return persona_registry.is_agent(speaker)
_TRAILING_STAMP = re.compile(r"[（(]\d{4}-\d{2}-\d{2}[^）)]*[）)]\s*$")
_TAG_BLOCK = re.compile(r"<(fact-workspace|memory-recall-workspace)[^>]*>(.*?)</\1>", re.S)


def _tokens(text: str) -> set[str]:
    text = str(text or "").lower()
    out: set[str] = set(re.findall(r"[a-z0-9]{3,}", text))
    cjk = "".join(re.findall(r"[一-鿿]", text))
    out.update(cjk[i:i + 2] for i in range(max(0, len(cjk) - 1)))
    return out


def _workspace_facts(messages: list[dict]) -> list[str]:
    facts: list[str] = []
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = str(msg.get("content", "") or "")
        blocks = [m.group(2) for m in _TAG_BLOCK.finditer(content)] or [content]
        for block in blocks:
            for line in block.splitlines():
                m = _FACT_LINE.match(line)
                if m and len(m.group(1)) >= 4:
                    facts.append(m.group(1))
    return facts


def offline_reply(messages: list[dict]) -> str:
    """Deterministic stand-in for a chat model (see module docstring)."""
    # the user's own latest message (room context renders other agents as "[name]：" user turns)
    user = next(
        (str(m.get("content", "")) for m in reversed(messages)
         if m.get("role") == "user" and not _is_other_agent_line(str(m.get("content", "")))),
        "",
    )
    current = _SPEAKER_TAG.sub("", user.split("\n\n<turn-execution-packet", 1)[0].strip())
    query = _tokens(current)
    best, best_score = "", 0
    for fact in _workspace_facts(messages):
        fact_tokens = _tokens(fact)
        if query and len(query & fact_tokens) >= 0.8 * len(query):
            continue  # the current message itself (already carded this turn), not a memory
        score = len(query & fact_tokens)
        if score > best_score:
            best, best_score = fact, score
    if best and best_score >= 2:
        return f"嗯，我记得——{_TRAILING_STAMP.sub('', best).strip()}"
    return f"嗯，我在听。（{current[:40]}）" if current else "嗯。"


_CORRECTION_CUE = re.compile(
    r"说错了|搞错了|记错了|写错了|口误|不对|不是的|更正|纠正|其实是|其实叫|应该是|"
    r"改成|换成|改叫|改名|搬(?:去|到|家)|已经不|不再"
)
_CLAUSE_SPLIT = re.compile(r"[。；;！!\n]")
_SAY_PREFIX = re.compile(r"^.{0,40}?说[:：]")
_STOP_CHARS = set("我你他她它的了是不对说错吗呢吧啊呀哦嗯也都就还其实应该改成换成现在已经")
_SLOTS = (
    ("住", "搬", "家住", "住在"),
    ("叫", "名字", "称呼"),
    ("喜欢", "爱喝", "爱吃", "最爱", "讨厌"),
    ("工作", "上班", "公司", "职业"),
    ("学校", "上学", "专业"),
)


def _content_bigrams(text: str) -> set[str]:
    cjk = "".join(ch for ch in str(text or "") if "一" <= ch <= "鿿" and ch not in _STOP_CHARS)
    return {cjk[i:i + 2] for i in range(max(0, len(cjk) - 1))}


def _same_slot(a: str, b: str) -> bool:
    return any(any(k in a for k in group) and any(k in b for k in group) for group in _SLOTS)


def offline_validity_review(character: str, user_entry: dict, cards: list) -> dict:
    """Rule-based stand-in for ``fact_validity.review_current_turn`` (no LLM, no network).

    Only an explicit self-correction in the current user message supersedes anything,
    and only the most recent earlier claim about the same thing; the action still goes
    through ``fact_validity.validate_review`` so the same evidence rules apply.
    """
    from tsuzuri.fact_validity import candidates_for_review, validate_review

    text = str((user_entry or {}).get("content", "") or "")
    candidates = candidates_for_review(cards, character, int((user_entry or {}).get("timestamp", 0) or 0))
    clause = next((c.strip() for c in _CLAUSE_SPLIT.split(text) if _CORRECTION_CUE.search(c)), "")
    raw: dict = {"actions": [], "developer_state_updates": []}
    if clause and not re.search(r"[?？]", text):
        corrected = _content_bigrams(clause)
        best = None
        for cand in candidates:  # newest first
            if cand.get("field") != "fact":
                continue
            claim = _SAY_PREFIX.sub("", str(cand.get("text", "") or ""))
            if claim and (corrected & _content_bigrams(claim) or _same_slot(clause, claim)):
                best = cand
                break
        if best is not None:
            raw["actions"].append({"target_id": best["id"], "status": "superseded",
                                   "evidence_span": clause, "confidence": 0.96})
    review = validate_review(raw, user_entry or {}, candidates)
    review["review_status"] = "offline_rules"
    return review


def openai_compatible(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    timeout: float = 60.0,
) -> tuple[Callable[[str], str], Callable[[list[dict]], str]]:
    """Return ``(text_llm, chat_llm)`` backed by any OpenAI-compatible endpoint.

    Reads ``LLM_MODEL`` / ``LLM_BASE_URL`` / ``LLM_API_KEY`` when arguments are omitted.
    Requires the ``openai`` package (``pip install tsuzuri[llm]``).
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.getenv("LLM_API_KEY", ""),
        base_url=base_url or os.getenv("LLM_BASE_URL") or None,
        timeout=timeout,
    )
    name = model or os.getenv("LLM_MODEL", "")
    if not name:
        raise ValueError("set LLM_MODEL or pass model=")

    def chat(messages: list[dict]) -> str:
        resp = client.chat.completions.create(model=name, messages=messages, temperature=temperature)
        return str(resp.choices[0].message.content or "")

    def text(prompt: str) -> str:
        return chat([{"role": "user", "content": prompt}])

    return text, chat


__all__ = ["offline_reply", "offline_validity_review", "openai_compatible"]
