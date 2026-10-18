"""Offline building blocks: a deterministic responder and an OpenAI-compatible client factory.

``offline_reply`` is not a language model. It answers from the compiled workspace so a
demo or a CI run can exercise the whole memory loop without network access: when the
fact workspace or recall block carries something relevant to the current message it
echoes that back, otherwise it acknowledges the message.
"""
from __future__ import annotations

import os
import re
from typing import Callable

_FACT_LINE = re.compile(r"^\s*[-•]\s*(?:\[[^\]]*\]\s*)*(.+?)\s*$")
_SPEAKER_TAG = re.compile(r"^\s*\[[^\]]{1,24}\][:：]\s*")
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
         if m.get("role") == "user" and not _SPEAKER_TAG.match(str(m.get("content", "")))),
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


__all__ = ["offline_reply", "openai_compatible"]
