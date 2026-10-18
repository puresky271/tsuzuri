"""Lossless normalization and bounded segmentation for archived chat days."""

from __future__ import annotations
from tsuzuri.persona_registry import user_name

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable


_ELLIPSIS_ONLY_RE = re.compile(r"^[·\.…—\s]*[·\.…—]{3,}[·\.…—\s？！?!]*$")
_HUM_ONLY_RE = re.compile(r"^[\s]*(嗯|啊|哦|诶|唉|呃)[\s，。．、！？!?]*$")


@dataclass(frozen=True)
class NormalizedMessage:
    source_message_id: str
    role: str
    content: str
    timestamp_ms: int
    character: str = ""
    fragment_index: int = 0
    fragment_count: int = 1


@dataclass(frozen=True)
class ContinuitySegment:
    segment_id: str
    messages: tuple[NormalizedMessage, ...]
    char_count: int
    source_message_ids: tuple[str, ...]


def _stable_synthetic_id(index: int, role: str, content: str, timestamp_ms: int) -> str:
    digest = hashlib.sha256(
        f"{index}\x1f{role}\x1f{timestamp_ms}\x1f{content}".encode("utf-8")
    ).hexdigest()[:16]
    return f"archive-msg-{index:06d}-{digest}"


def normalize_archive_history(history: Any) -> tuple[NormalizedMessage, ...]:
    if not isinstance(history, list):
        return ()
    output: list[NormalizedMessage] = []
    seen_ids: dict[str, int] = {}
    for index, raw in enumerate(history):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("audience_scope", "") or "").strip() not in ("", "developer"):
            continue
        role = str(raw.get("role", "") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = str(raw.get("content", "") or "").strip()
        if not content:
            continue
        if role == "assistant":
            if _ELLIPSIS_ONLY_RE.match(content) or _HUM_ONLY_RE.match(content):
                continue
            if raw.get("auto_greet") or (
                raw.get("proactive") and raw.get("source") == "web_auto_greet"
            ):
                continue
        try:
            timestamp_ms = max(0, int(raw.get("timestamp", 0) or 0))
        except (TypeError, ValueError):
            timestamp_ms = 0
        base_id = str(raw.get("id", "") or "").strip() or _stable_synthetic_id(
            index, role, content, timestamp_ms
        )
        duplicate_index = seen_ids.get(base_id, 0)
        seen_ids[base_id] = duplicate_index + 1
        source_id = base_id if duplicate_index == 0 else f"{base_id}#duplicate-{duplicate_index}"
        output.append(
            NormalizedMessage(
                source_message_id=source_id,
                role=role,
                content=content,
                timestamp_ms=timestamp_ms,
                character=str(raw.get("character", "") or "").strip(),
            )
        )
    return tuple(output)


def _split_content(content: str, max_chars: int) -> list[str]:
    if len(content) <= max_chars:
        return [content]
    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        window = remaining[:max_chars]
        split_at = -1
        for marker in ("\n\n", "\n", "。", "！", "？", ".", "，", ","):
            candidate = window.rfind(marker)
            if candidate >= int(max_chars * 0.55):
                split_at = candidate + len(marker)
                break
        if split_at <= 0:
            split_at = max_chars
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    return [chunk for chunk in chunks if chunk]


def fragment_messages(
    messages: Iterable[NormalizedMessage], *, max_content_chars: int
) -> tuple[NormalizedMessage, ...]:
    if max_content_chars < 200:
        raise ValueError("max_content_chars must be at least 200")
    output: list[NormalizedMessage] = []
    for message in messages:
        chunks = _split_content(message.content, max_content_chars)
        for index, chunk in enumerate(chunks):
            output.append(
                NormalizedMessage(
                    source_message_id=message.source_message_id,
                    role=message.role,
                    content=chunk,
                    timestamp_ms=message.timestamp_ms,
                    character=message.character,
                    fragment_index=index,
                    fragment_count=len(chunks),
                )
            )
    return tuple(output)


def _message_cost(message: NormalizedMessage) -> int:
    return len(message.content) + len(message.source_message_id) + 96


def segment_messages(
    messages: Iterable[NormalizedMessage],
    *,
    max_chars: int = 8000,
    overlap_messages: int = 1,
    gap_break_ms: int = 30 * 60 * 1000,
) -> tuple[ContinuitySegment, ...]:
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500")
    if overlap_messages < 0 or overlap_messages > 4:
        raise ValueError("overlap_messages must be between 0 and 4")
    fragments = fragment_messages(messages, max_content_chars=max(200, max_chars - 160))
    if not fragments:
        return ()

    raw_segments: list[list[NormalizedMessage]] = []
    current: list[NormalizedMessage] = []
    current_cost = 0
    for message in fragments:
        cost = _message_cost(message)
        gap_break = bool(
            current
            and message.timestamp_ms > 0
            and current[-1].timestamp_ms > 0
            and message.timestamp_ms - current[-1].timestamp_ms >= gap_break_ms
            and current_cost >= int(max_chars * 0.35)
        )
        budget_break = bool(current and current_cost + cost > max_chars)
        if gap_break or budget_break:
            raw_segments.append(current)
            overlap = current[-overlap_messages:] if overlap_messages else []
            current = list(overlap)
            current_cost = sum(_message_cost(row) for row in current)
            while current and current_cost + cost > max_chars:
                removed = current.pop(0)
                current_cost -= _message_cost(removed)
        current.append(message)
        current_cost += cost
    if current:
        raw_segments.append(current)

    output: list[ContinuitySegment] = []
    for index, rows in enumerate(raw_segments):
        source_ids: list[str] = []
        for row in rows:
            if row.source_message_id not in source_ids:
                source_ids.append(row.source_message_id)
        output.append(
            ContinuitySegment(
                segment_id=f"segment-{index + 1:04d}",
                messages=tuple(rows),
                char_count=sum(_message_cost(row) for row in rows),
                source_message_ids=tuple(source_ids),
            )
        )
    return tuple(output)


def render_segment_conversation(segment: ContinuitySegment, character: str) -> str:
    lines: list[str] = []
    for message in segment.messages:
        speaker = user_name() if message.role == "user" else (message.character or character or "角色")
        fragment = (
            f" fragment={message.fragment_index + 1}/{message.fragment_count}"
            if message.fragment_count > 1
            else ""
        )
        lines.append(
            f"[source_id={message.source_message_id} timestamp_ms={message.timestamp_ms}{fragment}] "
            f"{speaker}：{message.content}"
        )
    return "\n".join(lines)
