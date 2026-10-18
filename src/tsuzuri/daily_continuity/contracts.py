"""Strict, versioned contracts for daily continuity compaction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 2
GENERATOR_VERSION = "daily_continuity_v2_core_5"
COMPATIBLE_GENERATOR_VERSIONS = frozenset(
    {
        GENERATOR_VERSION,
        "daily_continuity_v2_core_4",
        "daily_continuity_v2_core_3",
        "daily_continuity_v2_core_2",
        "daily_continuity_v2_core_1",
    }
)

ITEM_KINDS = frozenset(
    {
        "event",
        "relation_delta",
        "open_thread",
        "user_fact",
        "commitment",
        "correction",
    }
)
TEMPORAL_STATUSES = frozenset({"past", "active", "planned", "unknown"})
ITEM_STATUSES = frozenset({"active", "fulfilled", "cancelled", "superseded"})
RESULT_STATUSES = frozenset({"completed", "idempotent", "retryable_failure"})
SEGMENT_MAP_STATUSES = frozenset({"validated", "not_required", "legacy_untracked"})
ACTORS = frozenset({"user", "character", "shared", "other", "unknown"})
DUE_PRECISIONS = frozenset({"unknown", "date", "datetime"})
UNSUPPORTED_SILENCE_MARKERS = ("未否认", "没有反驳")


class SchemaValidationError(ValueError):
    """Raised when model or persisted data violates the v2 contract."""


def _clean_text(value: Any, *, field_name: str, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if not text:
        raise SchemaValidationError(f"{field_name} must be non-empty")
    if len(text) > max_chars:
        raise SchemaValidationError(f"{field_name} exceeds {max_chars} chars")
    return text


def _clean_string_list(
    value: Any,
    *,
    field_name: str,
    max_items: int,
    max_chars: int,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{field_name} must be a list")
    if len(value) > max_items:
        raise SchemaValidationError(f"{field_name} exceeds {max_items} items")
    output: list[str] = []
    for raw in value:
        text = _clean_text(raw, field_name=field_name, max_chars=max_chars)
        if text not in output:
            output.append(text)
    if not allow_empty and not output:
        raise SchemaValidationError(f"{field_name} must be non-empty")
    return tuple(output)


@dataclass(frozen=True)
class CapsuleItem:
    item_id: str
    kind: str
    text: str
    subjects: tuple[str, ...]
    source_message_ids: tuple[str, ...]
    source_at_ms: int
    temporal_status: str
    status: str
    supersedes_source_message_ids: tuple[str, ...] = ()
    canonical_key: str = ""
    actor: str = "unknown"
    due_text: str = ""
    due_precision: str = "unknown"
    due_at_ms: int = 0
    confidence: float = 0.0
    ledger_refs: tuple[str, ...] = ()
    memory_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.item_id or "").strip():
            raise SchemaValidationError("item_id must be non-empty")
        if self.kind not in ITEM_KINDS:
            raise SchemaValidationError(f"invalid item kind: {self.kind}")
        if not self.text or len(self.text) > 320:
            raise SchemaValidationError("item text must contain 1-320 chars")
        if not self.source_message_ids:
            raise SchemaValidationError("source_message_ids must be non-empty")
        if self.source_at_ms < 0 or self.due_at_ms < 0:
            raise SchemaValidationError("timestamps must be non-negative")
        if self.temporal_status not in TEMPORAL_STATUSES:
            raise SchemaValidationError(f"invalid temporal_status: {self.temporal_status}")
        if self.status not in ITEM_STATUSES:
            raise SchemaValidationError(f"invalid item status: {self.status}")
        if len(self.canonical_key) > 120 or len(self.due_text) > 80:
            raise SchemaValidationError("canonical_key or due_text exceeds contract limit")
        if self.actor not in ACTORS:
            raise SchemaValidationError(f"invalid actor: {self.actor}")
        if self.due_precision not in DUE_PRECISIONS:
            raise SchemaValidationError(f"invalid due_precision: {self.due_precision}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SchemaValidationError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "text": self.text,
            "subjects": list(self.subjects),
            "source_message_ids": list(self.source_message_ids),
            "source_at_ms": self.source_at_ms,
            "temporal_status": self.temporal_status,
            "status": self.status,
            "supersedes_source_message_ids": list(self.supersedes_source_message_ids),
            "canonical_key": self.canonical_key,
            "actor": self.actor,
            "due_text": self.due_text,
            "due_precision": self.due_precision,
            "due_at_ms": self.due_at_ms,
            "confidence": round(float(self.confidence), 4),
            "ledger_refs": list(self.ledger_refs),
            "memory_refs": list(self.memory_refs),
        }


@dataclass(frozen=True)
class SegmentProvenance:
    segment_id: str
    source_message_ids: tuple[str, ...]
    start_ms: int
    end_ms: int
    map_status: str
    map_item_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not str(self.segment_id or "").strip():
            raise SchemaValidationError("segment_id must be non-empty")
        if len(set(self.source_message_ids)) != len(self.source_message_ids):
            raise SchemaValidationError("segment source_message_ids must be unique")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise SchemaValidationError("invalid segment time range")
        if self.map_status not in SEGMENT_MAP_STATUSES:
            raise SchemaValidationError(f"invalid segment map_status: {self.map_status}")
        if len(set(self.map_item_ids)) != len(self.map_item_ids):
            raise SchemaValidationError("segment map_item_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_message_ids": list(self.source_message_ids),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "map_status": self.map_status,
            "map_item_ids": list(self.map_item_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SegmentProvenance":
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("segment provenance must be an object")
        return cls(
            segment_id=str(raw.get("segment_id", "") or ""),
            source_message_ids=tuple(
                str(x) for x in raw.get("source_message_ids", []) or []
            ),
            start_ms=int(raw.get("start_ms", 0) or 0),
            end_ms=int(raw.get("end_ms", 0) or 0),
            map_status=str(raw.get("map_status", "") or ""),
            map_item_ids=tuple(str(x) for x in raw.get("map_item_ids", []) or []),
        )


@dataclass(frozen=True)
class DailyContinuityCapsule:
    session_id: str
    character: str
    date_world: str
    source_archive_sha256: str
    source_message_count: int
    source_message_ids: tuple[str, ...]
    source_start_ms: int
    source_end_ms: int
    segments: tuple[SegmentProvenance, ...]
    items: tuple[CapsuleItem, ...]
    rendered_summary: str
    quality: Mapping[str, Any]
    created_at_ms: int
    schema_version: int = SCHEMA_VERSION
    generator_version: str = GENERATOR_VERSION
    rendered_summary_authoritative: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError("unsupported capsule schema_version")
        if self.generator_version not in COMPATIBLE_GENERATOR_VERSIONS:
            raise SchemaValidationError("unsupported capsule generator_version")
        if not self.session_id or not self.character or not self.date_world:
            raise SchemaValidationError("session_id, character, and date_world must be non-empty")
        if len(self.source_archive_sha256) != 64:
            raise SchemaValidationError("source_archive_sha256 must be a SHA-256 hex digest")
        try:
            int(self.source_archive_sha256, 16)
        except ValueError as exc:
            raise SchemaValidationError("source_archive_sha256 is not hexadecimal") from exc
        if self.source_message_count != len(self.source_message_ids):
            raise SchemaValidationError("source_message_count does not match source_message_ids")
        if len(set(self.source_message_ids)) != len(self.source_message_ids):
            raise SchemaValidationError("source_message_ids must be unique")
        if self.source_start_ms < 0 or self.source_end_ms < self.source_start_ms:
            raise SchemaValidationError("invalid source time range")
        if self.rendered_summary_authoritative:
            raise SchemaValidationError("rendered_summary must remain non-authoritative")
        allowed_ids = set(self.source_message_ids)
        seen_segment_ids: set[str] = set()
        covered_source_ids: set[str] = set()
        for segment in self.segments:
            if segment.segment_id in seen_segment_ids:
                raise SchemaValidationError(f"duplicate segment_id: {segment.segment_id}")
            seen_segment_ids.add(segment.segment_id)
            if not set(segment.source_message_ids).issubset(allowed_ids):
                raise SchemaValidationError(
                    f"segment {segment.segment_id} references unknown source messages"
                )
            covered_source_ids.update(segment.source_message_ids)
        if covered_source_ids != allowed_ids:
            raise SchemaValidationError("segments do not cover every source message id")
        seen_item_ids: set[str] = set()
        for item in self.items:
            if item.item_id in seen_item_ids:
                raise SchemaValidationError(f"duplicate item_id: {item.item_id}")
            seen_item_ids.add(item.item_id)
            if not set(item.source_message_ids).issubset(allowed_ids):
                raise SchemaValidationError(f"item {item.item_id} references unknown source messages")
            if not set(item.supersedes_source_message_ids).issubset(allowed_ids):
                raise SchemaValidationError(f"item {item.item_id} supersedes unknown source messages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator_version": self.generator_version,
            "session_id": self.session_id,
            "character": self.character,
            "date_world": self.date_world,
            "source": {
                "archive_sha256": self.source_archive_sha256,
                "message_count": self.source_message_count,
                "message_ids": list(self.source_message_ids),
                "start_ms": self.source_start_ms,
                "end_ms": self.source_end_ms,
            },
            "segments": [segment.to_dict() for segment in self.segments],
            "items": [item.to_dict() for item in self.items],
            "rendered_summary": self.rendered_summary,
            "rendered_summary_authoritative": False,
            "quality": dict(self.quality),
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DailyContinuityCapsule":
        if not isinstance(raw, Mapping):
            raise SchemaValidationError("capsule must be an object")
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise SchemaValidationError("capsule source must be an object")
        item_rows = raw.get("items")
        if not isinstance(item_rows, list):
            raise SchemaValidationError("capsule items must be a list")
        items = tuple(capsule_item_from_persisted(row) for row in item_rows)
        segment_rows = raw.get("segments")
        if segment_rows is None:
            legacy_source_ids = tuple(str(x) for x in source.get("message_ids", []) or [])
            segments = (
                SegmentProvenance(
                    segment_id="legacy-untracked",
                    source_message_ids=legacy_source_ids,
                    start_ms=int(source.get("start_ms", 0) or 0),
                    end_ms=int(source.get("end_ms", 0) or 0),
                    map_status="legacy_untracked",
                    map_item_ids=tuple(item.item_id for item in items),
                ),
            ) if legacy_source_ids else ()
        elif isinstance(segment_rows, list):
            segments = tuple(SegmentProvenance.from_dict(row) for row in segment_rows)
        else:
            raise SchemaValidationError("capsule segments must be a list")
        quality = raw.get("quality")
        if not isinstance(quality, Mapping):
            raise SchemaValidationError("capsule quality must be an object")
        return cls(
            schema_version=int(raw.get("schema_version", 0) or 0),
            generator_version=str(raw.get("generator_version", "") or ""),
            session_id=str(raw.get("session_id", "") or ""),
            character=str(raw.get("character", "") or ""),
            date_world=str(raw.get("date_world", "") or ""),
            source_archive_sha256=str(source.get("archive_sha256", "") or ""),
            source_message_count=int(source.get("message_count", 0) or 0),
            source_message_ids=tuple(str(x) for x in source.get("message_ids", []) or []),
            source_start_ms=int(source.get("start_ms", 0) or 0),
            source_end_ms=int(source.get("end_ms", 0) or 0),
            segments=segments,
            items=items,
            rendered_summary=str(raw.get("rendered_summary", "") or ""),
            rendered_summary_authoritative=bool(raw.get("rendered_summary_authoritative", False)),
            quality=dict(quality),
            created_at_ms=int(raw.get("created_at_ms", 0) or 0),
        )


@dataclass(frozen=True)
class CompactionResult:
    status: str
    archive_sha256: str
    capsule: DailyContinuityCapsule | None = None
    reason: str = ""
    retry_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise SchemaValidationError(f"invalid result status: {self.status}")

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "idempotent"}


MODEL_ITEM_FIELDS = frozenset(
    {
        "kind",
        "text",
        "subjects",
        "source_message_ids",
        "source_at_ms",
        "temporal_status",
        "status",
        "supersedes_source_message_ids",
        "canonical_key",
        "actor",
        "due_text",
        "confidence",
    }
)


def capsule_item_from_model(
    raw: Mapping[str, Any],
    *,
    item_id: str,
    allowed_source_ids: Iterable[str],
    source_timestamps: Mapping[str, int],
    source_contents: Mapping[str, str],
    source_roles: Mapping[str, str],
    date_world: str,
) -> CapsuleItem:
    if not isinstance(raw, Mapping):
        raise SchemaValidationError("model item must be an object")
    fields = frozenset(raw.keys())
    if fields != MODEL_ITEM_FIELDS:
        missing = sorted(MODEL_ITEM_FIELDS - fields)
        extra = sorted(fields - MODEL_ITEM_FIELDS)
        raise SchemaValidationError(f"model item fields mismatch; missing={missing}, extra={extra}")
    source_ids = _clean_string_list(
        raw.get("source_message_ids"),
        field_name="source_message_ids",
        max_items=12,
        max_chars=160,
        allow_empty=False,
    )
    allowed = set(allowed_source_ids)
    if not set(source_ids).issubset(allowed):
        raise SchemaValidationError("model item references source ids outside its evidence window")
    source_at_ms = int(raw.get("source_at_ms", 0) or 0)
    expected_timestamps = {int(source_timestamps.get(source_id, 0) or 0) for source_id in source_ids}
    if source_at_ms not in expected_timestamps:
        raise SchemaValidationError("source_at_ms must equal a referenced message timestamp")
    supersedes = _clean_string_list(
        raw.get("supersedes_source_message_ids"),
        field_name="supersedes_source_message_ids",
        max_items=12,
        max_chars=160,
    )
    if not set(supersedes).issubset(allowed):
        raise SchemaValidationError("supersedes_source_message_ids contains unknown evidence")
    canonical_key = " ".join(str(raw.get("canonical_key", "") or "").split()).strip()
    if len(canonical_key) > 120 or any(char.isspace() for char in canonical_key):
        raise SchemaValidationError("canonical_key must be a compact 0-120 char identity")
    actor = str(raw.get("actor", "") or "").strip().lower()
    if actor not in ACTORS:
        raise SchemaValidationError(f"invalid actor: {actor}")
    if canonical_key and not canonical_key.startswith(f"{actor}:"):
        raise SchemaValidationError("canonical_key domain must match actor")
    kind = str(raw.get("kind", "") or "")
    temporal_status = str(raw.get("temporal_status", "") or "")
    item_text = _clean_text(raw.get("text"), field_name="text", max_chars=320)
    if any(marker in item_text for marker in UNSUPPORTED_SILENCE_MARKERS) or (
        temporal_status == "planned" and "暗示" in item_text
    ):
        raise SchemaValidationError("item text relies on unsupported indirect inference")
    if kind == "user_fact" and actor != "user":
        raise SchemaValidationError("user_fact actor must be user")
    cited_roles = {
        str(source_roles.get(source_id, ""))
        for source_id in source_ids
    }
    if actor == "shared" and not {"user", "assistant"}.issubset(cited_roles):
        raise SchemaValidationError("shared actor requires direct evidence from both speakers")
    if (
        kind in {"commitment", "correction", "user_fact"} or temporal_status == "planned"
    ) and actor in {"user", "character"}:
        expected_role = "user" if actor == "user" else "assistant"
        source_at_roles = {
            str(source_roles.get(source_id, ""))
            for source_id in source_ids
            if int(source_timestamps.get(source_id, 0) or 0) == source_at_ms
        }
        if expected_role not in source_at_roles:
            raise SchemaValidationError("actor conflicts with source_at speaker role")
    due_text = " ".join(str(raw.get("due_text", "") or "").split()).strip()
    if len(due_text) > 80:
        raise SchemaValidationError("due_text exceeds 80 chars")
    due_source_ids = [
        source_id
        for source_id in source_ids
        if due_text in " ".join(str(source_contents.get(source_id, "") or "").split())
    ] if due_text else []
    if due_text and not due_source_ids:
        raise SchemaValidationError("due_text must be copied from cited evidence")
    due_anchor_ms = source_at_ms
    if due_source_ids:
        source_at_due_ids = [
            source_id
            for source_id in due_source_ids
            if int(source_timestamps.get(source_id, 0) or 0) == source_at_ms
        ]
        if not source_at_due_ids:
            due_timestamps = {
                int(source_timestamps.get(source_id, 0) or 0)
                for source_id in due_source_ids
            }
            if len(due_timestamps) != 1:
                raise SchemaValidationError("due_text has ambiguous source timestamps")
            due_anchor_ms = next(iter(due_timestamps))
    from .temporal import derive_due_at_ms

    due = derive_due_at_ms(due_text, date_world=date_world, source_at_ms=due_anchor_ms)
    confidence_raw = raw.get("confidence")
    if isinstance(confidence_raw, bool) or not isinstance(confidence_raw, (int, float)):
        raise SchemaValidationError("confidence must be numeric")
    return CapsuleItem(
        item_id=item_id,
        kind=kind,
        text=item_text,
        subjects=_clean_string_list(
            raw.get("subjects"), field_name="subjects", max_items=8, max_chars=40
        ),
        source_message_ids=source_ids,
        source_at_ms=source_at_ms,
        temporal_status=temporal_status,
        status=str(raw.get("status", "") or ""),
        supersedes_source_message_ids=supersedes,
        canonical_key=canonical_key,
        actor=actor,
        due_text=due_text,
        due_precision=due.precision,
        due_at_ms=due.due_at_ms,
        confidence=float(confidence_raw),
    )


def capsule_item_from_persisted(raw: Mapping[str, Any]) -> CapsuleItem:
    if not isinstance(raw, Mapping):
        raise SchemaValidationError("persisted item must be an object")
    return CapsuleItem(
        item_id=str(raw.get("item_id", "") or ""),
        kind=str(raw.get("kind", "") or ""),
        text=str(raw.get("text", "") or ""),
        subjects=tuple(str(x) for x in raw.get("subjects", []) or []),
        source_message_ids=tuple(str(x) for x in raw.get("source_message_ids", []) or []),
        source_at_ms=int(raw.get("source_at_ms", 0) or 0),
        temporal_status=str(raw.get("temporal_status", "") or ""),
        status=str(raw.get("status", "") or ""),
        supersedes_source_message_ids=tuple(
            str(x) for x in raw.get("supersedes_source_message_ids", []) or []
        ),
        canonical_key=str(raw.get("canonical_key", "") or ""),
        actor=str(raw.get("actor", "unknown") or "unknown"),
        due_text=str(raw.get("due_text", "") or ""),
        due_precision=str(raw.get("due_precision", "unknown") or "unknown"),
        due_at_ms=int(raw.get("due_at_ms", 0) or 0),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        ledger_refs=tuple(str(x) for x in raw.get("ledger_refs", []) or []),
        memory_refs=tuple(str(x) for x in raw.get("memory_refs", []) or []),
    )
