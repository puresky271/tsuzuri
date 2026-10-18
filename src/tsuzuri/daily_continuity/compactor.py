"""Pure compaction pipeline for versioned daily continuity capsules."""

from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit
from tsuzuri.persona_registry import agent_names, setting, user_name

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .contracts import (
    GENERATOR_VERSION,
    CapsuleItem,
    CompactionResult,
    DailyContinuityCapsule,
    SchemaValidationError,
    SegmentProvenance,
    capsule_item_from_model,
    capsule_item_from_persisted,
)
from .segmenter import (
    ContinuitySegment,
    NormalizedMessage,
    normalize_archive_history,
    render_segment_conversation,
    segment_messages,
)
from .storage import (
    ContinuityStorage,
    checkpoint_path_for_capsule,
    compute_archive_sha256,
    retry_path_for_capsule,
)


LLMCaller = Callable[[str], str]


_ITEM_SCHEMA = """{
  "items": [
    {
      "kind": "event|relation_delta|open_thread|user_fact|commitment|correction",
      "text": "有主体、可独立理解的简体中文事实，最多320字",
      "subjects": ["{USER}", "角色名"],
      "source_message_ids": ["逐字复制上文 source_id"],
      "source_at_ms": 1234567890,
      "temporal_status": "past|active|planned|unknown",
      "status": "active|fulfilled|cancelled|superseded",
      "supersedes_source_message_ids": [],
      "canonical_key": "actor域:实体:属性，不含当前值、日期或状态；无法可靠确定时为空字符串",
      "actor": "user|character|shared|other|unknown",
      "due_text": "逐字复制引用消息中的单个日期或时段表达（如“明天十点半”“今晚”“放学后”）；没有明确表达时为空字符串",
      "confidence": 0.0
    }
  ]
}"""


@dataclass(frozen=True)
class CompactorConfig:
    small_day_char_threshold: int = 8000
    segment_char_budget: int = 8000
    overlap_messages: int = 1
    max_items_per_segment: int = 16
    max_final_items: int = 24
    reduce_batch_size: int = 32
    reduce_batch_target: int = 12
    rendered_summary_max_chars: int = 500

    def __post_init__(self) -> None:
        if self.small_day_char_threshold < 500 or self.segment_char_budget < 500:
            raise ValueError("compaction character budgets must be at least 500")
        if not 1 <= self.max_items_per_segment <= 40:
            raise ValueError("max_items_per_segment must be between 1 and 40")
        if not 1 <= self.max_final_items <= 80:
            raise ValueError("max_final_items must be between 1 and 80")
        if not 2 <= self.reduce_batch_size <= 80:
            raise ValueError("reduce_batch_size must be between 2 and 80")
        if not 1 <= self.reduce_batch_target < self.reduce_batch_size:
            raise ValueError("reduce_batch_target must be smaller than reduce_batch_size")
        if self.rendered_summary_max_chars < 100:
            raise ValueError("rendered_summary_max_chars must be at least 100")


def build_extract_prompt(
    *,
    character: str,
    date_world: str,
    segment: ContinuitySegment,
    max_items: int,
    whole_day: bool,
) -> str:
    _user = user_name()
    _canonical_example = str(setting("continuity_canonical_key_example", "") or "") or (
        f"character:{agent_names()[0]}:part_time_shift" if agent_names() else "character:角色:part_time_shift"
    )
    scope = "完整单日" if whole_day else f"分段 {segment.segment_id}"
    conversation = render_segment_conversation(segment, character)
    return f"""TASK=DAILY_CONTINUITY_V2_EXTRACT
你正在为角色对话系统构建跨日连续性 capsule。范围：{date_world} 的{scope}，角色：{character or '角色'}。

这不是角色台词，也不是自由叙事摘要。只抽取对未来对话仍有价值、且能被原文 source_id 直接证明的项目。

硬规则：
1. 只输出一个 JSON object；禁止 Markdown、代码围栏、analysis、解释或工具调用。
2. 顶层必须且只能有 items；items 最多 {max_items} 条。没有可靠项目时输出 {{"items":[]}}。
3. 每个 item 必须且只能包含 schema 中的全部字段；不得遗漏或增加字段。
4. source_message_ids 必须逐字复制下文 source_id；source_at_ms 必须等于其中一条引用消息的 timestamp_ms。
5. 严禁编造、泛化掉专名，或从角色回复反推{_user}没有亲口说出的事实。
6. canonical_key 标识“同一个可变 claim”，格式为 actor域:实体:属性，只用稳定名词，不含当前值、日期或 active/fulfilled 等状态；首段必须与 actor 相同。例如 user:{_user}:relationship_intent、{_canonical_example}、shared:花展:location。无法可靠确定时输出空字符串。
7. actor 是该事实/计划的责任主体：{_user}=user，当前角色=character，共同=shared；必须与 source_at_ms 对应消息的说话人一致，不要把提到某人误当成该人承诺。shared 只用于双方直接表达并确认的共同事实/计划，必须同时引用{_user}与角色的直接证据。user_fact 只能表示{_user}本人透露的事实，kind=user_fact 时 actor 必须是 user 且 source_at_ms 必须落在{_user}消息；角色自身事实请用 event 或 relation_delta，禁止写成 actor=character 的 user_fact。
8. correction 必须指出新值，并用相同 canonical_key 连接旧 claim；同一证据窗内有旧说法时，再用 supersedes_source_message_ids 指向旧 source_id。用户以“不是 X，是 Y”直接纠正前文错误理解时属于高优先级 correction，即使错误理解来自角色提问或推测，也要保留用户明确给出的新事实，不要重复旧误解。
9. planned 不得写成 past；已履约、取消或被替代的承诺必须用对应 status。
10. due_text 只能逐字复制引用消息中的单个日期或时段表达（如“明天十点半”“今晚”“傍晚”“放学后”）；不得计算、换算或输出 epoch。kind=commitment 且引用消息里存在这类表达时必须复制进 due_text；只有“以后／有空／将来／改天”这类无锚点说法才可以留空（禁止把无锚点说法改写成具体日期或时刻）。
11. 提醒、建议和安慰不是 commitment；只有说话者明确承诺之后会执行的动作才算。
12. 祝愿、提问、转述、猜测、假设、暗示，以及对方“未否认/没有反驳”都不能证明事实或计划；只能抽取责任主体亲口陈述、承诺或明确确认的内容。例：如果唯一证据只是{_user}说“祝你明天花火大会玩得开心”，而角色只回复晚安，则不能推导角色有花火大会行程，也不能改写成“{_user}知晓该行程”；对此应不生成 item。
13. rendered summary 由本地代码另行生成；不要输出 summary 字段。

严格输出 schema：
{_ITEM_SCHEMA.replace("{USER}", _user)}

带来源的对话：
<conversation>
{conversation}
</conversation>
"""


def build_reduce_prompt(items: Iterable[CapsuleItem], *, max_items: int) -> str:
    candidates = [
        {
            "candidate_id": item.item_id,
            "kind": item.kind,
            "text": item.text,
            "canonical_key": item.canonical_key,
            "actor": item.actor,
            "temporal_status": item.temporal_status,
            "status": item.status,
            "source_message_ids": list(item.source_message_ids),
            "source_at_ms": item.source_at_ms,
            "due_text": item.due_text,
            "due_precision": item.due_precision,
            "due_at_ms": item.due_at_ms,
            "confidence": item.confidence,
        }
        for item in items
    ]
    return f"""TASK=DAILY_CONTINUITY_V2_REDUCE
你只能从既有候选中选择，不能重写、合并或新增任何事实。

硬规则：
1. 只输出 JSON：{{"selected_candidate_ids":["candidate_id"]}}，禁止其他字段或文本。
2. 最多选择 {max_items} 条，candidate_id 必须逐字来自下方候选。
3. 优先级：correction > 未完成 commitment/open_thread > active user_fact > event > relation_delta。
4. fulfilled/cancelled/superseded 只在它能阻止旧计划复活时保留。
5. 同义重复只保留来源更新、confidence 更高的一条；不要按候选位置机械截头。

候选：
{json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}
"""


def _strict_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        raise SchemaValidationError("model returned empty output")
    if text.startswith("```") or "<analysis>" in text.lower():
        raise SchemaValidationError("model output contains forbidden wrapper")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"model output is not strict JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SchemaValidationError("model output root must be an object")
    return payload


def _stable_item_id(archive_sha256: str, raw: Mapping[str, Any]) -> str:
    identity = {
        "kind": raw.get("kind"),
        "text": " ".join(str(raw.get("text", "") or "").split()),
        "source_message_ids": raw.get("source_message_ids"),
        "source_at_ms": raw.get("source_at_ms"),
    }
    digest = hashlib.sha256(
        (archive_sha256 + "\x1f" + json.dumps(identity, ensure_ascii=False, sort_keys=True)).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return f"dc2-{digest}"


def _parse_extract_output(
    raw: str,
    *,
    archive_sha256: str,
    allowed_source_ids: Iterable[str],
    source_timestamps: Mapping[str, int],
    source_contents: Mapping[str, str],
    source_roles: Mapping[str, str],
    date_world: str,
    max_items: int,
) -> tuple[CapsuleItem, ...]:
    payload = _strict_json_object(raw)
    if frozenset(payload.keys()) != frozenset({"items"}):
        raise SchemaValidationError("extract output must contain only the items field")
    rows = payload.get("items")
    if not isinstance(rows, list):
        raise SchemaValidationError("items must be a list")
    if len(rows) > max_items:
        raise SchemaValidationError(f"model returned more than {max_items} items")
    items: list[CapsuleItem] = []
    dropped: dict[str, int] = {}
    first_error: SchemaValidationError | None = None
    for row in rows:
        try:
            items.append(capsule_item_from_model(
                row,
                item_id=_stable_item_id(archive_sha256, row),
                allowed_source_ids=allowed_source_ids,
                source_timestamps=source_timestamps,
                source_contents=source_contents,
                source_roles=source_roles,
                date_world=date_world,
            ))
        except SchemaValidationError as exc:
            # 2026-09-06: 单条 item 违规（如 canonical_key domain 与 actor 不匹配）
            # 不再连坐报废整批——否则同一确定性违规会被指数退避重试 5 次白烧
            # 调用，最终 quarantine，整天 continuity 全丢。丢弃坏条目、保留其余。
            if first_error is None:
                first_error = exc
            dropped[str(exc)[:80]] = dropped.get(str(exc)[:80], 0) + 1
    if dropped:
        _tsz_emit(f"[DailyContinuityV2] dropped {sum(dropped.values())} invalid extract item(s): {dropped}", flush=True)
    if rows and not items and first_error is not None:
        # 全部条目违规 = 模型系统性违约，保持 fail-closed（走重试/隔离），
        # 避免静默写出一个"看起来正常"的空 capsule。
        raise first_error
    return _dedupe_items(items)


def _parse_reduce_output(
    raw: str, candidates: Iterable[CapsuleItem], *, max_items: int
) -> tuple[CapsuleItem, ...]:
    payload = _strict_json_object(raw)
    if frozenset(payload.keys()) != frozenset({"selected_candidate_ids"}):
        raise SchemaValidationError("reduce output must contain only selected_candidate_ids")
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list) or any(not isinstance(value, str) for value in selected):
        raise SchemaValidationError("selected_candidate_ids must be a string list")
    if len(selected) > max_items:
        raise SchemaValidationError(f"reduce selected more than {max_items} candidates")
    if len(set(selected)) != len(selected):
        raise SchemaValidationError("reduce selected duplicate candidate ids")
    by_id = {item.item_id: item for item in candidates}
    unknown = [item_id for item_id in selected if item_id not in by_id]
    if unknown:
        raise SchemaValidationError(f"reduce selected unknown candidate ids: {unknown}")
    return tuple(by_id[item_id] for item_id in selected)


def _dedupe_items(items: Iterable[CapsuleItem]) -> tuple[CapsuleItem, ...]:
    output: list[CapsuleItem] = []
    seen_ids: set[str] = set()
    seen_semantics: set[tuple[str, str, tuple[str, ...]]] = set()
    for item in items:
        semantic_key = (
            item.kind,
            re.sub(r"[\s，。！？,.!?]+", "", item.text).lower(),
            tuple(sorted(item.source_message_ids)),
        )
        if item.item_id in seen_ids or semantic_key in seen_semantics:
            continue
        seen_ids.add(item.item_id)
        seen_semantics.add(semantic_key)
        output.append(item)
    return tuple(output)


def render_summary(items: Iterable[CapsuleItem], *, max_chars: int) -> str:
    rows = list(items)
    if not rows:
        return ""
    priorities = {
        "correction": 0,
        "commitment": 1,
        "open_thread": 2,
        "user_fact": 3,
        "event": 4,
        "relation_delta": 5,
    }
    labels = {
        "correction": "纠正",
        "commitment": "承诺",
        "open_thread": "待续",
        "user_fact": f"{user_name()}事实",
        "event": "事件",
        "relation_delta": "关系变化",
    }
    rows.sort(
        key=lambda item: (
            priorities.get(item.kind, 99),
            0 if item.status == "active" else 1,
            int(item.due_at_ms or 2**63 - 1),
            -int(item.source_at_ms or 0),
            -float(item.confidence),
        )
    )
    lines: list[str] = []
    for item in rows:
        status = "" if item.status == "active" else f"/{item.status}"
        line = f"- [{labels.get(item.kind, item.kind)}{status}] {item.text}"
        candidate = "\n".join(lines + [line])
        if len(candidate) <= max_chars:
            lines.append(line)
    if not lines:
        first = f"- [{labels.get(rows[0].kind, rows[0].kind)}] {rows[0].text}"
        return first[: max(1, max_chars - 1)].rstrip() + "…"
    return "\n".join(lines)


def _segment_provenance(
    segment: ContinuitySegment,
    items: Iterable[CapsuleItem],
    *,
    map_status: str,
) -> SegmentProvenance:
    timestamps = [message.timestamp_ms for message in segment.messages if message.timestamp_ms > 0]
    return SegmentProvenance(
        segment_id=segment.segment_id,
        source_message_ids=segment.source_message_ids,
        start_ms=min(timestamps) if timestamps else 0,
        end_ms=max(timestamps) if timestamps else 0,
        map_status=map_status,
        map_item_ids=tuple(item.item_id for item in items),
    )


class DailyContinuityCompactor:
    def __init__(
        self,
        *,
        config: CompactorConfig | None = None,
        storage: ContinuityStorage | None = None,
    ) -> None:
        self.config = config or CompactorConfig()
        self.storage = storage or ContinuityStorage()

    def compact_archive(
        self,
        archive: Mapping[str, Any],
        *,
        llm_caller: LLMCaller,
        capsule_path: str | Path | None = None,
        retry_state_path: str | Path | None = None,
        force: bool = False,
        now_ms: int | None = None,
    ) -> CompactionResult:
        if not isinstance(archive, Mapping):
            raise TypeError("archive must be a mapping")
        if not callable(llm_caller):
            raise TypeError("llm_caller must be callable")
        archive_sha = compute_archive_sha256(archive)
        capsule_file = Path(capsule_path) if capsule_path is not None else None
        retry_file = (
            Path(retry_state_path)
            if retry_state_path is not None
            else retry_path_for_capsule(capsule_file)
            if capsule_file is not None
            else None
        )
        checkpoint_file = (
            checkpoint_path_for_capsule(capsule_file)
            if capsule_file is not None and not force
            else None
        )
        if capsule_file is not None and not force:
            existing = self.storage.load_matching_capsule(capsule_file, archive_sha)
            if existing is not None:
                retry_state: Mapping[str, Any] = {}
                if retry_file is not None:
                    retry_state = self.storage.record_success(
                        retry_file,
                        archive_sha256=archive_sha,
                        now_ms=now_ms,
                    )
                if checkpoint_file is not None:
                    self.storage.clear_map_checkpoint(checkpoint_file)
                return CompactionResult(
                    status="idempotent",
                    archive_sha256=archive_sha,
                    capsule=existing,
                    retry_state=retry_state,
                )

        try:
            capsule = self._compact(
                archive,
                archive_sha,
                llm_caller,
                now_ms=now_ms,
                checkpoint_file=checkpoint_file,
            )
            if capsule_file is not None:
                self.storage.write_capsule(capsule_file, capsule)
            if checkpoint_file is not None:
                self.storage.clear_map_checkpoint(checkpoint_file)
            retry_state: Mapping[str, Any] = {
                "status": "succeeded",
                "archive_sha256": archive_sha,
                "attempts": 0,
                "next_retry_at_ms": 0,
            }
            if retry_file is not None:
                retry_state = self.storage.record_success(
                    retry_file, archive_sha256=archive_sha, now_ms=now_ms
                )
            return CompactionResult(
                status="completed",
                archive_sha256=archive_sha,
                capsule=capsule,
                retry_state=retry_state,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            retry_state = {
                "status": "retryable_failure",
                "archive_sha256": archive_sha,
                "attempts": 1,
                "last_error": error,
                "next_retry_at_ms": int(now_ms if now_ms is not None else time.time() * 1000),
            }
            if retry_file is not None:
                retry_state = self.storage.record_retryable_failure(
                    retry_file,
                    archive_sha256=archive_sha,
                    error=error,
                    now_ms=now_ms,
                )
            return CompactionResult(
                status="retryable_failure",
                archive_sha256=archive_sha,
                reason=error,
                retry_state=retry_state,
            )

    def _compact(
        self,
        archive: Mapping[str, Any],
        archive_sha: str,
        llm_caller: LLMCaller,
        *,
        now_ms: int | None,
        checkpoint_file: Path | None,
    ) -> DailyContinuityCapsule:
        messages = normalize_archive_history(archive.get("history", []))
        source_ids = tuple(message.source_message_id for message in messages)
        source_timestamps = {
            message.source_message_id: message.timestamp_ms for message in messages
        }
        source_contents = {
            message.source_message_id: message.content for message in messages
        }
        source_roles = {
            message.source_message_id: message.role for message in messages
        }
        character = str(archive.get("character", "") or "").strip()
        date_world = str(archive.get("date_world", "") or "").strip()
        session_id = str(archive.get("session_id", "") or "").strip()
        input_chars = sum(
            len(message.content) + len(message.source_message_id) + 96 for message in messages
        )
        model_call_count = 0
        reused_segment_count = 0

        if not messages:
            mode = "empty"
            segments: tuple[ContinuitySegment, ...] = ()
            segment_provenance: tuple[SegmentProvenance, ...] = ()
            items: tuple[CapsuleItem, ...] = ()
        elif input_chars <= self.config.small_day_char_threshold:
            mode = "single_pass"
            segment = ContinuitySegment(
                segment_id="whole-day",
                messages=messages,
                char_count=input_chars,
                source_message_ids=source_ids,
            )
            segments = (segment,)
            prompt = build_extract_prompt(
                character=character,
                date_world=date_world,
                segment=segment,
                max_items=self.config.max_final_items,
                whole_day=True,
            )
            items = _parse_extract_output(
                llm_caller(prompt),
                archive_sha256=archive_sha,
                allowed_source_ids=source_ids,
                source_timestamps=source_timestamps,
                source_contents=source_contents,
                source_roles=source_roles,
                date_world=date_world,
                max_items=self.config.max_final_items,
            )
            segment_provenance = (
                _segment_provenance(segment, items, map_status="validated"),
            )
            model_call_count = 1
        else:
            mode = "map_reduce"
            segments = segment_messages(
                messages,
                max_chars=self.config.segment_char_budget,
                overlap_messages=self.config.overlap_messages,
            )
            mapped: list[CapsuleItem] = []
            provenance_rows: list[SegmentProvenance] = []
            checkpoint = (
                self.storage.load_map_checkpoint(checkpoint_file, archive_sha)
                if checkpoint_file is not None
                else {}
            )
            checkpoint_segments = dict(checkpoint.get("segments", {}) or {})
            model_call_count = int(checkpoint.get("model_call_count", 0) or 0)
            for segment in segments:
                cached = checkpoint_segments.get(segment.segment_id)
                if isinstance(cached, Mapping):
                    try:
                        if tuple(cached.get("source_message_ids", []) or []) != segment.source_message_ids:
                            raise SchemaValidationError("checkpoint segment source ids changed")
                        segment_items = tuple(
                            capsule_item_from_persisted(row)
                            for row in cached.get("items", []) or []
                        )
                        provenance = SegmentProvenance.from_dict(
                            cached.get("provenance", {}) or {}
                        )
                        if provenance.source_message_ids != segment.source_message_ids:
                            raise SchemaValidationError("checkpoint provenance mismatch")
                        mapped.extend(segment_items)
                        provenance_rows.append(provenance)
                        reused_segment_count += 1
                        continue
                    except Exception:
                        checkpoint_segments.pop(segment.segment_id, None)
                prompt = build_extract_prompt(
                    character=character,
                    date_world=date_world,
                    segment=segment,
                    max_items=self.config.max_items_per_segment,
                    whole_day=False,
                )
                segment_items = _parse_extract_output(
                        llm_caller(prompt),
                        archive_sha256=archive_sha,
                        allowed_source_ids=segment.source_message_ids,
                        source_timestamps=source_timestamps,
                        source_contents=source_contents,
                        source_roles=source_roles,
                        date_world=date_world,
                        max_items=self.config.max_items_per_segment,
                    )
                mapped.extend(segment_items)
                provenance = _segment_provenance(
                    segment, segment_items, map_status="validated"
                )
                provenance_rows.append(provenance)
                model_call_count += 1
                if checkpoint_file is not None:
                    checkpoint_segments[segment.segment_id] = {
                        "source_message_ids": list(segment.source_message_ids),
                        "items": [item.to_dict() for item in segment_items],
                        "provenance": provenance.to_dict(),
                    }
                    self.storage.write_map_checkpoint(
                        checkpoint_file,
                        archive_sha256=archive_sha,
                        segments=checkpoint_segments,
                        model_call_count=model_call_count,
                        now_ms=now_ms,
                    )
            segment_provenance = tuple(provenance_rows)
            items, reduce_calls = self._reduce_items(_dedupe_items(mapped), llm_caller)
            model_call_count += reduce_calls

        source_times = [message.timestamp_ms for message in messages if message.timestamp_ms > 0]
        selected_sources = {
            source_id for item in items for source_id in item.source_message_ids
        }
        quality = {
            "mode": mode,
            "all_input_processed": True,
            "segment_count": len(segments),
            "model_call_count": model_call_count,
            "checkpoint_reused_segment_count": reused_segment_count,
            "input_character_count": input_chars,
            "input_message_count": len(messages),
            "selected_item_count": len(items),
            "selected_source_message_count": len(selected_sources),
            "selected_source_ratio": round(
                len(selected_sources) / max(1, len(source_ids)), 4
            ),
            "strict_schema_validated": True,
            "rendered_summary_authoritative": False,
        }
        rendered = render_summary(
            items, max_chars=self.config.rendered_summary_max_chars
        )
        return DailyContinuityCapsule(
            generator_version=GENERATOR_VERSION,
            session_id=session_id,
            character=character,
            date_world=date_world,
            source_archive_sha256=archive_sha,
            source_message_count=len(source_ids),
            source_message_ids=source_ids,
            source_start_ms=min(source_times) if source_times else 0,
            source_end_ms=max(source_times) if source_times else 0,
            segments=segment_provenance,
            items=items,
            rendered_summary=rendered,
            quality=quality,
            created_at_ms=int(now_ms if now_ms is not None else time.time() * 1000),
        )

    def _reduce_items(
        self, items: tuple[CapsuleItem, ...], llm_caller: LLMCaller
    ) -> tuple[tuple[CapsuleItem, ...], int]:
        if not items:
            return (), 0
        current = tuple(items)
        calls = 0
        while len(current) > self.config.max_final_items:
            reduced: list[CapsuleItem] = []
            for start in range(0, len(current), self.config.reduce_batch_size):
                batch = current[start : start + self.config.reduce_batch_size]
                target = min(self.config.reduce_batch_target, len(batch))
                selected = _parse_reduce_output(
                    llm_caller(build_reduce_prompt(batch, max_items=target)),
                    batch,
                    max_items=target,
                )
                calls += 1
                reduced.extend(selected)
            next_items = _dedupe_items(reduced)
            if len(next_items) >= len(current):
                raise SchemaValidationError("reduce stage made no progress")
            current = next_items
        selected = _parse_reduce_output(
            llm_caller(
                build_reduce_prompt(current, max_items=self.config.max_final_items)
            ),
            current,
            max_items=self.config.max_final_items,
        )
        return selected, calls + 1


__all__ = [
    "CompactorConfig",
    "DailyContinuityCompactor",
    "build_extract_prompt",
    "build_reduce_prompt",
    "render_summary",
]
