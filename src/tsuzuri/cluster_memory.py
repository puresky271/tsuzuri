"""cluster_memory.py — 事件/话题聚类记忆(2026-08-16, feat/cluster-memory)

设计文档: docs/brain/memory/cluster_memory_exploration/06-implementation-plan.md

对「某一次旅游 / 某一次告白」这类具体事件、或反复谈论的同一话题做总结回忆。
与 memory_core.cluster_and_summarize_memories(embedding-only + ≤15字短语、只写不读)
不同,本模块产出带事件边界 / 人物地点 / 时间跨度 / provenance 的叙事对象:

  候选过滤(topic_labels 非空 或 salience>=0.60 的 episodic)
  → 按 topic label 分桶 → 连续时间窗 / 未覆盖的跨日话题 → 桶内 embedding 细分
  → LLM 叙事契约(strict JSON,只叙事不分组) → 硬门槛(significance>=4)
  → 本地写链(_upsert_item 补 embedding/FTS/topic/cue) + member/parent 双向引用

消费侧: select_cluster_memory_candidates() 确定性触发(显式回忆+topic命中 /
≥2 label / 1 label+entity), 由 turn_context_workspace 的 CLUSTER_MEMORY_FRAME 装配。

flags(全部默认关;CREATE/RECALL 需总开关 ENABLE 同时开启才生效):
  MEM_CLUSTER_MEMORY_ENABLE         总开关
  MEM_CLUSTER_MEMORY_CREATE_ENABLE  KAIROS 夜间创建
  MEM_CLUSTER_MEMORY_RECALL_ENABLE  召回/装配
  MEM_CLUSTER_MEMORY_AUDIT_ENABLE   创建前忠实度审计(默认开,随创建生效)
  MEM_CLUSTER_MEMORY_MODEL          叙事模型(默认回落 LLM_MODEL)
  MEM_CLUSTER_MEMORY_MAX_CALLS      单角色单次运行 LLM 叙事上限(默认 3)

注意:被动召回通道(宿主应用 `_retrieve_relevant_memories` 候选循环)渲染前排除
cluster 行(2026-08-16 决策 B,`exclude_cluster_item_rows`)——裸文本行缺
「概括非逐字原文」标注,且会绕过专属 frame 的触发门控。底层 retrieve 仍可返回
旧 cluster_summary；event/topic parent 在通用 SQL 候选限额之前排除。
"""

from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from tsuzuri.memory_core import (
    _emb_decode,
    embedding_row_matches_current,
    _json_dumps,
    _json_loads,
    _utc_now_iso,
)

from tsuzuri.time_source import utc_now
from tsuzuri.persona_registry import agent_names, user_name

CLUSTER_KIND_EVENT = "event"
CLUSTER_KIND_TOPIC = "topic"
CLUSTER_ITEM_TYPES = ("event_cluster", "topic_cluster", "cluster_summary")
ITEM_TYPE_BY_KIND = {CLUSTER_KIND_EVENT: "event_cluster", CLUSTER_KIND_TOPIC: "topic_cluster"}


def exclude_cluster_item_rows(rows: list | None) -> list:
    """被动召回通道渲染前排除 cluster 叙事行(2026-08-16 决策 B)。

    cluster 行只走专属 CLUSTER_MEMORY_FRAME(带时间范围/member 数/「概括非逐字
    原文」标注);混进普通召回行会以裸文本呈现,诱导模型把事件概括当逐字事实引用。
    event/topic parent 已在通用 SQL 候选限额前排除；这里兼容旧结果及 cluster_summary。
    """
    return [
        r
        for r in (rows or [])
        if not isinstance(r, dict) or str(r.get("item_type", "") or "") not in CLUSTER_ITEM_TYPES
    ]

# 无 topic label 但 salience>=0.60 的候选进这个桶(embedding-only 兜底)
SALIENCE_ONLY_LABEL = "_salience_only"

SIGNIFICANCE_MIN = 4  # 仿 MemoryConstellations: <4 不建聚类(碎片多≠重要)
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_TIME_GAP_HOURS = 30.0
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MAX_CLUSTERS_PER_RUN = 3
DEFAULT_MAX_BUCKET_MEMBERS = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.62
NARRATIVE_MIN_CHARS = 40   # 硬下限(prompt 要求 120-400)
NARRATIVE_MAX_CHARS = 600  # 实测自然输出 409-520 字(2026-09-16 角色A轮),上限贴着分布中心会整批赌输且重试不收敛;巨桶(800+)属桶结构问题,另行走成员喂入上限
CAPSULE_MAX_CHARS = 60     # 硬上限(prompt 要求 ≤40)
SUMMARIZER_VERSION = "cluster_memory_v2_typed"

from tsuzuri.time_source import WORLD_TZ as _WORLD_TZ  # world-clock time zone
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# flags
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or ("1" if default else "0")).strip().lower()
    return raw not in ("0", "false", "off", "no")


def mem_cluster_memory_enabled() -> bool:
    return _bool_env("MEM_CLUSTER_MEMORY_ENABLE", False)


def mem_cluster_memory_create_enabled() -> bool:
    return mem_cluster_memory_enabled() and _bool_env("MEM_CLUSTER_MEMORY_CREATE_ENABLE", False)


def mem_cluster_memory_recall_enabled() -> bool:
    return mem_cluster_memory_enabled() and _bool_env("MEM_CLUSTER_MEMORY_RECALL_ENABLE", False)


def mem_cluster_memory_audit_enabled() -> bool:
    return _bool_env("MEM_CLUSTER_MEMORY_AUDIT_ENABLE", True)


def cluster_memory_model() -> str:
    return (
        str(os.getenv("MEM_CLUSTER_MEMORY_MODEL", "") or "").strip()
        or str(os.getenv("LLM_MODEL", "") or "").strip()
        or "deepseek-flash"
    )


def cluster_memory_max_calls() -> int:
    try:
        return max(1, int(str(os.getenv("MEM_CLUSTER_MEMORY_MAX_CALLS", "3") or "3")))
    except ValueError:
        return 3


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _parse_iso(ts: str) -> datetime | None:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _world_date(dt: datetime) -> str:
    return dt.astimezone(_WORLD_TZ).date().isoformat()


def _cluster_logical_day_world(now: datetime | None = None) -> str:
    """与 kairos_dream._kairos_logical_day_world 同一 02:30 WORLD_TZ rollover 口径。"""
    current = now or utc_now()
    local = current.astimezone(_WORLD_TZ)
    if (local.hour, local.minute) < (2, 30):
        local -= timedelta(days=1)
    return local.strftime("%Y-%m-%d")


def _canonical_key(character: str, kind: str, topic_labels: list[str], start_at: str, end_at: str, context_key: str = "") -> str:
    raw = f"{character}|{kind}|{','.join(sorted(topic_labels))}|{start_at}|{end_at}|{context_key}"
    return "clu_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


# ---------------------------------------------------------------------------
# Phase 1a: 候选过滤 + 分桶(确定性)
# ---------------------------------------------------------------------------

def _load_candidates(
    core,
    character: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """topic_labels 非空 或 salience>=0.60 的 episodic 候选(未聚类、有 embedding)。"""
    ref = now or utc_now()
    cutoff = (ref - timedelta(days=max(1, int(lookback_days)))).replace(microsecond=0).isoformat()
    with core._lock:
        conn = core._connect()
        try:
            rows = conn.execute(
                """SELECT item_id, text, created_at, salience, importance, confidence,
                          topic_labels_json, cue_json, evidence_id, payload_json,
                          embedding_text, embedding_model_id, embedding_dim
                   FROM memory_items
                   WHERE character=? AND layer='episodic' AND archived=0 AND status='active'
                     AND cluster_memory_parent_id=''
                     AND embedding_text != ''
                     AND source NOT LIKE 'cross_char_daily_digest%'
                     AND created_at >= ?
                     AND (topic_labels_json != '[]' OR salience >= 0.60)
                   ORDER BY created_at ASC""",
                (str(character), cutoff),
            ).fetchall()
        finally:
            conn.close()
    out: list[dict[str, Any]] = []
    roster = agent_names()
    for r in rows:
        labels = [str(x) for x in (_json_loads(r["topic_labels_json"] or "[]", []) or []) if str(x).strip()]
        created = _parse_iso(r["created_at"])
        if created is None:
            continue
        payload = _json_loads(r["payload_json"] or "{}", {}) or {}
        cue = _json_loads(r["cue_json"] or "{}", {}) or {}
        location = str(cue.get("location_anchor") or payload.get("location") or "").strip()
        people = sorted({name for name in roster if name in str(r["text"] or "") and name != character})
        context_key = "|".join([location, ",".join(people)]).strip("|")
        out.append(
            {
                "item_id": str(r["item_id"]),
                "text": str(r["text"] or ""),
                "created_at": created,
                "salience": float(r["salience"] or 0.5),
                "importance": float(r["importance"] or 0.5),
                "confidence": float(r["confidence"] or 0.5),
                "topic_labels": labels,
                "evidence_id": str(r["evidence_id"] or ""),
                "embedding_text": str(r["embedding_text"] or ""),
                "embedding_model_id": str(r["embedding_model_id"] or ""),
                "embedding_dim": int(r["embedding_dim"] or 0),
                "payload": _json_loads(r["payload_json"] or "{}", {}) or {},
                "cue": cue,
                "context_key": context_key,
            }
        )
    return out


def _split_by_time_gap(rows: list[dict[str, Any]], gap_hours: float) -> list[list[dict[str, Any]]]:
    """按 created_at 排序后,相邻间隔 > gap_hours 即切成新时间窗桶。"""
    ordered = sorted(rows, key=lambda r: r["created_at"])
    buckets: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in ordered:
        if current:
            gap = (row["created_at"] - current[-1]["created_at"]).total_seconds() / 3600.0
            if gap > gap_hours:
                buckets.append(current)
                current = []
        current.append(row)
    if current:
        buckets.append(current)
    return buckets


def _split_by_embedding(
    rows: list[dict[str, Any]],
    *,
    min_cluster_size: int,
    similarity_threshold: float,
) -> list[list[dict[str, Any]]]:
    """桶内 embedding 层次聚类(复用 memory_core 既有 scipy 路径的口径)。

    桶太小(< 2*min_size)或有效 embedding 不足时,整桶作为单组返回——
    同 topic + 同时间窗已经足够相似,细分没有意义。
    """
    if len(rows) < 2 * min_cluster_size:
        return [rows]
    valid: list[dict[str, Any]] = []
    vecs: list[Any] = []
    for row in rows:
        if not embedding_row_matches_current(row):
            continue
        vec = _emb_decode(row.get("embedding_text") or "")
        if vec is None or len(vec) == 0:
            continue
        valid.append(row)
        vecs.append(vec)
    if len(valid) < 2 * min_cluster_size:
        return [rows]
    try:
        import numpy as np
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        mat = np.array(vecs)
        dist = pdist(mat, metric="cosine")
        link = linkage(dist, method="average")
        labels = fcluster(link, t=1.0 - similarity_threshold, criterion="distance")
    except Exception as err:  # numpy/scipy 缺失时退化为整桶单组
        _tsz_emit(f"[ClusterMemory] embedding split fallback: {err}", flush=True)
        return [rows]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, cid in enumerate(labels):
        groups[int(cid)].append(valid[idx])
    return [members for members in groups.values() if len(members) >= min_cluster_size]


def build_cluster_plan(
    core,
    character: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    time_gap_hours: float = DEFAULT_TIME_GAP_HOURS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_clusters: int = DEFAULT_MAX_CLUSTERS_PER_RUN,
    max_bucket_members: int = DEFAULT_MAX_BUCKET_MEMBERS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    now: datetime | None = None,
) -> dict[str, Any]:
    """候选 → topic 桶 → 连续时间窗或稀疏跨日话题 → embedding 细分 → 门槛。

    返回 {"scanned", "groups":[{status, reason, member_ids, ...}]}, 不写库、不调 LLM。
    status: passed / skipped(reason 见每组)。
    """
    char = str(character or "").strip()
    candidates = _load_candidates(core, char, lookback_days=lookback_days, now=now)
    plan: dict[str, Any] = {
        "character": char,
        "scanned": len(candidates),
        "lookback_days": lookback_days,
        "time_gap_hours": time_gap_hours,
        "groups": [],
        "diagnostics": {"bucket_count": 0, "buckets": [], "rejection_counts": {}},
    }
    def reject(reason):
        counts = plan["diagnostics"]["rejection_counts"]
        counts[reason] = counts.get(reason, 0) + 1

    if len(candidates) < min_cluster_size:
        reject("insufficient_candidates")
        return plan

    # 1) 按 topic label 分桶(一条 row 含多个 label 时进多个桶;无 label 进 salience 兜底桶)
    label_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["topic_labels"]:
            for label in row["topic_labels"]:
                label_groups[f"{label}|{row.get('context_key', '')}"] .append(row)
        else:
            label_groups[SALIENCE_ONLY_LABEL].append(row)

    plan["diagnostics"]["bucket_count"] = len(label_groups)
    for key, rows in sorted(label_groups.items()):
        bounded = sorted(rows, key=lambda r: r["created_at"])[-max_bucket_members:]
        distinct = len({re.sub(r"\s+", "", str(m["text"])) for m in bounded})
        plan["diagnostics"]["buckets"].append({
            "bucket": key, "input_members": len(rows), "bounded_members": len(bounded),
            "distinct_members": distinct, "duplicate_members": len(bounded) - distinct,
            "trimmed_members": len(rows) - len(bounded),
        })

    # 2) 桶内:时间窗切分 → 每窗 embedding 细分,收集候选组
    raw_groups: list[dict[str, Any]] = []
    for bucket_key in sorted(label_groups):
        rows = label_groups[bucket_key]
        label = bucket_key.split("|", 1)[0]
        if len(rows) < min_cluster_size:
            reject("bucket_too_small")
            continue
        # 桶过大时取最近 max_bucket_members 条(确定性)
        rows = sorted(rows, key=lambda r: r["created_at"])
        if len(rows) > max_bucket_members:
            rows = rows[-max_bucket_members:]
        for window in _split_by_time_gap(rows, time_gap_hours):
            if len(window) < min_cluster_size:
                reject("window_too_small")
                continue
            for members in _split_by_embedding(
                window,
                min_cluster_size=min_cluster_size,
                similarity_threshold=similarity_threshold,
            ):
                if len(members) < min_cluster_size:
                    reject("embedding_group_too_small")
                    continue
                # live 数据存在同文重复行(同一事实被重复抽取):去重后不足
                # min_cluster_size 条的组不成立(3 条相同文本不是一个事件)
                distinct_texts = {re.sub(r"\s+", "", str(m["text"])) for m in members}
                if len(distinct_texts) < min_cluster_size:
                    reject("insufficient_distinct_members")
                    continue
                members = sorted(members, key=lambda r: r["created_at"])
                raw_groups.append(
                    {
                        "label": label,
                        "generation_scope": "time_window",
                        "members": members,
                        "context_key": members[0].get("context_key", ""),
                        "start_at": _world_date(members[0]["created_at"]),
                        "end_at": _world_date(members[-1]["created_at"]),
                    }
                )

    # Sparse recurring discussions should not disappear between event windows.
    # Existing window groups keep ownership; topic candidates use uncovered evidence.
    window_ids = {m["item_id"] for group in raw_groups for m in group["members"]}
    for bucket_key in sorted(label_groups):
        label = bucket_key.split("|", 1)[0]
        if label == SALIENCE_ONLY_LABEL:
            continue
        remaining = [m for m in label_groups[bucket_key] if m["item_id"] not in window_ids]
        remaining = sorted(remaining, key=lambda m: m["created_at"])[-max_bucket_members:]
        if len(remaining) < min_cluster_size:
            reject("recurring_uncovered_too_small")
            continue
        for members in _split_by_embedding(remaining, min_cluster_size=min_cluster_size,
                                           similarity_threshold=similarity_threshold):
            days = {_world_date(m["created_at"]) for m in members}
            distinct = {re.sub(r"\s+", "", str(m["text"])) for m in members}
            if len(days) < 2 or len(distinct) < min_cluster_size:
                reject("recurring_insufficient_days" if len(days) < 2 else "recurring_insufficient_distinct_members")
                continue
            members = sorted(members, key=lambda m: m["created_at"])
            raw_groups.append({"label": label, "generation_scope": "recurring_topic", "members": members,
                               "start_at": _world_date(members[0]["created_at"]), "end_at": _world_date(members[-1]["created_at"]),
                               "context_key": members[0].get("context_key", "")})

    # Event windows retain precedence over the new, broader topic route.
    raw_groups.sort(key=lambda g: (g["generation_scope"] != "time_window", -len(g["members"]), g["start_at"], g["label"]))
    used: set[str] = set()
    passed = 0
    for group in raw_groups:
        member_ids = [m["item_id"] for m in group["members"]]
        entry = {
            "label": group["label"],
            "generation_scope": group["generation_scope"],
            "member_ids": member_ids,
            "member_count": len(member_ids),
            "start_at": group["start_at"],
            "end_at": group["end_at"],
            "sample_texts": [m["text"][:80] for m in group["members"][:3]],
            "status": "skipped",
            "reason": "",
        }
        if passed >= max_clusters:
            entry["reason"] = "run_cap_reached"
        elif any(mid in used for mid in member_ids):
            entry["reason"] = "members_consumed"
        else:
            entry["status"] = "passed"
            used.update(member_ids)
            passed += 1
        plan["groups"].append(entry)
        if entry["reason"]:
            reject(entry["reason"])
    return plan


# ---------------------------------------------------------------------------
# Phase 1b: LLM 叙事契约(strict JSON,只叙事不分组)
# ---------------------------------------------------------------------------

_NARRATIVE_PROMPT = """你是记忆整理器。下面是角色「{character}」的 {count} 条记忆片段，具有候选标签「{label}」，按时间升序给出。标签相同不证明叙事连贯。

候选范围：{scope_guidance}

任务：判断这些片段是否构成「同一件事 / 同一话题脉络」，并产出一段事件叙事。

硬规则：
1. 只能复述片段里出现的事实，禁止编造、禁止推断性格或稳定偏好
2. 第三人称（用户叫「{user}」、角色叫「{character}」），禁用「我 / 你」
3. 保留具体地点 / 活动 / 时间范围原样；禁止相对日期（昨天 / 上周 / 刚才），日期一律 YYYY-MM-DD
4. event 必须是同一具体事件；topic 必须围绕同一具体问题/计划反复讨论，保留每次讨论的时间和变化。只有宽泛领域相同（如「妈妈做饭」和「妈妈打电话」）、没有共同问题的片段必须 coherent=false。
5. significance 1-10 分类型判断：event 的明确事件经过=4-6，重要人生事件=7-10；topic 的同一具体问题/计划持续推进、约束或观点有新增=4-6，重要人生选择的长期讨论=7-10。两类的寒暄、重复而无新内容的碎片都为1-3；计划尚未执行不等于讨论没有记忆价值。片段数量多不等于重要。
6. cluster_kind：event = 有明确时间边界的一次事件（旅行 / 告白 / 演出）；topic = 围绕同一话题的多次谈论
7. timeline 2-6 条，按日期排列，detail ≤40 字，significance ≤30 字（为什么值得记住）
8. topic 中的提议、计划、猜测不能写成已发生的行动或稳定偏好；明确保留未定、未成行的状态。
9. 叙事 120-{narrative_budget} 字，目标 250 字左右：只写主线、经过与结果，逐日细节交给 timeline，不要在叙事里逐条复述 timeline；压缩重复表达但保留所有关键事实。超过 {narrative_budget} 字将整体拒绝，不会截断结尾。

只输出严格 JSON，不要任何其他文字：
{{
  "coherent": true 或 false,
  "cluster_kind": "event" 或 "topic",
  "title": "≤20字标题",
  "narrative": "120-{narrative_budget}字事件叙事",
  "timeline": [{{"date": "YYYY-MM-DD", "detail": "...", "significance": "..."}}],
  "people": ["出现的人物"],
  "places": ["出现的地点"],
  "start_at": "YYYY-MM-DD",
  "end_at": "YYYY-MM-DD",
  "capsule": "≤40字一句话概括",
  "significance": 1到10的整数,
  "confidence": 0.0到1.0
}}

记忆片段：
{members_text}"""


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    """容错解析 LLM 输出中的第一个 JSON object(去 fence / 截取花括号段)。"""
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except (ValueError, TypeError):
            return None
    return None


def build_narrative_prompt(member_items: list[dict[str, Any]], character: str, label: str) -> str:
    """渲染叙事契约 prompt(确定性;_default_llm_narrator 与 prompt-diff 门禁 dump 共用)。"""
    # 同文重复行只给 LLM 看一遍(member_ids 保持原样,provenance 不受影响)
    seen_texts: set[str] = set()
    member_lines: list[str] = []
    for m in member_items[:DEFAULT_MAX_BUCKET_MEMBERS]:
        key = re.sub(r"\s+", "", str(m["text"]))
        if key in seen_texts:
            continue
        seen_texts.add(key)
        member_lines.append(f"- [{_world_date(m['created_at'])}] {m['text'][:200]}")
    members_text = "\n".join(member_lines)
    scope = member_items[0].get("_generation_scope", "time_window") if member_items else "time_window"
    scope_guidance = (
        "跨日期的稀疏讨论，只能生成 topic。即使有相同人物地点，也不能拼成一次已发生的事件；没有具体话题脉络时 coherent=false。"
        if scope == "recurring_topic" else
        "连续时间窗。一次完整经历用 event；同一具体问题的多次讨论用 topic。禁止用时间长短代替语义判断。"
    )
    return _NARRATIVE_PROMPT.format(character=character, user=user_name(), count=len(member_items), label=label,
                                    members_text=members_text, scope_guidance=scope_guidance,
                                    narrative_budget=NARRATIVE_MAX_CHARS)


def _narrative_violations(parsed: dict[str, Any], scope: str) -> list[str]:
    """返回 coherent 草稿违反叙事契约的具体原因;空列表 = 合规。

    只做描述,不改判定语义(与 _normalize_draft 的上界保持一致)。
    """
    out: list[str] = []
    narrative = str(parsed.get("narrative") or "")
    if len(narrative) > NARRATIVE_MAX_CHARS:
        out.append(
            f"叙事 {len(narrative)} 字超过 {NARRATIVE_MAX_CHARS} 上限，压缩重复表达、逐日细节交给 timeline，写 250 字左右"
        )
    kind = parsed.get("cluster_kind")
    if kind not in (CLUSTER_KIND_EVENT, CLUSTER_KIND_TOPIC):
        out.append(f"cluster_kind {kind!r} 必须是 event 或 topic")
    elif scope == "recurring_topic" and kind != CLUSTER_KIND_TOPIC:
        out.append("recurring_topic 候选只能生成 topic，不能把碎片拼成一次已发生的事件")
    return out


def _narrative_retry_note(violations: list[str] | None = None) -> str:
    """叙事器重试时追加在原 prompt 末尾的纠偏行;violations 非空时回传具体违规。"""
    detail = f"（{'；'.join(violations)}）" if violations else ""
    return (
        f"\n上次输出类型或叙事字数不合规{detail}：叙事须 120-{NARRATIVE_MAX_CHARS} 字、目标 250 字左右，"
        "逐日细节交给 timeline；请按候选范围重写，保留关键事实，不得把计划写成经历。"
    )


def _narrative_truncation_note() -> str:
    """输出被 max_tokens 截断时追加的压缩提示(截断的 JSON 无法解析,原样重发低温下近似复现)。"""
    return (
        f"\n上次输出 JSON 未写完就被截断：整篇太长。请先压缩叙事到 {NARRATIVE_MAX_CHARS} 字内、"
        "timeline 压到 2-4 条，再输出完整 JSON。"
    )


def _default_llm_narrator(member_items: list[dict[str, Any]], character: str, label: str) -> dict[str, Any] | None:
    """默认叙事器:OpenAI 兼容客户端,低温 strict JSON,空响应重试 1 次。

    返回解析后的 raw draft dict;LLM 不可用 / 解析失败返回 None。
    """
    api_key = str(os.getenv("LLM_API_KEY", "") or "").strip()
    if not api_key:
        _tsz_emit("[ClusterMemory] LLM_API_KEY missing, narrator unavailable", flush=True)
        return None
    base_url = str(os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1") or "").strip()
    model = cluster_memory_model()
    scope = member_items[0].get("_generation_scope", "time_window") if member_items else "time_window"
    prompt = build_narrative_prompt(member_items, character, label)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        parsed: dict[str, Any] | None = None
        # 1600 会截断大桶的完整 JSON(2026-09-16 角色A completion 恰好=1600):放开到 2400/3200
        for attempt, max_tokens in enumerate((2400, 3200), start=1):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},
            )
            try:
                from tsuzuri.llm_usage import capture_llm_usage as _cap

                _cap(resp, label="cluster_memory_narrative")
            except Exception:
                pass
            choice = resp.choices[0]
            raw = str(choice.message.content or "").strip()
            if not raw:
                _tsz_emit(f"[ClusterMemory] empty narrative response, retry (attempt {attempt}/2)", flush=True)
                continue
            parsed = _extract_json_object(raw)
            if parsed is not None:
                violations = _narrative_violations(parsed, scope) if parsed.get("coherent") is True else []
                if violations:
                    parsed = None
                    prompt += _narrative_retry_note(violations)
                    _tsz_emit(
                        f"[ClusterMemory] narrative constraint violated, retry (attempt {attempt}/2): "
                        f"{'; '.join(violations)}",
                        flush=True,
                    )
                    continue
                break
            if str(getattr(choice, "finish_reason", "") or "") == "length":
                prompt += _narrative_truncation_note()
            _tsz_emit(f"[ClusterMemory] narrative JSON parse failed, retry (attempt {attempt}/2): {raw[:120]}", flush=True)
        return parsed
    except Exception as err:
        _tsz_emit(f"[ClusterMemory] narrator error: {err}", flush=True)
        return None


def _normalize_draft(
    raw: dict[str, Any],
    members: list[dict[str, Any]],
    character: str,
    label: str,
    *,
    generation_scope: str = "time_window",
) -> dict[str, Any] | None:
    """把 LLM raw 输出规整成 draft;不连贯 / 低显著 / 字段残缺返回 None。

    topic_labels / salience / importance / evidence / canonical_key 全部确定性
    从 member 行推导,不信 LLM。
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("coherent") is not True:
        return None

    start_at = str(raw.get("start_at") or "").strip()
    end_at = str(raw.get("end_at") or "").strip()
    member_start = _world_date(members[0]["created_at"])
    member_end = _world_date(members[-1]["created_at"])
    if not _DATE_RE.match(start_at):
        start_at = member_start
    if not _DATE_RE.match(end_at):
        end_at = member_end
    if start_at > end_at:
        start_at, end_at = end_at, start_at

    kind = str(raw.get("cluster_kind") or "").strip().lower()
    if kind not in (CLUSTER_KIND_EVENT, CLUSTER_KIND_TOPIC):
        return None
    if generation_scope == "recurring_topic" and kind != CLUSTER_KIND_TOPIC:
        return None

    narrative = str(raw.get("narrative") or "").strip()
    if len(narrative) < NARRATIVE_MIN_CHARS:
        return None
    if len(narrative) > NARRATIVE_MAX_CHARS:
        return None

    try:
        significance = max(1, min(10, int(raw.get("significance") or 0)))
    except (TypeError, ValueError):
        significance = 0
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.6)))
    except (TypeError, ValueError):
        confidence = 0.6

    def _str_list(value: Any, cap: int) -> list[str]:
        if not isinstance(value, list):
            return []
        out = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text[:40])
            if len(out) >= cap:
                break
        return out

    timeline: list[dict[str, str]] = []
    for entry in (raw.get("timeline") or [])[:8]:
        if not isinstance(entry, dict):
            continue
        date_s = str(entry.get("date") or "").strip()
        if not _DATE_RE.match(date_s):
            continue
        timeline.append(
            {
                "date": date_s,
                "detail": str(entry.get("detail") or "")[:80],
                "significance": str(entry.get("significance") or "")[:60],
            }
        )

    # member label 并集按频次收敛到 top-5:过宽会让召回侧 ≥2 label 规则轻易误触发
    label_counts = Counter(lbl for m in members for lbl in m["topic_labels"])
    topic_labels = [lbl for lbl, _ in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]]
    avg_sal = sum(m["salience"] for m in members) / len(members)
    avg_imp = sum(m["importance"] for m in members) / len(members)
    member_ids = [m["item_id"] for m in members]
    evidence_ids = [m["evidence_id"] for m in members if m["evidence_id"]][:40]

    return {
        "cluster_kind": kind,
        "generation_scope": generation_scope,
        "context_key": str(members[0].get("context_key", "") or ""),
        "item_type": ITEM_TYPE_BY_KIND[kind],
        "title": str(raw.get("title") or "").strip()[:30],
        "text": narrative,
        "capsule": str(raw.get("capsule") or "").strip()[:CAPSULE_MAX_CHARS],
        "timeline": timeline,
        "people": _str_list(raw.get("people"), 10),
        "places": _str_list(raw.get("places"), 10),
        "start_at": start_at,
        "end_at": end_at,
        "topic_labels": topic_labels,
        "significance": significance,
        "confidence": confidence,
        "salience": min(1.0, avg_sal + 0.10),
        "importance": min(1.0, avg_imp + 0.10),
        "member_ids": member_ids,
        "evidence_ids": evidence_ids,
        "canonical_key": _canonical_key(character, kind, topic_labels, start_at, end_at, str(members[0].get("context_key", "") or "")),
    }


_AUDIT_PROMPT = """你是记忆质检员。下面是一段由记忆片段汇总出的「事件叙事」,以及用于汇总的全部 {count} 条原始片段（完整、无抽样）。

判定叙事与原始片段的关系：
- faithful：叙事中的事实都能在片段里找到来源
- distorted：叙事有事实性偏差（张冠李戴 / 时间地点错误 / 夸大）
- fabricated：叙事包含片段里完全没有的编造内容

注意：片段集是全量的，「片段里没有」可以作为判定依据；但叙事允许跨片段组合与合理概括，只有事实性错误才判 distorted / fabricated。片段前缀 [YYYY-MM-DD] 是片段的元数据日期（来自记录时间，正文未必含日期），叙事中的具体日期以此为准，不算编造。判 distorted / fabricated 时 reason 必须引用冲突片段的原句。

只输出严格 JSON：{{"verdict": "faithful" 或 "distorted" 或 "fabricated", "reason": "≤60字"}}

事件叙事：
{narrative}

全部原始片段：
{samples_text}"""


def _default_llm_auditor(narrative: str, sample_texts: list[str], character: str) -> dict[str, Any] | None:
    """忠实度审计(仿 MemoryConstellations auditNewEpisodes)。

    全量对照(去重后 ≤30 条):抽样 ≤5 条在大组上会把「未抽样片段里的
    事实」误判为无来源(2026-08-16 快照实测 6 例假 distorted)。
    """
    api_key = str(os.getenv("LLM_API_KEY", "") or "").strip()
    if not api_key:
        return None
    base_url = str(os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1") or "").strip()
    seen: set[str] = set()
    uniq: list[str] = []
    for t in sample_texts:
        key = re.sub(r"\s+", "", str(t))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(str(t))
    samples = "\n".join(f"- {t[:200]}" for t in uniq[:DEFAULT_MAX_BUCKET_MEMBERS])
    prompt = _AUDIT_PROMPT.format(count=len(uniq[:DEFAULT_MAX_BUCKET_MEMBERS]), narrative=narrative, samples_text=samples)
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        resp = client.chat.completions.create(
            model=cluster_memory_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )
        try:
            from tsuzuri.llm_usage import capture_llm_usage as _cap

            _cap(resp, label="cluster_memory_audit")
        except Exception:
            pass
        return _extract_json_object(str(resp.choices[0].message.content or ""))
    except Exception as err:
        _tsz_emit(f"[ClusterMemory] auditor error: {err}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Phase 1c: 编排(dry_run 优先)
# ---------------------------------------------------------------------------

def create_cluster_memories(
    core,
    character: str,
    *,
    dry_run: bool = True,
    llm_narrator: Callable[[list[dict[str, Any]], str, str], dict[str, Any] | None] | None = None,
    llm_auditor: Callable[[str, list[str], str], dict[str, Any] | None] | None = None,
    audit: bool | None = None,
    **plan_kwargs: Any,
) -> dict[str, Any]:
    """事件聚类主入口。dry_run=True 只产 draft 不写库(默认)。

    llm_narrator(member_items, character, label) -> raw draft dict | None
    llm_auditor(narrative, sample_texts, character) -> {"verdict": ...} | None
    """
    char = str(character or "").strip()
    narrator = llm_narrator or _default_llm_narrator
    auditor = llm_auditor or _default_llm_auditor
    do_audit = mem_cluster_memory_audit_enabled() if audit is None else bool(audit)
    max_calls = cluster_memory_max_calls()

    plan = build_cluster_plan(core, char, **plan_kwargs)
    result: dict[str, Any] = {
        "character": char,
        "dry_run": dry_run,
        "plan": plan,
        "drafts": [],
        "created": 0,
        "parents_created": [],
        "quarantined": [],
        "narrator_calls": 0,
        "narrator_failed": 0,
        "write_outcomes": [],
    }
    # 编排时把 member 对象挂回组(plan 里只存 id,便于 JSON 报告)
    id_to_member = {m["item_id"]: m for m in _load_candidates(
        core, char,
        lookback_days=int(plan_kwargs.get("lookback_days", DEFAULT_LOOKBACK_DAYS)),
        now=plan_kwargs.get("now"),
    )}

    for group in plan["groups"]:
        if group["status"] != "passed":
            continue
        if result["narrator_calls"] >= max_calls:
            group["status"] = "skipped"
            group["reason"] = "max_calls_reached"
            continue
        scope = group.get("generation_scope", "time_window")
        members = [{**id_to_member[mid], "_generation_scope": scope} for mid in group["member_ids"] if mid in id_to_member]
        if len(members) < DEFAULT_MIN_CLUSTER_SIZE:
            group["status"] = "skipped"
            group["reason"] = "members_missing"
            continue
        result["narrator_calls"] += 1
        raw = narrator(members, char, group["label"])
        draft = _normalize_draft(raw, members, char, label=group["label"], generation_scope=scope)
        if draft is None:
            group["status"] = "skipped"
            group["reason"] = "narrator_rejected"
            result["narrator_failed"] += 1
            continue
        if draft["significance"] < SIGNIFICANCE_MIN:
            group["status"] = "skipped"
            group["reason"] = "low_significance"
            result["drafts"].append({**draft, "dropped": "low_significance"})
            continue
        group["status"] = "narrated"
        result["drafts"].append(draft)

    if dry_run:
        return result

    for draft in result["drafts"]:
        if draft.get("dropped"):
            continue
        outcome = {"canonical_key": draft["canonical_key"], "audit_verdict": "disabled" if not do_audit else "unavailable", "status": "pending"}
        result["write_outcomes"].append(outcome)
        if do_audit:
            # 样本带元数据日期前缀:日期来自 created_at,审计不应把正文无日期当编造
            sample_lines = [
                f"[{_world_date(id_to_member[mid]['created_at'])}] {id_to_member[mid]['text']}"
                for mid in draft["member_ids"]
                if mid in id_to_member
            ]
            verdict = auditor(draft["text"], sample_lines, char)
            draft["audit"] = verdict
            if isinstance(verdict, dict):
                v = str(verdict.get("verdict") or "").strip().lower()
                outcome["audit_verdict"] = v or "invalid"
                if v == "fabricated":
                    outcome["status"] = "quarantined"
                    result["quarantined"].append(
                        {"canonical_key": draft["canonical_key"], "reason": str(verdict.get("reason") or "")[:120]}
                    )
                    continue
                if v == "distorted":
                    draft["confidence"] = round(draft["confidence"] * 0.5, 3)
        if _canonical_key_exists(core, char, draft["canonical_key"]):
            outcome["status"] = "duplicate_canonical_key"
            result["quarantined"].append({"canonical_key": draft["canonical_key"], "reason": "duplicate_canonical_key"})
            continue
        try:
            item_id = write_cluster_memory(core, char, draft)
        except Exception as err:
            outcome["status"] = "write_error"
            outcome["error_type"] = type(err).__name__
            _tsz_emit(f"[ClusterMemory] write failed ({draft['canonical_key']}): {err}", flush=True)
            continue
        result["created"] += 1
        outcome.update(status="created", item_id=item_id)
        result["parents_created"].append(
            {
                "item_id": item_id,
                "item_type": draft["item_type"],
                "title": draft["title"],
                "member_count": len(draft["member_ids"]),
                "significance": draft["significance"],
                "canonical_key": draft["canonical_key"],
            }
        )
    return result


# ---------------------------------------------------------------------------
# Phase 2: 写链(_upsert_item 完整写链 + cluster 列 + children 双向)
# ---------------------------------------------------------------------------

def _canonical_key_exists(core, character: str, canonical_key: str) -> bool:
    with core._lock:
        conn = core._connect()
        try:
            row = conn.execute(
                """SELECT item_id FROM memory_items
                   WHERE character=? AND item_type IN ('event_cluster', 'topic_cluster')
                     AND archived=0 AND status='active'
                     AND cluster_meta_json LIKE '%' || ? || '%'
                   LIMIT 1""",
                (str(character), str(canonical_key)),
            ).fetchone()
        finally:
            conn.close()
    return row is not None


def write_cluster_memory(core, character: str, draft: dict[str, Any]) -> str:
    """单个 draft 落库,返 item_id。

    走 _upsert_item(自动补 cue/lifecycle/dense+sparse embedding/FTS),
    再补 _upsert_item 不管的 cluster 专属列(member_ids_json/cluster_meta_json),
    最后把 member 行标 cluster_memory_parent_id(member 保持只读证据,不改文本)。
    该专用边与旧 embedding-summary 的 parent_id 隔离，两个管线可独立消费证据。
    """
    char = str(character or "").strip()
    item_id = uuid.uuid4().hex[:24]
    now_iso = _utc_now_iso()
    payload = {
        "title": draft["title"],
        "capsule": draft["capsule"],
        "timeline": draft["timeline"],
        "people": draft["people"],
        "places": draft["places"],
        "start_at": draft["start_at"],
        "end_at": draft["end_at"],
        "cluster_kind": draft["cluster_kind"],
        "context_key": draft.get("context_key", ""),
        "canonical_key": draft["canonical_key"],
        "member_count": len(draft["member_ids"]),
        "memory_meta": {"source_kind": "cluster_memory", "evidence_tier": "derived"},
    }
    cluster_meta = {
        "method": "topic_bucket+time_gap+embedding",
        "generation_scope": draft.get("generation_scope", "time_window"),
        "context_key": draft.get("context_key", ""),
        "trigger": "kairos_nightly",
        "canonical_key": draft["canonical_key"],
        "significance": draft["significance"],
        "label": draft["topic_labels"][0] if draft["topic_labels"] else SALIENCE_ONLY_LABEL,
        "generated_at": now_iso,
        "summarizer_version": SUMMARIZER_VERSION,
        "audit": draft.get("audit"),
        "member_preview": [mid for mid in draft["member_ids"][:3]],
    }
    evidence_hash = hashlib.sha256(
        json.dumps(draft["member_ids"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    item = {
        "item_id": item_id,
        "character": char,
        "layer": "semantic",
        "memory_role": "semantic",
        "engram_type": draft["item_type"],
        "item_type": draft["item_type"],
        "text": draft["text"],
        "payload_json": _json_dumps(payload),
        "salience": draft["salience"],
        "confidence": draft["confidence"],
        "importance": draft["importance"],
        "novelty": 0.6,
        "status": "active",
        "source": "cluster_memory",
        "evidence_id": evidence_hash,
        "created_at": now_iso,
        "updated_at": now_iso,
        "topic_labels_json": _json_dumps(draft["topic_labels"]),
    }
    core._upsert_item(item)

    with core._lock:
        conn = core._connect()
        try:
            conn.execute(
                "UPDATE memory_items SET member_ids_json=?, cluster_meta_json=? WHERE item_id=?",
                (_json_dumps(draft["member_ids"]), _json_dumps(cluster_meta), item_id),
            )
            for mid in draft["member_ids"]:
                conn.execute(
                    "UPDATE memory_items SET cluster_memory_parent_id=?, updated_at=? WHERE item_id=?",
                    (item_id, now_iso, mid),
                )
            try:
                conn.execute(
                    """INSERT INTO evolution_action_log
                    (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        f"cluster_memory_{now_iso}", 0, "cluster_memory_create",
                        item_id,
                        _json_dumps({"members": draft["member_ids"], "canonical_key": draft["canonical_key"]}),
                        _json_dumps({"title": draft["title"], "text_preview": draft["text"][:120]}),
                        "applied", now_iso,
                    ),
                )
            except Exception:
                pass
            conn.commit()
        finally:
            conn.close()
    return item_id


def run_kairos_cluster_memory(core, character: str, *, llm_narrator=None, now: datetime | None = None) -> dict[str, Any]:
    """KAIROS 夜间入口:02:30 WORLD_TZ 逻辑日幂等 + 预算事件。"""
    char = str(character or "").strip()
    logical_day = _cluster_logical_day_world(now)
    meta_key = f"kairos.cluster_memory.created.{char}.{logical_day}"
    previous = core.get_meta_state_json(meta_key, {}) or {}
    if isinstance(previous, dict) and previous.get("terminal"):
        return {"ok": True, "status": "idempotent", "created": 0, "meta_key": meta_key}
    result = create_cluster_memories(core, char, dry_run=False, llm_narrator=llm_narrator, now=now)
    # 全部 narrator 调用都失败(传输错误/解析失败/内容被拒)时不消费幂等键,
    # 下一个逻辑日或手动重跑可重试(00 §6.3:失败重试不消费幂等键)。
    narrator_calls = int(result.get("narrator_calls", 0))
    total_failure = (
        narrator_calls > 0
        and int(result.get("narrator_failed", 0)) == narrator_calls
        and int(result.get("created", 0)) == 0
    )
    if total_failure:
        status = "failed"
    else:
        status = "created" if result.get("created") else "empty"
    generation_diagnostics = {
        "logical_day": logical_day, "status": status, "scanned": result["plan"]["scanned"],
        "plan": result["plan"]["diagnostics"], "narrator_calls": narrator_calls,
        "narrator_failed": result["narrator_failed"],
        "group_reasons": dict(Counter(g["reason"] for g in result["plan"]["groups"] if g["reason"])),
        "write_outcomes": result["write_outcomes"],
    }
    # Keep the latest attempt, including failure; never turn diagnostics into a terminal marker.
    core.set_meta_state_json(f"kairos.cluster_memory.diagnostics.{char}", generation_diagnostics)
    if total_failure:
        return {"ok": False, "status": "failed", "created": 0, "meta_key": meta_key, "diagnostics": generation_diagnostics}
    core.set_meta_state_json(
        meta_key,
        {
            "terminal": True,
            "logical_day": logical_day,
            "created": int(result.get("created", 0)),
            "quarantined": len(result.get("quarantined", [])),
            "ran_at": _utc_now_iso(),
        },
    )
    try:
        from tsuzuri.memory_api import get_memory_api

        get_memory_api(core).record_budget_event(
            char, "cluster_memory_creations", amount=int(result.get("created", 0)), tokens=0
        )
    except Exception:
        pass
    return {
        "ok": True,
        "status": "created" if result.get("created") else "empty",
        "created": int(result.get("created", 0)),
        "parents_created": result.get("parents_created", []),
        "meta_key": meta_key,
        "diagnostics": generation_diagnostics,
    }


# ---------------------------------------------------------------------------
# Phase 3: 召回触发(确定性选择器,由 workspace 装配消费)
# ---------------------------------------------------------------------------

_CLUSTER_QUERY_STOP = frozenset({
    "还记得", "记得", "那次", "后来", "之前", "有没有", "什么", "确认", "事情", "那件",
    "一起", "两人", "我们", "自己", "当时", "一次", "一直", "可以", "表示", "已经",
    "旅行", "旅游", "计划", "安排", "集训", "练习", "话题", "讨论", "对话",
})


def _cluster_query_terms(text: str, names: set[str]) -> set[str]:
    # Reuse the recall keyword extractor: word boundaries, never arbitrary bigrams.
    from tsuzuri.memory_core import _extract_query_keywords

    return {word.casefold() for word in _extract_query_keywords(text, max_keywords=32)
            if word not in _CLUSTER_QUERY_STOP and word.casefold() not in names}


def select_cluster_memory_candidates(
    core,
    character: str,
    user_text: str,
    *,
    explicit_recall: bool = False,
    entity_names: list[str] | None = None,
    max_items: int = 2,
    diagnostics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """「明确提到相关话题且需要聚类记忆」的确定性触发。

    保留双 topic 强路径；显式回忆还需具体内容、地点或非对话双方的人物。
    查询明确点名的已知地点/人物以及 ISO 日期优先约束候选，成员或同文重复去重。
    """
    from tsuzuri.memory_core import infer_topic_labels_from_query, _developer_display_name

    char = str(character or "").strip()
    text = str(user_text or "")
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(scanned=0, eligible=0, selected_ids=[], rejection_counts={}, candidates=[], candidates_truncated=0,
                           explicit_recall=bool(explicit_recall))
    def reject(reason):
        if diagnostics is not None:
            counts = diagnostics["rejection_counts"]
            counts[reason] = counts.get(reason, 0) + 1

    if not char or not text or max_items <= 0:
        reject("empty_input_or_budget")
        return []
    query_labels = set(infer_topic_labels_from_query(text))
    if diagnostics is not None:
        diagnostics["query_labels"] = sorted(query_labels)
    entities = {str(e).strip().casefold() for e in (entity_names or []) if str(e or "").strip()}
    # explicit_recall 时即使无 label/entity 参数也要扫库(places/people 与文本直接匹配)
    if not query_labels and not entities and not explicit_recall:
        reject("no_query_signal")
        return []

    with core._lock:
        conn = core._connect()
        try:
            rows = conn.execute(
                """SELECT item_id, item_type, text, payload_json, topic_labels_json,
                          salience, confidence, importance, created_at, member_ids_json
                   FROM memory_items
                   WHERE character=? AND item_type IN ('event_cluster', 'topic_cluster')
                     AND status='active' AND archived=0""",
                (char,),
            ).fetchall()
        finally:
            conn.close()

    payloads = {str(r["item_id"]): _json_loads(r["payload_json"] or "{}", {}) for r in rows}
    payloads = {iid: p if isinstance(p, dict) else {} for iid, p in payloads.items()}
    def names_in(payload, key):
        values = payload.get(key)
        return [str(n).strip() for n in values if str(n).strip()] if isinstance(values, list) else []
    all_people = {n.casefold() for p in payloads.values() for n in names_in(p, "people")}
    all_places = {n.casefold() for p in payloads.values() for n in names_in(p, "places")}
    query_lower = text.casefold()
    mentioned_places = {n for n in all_places if n in query_lower or n in entities}
    # A speaker/addressee name is background context, not an event discriminator.
    background_people = {char.casefold(), _developer_display_name().casefold(), user_name().casefold(), "user"}
    mentioned_people = {n for n in all_people - background_people if n in query_lower or n in entities}
    terms = _cluster_query_terms(text, all_people | background_people)
    query_dates = set(re.findall(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)", text))
    if diagnostics is not None:
        diagnostics.update(scanned=len(rows), query_dates=sorted(query_dates))
    scored: list[tuple[float, float, str, dict[str, Any]]] = []
    for r in rows:
        row_labels = set(str(x) for x in (_json_loads(r["topic_labels_json"] or "[]", []) or []))
        overlap = len(query_labels & row_labels)
        payload = payloads[str(r["item_id"])]
        people = names_in(payload, "people")
        places = names_in(payload, "places")
        names = people + places
        place_hits = mentioned_places & {n.casefold() for n in places}
        person_hits = mentioned_people & {n.casefold() for n in people}
        start_at = str(payload.get("start_at") or "")
        end_at = str(payload.get("end_at") or start_at)
        date_hit = bool(query_dates) and any(start_at <= day <= end_at for day in query_dates)
        conflict = ("place_mismatch" if mentioned_places and not place_hits else
                    "person_mismatch" if mentioned_people and not person_hits else
                    "date_mismatch" if query_dates and not date_hit else "")
        if conflict:
            reject(conflict)
            continue
        entity_hit = any(n.casefold() in entities or n.casefold() in query_lower for n in names)
        title = str(payload.get("title", "") or "")
        capsule = str(payload.get("capsule", "") or "")
        haystack = " ".join([title, capsule, str(r["text"] or ""), *places]).casefold()
        lexical_hits = sorted(term for term in terms if term in haystack)
        lexical_hit = bool(lexical_hits)
        specific_hit = bool(place_hits or person_hits or date_hit or lexical_hit)
        reason = ""
        if overlap >= 2 and (not explicit_recall or specific_hit):
            reason = "topic_overlap>=2"
        elif overlap == 1 and (specific_hit if explicit_recall else bool(place_hits or person_hits)):
            reason = "topic_overlap+explicit" if explicit_recall else ("topic_overlap+entity" if entity_hit else "topic_overlap+lexical")
        elif explicit_recall and specific_hit:
            reason = "explicit_recall+entity" if entity_hit else "explicit_recall+content"
        if not reason:
            reject("insufficient_specific_signal")
            continue
        member_ids = [str(x) for x in (_json_loads(r["member_ids_json"] or "[]", []) or []) if str(x or "").strip()]
        member_count = len(member_ids)
        scored.append(
            (
                float(overlap) + 2 * len(place_hits) + len(person_hits) + 2 * date_hit + min(2, len(lexical_hits)) * 0.6,
                float(r["salience"] or 0.5),
                str(r["created_at"] or ""),
                {
                    "item_id": str(r["item_id"]),
                    "item_type": str(r["item_type"] or ""),
                    "text": str(r["text"] or ""),
                    "title": str(payload.get("title") or ""),
                    "capsule": str(payload.get("capsule") or ""),
                    "start_at": str(payload.get("start_at") or ""),
                    "end_at": str(payload.get("end_at") or ""),
                    "people": people,
                    "places": places,
                    "topic_labels": sorted(row_labels),
                    "member_count": member_count,
                    "member_ids": member_ids,
                    "salience": float(r["salience"] or 0.5),
                    "confidence": float(r["confidence"] or 0.5),
                    "trigger_reason": reason,
                    "query_entity_hit": entity_hit,
                    "query_lexical_hit": lexical_hit,
                    "matched_keywords": lexical_hits,
                    "matched_places": sorted(place_hits),
                    "matched_people": sorted(person_hits),
                    "query_date_hit": date_hit,
                },
            )
        )
    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]["item_id"]), reverse=True)
    if diagnostics is not None:
        diagnostics["eligible"] = len(scored)
    selected = []
    seen_members = set()
    seen_texts = set()
    for _score, _sal, _created, entry in scored:
        members = tuple(sorted(entry["member_ids"]))
        text_key = (entry["item_type"], entry["start_at"], entry["end_at"], re.sub(r"\s+", "", entry["text"]))
        if (members and members in seen_members) or text_key in seen_texts:
            reject("duplicate_parent")
            continue
        seen_members.add(members)
        seen_texts.add(text_key)
        if len(selected) >= max_items:
            reject("item_cap")
            continue
        selected.append(entry)
    if diagnostics is not None:
        diagnostics["selected_ids"] = [r["item_id"] for r in selected]
        diagnostics["candidates"] = [{"item_id": entry["item_id"], "score": round(score, 3),
            "trigger_reason": entry["trigger_reason"], "matched_labels": sorted(query_labels & set(entry["topic_labels"])),
            "matched_keywords": entry["matched_keywords"], "matched_places": entry["matched_places"],
            "matched_people": entry["matched_people"], "date_hit": entry["query_date_hit"],
            "selected": entry["item_id"] in diagnostics["selected_ids"]} for score, _sal, _created, entry in scored[:50]]
        diagnostics["candidates_truncated"] = max(0, len(scored) - 50)
    return selected


def select_cluster_rethink_candidates(
    core,
    character: str,
    *,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    """Select editable cluster parents for the dedicated rethink flow.

    Cluster members remain evidence only; callers must target the returned
    parent ``item_id`` when proposing a rewrite or supersede action.
    """
    char = str(character or "").strip()
    if not char:
        return []
    with core._lock:
        conn = core._connect()
        try:
            rows = conn.execute(
                """SELECT item_id, item_type, text, payload_json, cluster_meta_json,
                          member_ids_json, salience, confidence, importance,
                          created_at, updated_at
                   FROM memory_items
                   WHERE character=? AND item_type IN ('event_cluster','topic_cluster')
                     AND layer='semantic' AND status='active' AND archived=0
                   ORDER BY updated_at DESC
                   LIMIT ?""",
                (char, max(1, int(max_items))),
            ).fetchall()
        finally:
            conn.close()
    result: list[dict[str, Any]] = []
    for row in rows:
        payload = _json_loads(row["payload_json"] or "{}", {}) or {}
        member_ids = [str(x) for x in (_json_loads(row["member_ids_json"] or "[]", []) or []) if str(x or "").strip()]
        result.append({
            "item_id": str(row["item_id"]),
            "item_type": str(row["item_type"] or ""),
            "text": str(row["text"] or ""),
            "payload": payload,
            "cluster_meta": _json_loads(row["cluster_meta_json"] or "{}", {}) or {},
            "member_ids": member_ids,
            "member_count": len(member_ids),
            "salience": float(row["salience"] or 0.5),
            "confidence": float(row["confidence"] or 0.5),
            "importance": float(row["importance"] or 0.5),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
            "editable_target": "parent_only",
        })
    return result


def fold_member_prefixes(
    core,
    selected_records: list[dict[str, Any]],
    *,
    exempt_hours: float = 48.0,
    prefix_chars: int = 36,
) -> list[str]:
    """级联折叠(仿 BaiShou 高层覆盖低层,Phase 4):被选中 cluster 的 member 文本前缀,
    供 compile_fact_workspace 抑制已被概括覆盖的 member 原文行(原文+概括同现=复读)。

    最近 exempt_hours 内创建的 member 豁免(近期细节保留原文)。纯规则、零 LLM;
    前缀归一化(去非字符+小写)与 turn_context_workspace 的文本去重口径一致。
    """
    member_ids: list[str] = []
    for rec in selected_records or []:
        for mid in rec.get("member_ids") or []:
            mid_s = str(mid or "").strip()
            if mid_s:
                member_ids.append(mid_s)
    if not member_ids:
        return []
    cutoff = utc_now() - timedelta(hours=max(0.0, float(exempt_hours)))
    with core._lock:
        conn = core._connect()
        try:
            placeholders = ",".join(["?"] * len(member_ids))
            rows = conn.execute(
                f"SELECT text, created_at FROM memory_items "
                f"WHERE item_id IN ({placeholders}) AND archived=0 AND status='active'",
                tuple(member_ids),
            ).fetchall()
        finally:
            conn.close()
    prefixes: set[str] = set()
    for r in rows:
        created = _parse_iso(str(r["created_at"] or ""))
        if created is not None and created >= cutoff:
            continue  # 最近 member 豁免:保留近期细节原文
        norm = re.sub(r"\W+", "", str(r["text"] or "").lower())
        if len(norm) >= 8:  # 过短前缀无区分度,放弃折叠该条
            prefixes.add(norm[: max(8, int(prefix_chars))])
    return sorted(prefixes)


def load_cluster_capsule_evidence(
    core,
    character: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_items: int = 10,
    min_significance: int = 5,
) -> list[dict[str, Any]]:
    """向上胶囊证据(Phase 4):近期 cluster 的 capsule,供 KAIROS core_summary rebuild 消费。

    capsule 是设计给上游消费的一句话概括;只取窗口内、significance>=min_significance
    的 active cluster(core summary 只要稳定重要背景,sig=4 的闲谈向上是噪音)。
    输出 continuity 证据行形状 {kind, status, text, item_id}。
    """
    char = str(character or "").strip()
    if not char:
        return []
    cutoff = (utc_now() - timedelta(days=max(1, int(lookback_days)))).date().isoformat()
    with core._lock:
        conn = core._connect()
        try:
            rows = conn.execute(
                """SELECT item_id, item_type, payload_json, cluster_meta_json
                   FROM memory_items
                   WHERE character=? AND item_type IN ('event_cluster', 'topic_cluster')
                     AND status='active' AND archived=0
                   ORDER BY created_at DESC""",
                (char,),
            ).fetchall()
        finally:
            conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        payload = _json_loads(r["payload_json"] or "{}", {}) or {}
        meta = _json_loads(r["cluster_meta_json"] or "{}", {}) or {}
        try:
            significance = int(meta.get("significance", 0) or 0)
        except (TypeError, ValueError):
            significance = 0
        if significance < min_significance:
            continue
        start_at = str(payload.get("start_at") or "")
        end_at = str(payload.get("end_at") or "")
        # 事件结束日在窗口内才向上消费
        if (start_at or end_at) and max(start_at, end_at) < cutoff:
            continue
        title = str(payload.get("title") or "").strip()
        capsule = str(payload.get("capsule") or "").strip()
        line = "——".join(x for x in (title, capsule) if x)
        if not line:
            continue
        span = start_at if (not end_at or start_at == end_at) else f"{start_at}~{end_at}"
        if span:
            line = f"{line}({span})"
        out.append(
            {
                "kind": str(r["item_type"] or "event_cluster"),
                "status": "active",
                "text": line,
                "item_id": str(r["item_id"]),
            }
        )
        if len(out) >= max(1, int(max_items)):
            break
    return out
