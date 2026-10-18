"""Derived SQLite entity index for memory recall and developer inspection.

The tables in this module are rebuildable projections of memory_items. They
never own source facts and are only read on the reply path when the entity
boost flag is enabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import collections.abc
import uuid
from tsuzuri.time_source import utc_now
from typing import Any, Iterable
from tsuzuri.persona_registry import agent_names, collective_scope, list_with_extras, live_set, setting, user_name


_ENTITY_LINK_TYPES = {"merge_into", "rethink_merge_into", "rethink_evidence_for"}
_ENTITY_TOPIC_HINTS = live_set("entity_topic_hints", ["class_school", "home_life"])


class _CastNames(collections.abc.Set):
    """Agent names plus their full names (setting "agent_full_names"); tracks runtime config."""

    def _names(self) -> set[str]:
        return set(agent_names()) | {str(n) for n in (setting("agent_full_names", []) or [])}

    def __iter__(self):
        return iter(self._names())

    def __contains__(self, item) -> bool:
        return item in self._names()

    def __len__(self) -> int:
        return len(self._names())


_TSUZURI_NAMES = _CastNames()

# 复用 memory_core._find_co_occurring 的 generic 实体黑名单语义（见该函数
# 2026-05-09 fix）："两个记忆都提到 user/用户/开发者/代词" 是弱共现信号。
# 一跳扩展面对全库，同样不能把这些名字当可穿越的联想边。
_GENERIC_HOP1_EXCLUDE_NAMES = {
    "user", "开发者",
    "你", "我", "他", "她", "它", "我们", "你们", "她们", "他们",
}
# 小组成员名是 canon 上有意义的 person 边，但作为"共现键"是弱信号（度数 43-81
# 洪泛）。不做完全黑名单（那会杀掉 groupmate 联想），改强降权 + 靠 idf 兜底。
_GENERIC_PERSON_HUB_PENALTY = 0.15
# 与 memory_core._CO_STRENGTH_DEFAULT 对齐：无 Hebbian 共现记录时的边权重。
_HOP1_CO_STRENGTH_DEFAULT = 0.10


def mem_entity_boost_enabled() -> bool:
    return os.getenv("MEM_ENTITY_BOOST_ENABLE", "0").strip() == "1"


def mem_entity_rebuild_enabled() -> bool:
    return os.getenv("MEM_ENTITY_REBUILD_ENABLE", "0").strip() == "1"


def mem_assoc_hop1_shadow_enabled() -> bool:
    return os.getenv("MEM_ASSOC_HOP1_SHADOW_ENABLE", "0").strip() == "1"


def _loads(value: Any, fallback: Any) -> Any:
    try:
        parsed = json.loads(str(value or ""))
        return parsed
    except Exception:
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_name(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:64]


def _entity_id(name: str, entity_type: str, char_scope: str) -> str:
    seed = f"{char_scope}|{entity_type}|{name}".encode("utf-8")
    return "entity:" + hashlib.sha256(seed).hexdigest()[:24]


def _person_roster() -> set[str]:
    names = set(_TSUZURI_NAMES)
    try:
        from tsuzuri.turn_fact_ledger import _load_person_name_roster

        names.update(str(name or "").strip() for name in _load_person_name_roster())
    except Exception:
        pass
    return {name for name in names if name}


def _classify_entity(name: str, cue: dict, roster: set[str]) -> str:
    if name in roster:
        return "person"
    location = _normalize_name(cue.get("location_anchor", ""))
    if location and name == location:
        return "location"
    lowered = name.lower()
    if any(token in lowered for token in list_with_extras("location_tokens", ("学校", "学园", "公园", "车站", "駅", "家", "店"))):
        return "location"
    return "object"


def ensure_entity_schema(store) -> None:
    """Install the derived schema on old and new MemoryCore databases."""
    with store._lock:
        conn = store._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entities (
                    entity_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    char_scope TEXT NOT NULL,
                    count INTEGER NOT NULL DEFAULT 1,
                    cooc_strength REAL NOT NULL DEFAULT 0.0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entity_links (
                    entity_id TEXT NOT NULL,
                    memory_item_id TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 0.5,
                    PRIMARY KEY (entity_id, memory_item_id)
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entity_links_item "
                "ON memory_entity_links(memory_item_id);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_scope "
                "ON memory_entities(char_scope, entity_type, name);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_association_shadow_runs (
                    run_id TEXT PRIMARY KEY,
                    retrieval_run_id TEXT NOT NULL DEFAULT '',
                    char_scope TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    seed_item_ids_json TEXT NOT NULL DEFAULT '[]',
                    candidate_paths_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_assoc_shadow_scope_created "
                "ON memory_association_shadow_runs(char_scope, created_at DESC);"
            )
            conn.commit()
        finally:
            conn.close()


def _active_scope_rows(conn, char_scope: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT item_id, character, text, payload_json, cue_json, topic_labels_json
        FROM memory_items
        WHERE character=? AND archived=0 AND status='active';
        """,
        (char_scope,),
    ).fetchall()
    result: list[dict] = []
    for row in rows:
        payload = _loads(row["payload_json"], {})
        meta = payload.get("memory_meta", {}) if isinstance(payload, dict) else {}
        if isinstance(meta, dict) and str(meta.get("mode_scope", "") or "").startswith("guest"):
            continue
        result.append({key: row[key] for key in row.keys()})
    return result


def _build_projection(
    rows: Iterable[dict],
    char_scope: str,
) -> tuple[dict[str, dict], dict[str, dict[str, float]], dict[str, set[str]]]:
    roster = _person_roster()
    entities: dict[str, dict] = {}
    links: dict[str, dict[str, float]] = {}

    def add(name: str, entity_type: str, item_id: str, *, source_kind: str, source_labels: list[str]) -> str:
        clean = _normalize_name(name)
        if not clean or clean == char_scope:
            return ""
        eid = _entity_id(clean, entity_type, char_scope)
        row = entities.setdefault(
            eid,
            {
                "entity_id": eid,
                "name": clean,
                "entity_type": entity_type,
                "char_scope": char_scope,
                "count": 0,
                "cooc_strength": 0.0,
                "source_item_ids": set(),
                "source_kinds": set(),
                "source_topic_labels": set(),
            },
        )
        row["count"] += 1
        row["source_item_ids"].add(item_id)
        row["source_kinds"].add(source_kind)
        row["source_topic_labels"].update(source_labels)
        links.setdefault(eid, {})[item_id] = max(1.0, links.setdefault(eid, {}).get(item_id, 0.0))
        return eid

    item_entities: dict[str, set[str]] = {}
    for row in rows:
        item_id = str(row.get("item_id", "") or "").strip()
        if not item_id:
            continue
        cue = _loads(row.get("cue_json", ""), {})
        cue = cue if isinstance(cue, dict) else {}
        topic_labels = _loads(row.get("topic_labels_json", ""), [])
        labels = [str(value or "").strip() for value in topic_labels if str(value or "").strip()] if isinstance(topic_labels, list) else []
        current: set[str] = set()
        for label in labels:
            eid = add(label, "topic", item_id, source_kind="topic_labels", source_labels=labels)
            if eid:
                current.add(eid)
        raw_entities = cue.get("entities", []) if isinstance(cue.get("entities", []), list) else []
        location = _normalize_name(cue.get("location_anchor", ""))
        if location:
            raw_entities = list(raw_entities) + [location]
        seen_cues: set[tuple[str, str]] = set()
        for value in raw_entities:
            name = _normalize_name(value)
            if not name:
                continue
            entity_type = _classify_entity(name, cue, roster)
            cue_key = (name, entity_type)
            if cue_key in seen_cues:
                continue
            seen_cues.add(cue_key)
            eid = add(
                name,
                entity_type,
                item_id,
                source_kind="cue_json",
                source_labels=labels,
            )
            if eid:
                current.add(eid)
        item_entities[item_id] = current
    return entities, links, item_entities


def rebuild_entities(store, char_scope: str) -> dict:
    """Atomically replace one exact-scope projection from active memory rows."""
    scope = _normalize_name(char_scope)
    if not scope or scope.startswith("guest"):
        return {"ok": False, "status": "invalid_scope", "entity_count": 0, "link_count": 0}
    ensure_entity_schema(store)
    with store._lock:
        conn = store._connect()
        try:
            rows = _active_scope_rows(conn, scope)
            entities, links, item_entities = _build_projection(rows, scope)

            item_ids = set(item_entities)
            if item_ids:
                cooc_rows = conn.execute(
                    """
                    SELECT edge.item_a, edge.item_b, edge.strength
                    FROM co_occurrence_strength edge
                    JOIN memory_items a ON a.item_id=edge.item_a
                    JOIN memory_items b ON b.item_id=edge.item_b
                    WHERE a.character=? AND b.character=?
                      AND a.archived=0 AND b.archived=0
                      AND a.status='active' AND b.status='active';
                    """,
                    (scope, scope),
                ).fetchall()
                for edge in cooc_rows:
                    shared = item_entities.get(str(edge["item_a"]), set()) & item_entities.get(str(edge["item_b"]), set())
                    strength = max(0.0, float(edge["strength"] or 0.0))
                    for eid in shared:
                        entities[eid]["cooc_strength"] += strength

                memory_link_rows = conn.execute(
                    """
                    SELECT edge.src_item_id, edge.dst_item_id, edge.link_type, edge.weight
                    FROM memory_links edge
                    JOIN memory_items src ON src.item_id=edge.src_item_id
                    JOIN memory_items dst ON dst.item_id=edge.dst_item_id
                    WHERE src.character=? AND dst.character=?
                      AND src.archived=0 AND dst.archived=0
                      AND src.status='active' AND dst.status='active';
                    """,
                    (scope, scope),
                ).fetchall()
                for edge in memory_link_rows:
                    if str(edge["link_type"] or "") not in _ENTITY_LINK_TYPES:
                        continue
                    src = str(edge["src_item_id"] or "")
                    dst = str(edge["dst_item_id"] or "")
                    weight = max(0.0, float(edge["weight"] or 0.0))
                    for eid in item_entities.get(src, set()):
                        links.setdefault(eid, {})[dst] = max(weight, links.setdefault(eid, {}).get(dst, 0.0))
                    for eid in item_entities.get(dst, set()):
                        links.setdefault(eid, {})[src] = max(weight, links.setdefault(eid, {}).get(src, 0.0))

            now = utc_now().replace(microsecond=0).isoformat()
            conn.execute(
                "DELETE FROM memory_entity_links WHERE entity_id IN "
                "(SELECT entity_id FROM memory_entities WHERE char_scope=?);",
                (scope,),
            )
            conn.execute("DELETE FROM memory_entities WHERE char_scope=?;", (scope,))
            for entity in sorted(entities.values(), key=lambda value: value["entity_id"]):
                payload = {
                    "source_item_ids": sorted(entity["source_item_ids"]),
                    "source_kinds": sorted(entity["source_kinds"]),
                    "source_topic_labels": sorted(entity["source_topic_labels"]),
                }
                conn.execute(
                    """
                    INSERT INTO memory_entities(
                        entity_id, name, entity_type, char_scope, count, cooc_strength,
                        payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entity["entity_id"], entity["name"], entity["entity_type"], scope,
                        int(entity["count"]), round(float(entity["cooc_strength"]), 6),
                        _dumps(payload), now, now,
                    ),
                )
                for item_id, weight in sorted(links.get(entity["entity_id"], {}).items()):
                    conn.execute(
                        "INSERT INTO memory_entity_links(entity_id, memory_item_id, weight) VALUES (?, ?, ?);",
                        (entity["entity_id"], item_id, round(float(weight), 6)),
                    )
            conn.commit()
            return {
                "ok": True,
                "status": "rebuilt",
                "char_scope": scope,
                "entity_count": len(entities),
                "link_count": sum(len(values) for values in links.values()),
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def list_entities(store, char_scope: str, *, entity_type: str = "", limit: int = 50) -> list[dict]:
    scope = _normalize_name(char_scope)
    if not scope or scope.startswith("guest"):
        return []
    ensure_entity_schema(store)
    where = ["char_scope=?"]
    params: list[Any] = [scope]
    kind = str(entity_type or "").strip().lower()
    if kind:
        where.append("entity_type=?")
        params.append(kind)
    params.append(max(1, min(500, int(limit or 50))))
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM memory_entities WHERE " + " AND ".join(where)
                + " ORDER BY count DESC, cooc_strength DESC, name ASC LIMIT ?;",
                tuple(params),
            ).fetchall()
            return [
                {
                    **{key: row[key] for key in row.keys() if key != "payload_json"},
                    "payload": _loads(row["payload_json"], {}),
                }
                for row in rows
            ]
        finally:
            conn.close()


def get_entity(store, entity_id: str, *, char_scope: str = "") -> dict | None:
    eid = str(entity_id or "").strip()
    scope = _normalize_name(char_scope)
    if not eid or scope.startswith("guest"):
        return None
    ensure_entity_schema(store)
    with store._lock:
        conn = store._connect()
        try:
            if scope:
                row = conn.execute(
                    "SELECT * FROM memory_entities WHERE entity_id=? AND char_scope=? LIMIT 1;",
                    (eid, scope),
                ).fetchone()
            else:
                row = conn.execute("SELECT * FROM memory_entities WHERE entity_id=? LIMIT 1;", (eid,)).fetchone()
            if row is None:
                return None
            return {
                **{key: row[key] for key in row.keys() if key != "payload_json"},
                "payload": _loads(row["payload_json"], {}),
            }
        finally:
            conn.close()


def entity_links(store, entity_id: str, *, char_scope: str = "", limit: int = 100) -> list[dict]:
    entity = get_entity(store, entity_id, char_scope=char_scope)
    if entity is None:
        return []
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                """
                SELECT l.entity_id, l.memory_item_id, l.weight, m.character, m.layer,
                       m.item_type, m.text, m.source, m.updated_at
                FROM memory_entity_links l
                JOIN memory_items m ON m.item_id=l.memory_item_id
                WHERE l.entity_id=? AND m.archived=0 AND m.status='active'
                ORDER BY l.weight DESC, m.updated_at DESC
                LIMIT ?;
                """,
                (entity["entity_id"], max(1, min(500, int(limit or 100)))),
            ).fetchall()
            return [{key: row[key] for key in row.keys()} for row in rows]
        finally:
            conn.close()


def _query_entity_ids(conn, char_scope: str, query: str) -> set[str]:
    try:
        from tsuzuri.memory_core import infer_topic_labels_from_query

        query_labels = set(infer_topic_labels_from_query(query))
    except Exception:
        query_labels = set()
    lowered = query.casefold()
    matched: set[str] = set()
    rows = conn.execute(
        "SELECT entity_id, name, entity_type, payload_json FROM memory_entities "
        "WHERE char_scope=? AND entity_type!='topic';",
        (char_scope,),
    ).fetchall()
    roster = _person_roster()
    for row in rows:
        name = str(row["name"] or "").strip()
        if not name:
            continue
        direct = name.casefold() in lowered
        if str(row["entity_type"] or "") == "person" and name in roster:
            direct = direct or any(alias.casefold() in lowered for alias in roster if alias in name or name in alias)
        payload = _loads(row["payload_json"], {})
        source_labels = set(payload.get("source_topic_labels", []) if isinstance(payload, dict) else [])
        hinted = bool(
            str(row["entity_type"] or "") in {"location", "object"}
            and query_labels.intersection(source_labels).intersection(_ENTITY_TOPIC_HINTS)
        )
        if direct or hinted:
            matched.add(str(row["entity_id"] or ""))
    return matched


def annotate_entity_boosts(store, character: str, query: str, rows: list[dict]) -> list[dict]:
    """Mark already-retrieved hybrid candidates; never adds a recall lane."""
    if not mem_entity_boost_enabled() or not rows:
        return rows
    scope = _normalize_name(character)
    probe = str(query or "").strip()
    if not scope or not probe or scope.startswith("guest"):
        return rows
    with store._lock:
        conn = store._connect()
        try:
            query_ids = _query_entity_ids(conn, scope, probe)
            if not query_ids:
                return rows
            item_ids = [str(row.get("item_id", "") or "").strip() for row in rows]
            item_ids = [item_id for item_id in item_ids if item_id]
            if not item_ids:
                return rows
            entity_placeholders = ",".join("?" for _ in query_ids)
            item_placeholders = ",".join("?" for _ in item_ids)
            hits = conn.execute(
                f"SELECT memory_item_id, COUNT(DISTINCT entity_id) AS overlap "
                f"FROM memory_entity_links WHERE entity_id IN ({entity_placeholders}) "
                f"AND memory_item_id IN ({item_placeholders}) GROUP BY memory_item_id;",
                tuple(query_ids) + tuple(item_ids),
            ).fetchall()
            overlap_by_item = {str(row["memory_item_id"]): int(row["overlap"] or 0) for row in hits}
        finally:
            conn.close()
    for row in rows:
        overlap = overlap_by_item.get(str(row.get("item_id", "") or ""), 0)
        if overlap >= 2:
            row["_entity_boost"] = 1.35
        elif overlap == 1:
            row["_entity_boost"] = 1.18
    return rows


def _shadow_seed_score(row: dict) -> float:
    values = [
        row.get("score"),
        row.get("final_score"),
        row.get("_hybrid_dense_score"),
        row.get("_hybrid_sparse_score"),
        row.get("_hybrid_structured_score"),
    ]
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(parsed, default=0.0)


def evaluate_association_hop1_shadow(
    store,
    character: str,
    query: str,
    rows: list[dict],
    *,
    retrieval_run_id: str = "",
    persist: bool = False,
    seed_score_floor: float = 0.30,
    max_seeds: int = 3,
    max_candidates: int = 20,
    min_shadow_score: float = 0.01,
) -> dict:
    """Evaluate exact-scope one-hop neighbors without changing recall candidates."""
    scope = _normalize_name(character)
    probe = str(query or "").strip()
    base = {
        "status": "disabled",
        "char_scope": scope,
        "seed_item_ids": [],
        "candidates": [],
        "metrics": {
            "input_rows": len(rows or []),
            "eligible_seeds": 0,
            "traversed_paths": 0,
            "candidate_count": 0,
            "excluded_topic_paths": 0,
            "excluded_high_frequency_paths": 0,
            "rejected_cross_scope_paths": 0,
            "excluded_generic_paths": 0,
            "downweighted_person_hub_paths": 0,
        },
    }
    if not scope or not probe:
        return {**base, "status": "invalid_input"}
    if scope.startswith("guest"):
        return {**base, "status": "invalid_scope"}
    if scope == collective_scope():
        return {**base, "status": "collective_scope_blocked"}

    scored_seeds = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        item_id = str(row.get("item_id", "") or "").strip()
        score = _shadow_seed_score(row)
        row_scope = _normalize_name(row.get("character", scope))
        if item_id and row_scope == scope and score >= float(seed_score_floor):
            scored_seeds.append((score, item_id))
    scored_seeds.sort(key=lambda pair: (-pair[0], pair[1]))
    scored_seeds = scored_seeds[:max(1, min(8, int(max_seeds or 3)))]
    if not scored_seeds:
        return {**base, "status": "no_eligible_seeds"}

    seed_scores = {item_id: score for score, item_id in scored_seeds}
    seed_ids = list(seed_scores)
    report = {**base, "seed_item_ids": seed_ids, "metrics": dict(base["metrics"])}
    report["metrics"]["eligible_seeds"] = len(seed_ids)
    ensure_entity_schema(store)
    with store._lock:
        conn = store._connect()
        try:
            counts = [
                int(row["count"] or 0)
                for row in conn.execute(
                    "SELECT count FROM memory_entities WHERE char_scope=? AND entity_type!='topic' ORDER BY count;",
                    (scope,),
                ).fetchall()
            ]
            if counts:
                p90_index = max(0, min(len(counts) - 1, math.ceil(len(counts) * 0.90) - 1))
                high_frequency_cutoff = counts[p90_index]
            else:
                high_frequency_cutoff = 0
            placeholders = ",".join("?" for _ in seed_ids)
            path_rows = conn.execute(
                f"""
                SELECT sl.memory_item_id AS seed_item_id, sl.weight AS seed_weight,
                       e.entity_id, e.name AS entity_name, e.entity_type, e.count AS entity_count,
                       nl.memory_item_id AS candidate_item_id, nl.weight AS candidate_weight,
                       m.character AS candidate_scope
                FROM memory_entity_links sl
                JOIN memory_entities e ON e.entity_id=sl.entity_id
                JOIN memory_entity_links nl ON nl.entity_id=e.entity_id
                JOIN memory_items m ON m.item_id=nl.memory_item_id
                WHERE sl.memory_item_id IN ({placeholders})
                  AND e.char_scope=?
                  AND nl.memory_item_id NOT IN ({placeholders})
                  AND m.archived=0 AND m.status='active';
                """,
                tuple(seed_ids) + (scope,) + tuple(seed_ids),
            ).fetchall()
            candidate_map: dict[str, dict] = {}
            for row in path_rows:
                report["metrics"]["traversed_paths"] += 1
                if str(row["candidate_scope"] or "") != scope:
                    report["metrics"]["rejected_cross_scope_paths"] += 1
                    continue
                if str(row["entity_type"] or "") == "topic":
                    report["metrics"]["excluded_topic_paths"] += 1
                    continue
                entity_name = str(row["entity_name"] or "").strip()
                if entity_name in _GENERIC_HOP1_EXCLUDE_NAMES or entity_name == user_name():
                    report["metrics"]["excluded_generic_paths"] += 1
                    continue
                generic_penalty = 1.0
                if entity_name in _TSUZURI_NAMES:
                    generic_penalty = _GENERIC_PERSON_HUB_PENALTY
                    report["metrics"]["downweighted_person_hub_paths"] += 1
                entity_count = int(row["entity_count"] or 0)
                if high_frequency_cutoff and entity_count > high_frequency_cutoff:
                    report["metrics"]["excluded_high_frequency_paths"] += 1
                    continue
                seed_id = str(row["seed_item_id"] or "")
                candidate_id = str(row["candidate_item_id"] or "")
                # 度数 IDF（比 log1p 更陡）直接压制 groupmate 洪泛；generic person 再乘降权。
                idf = 1.0 / math.sqrt(max(1.0, entity_count))
                entity_path_score = seed_scores.get(seed_id, 0.0) * idf * generic_penalty
                candidate = candidate_map.setdefault(candidate_id, {
                    "candidate_item_id": candidate_id,
                    "shadow_score": 0.0,
                    "best_seed_id": "",
                    "seed_item_ids": [],
                    "entity_paths": [],
                })
                if entity_path_score > float(candidate["shadow_score"]):
                    candidate["shadow_score"] = entity_path_score
                    candidate["best_seed_id"] = seed_id
                if seed_id not in candidate["seed_item_ids"]:
                    candidate["seed_item_ids"].append(seed_id)
                candidate["entity_paths"].append({
                    "entity_id": str(row["entity_id"] or ""),
                    "entity_name": entity_name[:64],
                    "entity_type": str(row["entity_type"] or ""),
                    "entity_count": entity_count,
                    "generic_penalty": round(generic_penalty, 4),
                    "idf": round(idf, 5),
                })

            # 用 Hebbian 学到的 item 对共现强度当"真边权重"，替代投影层恒为 1.0 的
            # memory_entity_links.weight。co_occurrence_strength 只在扩散激活跑过时才有，
            # 无记录默认 0.10（对齐 memory_core._CO_STRENGTH_DEFAULT），是弱 boost 而非硬门。
            pair_strengths: dict[tuple[str, str], float] = {}
            candidate_ids = list(candidate_map)
            if candidate_ids:
                seed_ph = ",".join("?" for _ in seed_ids)
                cand_ph = ",".join("?" for _ in candidate_ids)
                try:
                    cs_rows = conn.execute(
                        f"""SELECT item_a, item_b, strength FROM co_occurrence_strength
                            WHERE (item_a IN ({seed_ph}) AND item_b IN ({cand_ph}))
                               OR (item_a IN ({cand_ph}) AND item_b IN ({seed_ph}));""",
                        tuple(seed_ids) + tuple(candidate_ids) + tuple(candidate_ids) + tuple(seed_ids),
                    ).fetchall()
                    for _row in cs_rows:
                        _a, _b = str(_row["item_a"] or ""), str(_row["item_b"] or "")
                        if _a and _b:
                            pair_strengths[(_a, _b)] = float(_row["strength"] or _HOP1_CO_STRENGTH_DEFAULT)
                except Exception:
                    pass  # co_occurrence_strength 表在极旧库可能不存在，静默降级

            for candidate in candidate_map.values():
                best_seed = str(candidate.pop("best_seed_id", "") or "")
                cid = str(candidate["candidate_item_id"] or "")
                pair_key = (min(best_seed, cid), max(best_seed, cid)) if best_seed else ("", "")
                edge_weight = pair_strengths.get(pair_key, _HOP1_CO_STRENGTH_DEFAULT)
                candidate["shadow_score"] = round(float(candidate["shadow_score"]) * edge_weight, 6)

            candidates = sorted(
                candidate_map.values(),
                key=lambda row: (-float(row["shadow_score"]), str(row["candidate_item_id"])),
            )
            # 分数地板：宁可不填满 max_candidates，也不把近零的泛化边当"有效联想"。
            candidates = [c for c in candidates if float(c["shadow_score"]) >= float(min_shadow_score)]
            candidates = candidates[:max(1, min(100, int(max_candidates or 20)))]
            for candidate in candidates:
                candidate["seed_item_ids"].sort()
                candidate["entity_paths"] = sorted(
                    candidate["entity_paths"],
                    key=lambda path: (str(path["entity_type"]), str(path["entity_id"])),
                )[:8]
            report["candidates"] = candidates
            report["metrics"]["candidate_count"] = len(candidates)
            report["metrics"]["high_frequency_cutoff"] = high_frequency_cutoff
            report["status"] = "ok" if candidates else "empty"
            if persist:
                now = utc_now().isoformat()
                run_id = "assoc-shadow:" + uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO memory_association_shadow_runs(
                        run_id, retrieval_run_id, char_scope, query_hash,
                        seed_item_ids_json, candidate_paths_json, metrics_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        run_id,
                        str(retrieval_run_id or "")[:128],
                        scope,
                        hashlib.sha256(probe.encode("utf-8")).hexdigest(),
                        _dumps(seed_ids),
                        _dumps(candidates),
                        _dumps(report["metrics"]),
                        now,
                    ),
                )
                conn.execute(
                    "DELETE FROM memory_association_shadow_runs WHERE run_id NOT IN "
                    "(SELECT run_id FROM memory_association_shadow_runs ORDER BY created_at DESC LIMIT 2000);"
                )
                conn.commit()
                report["run_id"] = run_id
        finally:
            conn.close()
    return report


__all__ = [
    "annotate_entity_boosts",
    "evaluate_association_hop1_shadow",
    "ensure_entity_schema",
    "entity_links",
    "get_entity",
    "list_entities",
    "mem_entity_boost_enabled",
    "mem_entity_rebuild_enabled",
    "mem_assoc_hop1_shadow_enabled",
    "rebuild_entities",
]
