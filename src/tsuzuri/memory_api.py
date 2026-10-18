"""显式记忆服务 façade（P0：docs/brain/memory/letta_mem0_memory_plan/07-phase-0-memory-api-and-budget.md）。

设计约束：
- 只读/受控写；不提供公开 insert（写入入口仍只有 plan_turn_memory_writes）。
- façade 不自建连接池：连接/锁复用 store._lock + store._connect()（kairos_* 同款 in-house 模式）。
- guest 隔离：mode_scope 在 payload_json.memory_meta 里（非表列），Python 侧统一剔除。
- 无硬删除：delete 一律状态翻转 archived=1 + status='archived'，审计行留 before 快照可手工恢复。
- 预算计数走 meta_state（KAIROS audit-meta 同款读写对），软上限只 warn 不硬拦。
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone

from tsuzuri.memory_core import (
    _json_dumps,
    _json_loads,
    _memory_payload_meta,
    _utc_now_iso,
    get_memory_core,
)
from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH
from tsuzuri.time_source import now_world, utc_now
from tsuzuri.persona_registry import collective_scope

log = logging.getLogger(__name__)

from tsuzuri.time_source import WORLD_TZ as _WORLD_TZ  # world-clock time zone

# 列表/详情共用的列清单（照 memory_core.py:_query_items :8144，去掉 embedding 大字段）
_ITEM_COLUMNS = (
    "item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, "
    "cue_json, lifecycle_json, salience, confidence, novelty, importance, status, source, "
    "evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, topic_labels_json, "
    "parent_id, cluster_memory_parent_id, member_ids_json, cluster_meta_json"
)

_MAX_UPDATE_TEXT_LEN = 2000
_ARCHIVE_GRACE_HOURS = 168.0
_BUDGET_KINDS = (
    "turn_time_llm_calls",
    "tool_loop_llm_calls",
    "rethink_actions",
    "rethink_loop_llm_calls",
    "core_summary_rebuilds",
    "entity_rebuilds",
    "forgetting_advances",
    "cluster_memory_creations",
)
_BUDGET_WARN_ENV = "MEM_BUDGET_DAILY_TOKEN_WARN"
_BUDGET_WARN_DEFAULT = 300000


def _archive_enabled() -> bool:
    """删除门禁（计划 07 字面落地）：live archive 默认关，与 rethink 共用同一 flag。"""
    return os.getenv("MEM_RETHINK_ALLOW_ARCHIVE", "0").strip() == "1"


def _world_today() -> str:
    return now_world().strftime("%Y-%m-%d")


def _escape_like(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _is_guest_payload(payload: dict) -> bool:
    meta = _memory_payload_meta(payload if isinstance(payload, dict) else {})
    return str(meta.get("mode_scope", "") or "").startswith("guest")


def _item_is_guest(item: dict) -> bool:
    return str((item.get("memory_meta") or {}).get("mode_scope", "") or "").startswith("guest")


class MemoryApi:
    """MemoryCoreStore 之上的显式记忆管理 façade。dev-only；guest 行在所有读取路径被剔除。"""

    def __init__(self, store) -> None:
        self._store = store
        self._ensure_audit_table()

    # ---------- 基础设施 ----------

    def _ensure_audit_table(self) -> None:
        # 与 memory_core 同库，但表由本模块自管（IF NOT EXISTS 幂等）
        with self._store._lock:
            conn = self._store._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_api_action_log (
                        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT NOT NULL,
                        character TEXT NOT NULL DEFAULT '',
                        action TEXT NOT NULL,
                        actor TEXT NOT NULL DEFAULT 'dev_api',
                        before_json TEXT NOT NULL DEFAULT '',
                        after_json TEXT NOT NULL DEFAULT '',
                        reason TEXT NOT NULL DEFAULT '',
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_api_action_log_item "
                    "ON memory_api_action_log(item_id, action_id);"
                )
                conn.commit()
            finally:
                conn.close()

    def _fetch_row(self, conn, item_id: str):
        return conn.execute(
            f"SELECT {_ITEM_COLUMNS}, "
            "CASE WHEN embedding_text IS NULL OR embedding_text='' THEN 0 ELSE 1 END AS has_embedding "
            "FROM memory_items WHERE item_id=? LIMIT 1;",
            (str(item_id or "").strip(),),
        ).fetchone()

    def _snapshot(self, conn, item_id: str) -> dict:
        # 形状对齐 memory_core._snapshot_item_row：{"exists": bool, "row": {...}}
        row = self._fetch_row(conn, item_id)
        if row is None:
            return {"exists": False}
        return {"exists": True, "row": {k: row[k] for k in row.keys()}}

    def _write_audit(self, conn, *, item_id: str, character: str, action: str, actor: str,
                     before: dict, after: dict, reason: str = "", meta: dict | None = None) -> None:
        conn.execute(
            """
            INSERT INTO memory_api_action_log(item_id, character, action, actor, before_json, after_json, reason, meta_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(item_id or ""), str(character or ""), str(action or ""), str(actor or "dev_api"),
                _json_dumps(before or {}), _json_dumps(after or {}), str(reason or ""),
                _json_dumps(meta or {}), _utc_now_iso(),
            ),
        )

    @staticmethod
    def _normalize_row(row, *, include_payload: bool = False) -> dict:
        payload = _json_loads(str(row["payload_json"] or ""), {})
        legacy_parent_id = str(row["parent_id"] or "")
        cluster_parent_id = str(row["cluster_memory_parent_id"] or "")
        item = {
            "item_id": str(row["item_id"]),
            "character": str(row["character"] or ""),
            "layer": str(row["layer"] or ""),
            "memory_role": str(row["memory_role"] or ""),
            "engram_type": str(row["engram_type"] or ""),
            "item_type": str(row["item_type"] or ""),
            "text": str(row["text"] or ""),
            "salience": row["salience"],
            "confidence": row["confidence"],
            "novelty": row["novelty"],
            "importance": row["importance"],
            "status": str(row["status"] or ""),
            "source": str(row["source"] or ""),
            "evidence_id": str(row["evidence_id"] or ""),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
            "last_accessed_at": str(row["last_accessed_at"] or ""),
            "access_count": row["access_count"],
            "archived": int(row["archived"] or 0),
            "topic_labels": _json_loads(str(row["topic_labels_json"] or ""), []),
            "cue": _json_loads(str(row["cue_json"] or ""), {}),
            "lifecycle": _json_loads(str(row["lifecycle_json"] or ""), {}),
            # Keep the legacy API field useful for callers while exposing the
            # dedicated cluster edge explicitly. The database columns remain
            # separate because embedding summaries still own parent_id.
            "parent_id": legacy_parent_id or cluster_parent_id,
            "cluster_memory_parent_id": cluster_parent_id,
            "member_count": len(_json_loads(str(row["member_ids_json"] or ""), []) or []),
            "memory_meta": _memory_payload_meta(payload if isinstance(payload, dict) else {}),
        }
        try:
            item["has_embedding"] = bool(row["has_embedding"])
        except Exception:
            pass
        if include_payload:
            item["payload"] = payload if isinstance(payload, dict) else {}
            item["member_ids"] = _json_loads(str(row["member_ids_json"] or ""), []) or []
            item["cluster_meta"] = _json_loads(str(row["cluster_meta_json"] or ""), {}) or {}
        return item

    def _visible_row(self, conn, char: str, item_id: str, *, include_payload: bool = False):
        """按 id 取行并强制 per-char 作用域 + guest 剔除；不可见返回 None。"""
        row = self._fetch_row(conn, item_id)
        if row is None:
            return None
        row_char = str(row["character"] or "")
        if row_char not in (char, "", "*"):
            return None
        payload = _json_loads(str(row["payload_json"] or ""), {})
        if _is_guest_payload(payload if isinstance(payload, dict) else {}):
            return None
        return self._normalize_row(row, include_payload=include_payload)

    def list_entities(self, char: str, *, entity_type: str = "", limit: int = 50) -> dict:
        from tsuzuri.memory_entities import list_entities

        scope = str(char or "").strip()
        if not scope:
            return {"ok": False, "error": "character_required", "entities": []}
        return {
            "ok": True,
            "character": scope,
            "entities": list_entities(self._store, scope, entity_type=entity_type, limit=limit),
        }

    def get_entity(self, entity_id: str, *, char: str = "") -> dict | None:
        from tsuzuri.memory_entities import get_entity

        return get_entity(self._store, entity_id, char_scope=str(char or "").strip())

    def entity_links(self, entity_id: str, *, char: str = "", limit: int = 100) -> dict:
        from tsuzuri.memory_entities import entity_links, get_entity

        scope = str(char or "").strip()
        if get_entity(self._store, entity_id, char_scope=scope) is None:
            return {"ok": False, "error": "not_found", "entity_id": str(entity_id or "").strip(), "links": []}
        rows = entity_links(
            self._store,
            entity_id,
            char_scope=scope,
            limit=limit,
        )
        return {"ok": True, "entity_id": str(entity_id or "").strip(), "links": rows}

    # ---------- 读 ----------

    def list_memories(self, char: str, *, layer: str = "", item_type: str = "", since: str = "", until: str = "",
                      query: str = "", include_archived: bool = False, include_collective: bool = False,
                      limit: int = 200, offset: int = 0) -> dict:
        char = str(char or "").strip()
        if not char:
            return {"ok": False, "error": "character_required", "items": []}
        lim = max(1, min(500, int(limit or 200)))
        off = max(0, int(offset or 0))
        scope_sql = "(character=? OR character='' OR character='*')"
        params: list = [char]
        if include_collective:
            scope_sql = "(character=? OR character='' OR character='*' OR character=?)"
            params.append(collective_scope())
        where = [scope_sql]
        if not include_archived:
            where.append("archived=0 AND status='active'")
        layer = str(layer or "").strip()
        if layer:
            where.append("layer=?")
            params.append(layer)
        item_type = str(item_type or "").strip()
        if item_type:
            where.append("item_type=?")
            params.append(item_type)
        since = str(since or "").strip()
        if since:
            where.append("created_at>=?")
            params.append(since)
        until = str(until or "").strip()
        if until:
            where.append("created_at<=?")
            params.append(until)
        query = str(query or "").strip()
        if query:
            # LIKE 子串匹配：FTS5 unicode61 不切中文，子串查询对 CJK 不可靠（memory_core 召回也用 LIKE 兜底）
            where.append("text LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(query)}%")
        sql = (
            f"SELECT {_ITEM_COLUMNS}, "
            "CASE WHEN embedding_text IS NULL OR embedding_text='' THEN 0 ELSE 1 END AS has_embedding "
            f"FROM memory_items WHERE {' AND '.join(where)} "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?;"
        )
        with self._store._lock:
            conn = self._store._connect()
            try:
                rows = conn.execute(sql, (*params, lim + 1, off)).fetchall()
            finally:
                conn.close()
        items = []
        for row in rows[: lim + 1]:
            item = self._normalize_row(row)
            if _item_is_guest(item):
                continue
            items.append(item)
            if len(items) >= lim:
                break
        return {"ok": True, "items": items, "count": len(items), "has_more": len(rows) > lim}

    def get_memory(self, char: str, item_id: str) -> dict | None:
        char = str(char or "").strip()
        with self._store._lock:
            conn = self._store._connect()
            try:
                return self._visible_row(conn, char, item_id, include_payload=True)
            finally:
                conn.close()

    def get_cluster_members(self, char: str, item_id: str, *, limit: int = 100) -> dict | None:
        """cluster member 展开回读:event/topic_cluster 行的成员预览(按 created_at 升序)。

        不可见(越角色/guest)返回 None;非 cluster 行返回 not_a_cluster。
        """
        char = str(char or "").strip()
        lim = max(1, min(200, int(limit or 100)))
        with self._store._lock:
            conn = self._store._connect()
            try:
                cluster = self._visible_row(conn, char, item_id, include_payload=True)
                if cluster is None:
                    return None
                from tsuzuri.cluster_memory import CLUSTER_ITEM_TYPES

                if str(cluster.get("item_type", "") or "") not in CLUSTER_ITEM_TYPES:
                    return {"ok": False, "error": "not_a_cluster", "item_id": str(item_id or "").strip()}
                member_ids = [str(x) for x in (cluster.get("member_ids") or []) if str(x or "").strip()]
                members: list[dict] = []
                if member_ids:
                    placeholders = ",".join(["?"] * len(member_ids))
                    rows = conn.execute(
                        f"SELECT {_ITEM_COLUMNS} FROM memory_items "
                        f"WHERE item_id IN ({placeholders}) "
                        "AND (character=? OR character='' OR character='*') "
                        "AND archived=0 AND status='active' "
                        "ORDER BY created_at ASC LIMIT ?;",
                        (*member_ids, char, lim),
                    ).fetchall()
                    for row in rows:
                        item = self._normalize_row(row)
                        if _item_is_guest(item):
                            continue
                        # 2026-08-16 cluster Phase4-B2: 透出溯源字段——
                        # location_anchor 来自 cue,turn_fact_turn_id 来自 payload
                        # (cluster→turn 溯源映射,写路径无信号时为空串)。
                        _member_payload = _json_loads(str(row["payload_json"] or ""), {})
                        _member_cue = item.get("cue") if isinstance(item.get("cue"), dict) else {}
                        members.append(
                            {
                                "item_id": item["item_id"],
                                "item_type": item["item_type"],
                                "text": item["text"][:200],
                                "created_at": item["created_at"],
                                "salience": item["salience"],
                                "confidence": item["confidence"],
                                "source": item["source"],
                                "evidence_id": item["evidence_id"],
                                "parent_id": item["parent_id"],
                                "location_anchor": str(_member_cue.get("location_anchor", "") or ""),
                                "turn_fact_turn_id": str(_member_payload.get("turn_fact_turn_id", "") or "") if isinstance(_member_payload, dict) else "",
                            }
                        )
                return {
                    "ok": True,
                    "cluster": cluster,
                    "members": members,
                    "member_count": len(member_ids),
                    "members_returned": len(members),
                }
            finally:
                conn.close()

    @staticmethod
    def _snapshot_texts(snapshot_json: str) -> tuple[str, str]:
        """从 before/after 快照 JSON 提取代表文本（容忍 {"row":...} 与 {iid: {...}} 两种形状）。"""
        data = _json_loads(str(snapshot_json or ""), {})
        if not isinstance(data, dict):
            return "", ""
        texts: list[str] = []
        if isinstance(data.get("row"), dict):
            texts.append(str(data["row"].get("text", "") or ""))
        else:
            for value in data.values():
                if isinstance(value, dict) and isinstance(value.get("row"), dict):
                    texts.append(str(value["row"].get("text", "") or ""))
        return (texts[0] if texts else "", " | ".join(texts[1:]) if len(texts) > 1 else "")

    def get_history(self, char: str, item_id: str) -> dict | None:
        char = str(char or "").strip()
        iid = str(item_id or "").strip()
        with self._store._lock:
            conn = self._store._connect()
            try:
                item = self._visible_row(conn, char, iid, include_payload=True)
                if item is None:
                    return None
                entries: list[dict] = []
                # 1) evolution_action_log：有规范 item_id 列
                try:
                    for r in conn.execute(
                        "SELECT action_id, run_id, op, before_json, after_json, status, created_at, reverted_at "
                        "FROM evolution_action_log WHERE item_id=? ORDER BY action_id DESC LIMIT 100;",
                        (iid,),
                    ).fetchall():
                        before_text, _ = self._snapshot_texts(r["before_json"])
                        after_text, _ = self._snapshot_texts(r["after_json"])
                        entries.append({
                            "ts": str(r["created_at"] or ""), "source": "evolution",
                            "action": str(r["op"] or ""), "run_id": str(r["run_id"] or ""),
                            "status": str(r["status"] or ""), "reverted_at": str(r["reverted_at"] or ""),
                            "before_text": before_text, "after_text": after_text,
                        })
                except Exception:
                    pass
                # 2) memory_rethink_actions：target 是 JSON 数组列，只能 LIKE（转义通配符）
                try:
                    for r in conn.execute(
                        "SELECT action_pk, run_id, action_type, old_texts_json, new_text, reason, "
                        "validation_status, applied, applied_at, reverted_at, created_at "
                        "FROM memory_rethink_actions WHERE target_item_ids_json LIKE ? ESCAPE '\\' "
                        "ORDER BY action_pk DESC LIMIT 100;",
                        (f"%{_escape_like(iid)}%",),
                    ).fetchall():
                        entries.append({
                            "ts": str(r["created_at"] or ""), "source": "rethink",
                            "action": str(r["action_type"] or ""), "run_id": str(r["run_id"] or ""),
                            "validation_status": str(r["validation_status"] or ""),
                            "applied": int(r["applied"] or 0), "applied_at": str(r["applied_at"] or ""),
                            "reverted_at": str(r["reverted_at"] or ""),
                            "before_text": " | ".join(str(t) for t in (_json_loads(str(r["old_texts_json"] or ""), []) or [])),
                            "after_text": str(r["new_text"] or ""), "reason": str(r["reason"] or ""),
                        })
                except Exception:
                    pass
                # 3) API 自身审计
                for r in conn.execute(
                    "SELECT action_id, action, actor, before_json, after_json, reason, meta_json, created_at "
                    "FROM memory_api_action_log WHERE item_id=? ORDER BY action_id DESC LIMIT 100;",
                    (iid,),
                ).fetchall():
                    before_text, _ = self._snapshot_texts(r["before_json"])
                    after_text, _ = self._snapshot_texts(r["after_json"])
                    entries.append({
                        "ts": str(r["created_at"] or ""), "source": "api",
                        "action": str(r["action"] or ""), "actor": str(r["actor"] or ""),
                        "reason": str(r["reason"] or ""), "meta": _json_loads(str(r["meta_json"] or ""), {}),
                        "before_text": before_text, "after_text": after_text,
                    })
                # 4) payload 内 rethink lineage（最近 12 条，原样透传）
                payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
                lineage = payload.get("memory_rethink_history", []) if isinstance(payload, dict) else []
                if isinstance(lineage, list):
                    for entry in lineage:
                        if isinstance(entry, dict):
                            entries.append({"ts": str(entry.get("ts", "") or entry.get("applied_at", "") or ""),
                                            "source": "rethink_lineage", "action": str(entry.get("action_type", "") or ""),
                                            "detail": entry})
                entries.sort(key=lambda e: str(e.get("ts", "")), reverse=True)
                return {"ok": True, "item_id": iid, "character": char, "entries": entries, "count": len(entries)}
            finally:
                conn.close()

    # ---------- 受控写 ----------

    def update_memory(
        self,
        char: str,
        item_id: str,
        text: str,
        *,
        expected_updated_at: str = "",
        reason: str = "",
        actor: str = "dev_api",
    ) -> dict:
        char = str(char or "").strip()
        new_text = str(text or "").strip()
        expected_version = str(expected_updated_at or "").strip()
        if not new_text:
            return {"ok": False, "error": "text_required"}
        if len(new_text) > _MAX_UPDATE_TEXT_LEN:
            return {"ok": False, "error": "text_too_long", "max_len": _MAX_UPDATE_TEXT_LEN}

        def _write(conn) -> dict:
            item = self._visible_row(conn, char, item_id)
            if item is None:
                return {"ok": False, "error": "not_found"}
            if int(item["archived"] or 0) or item["status"] != "active":
                return {"ok": False, "error": "not_active", "status": item["status"], "archived": item["archived"]}
            iid = item["item_id"]
            current_version = str(item.get("updated_at", "") or "")
            compare_version = expected_version or current_version
            if expected_version and expected_version != current_version:
                return {
                    "ok": False,
                    "error": "stale_conflict",
                    "item_id": iid,
                    "expected_updated_at": expected_version,
                    "actual_updated_at": current_version,
                }
            before = self._snapshot(conn, iid)
            now = utc_now().isoformat(timespec="microseconds").replace("+00:00", "Z")
            if now == current_version:
                now = (utc_now() + timedelta(microseconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
            # 照 rethink rewrite：清 embedding 让 backfill 重算 + 同步 FTS。
            cursor = conn.execute(
                """
                UPDATE memory_items
                SET text=?, embedding_text='', embedding_sparse_text='',
                    embedding_model_id='', embedding_dim=0, updated_at=?
                WHERE item_id=? AND updated_at=?;
                """,
                (new_text, now, iid, compare_version),
            )
            if cursor.rowcount != 1:
                current = self._visible_row(conn, char, iid)
                return {
                    "ok": False,
                    "error": "stale_conflict",
                    "item_id": iid,
                    "expected_updated_at": compare_version,
                    "actual_updated_at": str((current or {}).get("updated_at", "") or ""),
                }
            after = self._snapshot(conn, iid)
            self._write_audit(
                conn,
                item_id=iid,
                character=item["character"],
                action="update",
                actor=actor,
                before=before,
                after=after,
                reason=reason,
                meta={"expected_updated_at": compare_version},
            )
            return {
                "ok": True,
                "item_id": iid,
                "before_text": item["text"],
                "after_text": new_text,
                "updated_at": now,
            }

        return self._store._run_write_transaction(_write)

    def update_core_summary(
        self,
        char: str,
        text: str,
        *,
        reason: str = "",
        actor: str = "dev_api",
    ) -> dict:
        """Update only the fixed per-character core_summary row."""
        character = str(char or "").strip()
        if not character:
            return {"ok": False, "error": "character_required"}
        item_id = self._store.core_summary_item_id(character)
        with self._store._lock:
            conn = self._store._connect()
            try:
                item = self._visible_row(conn, character, item_id)
            finally:
                conn.close()
        if item is None:
            return {"ok": False, "error": "not_found"}
        if str(item.get("item_type", "") or "") != "core_summary":
            return {"ok": False, "error": "not_core_summary"}
        return self.update_memory(
            character,
            item_id,
            text,
            expected_updated_at=str(item.get("updated_at", "") or ""),
            reason=reason,
            actor=actor,
        )

    def _archive_row(self, conn, item: dict, *, actor: str, reason: str, action: str) -> None:
        iid = item["item_id"]
        before = self._snapshot(conn, iid)
        now = _utc_now_iso()
        # 软删模式照 decay（memory_core.py:13414-13418）：状态翻转 + 删 FTS 行，绝不物理删除
        conn.execute(
            "UPDATE memory_items SET archived=1, status='archived', updated_at=? WHERE item_id=?;",
            (now, iid),
        )
        after = self._snapshot(conn, iid)
        created = _parse_iso(item.get("created_at", ""))
        age_h = None
        if created is not None:
            try:
                age_h = max(0.0, (utc_now() - created.astimezone(timezone.utc)).total_seconds() / 3600.0)
            except Exception:
                age_h = None
        within_grace = bool(
            item.get("layer") == "working" and age_h is not None and age_h < _ARCHIVE_GRACE_HOURS
        )
        self._write_audit(conn, item_id=iid, character=item["character"], action=action,
                          actor=actor, before=before, after=after, reason=reason,
                          meta={"within_grace": within_grace, "age_hours": age_h})

    def delete_memory(self, char: str, item_id: str, *, reason: str = "", actor: str = "dev_api") -> dict:
        if not _archive_enabled():
            return {"ok": False, "error": "archive_disabled",
                    "detail": "MEM_RETHINK_ALLOW_ARCHIVE!=1，软删除按 P0 计划默认关闭"}
        char = str(char or "").strip()
        with self._store._lock:
            conn = self._store._connect()
            try:
                item = self._visible_row(conn, char, item_id)
                if item is None:
                    return {"ok": False, "error": "not_found"}
                if int(item["archived"] or 0):
                    return {"ok": False, "error": "already_archived"}
                self._archive_row(conn, item, actor=actor, reason=reason, action="delete")
                conn.commit()
                return {"ok": True, "item_id": item["item_id"], "archived": True}
            finally:
                conn.close()

    def delete_all_memories(self, char: str, *, confirm: bool = False, reason: str = "",
                            actor: str = "dev_api") -> dict:
        if not _archive_enabled():
            return {"ok": False, "error": "archive_disabled",
                    "detail": "MEM_RETHINK_ALLOW_ARCHIVE!=1，软删除按 P0 计划默认关闭"}
        if not confirm:
            return {"ok": False, "error": "confirm_required"}
        char = str(char or "").strip()
        if not char:
            return {"ok": False, "error": "character_required"}
        with self._store._lock:
            conn = self._store._connect()
            try:
                # 只清该角色自己的行；共享行（''/'*'）必须逐条删，防止 per-char 操作误伤全局
                rows = conn.execute(
                    f"SELECT {_ITEM_COLUMNS} FROM memory_items "
                    "WHERE character=? AND archived=0 AND status='active' AND item_type!='core_summary';",
                    (char,),
                ).fetchall()
                archived_count = 0
                for row in rows:
                    item = self._normalize_row(row)
                    if _item_is_guest(item):
                        continue
                    self._archive_row(conn, item, actor=actor, reason=reason, action="delete_all")
                    archived_count += 1
                conn.commit()
                return {"ok": True, "character": char, "archived_count": archived_count}
            finally:
                conn.close()

    # ---------- 预算计数（meta_state KV，KAIROS audit-meta 同款） ----------

    @staticmethod
    def _budget_key(char: str, day: str) -> str:
        return f"memory_budget.{char}.{day}"

    def record_budget_event(self, char: str, kind: str, *, amount: int = 1, tokens: int = 0,
                            day: str = "") -> dict:
        char = str(char or "").strip()
        kind = str(kind or "").strip()
        if not char or kind not in _BUDGET_KINDS:
            return {"ok": False, "error": "bad_budget_event", "kinds": list(_BUDGET_KINDS)}
        day = str(day or "").strip() or _world_today()
        key = self._budget_key(char, day)
        data = self._store.get_meta_state_json(key, None)
        if not isinstance(data, dict):
            data = {}
        for field in _BUDGET_KINDS + ("approx_tokens",):
            data[field] = int(data.get(field, 0) or 0)
        data[kind] += max(0, int(amount or 0))
        data["approx_tokens"] += max(0, int(tokens or 0))
        data["updated_at"] = _utc_now_iso()
        self._store.set_meta_state_json(key, data)
        warn_at = int(os.getenv(_BUDGET_WARN_ENV, str(_BUDGET_WARN_DEFAULT)) or _BUDGET_WARN_DEFAULT)
        if data["approx_tokens"] > warn_at:
            log.warning("memory budget soft limit: %s %s approx_tokens=%s > %s",
                        char, day, data["approx_tokens"], warn_at)
        return {"ok": True, "day": day, "budget": data}

    def get_budget(self, char: str, *, days: int = 7) -> dict:
        char = str(char or "").strip()
        if not char:
            return {"ok": False, "error": "character_required"}
        days = max(1, min(31, int(days or 7)))
        today = now_world()
        history: list[dict] = []
        for i in range(days):
            day = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            data = self._store.get_meta_state_json(self._budget_key(char, day), None)
            if isinstance(data, dict) and data:
                entry = {"day": day}
                for field in _BUDGET_KINDS + ("approx_tokens",):
                    entry[field] = int(data.get(field, 0) or 0)
                history.append(entry)
        today_key = today.strftime("%Y-%m-%d")
        today_data = next((h for h in history if h["day"] == today_key), None)
        return {
            "ok": True, "character": char, "day": today_key,
            "today": today_data or {field: 0 for field in _BUDGET_KINDS + ("approx_tokens",)},
            "history": history,
        }

    # ---------- Phase E/P4 forgetting lifecycle ----------

    def advance_forgetting(
        self,
        char: str,
        *,
        dry_run: bool = True,
        frozen_min_days: float = 90.0,
        tombstone_min_days: float = 180.0,
        actor: str = "dev_api",
    ) -> dict:
        char = str(char or "").strip()
        if not char:
            return {"ok": False, "error": "character_required"}
        if not dry_run and os.getenv("MEM_FORGETTING_APPLY", "0").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            return {"ok": False, "error": "forgetting_apply_disabled"}
        result = self._store.advance_forgetting_lifecycle(
            char,
            frozen_min_days=frozen_min_days,
            tombstone_min_days=tombstone_min_days,
            dry_run=bool(dry_run),
        )
        result["actor"] = str(actor or "dev_api")[:80]
        return result

    def advance_forgetting_dry_run(
        self,
        char: str,
        *,
        frozen_min_days: float = 90.0,
        tombstone_min_days: float = 180.0,
    ) -> dict:
        return self.advance_forgetting(
            char,
            dry_run=True,
            frozen_min_days=frozen_min_days,
            tombstone_min_days=tombstone_min_days,
            actor="dev_api",
        )


_MEMORY_API_SINGLETON: MemoryApi | None = None
_MEMORY_API_LOCK = threading.Lock()


def get_memory_api(store=None) -> MemoryApi:
    """默认绑全库单例；测试显式传 store（直实例化 MemoryApi(store) 亦可）。"""
    global _MEMORY_API_SINGLETON
    if store is not None:
        return MemoryApi(store)
    with _MEMORY_API_LOCK:
        if _MEMORY_API_SINGLETON is None:
            core = get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)
            _MEMORY_API_SINGLETON = MemoryApi(core)
        return _MEMORY_API_SINGLETON


__all__ = ["MemoryApi", "get_memory_api"]
