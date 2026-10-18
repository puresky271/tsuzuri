"""Store mixin: evidence-pool maintenance, JSON snapshots and memory item upserts.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import json
import os
import sqlite3
import time
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import (
    _current_embedding_spec, _emb_encode_both, _emb_sparse_to_text, _emb_to_text,
    _embedding_callable, _safe_float, _safe_int, _utc_now_iso,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _memory_row_has_durable_turn_signal, _memory_snapshot_text_index,
    _memory_text_is_low_value_generalization, _memory_working_text_is_ephemeral_noise,
    _norm_text,
)
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads, _parse_dt
from tsuzuri.memory_core_impl.maintenance_seeds import load_maintenance_seeds
from tsuzuri.memory_core_impl.meta_policy import (
    _derive_time_anchor, _memory_record_meta, _MEMORY_ROLE_MAP, _MIGRATION_DOMAIN_PATHS,
    _NATIVE_MECHANISM_PATHS, _path_get,
)
from tsuzuri.persona_registry import agent_names, collective_scope, default_agent, resolve_agent

class _ItemsMixin:
    """Evidence-pool maintenance, JSON snapshots and memory item upserts."""


    def _append_unique_text(self, rows: list, text: str, *, limit: int = 40) -> bool:
        tx = str(text or "").strip()
        if not isinstance(rows, list) or not tx:
            return False
        norm = _norm_text(tx)
        for row in rows:
            if _norm_text(str(row.get("text", row.get("trait", row)) if isinstance(row, dict) else row)) == norm:
                return False
        rows.append(tx)
        if len(rows) > max(1, int(limit or 40)):
            del rows[:-max(1, int(limit or 40))]
        return True

    def _append_user_model_observation(
        self,
        rel: dict,
        text: str,
        *,
        source_item_ids: list[str] | None = None,
        now_iso: str = "",
    ) -> bool:
        if not isinstance(rel, dict):
            return False
        tx = str(text or "").strip()
        if not tx:
            return False
        now = str(now_iso or _utc_now_iso())
        model = rel.setdefault("user_model", {})
        if not isinstance(model, dict):
            model = {}
            rel["user_model"] = model
        entries = model.get("observation_entries", [])
        if not isinstance(entries, list):
            entries = []
        norm = _norm_text(tx)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if _norm_text(str(entry.get("text", "") or "")) != norm:
                continue
            refs = entry.setdefault("source_item_ids", [])
            changed = False
            if isinstance(refs, list):
                for iid in source_item_ids or []:
                    if iid and iid not in refs:
                        refs.append(iid)
                        changed = True
            if changed:
                entry["last_seen"] = str(entry.get("last_seen", "") or now) or now
                model["updated_at"] = now
            return changed
        entry = {
            "text": tx,
            "weight": 1.0,
            "first_seen": now,
            "last_seen": now,
        }
        refs = [str(x or "").strip() for x in (source_item_ids or []) if str(x or "").strip()]
        if refs:
            entry["source_item_ids"] = refs
        entries.append(entry)
        model["observation_entries"] = entries[:20]
        model["observations"] = [str(x.get("text", "") or "").strip() for x in model["observation_entries"] if isinstance(x, dict) and str(x.get("text", "") or "").strip()]
        model["updated_at"] = now
        return True

    # memory_items_fts 是 external-content 表（content_rowid=rowid），索引必须按 rowid
    # 对齐。2026-10-19 前各写入方手写 `INSERT INTO memory_items_fts(item_id, text)`
    # 不带 rowid、`DELETE ... WHERE item_id=?` 又按当前内容删旧词，索引漂移出孤儿
    # rowid，MATCH 抛 "fts5: missing row" 后整条 FTS 通道被静默吞掉。现由 trigger
    # 统一维护：只索引 active 且非空文本的行，与 _rebuild_memory_items_fts_for_active 一致。
    _FTS_ROW_INDEXED = "{r}.archived=0 AND {r}.status='active' AND COALESCE({r}.text, '') != ''"
    _MEMORY_ITEMS_FTS_TRIGGERS = (
        (
            "memory_items_fts_ai",
            "AFTER INSERT ON memory_items BEGIN "
            "INSERT INTO memory_items_fts(rowid, item_id, text) "
            "SELECT new.rowid, new.item_id, new.text WHERE " + _FTS_ROW_INDEXED.format(r="new") + "; END",
        ),
        (
            "memory_items_fts_ad",
            "AFTER DELETE ON memory_items BEGIN "
            "INSERT INTO memory_items_fts(memory_items_fts, rowid, item_id, text) "
            "SELECT 'delete', old.rowid, old.item_id, old.text WHERE " + _FTS_ROW_INDEXED.format(r="old") + "; END",
        ),
        (
            "memory_items_fts_au",
            "AFTER UPDATE OF item_id, text, archived, status ON memory_items BEGIN "
            "INSERT INTO memory_items_fts(memory_items_fts, rowid, item_id, text) "
            "SELECT 'delete', old.rowid, old.item_id, old.text WHERE " + _FTS_ROW_INDEXED.format(r="old") + "; "
            "INSERT INTO memory_items_fts(rowid, item_id, text) "
            "SELECT new.rowid, new.item_id, new.text WHERE " + _FTS_ROW_INDEXED.format(r="new") + "; END",
        ),
    )

    def _ensure_memory_items_fts_triggers(self, conn: sqlite3.Connection) -> bool:
        """Install the FTS sync triggers; rebuild the index once when they are new."""
        try:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_items_fts';"
            ).fetchone():
                return False
            existing = {
                str(r[0])
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='memory_items';"
                ).fetchall()
            }
            missing = [(name, body) for name, body in self._MEMORY_ITEMS_FTS_TRIGGERS if name not in existing]
            if not missing:
                return False
            for name, body in missing:
                conn.execute(f"CREATE TRIGGER IF NOT EXISTS {name} {body};")
            # 旧索引是手写同步留下的、rowid 不可信：装 trigger 时整体重建一次。
            return self._rebuild_memory_items_fts_for_active(conn)
        except Exception as exc:
            _tsz_emit(f"[MemoryCore][FTS] trigger install failed: {exc}", flush=True)
            return False

    def _rebuild_memory_items_fts_for_active(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.execute("INSERT INTO memory_items_fts(memory_items_fts) VALUES('delete-all');")
            conn.execute(
                """
                INSERT INTO memory_items_fts(rowid, item_id, text)
                SELECT rowid, item_id, text
                FROM memory_items
                WHERE archived=0 AND status='active' AND COALESCE(text, '') != '';
                """
            )
            return True
        except Exception:
            return False

    def _archive_memory_items_for_maintenance(
        self,
        conn: sqlite3.Connection,
        rows: list[dict],
        *,
        category: str,
        reason: str,
        now_iso: str,
        covered_by_dev_memory_path: str = "",
        migration_target: str = "",
    ) -> int:
        archived = 0
        for raw in rows:
            item = dict(raw) if isinstance(raw, dict) else {}
            item_id = str(item.get("item_id", "") or "").strip()
            if not item_id:
                continue
            payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
            if not isinstance(payload, dict):
                payload = {}
            policy = payload.get("evidence_policy", {}) if isinstance(payload.get("evidence_policy", {}), dict) else {}
            policy.update(
                {
                    "category": str(category or "exclude"),
                    "visibility": "archived_source",
                    "reason": str(reason or ""),
                    "source_item_ids": [item_id],
                    "archived_by": "memory_evidence_pool_maintenance",
                    "archived_at": now_iso,
                }
            )
            if covered_by_dev_memory_path:
                policy["covered_by_dev_memory_path"] = str(covered_by_dev_memory_path)
            if migration_target:
                policy["migration_target"] = str(migration_target)
            payload["evidence_policy"] = policy
            history = payload.get("maintenance_history", [])
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "at": now_iso,
                    "action": "archive",
                    "category": str(category or "exclude"),
                    "reason": str(reason or ""),
                }
            )
            payload["maintenance_history"] = history[-8:]
            conn.execute(
                """
                UPDATE memory_items
                SET archived=1, status='archived', payload_json=?, updated_at=?
                WHERE item_id=?;
                """,
                (_json_dumps(payload), now_iso, item_id),
            )
            archived += 1
        return archived

    def maintain_memory_evidence_pool(
        self,
        *,
        dry_run: bool = True,
        working_ttl_days: float = 7.0,
        export_json: bool = True,
    ) -> dict:
        """Curate DB-grounded evidence without treating every row as durable memory."""
        now_iso = _utc_now_iso()
        now_dt = utc_now_naive()
        snapshot = self.load_json_snapshot()
        if not isinstance(snapshot, dict):
            snapshot = {}
        snapshot_before = _json_dumps(snapshot)
        snapshot_index = _memory_snapshot_text_index(snapshot)
        report: dict[str, object] = {
            "ok": True,
            "dry_run": bool(dry_run),
            "created_items": [],
            "archived": {
                "working_ttl": 0,
                "profile_covered": 0,
                "profile_candidate": 0,
                "key_event": 0,
                "explicit_recall_archive": 0,
                "exclude": 0,
            },
            "profile_candidates_written": [],
            "key_events_written": [],
            "generic_candidates_seen": 0,
            "source_rows_missing": [],
            "fts_rebuilt": False,
            "role_diagnostics": {
                "characters": {
                    name: {
                        "old_light_callback_episodic": 0,
                        "short_empty_payload_semantic": 0,
                        "low_value_generic_candidates": 0,
                        "profile_covered_candidates": 0,
                        "working_ttl_candidates": 0,
                        "sample_item_ids": [],
                    }
                    for name in agent_names()
                },
                "collective": {
                    "scope": collective_scope(),
                    "active_collective_items": 0,
                    "sample_item_ids": [],
                },
            },
        }

        lt = snapshot.setdefault("long_term", {})
        if not isinstance(lt, dict):
            lt = {}
            snapshot["long_term"] = lt
        rels = lt.setdefault("character_relationships", {})
        if not isinstance(rels, dict):
            rels = {}
            lt["character_relationships"] = rels
        # Phase D fix #4 (2026-05-01): 维护 helper 不再只对角色A写、所有角色平等。
        # source-backed 种子画像描述的是开发者的通用特征、不是角色独占视角、
        # 所有角色都该共享同一份认知。
        _maintenance_chars = agent_names()
        _maintenance_rels = {}
        for _name in _maintenance_chars:
            _r = rels.setdefault(_name, {})
            if not isinstance(_r, dict):
                _r = {}
                rels[_name] = _r
            _maintenance_rels[_name] = _r
        # backward-compat 别名（默认角色的关系条目）
        default_rel = _maintenance_rels[default_agent()]
        developer_profile = lt.setdefault("developer_profile", {})
        if not isinstance(developer_profile, dict):
            developer_profile = {}
            lt["developer_profile"] = developer_profile
        developer_profile.setdefault("known_preferences", [])
        if not isinstance(developer_profile.get("known_preferences", []), list):
            developer_profile["known_preferences"] = []
        developer_profile.setdefault("profile_evidence", [])
        if not isinstance(developer_profile.get("profile_evidence", []), list):
            developer_profile["profile_evidence"] = []
        key_events = lt.setdefault("key_events", [])
        if not isinstance(key_events, list):
            key_events = []
            lt["key_events"] = key_events

        # 手工整理的开发者画像 / key event / 显式回忆归档及其来源行属于部署数据，
        # 从 gitignored 本地文件读取（memory_core_impl/maintenance_seeds.py）；缺失即无种子。
        _seeds = load_maintenance_seeds()
        fixed_profile_candidates = _seeds["profile_candidates"]
        fixed_key_events = _seeds["key_events"]
        consolidated_items = _seeds["consolidated_items"]
        fixed_exclude_source_ids = _seeds["exclude_source_ids"]

        for row in fixed_profile_candidates:
            text = str(row.get("text", "") or "")
            if self._append_unique_text(developer_profile["known_preferences"], text, limit=30):
                report["profile_candidates_written"].append(text)
            # Phase D fix #4: 所有角色都写、不只角色A
            for _name, _rel in _maintenance_rels.items():
                self._append_user_model_observation(
                    _rel, text,
                    source_item_ids=list(row.get("source_item_ids", [])),
                    now_iso=now_iso,
                )
            evidence = {
                "text": text,
                "source_item_ids": list(row.get("source_item_ids", [])),
                "target_path": str(row.get("target_path", "")),
                "added_at": now_iso,
            }
            existing_evidence_norm = {
                _norm_text(str(x.get("text", "") or "") + "|" + ",".join(str(y or "") for y in (x.get("source_item_ids", []) if isinstance(x.get("source_item_ids", []), list) else [])))
                for x in (developer_profile.get("profile_evidence", []) if isinstance(developer_profile.get("profile_evidence", []), list) else [])
                if isinstance(x, dict)
            }
            evidence_key = _norm_text(text + "|" + ",".join(str(y or "") for y in evidence.get("source_item_ids", [])))
            if evidence_key and evidence_key not in existing_evidence_norm:
                developer_profile["profile_evidence"].append(evidence)
        if report["profile_candidates_written"]:
            developer_profile["updated_at"] = now_iso

        for row in fixed_key_events:
            event_text = str(row.get("event", "") or "")
            if event_text and _norm_text(event_text) not in snapshot_index:
                key_events.append(dict(row))
                report["key_events_written"].append(event_text)

        archive_sources: dict[str, tuple[str, str, str, str]] = {}
        for item in consolidated_items:
            category = str(item.get("category", "") or "")
            source_ids = [str(x or "").strip() for x in (item.get("source_item_ids", []) if isinstance(item.get("source_item_ids", []), list) else []) if str(x or "").strip()]
            for source_id in source_ids:
                if source_id.startswith("shared_info:"):
                    continue
                if category == "profile":
                    archive_sources[source_id] = ("profile_candidate", "promoted_to_dev_memory_profile", "", "dev_memory.json.long_term.developer_profile")
                elif category == "key_event":
                    archive_sources[source_id] = ("key_event", "promoted_to_dev_memory_key_events", "", "dev_memory.json.long_term.key_events")
                elif category == "explicit_recall_archive":
                    archive_sources[source_id] = ("explicit_recall_archive", "consolidated_to_explicit_recall_archive", "", str(item.get("item_id", "")))
        for source_id, reason in fixed_exclude_source_ids.items():
            archive_sources[source_id] = ("exclude", reason, "", "")

        with self._lock:
            conn = self._connect()
            try:
                conn.row_factory = sqlite3.Row
                source_rows = {}
                if archive_sources:
                    ids = list(archive_sources.keys())
                    placeholders = ",".join(["?"] * len(ids))
                    existing_source_ids = set()
                    try:
                        existing_source_ids = {
                            str(r["item_id"] or "")
                            for r in conn.execute(
                                f"SELECT item_id FROM memory_items WHERE item_id IN ({placeholders});",
                                ids,
                            ).fetchall()
                        }
                    except Exception:
                        existing_source_ids = set()
                    for r in conn.execute(
                        f"SELECT * FROM memory_items WHERE archived=0 AND status='active' AND item_id IN ({placeholders});",
                        ids,
                    ).fetchall():
                        source_rows[str(r["item_id"])] = {k: r[k] for k in r.keys()}
                    report["source_rows_missing"] = [iid for iid in ids if iid not in existing_source_ids]

                generic_rows = conn.execute(
                    """
                    SELECT *
                    FROM memory_items
                    WHERE archived=0 AND status='active'
                          AND layer IN ('semantic', 'episodic', 'working', 'collective');
                    """
                ).fetchall()
                rows_by_bucket: dict[str, list[dict]] = {
                    "working_ttl": [],
                    "profile_covered": [],
                    "exclude": [],
                }
                role_diagnostics = report.get("role_diagnostics", {}) if isinstance(report.get("role_diagnostics", {}), dict) else {}
                character_diagnostics = role_diagnostics.get("characters", {}) if isinstance(role_diagnostics.get("characters", {}), dict) else {}
                collective_diagnostics = role_diagnostics.get("collective", {}) if isinstance(role_diagnostics.get("collective", {}), dict) else {}
                for r in generic_rows:
                    item = {k: r[k] for k in r.keys()}
                    text = str(item.get("text", "") or "")
                    item_payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
                    item_payload = item_payload if isinstance(item_payload, dict) else {}
                    char_name = str(item.get("character", "") or "").strip()
                    diag = character_diagnostics.get(char_name) if isinstance(character_diagnostics.get(char_name), dict) else None
                    if char_name == collective_scope():
                        collective_diagnostics["active_collective_items"] = int(collective_diagnostics.get("active_collective_items", 0) or 0) + 1
                        samples = collective_diagnostics.get("sample_item_ids", []) if isinstance(collective_diagnostics.get("sample_item_ids", []), list) else []
                        if len(samples) < 8:
                            samples.append(str(item.get("item_id", "") or ""))
                            collective_diagnostics["sample_item_ids"] = samples
                    if diag is not None:
                        samples = diag.get("sample_item_ids", []) if isinstance(diag.get("sample_item_ids", []), list) else []
                        if len(samples) < 8:
                            samples.append(str(item.get("item_id", "") or ""))
                            diag["sample_item_ids"] = samples
                        payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
                        memory_meta = payload.get("memory_meta", {}) if isinstance(payload, dict) and isinstance(payload.get("memory_meta", {}), dict) else {}
                        surface_default = str(memory_meta.get("surface_default", "") or "")
                        if (
                            str(item.get("layer", "") or "") == "episodic"
                            and str(item.get("memory_role", "") or "") == "episodic"
                            and surface_default == "light_callback"
                        ):
                            diag["old_light_callback_episodic"] = int(diag.get("old_light_callback_episodic", 0) or 0) + 1
                        if (
                            str(item.get("layer", "") or "") == "semantic"
                            and str(item.get("payload_json", "") or "").strip() in ("", "{}")
                            and len(str(text or "")) <= 14
                        ):
                            diag["short_empty_payload_semantic"] = int(diag.get("short_empty_payload_semantic", 0) or 0) + 1
                    if str(item.get("item_id", "") or "") in archive_sources:
                        continue
                    if str(item.get("layer", "") or "") == "collective" or char_name == collective_scope():
                        continue
                    updated = _parse_dt(str(item.get("updated_at", "") or ""))
                    age_days = ((now_dt - updated).total_seconds() / 86400.0) if updated is not None else 999.0
                    if str(item.get("layer", "") or "") == "working":
                        if _memory_row_has_durable_turn_signal(item, item_payload):
                            continue
                        if age_days >= 1.0 and (
                            age_days >= float(working_ttl_days or 7.0)
                            or _memory_working_text_is_ephemeral_noise(text)
                        ):
                            rows_by_bucket["working_ttl"].append(item)
                            if diag is not None:
                                diag["working_ttl_candidates"] = int(diag.get("working_ttl_candidates", 0) or 0) + 1
                        continue
                    if not _memory_text_is_low_value_generalization(text):
                        continue
                    report["generic_candidates_seen"] = int(report.get("generic_candidates_seen", 0) or 0) + 1
                    if diag is not None:
                        diag["low_value_generic_candidates"] = int(diag.get("low_value_generic_candidates", 0) or 0) + 1
                    if (
                        str(item.get("character", "") or "") == default_agent()
                        and (not str(text).startswith(("话题:", "topic:")))
                        and _norm_text(text)
                        and _norm_text(text) in snapshot_index
                    ):
                        rows_by_bucket["profile_covered"].append(item)
                        if diag is not None:
                            diag["profile_covered_candidates"] = int(diag.get("profile_covered_candidates", 0) or 0) + 1
                    else:
                        rows_by_bucket["exclude"].append(item)

                if not dry_run:
                    for item in consolidated_items:
                        payload = {
                            "evidence_policy": {
                                "category": str(item.get("category", "") or ""),
                                "visibility": str(item.get("visibility", "normal") or "normal"),
                                "source_item_ids": list(item.get("source_item_ids", [])),
                                "created_by": "memory_evidence_pool_maintenance",
                                "created_at": now_iso,
                            },
                            "memory_meta": _memory_record_meta(
                                memory_type=str(item.get("item_type", "") or "shared_fact"),
                                owner_scope="shared",
                                evidence_tier="medium",
                                source_kind=("source_backed_archive" if str(item.get("category", "")) == "explicit_recall_archive" else str(item.get("category", "maintenance") or "maintenance")),
                                temporal_validity=("archival" if str(item.get("category", "")) == "explicit_recall_archive" else "persistent"),
                                surface_default=("explicit_recall" if str(item.get("category", "")) == "explicit_recall_archive" else "silent_grounding"),
                                promotion_candidate=False,
                            ),
                        }
                        self._upsert_item(
                            {
                                "item_id": str(item.get("item_id", "") or ""),
                                "character": str(item.get("character", "*") or "*"),
                                "layer": str(item.get("layer", "semantic") or "semantic"),
                                "memory_role": str(item.get("memory_role", "semantic") or "semantic"),
                                "engram_type": str(item.get("engram_type", item.get("item_type", "fact")) or "fact"),
                                "item_type": str(item.get("item_type", "fact") or "fact"),
                                "text": str(item.get("text", "") or "")[:320],
                                "payload_json": _json_dumps(payload),
                                "cue_json": _json_dumps(self._build_cue_bundle(str(item.get("text", "") or ""), character="*", payload=payload, meta={"topics": [str(item.get("category", "") or "")]}, engram_type=str(item.get("engram_type", "") or ""), memory_role=str(item.get("memory_role", "") or ""))),
                                "lifecycle_json": _json_dumps(self._build_lifecycle(created_at=now_iso, updated_at=now_iso, source="memory_evidence_pool_maintenance", access_count=0, state="consolidation")),
                                "salience": 0.72,
                                "confidence": 0.76,
                                "novelty": 0.28,
                                "importance": float(_safe_float(item.get("importance", 0.68), 0.68)),
                                "source": "memory_evidence_pool_maintenance",
                                "evidence_id": "",
                                "created_at": now_iso,
                                "updated_at": now_iso,
                            },
                            conn=conn,
                            sync_fts=False,
                        )
                        report["created_items"].append(str(item.get("item_id", "") or ""))

                    bucket_rows: dict[str, list[dict]] = {"profile_candidate": [], "key_event": [], "explicit_recall_archive": [], "exclude": []}
                    for source_id, (category, reason, covered_path, target) in archive_sources.items():
                        row = source_rows.get(source_id)
                        if row:
                            bucket_rows.setdefault(category, []).append(row)
                    for bucket, rows in bucket_rows.items():
                        report["archived"][bucket] = int(report["archived"].get(bucket, 0)) + self._archive_memory_items_for_maintenance(
                            conn,
                            rows,
                            category=bucket,
                            reason="fixed_candidate_source",
                            now_iso=now_iso,
                            migration_target=("dev_memory.json" if bucket in ("profile_candidate", "key_event") else ""),
                        )
                    report["archived"]["working_ttl"] = self._archive_memory_items_for_maintenance(
                        conn,
                        rows_by_bucket["working_ttl"],
                        category="working_ttl",
                        reason=f"working_older_than_{working_ttl_days}_days_or_ephemeral",
                        now_iso=now_iso,
                    )
                    report["archived"]["profile_covered"] = self._archive_memory_items_for_maintenance(
                        conn,
                        rows_by_bucket["profile_covered"],
                        category="profile_covered",
                        reason="covered_by_existing_dev_memory_profile",
                        now_iso=now_iso,
                        covered_by_dev_memory_path=f"dev_memory.json.long_term.character_relationships.{default_agent()}",
                    )
                    report["archived"]["exclude"] = int(report["archived"].get("exclude", 0)) + self._archive_memory_items_for_maintenance(
                        conn,
                        rows_by_bucket["exclude"],
                        category="exclude",
                        reason="low_value_generic_or_topic_only",
                        now_iso=now_iso,
                    )
                    report["fts_rebuilt"] = self._rebuild_memory_items_fts_for_active(conn)
                    conn.commit()
                else:
                    report["archived"]["working_ttl"] = len(rows_by_bucket["working_ttl"])
                    report["archived"]["profile_covered"] = len(rows_by_bucket["profile_covered"])
                    report["archived"]["exclude"] = len(rows_by_bucket["exclude"]) + sum(1 for sid, meta in archive_sources.items() if meta[0] == "exclude" and sid in source_rows)
                    report["archived"]["profile_candidate"] = sum(1 for sid, meta in archive_sources.items() if meta[0] == "profile_candidate" and sid in source_rows)
                    report["archived"]["key_event"] = sum(1 for sid, meta in archive_sources.items() if meta[0] == "key_event" and sid in source_rows)
                    report["archived"]["explicit_recall_archive"] = sum(1 for sid, meta in archive_sources.items() if meta[0] == "explicit_recall_archive" and sid in source_rows)
            finally:
                conn.close()

        snapshot_changed = _json_dumps(snapshot) != snapshot_before
        report["snapshot_changed"] = bool(snapshot_changed)
        if snapshot_changed and not dry_run:
            self.save_json_snapshot(snapshot, source="memory_evidence_pool_maintenance")
            if export_json:
                self.export_json_snapshot(self.snapshot_path)
        if not dry_run:
            self._set_meta_json("memory_maintenance.evidence_pool.last_run", report)
        return report

    def get_recent_ooc_audits(self, character: str, limit: int = 5) -> list[dict]:
        char = resolve_agent(character)
        row_limit = max(1, min(50, _safe_int(limit, 5)))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT audit_id, character, run_id, ooc_score, style_deviation_json, evidence_hits_json,
                           high_conflict_claims_json, penalty_json, applied, shadow, created_at, updated_at
                    FROM ooc_audit_runs
                    WHERE character=?
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (char, row_limit),
                ).fetchall()
            finally:
                conn.close()
        out: list[dict] = []
        for row in rows:
            penalty = _json_loads(str(row["penalty_json"] or "{}"), {})
            out.append(
                {
                    "audit_id": str(row["audit_id"] or ""),
                    "character": str(row["character"] or ""),
                    "run_id": str(row["run_id"] or ""),
                    "ooc_score": round(_safe_float(row["ooc_score"], 0.0), 3),
                    "style_deviation": _json_loads(str(row["style_deviation_json"] or "{}"), {}),
                    "style_deviation_axes": dict(
                        penalty.get("style_deviation_axes", {})
                        if isinstance(penalty.get("style_deviation_axes", {}), dict)
                        else {}
                    ),
                    "evidence_hits": _json_loads(str(row["evidence_hits_json"] or "[]"), []),
                    "high_conflict_claims": _json_loads(str(row["high_conflict_claims_json"] or "[]"), []),
                    "penalty_recommendation": penalty if isinstance(penalty, dict) else {},
                    "applied": bool(int(row["applied"] or 0)),
                    "shadow": bool(int(row["shadow"] or 0)),
                    "created_at": str(row["created_at"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                }
            )
        return out

    def get_recent_evolution_runs(self, character: str, limit: int = 5) -> list[dict]:
        char = resolve_agent(character)
        row_limit = max(1, min(50, _safe_int(limit, 5)))
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT run_id, character, shadow, status, proposal_json, evaluation_json,
                           applied_actions_json, base_snapshot_hash, created_at, updated_at
                    FROM evolution_runs
                    WHERE character=?
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (char, row_limit),
                ).fetchall()
            finally:
                conn.close()
        out: list[dict] = []
        for row in rows:
            proposal = _json_loads(str(row["proposal_json"] or "[]"), [])
            evaluation = _json_loads(str(row["evaluation_json"] or "{}"), {})
            applied_actions = _json_loads(str(row["applied_actions_json"] or "[]"), [])
            out.append(
                {
                    "run_id": str(row["run_id"] or ""),
                    "character": str(row["character"] or ""),
                    "shadow": bool(int(row["shadow"] or 0)),
                    "status": str(row["status"] or ""),
                    "proposal": proposal if isinstance(proposal, list) else [],
                    "evaluation": evaluation if isinstance(evaluation, dict) else {},
                    "applied_actions": applied_actions if isinstance(applied_actions, list) else [],
                    "action_count": len(proposal if isinstance(proposal, list) else []),
                    "applied_action_count": len(applied_actions if isinstance(applied_actions, list) else []),
                    "quality": round(
                        _safe_float(
                            evaluation.get("quality", 0.0) if isinstance(evaluation, dict) else 0.0,
                            0.0,
                        ),
                        3,
                    ),
                    "base_snapshot_hash": str(row["base_snapshot_hash"] or ""),
                    "created_at": str(row["created_at"] or ""),
                    "updated_at": str(row["updated_at"] or ""),
                }
            )
        return out

    def _bootstrap_from_snapshot_if_needed(self) -> None:
        done = bool(self._get_meta_json("bootstrap_done", False))
        if done:
            return
        mem = {}
        if os.path.isfile(self.snapshot_path):
            try:
                with open(self.snapshot_path, "r", encoding="utf-8") as f:
                    mem = json.loads(f.read())
            except Exception:
                mem = {}
        if isinstance(mem, dict) and mem:
            self.save_json_snapshot(mem, source="bootstrap")
            self._ingest_snapshot_to_items(mem)
            self._sync_domains_from_snapshot(mem)
            self._sync_mechanisms_from_snapshot(mem)
        self._set_meta_json("bootstrap_done", True)

    def load_json_snapshot(self) -> dict:
        snap = self._get_meta_json("latest_snapshot", {})
        if not isinstance(snap, dict):
            snap = {}
        snap = self._overlay_domains_into_snapshot(snap)
        snap = self._overlay_mechanisms_into_snapshot(snap)
        return snap

    def save_json_snapshot(self, snapshot: dict, *, source: str = "runtime") -> None:
        """把 snapshot 写入 memory_core.db。

        历史实现里 4 段子操作（meta_state×2 + domain×N + mechanism×1）各开一次
        连接 + 各自 commit，10+ 次抢锁让 WAL writer lock 频繁 acquire/release，
        与外部并发写入（_save_dev_memory 多 call site / kairos / news / autogreet）
        叠加时容易 BUSY 超 30s 导致 fallback。

        现改为：单连接 + 单事务 + 顶层 retry。同一次 snapshot 写入只抢一次锁，
        BUSY 时 sleep 0.5/1.0/2.0s 重试 3 次再上抛，让上层 fallback 路径更难触发。
        """
        if not isinstance(snapshot, dict):
            return
        payload = {
            "source": str(source or "runtime"),
            "saved_at": _utc_now_iso(),
            "snapshot": snapshot,
        }
        now = _utc_now_iso()

        # 收集所有要写的 domain / mechanism 数据（在拿锁之前完成）
        domain_writes: list[tuple[str, str]] = []
        for path in _MIGRATION_DOMAIN_PATHS:
            v = _path_get(snapshot, path)
            if v is None:
                continue
            domain_writes.append((str(path), _json_dumps(v)))

        mechanism_writes: list[tuple[str, str, str]] = []  # (path, phase, value_json)
        mechanism_entries: dict[str, list[dict]] = {}
        for path in _NATIVE_MECHANISM_PATHS:
            v = _path_get(snapshot, path)
            if v is None:
                continue
            phase = self._mechanism_phase(path)
            mechanism_writes.append((str(path), str(phase), _json_dumps(v)))
            try:
                mechanism_entries[path] = self._extract_mechanism_entries(path, v) or []
            except Exception:
                mechanism_entries[path] = []

        snapshot_blob = _json_dumps(payload.get("snapshot", {}))
        meta_blob = _json_dumps(payload)
        domain_meta_blob = _json_dumps({"at": now, "paths": len(_MIGRATION_DOMAIN_PATHS)})
        mech_meta_blob = _json_dumps(
            {
                "at": now,
                "paths": len(mechanism_writes),
                "entry_counts": {p: len(mechanism_entries.get(p, [])) for p, _, _ in mechanism_writes},
            }
        )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with self._lock:
                    conn = self._connect()
                    try:
                        # 单事务：BEGIN IMMEDIATE 拿到 reserved lock，之后所有写入连续完成
                        conn.execute("BEGIN IMMEDIATE;")
                        # ── meta_state ──
                        conn.execute(
                            """
                            INSERT INTO meta_state(state_key, value_json, updated_at)
                            VALUES (?, ?, ?)
                            ON CONFLICT(state_key) DO UPDATE SET
                                value_json=excluded.value_json,
                                updated_at=excluded.updated_at;
                            """,
                            ("latest_snapshot", snapshot_blob, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO meta_state(state_key, value_json, updated_at)
                            VALUES (?, ?, ?)
                            ON CONFLICT(state_key) DO UPDATE SET
                                value_json=excluded.value_json,
                                updated_at=excluded.updated_at;
                            """,
                            ("latest_snapshot_meta", meta_blob, now),
                        )
                        # ── memory_domains ──
                        for dpath, dblob in domain_writes:
                            conn.execute(
                                """
                                INSERT INTO memory_domains(domain_key, domain_json, updated_at, version)
                                VALUES (?, ?, ?, 1)
                                ON CONFLICT(domain_key) DO UPDATE SET
                                    domain_json=excluded.domain_json,
                                    updated_at=excluded.updated_at;
                                """,
                                (dpath, dblob, now),
                            )
                        if domain_writes:
                            conn.execute(
                                """
                                INSERT INTO meta_state(state_key, value_json, updated_at)
                                VALUES (?, ?, ?)
                                ON CONFLICT(state_key) DO UPDATE SET
                                    value_json=excluded.value_json,
                                    updated_at=excluded.updated_at;
                                """,
                                ("domain_sync_at", domain_meta_blob, now),
                            )
                        # ── mechanism_state + mechanism_entries ──
                        for mpath, mphase, mblob in mechanism_writes:
                            conn.execute(
                                """
                                INSERT INTO mechanism_state(mechanism_key, phase, value_json, updated_at, version)
                                VALUES (?, ?, ?, ?, 1)
                                ON CONFLICT(mechanism_key) DO UPDATE SET
                                    phase=excluded.phase,
                                    value_json=excluded.value_json,
                                    updated_at=excluded.updated_at,
                                    version=mechanism_state.version+1;
                                """,
                                (mpath, mphase, mblob, now),
                            )
                            conn.execute("DELETE FROM mechanism_entries WHERE mechanism_key=?;", (mpath,))
                            for r in mechanism_entries.get(mpath, []):
                                conn.execute(
                                    """
                                    INSERT INTO mechanism_entries
                                    (mechanism_key, phase, owner_key, entry_key, content_text, payload_json, ts, weight, confidence, updated_at, version)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ON CONFLICT(mechanism_key, owner_key, entry_key) DO UPDATE SET
                                        phase=excluded.phase,
                                        content_text=excluded.content_text,
                                        payload_json=excluded.payload_json,
                                        ts=excluded.ts,
                                        weight=excluded.weight,
                                        confidence=excluded.confidence,
                                        updated_at=excluded.updated_at,
                                        version=mechanism_entries.version+1;
                                    """,
                                    (
                                        str(r.get("mechanism_key", "")),
                                        str(r.get("phase", "phase1")),
                                        str(r.get("owner_key", "")),
                                        str(r.get("entry_key", "")),
                                        str(r.get("content_text", "")),
                                        str(r.get("payload_json", "{}")),
                                        str(r.get("ts", "")),
                                        float(_safe_float(r.get("weight", 0.0), 0.0)),
                                        float(_safe_float(r.get("confidence", 0.0), 0.0)),
                                        str(r.get("updated_at", now)),
                                        int(_safe_int(r.get("version", 1), 1)),
                                    ),
                                )
                        if mechanism_writes:
                            conn.execute(
                                """
                                INSERT INTO meta_state(state_key, value_json, updated_at)
                                VALUES (?, ?, ?)
                                ON CONFLICT(state_key) DO UPDATE SET
                                    value_json=excluded.value_json,
                                    updated_at=excluded.updated_at;
                                """,
                                ("mechanism_sync_at", mech_meta_blob, now),
                            )
                        conn.commit()
                        return
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
            except sqlite3.OperationalError as e:
                last_err = e
                msg = str(e or "").lower()
                if "locked" in msg or "busy" in msg:
                    # BUSY/locked：退避后重试 3 次（0.5 / 1.0 / 2.0s）
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise
            except Exception:
                raise
        # 3 次都 BUSY：上抛、由调用方走 JSON fallback
        if last_err is not None:
            raise last_err

    def export_json_snapshot(self, out_path: str | None = None) -> dict:
        snap = self.load_json_snapshot()
        path = str(out_path or self.snapshot_path or "").strip()
        if path:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(snap, ensure_ascii=False, indent=2))
            os.replace(tmp, path)
        return snap

    def _make_item_id(self, character: str, layer: str, item_type: str, text: str, stamp: str = "") -> str:
        seed = "|".join(
            [
                str(character or "").strip(),
                str(layer or "").strip(),
                str(item_type or "").strip(),
                _norm_text(text)[:220],
                str(stamp or "").strip(),
            ]
        )
        return hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()[:24]

    def _normalize_memory_role(self, layer: str, explicit_role: str = "") -> str:
        role = str(explicit_role or "").strip().lower()
        if role in ("working", "episodic", "semantic", "control", "collective"):
            return role
        return _MEMORY_ROLE_MAP.get(str(layer or "").strip().lower(), "episodic")

    def _build_cue_bundle(
        self,
        text: str,
        *,
        character: str = "",
        payload: dict | None = None,
        meta: dict | None = None,
        engram_type: str = "",
        memory_role: str = "",
    ) -> dict:
        txt = str(text or "").strip()
        pl = payload if isinstance(payload, dict) else {}
        mt = meta if isinstance(meta, dict) else {}
        raw_entities = []
        for v in (
            pl.get("actors", []),
            pl.get("targets", []),
            mt.get("entities", []),
        ):
            if isinstance(v, list):
                raw_entities.extend([str(x or "").strip() for x in v if str(x or "").strip()])
        for candidate in [character, pl.get("location", ""), pl.get("school_phase", ""), pl.get("session_phase", ""), pl.get("group_status", "")]:
            s = str(candidate or "").strip()
            if s:
                raw_entities.append(s)
        entities = []
        seen = set()
        for item in raw_entities:
            if (not item) or (item in seen):
                continue
            seen.add(item)
            entities.append(item[:32])
        cue = {
            "entities": entities[:12],
            "topics": [str(x or "").strip()[:24] for x in (mt.get("topics", []) if isinstance(mt.get("topics", []), list) else []) if str(x or "").strip()][:10],
            "time_anchor": _derive_time_anchor(txt, pl),
            "location_anchor": str(pl.get("location", "") or mt.get("location_anchor", "") or "").strip()[:24],
            "emotion_anchor": str(pl.get("emotion_anchor", "") or mt.get("emotion_anchor", "") or "").strip()[:24],
            "school_phase": str(pl.get("school_phase", "") or mt.get("school_phase", "") or "").strip()[:24],
            "session_phase": str(pl.get("session_phase", "") or mt.get("session_phase", "") or "").strip()[:24],
            "group_status": str(pl.get("group_status", "") or mt.get("group_status", "") or "").strip()[:24],
            "memory_role": str(memory_role or "").strip()[:16],
            "engram_type": str(engram_type or "").strip()[:24],
        }
        return cue

    def _build_lifecycle(
        self,
        *,
        created_at: str,
        updated_at: str,
        source: str,
        access_count: int = 0,
        last_retrieved_at: str = "",
        state: str = "encoding",
    ) -> dict:
        return {
            "encoding_at": str(created_at or updated_at or _utc_now_iso()),
            "consolidated_at": str(updated_at or created_at or _utc_now_iso()),
            "last_retrieved_at": str(last_retrieved_at or ""),
            "last_updated_at": str(updated_at or created_at or _utc_now_iso()),
            "state": str(state or "encoding"),
            "access_count": int(max(0, _safe_int(access_count, 0))),
            "source": str(source or "").strip()[:40],
        }

    def _refresh_lifecycle_on_retrieval(self, lifecycle: dict | None, *, now: str, source: str = "retrieval") -> dict:
        data = dict(lifecycle or {}) if isinstance(lifecycle, dict) else {}
        encoding_at = str(data.get("encoding_at", "") or data.get("created_at", "") or now)
        consolidated_at = str(data.get("consolidated_at", "") or data.get("last_updated_at", "") or encoding_at or now)
        access_count = int(max(0, _safe_int(data.get("access_count", 0), 0))) + 1
        # Store real access timestamps for accurate ACT-R activation
        _access_times = data.get("access_times", [])
        if not isinstance(_access_times, list):
            _access_times = []
        _access_times.append(str(now or ""))
        if len(_access_times) > 20:
            _access_times = _access_times[-20:]  # Keep last 20 access timestamps
        refreshed = {
            "encoding_at": encoding_at,
            "consolidated_at": consolidated_at,
            "last_retrieved_at": str(now or ""),
            "last_updated_at": str(data.get("last_updated_at", "") or consolidated_at or now),
            "state": "retrieval",
            "access_count": access_count,
            "access_times": _access_times,
            "source": str(data.get("source", "") or source or "retrieval").strip()[:40],
        }
        for key, value in data.items():
            if key not in refreshed:
                refreshed[key] = value
        return refreshed

    def _upsert_item(self, item: dict, conn: sqlite3.Connection | None = None, *, sync_fts: bool = True) -> None:
        now = _utc_now_iso()
        row = dict(item or {})
        row.setdefault("item_id", self._make_item_id(row.get("character", ""), row.get("layer", ""), row.get("item_type", ""), row.get("text", ""), row.get("created_at", "")))
        row.setdefault("character", "")
        row.setdefault("layer", "episodic")
        row.setdefault("memory_role", self._normalize_memory_role(row.get("layer", ""), row.get("memory_role", "")))
        row.setdefault("engram_type", str(row.get("item_type", "note") or "note"))
        row.setdefault("item_type", "note")
        row.setdefault("text", "")
        row.setdefault("payload_json", "{}")
        row.setdefault("cue_json", _json_dumps(self._build_cue_bundle(str(row.get("text", "") or ""), character=str(row.get("character", "") or ""), payload=_json_loads(str(row.get("payload_json", "{}") or "{}"), {}), engram_type=str(row.get("engram_type", "") or str(row.get("item_type", "note") or "note")), memory_role=str(row.get("memory_role", "") or ""))))
        row.setdefault("lifecycle_json", _json_dumps(self._build_lifecycle(created_at=str(row.get("created_at", "") or now), updated_at=str(row.get("updated_at", "") or now), source=str(row.get("source", "") or "runtime"), access_count=_safe_int(row.get("access_count", 0), 0), last_retrieved_at=str(row.get("last_accessed_at", "") or ""), state="consolidation")))
        row.setdefault("salience", 0.5)
        row.setdefault("confidence", 0.5)
        row.setdefault("novelty", 0.5)
        row.setdefault("importance", 0.5)
        row.setdefault("status", "active")
        row.setdefault("source", "runtime")
        row.setdefault("evidence_id", "")
        row.setdefault("created_at", now)
        row.setdefault("updated_at", now)
        row.setdefault("last_accessed_at", "")
        row.setdefault("access_count", 0)
        row.setdefault("archived", 0)
        row.setdefault("topic_labels_json", "[]")
        # Compute dense+sparse features before acquiring the write lock.
        if (
            str(row.get("item_type", "") or "") != "core_summary"
            and not row.get("embedding_text")
            and _embedding_callable()
        ):
            try:
                dense, sparse = _emb_encode_both(str(row.get("text", "") or ""))
                if dense is not None:
                    model_id, declared_dim = _current_embedding_spec()
                    actual_dim = len(dense)
                    if declared_dim > 0 and actual_dim != declared_dim:
                        _tsz_emit(
                            f"[MemoryCore][Embedding] encoder dimension mismatch: "
                            f"model={model_id} actual={actual_dim} declared={declared_dim}",
                            flush=True,
                        )
                    else:
                        row["embedding_text"] = _emb_to_text(dense)
                        row["embedding_sparse_text"] = _emb_sparse_to_text(sparse) if sparse else ""
                        row["embedding_model_id"] = model_id
                        row["embedding_dim"] = actual_dim
            except Exception as exc:
                _tsz_emit(f"[MemoryCore][Embedding] encode_both failed: {exc}", flush=True)
        row.setdefault("embedding_text", "")
        row.setdefault("embedding_sparse_text", "")
        row.setdefault("embedding_model_id", "")
        row.setdefault("embedding_dim", 0)

        def _write(db: sqlite3.Connection) -> None:
            write_cursor = db.execute(
                    """
                    INSERT INTO memory_items
                    (item_id, character, layer, memory_role, engram_type, item_type, text, payload_json, cue_json, lifecycle_json, salience, confidence, novelty, importance, status, source, evidence_id, created_at, updated_at, last_accessed_at, access_count, archived, embedding_text, embedding_sparse_text, embedding_model_id, embedding_dim, topic_labels_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(item_id) DO UPDATE SET
                        character=excluded.character,
                        layer=excluded.layer,
                        memory_role=excluded.memory_role,
                        engram_type=excluded.engram_type,
                        item_type=excluded.item_type,
                        text=excluded.text,
                        payload_json=excluded.payload_json,
                        cue_json=excluded.cue_json,
                        lifecycle_json=excluded.lifecycle_json,
                        salience=excluded.salience,
                        confidence=excluded.confidence,
                        novelty=excluded.novelty,
                        importance=excluded.importance,
                        status=excluded.status,
                        source=excluded.source,
                        evidence_id=excluded.evidence_id,
                        updated_at=excluded.updated_at,
                        archived=excluded.archived,
                        embedding_text=CASE WHEN excluded.embedding_text != '' THEN excluded.embedding_text ELSE memory_items.embedding_text END,
                        embedding_sparse_text=CASE WHEN excluded.embedding_text != '' THEN excluded.embedding_sparse_text ELSE memory_items.embedding_sparse_text END,
                        embedding_model_id=CASE WHEN excluded.embedding_text != '' THEN excluded.embedding_model_id ELSE memory_items.embedding_model_id END,
                        embedding_dim=CASE WHEN excluded.embedding_text != '' THEN excluded.embedding_dim ELSE memory_items.embedding_dim END,
                        topic_labels_json=CASE WHEN excluded.topic_labels_json != '[]' THEN excluded.topic_labels_json ELSE memory_items.topic_labels_json END
                    WHERE memory_items.status != 'tombstone'
                       OR excluded.source IN ('core_summary_rebuild');
                    """,
                    (
                        str(row["item_id"]),
                        str(row["character"]),
                        str(row["layer"]),
                        str(row["memory_role"]),
                        str(row["engram_type"]),
                        str(row["item_type"]),
                        str(row["text"]),
                        str(row["payload_json"]),
                        str(row["cue_json"]),
                        str(row["lifecycle_json"]),
                        float(_clamp(_safe_float(row["salience"], 0.5))),
                        float(_clamp(_safe_float(row["confidence"], 0.5))),
                        float(_clamp(_safe_float(row["novelty"], 0.5))),
                        float(_clamp(_safe_float(row["importance"], 0.5))),
                        str(row["status"]),
                        str(row["source"]),
                        str(row["evidence_id"]),
                        str(row["created_at"]),
                        str(row["updated_at"]),
                        str(row["last_accessed_at"]),
                        int(_safe_int(row["access_count"], 0)),
                        int(_safe_int(row["archived"], 0)),
                        str(row.get("embedding_text", "") or ""),
                        str(row.get("embedding_sparse_text", "") or ""),
                        str(row.get("embedding_model_id", "") or ""),
                        int(_safe_int(row.get("embedding_dim", 0), 0)),
                        str(row.get("topic_labels_json", "[]") or "[]"),
                    ),
                )
            # FTS5 索引由 memory_items 上的 trigger 同步；sync_fts 仅为兼容旧调用保留。

        if conn is not None:
            with self._lock:
                _write(conn)
            return
        self._run_write_transaction(_write)

    def _ingest_snapshot_to_items(self, memory: dict) -> None:
        if not isinstance(memory, dict):
            return
        chars = list(agent_names())
        now = _utc_now_iso()
        for c in chars:
            bw = memory.get("_brain_working", {}).get(c, {}) if isinstance(memory.get("_brain_working"), dict) else {}
            if isinstance(bw, dict):
                for i, msg in enumerate((bw.get("recent_user_msgs", []) if isinstance(bw.get("recent_user_msgs", []), list) else [])[-12:]):
                    tx = str(msg or "").strip()
                    if not tx:
                        continue
                    self._upsert_item(
                        {
                            "item_id": self._make_item_id(c, "working", "recent_user_msg", tx, str(i)),
                            "character": c,
                            "layer": "working",
                            "item_type": "recent_user_msg",
                            "text": tx[:220],
                            "payload_json": _json_dumps({"kind": "recent_user_msg"}),
                            "salience": 0.45,
                            "confidence": 0.65,
                            "novelty": 0.30,
                            "importance": 0.40,
                            "source": "snapshot_bootstrap",
                            "created_at": now,
                            "updated_at": now,
                        }
                    )
            be = memory.get("_brain_episodic", {}).get(c, []) if isinstance(memory.get("_brain_episodic"), dict) else []
            if isinstance(be, list):
                for e in be[-140:]:
                    if not isinstance(e, dict):
                        continue
                    tx = str(e.get("text", "") or "").strip()
                    if not tx:
                        continue
                    created = str(e.get("ts", "") or now)
                    self._upsert_item(
                        {
                            "item_id": self._make_item_id(c, "episodic", "event", tx, created),
                            "character": c,
                            "layer": "episodic",
                            "item_type": "event",
                            "text": tx[:260],
                            "payload_json": _json_dumps(e),
                            "salience": _clamp(_safe_float(e.get("salience", 0.5), 0.5)),
                            "confidence": _clamp(_safe_float(e.get("confidence", 0.55), 0.55)),
                            "novelty": _clamp(_safe_float(e.get("surprise", 0.45), 0.45)),
                            "importance": _clamp(0.45 * _safe_float(e.get("salience", 0.5), 0.5) + 0.30 * _safe_float(e.get("surprise", 0.5), 0.5) + 0.25),
                            "source": "snapshot_bootstrap",
                            "created_at": created,
                            "updated_at": now,
                            "evidence_id": str(e.get("id", "") or ""),
                        }
                    )
            bs = memory.get("_brain_semantic", {}).get(c, []) if isinstance(memory.get("_brain_semantic"), dict) else []
            if isinstance(bs, list):
                for f in bs[-120:]:
                    if not isinstance(f, dict):
                        continue
                    fact = str(f.get("fact", "") or "").strip()
                    if not fact:
                        continue
                    created = str(f.get("at", "") or now)
                    weight = _clamp(_safe_float(f.get("weight", 0.6), 0.6))
                    self._upsert_item(
                        {
                            "item_id": self._make_item_id(c, "semantic", "fact", fact, created),
                            "character": c,
                            "layer": "semantic",
                            "item_type": "fact",
                            "text": fact[:240],
                            "payload_json": _json_dumps(f),
                            "salience": max(0.55, weight),
                            "confidence": max(0.60, weight),
                            "novelty": 0.28,
                            "importance": max(0.60, weight),
                            "source": "snapshot_bootstrap",
                            "created_at": created,
                            "updated_at": now,
                        }
                    )
            bp = memory.get("_brain_procedural", {}).get(c, {}) if isinstance(memory.get("_brain_procedural"), dict) else {}
            rules = bp.get("rules_v2", []) if isinstance(bp, dict) and isinstance(bp.get("rules_v2", []), list) else []
            for r in rules[-80:]:
                if not isinstance(r, dict):
                    continue
                tx = str(r.get("text", "") or "").strip()
                if not tx:
                    continue
                created = str(r.get("added_at", "") or now)
                strength = _clamp(_safe_float(r.get("strength", 0.55), 0.55))
                self._upsert_item(
                    {
                        "item_id": self._make_item_id(c, "procedural", "rule", tx, created),
                        "character": c,
                        "layer": "procedural",
                        "item_type": "rule",
                        "text": tx[:200],
                        "payload_json": _json_dumps(r),
                        "salience": strength,
                        "confidence": max(0.55, strength),
                        "novelty": 0.20,
                        "importance": max(0.58, strength),
                        "source": "snapshot_bootstrap",
                        "created_at": created,
                        "updated_at": now,
                    }
                )

    # ═════════════════════════════════════════════════════════════════════════
    # Phase Q.W2 / Option E Layer 3: Reuse Signal API (2026-05-04)
    # ═════════════════════════════════════════════════════════════════════════

    def record_reuse_signal(
        self,
        working_item_id: str,
        character: str,
        signal_type: str,
        *,
        confidence: float = 0.7,
        triggering_turn_idx: int = 0,
        evidence: str = "",
    ) -> bool:
        """Record a reuse signal for a working turn (Layer 3).

        signal_type one of:
          - "confirmed_reuse": user 在后续 turn 提到了 working 的 entity
          - "direct_recall_hit": user 显式问到 + 命中此 working
          - "assistant_reference_accepted": 角色引用、user 接续未否认
          - "assistant_reference_rejected": 角色引用、user 否认（contra signal）

        Side effect: bumps payload.memory_policy.confirmed_reuse_count (or 对应 counter)
        for accepted-type signals (rejected does NOT bump).
        """
        iid = str(working_item_id or "").strip()
        char = str(character or "").strip()
        st = str(signal_type or "").strip()
        if not iid or not char or st not in (
            "confirmed_reuse", "direct_recall_hit",
            "assistant_reference_accepted", "assistant_reference_rejected",
        ):
            return False
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                # Insert log row
                conn.execute(
                    """
                    INSERT INTO reuse_signal_log
                    (working_item_id, character, signal_type, confidence, triggering_turn_idx, evidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (iid, char, st, float(confidence), int(triggering_turn_idx), str(evidence)[:300], now),
                )
                # Bump counter in payload (positive signals only)
                counter_field = {
                    "confirmed_reuse": "confirmed_reuse_count",
                    "direct_recall_hit": "direct_recall_hit_count",
                    "assistant_reference_accepted": "assistant_reference_accepted_count",
                }.get(st)
                if counter_field:
                    row = conn.execute(
                        "SELECT payload_json FROM memory_items WHERE item_id=?;", (iid,)
                    ).fetchone()
                    if row is not None:
                        try:
                            payload = _json_loads(str(row[0] or "{}"), {})
                            if not isinstance(payload, dict):
                                payload = {}
                            mp = payload.get("memory_policy", {})
                            if not isinstance(mp, dict):
                                mp = {}
                            mp[counter_field] = int(mp.get(counter_field, 0) or 0) + 1
                            payload["memory_policy"] = mp
                            conn.execute(
                                "UPDATE memory_items SET payload_json=? WHERE item_id=?;",
                                (_json_dumps(payload), iid)
                            )
                        except Exception:
                            pass
                conn.commit()
                return True
            finally:
                conn.close()

    def get_reuse_signal_count(self, working_item_id: str, signal_type: str = "") -> int:
        """Read total reuse signal count for an item (across all signal types or filtered)."""
        iid = str(working_item_id or "").strip()
        if not iid:
            return 0
        with self._lock:
            conn = self._connect()
            try:
                if signal_type:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM reuse_signal_log WHERE working_item_id=? AND signal_type=?;",
                        (iid, signal_type)
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM reuse_signal_log WHERE working_item_id=?;",
                        (iid,)
                    ).fetchone()
                return int(row[0]) if row else 0
            finally:
                conn.close()

    def get_recent_working_items(self, character: str, limit: int = 5,
                                  exclude_ephemeral: bool = True) -> list[dict]:
        """Fetch recent working turns for a character (for Layer 3 chat_server lookup)."""
        char = str(character or "").strip()
        if not char:
            return []
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, text, payload_json, created_at, updated_at
                    FROM memory_items
                    WHERE character=? AND memory_role='working' AND layer='working'
                      AND engram_type='turn_exchange' AND archived=0
                    ORDER BY created_at DESC
                    LIMIT ?;
                    """,
                    (char, max(1, int(limit or 5)))
                ).fetchall()
                out = []
                for r in rows:
                    text = str(r["text"] or "")
                    if exclude_ephemeral and _memory_working_text_is_ephemeral_noise(text):
                        continue
                    out.append({
                        "item_id": str(r["item_id"]),
                        "text": text,
                        "payload_json": str(r["payload_json"] or ""),
                        "created_at": str(r["created_at"] or ""),
                        "updated_at": str(r["updated_at"] or ""),
                    })
                return out
            finally:
                conn.close()
