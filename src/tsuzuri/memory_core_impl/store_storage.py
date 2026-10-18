"""Store mixin: connections, schema, domains/mechanisms, meta state, audits and retention.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Callable
from tsuzuri.reliable_io import interprocess_file_lock
from tsuzuri.sqlite_reliability import connect_sqlite, is_sqlite_busy_error
from tsuzuri.time_source import WORLD_TZ, utc_now
from tsuzuri.memory_core_impl.config import _safe_float, _safe_int, _utc_now_iso
from tsuzuri.memory_core_impl.recall_scoring import _json_dumps, _json_loads
from tsuzuri.memory_core_impl.meta_policy import (
    _DEFAULT_POLICY_PROFILE_V2, _derive_confidence, _derive_text, _derive_ts, _derive_weight,
    _GOSSIP_AUDIT_HASH_LEN, _GOSSIP_AUDIT_PREVIEW_LEN, _json_clone, _MECHANISM_PHASES,
    _MIGRATION_DOMAIN_PATHS, _NATIVE_MECHANISM_PATHS, _normalize_policy_profile, _parse_bool,
    _path_get, _path_set, _stable_hash,
)

class _StorageMixin:
    """Connections, schema, domains, meta state, audits and retention."""


    def _connect(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        self._maybe_quick_check_connection()
        return conn

    @staticmethod
    def _is_sqlite_busy_error(exc: BaseException) -> bool:
        return is_sqlite_busy_error(exc)

    def _write_retry_attempts(self) -> int:
        try:
            return max(1, int(os.getenv("MEM_SQLITE_WRITE_ATTEMPTS", "4")))
        except Exception:
            return 4

    def _run_write_transaction(self, operation: Callable[[sqlite3.Connection], object]):
        """Run one write transaction behind the process and cross-process locks."""
        attempts = self._write_retry_attempts()
        try:
            base_delay = max(0.01, float(os.getenv("MEM_SQLITE_WRITE_BACKOFF_SEC", "0.1")))
        except Exception:
            base_delay = 0.1
        try:
            lock_timeout = max(0.1, float(os.getenv("MEM_SQLITE_WRITE_LOCK_TIMEOUT_SEC", "35")))
        except Exception:
            lock_timeout = 35.0
        last_error: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self._lock:
                    with interprocess_file_lock(self.db_path, timeout=lock_timeout):
                        conn = self._connect()
                        try:
                            conn.execute("BEGIN IMMEDIATE;")
                            result = operation(conn)
                            conn.commit()
                            return result
                        except Exception:
                            try:
                                conn.rollback()
                            except Exception:
                                pass
                            raise
                        finally:
                            conn.close()
            except (sqlite3.OperationalError, TimeoutError) as exc:
                last_error = exc
                retryable = isinstance(exc, TimeoutError) or self._is_sqlite_busy_error(exc)
                if not retryable or attempt >= attempts - 1:
                    raise
                time.sleep(base_delay * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("SQLite write transaction did not run")

    def _quick_check_interval(self) -> int:
        try:
            return max(0, int(os.getenv("MEM_SQLITE_QUICK_CHECK_EVERY_CONNECTIONS", "250")))
        except Exception:
            return 250

    def _maybe_quick_check_connection(self) -> None:
        interval = self._quick_check_interval()
        if interval <= 0:
            return
        with self._health_lock:
            self._connection_count += 1
            should_check = self._connection_count % interval == 0
        if not should_check or not os.path.isfile(self.db_path):
            return
        checked_at = _utc_now_iso()
        try:
            from tsuzuri.runtime_backup import sqlite_quick_check

            rows = sqlite_quick_check(self.db_path, timeout=2.0)
            self._last_quick_check_report = {"checked_at": checked_at, "rows": rows, "ok": rows == ["ok"]}
            if rows != ["ok"]:
                _tsz_emit(f"[MemoryCore][Health] quick_check failed: {rows}", flush=True)
        except Exception as exc:
            self._last_quick_check_report = {"checked_at": checked_at, "rows": [], "ok": False, "error": str(exc)}
            _tsz_emit(f"[MemoryCore][Health] quick_check unavailable: {exc}", flush=True)

    def _backup_before_schema_migration_if_needed(self) -> str:
        if not os.path.isfile(self.db_path):
            return ""
        required_columns = {
            "memory_items": {
                "parent_id",
                "cluster_memory_parent_id",
                "member_ids_json",
                "cluster_meta_json",
                "embedding_text",
                "embedding_sparse_text",
                "embedding_model_id",
                "embedding_dim",
                "topic_labels_json",
                "memory_role",
                "engram_type",
                "cue_json",
                "lifecycle_json",
            },
            "memory_rethink_runs": {"evidence_json", "finish_confirmed", "protocol_version"},
            "memory_rethink_actions": {
                "resolution_type",
                "evidence_item_ids_json",
                "expected_before_hash",
                "evidence_before_hash",
                "loop_pass",
                "apply_status",
                "apply_error",
            },
        }
        missing: dict[str, list[str]] = {}
        try:
            uri = f"file:{os.path.abspath(self.db_path).replace(os.sep, '/')}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                tables = {
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                if "memory_items" not in tables:
                    return ""
                for table, expected in required_columns.items():
                    if table not in tables:
                        continue
                    actual = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                    absent = sorted(expected - actual)
                    if absent:
                        missing[table] = absent
            finally:
                conn.close()
        except Exception as exc:
            _tsz_emit(f"[MemoryCore][Schema] preflight unavailable, schema migration not started: {exc}", flush=True)
            raise
        if not missing:
            return ""
        from tsuzuri.runtime_backup import backup_sqlite, prune_rotating_files

        timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        destination = f"{self.db_path}.pre_migrate_{timestamp}.bak"
        backup_sqlite(self.db_path, destination)
        prune_rotating_files(
            os.path.dirname(os.path.abspath(self.db_path)) or ".",
            f"{os.path.basename(self.db_path)}.pre_migrate_*.bak",
            keep=2,
        )
        _tsz_emit(f"[MemoryCore][Schema] backup={destination} missing={missing}", flush=True)
        return destination

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_items (
                        item_id TEXT PRIMARY KEY,
                        character TEXT NOT NULL,
                        layer TEXT NOT NULL,
                        memory_role TEXT NOT NULL DEFAULT 'episodic',
                        engram_type TEXT NOT NULL DEFAULT 'note',
                        item_type TEXT NOT NULL,
                        text TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        cue_json TEXT NOT NULL DEFAULT '{}',
                        lifecycle_json TEXT NOT NULL DEFAULT '{}',
                        salience REAL NOT NULL DEFAULT 0.5,
                        confidence REAL NOT NULL DEFAULT 0.5,
                        novelty REAL NOT NULL DEFAULT 0.5,
                        importance REAL NOT NULL DEFAULT 0.5,
                        status TEXT NOT NULL DEFAULT 'active',
                        source TEXT NOT NULL DEFAULT '',
                        evidence_id TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL DEFAULT '',
                        access_count INTEGER NOT NULL DEFAULT 0,
                        archived INTEGER NOT NULL DEFAULT 0
                    );
                    """
                )
                # Phase Q.1.A (2026-05-03): graph hierarchy for cluster + parent summary
                # 借鉴 MemOS reorganizer.py、给 memory_items 加 3 列实现 parent-child 层级
                # ALTER ADD COLUMN 兼容老 db、try/except 防御已建过的字段
                _phase_q_columns = [
                    ("parent_id", "TEXT NOT NULL DEFAULT ''"),
                    # The narrative cluster pipeline owns a separate membership edge.
                    # ``parent_id`` remains the legacy embedding-summary edge.
                    ("cluster_memory_parent_id", "TEXT NOT NULL DEFAULT ''"),
                    ("member_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
                    ("cluster_meta_json", "TEXT NOT NULL DEFAULT '{}'"),
                ]
                for _col_name, _col_def in _phase_q_columns:
                    try:
                        conn.execute(f"ALTER TABLE memory_items ADD COLUMN {_col_name} {_col_def};")
                    except Exception:
                        pass  # 已存在
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_parent_id ON memory_items(parent_id);")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_items_cluster_memory_parent_id "
                    "ON memory_items(cluster_memory_parent_id);"
                )
                # Earlier narrative clusters reused the legacy parent_id column. Move
                # only those edges to their dedicated column so both pipelines can
                # evaluate the same episodic evidence independently.
                conn.execute(
                    """
                    UPDATE memory_items AS member
                    SET cluster_memory_parent_id = member.parent_id,
                        parent_id = ''
                    WHERE member.parent_id != ''
                      AND member.cluster_memory_parent_id = ''
                      AND EXISTS (
                          SELECT 1
                          FROM memory_items AS parent
                          WHERE parent.item_id = member.parent_id
                            AND parent.source = 'cluster_memory'
                            AND parent.item_type IN ('event_cluster', 'topic_cluster')
                      )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_char_layer ON memory_items(character, layer);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_updated ON memory_items(updated_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_archived ON memory_items(archived, status);")
                # Optimized composite indexes for retrieval queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_retrieval ON memory_items(status, archived, character, updated_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_importance ON memory_items(character, importance DESC) WHERE archived=0 AND status='active';")
                # Embedding column for semantic recall
                try:
                    conn.execute("ALTER TABLE memory_items ADD COLUMN embedding_text TEXT NOT NULL DEFAULT '';")
                except Exception:
                    pass  # Column already exists
                # Phase A (2026-05-14): BGE-M3 sparse embedding column
                # 存 JSON 序列化的 dict[token_id, weight]、给 BM25-like 召回用
                try:
                    conn.execute("ALTER TABLE memory_items ADD COLUMN embedding_sparse_text TEXT NOT NULL DEFAULT '';")
                except Exception:
                    pass  # Column already exists
                for _column, _ddl in (
                    ("embedding_model_id", "TEXT NOT NULL DEFAULT ''"),
                    ("embedding_dim", "INTEGER NOT NULL DEFAULT 0"),
                ):
                    try:
                        conn.execute(f"ALTER TABLE memory_items ADD COLUMN {_column} {_ddl};")
                    except Exception:
                        pass
                # topic_labels_json: fine-grained topic label list from TurnFactCard LLM extraction
                try:
                    conn.execute("ALTER TABLE memory_items ADD COLUMN topic_labels_json TEXT NOT NULL DEFAULT '[]';")
                except Exception:
                    pass  # Column already exists
                try:
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_topic_labels ON memory_items(character, topic_labels_json) WHERE archived=0;")
                except Exception:
                    pass
                # FTS5 full-text search for fast pre-filtering
                try:
                    conn.execute("""
                        CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
                            item_id, text, content=memory_items, content_rowid=rowid,
                            tokenize='unicode61'
                        );
                    """)
                except Exception:
                    pass  # FTS5 may not be available in all SQLite builds


                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_links (
                        link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        src_item_id TEXT NOT NULL,
                        dst_item_id TEXT NOT NULL,
                        link_type TEXT NOT NULL,
                        weight REAL NOT NULL DEFAULT 0.5,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        UNIQUE(src_item_id, dst_item_id, link_type)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_src ON memory_links(src_item_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_dst ON memory_links(dst_item_id);")

                # 2026-05-09 fix: Hebbian co-occurrence persistence
                # 替代旧 in-memory _CO_STRENGTH_CACHE（重启清空、长期 Hebbian 学习失效）
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS co_occurrence_strength (
                        item_a TEXT NOT NULL,
                        item_b TEXT NOT NULL,
                        strength REAL NOT NULL DEFAULT 0.10,
                        last_updated TEXT NOT NULL,
                        update_count INTEGER NOT NULL DEFAULT 1,
                        PRIMARY KEY (item_a, item_b)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_co_strength_a ON co_occurrence_strength(item_a);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_co_strength_b ON co_occurrence_strength(item_b);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_co_strength_updated ON co_occurrence_strength(last_updated DESC);")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS retrieval_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        character TEXT NOT NULL,
                        query_text TEXT NOT NULL,
                        query_bundle_json TEXT NOT NULL,
                        candidate_item_id TEXT NOT NULL,
                        candidate_text TEXT NOT NULL,
                        source TEXT NOT NULL,
                        score REAL NOT NULL DEFAULT 0.0,
                        relevance REAL NOT NULL DEFAULT 0.0,
                        recency REAL NOT NULL DEFAULT 0.0,
                        importance REAL NOT NULL DEFAULT 0.0,
                        llm_score REAL NOT NULL DEFAULT 0.0,
                        selected INTEGER NOT NULL DEFAULT 0,
                        evidence_id TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_logs_run ON retrieval_logs(run_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_retrieval_logs_created ON retrieval_logs(created_at DESC);")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evolution_runs (
                        run_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL UNIQUE,
                        character TEXT NOT NULL,
                        shadow INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL,
                        proposal_json TEXT NOT NULL,
                        evaluation_json TEXT NOT NULL,
                        applied_actions_json TEXT NOT NULL DEFAULT '[]',
                        base_snapshot_hash TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evolution_runs_created ON evolution_runs(created_at DESC);")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS policy_profiles (
                        profile_id TEXT PRIMARY KEY,
                        profile_name TEXT NOT NULL,
                        active INTEGER NOT NULL DEFAULT 0,
                        config_json TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS meta_state (
                        state_key TEXT PRIMARY KEY,
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_domains (
                        domain_key TEXT PRIMARY KEY,
                        domain_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mechanism_state (
                        mechanism_key TEXT PRIMARY KEY,
                        phase TEXT NOT NULL DEFAULT 'phase1',
                        value_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mechanism_entries (
                        entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mechanism_key TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT 'phase1',
                        owner_key TEXT NOT NULL DEFAULT '',
                        entry_key TEXT NOT NULL,
                        content_text TEXT NOT NULL DEFAULT '',
                        payload_json TEXT NOT NULL,
                        ts TEXT NOT NULL DEFAULT '',
                        weight REAL NOT NULL DEFAULT 0.0,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        updated_at TEXT NOT NULL,
                        version INTEGER NOT NULL DEFAULT 1,
                        UNIQUE(mechanism_key, owner_key, entry_key)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mechanism_entries_mech ON mechanism_entries(mechanism_key, owner_key);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mechanism_entries_ts ON mechanism_entries(mechanism_key, ts DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_mechanism_entries_weight ON mechanism_entries(mechanism_key, weight DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS policy_decision_logs (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        decision_kind TEXT NOT NULL,
                        phase TEXT NOT NULL DEFAULT '',
                        character TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL DEFAULT '',
                        input_json TEXT NOT NULL DEFAULT '{}',
                        llm_json TEXT NOT NULL DEFAULT '{}',
                        fallback_reason TEXT NOT NULL DEFAULT '',
                        final_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_decision_kind ON policy_decision_logs(decision_kind, created_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_context_diff_logs (
                        diff_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        character TEXT NOT NULL DEFAULT '',
                        retrieval_run_id TEXT NOT NULL DEFAULT '',
                        legacy_json TEXT NOT NULL DEFAULT '[]',
                        context_json TEXT NOT NULL DEFAULT '[]',
                        diff_json TEXT NOT NULL DEFAULT '{}',
                        severity TEXT NOT NULL DEFAULT 'info',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_context_diff_run ON memory_context_diff_logs(character, retrieval_run_id, created_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evolution_action_log (
                        action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        action_index INTEGER NOT NULL DEFAULT 0,
                        op TEXT NOT NULL,
                        item_id TEXT NOT NULL,
                        before_json TEXT NOT NULL,
                        after_json TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'applied',
                        created_at TEXT NOT NULL,
                        reverted_at TEXT NOT NULL DEFAULT ''
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_evolution_action_run ON evolution_action_log(run_id, action_index, action_id);")

                # ─────────────────────────────────────────────────────────────
                # Phase R.2 (2026-07-25) — evidence-bound memory rethink
                # ─────────────────────────────────────────────────────────────
                # 借鉴 Letta sleeptime agent + memory_rethink 设计，让 LLM 在
                # KAIROS 凌晨 dream cycle 中审视 semantic 层记忆并提出 rewrite /
                # merge / archive 三类编辑建议。规则启发式（propose_memory_evolution）
                # 是被动 promote/archive，无法改 text；LLM rethink 补的就是"语义
                # 化纠错 + 同主题合并"的能力。
                #
                # 双表结构（与 evolution_runs / evolution_action_log 同构）：
                #   memory_rethink_runs    — 一次 LLM 调用一行，存 candidates + raw response + summary
                #   memory_rethink_actions — 每个 action 一行，含 before/after 快照 + OOC 二审分数
                #
                # terminal status 包括 completed_shadow / completed_applied /
                # completed_no_change / failed_atomic / failed_validation /
                # failed_incomplete / revert_conflict / reverted。
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_rethink_runs (
                        run_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL UNIQUE,
                        character TEXT NOT NULL,
                        shadow INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL,
                        candidates_count INTEGER NOT NULL DEFAULT 0,
                        candidates_json TEXT NOT NULL DEFAULT '[]',
                        actions_proposed_count INTEGER NOT NULL DEFAULT 0,
                        actions_applied_count INTEGER NOT NULL DEFAULT 0,
                        actions_rejected_count INTEGER NOT NULL DEFAULT 0,
                        llm_model TEXT NOT NULL DEFAULT '',
                        llm_total_tokens INTEGER NOT NULL DEFAULT 0,
                        llm_raw_response TEXT NOT NULL DEFAULT '',
                        summary TEXT NOT NULL DEFAULT '',
                        fallback_reason TEXT NOT NULL DEFAULT '',
                        ooc_threshold REAL NOT NULL DEFAULT 0.45,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rethink_runs_char_created ON memory_rethink_runs(character, created_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rethink_runs_status ON memory_rethink_runs(status, created_at DESC);")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_rethink_actions (
                        action_pk INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        action_index INTEGER NOT NULL DEFAULT 0,
                        action_type TEXT NOT NULL,
                        target_item_ids_json TEXT NOT NULL DEFAULT '[]',
                        old_texts_json TEXT NOT NULL DEFAULT '[]',
                        new_text TEXT NOT NULL DEFAULT '',
                        reason TEXT NOT NULL DEFAULT '',
                        ooc_score REAL NOT NULL DEFAULT 0.0,
                        ooc_verdict TEXT NOT NULL DEFAULT '',
                        ooc_features_json TEXT NOT NULL DEFAULT '[]',
                        validation_status TEXT NOT NULL DEFAULT 'pending',
                        validation_reason TEXT NOT NULL DEFAULT '',
                        applied INTEGER NOT NULL DEFAULT 0,
                        applied_at TEXT NOT NULL DEFAULT '',
                        reverted_at TEXT NOT NULL DEFAULT '',
                        revert_reason TEXT NOT NULL DEFAULT '',
                        before_json TEXT NOT NULL DEFAULT '{}',
                        after_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rethink_actions_run ON memory_rethink_actions(run_id, action_index, action_pk);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rethink_actions_type ON memory_rethink_actions(action_type, applied);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_rethink_actions_validation ON memory_rethink_actions(validation_status, ooc_verdict);")

                # Phase R.2: evidence-bound edit protocol. Guarded ALTER keeps old
                # runtime DBs compatible without assuming a fresh schema.
                _rethink_run_columns = {
                    str(row[1]) for row in conn.execute("PRAGMA table_info(memory_rethink_runs);").fetchall()
                }
                for _column, _ddl in {
                    "evidence_json": "TEXT NOT NULL DEFAULT '[]'",
                    "finish_confirmed": "INTEGER NOT NULL DEFAULT 0",
                    "protocol_version": "TEXT NOT NULL DEFAULT 'rethink_v2'",
                }.items():
                    if _column not in _rethink_run_columns:
                        conn.execute(f"ALTER TABLE memory_rethink_runs ADD COLUMN {_column} {_ddl};")
                _rethink_action_columns = {
                    str(row[1]) for row in conn.execute("PRAGMA table_info(memory_rethink_actions);").fetchall()
                }
                for _column, _ddl in {
                    "resolution_type": "TEXT NOT NULL DEFAULT ''",
                    "evidence_item_ids_json": "TEXT NOT NULL DEFAULT '[]'",
                    "expected_before_hash": "TEXT NOT NULL DEFAULT ''",
                    "evidence_before_hash": "TEXT NOT NULL DEFAULT ''",
                    "loop_pass": "INTEGER NOT NULL DEFAULT 0",
                    "apply_status": "TEXT NOT NULL DEFAULT 'pending'",
                    "apply_error": "TEXT NOT NULL DEFAULT ''",
                }.items():
                    if _column not in _rethink_action_columns:
                        conn.execute(f"ALTER TABLE memory_rethink_actions ADD COLUMN {_column} {_ddl};")

                # Phase Q.W2 / Option E Layer 3: reuse signal tracking
                # 跟踪 working turn 在后续对话中被实际复用的信号
                # signal_type:
                #   - confirmed_reuse: user 在 N+1 turn 提到了 working_N 的 entity
                #   - direct_recall_hit: user 显式问"还记得 X 吗" + 命中
                #   - assistant_reference_accepted: 角色引用 + user 接续未否认
                #   - assistant_reference_rejected: 角色引用 + user 否认（contra signal）
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reuse_signal_log (
                        signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        working_item_id TEXT NOT NULL,
                        character TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        triggering_turn_idx INTEGER NOT NULL DEFAULT 0,
                        evidence TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reuse_signal_item ON reuse_signal_log(working_item_id, signal_type, created_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reuse_signal_char ON reuse_signal_log(character, created_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS canon_sources (
                        source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        character TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT '',
                        snippet TEXT NOT NULL DEFAULT '',
                        fetched_at TEXT NOT NULL,
                        freshness REAL NOT NULL DEFAULT 0.0,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        weight REAL NOT NULL DEFAULT 0.0,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(character, source_type, source_url)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_canon_sources_char ON canon_sources(character, fetched_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS canon_claims (
                        claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        character TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_url TEXT NOT NULL,
                        claim_type TEXT NOT NULL,
                        claim_text TEXT NOT NULL,
                        confidence REAL NOT NULL DEFAULT 0.0,
                        freshness REAL NOT NULL DEFAULT 0.0,
                        weight REAL NOT NULL DEFAULT 0.0,
                        status TEXT NOT NULL DEFAULT 'active',
                        extracted_at TEXT NOT NULL,
                        meta_json TEXT NOT NULL DEFAULT '{}',
                        UNIQUE(character, source_url, claim_type, claim_text)
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_canon_claims_char ON canon_claims(character, claim_type, extracted_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS canon_calibration_runs (
                        run_id TEXT PRIMARY KEY,
                        character TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'planned',
                        source_summary_json TEXT NOT NULL DEFAULT '{}',
                        claims_json TEXT NOT NULL DEFAULT '[]',
                        profile_json TEXT NOT NULL DEFAULT '{}',
                        shadow INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_canon_calibration_runs_char ON canon_calibration_runs(character, created_at DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ooc_audit_runs (
                        audit_id TEXT PRIMARY KEY,
                        character TEXT NOT NULL,
                        run_id TEXT NOT NULL DEFAULT '',
                        ooc_score REAL NOT NULL DEFAULT 0.0,
                        style_deviation_json TEXT NOT NULL DEFAULT '{}',
                        evidence_hits_json TEXT NOT NULL DEFAULT '[]',
                        high_conflict_claims_json TEXT NOT NULL DEFAULT '[]',
                        penalty_json TEXT NOT NULL DEFAULT '{}',
                        applied INTEGER NOT NULL DEFAULT 0,
                        shadow INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ooc_audit_runs_char ON ooc_audit_runs(character, created_at DESC);")
                # ── Motivation layer: persistent character goals ──
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS character_motivations (
                        motivation_id TEXT PRIMARY KEY,
                        character TEXT NOT NULL,
                        goal_text TEXT NOT NULL,
                        origin TEXT NOT NULL DEFAULT 'reflection',
                        priority REAL NOT NULL DEFAULT 0.5,
                        urgency REAL NOT NULL DEFAULT 0.3,
                        created_at TEXT NOT NULL,
                        last_reinforced_at TEXT,
                        reinforcement_count INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'active',
                        decay_rate REAL NOT NULL DEFAULT 0.02,
                        related_memory_ids TEXT NOT NULL DEFAULT '[]',
                        meta_json TEXT NOT NULL DEFAULT '{}'
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_motivations_char ON character_motivations(character, status, priority DESC);")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_appointments (
                        appointment_id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL DEFAULT 0,
                        character TEXT NOT NULL,
                        text TEXT NOT NULL,
                        due_date TEXT NOT NULL DEFAULT '',
                        due_period TEXT NOT NULL DEFAULT 'someday',
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        reminded_at TEXT
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_appointments_due ON user_appointments(due_date, status);")
                try:
                    existing_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(memory_items);").fetchall()}
                except Exception:
                    existing_cols = set()
                alter_specs = [
                    ("memory_role", "TEXT NOT NULL DEFAULT 'episodic'"),
                    ("engram_type", "TEXT NOT NULL DEFAULT 'note'"),
                    ("cue_json", "TEXT NOT NULL DEFAULT '{}'"),
                    ("lifecycle_json", "TEXT NOT NULL DEFAULT '{}'"),
                ]
                for col, ddl in alter_specs:
                    if col not in existing_cols:
                        conn.execute(f"ALTER TABLE memory_items ADD COLUMN {col} {ddl};")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_items_role ON memory_items(character, memory_role, archived, status);")

                now = _utc_now_iso()
                row = conn.execute("SELECT profile_id, profile_name, config_json, version, created_at FROM policy_profiles WHERE active=1 LIMIT 1;").fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO policy_profiles
                        (profile_id, profile_name, active, config_json, version, created_at, updated_at)
                        VALUES (?, ?, 1, ?, 2, ?, ?);
                        """,
                        ("default_v2", "default", _json_dumps(_DEFAULT_POLICY_PROFILE_V2), now, now),
                    )
                else:
                    cfg_now = _json_loads(str(row["config_json"] or "{}"), {})
                    cfg_v2 = _normalize_policy_profile(cfg_now if isinstance(cfg_now, dict) else {})
                    if _json_dumps(cfg_v2) != _json_dumps(cfg_now if isinstance(cfg_now, dict) else {}):
                        conn.execute(
                            """
                            UPDATE policy_profiles
                            SET config_json=?, version=MAX(version, 2), updated_at=?
                            WHERE profile_id=?;
                            """,
                            (_json_dumps(cfg_v2), now, str(row["profile_id"])),
                        )

                conn.commit()
            finally:
                conn.close()

    def _upsert_domain(self, domain_key: str, value) -> None:
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO memory_domains(domain_key, domain_json, updated_at, version)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(domain_key) DO UPDATE SET
                        domain_json=excluded.domain_json,
                        updated_at=excluded.updated_at;
                    """,
                    (str(domain_key or "").strip(), _json_dumps(value), now),
                )
                conn.commit()
            finally:
                conn.close()

    def _load_domain_map(self) -> dict[str, object]:
        out: dict[str, object] = {}
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT domain_key, domain_json FROM memory_domains;").fetchall()
            except Exception:
                rows = []
            finally:
                conn.close()
        for r in rows:
            k = str(r["domain_key"] or "").strip()
            if not k:
                continue
            out[k] = _json_loads(str(r["domain_json"] or ""), None)
        return out

    def _sync_domains_from_snapshot(self, snapshot: dict) -> None:
        if not isinstance(snapshot, dict):
            return
        for path in _MIGRATION_DOMAIN_PATHS:
            v = _path_get(snapshot, path)
            if v is None:
                continue
            self._upsert_domain(path, v)
        self._set_meta_json("domain_sync_at", {"at": _utc_now_iso(), "paths": len(_MIGRATION_DOMAIN_PATHS)})

    def _overlay_domains_into_snapshot(self, snapshot: dict) -> dict:
        base = snapshot if isinstance(snapshot, dict) else {}
        merged = _json_clone(base) if isinstance(base, dict) else {}
        if not isinstance(merged, dict):
            merged = {}
        dmap = self._load_domain_map()
        if not isinstance(dmap, dict) or (not dmap):
            return merged
        for path, v in dmap.items():
            if v is None:
                continue
            _path_set(merged, path, v)
        return merged

    def _mechanism_phase(self, mechanism_key: str) -> str:
        k = str(mechanism_key or "").strip()
        return str(_MECHANISM_PHASES.get(k, "phase1"))

    def _extract_mechanism_entries(self, mechanism_key: str, value) -> list[dict]:
        key = str(mechanism_key or "").strip()
        if not key:
            return []

        rows: list[dict] = []

        def _push(owner_key: str, entry_seed: str, payload, seq_idx: int) -> None:
            payload_obj = _json_clone(payload)
            seed_base = str(entry_seed or "").strip() or str(seq_idx)
            ek = f"{seed_base}_{_stable_hash(payload_obj)}"
            rows.append(
                {
                    "mechanism_key": key,
                    "phase": self._mechanism_phase(key),
                    "owner_key": str(owner_key or "").strip(),
                    "entry_key": ek[:96],
                    "content_text": _derive_text(payload_obj),
                    "payload_json": _json_dumps(payload_obj),
                    "ts": _derive_ts(payload_obj),
                    "weight": _safe_float(_derive_weight(payload_obj), 0.0),
                    "confidence": _safe_float(_derive_confidence(payload_obj), 0.0),
                    "updated_at": _utc_now_iso(),
                    "version": 1,
                }
            )

        if isinstance(value, list):
            for i, item in enumerate(value):
                seed = ""
                if isinstance(item, dict):
                    seed = str(item.get("id", "") or item.get("fp", "") or item.get("date", "") or item.get("at", "") or item.get("timestamp", "") or "").strip()
                _push("", seed or str(i), item, i)
            return rows

        if isinstance(value, dict):
            split_owner_mechanisms = {
                "gossip_queue",
                "_topic_fingerprints",
                "_micro_reflections",
                "_town_projection",
                "_town_mailbox_seen",
                "long_term.character_relationships",
            }
            if key in split_owner_mechanisms:
                seq = 0
                for owner, owner_val in value.items():
                    owner_key = str(owner or "").strip()
                    if isinstance(owner_val, list):
                        for i, item in enumerate(owner_val):
                            seed = ""
                            if isinstance(item, dict):
                                seed = str(item.get("id", "") or item.get("fp", "") or item.get("at", "") or item.get("timestamp", "") or item.get("date", "") or "").strip()
                            _push(owner_key, seed or str(i), item, seq)
                            seq += 1
                    else:
                        _push(owner_key, owner_key or str(seq), owner_val, seq)
                        seq += 1
                return rows

            seq = 0
            for sub_key, sub_val in value.items():
                sk = str(sub_key or "").strip()
                if isinstance(sub_val, list):
                    for i, item in enumerate(sub_val):
                        seed = ""
                        if isinstance(item, dict):
                            seed = str(item.get("id", "") or item.get("fp", "") or item.get("date", "") or item.get("at", "") or "").strip()
                        _push(sk, seed or str(i), item, seq)
                        seq += 1
                else:
                    _push("", sk or str(seq), {"key": sk, "value": sub_val}, seq)
                    seq += 1
            return rows

        _push("", "value", value, 0)
        return rows

    def _sync_mechanisms_from_snapshot(self, snapshot: dict) -> None:
        if not isinstance(snapshot, dict):
            return
        now = _utc_now_iso()
        stats: dict[str, int] = {}
        with self._lock:
            conn = self._connect()
            try:
                for path in _NATIVE_MECHANISM_PATHS:
                    v = _path_get(snapshot, path)
                    if v is None:
                        continue
                    phase = self._mechanism_phase(path)
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
                        (path, phase, _json_dumps(v), now),
                    )
                    conn.execute("DELETE FROM mechanism_entries WHERE mechanism_key=?;", (path,))
                    rows = self._extract_mechanism_entries(path, v)
                    for r in rows:
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
                    stats[path] = len(rows)
                conn.commit()
            finally:
                conn.close()
        self._set_meta_json(
            "mechanism_sync_at",
            {"at": now, "paths": len(stats), "entry_counts": stats},
        )

    def _load_mechanism_state_map(self) -> dict[str, object]:
        out: dict[str, object] = {}
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute("SELECT mechanism_key, value_json FROM mechanism_state;").fetchall()
            except Exception:
                rows = []
            finally:
                conn.close()
        for r in rows:
            k = str(r["mechanism_key"] or "").strip()
            if not k:
                continue
            out[k] = _json_loads(str(r["value_json"] or ""), None)
        return out

    def _overlay_mechanisms_into_snapshot(self, snapshot: dict) -> dict:
        base = snapshot if isinstance(snapshot, dict) else {}
        merged = _json_clone(base) if isinstance(base, dict) else {}
        if not isinstance(merged, dict):
            merged = {}
        mmap = self._load_mechanism_state_map()
        if not isinstance(mmap, dict) or (not mmap):
            return merged
        for path, v in mmap.items():
            if v is None:
                continue
            _path_set(merged, path, v)
        return merged

    def _count_native_mechanisms(self) -> int:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT COUNT(*) AS c FROM mechanism_state;").fetchone()
                return int(row["c"]) if row is not None else 0
            except Exception:
                return 0
            finally:
                conn.close()

    def _backfill_native_mechanisms_if_needed(self) -> None:
        meta = self._get_meta_json("native_mechanism_backfill_v1", {})
        done = _parse_bool(meta.get("done", False), False) if isinstance(meta, dict) else False
        current_count = self._count_native_mechanisms()
        if done and current_count >= len(_NATIVE_MECHANISM_PATHS):
            return
        snap = self._get_meta_json("latest_snapshot", {})
        if not isinstance(snap, dict):
            snap = {}
        seed = self._overlay_domains_into_snapshot(snap)
        self._sync_mechanisms_from_snapshot(seed if isinstance(seed, dict) else {})
        self._set_meta_json(
            "native_mechanism_backfill_v1",
            {
                "done": True,
                "at": _utc_now_iso(),
                "native_mechanism_count": self._count_native_mechanisms(),
            },
        )

    def _get_meta_json(self, key: str, default):
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT value_json FROM meta_state WHERE state_key=?;", (str(key),)).fetchone()
            finally:
                conn.close()
        if row is None:
            return default
        return _json_loads(str(row["value_json"] or ""), default)

    def _set_meta_json(self, key: str, value) -> None:
        now = _utc_now_iso()
        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """
                INSERT INTO meta_state(state_key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at;
                """,
                (str(key), _json_dumps(value), now),
            )

        self._run_write_transaction(_write)

    def get_meta_state_json(self, key: str, default=None):
        return self._get_meta_json(str(key or "").strip(), default)

    def set_meta_state_json(self, key: str, value) -> None:
        self._set_meta_json(str(key or "").strip(), value)

    def retrieval_log_retention_days(self) -> int:
        try:
            return max(1, int(os.getenv("MEMORY_RETRIEVAL_LOG_RETENTION_DAYS", "7")))
        except Exception:
            return 7

    def _retrieval_log_prune_interval_hours(self) -> float:
        try:
            return max(0.0, float(os.getenv("MEMORY_RETRIEVAL_LOG_PRUNE_INTERVAL_HOURS", "24")))
        except Exception:
            return 24.0

    def prune_retrieval_logs(self, *, max_age_days: int | None = None, vacuum: bool = False) -> int:
        """删除超出保留窗口的召回诊断日志，返回删除行数。"""
        days = int(max_age_days) if max_age_days is not None else self.retrieval_log_retention_days()
        days = max(1, days)
        cutoff = (utc_now() - timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        deleted = 0

        def _delete(conn: sqlite3.Connection) -> None:
            nonlocal deleted
            cursor = conn.execute("DELETE FROM retrieval_logs WHERE created_at < ?;", (cutoff,))
            deleted = int(cursor.rowcount or 0)

        self._run_write_transaction(_delete)
        if vacuum and deleted:
            with self._lock:
                with interprocess_file_lock(self.db_path, timeout=35.0):
                    conn = self._connect()
                    try:
                        conn.execute("VACUUM;")
                    finally:
                        conn.close()
        self._set_meta_json(
            self.RETRIEVAL_LOG_PRUNE_STATE_KEY,
            {"at": _utc_now_iso(), "deleted": deleted, "retention_days": days},
        )
        return deleted

    def _maybe_prune_retrieval_logs(self) -> None:
        interval_hours = self._retrieval_log_prune_interval_hours()
        if interval_hours <= 0:
            return
        meta = self._get_meta_json(self.RETRIEVAL_LOG_PRUNE_STATE_KEY, {})
        last_at = str(meta.get("at", "") or "") if isinstance(meta, dict) else ""
        if last_at:
            try:
                last_dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
            except Exception:
                last_dt = None
            if last_dt is not None and (utc_now() - last_dt) < timedelta(hours=interval_hours):
                return
        try:
            deleted = self.prune_retrieval_logs()
        except Exception as exc:
            # 卫生清理失败不能阻塞启动。
            _tsz_emit(f"[MemoryCore] retrieval_logs prune skipped: {exc}", flush=True)
            return
        if deleted:
            _tsz_emit(
                f"[MemoryCore] pruned {deleted} retrieval_logs rows older than "
                f"{self.retrieval_log_retention_days()}d",
                flush=True,
            )

    @staticmethod
    def _ensure_shared_info_schema_on_conn(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_info (
                info_id TEXT PRIMARY KEY,
                source_character TEXT NOT NULL,
                original_text TEXT NOT NULL,
                paraphrased_text TEXT DEFAULT '',
                privacy_level TEXT DEFAULT 'public',
                created_at REAL NOT NULL,
                delivered_to_json TEXT DEFAULT '{}',
                mentions_count INTEGER DEFAULT 0,
                max_mentions INTEGER DEFAULT 2,
                expired INTEGER DEFAULT 0,
                confidence REAL DEFAULT 1.0,
                hop INTEGER DEFAULT 1,
                eligible_targets_json TEXT DEFAULT '',
                town_msg_id TEXT DEFAULT ''
            );
            """
        )
        # 2026-08-14 gossip 链路统一：存量库补列（幂等）。
        # eligible_targets_json 空串 = 全员可见（兼容统一前 chain A 写入的旧行）。
        existing = {str(row[1]) for row in conn.execute("PRAGMA table_info(shared_info);").fetchall()}
        for _col, _ddl in (
            ("confidence", "ALTER TABLE shared_info ADD COLUMN confidence REAL DEFAULT 1.0;"),
            ("hop", "ALTER TABLE shared_info ADD COLUMN hop INTEGER DEFAULT 1;"),
            ("eligible_targets_json", "ALTER TABLE shared_info ADD COLUMN eligible_targets_json TEXT DEFAULT '';"),
            ("town_msg_id", "ALTER TABLE shared_info ADD COLUMN town_msg_id TEXT DEFAULT '';"),
        ):
            if _col not in existing:
                conn.execute(_ddl)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_shared_info_hop_created_at "
            "ON shared_info(hop, created_at DESC);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_shared_info_created_at "
            "ON shared_info(created_at DESC);"
        )

    def ensure_shared_info_schema(self) -> None:
        self._run_write_transaction(self._ensure_shared_info_schema_on_conn)

    def store_shared_info(
        self,
        *,
        info_id: str,
        source_character: str,
        original_text: str,
        privacy_level: str = "public",
        created_at: float | None = None,
        confidence: float = 1.0,
        hop: int = 1,
        eligible_targets: list | tuple | None = None,
        town_msg_id: str = "",
    ) -> bool:
        created = time.time() if created_at is None else float(created_at)
        eligible_json = ""
        if eligible_targets:
            eligible_json = _json_dumps([str(t) for t in eligible_targets if str(t or "").strip()])

        def _write(conn: sqlite3.Connection) -> bool:
            self._ensure_shared_info_schema_on_conn(conn)
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO shared_info
                (info_id, source_character, original_text, privacy_level, created_at,
                 confidence, hop, eligible_targets_json, town_msg_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(info_id),
                    str(source_character),
                    str(original_text),
                    str(privacy_level or "public"),
                    created,
                    float(confidence),
                    int(hop),
                    eligible_json,
                    str(town_msg_id or ""),
                ),
            )
            return cursor.rowcount > 0

        return bool(self._run_write_transaction(_write))

    def has_town_mailbox_message(self, town_msg_id: str, character: str) -> bool:
        """邮箱投影去重：同一 (town_msg_id, 目标角色) 只投影一次。"""
        mid = str(town_msg_id or "").strip()
        char = str(character or "").strip()
        if not mid or not char:
            return False

        def _read(conn: sqlite3.Connection) -> bool:
            self._ensure_shared_info_schema_on_conn(conn)
            rows = conn.execute(
                "SELECT eligible_targets_json FROM shared_info WHERE town_msg_id=?;",
                (mid,),
            ).fetchall()
            for row in rows:
                eligible = _json_loads(str(row[0] or ""), [])
                if isinstance(eligible, list) and char in eligible:
                    return True
            return False

        return bool(self._run_write_transaction(_read))

    def mark_shared_info_delivered(
        self,
        info_id: str,
        character: str,
        *,
        delivered_at: float | None = None,
    ) -> bool:
        delivered_ts = time.time() if delivered_at is None else float(delivered_at)

        def _write(conn: sqlite3.Connection) -> bool:
            self._ensure_shared_info_schema_on_conn(conn)
            row = conn.execute(
                "SELECT delivered_to_json, mentions_count, max_mentions FROM shared_info WHERE info_id=?;",
                (str(info_id),),
            ).fetchone()
            if row is None:
                return False
            delivered = _json_loads(str(row["delivered_to_json"] or "{}"), {})
            if not isinstance(delivered, dict):
                delivered = {}
            delivered[str(character)] = delivered_ts
            mentions = int(_safe_int(row["mentions_count"], 0)) + 1
            max_mentions = max(1, int(_safe_int(row["max_mentions"], 2)))
            conn.execute(
                """
                UPDATE shared_info
                SET delivered_to_json=?, mentions_count=?, expired=?
                WHERE info_id=?;
                """,
                (_json_dumps(delivered), mentions, int(mentions >= max_mentions), str(info_id)),
            )
            return True

        return bool(self._run_write_transaction(_write))

    def shared_info_stale_hours(self) -> int:
        try:
            return max(48, int(os.getenv("MEMORY_SHARED_INFO_STALE_HOURS", "72")))
        except Exception:
            return 72

    def _shared_info_prune_interval_hours(self) -> float:
        try:
            return max(0.0, float(os.getenv("MEMORY_SHARED_INFO_PRUNE_INTERVAL_HOURS", "24")))
        except Exception:
            return 24.0

    def prune_stale_shared_info(self, *, max_age_hours: int | None = None) -> int:
        """把超过新鲜度窗口仍零投递的 shared_info 行置 expired=1，返回命中行数。

        下限 48h（对齐 gossip_runtime 的 freshness 窗口），避免误杀仍可投递的行。
        """
        hours = int(max_age_hours) if max_age_hours is not None else self.shared_info_stale_hours()
        hours = max(48, hours)
        cutoff = time.time() - hours * 3600.0
        expired = 0

        def _write(conn: sqlite3.Connection) -> None:
            nonlocal expired
            self._ensure_shared_info_schema_on_conn(conn)
            cursor = conn.execute(
                """
                UPDATE shared_info
                SET expired=1
                WHERE expired=0
                  AND (delivered_to_json IS NULL OR delivered_to_json IN ('', '{}'))
                  AND created_at < ?;
                """,
                (cutoff,),
            )
            expired = int(cursor.rowcount or 0)

        self._run_write_transaction(_write)
        self._set_meta_json(
            self.SHARED_INFO_STALE_PRUNE_STATE_KEY,
            {"at": _utc_now_iso(), "expired": expired, "stale_hours": hours},
        )
        return expired

    def _maybe_prune_stale_shared_info(self) -> None:
        interval_hours = self._shared_info_prune_interval_hours()
        if interval_hours <= 0:
            return
        meta = self._get_meta_json(self.SHARED_INFO_STALE_PRUNE_STATE_KEY, {})
        last_at = str(meta.get("at", "") or "") if isinstance(meta, dict) else ""
        if last_at:
            try:
                last_dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
            except Exception:
                last_dt = None
            if last_dt is not None and (utc_now() - last_dt) < timedelta(hours=interval_hours):
                return
        try:
            expired = self.prune_stale_shared_info()
        except Exception as exc:
            _tsz_emit(f"[MemoryCore] shared_info stale prune skipped: {exc}", flush=True)
            return
        if expired:
            _tsz_emit(
                f"[MemoryCore] expired {expired} shared_info rows undelivered for "
                f">{self.shared_info_stale_hours()}h",
                flush=True,
            )

    @staticmethod
    def _ensure_gossip_audit_schema_on_conn(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gossip_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_kind TEXT NOT NULL,
                date_world TEXT NOT NULL,
                source_character TEXT NOT NULL DEFAULT '',
                target_character TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL DEFAULT '',
                text_preview TEXT NOT NULL DEFAULT '',
                detail_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gossip_audit_date ON gossip_audit_log(date_world, event_kind);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_gossip_audit_source ON gossip_audit_log(source_character, date_world);")

    def ensure_gossip_audit_schema(self) -> None:
        self._run_write_transaction(self._ensure_gossip_audit_schema_on_conn)

    def record_gossip_audit(
        self,
        *,
        event_kind: str,
        source_character: str = "",
        target_character: str = "",
        text: str = "",
        detail: dict | None = None,
        created_at: float | None = None,
    ) -> bool:
        """写一条 gossip 链路审计事件（fail-soft 由调用方负责兜底）。"""
        ts = time.time() if created_at is None else float(created_at)
        world_date = datetime.fromtimestamp(ts, WORLD_TZ).strftime("%Y-%m-%d")
        text_preview = str(text or "").strip()[:_GOSSIP_AUDIT_PREVIEW_LEN]
        text_hash = ""
        if text:
            text_hash = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:_GOSSIP_AUDIT_HASH_LEN]

        def _write(conn: sqlite3.Connection) -> bool:
            self._ensure_gossip_audit_schema_on_conn(conn)
            conn.execute(
                """
                INSERT INTO gossip_audit_log
                (event_kind, date_world, source_character, target_character,
                 text_hash, text_preview, detail_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(event_kind or "unknown"),
                    world_date,
                    str(source_character or ""),
                    str(target_character or ""),
                    text_hash,
                    text_preview,
                    _json_dumps(detail if isinstance(detail, dict) else {}),
                    ts,
                ),
            )
            return True

        return bool(self._run_write_transaction(_write))

    @staticmethod
    def _ensure_gossip_extract_audit_schema_on_conn(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gossip_extract_audit (
                date_world TEXT NOT NULL,
                character TEXT NOT NULL,
                extractions INTEGER NOT NULL DEFAULT 0,
                gossip_outputs INTEGER NOT NULL DEFAULT 0,
                blocked_gate INTEGER NOT NULL DEFAULT 0,
                blocked_present_state INTEGER NOT NULL DEFAULT 0,
                validator_drop INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (date_world, character)
            );
            """
        )

    def ensure_gossip_extract_audit_schema(self) -> None:
        self._run_write_transaction(self._ensure_gossip_extract_audit_schema_on_conn)

    def record_gossip_extract_audit(
        self,
        *,
        date_world: str = "",
        character: str = "",
        extractions: int = 0,
        gossip_outputs: int = 0,
        blocked_gate: int = 0,
        blocked_present_state: int = 0,
        validator_drop: int = 0,
        created_at: float | None = None,
    ) -> bool:
        """按 (WORLD_TZ 日, 角色) 累加 gossip 提取侧审计计数（幂等 upsert）。"""
        ts = time.time() if created_at is None else float(created_at)
        if not str(date_world or "").strip():
            date_world = datetime.fromtimestamp(ts, WORLD_TZ).strftime("%Y-%m-%d")

        def _write(conn: sqlite3.Connection) -> bool:
            self._ensure_gossip_extract_audit_schema_on_conn(conn)
            conn.execute(
                """
                INSERT INTO gossip_extract_audit
                (date_world, character, extractions, gossip_outputs, blocked_gate,
                 blocked_present_state, validator_drop, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date_world, character) DO UPDATE SET
                    extractions = extractions + excluded.extractions,
                    gossip_outputs = gossip_outputs + excluded.gossip_outputs,
                    blocked_gate = blocked_gate + excluded.blocked_gate,
                    blocked_present_state = blocked_present_state + excluded.blocked_present_state,
                    validator_drop = validator_drop + excluded.validator_drop,
                    updated_at = excluded.updated_at;
                """,
                (
                    str(date_world),
                    str(character or ""),
                    max(0, int(extractions)),
                    max(0, int(gossip_outputs)),
                    max(0, int(blocked_gate)),
                    max(0, int(blocked_present_state)),
                    max(0, int(validator_drop)),
                    ts,
                ),
            )
            return True

        return bool(self._run_write_transaction(_write))

    def get_domain_json(self, domain_key: str, default=None):
        target = str(domain_key or "").strip()
        if not target:
            return default
        domain_map = self._load_domain_map()
        if target not in domain_map:
            return default
        return domain_map.get(target, default)

    def set_domain_json(self, domain_key: str, value) -> None:
        target = str(domain_key or "").strip()
        if not target:
            return
        self._upsert_domain(target, value)
