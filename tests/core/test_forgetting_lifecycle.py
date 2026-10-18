from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from unittest import mock

from tsuzuri import kairos_dream as kairos_dream
from tsuzuri.kairos_dream import (
    _ensure_llm_env_from_secrets,
    _load_recent_working_memory_texts,
    _run_forgetting_lifecycle,
)
from tsuzuri.memory_api import MemoryApi
from tsuzuri.memory_core import MemoryCoreStore, _forgetting_protected
from tsuzuri.secrets_loader import _RUNTIME_ENV_KEYS, env_bool, sync_env_from_secrets


class SecretsLoaderTests(unittest.TestCase):
    pass  # upstream-only test removed

    def test_runtime_sync_can_explicitly_overwrite_and_env_bool_keeps_env_priority(self):
        config = {"MEM_FORGETTING_ENABLED": True}
        with (
            mock.patch.dict(os.environ, {"MEM_FORGETTING_ENABLED": "0"}, clear=True),
            mock.patch("tsuzuri.secrets_loader.load_project_secrets", return_value=config),
        ):
            self.assertFalse(env_bool("MEM_FORGETTING_ENABLED", True))
            self.assertEqual(sync_env_from_secrets(overwrite=True), 1)
            self.assertEqual(os.environ["MEM_FORGETTING_ENABLED"], "True")
            self.assertTrue(env_bool("MEM_FORGETTING_ENABLED", False))

    def test_env_bool_falls_back_to_secrets_and_default(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("tsuzuri.secrets_loader.load_project_secrets", return_value={"SCHEDULE_TOOLS_ENABLE": "yes"}),
        ):
            self.assertTrue(env_bool("SCHEDULE_TOOLS_ENABLE", False))
            self.assertFalse(env_bool("MISSING_FLAG", False))
            self.assertTrue(env_bool("MISSING_FLAG", True))


class ForgettingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "memory.db")
        self.store = MemoryCoreStore(
            db_path=self.db_path,
            snapshot_path=os.path.join(self.tmpdir.name, "snapshot.json"),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _seed(
        self,
        item_id: str,
        *,
        character: str = "张三",
        layer: str = "episodic",
        status: str = "archived_decayed",
        text: str = "user: 一条已经久远的普通对话 | assistant: 收到了",
        payload: dict | None = None,
        salience: float = 0.3,
        access_count: int = 0,
        engram_type: str = "turn_exchange",
        created_at: str = "2025-01-01T00:00:00Z",
        importance: float = 0.2,
    ) -> None:
        self.store._upsert_item(
            {
                "item_id": item_id,
                "character": character,
                "layer": layer,
                "memory_role": layer,
                "engram_type": engram_type,
                "item_type": engram_type,
                "text": text,
                "payload_json": json.dumps(payload or {}, ensure_ascii=False),
                "cue_json": "{}",
                "lifecycle_json": "{}",
                "salience": salience,
                "confidence": 0.8,
                "novelty": 0.2,
                "importance": importance,
                "status": status,
                "source": "forgetting_test",
                "evidence_id": f"evidence:{item_id}",
                "created_at": created_at,
                "updated_at": created_at,
                "last_accessed_at": "",
                "access_count": access_count,
                "archived": 1,
                "embedding_text": "dense-vector",
                "topic_labels_json": "[]",
            }
        )
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "UPDATE memory_items SET embedding_sparse_text='sparse-vector' WHERE item_id=?",
                (item_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def _row(self, item_id: str) -> sqlite3.Row:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM memory_items WHERE item_id=?", (item_id,)).fetchone()
            self.assertIsNotNone(row)
            return row
        finally:
            conn.close()

    def test_protected_layers_never_advance(self):
        for layer in ("working", "semantic", "collective", "procedural", ""):
            with self.subTest(layer=layer):
                protected, reason = _forgetting_protected({"layer": layer})
                self.assertTrue(protected)
                self.assertEqual(reason, f"protected_layer:{layer or 'unknown'}")

    def test_protected_evidence_classes(self):
        cases = (
            ({"layer": "episodic", "engram_type": "strategy_rule"}, "protected_engram_type:strategy_rule"),
            (
                {
                    "layer": "episodic",
                    "engram_type": "turn_exchange",
                    "payload_json": json.dumps({"turn_fact_card": {"corrections": ["旧信息有误"]}}, ensure_ascii=False),
                },
                "correction",
            ),
            (
                {
                    "layer": "episodic",
                    "engram_type": "turn_exchange",
                    "payload_json": json.dumps({"evidence_policy": {"visibility": "explicit_recall_only"}}),
                },
                "explicit_recall_archive",
            ),
            (
                {
                    "layer": "episodic",
                    "engram_type": "turn_exchange",
                    "payload_json": json.dumps({"memory_meta": {"source_kind": "source_backed_archive"}}),
                },
                "source_backed_archive",
            ),
        )
        for row, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                protected, reason = _forgetting_protected(row)
                self.assertTrue(protected)
                self.assertEqual(reason, expected_reason)

    def test_protected_salience_and_access(self):
        protected, reason = _forgetting_protected({"layer": "episodic", "salience": 0.70})
        self.assertEqual((protected, reason), (True, "high_salience"))
        protected, reason = _forgetting_protected(
            {"layer": "episodic", "salience": 0.2, "access_count": 1}
        )
        self.assertEqual((protected, reason), (True, "accessed"))

    def test_unprotected_episodic_passes(self):
        self.assertEqual(
            _forgetting_protected(
                {"layer": "episodic", "engram_type": "turn_exchange", "salience": 0.2, "access_count": 0}
            ),
            (False, ""),
        )

    def test_dry_run_reports_both_archive_sources_without_writes(self):
        self._seed("decayed", status="archived_decayed")
        self._seed("evidence-archive", status="archived")
        result = self.store.advance_forgetting_lifecycle("张三", dry_run=True)
        self.assertEqual(result["frozen_candidates"], 2)
        self.assertEqual(result["frozen_applied"], 0)
        self.assertEqual({row["item_id"] for row in result["candidates"]}, {"decayed", "evidence-archive"})
        self.assertEqual(self._row("decayed")["status"], "archived_decayed")
        self.assertEqual(self._row("evidence-archive")["status"], "archived")

    def test_archived_to_frozen_retains_text_and_clears_embeddings(self):
        self._seed("freeze-me")
        original = self._row("freeze-me")["text"]
        result = self.store.advance_forgetting_lifecycle("张三", dry_run=False)
        row = self._row("freeze-me")
        self.assertEqual(result["frozen_applied"], 1)
        self.assertEqual(result["tombstone_applied"], 0)
        self.assertEqual(row["status"], "frozen")
        self.assertEqual(row["text"], original)
        self.assertEqual(row["embedding_text"], "")
        self.assertEqual(row["embedding_sparse_text"], "")

    def test_non_episodic_archived_rows_never_enter_candidate_pools(self):
        self._seed("archived-working", layer="working", status="archived")
        self._seed("frozen-working", layer="working", status="frozen")

        result = self.store.advance_forgetting_lifecycle("张三", dry_run=False)

        self.assertEqual(result["frozen_candidates"], 0)
        self.assertEqual(result["frozen_applied"], 0)
        self.assertEqual(result["tombstone_candidates"], 0)
        self.assertEqual(result["tombstone_applied"], 0)
        self.assertEqual(self._row("archived-working")["status"], "archived")
        frozen = self._row("frozen-working")
        self.assertEqual(frozen["status"], "frozen")
        self.assertNotEqual(frozen["text"], "[forgotten]")

    def test_newly_frozen_row_cannot_tombstone_in_same_call(self):
        self._seed("two-stage")
        first = self.store.advance_forgetting_lifecycle("张三", dry_run=False)
        self.assertEqual(first["frozen_applied"], 1)
        self.assertEqual(first["tombstone_applied"], 0)
        self.assertEqual(self._row("two-stage")["status"], "frozen")
        second = self.store.advance_forgetting_lifecycle("张三", dry_run=False)
        self.assertEqual(second["tombstone_applied"], 1)
        self.assertEqual(self._row("two-stage")["status"], "tombstone")

    def test_frozen_to_tombstone_retains_provenance_and_history(self):
        original = "一条应该被真正遗忘的旧内容。"
        self._seed("tombstone-me", status="frozen", text=original, payload={"memory_meta": {"source_kind": "turn_event"}})
        result = self.store.advance_forgetting_lifecycle("张三", dry_run=False)
        row = self._row("tombstone-me")
        payload = json.loads(row["payload_json"])
        self.assertEqual(result["tombstone_applied"], 1)
        self.assertEqual(row["status"], "tombstone")
        self.assertEqual(row["text"], "[forgotten]")
        self.assertEqual(row["evidence_id"], "evidence:tombstone-me")
        self.assertEqual(row["created_at"], "2025-01-01T00:00:00Z")
        self.assertEqual(row["layer"], "episodic")
        self.assertEqual(payload["memory_meta"]["source_kind"], "turn_event")
        history = payload["forgetting_history"][-1]
        self.assertEqual(history["action"], "tombstone")
        self.assertEqual(history["original_text_len"], len(original))
        self.assertEqual(history["evidence_id"], "evidence:tombstone-me")

    def test_age_character_and_invalid_timestamp_guards(self):
        self._seed("too-young", created_at="2026-08-01T00:00:00Z")
        self._seed("other-char", character="李四")
        self._seed("invalid-ts", created_at="not-a-timestamp")
        result = self.store.advance_forgetting_lifecycle("张三", dry_run=True)
        self.assertEqual(result["frozen_candidates"], 0)
        self.assertEqual(result["skipped_invalid_timestamp"], 1)
        self.assertEqual(self._row("other-char")["status"], "archived_decayed")

    def test_retrieval_excludes_frozen_and_tombstone(self):
        self._seed("frozen-hidden", status="frozen", text="user: 隐藏的蓝色票根")
        self._seed("tombstone-hidden", status="tombstone", text="[forgotten]")
        rows = self.store._query_items("张三", limit=20, query_text="蓝色票根")
        ids = {str(row.get("item_id", "")) for row in rows}
        self.assertNotIn("frozen-hidden", ids)
        self.assertNotIn("tombstone-hidden", ids)

    def test_upsert_does_not_revive_tombstone_or_restore_fts(self):
        self._seed("no-revive", status="frozen", text="原始内容")
        self.store.advance_forgetting_lifecycle("张三", dry_run=False)
        self.store._upsert_item(
            {
                "item_id": "no-revive",
                "character": "张三",
                "layer": "episodic",
                "item_type": "turn_exchange",
                "text": "试图复活的内容",
                "status": "active",
                "archived": 0,
                "source": "turn_append",
            }
        )
        row = self._row("no-revive")
        self.assertEqual(row["status"], "tombstone")
        self.assertEqual(row["archived"], 1)
        self.assertEqual(row["text"], "[forgotten]")
        conn = sqlite3.connect(self.db_path)
        try:
            old_hits = conn.execute(
                "SELECT item_id FROM memory_items_fts WHERE memory_items_fts MATCH '原始内容'"
            ).fetchall()
            revive_hits = conn.execute(
                "SELECT item_id FROM memory_items_fts WHERE memory_items_fts MATCH '试图复活'"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(old_hits, [])
        self.assertEqual(revive_hits, [])

    def test_api_apply_requires_env_gate(self):
        self._seed("api-candidate")
        api = MemoryApi(self.store)
        with mock.patch.dict(os.environ, {"MEM_FORGETTING_APPLY": "0"}):
            denied = api.advance_forgetting("张三", dry_run=False)
        self.assertEqual(denied["error"], "forgetting_apply_disabled")
        preview = api.advance_forgetting_dry_run("张三")
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["frozen_candidates"], 1)

    def test_admin_cli_exposes_preview_and_confirmed_advance(self):
        from tsuzuri.memory_admin import _build_parser

        preview = _build_parser().parse_args(["forgetting", "preview", "--char", "张三"])
        advance = _build_parser().parse_args(
            ["forgetting", "advance", "--char", "张三", "--yes", "--frozen-days", "120"]
        )
        self.assertEqual((preview.command, preview.action, preview.yes), ("forgetting", "preview", False))
        self.assertEqual((advance.action, advance.yes, advance.frozen_days), ("advance", True, 120.0))

    def test_kairos_job_is_disabled_by_default_and_idempotent_when_enabled(self):
        self._seed("kairos-candidate")
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("tsuzuri.secrets_loader.load_project_secrets", return_value={}),
        ):
            os.environ.pop("MEM_FORGETTING_ENABLED", None)
            disabled = _run_forgetting_lifecycle(self.store, "张三")
        self.assertEqual(disabled["status"], "disabled")
        with mock.patch.dict(
            os.environ,
            {"MEM_FORGETTING_ENABLED": "1", "MEM_FORGETTING_APPLY": "0"},
        ):
            first = _run_forgetting_lifecycle(self.store, "张三")
            second = _run_forgetting_lifecycle(self.store, "张三")
        self.assertEqual(first["status"], "dry_run")
        self.assertEqual(first["frozen_candidates"], 1)
        self.assertEqual(second["status"], "idempotent")
        self.assertTrue(str(first["meta_key"]).startswith("kairos.forgetting.张三."))

    def test_kairos_loads_dream_flags_when_llm_env_is_already_set(self):
        config = {
            "MEM_CORE_SUMMARY_ENABLE": True,
            "MEM_CORE_SUMMARY_MODEL": "summary-model",
            "MEM_ENTITY_REBUILD_ENABLE": True,
            "MEM_RETHINK_SHADOW_ONLY": False,
            "MEM_EVIDENCE_MAINTENANCE_APPLY": False,
            "MEM_FORGETTING_ENABLED": True,
            "MEM_FORGETTING_APPLY": False,
            "MEM_FORGETTING_FROZEN_DAYS": 90,
            "MEM_FORGETTING_TOMBSTONE_DAYS": 180,
        }
        existing = {
            "LLM_API_KEY": "already-set",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "fixture-model",
        }
        with (
            mock.patch.dict(os.environ, existing, clear=True),
            mock.patch("tsuzuri.secrets_loader.load_project_secrets", return_value=config),
        ):
            _ensure_llm_env_from_secrets()
            self.assertEqual(os.environ["MEM_CORE_SUMMARY_ENABLE"], "True")
            self.assertEqual(os.environ["MEM_CORE_SUMMARY_MODEL"], "summary-model")
            self.assertEqual(os.environ["MEM_ENTITY_REBUILD_ENABLE"], "True")
            self.assertEqual(os.environ["MEM_RETHINK_SHADOW_ONLY"], "False")
            self.assertEqual(os.environ["MEM_EVIDENCE_MAINTENANCE_APPLY"], "False")
            self.assertEqual(os.environ["MEM_FORGETTING_ENABLED"], "True")
            self.assertEqual(os.environ["MEM_FORGETTING_APPLY"], "False")
            self.assertEqual(os.environ["MEM_FORGETTING_FROZEN_DAYS"], "90")
            self.assertEqual(os.environ["MEM_FORGETTING_TOMBSTONE_DAYS"], "180")

    def test_kairos_dream_flags_do_not_override_process_environment(self):
        config = {
            "MEM_ENTITY_REBUILD_ENABLE": True,
            "MEM_RETHINK_SHADOW_ONLY": False,
        }
        with (
            mock.patch.dict(
                os.environ,
                {
                    "MEM_ENTITY_REBUILD_ENABLE": "0",
                    "MEM_RETHINK_SHADOW_ONLY": "1",
                },
                clear=True,
            ),
            mock.patch("tsuzuri.secrets_loader.load_project_secrets", return_value=config),
        ):
            _ensure_llm_env_from_secrets()
            self.assertEqual(os.environ["MEM_ENTITY_REBUILD_ENABLE"], "0")
            self.assertEqual(os.environ["MEM_RETHINK_SHADOW_ONLY"], "1")

    pass  # upstream-only test removed

    def test_kairos_profile_context_excludes_archived_rows(self):
        self._seed("archived-working", layer="working", text="archived secret", importance=0.99)
        self.store._upsert_item(
            {
                "item_id": "active-working",
                "character": "张三",
                "layer": "working",
                "item_type": "turn_exchange",
                "text": "active context",
                "status": "active",
                "archived": 0,
                "source": "forgetting_test",
                "created_at": "2026-08-03T00:00:00Z",
            }
        )
        with mock.patch.object(kairos_dream, "MEMORY_CORE_DB_PATH", self.db_path):
            texts = _load_recent_working_memory_texts("张三")
        self.assertEqual(texts, ["active context"])

    pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
