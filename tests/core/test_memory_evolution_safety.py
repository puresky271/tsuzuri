from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from tsuzuri.memory_core import MemoryCoreStore


def _iso_ago(*, hours: float = 0.0, days: float = 0.0) -> str:
    value = datetime.now(timezone.utc) - timedelta(hours=hours, days=days)
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _working_row(item_id: str, *, payload: dict | None = None, age_hours: float = 25.0) -> dict:
    return {
        "item_id": item_id,
        "character": "张三",
        "layer": "working",
        "memory_role": "working",
        "engram_type": "turn_exchange",
        "item_type": "turn",
        "text": f"user:{item_id} | assistant:嗯",
        "payload_json": json.dumps(payload or {
            "memory_policy": {
                "memory_features": {"ephemeral_penalty": 0.0},
                "typed_score": 0.0,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            }
        }, ensure_ascii=False),
        "salience": 0.5,
        "importance": 0.5,
        "confidence": 0.7,
        "access_count": 0,
        "created_at": _iso_ago(hours=age_hours),
        "updated_at": _iso_ago(hours=age_hours),
    }


def _trusted_fact_row(item_id: str, *, access_count: int = 0) -> dict:
    return {
        "item_id": item_id,
        "character": "张三",
        "layer": "episodic",
        "memory_role": "episodic",
        "engram_type": "fact_observation",
        "item_type": "fact_observation",
        "text": f"阿澈的事实 {item_id}",
        "payload_json": json.dumps({"fact": f"阿澈的事实 {item_id}"}, ensure_ascii=False),
        "source": "turn_append",
        "salience": 0.75,
        "importance": 0.66,
        "confidence": 0.72,
        "access_count": access_count,
        "created_at": _iso_ago(days=2),
        "updated_at": _iso_ago(days=2),
    }


class MemoryEvolutionSelectionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryCoreStore(
            db_path=os.path.join(self.tmpdir.name, "memory.db"),
            snapshot_path=os.path.join(self.tmpdir.name, "memory.json"),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_promotion_quota_survives_working_pressure(self):
        rows = [_working_row(f"working_{index}") for index in range(8)]
        rows.extend([_trusted_fact_row("fact_a"), _trusted_fact_row("fact_b")])
        actions = self.store._evolution_actions_from_items(rows, max_actions=8)
        promoted = {
            item_id
            for action in actions
            if action.get("op") == "promote" and action.get("target_layer") == "semantic"
            for item_id in action.get("item_ids", [])
        }
        self.assertTrue({"fact_a", "fact_b"}.issubset(promoted))
        self.assertLessEqual(len(actions), 8)

    def test_trusted_fact_observation_promotes_without_recall(self):
        actions = self.store._evolution_actions_from_items(
            [_trusted_fact_row("fact_no_recall", access_count=0)], max_actions=4
        )
        self.assertTrue(any(
            action.get("op") == "promote"
            and "fact_no_recall" in action.get("item_ids", [])
            for action in actions
        ))

    def test_structured_correction_overrides_low_score_archive(self):
        payload = {
            "turn_fact_card": {"corrections": ["阿澈明确纠正：高数是 81 分，不是 71 分。"]},
            "source_trust": {"user_text": "verified"},
            "memory_policy": {
                "memory_features": {"ephemeral_penalty": 1.0},
                "typed_score": 0.0,
                "consolidation_state": "transient",
                "confirmed_reuse_count": 0,
            },
        }
        actions = self.store._evolution_actions_from_items(
            [_working_row("correction", payload=payload)], max_actions=4
        )
        self.assertTrue(any(
            action.get("op") == "promote"
            and action.get("target_layer") == "episodic"
            for action in actions
        ))
        self.assertFalse(any(action.get("op") == "archive" for action in actions))

    def test_zero_action_run_is_terminal(self):
        result = self.store.propose_memory_evolution("张三", max_actions=4)
        conn = sqlite3.connect(self.store.db_path)
        try:
            status = conn.execute(
                "SELECT status FROM evolution_runs WHERE run_id=?", (result["run_id"],)
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(status, "skipped_empty")


class MemoryPromotionApplySafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = MemoryCoreStore(
            db_path=os.path.join(self.tmpdir.name, "memory.db"),
            snapshot_path=os.path.join(self.tmpdir.name, "memory.json"),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_promotion_migrates_role_meta_cue_and_lifecycle(self):
        now = _iso_ago(days=2)
        payload = {
            "memory_meta": {
                "memory_type": "working_turn",
                "owner_scope": "character:张三",
                "evidence_tier": "medium",
                "source_kind": "turn_exchange",
                "temporal_validity": "transient",
                "surface_default": "silent_grounding",
                "mode_scope": "normal",
                "transient_context_only": True,
            },
            "memory_policy": {"consolidation_state": "candidate"},
        }
        conn = sqlite3.connect(self.store.db_path)
        try:
            conn.execute(
                """
                INSERT INTO memory_items
                (item_id, character, layer, memory_role, engram_type, item_type, text,
                 payload_json, cue_json, lifecycle_json, salience, confidence, novelty,
                 importance, status, source, evidence_id, created_at, updated_at,
                 last_accessed_at, access_count, archived)
                VALUES ('promote_me', '张三', 'working', 'working', 'turn_exchange', 'turn',
                        '阿澈明确说高数考了 81 分', ?, '{}', '{}', 0.7, 0.8, 0.4,
                        0.7, 'active', 'turn_append', '', ?, ?, '', 0, 0)
                """,
                (json.dumps(payload, ensure_ascii=False), now, now),
            )
            conn.commit()
        finally:
            conn.close()

        result = self.store.apply_evolution_plan(
            plan=[{
                "op": "promote",
                "item_ids": ["promote_me"],
                "target_layer": "semantic",
                "target_engram_type": "user_profile_fact",
                "target_state": "profile_promoted",
            }],
            shadow_only=False,
        )
        self.assertTrue(result["ok"])
        conn = sqlite3.connect(self.store.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM memory_items WHERE item_id='promote_me'").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["layer"], "semantic")
        self.assertEqual(row["memory_role"], "semantic")
        promoted_payload = json.loads(row["payload_json"])
        self.assertFalse(promoted_payload["memory_meta"]["transient_context_only"])
        self.assertEqual(json.loads(row["cue_json"])["memory_role"], "semantic")
        self.assertEqual(json.loads(row["lifecycle_json"])["state"], "consolidation")


if __name__ == "__main__":
    unittest.main()
