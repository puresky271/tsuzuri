from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from tsuzuri.kairos_signal_hub import KairosSignalRequest, build_kairos_signal_bundle
from tsuzuri.kairos_state import ensure_kairos_session, get_kairos_state_store
from tsuzuri.memory_core import MemoryCoreStore
from tsuzuri.runtime_state import get_runtime_state_store


class KairosSignalHubTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.core = MemoryCoreStore(
            db_path=os.path.join(self.tmpdir.name, "memory_core_test.db"),
            snapshot_path=os.path.join(self.tmpdir.name, "dev_memory_test.json"),
        )
        get_runtime_state_store().clear()
        get_kairos_state_store().clear()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reads_self_eval_canon_evolution_runtime_and_kairos_state(self):
        self.core.save_json_snapshot(
            {
                "self_eval_stats": {
                    "total_evals": 4,
                    "recent_overall_ema": 4.4,
                    "injected_constraints": ["避免：语气过软", "避免：回复太长"],
                    "canon_alignment_score": 0.81,
                },
                "self_eval_log": [{"date": "2026-04-02T00:00:00Z", "overall": 4.6}],
                "self_eval_exemplars": [{"character": "张三", "response_snippet": "先把节奏稳住。", "overall": 4.9, "date": "2026-04-02"}],
                "character_states": {
                    "张三": {
                        "evolved_traits": {
                            "note_pool": [{"note": "收紧边界表达", "axis": "boundary_firmness", "direction": "up", "strength": 0.16}],
                            "drift_axes": {"boundary_firmness": 0.12},
                        }
                    }
                },
            },
            source="unit",
        )
        self.core.set_domain_json("kairos.session.sig-1", {"updated_at": "2026-04-02T00:10:00Z", "durable_brief": {"summary": {"headline": "carry over"}}})
        self.core.set_domain_json("kairos.character.张三", {"current_guardrails": ["避免：语气过软"]})

        conn = sqlite3.connect(self.core.db_path)
        try:
            conn.execute(
                """
                INSERT INTO canon_claims
                (character, source_type, source_url, claim_type, claim_text, confidence, freshness, weight, status, extracted_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                ("张三", "wiki", "https://example.com/1", "speech_style", "说话带一点轻快的自我修正", 0.82, 0.75, 0.88, "active", "2026-04-02T00:00:00Z", "{}"),
            )
            conn.execute(
                """
                INSERT INTO ooc_audit_runs
                (audit_id, character, run_id, ooc_score, style_deviation_json, evidence_hits_json,
                 high_conflict_claims_json, penalty_json, applied, shadow, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "ooc_sig_1",
                    "张三",
                    "run_1",
                    0.21,
                    json.dumps({"too_soft": 1.0}, ensure_ascii=False),
                    json.dumps([{"reply_text": "最爱你"}], ensure_ascii=False),
                    json.dumps(["speech_style"], ensure_ascii=False),
                    json.dumps({"style_deviation_axes": {"too_soft": 0.12}, "recommendation": "tighten_internal_style_control"}, ensure_ascii=False),
                    0,
                    1,
                    "2026-04-02T00:20:00Z",
                    "2026-04-02T00:20:00Z",
                ),
            )
            conn.execute(
                """
                INSERT INTO evolution_runs
                (run_id, character, shadow, status, proposal_json, evaluation_json, applied_actions_json, base_snapshot_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    "evo_sig_1",
                    "张三",
                    1,
                    "shadow_planned",
                    json.dumps([{"op": "merge"}], ensure_ascii=False),
                    json.dumps({"quality": 0.73}, ensure_ascii=False),
                    json.dumps([], ensure_ascii=False),
                    "hash_sig_1",
                    "2026-04-02T00:30:00Z",
                    "2026-04-02T00:30:00Z",
                ),
            )
            conn.commit()
        finally:
            conn.close()

        runtime_store = get_runtime_state_store()
        runtime_store.ensure_session("sig-1", channel_kind="qq_private", character="张三")
        runtime_store.set_prompt_summary("sig-1", {"profile": "qq_private"})
        ensure_kairos_session("sig-1", channel_kind="qq_private", character="张三", kairos_enabled=True, brief_enabled=True, bump_turn=True)

        with patch("tsuzuri.kairos_signal_hub._get_memory_core_store", return_value=self.core):
            bundle = build_kairos_signal_bundle(
                KairosSignalRequest(
                    session_id="sig-1",
                    channel_kind="qq_private",
                    character="张三",
                )
            )

        self.assertIn("避免：语气过软", bundle.behavior_guardrails.get("injected_constraints", []))
        self.assertTrue(bundle.behavior_guardrails.get("note_pool_hints"))
        self.assertEqual(bundle.canon_feedback.get("canon_profile", {}).get("claim_count"), 1)
        self.assertEqual(bundle.memory_management.get("latest_evolution", {}).get("run_id"), "evo_sig_1")
        self.assertEqual(bundle.memory_management.get("durable_session", {}).get("durable_brief", {}).get("summary", {}).get("headline"), "carry over")

    def test_missing_sources_return_empty_signals_and_review_flags(self):
        with patch("tsuzuri.kairos_signal_hub._get_memory_core_store", return_value=self.core):
            bundle = build_kairos_signal_bundle(
                KairosSignalRequest(
                    session_id="sig-empty",
                    channel_kind="pet",
                    character="张三",
                )
            )

        self.assertEqual(bundle.behavior_guardrails.get("injected_constraints", []), [])
        self.assertEqual(bundle.canon_feedback.get("recent_ooc_audits", []), [])
        self.assertEqual(bundle.memory_management.get("recent_evolution_runs", []), [])
        self.assertIn("missing_self_eval_stats", bundle.review_flags)
        self.assertIn("missing_ooc_audits", bundle.review_flags)
        self.assertIn("missing_evolution_runs", bundle.review_flags)


if __name__ == "__main__":
    unittest.main()
