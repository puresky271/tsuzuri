from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from tsuzuri.kairos_state import (
    ensure_kairos_session,
    get_kairos_state_store,
    hydrate_kairos_state_from_persistent,
    is_kairos_default_enabled,
    list_recent_kairos_dreams,
    record_kairos_brief,
    record_kairos_dream,
    resolve_kairos_switches,
    snapshot_kairos_session,
)
from tsuzuri.memory_core import MemoryCoreStore


class KairosStateStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = get_kairos_state_store()
        self.store.clear()

    def test_session_lifecycle_and_snapshots(self):
        ensure_kairos_session(
            "kairos-sess-1",
            channel_kind="pet",
            character="张三",
            kairos_enabled=True,
            brief_enabled=True,
            bump_turn=True,
        )
        record_kairos_brief(
            "kairos-sess-1",
            {
                "brief_text": "brief-1",
                "summary": {"headline": "pet brief"},
                "sections": {"runtime_continuity": "ok"},
                "source_usage": {"runtime_state": {"has_prompt_summary": True}},
            },
        )
        record_kairos_dream(
            "kairos-sess-1",
            {
                "ok": True,
                "summary": {"reason_code": "dream_completed"},
                "domain_writes": ["kairos.session.kairos-sess-1"],
                "meta_writes": ["kairos.session.kairos-sess-1"],
                "refreshed_brief": {
                    "brief_text": "brief-2",
                    "summary": {"headline": "dream brief"},
                    "sections": {"world_state": "stable"},
                    "source_usage": {"world_state": {"reflection_count": 1}},
                },
                "turn_count": 1,
            },
        )

        snapshot = snapshot_kairos_session("kairos-sess-1")
        self.assertEqual(snapshot["session_id"], "kairos-sess-1")
        self.assertEqual(snapshot["channel_kind"], "pet")
        self.assertEqual(snapshot["character"], "张三")
        self.assertTrue(snapshot["kairos_enabled"])
        self.assertTrue(snapshot["brief_enabled"])
        self.assertEqual(snapshot["turn_count"], 1)
        self.assertEqual(snapshot["last_brief"]["brief_text"], "brief-2")
        self.assertEqual(snapshot["last_brief_summary"]["headline"], "dream brief")
        self.assertEqual(snapshot["last_dream_record"]["summary"]["reason_code"], "dream_completed")
        self.assertGreater(float(snapshot["last_brief_at"]), 0.0)
        self.assertGreater(float(snapshot["last_dream_at"]), 0.0)

    def test_disabled_enabled_isolated_per_session(self):
        ensure_kairos_session("sess-a", channel_kind="qq_private", character="张三", kairos_enabled=False, brief_enabled=False)
        ensure_kairos_session("sess-b", channel_kind="pet", character="赵六", kairos_enabled=True, brief_enabled=True, bump_turn=True)
        snap_a = snapshot_kairos_session("sess-a")
        snap_b = snapshot_kairos_session("sess-b")
        self.assertFalse(snap_a["kairos_enabled"])
        self.assertFalse(snap_a["brief_enabled"])
        self.assertEqual(snap_a["turn_count"], 0)
        self.assertTrue(snap_b["kairos_enabled"])
        self.assertTrue(snap_b["brief_enabled"])
        self.assertEqual(snap_b["turn_count"], 1)

    def test_default_enabled_and_explicit_overrides_are_stable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TSUZURI_KAIROS_ENABLE", None)
            self.assertTrue(is_kairos_default_enabled())
            self.assertEqual(
                resolve_kairos_switches({"brief_enabled": False}),
                {"kairos_enabled": True, "brief_enabled": False},
            )
            self.assertEqual(
                resolve_kairos_switches({"kairos_enabled": False}),
                {"kairos_enabled": False, "brief_enabled": False},
            )

        with patch.dict(os.environ, {"TSUZURI_KAIROS_ENABLE": "off"}, clear=False):
            self.assertFalse(is_kairos_default_enabled())
            self.assertEqual(
                resolve_kairos_switches({"kairos_enabled": True}),
                {"kairos_enabled": True, "brief_enabled": True},
            )

    def test_hydrate_from_persistent_rebuilds_display_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCoreStore(
                db_path=os.path.join(tmpdir, "memory_core_test.db"),
                snapshot_path=os.path.join(tmpdir, "dev_memory_test.json"),
            )
            core.set_meta_state_json(
                "kairos.session.pet_api:张三",
                {
                    "session_id": "pet_api:张三",
                    "channel_kind": "pet",
                    "character": "张三",
                    "last_dream_at": "2026-04-11T17:22:37+00:00",
                    "turn_count": 20,
                    "signal_summary": {"review_flags": ["ok"]},
                },
            )
            core.set_domain_json(
                "kairos.session.pet_api:张三",
                {
                    "session_id": "pet_api:张三",
                    "channel_kind": "pet",
                    "character": "张三",
                    "updated_at": "2026-04-11T17:22:37+00:00",
                    "turn_count": 20,
                    "durable_brief": {
                        "brief_text": "brief",
                        "summary": {"headline": "persistent brief"},
                        "sections": {"memory_surface": "ok"},
                        "source_usage": {},
                    },
                },
            )
            self.store.clear()
            result = hydrate_kairos_state_from_persistent(core)
            self.assertEqual(result.get("hydrated_sessions"), 1)
            snapshot = snapshot_kairos_session("pet_api:张三")
            self.assertTrue(snapshot.get("hydrated_from_persistent"))
            self.assertEqual(snapshot.get("character"), "张三")
            self.assertEqual(snapshot.get("last_brief_summary", {}).get("headline"), "persistent brief")
            dreams = list_recent_kairos_dreams(limit=3)
            self.assertTrue(any(row.get("session_id") == "pet_api:张三" for row in dreams))


if __name__ == "__main__":
    unittest.main()
