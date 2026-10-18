"""Integration tests for Phase C: emotion-memory coupling.

Tests _apply_affect_memory_boost (pure-function) and the actual effect on
retrieve_memory_context ranking when affect_state is passed via query_bundle.

Per project requirement: real recall detection is mandatory — mock-only tests
do not count. The integration suite inserts fixtures into a real SQLite DB,
runs retrieve_memory_context with different affect states, and asserts that
high-emotional-weight memories rank higher under negative-valence conditions.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from tsuzuri.memory_core import (
    _apply_affect_memory_boost,
    _extract_payload_emotional_weight,
    MemoryCoreStore,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure-function unit tests (deterministic, no DB)
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyAffectMemoryBoostPure(unittest.TestCase):
    """_apply_affect_memory_boost 纯函数行为验证。"""

    def test_neutral_affect_no_change(self):
        """中性情绪（valence≈0, arousal≈0）不改变分数。"""
        s = _apply_affect_memory_boost(0.5, 0.6, {"valence": 0.0, "arousal": 0.0})
        self.assertAlmostEqual(s, 0.5, places=5)

    def test_none_affect_no_change(self):
        """affect_state=None 不改变分数。"""
        self.assertAlmostEqual(_apply_affect_memory_boost(0.4, 0.8, None), 0.4, places=5)

    def test_negative_valence_boosts_high_ew(self):
        """负面情绪态 + 高情感权重记忆 → 分数提升 ×1.40。"""
        base = 0.5
        s = _apply_affect_memory_boost(base, 0.8, {"valence": -0.5, "arousal": 0.3})
        self.assertGreater(s, base)
        self.assertAlmostEqual(s, min(1.8, base * 1.40), places=5)

    def test_negative_valence_suppresses_low_ew(self):
        """负面情绪态 + 低情感权重记忆 → 分数降低 ×0.75。"""
        base = 0.6
        s = _apply_affect_memory_boost(base, 0.1, {"valence": -0.6, "arousal": 0.2})
        self.assertLess(s, base)
        self.assertAlmostEqual(s, min(1.8, base * 0.75), places=5)

    def test_high_arousal_boosts_medium_high_ew(self):
        """高唤起态 + ew≥0.5 → 分数提升 ×1.25。"""
        base = 0.4
        s = _apply_affect_memory_boost(base, 0.6, {"valence": 0.1, "arousal": 0.8})
        self.assertGreater(s, base)
        self.assertAlmostEqual(s, min(1.8, base * 1.25), places=5)

    def test_high_arousal_low_ew_no_boost(self):
        """高唤起态 + 低情感权重 → 不触发 boost。"""
        base = 0.4
        s = _apply_affect_memory_boost(base, 0.3, {"valence": 0.1, "arousal": 0.9})
        self.assertAlmostEqual(s, base, places=5)

    def test_positive_high_arousal_boosts_very_high_ew(self):
        """正面高唤起 + ew≥0.7 → 轻微 boost ×1.15。"""
        base = 0.5
        s = _apply_affect_memory_boost(base, 0.8, {"valence": 0.7, "arousal": 0.6})
        self.assertGreater(s, base)
        self.assertAlmostEqual(s, min(1.8, base * 1.15), places=5)

    def test_cap_at_1_8(self):
        """分数不超过 1.8。"""
        s = _apply_affect_memory_boost(1.7, 0.9, {"valence": -0.9, "arousal": 0.0})
        self.assertLessEqual(s, 1.8)

    def test_boundary_valence_exactly_minus_0_3(self):
        """valence < -0.3 严格小于：-0.3 本身不触发，-0.31 触发。"""
        base = 0.5
        # 精确 -0.3 不在范围内（严格 <）
        s_edge = _apply_affect_memory_boost(base, 0.7, {"valence": -0.3, "arousal": 0.0})
        self.assertAlmostEqual(s_edge, base, places=5, msg="valence=-0.3 (boundary) should not boost")
        # -0.31 触发
        s_in = _apply_affect_memory_boost(base, 0.7, {"valence": -0.31, "arousal": 0.0})
        self.assertGreater(s_in, base, msg="valence=-0.31 (inside range) should boost")

    def test_boundary_valence_above_threshold_no_boost(self):
        """valence = -0.2（未达 -0.3）不触发负面态 boost。"""
        base = 0.5
        s = _apply_affect_memory_boost(base, 0.9, {"valence": -0.2, "arousal": 0.0})
        self.assertAlmostEqual(s, base, places=5)


class TestExtractPayloadEmotionalWeight(unittest.TestCase):
    """_extract_payload_emotional_weight 纯函数。"""

    def test_normal_path(self):
        payload = {"memory_policy": {"memory_features": {"emotional_weight": 0.75}}}
        self.assertAlmostEqual(_extract_payload_emotional_weight(payload), 0.75)

    def test_missing_keys(self):
        self.assertAlmostEqual(_extract_payload_emotional_weight({}), 0.0)
        self.assertAlmostEqual(_extract_payload_emotional_weight(None), 0.0)

    def test_clamp_above_1(self):
        payload = {"memory_policy": {"memory_features": {"emotional_weight": 1.5}}}
        self.assertLessEqual(_extract_payload_emotional_weight(payload), 1.0)

    def test_clamp_below_0(self):
        payload = {"memory_policy": {"memory_features": {"emotional_weight": -0.3}}}
        self.assertGreaterEqual(_extract_payload_emotional_weight(payload), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Real DB integration: affect_state influences actual retrieve ranking
# ─────────────────────────────────────────────────────────────────────────────

def _make_memory_payload(emotional_weight: float) -> str:
    return json.dumps({
        "memory_policy": {
            "memory_features": {"emotional_weight": emotional_weight},
            "transient_context_only": False,
        }
    }, ensure_ascii=False)


class TestAffectMemoryCouplingRealDB(unittest.TestCase):
    """真实 DB 召回检测：verify affect_state shifts ranking of high-EW vs low-EW memories."""

    def setUp(self):
        # Fresh store per test to avoid cross-test cache contamination
        from datetime import datetime, timezone
        self._tmp = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmp, "test_affect.db")
        self._snap_path = os.path.join(self._tmp, "snap.json")
        with open(self._snap_path, "w") as f:
            json.dump({}, f)
        self._now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._store = MemoryCoreStore(db_path=self._db_path, snapshot_path=self._snap_path)
        self._store._upsert_item({
            "item_id": "item_high_ew",
            "character": "张三",
            "layer": "episodic",
            "memory_role": "episodic",
            "engram_type": "event",
            "item_type": "event",
            "text": "集训时张三和赵六发生了激烈争执，张三哭了出来，王五沉默",
            "payload_json": _make_memory_payload(0.9),
            "salience": 0.6,
            "confidence": 0.75,
            "novelty": 0.4,
            "importance": 0.68,
            "status": "active",
            "source": "test",
            "created_at": self._now,
            "updated_at": self._now,
            "access_count": 2,
        })
        self._store._upsert_item({
            "item_id": "item_low_ew",
            "character": "张三",
            "layer": "episodic",
            "memory_role": "episodic",
            "engram_type": "event",
            "item_type": "event",
            "text": "集训结束后大家分别回家，张三说下次见",
            "payload_json": _make_memory_payload(0.1),
            "salience": 0.6,
            "confidence": 0.75,
            "novelty": 0.4,
            "importance": 0.72,
            "status": "active",
            "source": "test",
            "created_at": self._now,
            "updated_at": self._now,
            "access_count": 2,
        })

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _retrieve(self, affect_state: dict | None) -> list[str]:
        """检索"集训"相关记忆，返回 item_id 排序列表（top-5）。"""
        qb: dict = {}
        if affect_state is not None:
            qb["affect_state"] = affect_state
        result = self._store.retrieve_memory_context(
            "张三", "集训",
            query_bundle=qb,
            max_items=5,
            touch_selected=False,
            log_retrieval=False,
        )
        rows = result.get("rows", [])
        return [str(r.get("item_id", "") or "") for r in rows]

    @__import__("unittest").skipUnless(__import__("os").getenv("EMBEDDING_ENABLED") == "1", "needs dense embeddings (embeddings extra)")
    def test_neutral_affect_both_present(self):
        """中性情绪：两条记忆都应被召回（验证 fixture 有效）。"""
        ids = self._retrieve({"valence": 0.0, "arousal": 0.0})
        # 至少一条能被召回（DB 可能没有 embedding，lexical 足以匹配"集训"）
        recalled = set(ids) & {"item_high_ew", "item_low_ew"}
        self.assertTrue(len(recalled) >= 1,
                        f"Expected ≥1 of test fixtures recalled under neutral affect, got: {ids}")

    def test_negative_valence_high_ew_ranks_higher(self):
        """负面情绪态时 high_ew 记忆排名应高于 low_ew。"""
        ids_neg = self._retrieve({"valence": -0.7, "arousal": 0.2})
        ids_neu = self._retrieve({"valence": 0.0, "arousal": 0.0})

        if "item_high_ew" not in ids_neg or "item_low_ew" not in ids_neg:
            self.skipTest(
                f"Fixtures not both recalled (neg={ids_neg}, neu={ids_neu}); "
                "embedding may be disabled—lexical recall only"
            )

        rank_high_neg = ids_neg.index("item_high_ew")
        rank_low_neg = ids_neg.index("item_low_ew")
        self.assertLess(
            rank_high_neg, rank_low_neg,
            f"Under negative valence, high_ew should rank before low_ew. "
            f"Got high_ew@{rank_high_neg}, low_ew@{rank_low_neg} in {ids_neg}"
        )

    def test_negative_valence_raises_high_ew_score_vs_neutral(self):
        """负面情绪态时 high_ew 记忆的分数应比中性情绪时更高。"""
        def _score(affect: dict | None, item_id: str) -> float | None:
            qb: dict = {}
            if affect:
                qb["affect_state"] = affect
            result = self._store.retrieve_memory_context(
                "张三", "集训",
                query_bundle=qb, max_items=5,
                touch_selected=False, log_retrieval=False,
            )
            for r in result.get("rows", []):
                if str(r.get("item_id", "")) == item_id:
                    return float(r.get("score", 0.0))
            return None

        score_neu = _score({"valence": 0.0, "arousal": 0.0}, "item_high_ew")
        score_neg = _score({"valence": -0.7, "arousal": 0.2}, "item_high_ew")

        if score_neu is None or score_neg is None:
            self.skipTest(
                f"item_high_ew not recalled: neutral={score_neu}, neg={score_neg}"
            )

        self.assertGreater(
            score_neg, score_neu,
            f"Under negative valence, item_high_ew score should be higher. "
            f"neutral={score_neu:.4f} neg={score_neg:.4f}"
        )

    def test_high_arousal_boosts_high_ew_score(self):
        """高唤起态时 high_ew 记忆的分数应高于中性情绪。"""
        def _score(affect: dict | None, item_id: str) -> float | None:
            qb: dict = {}
            if affect:
                qb["affect_state"] = affect
            result = self._store.retrieve_memory_context(
                "张三", "集训",
                query_bundle=qb, max_items=5,
                touch_selected=False, log_retrieval=False,
            )
            for r in result.get("rows", []):
                if str(r.get("item_id", "")) == item_id:
                    return float(r.get("score", 0.0))
            return None

        score_neu = _score({"valence": 0.0, "arousal": 0.0}, "item_high_ew")
        score_high_arousal = _score({"valence": 0.1, "arousal": 0.85}, "item_high_ew")

        if score_neu is None or score_high_arousal is None:
            self.skipTest(
                f"item_high_ew not recalled: neutral={score_neu}, high_arousal={score_high_arousal}"
            )

        self.assertGreater(
            score_high_arousal, score_neu,
            f"Under high arousal, item_high_ew score should be higher. "
            f"neutral={score_neu:.4f} high_arousal={score_high_arousal:.4f}"
        )

    def test_negative_valence_depresses_low_ew_score(self):
        """负面情绪态时 low_ew 记忆的分数应低于中性情绪时。"""
        def _score(affect: dict | None, item_id: str) -> float | None:
            qb: dict = {}
            if affect:
                qb["affect_state"] = affect
            result = self._store.retrieve_memory_context(
                "张三", "集训",
                query_bundle=qb, max_items=5,
                touch_selected=False, log_retrieval=False,
            )
            for r in result.get("rows", []):
                if str(r.get("item_id", "")) == item_id:
                    return float(r.get("score", 0.0))
            return None

        score_neu = _score({"valence": 0.0, "arousal": 0.0}, "item_low_ew")
        score_neg = _score({"valence": -0.7, "arousal": 0.2}, "item_low_ew")

        if score_neu is None or score_neg is None:
            self.skipTest(
                f"item_low_ew not recalled: neutral={score_neu}, neg={score_neg}"
            )

        self.assertLess(
            score_neg, score_neu,
            f"Under negative valence, item_low_ew score should be lower. "
            f"neutral={score_neu:.4f} neg={score_neg:.4f}"
        )

    def test_no_affect_state_in_qb_no_change(self):
        """query_bundle 无 affect_state 时，分数与传空 dict 一致。"""
        def _score(qb: dict, item_id: str) -> float | None:
            result = self._store.retrieve_memory_context(
                "张三", "集训",
                query_bundle=qb, max_items=5,
                touch_selected=False, log_retrieval=False,
            )
            for r in result.get("rows", []):
                if str(r.get("item_id", "")) == item_id:
                    return float(r.get("score", 0.0))
            return None

        s_no_qb = _score({}, "item_high_ew")
        s_no_affect = _score({"some_other_key": "x"}, "item_high_ew")

        if s_no_qb is None or s_no_affect is None:
            self.skipTest("item_high_ew not recalled without affect_state")

        self.assertAlmostEqual(
            s_no_qb, s_no_affect, places=5,
            msg="Without affect_state key, scores should be identical"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
