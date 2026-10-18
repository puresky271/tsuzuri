"""Unit tests for Phase B: BM25 + RRF fusion (offline, no model required).

Tests:
  · _rrf_fuse: rank fusion math correctness
  · _bm25_score_pair: sparse dot product wiring
  · use_rrf_fusion: feature flag gating
"""
from __future__ import annotations

import os
import sys
import unittest

# Allow running from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


class TestRRFFusion(unittest.TestCase):
    """RRF rank fusion math."""

    def test_rrf_empty_input(self):
        from tsuzuri.memory_core import _rrf_fuse
        self.assertEqual(_rrf_fuse([]), [])
        self.assertEqual(_rrf_fuse([[]]), [])

    def test_rrf_single_rank_list(self):
        from tsuzuri.memory_core import _rrf_fuse
        # Single ranked list of 3 items
        result = _rrf_fuse([["a", "b", "c"]], k=60)
        self.assertEqual(len(result), 3)
        # Order preserved (a first)
        ids = [iid for iid, _ in result]
        self.assertEqual(ids, ["a", "b", "c"])
        # Scores: 1/(60+1) > 1/(60+2) > 1/(60+3)
        scores = [s for _, s in result]
        self.assertAlmostEqual(scores[0], 1 / 61, places=5)
        self.assertAlmostEqual(scores[1], 1 / 62, places=5)
        self.assertAlmostEqual(scores[2], 1 / 63, places=5)

    def test_rrf_two_lists_agree(self):
        from tsuzuri.memory_core import _rrf_fuse
        # Both lists rank "a" first → a should top fused
        result = _rrf_fuse([["a", "b", "c"], ["a", "c", "b"]], k=60)
        ids = [iid for iid, _ in result]
        self.assertEqual(ids[0], "a")
        # b in list1 rank=2, in list2 rank=3 → score = 1/62 + 1/63
        # c in list1 rank=3, in list2 rank=2 → score = 1/63 + 1/62
        # b == c rrf score → either order OK
        self.assertIn(ids[1], ("b", "c"))

    def test_rrf_complementary_signals(self):
        from tsuzuri.memory_core import _rrf_fuse
        # list1 ranks "a" first, "x" last; list2 ranks "x" first, "a" last
        result = _rrf_fuse([
            ["a", "b", "c", "x"],
            ["x", "c", "b", "a"],
        ], k=60)
        ids = [iid for iid, _ in result]
        # Both "a" and "x" top one list、应该都进 top 2
        # b/c 都在中间、也在中间
        score_map = dict(result)
        # a: 1/61 + 1/64
        # x: 1/64 + 1/61
        # b: 1/62 + 1/63
        # c: 1/63 + 1/62
        self.assertAlmostEqual(score_map["a"], 1/61 + 1/64, places=5)
        self.assertAlmostEqual(score_map["x"], 1/64 + 1/61, places=5)

    def test_rrf_skips_empty_ids(self):
        from tsuzuri.memory_core import _rrf_fuse
        result = _rrf_fuse([["a", "", "b"]], k=60)
        # 空字符串被跳过、所以 b 的 rank 应该是 3（不是 2、因为我们用 enumerate start=1）
        # 但实现是 if not item_id: continue、所以 b 拿 rank=3 的分
        ids = [iid for iid, _ in result]
        self.assertEqual(ids, ["a", "b"])

    def test_rrf_k_parameter(self):
        from tsuzuri.memory_core import _rrf_fuse
        # k=1 (extreme): 1/(1+rank) = 0.5, 0.33, 0.25
        result = _rrf_fuse([["a", "b", "c"]], k=1)
        scores = [s for _, s in result]
        self.assertAlmostEqual(scores[0], 0.5, places=5)
        self.assertAlmostEqual(scores[1], 1/3, places=5)


class TestBM25ScorePair(unittest.TestCase):
    """Sparse dot product wiring (without bge-m3 model)."""

    def test_bm25_empty_inputs(self):
        from tsuzuri.memory_core import _bm25_score_pair
        # Empty query or empty item → 0
        self.assertEqual(_bm25_score_pair(None, "any"), 0.0)
        self.assertEqual(_bm25_score_pair({}, "any"), 0.0)
        self.assertEqual(_bm25_score_pair({1: 0.5}, ""), 0.0)
        self.assertEqual(_bm25_score_pair({1: 0.5}, None), 0.0)


class TestFeatureFlag(unittest.TestCase):
    """USE_RRF_FUSION flag respected."""

    def setUp(self):
        self._old = os.environ.get("USE_RRF_FUSION", "")

    def tearDown(self):
        if self._old:
            os.environ["USE_RRF_FUSION"] = self._old
        elif "USE_RRF_FUSION" in os.environ:
            del os.environ["USE_RRF_FUSION"]

    def test_flag_off_by_default(self):
        # Re-import to pick up clean env
        os.environ.pop("USE_RRF_FUSION", None)
        # Need to re-import memory_core to get use_rrf_fusion reading current env
        # But _bool_env reads env at call time so just call
        from tsuzuri.memory_core import use_rrf_fusion
        self.assertFalse(use_rrf_fusion())

    def test_flag_on_when_true(self):
        os.environ["USE_RRF_FUSION"] = "true"
        from tsuzuri.memory_core import use_rrf_fusion
        self.assertTrue(use_rrf_fusion())

    def test_flag_on_variants(self):
        from tsuzuri.memory_core import use_rrf_fusion
        for val in ("true", "1", "yes", "TRUE", "YES"):
            os.environ["USE_RRF_FUSION"] = val
            self.assertTrue(use_rrf_fusion(), f"failed for {val!r}")

    def test_flag_off_variants(self):
        from tsuzuri.memory_core import use_rrf_fusion
        for val in ("false", "0", "no", "off", ""):
            os.environ["USE_RRF_FUSION"] = val
            self.assertFalse(use_rrf_fusion(), f"failed for {val!r}")


class TestSparseSerialization(unittest.TestCase):
    """Sparse encode/decode roundtrip (no model required)."""

    def test_roundtrip(self):
        from tsuzuri.embedding_engine import sparse_to_text, text_to_sparse
        original = {1: 0.5, 42: 0.123, 7: 1.0}
        encoded = sparse_to_text(original)
        decoded = text_to_sparse(encoded)
        self.assertEqual(decoded, original)

    def test_empty(self):
        from tsuzuri.embedding_engine import sparse_to_text, text_to_sparse
        self.assertEqual(sparse_to_text({}), "")
        self.assertEqual(sparse_to_text(None), "")
        self.assertIsNone(text_to_sparse(""))
        self.assertIsNone(text_to_sparse(None))

    def test_invalid_json(self):
        from tsuzuri.embedding_engine import text_to_sparse
        self.assertIsNone(text_to_sparse("not json"))
        self.assertIsNone(text_to_sparse("[1,2,3]"))  # 不是 dict


class TestSparseDotProduct(unittest.TestCase):
    """Sparse dot product math."""

    def test_dot_basic(self):
        from tsuzuri.embedding_engine import sparse_dot_product
        a = {1: 0.5, 2: 1.0, 3: 0.2}
        b = {1: 0.4, 2: 0.5, 4: 0.9}
        # 0.5*0.4 + 1.0*0.5 + 0 + 0 = 0.2 + 0.5 = 0.7
        self.assertAlmostEqual(sparse_dot_product(a, b), 0.7, places=5)

    def test_dot_empty(self):
        from tsuzuri.embedding_engine import sparse_dot_product
        self.assertEqual(sparse_dot_product({}, {1: 0.5}), 0.0)
        self.assertEqual(sparse_dot_product({1: 0.5}, {}), 0.0)
        self.assertEqual(sparse_dot_product(None, None), 0.0)

    def test_dot_no_overlap(self):
        from tsuzuri.embedding_engine import sparse_dot_product
        self.assertEqual(sparse_dot_product({1: 0.5}, {2: 0.5}), 0.0)

    def test_dot_symmetric(self):
        from tsuzuri.embedding_engine import sparse_dot_product
        a = {1: 0.5, 2: 1.0}
        b = {1: 0.4, 2: 0.5}
        self.assertAlmostEqual(
            sparse_dot_product(a, b), sparse_dot_product(b, a), places=5
        )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    unittest.main(verbosity=2)
