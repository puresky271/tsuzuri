"""Reranker Engine: Cross-encoder reranker for RAG top-K rerank.

Phase C (2026-05-14) of memory retrieval upgrade plan.
Replaces LLM-based reranker with FlagEmbedding's bge-reranker-v2-m3.

Cross-encoder takes (query, doc) pair as joint input → outputs single relevance
score. Slower than bi-encoder embedding but much more accurate (typical +5-10pp
NDCG vs cosine-only). For top-K rerank (K=16) total latency ~800ms (CPU),
vs LLM reranker 1-2s + API cost.

Environment variables:
  USE_CROSS_ENCODER_RERANK=true     — enable Phase C reranker
  RERANKER_MODEL_PATH=...           — model path (default BAAI/bge-reranker-v2-m3)
  RERANKER_TOP_K=16                 — top-K candidates to rerank (default 16)
  RERANKER_USE_FP16=false           — fp16 inference (GPU only、CPU 强制 fp32)

Design:
  · Lazy load + 60s cooldown on failure (与 embedding_engine 一致)
  · Graceful fallback: 失败时返回 None、调用方自动 fallback 到原 LLM rerank
  · 模型路径优先级：RERANKER_MODEL_PATH > ModelScope cache > HF cache
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import os
import threading
import time
from typing import Optional


_RERANKER_FLAG_KEY = "USE_CROSS_ENCODER_RERANK"
_RERANKER_PATH_KEY = "RERANKER_MODEL_PATH"
_RERANKER_TOP_K_KEY = "RERANKER_TOP_K"
_RERANKER_FP16_KEY = "RERANKER_USE_FP16"
_DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
_DEFAULT_TOP_K = 16

_reranker_lock = threading.Lock()
_reranker_instance = None
_reranker_last_fail: float = 0.0
_RETRY_COOLDOWN = 60.0

_flag_embedding_importable: Optional[bool] = None
_import_error: str = ""


def _check_flag_embedding() -> bool:
    """One-shot import probe."""
    global _flag_embedding_importable, _import_error
    if _flag_embedding_importable is not None:
        return _flag_embedding_importable
    try:
        import importlib
        importlib.import_module("FlagEmbedding")
        _flag_embedding_importable = True
        _import_error = ""
    except Exception as _err:
        _flag_embedding_importable = False
        _import_error = f"{type(_err).__name__}: {_err}"
        _tsz_emit(f"[Reranker] FlagEmbedding unavailable: {_import_error}", flush=True)
    return _flag_embedding_importable


def is_reranker_enabled() -> bool:
    """Check USE_CROSS_ENCODER_RERANK=true + FlagEmbedding importable."""
    explicit = os.getenv(_RERANKER_FLAG_KEY, "").strip().lower()
    if explicit not in ("true", "1", "yes"):
        return False
    return _check_flag_embedding()


def is_reranker_ready() -> bool:
    """Return whether inference can run without loading a model on the reply thread."""
    return _reranker_instance is not None


def get_reranker():
    """Lazy load FlagReranker. Returns None on failure (with cooldown)."""
    global _reranker_instance, _reranker_last_fail
    if _reranker_instance is not None:
        return _reranker_instance
    if _reranker_last_fail > 0 and (time.monotonic() - _reranker_last_fail) < _RETRY_COOLDOWN:
        return None
    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        if _reranker_last_fail > 0 and (time.monotonic() - _reranker_last_fail) < _RETRY_COOLDOWN:
            return None
        try:
            # 2026-05-14: FlagEmbedding 1.4.0 calls tokenizer.prepare_for_model()
            # which was removed in transformers >= 5.x (current: 5.5.3).
            # Bridge: decode token ids → re-encode via __call__ (modern API).
            import transformers
            _tok = transformers.XLMRobertaTokenizer
            if not hasattr(_tok, "prepare_for_model"):
                def _prepare_for_model_patch(self, ids, pair_ids=None, **kw):
                    text = self.decode(ids, skip_special_tokens=False)
                    if pair_ids:
                        pair_text = self.decode(pair_ids, skip_special_tokens=False)
                        return self(text, text_pair=pair_text, **kw)
                    return self(text, **kw)
                _tok.prepare_for_model = _prepare_for_model_patch
            from FlagEmbedding import FlagReranker
            model_path = os.getenv(_RERANKER_PATH_KEY, _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
            use_fp16 = os.getenv(_RERANKER_FP16_KEY, "false").strip().lower() in ("true", "1", "yes")
            _reranker_instance = FlagReranker(model_path, use_fp16=use_fp16)
            _reranker_last_fail = 0.0
            _tsz_emit(f"[Reranker] loaded {model_path} (fp16={use_fp16})", flush=True)
            return _reranker_instance
        except Exception as e:
            _reranker_last_fail = time.monotonic()
            _tsz_emit(f"[Reranker] model load failed (retry in {_RETRY_COOLDOWN:.0f}s): {e}", flush=True)
            return None



def force_retry() -> bool:
    """Clear failure state and force reload."""
    global _reranker_instance, _reranker_last_fail, _flag_embedding_importable
    with _reranker_lock:
        _reranker_last_fail = 0.0
        _reranker_instance = None
        _flag_embedding_importable = None
    return get_reranker() is not None


def get_top_k() -> int:
    raw = os.getenv(_RERANKER_TOP_K_KEY, "").strip()
    if not raw:
        return _DEFAULT_TOP_K
    try:
        v = int(raw)
        return max(2, min(64, v))
    except Exception:
        return _DEFAULT_TOP_K


def rerank_pairs(query: str, docs: list[str]) -> Optional[list[float]]:
    """Compute relevance scores for (query, doc) pairs.

    Returns:
      list[float] — relevance scores per doc (logit-style、range ~[-15, +15])
      None — reranker disabled / load failed / inference error

    Caller responsibility:
      · normalize / sort by score
      · graceful fallback when result is None
    """
    if not is_reranker_enabled():
        return None
    if not query or not docs:
        return None
    model = get_reranker()
    if model is None:
        return None
    try:
        # Truncate to avoid OOM on very long docs
        q = str(query)[:512]
        truncated_docs = [str(d or "")[:1024] for d in docs]
        pairs = [(q, d) for d in truncated_docs]
        scores = model.compute_score(pairs, normalize=False)
        # FlagReranker 单条返回 float、多条返回 list[float]
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(s) for s in scores]
    except Exception as e:
        _tsz_emit(f"[Reranker] inference failed: {e}", flush=True)
        return None


def rerank_candidates(
    query: str,
    candidates: list[dict],
    text_field: str = "text",
    top_k: Optional[int] = None,
) -> Optional[list[dict]]:
    """High-level: rerank candidate dicts by cross-encoder.

    Args:
      query: query text
      candidates: list of dicts (each with `text_field`)
      text_field: which field to use as doc (default "text")
      top_k: limit to top-K results (default: env RERANKER_TOP_K)

    Returns:
      Sorted list of dicts (each augmented with `reranker_score`)、或 None on failure。
      Caller can use directly or fall back when None.
    """
    if not is_reranker_enabled():
        return None
    if not candidates:
        return []
    limit = top_k if top_k is not None else get_top_k()
    # Only rerank top-N to keep latency bounded
    pool = candidates[:limit]
    docs = [str(c.get(text_field, "") or "") for c in pool]
    scores = rerank_pairs(query, docs)
    if scores is None:
        return None
    augmented = []
    for cand, score in zip(pool, scores):
        c = dict(cand)
        c["reranker_score"] = float(score)
        augmented.append(c)
    augmented.sort(key=lambda c: -c.get("reranker_score", 0.0))
    return augmented


# ── self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    _tsz_emit(f"[selftest] USE_CROSS_ENCODER_RERANK={os.getenv(_RERANKER_FLAG_KEY)}")
    _tsz_emit(f"[selftest] FlagEmbedding importable: {_check_flag_embedding()}")
    _tsz_emit(f"[selftest] enabled: {is_reranker_enabled()}")
    _tsz_emit(f"[selftest] top-K: {get_top_k()}")

    if not is_reranker_enabled():
        _tsz_emit("[selftest] Reranker disabled、skip inference test.")
        _tsz_emit("           启用: setx USE_CROSS_ENCODER_RERANK true  (重开 shell)")
        sys.exit(0)

    cases = [
        ("梦境系统", "梦境巩固机制"),  # 应该高
        ("梦境系统", "今天去吃了拉面"),  # 应该低
        ("周末去爬山", "上周末我们一起去爬了山"),  # 应该高
        ("周末去爬山", "晚饭吃了咖喱饭"),  # 应该低
    ]
    qs = [q for q, _ in cases]
    docs = [d for _, d in cases]
    _tsz_emit("\nRunning reranker on 4 pairs...")
    for q, d in cases:
        scores = rerank_pairs(q, [d])
        if scores:
            _tsz_emit(f"  query='{q}' / doc='{d}' → score={scores[0]:.3f}")
        else:
            _tsz_emit(f"  query='{q}' / doc='{d}' → FAILED")
