"""Per-turn semantic scorer for the fact workspace (offline eval 2026-08-18).

Eval evidence (`_offline_smoke_out/semantic_channel_eval/`, 150 real queries x
726 candidates): cosine >= 0.65 on lexically blocked rows is ~65% precision and
almost all commitment/open_thread rows that paraphrase misses; weak-overlap
selected rows below cos 0.40 are mostly noise. The scorer here feeds exactly
those two decisions inside `turn_context_workspace.compile_fact_workspace`.

Design rules:
- Reuses the already-warm bge-m3 singleton from `embedding_engine` (same model
  the memory provider uses); never cold-loads inside a turn unless
  TURN_FACT_SEMANTIC_ALLOW_COLD_LOAD=1 (dump/analysis use).
- Candidate vectors are cached by text hash so the ~240 mostly-static
  candidates are encoded once across turns.
- Any failure (no model, encode error, numpy missing) degrades to scorer=None,
  which leaves the workspace on the lexical-only path.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Any, Callable

_CACHE_MAX = 2048
_vector_cache: OrderedDict[str, Any] = OrderedDict()

SemanticScorer = Callable[[list[str]], "list[float | None]"]


def semantic_channel_enabled() -> bool:
    raw = str(os.environ.get("TURN_FACT_SEMANTIC_ENABLE", "1") or "").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _allow_cold_load() -> bool:
    raw = str(os.environ.get("TURN_FACT_SEMANTIC_ALLOW_COLD_LOAD", "0") or "").strip().lower()
    return raw in ("1", "true", "on", "yes")


def _cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _encode_texts(texts: list[str]) -> "list[Any | None]":
    """Dense vectors for texts, using the hash cache; batch-encodes misses."""
    import numpy as np  # local import: numpy absence must not break the module

    from tsuzuri import embedding_engine as embedding_engine

    out: list[Any | None] = [None] * len(texts)
    missing: list[tuple[int, str, str]] = []
    for idx, text in enumerate(texts):
        key = _cache_key(text)
        if key in _vector_cache:
            vec = _vector_cache.pop(key)
            _vector_cache[key] = vec
            out[idx] = vec
        else:
            missing.append((idx, key, text))
    if missing:
        pairs = embedding_engine.encode_both_batch([text for _, _, text in missing])
        for (idx, key, _), pair in zip(missing, pairs or []):
            dense = pair[0] if pair else None
            if dense is None:
                continue
            vec = np.asarray(dense, dtype=np.float32)
            _vector_cache[key] = vec
            _vector_cache.move_to_end(key)
            out[idx] = vec
        while len(_vector_cache) > _CACHE_MAX:
            _vector_cache.popitem(last=False)
    return out


def build_turn_semantic_scorer(query: str) -> SemanticScorer | None:
    """Return a cosine scorer against query, or None when semantics unavailable."""
    query = str(query or "").strip()
    if not query or not semantic_channel_enabled():
        return None
    try:
        import numpy as np

        from tsuzuri import embedding_engine as embedding_engine

        # Never pay the 30s+ cold model load inside a chat turn; the memory
        # provider keeps the model warm in the server process anyway. Check
        # readiness before is_m3_mode(), whose import probe is also expensive.
        if getattr(embedding_engine, "_m3_model_instance", None) is None and not _allow_cold_load():
            return None
        if not embedding_engine.is_m3_mode():
            return None
        query_vec = _encode_texts([query])[0]
        if query_vec is None:
            return None
        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm <= 0.0:
            return None
    except Exception:
        return None

    def _score(texts: list[str]) -> "list[float | None]":
        try:
            vectors = _encode_texts([str(t or "") for t in texts])
            scores: list[float | None] = []
            for vec in vectors:
                if vec is None:
                    scores.append(None)
                    continue
                denom = float(np.linalg.norm(vec)) * query_norm
                scores.append(float(np.dot(vec, query_vec) / denom) if denom > 0.0 else None)
            return scores
        except Exception:
            return [None] * len(texts)

    return _score
