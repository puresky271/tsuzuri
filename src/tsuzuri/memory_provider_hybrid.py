"""Hybrid indexed memory provider with full-store dense/sparse/fact lanes."""
from __future__ import annotations

import json
import threading
from typing import Any, Callable

from tsuzuri.memory_core import (
    MemoryCoreStore,
    embedding_row_matches_current,
    lexical_similarity,
    memory_query_analysis,
    memory_explicit_query_has_bounded_slot,
    memory_explicit_row_is_answer_bearing,
    memory_row_evidence_role,
    memory_row_has_durable_turn_signal,
    recall_focus_keyword_score,
    recall_focus_text,
)
from tsuzuri.memory_provider import MemoryProvider
from tsuzuri.persona_registry import resolve_agent


DenseEncoder = Callable[[str], Any]
SparseEncoder = Callable[[str], dict | None]


def _default_dense_encoder(text: str):
    try:
        from tsuzuri.embedding_engine import encode_dense, is_embedding_enabled, is_m3_mode

        if not (is_m3_mode() or is_embedding_enabled()):
            return None
        return encode_dense(text)
    except Exception:
        return None


def _default_sparse_encoder(text: str) -> dict | None:
    try:
        from tsuzuri.embedding_engine import encode_sparse, is_m3_mode

        return encode_sparse(text) if is_m3_mode() else None
    except Exception:
        return None


def _decode_payload(row: dict) -> dict:
    try:
        value = json.loads(str(row.get("payload_json", "") or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _semantic_has_stale_working_metadata(row: dict, payload: dict) -> bool:
    if str(row.get("layer", "") or "").strip().lower() != "semantic":
        return False
    meta = payload.get("memory_meta", {}) if isinstance(payload.get("memory_meta", {}), dict) else {}
    return bool(
        str(row.get("memory_role", "") or "").strip().lower() == "working"
        or str(meta.get("memory_type", "") or "").strip().lower() == "working_turn"
        or str(meta.get("source_kind", "") or "").strip().lower() == "turn_exchange"
    )


class HybridIndexedMemoryStore(MemoryCoreStore):
    """Extend the recent/FTS pool with conservative full-store retrieval lanes."""

    def __init__(
        self,
        db_path: str,
        snapshot_path: str,
        *,
        dense_query_encoder: DenseEncoder | None = None,
        sparse_query_encoder: SparseEncoder | None = None,
        dense_top_k: int = 64,
        sparse_top_k: int = 48,
        structured_top_k: int = 32,
    ):
        super().__init__(db_path=db_path, snapshot_path=snapshot_path)
        self._hybrid_dense_encoder = dense_query_encoder or _default_dense_encoder
        self._hybrid_sparse_encoder = sparse_query_encoder or _default_sparse_encoder
        self._hybrid_dense_top_k = max(8, min(256, int(dense_top_k)))
        self._hybrid_sparse_top_k = max(8, min(256, int(sparse_top_k)))
        self._hybrid_structured_top_k = max(4, min(128, int(structured_top_k)))
        self._hybrid_query_cache: dict[str, tuple[Any, dict | None]] = {}
        self._hybrid_query_cache_lock = threading.RLock()

    def _encode_memory_query_features(self, text: str) -> tuple[Any, dict | None]:
        query = str(text or "").strip()
        if not query:
            return None, None
        with self._hybrid_query_cache_lock:
            cached = self._hybrid_query_cache.get(query)
            if cached is not None:
                return cached
            if self._hybrid_dense_encoder is _default_dense_encoder and self._hybrid_sparse_encoder is _default_sparse_encoder:
                try:
                    from tsuzuri.embedding_engine import encode_both, is_embedding_enabled, is_m3_mode

                    features = encode_both(query) if (is_m3_mode() or is_embedding_enabled()) else (None, None)
                except Exception:
                    features = (None, None)
            else:
                dense = self._hybrid_dense_encoder(query) if callable(self._hybrid_dense_encoder) else None
                sparse = self._hybrid_sparse_encoder(query) if callable(self._hybrid_sparse_encoder) else None
                features = (dense, sparse)
            if features[0] is None and features[1] is None:
                return features
            if len(self._hybrid_query_cache) >= 24:
                self._hybrid_query_cache.clear()
            self._hybrid_query_cache[query] = features
            return features

    def _encode_memory_query_dense(self, text: str):
        return self._encode_memory_query_features(text)[0]

    def _encode_memory_query_sparse(self, text: str):
        return self._encode_memory_query_features(text)[1]

    def _query_all_active_items(self, character: str) -> list[dict]:
        char = resolve_agent(character)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, character, layer, memory_role, engram_type, item_type, text,
                           payload_json, cue_json, lifecycle_json, salience, confidence, novelty,
                           importance, status, source, evidence_id, created_at, updated_at,
                           last_accessed_at, access_count, archived, embedding_text,
                           embedding_sparse_text, embedding_model_id, embedding_dim
                    FROM memory_items
                    WHERE archived=0 AND status='active'
                      AND COALESCE(item_type, '') NOT IN ('core_summary', 'event_cluster', 'topic_cluster')
                      AND (character=? OR character='' OR character='*')
                    ORDER BY updated_at DESC;
                    """,
                    (char,),
                ).fetchall()
            finally:
                conn.close()
        return [{key: row[key] for key in row.keys()} for row in rows]

    def _query_items(self, character: str, limit: int = 220, query_text: str = "", extra_search_texts=()) -> list[dict]:
        base_rows = super()._query_items(character, limit=limit, query_text=query_text,
                                         extra_search_texts=extra_search_texts)
        query = str(query_text or "").strip()
        if len(query) < 2:
            return base_rows

        all_rows = self._query_all_active_items(character)
        analysis = memory_query_analysis(query, {}, "normal")
        explicit = bool(analysis.get("explicit_recall_intent", False))
        focus = recall_focus_text(query) if explicit else query

        query_dense, query_sparse = self._encode_memory_query_features(query)
        default_dense = self._hybrid_dense_encoder is _default_dense_encoder
        default_sparse = self._hybrid_sparse_encoder is _default_sparse_encoder
        dense_threshold = 0.52 if explicit else 0.42

        dense_ranked: list[tuple[float, dict]] = []
        sparse_ranked: list[tuple[float, dict]] = []
        structured_ranked: list[tuple[float, dict]] = []
        for raw in all_rows:
            row = dict(raw)
            text = str(row.get("text", "") or "").strip()
            if not text:
                continue
            importance = float(row.get("importance", 0.0) or 0.0)

            dense_score = 0.0
            dense_row_current = (not default_dense) or embedding_row_matches_current(row)
            if importance > 0.15 and query_dense is not None and dense_row_current and str(row.get("embedding_text", "") or ""):
                try:
                    from tsuzuri.embedding_engine import cosine_similarity, text_to_embedding

                    item_dense = text_to_embedding(
                        str(row.get("embedding_text", "") or ""),
                        expected_dim=len(query_dense),
                    )
                    if item_dense is not None:
                        similarity = cosine_similarity(query_dense, item_dense)
                        if similarity is not None:
                            dense_score = max(0.0, float(similarity))
                except Exception:
                    dense_score = 0.0
            if dense_score >= dense_threshold:
                row["_hybrid_dense_hit"] = True
                row["_hybrid_dense_score"] = round(dense_score, 5)
                dense_ranked.append((dense_score, row))

            sparse_score = 0.0
            sparse_row_current = (not default_sparse) or embedding_row_matches_current(row)
            if importance > 0.15 and query_sparse and sparse_row_current and str(row.get("embedding_sparse_text", "") or ""):
                try:
                    from tsuzuri.embedding_engine import sparse_dot_product, text_to_sparse

                    item_sparse = text_to_sparse(str(row.get("embedding_sparse_text", "") or ""))
                    sparse_score = float(sparse_dot_product(query_sparse, item_sparse)) if item_sparse else 0.0
                except Exception:
                    sparse_score = 0.0
            if sparse_score >= 1.2:
                sparse_row = dict(row)
                sparse_row["_hybrid_sparse_hit"] = True
                sparse_row["_hybrid_sparse_score"] = round(sparse_score, 5)
                sparse_ranked.append((sparse_score, sparse_row))

            payload = _decode_payload(row)
            policy = payload.get("memory_policy", {}) if isinstance(payload.get("memory_policy", {}), dict) else {}
            layer = str(row.get("layer", "") or "").strip().lower()
            confidence = float(row.get("confidence", 0.0) or 0.0)
            stable_semantic = bool(
                layer == "semantic"
                and (
                    bool(policy.get("manual_promote", False))
                    or str(policy.get("promotion_reason", "") or "") == "explicit_user_memory_request"
                    or confidence >= 0.72
                    or (confidence >= 0.64 and importance >= 0.55)
                )
            )
            manual_promoted = bool(layer in ("semantic", "episodic") and policy.get("manual_promote", False))
            durable = bool(
                memory_row_has_durable_turn_signal(row, payload)
                or memory_row_evidence_role(row) == "correction"
            )
            lexical = lexical_similarity(focus, text)
            focus_keyword = recall_focus_keyword_score(focus, text)
            structured_score = max(lexical, focus_keyword, dense_score)
            slot_answer = bool(
                explicit
                and memory_explicit_query_has_bounded_slot(query)
                and memory_explicit_row_is_answer_bearing(
                    {
                        **row,
                        "embedding_score": dense_score,
                    },
                    query,
                )
            )
            if (stable_semantic or manual_promoted or durable) and (
                lexical >= 0.10
                or focus_keyword >= 0.18
                or dense_score >= dense_threshold
                or slot_answer
            ):
                structured_row = dict(row)
                structured_row["_hybrid_structured_hit"] = True
                structured_row["_hybrid_structured_score"] = round(structured_score, 5)
                if _semantic_has_stale_working_metadata(row, payload):
                    structured_row["_hybrid_promoted_layer"] = True
                structured_ranked.append((structured_score, structured_row))

        lane_rows = [row for _, row in sorted(dense_ranked, key=lambda pair: pair[0], reverse=True)[: self._hybrid_dense_top_k]]
        lane_rows.extend(row for _, row in sorted(sparse_ranked, key=lambda pair: pair[0], reverse=True)[: self._hybrid_sparse_top_k])
        lane_rows.extend(row for _, row in sorted(structured_ranked, key=lambda pair: pair[0], reverse=True)[: self._hybrid_structured_top_k])

        base_ids = {str(row.get("item_id", "") or "").strip() for row in base_rows if isinstance(row, dict)}
        merged: dict[str, dict] = {}
        order: list[str] = []
        for row in list(base_rows) + lane_rows:
            item_id = str(row.get("item_id", "") or "").strip()
            if not item_id:
                continue
            if item_id not in merged:
                merged[item_id] = dict(row)
                order.append(item_id)
            else:
                for key, value in row.items():
                    if key.startswith("_hybrid_") or key == "_fts_hit":
                        merged[item_id][key] = value
        result = [merged[item_id] for item_id in order]
        if result:
            from tsuzuri.memory_entities import annotate_entity_boosts, mem_entity_boost_enabled

            if mem_entity_boost_enabled():
                annotate_entity_boosts(self, character, query, result)
        for row in result:
            # Full-store expansion must never trigger per-item model inference on the reply path.
            if str(row.get("item_id", "") or "").strip() not in base_ids:
                row["_skip_embedding_backfill"] = True
        return result


class HybridIndexedMemoryProvider(MemoryProvider):
    """Provider facade for the hybrid v2 store."""

    def __init__(
        self,
        *,
        db_path: str | None = None,
        snapshot_path: str | None = None,
        dense_query_encoder: DenseEncoder | None = None,
        sparse_query_encoder: SparseEncoder | None = None,
    ):
        self._db_path = db_path
        self._snapshot_path = snapshot_path
        self._dense_query_encoder = dense_query_encoder
        self._sparse_query_encoder = sparse_query_encoder
        self._store: HybridIndexedMemoryStore | None = None
        self._store_lock = threading.RLock()

    @property
    def name(self) -> str:
        return "hybrid_v2"

    def _get_store(self) -> HybridIndexedMemoryStore:
        with self._store_lock:
            if self._store is None:
                from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH

                self._store = HybridIndexedMemoryStore(
                    db_path=self._db_path or MEMORY_CORE_DB_PATH,
                    snapshot_path=self._snapshot_path or DEV_MEMORY_PATH,
                    dense_query_encoder=self._dense_query_encoder,
                    sparse_query_encoder=self._sparse_query_encoder,
                )
            return self._store

    def build_memory_context_pack(
        self,
        character: str,
        query_text: str,
        query_bundle: dict | None = None,
        max_items: int = 6,
        *,
        llm_selector: Any = None,
        world_signals: dict | None = None,
        risk_bundle: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        result = self._get_store().build_memory_context_pack(
            character,
            query_text,
            query_bundle=query_bundle,
            max_items=max_items,
            llm_selector=llm_selector,
            world_signals=world_signals,
            risk_bundle=risk_bundle,
        )
        pack = dict(result) if isinstance(result, dict) else {}
        try:
            from tsuzuri.memory_entities import (
                evaluate_association_hop1_shadow,
                mem_assoc_hop1_shadow_enabled,
            )

            if mem_assoc_hop1_shadow_enabled():
                candidate_rows = pack.get("filtered_memory_rows", [])
                if not isinstance(candidate_rows, list):
                    candidate_rows = []
                pack["association_hop1_shadow"] = evaluate_association_hop1_shadow(
                    self._get_store(),
                    character,
                    query_text,
                    [dict(row) for row in candidate_rows if isinstance(row, dict)],
                    retrieval_run_id=str(pack.get("retrieval_run_id", "") or ""),
                    persist=True,
                )
        except Exception as exc:
            pack["association_hop1_shadow"] = {
                "status": "error",
                "error": f"{type(exc).__name__}:{exc}"[:160],
            }
        return pack


__all__ = ["HybridIndexedMemoryProvider", "HybridIndexedMemoryStore"]
