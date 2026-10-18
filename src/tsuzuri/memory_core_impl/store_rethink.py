"""Store mixin: R.1 rethink evidence/candidates, actions, runs and core summaries.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import uuid
from datetime import timedelta
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import (
    _safe_float, _safe_int, _utc_now_iso, mem_rethink_min_shadow_days, mem_rethink_shadow_only,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _lexical_similarity, _memory_row_is_trusted_fact_observation,
)
from tsuzuri.memory_core_impl.features import _extract_entities_from_text, _fact_is_dialogue_residue
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads, _parse_dt
from tsuzuri.persona_registry import resolve_agent

class _RethinkMixin:
    """R.1 rethink evidence/candidates, actions, runs and core summaries."""


    # ═══════════════════════════════════════════════════════════════════
    # Phase R.1 (2026-05-16) — LLM-driven memory rethink helpers
    # ═══════════════════════════════════════════════════════════════════
    # 这一组方法是 Letta sleeptime + memory_rethink 在 宿主应用 的对应物。
    # 总体流程（KAIROS dream cycle 凌晨触发）：
    #
    #   select_rethink_candidates(char, k=15)
    #         ↓ 返回 candidates（semantic 层、高决策价值 + 高冗余 cluster）
    #   memory_rethink_engine.propose_memory_rethink(char, candidates)
    #         ↓ LLM 输出 actions
    #   _record_rethink_run_open(...) → run_id
    #         ↓
    #   for action in actions:
    #       validate_action(action) → ok / scope_invalid / canon_violation
    #       score_action_ooc(action.new_text, char) → (ooc_score, verdict)
    #       _record_rethink_action(run_id, ...) → action_pk
    #       if validation == ok and ooc_verdict != reject:
    #           apply_rethink_action(action_pk, shadow=...)
    #   _record_rethink_run_complete(run_id, applied_n, rejected_n, ...)
    #
    # revert_rethink_run 用于 audit 异常时整批回滚（脚本侧调用）。
    #
    # 三类 action 的语义（与 evolution_plan 区分）：
    #   rewrite — 改 memory_items.text + 重写 embedding_text、保留 item_id
    #   merge   — 多条 archive、status='merged_by_rethink' + 创建 1 条新 item 承载 new_text
    #   archive — 直接 status='archived_by_rethink' + archived=1
    #
    # 与 propose_memory_evolution 的分工：
    #   evolution = 规则启发式调权重 / 层级 (promote/demote/archive/drop/merge)
    #   rethink   = LLM 改文本 (rewrite/merge text/archive with reason)

    def _effective_rethink_shadow_mode(self) -> tuple[bool, dict]:
        """Mirror of _effective_shadow_mode 但用 MEM_RETHINK_* env vars。

        shadow=True 时所有 action 仅记 audit、不真改 memory_items。
        可选的强制 shadow 窗口（默认 0 天）允许紧急回退后保留观察期。
        """
        now_dt = utc_now_naive()
        base_shadow = bool(mem_rethink_shadow_only())
        min_days = max(0, int(mem_rethink_min_shadow_days()))
        meta = self._get_meta_json("rethink_shadow_window", {})
        if not isinstance(meta, dict):
            meta = {}
        started_at = _parse_dt(str(meta.get("started_at", "") or ""))
        if started_at is None:
            started_at = now_dt
            self._set_meta_json(
                "rethink_shadow_window",
                {"started_at": _utc_now_iso(), "min_days": min_days},
            )
        elapsed_days = max(0.0, (now_dt - started_at).total_seconds() / 86400.0)
        forced_shadow = bool(elapsed_days < float(min_days))
        return (
            bool(base_shadow or forced_shadow),
            {
                "base_shadow": base_shadow,
                "forced_shadow": forced_shadow,
                "elapsed_days": round(elapsed_days, 3),
                "min_days": int(min_days),
            },
        )

    def select_rethink_evidence(
        self,
        character: str,
        *,
        k: int = 24,
        recent_days: float = 21.0,
        since_ts: str = "",
    ) -> list[dict]:
        """Select recent working/episodic rows as read-only rethink evidence."""
        char = resolve_agent(character)
        kk = max(1, min(80, int(k or 24)))
        cutoff = str(since_ts or "").strip()
        cutoff_operator = ">" if cutoff else ">="
        if not cutoff:
            cutoff = (
                utc_now_naive()
                - timedelta(days=max(1.0, float(recent_days or 21.0)))
            ).replace(microsecond=0).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT item_id, layer, text, payload_json, confidence, source,
                           engram_type, item_type, salience, access_count, created_at, updated_at
                    FROM memory_items
                    WHERE archived=0 AND status='active' AND character=?
                          AND layer IN ('working', 'episodic')
                          AND updated_at{cutoff_operator}? AND text!=''
                    ORDER BY updated_at DESC
                    LIMIT 180;
                    """,
                    (char, cutoff),
                ).fetchall()
            finally:
                conn.close()

        evidence: list[dict] = []
        for row in rows:
            item = {key: row[key] for key in row.keys()}
            payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
            payload = payload if isinstance(payload, dict) else {}
            meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
            memory_meta = payload.get("memory_meta", {}) if isinstance(payload.get("memory_meta", {}), dict) else {}
            if str(memory_meta.get("mode_scope", "normal") or "normal").strip().lower().startswith("guest"):
                continue
            card = payload.get("turn_fact_card", {}) if isinstance(payload.get("turn_fact_card", {}), dict) else {}
            if not card and isinstance(meta.get("turn_fact_card", {}), dict):
                card = meta.get("turn_fact_card", {})
            if str(card.get("audience_scope", "") or "").strip().lower().startswith("guest"):
                continue

            evidence_text = ""
            evidence_role = "observation"
            corrections = card.get("corrections", []) if isinstance(card.get("corrections", []), list) else []
            commitments = card.get("commitments", []) if isinstance(card.get("commitments", []), list) else []
            if corrections:
                evidence_text = "；".join(str(x or "").strip() for x in corrections[:3] if str(x or "").strip())
                evidence_role = "correction"
            elif str(item.get("engram_type", "") or "") == "fact_observation":
                evidence_text = str(payload.get("fact", "") or item.get("text", "") or "")
                evidence_role = "fact_observation"
            elif commitments:
                evidence_text = "；".join(str(x or "").strip() for x in commitments[:3] if str(x or "").strip())
                evidence_role = "commitment"
            elif str(card.get("state_delta", "") or "").strip():
                evidence_text = str(card.get("state_delta", "") or "").strip()
                evidence_role = "state_update"
            elif str(card.get("fact", "") or "").strip():
                evidence_text = str(card.get("fact", "") or "").strip()
                evidence_role = "fact_observation"
            elif str(item.get("layer", "") or "") == "episodic":
                evidence_text = str(item.get("text", "") or "")

            evidence_text = re.sub(r"\s+", " ", evidence_text).strip()[:320]
            if not evidence_text:
                continue
            canonical_key = str(
                card.get("canonical_key", "")
                or payload.get("canonical_key", "")
                or meta.get("canonical_key", "")
                or ""
            ).strip()[:120]
            role_weight = {
                "correction": 5.0,
                "fact_observation": 3.0,
                "commitment": 2.5,
                "state_update": 2.0,
                "observation": 1.0,
            }.get(evidence_role, 1.0)
            trusted_fact = _memory_row_is_trusted_fact_observation(
                item,
                payload,
            )
            evidence.append(
                {
                    "item_id": str(item.get("item_id", "") or ""),
                    "text": evidence_text,
                    "layer": str(item.get("layer", "") or ""),
                    "source": str(item.get("source", "") or ""),
                    "confidence": _safe_float(item.get("confidence", 0.5), 0.5),
                    "created_at": str(item.get("created_at", "") or ""),
                    "updated_at": str(item.get("updated_at", "") or ""),
                    "canonical_key": canonical_key,
                    "evidence_role": evidence_role,
                    "turn_card_id": str(card.get("card_id", card.get("id", "")) or "")[:120],
                    "salience": _safe_float(item.get("salience", 0.0), 0.0),
                    "access_count": _safe_int(item.get("access_count", 0), 0),
                    "trusted_fact": bool(trusted_fact),
                    "selection_score": role_weight,
                }
            )
        evidence.sort(
            key=lambda item: (float(item.get("selection_score", 0.0)), str(item.get("updated_at", ""))),
            reverse=True,
        )
        return evidence[:kk]

    def select_rethink_candidates(
        self,
        character: str,
        *,
        k: int = 15,
        redundancy_threshold: float = 0.70,
        evidence: list[dict] | None = None,
        protected_item_types: tuple[str, ...] = ("core_summary",),
    ) -> list[dict]:
        """选 semantic 层的 rethink 候选（高决策价值 + 高冗余 cluster）。

        两路候选合并：
          A) 高决策价值：importance 在 [0.40, 0.75] 中间区 + access_count >= 3。
             不高不低、最需要 LLM 介入纠正。高 importance 不动（已稳定），
             低 importance 让规则衰减处理。
          B) 高冗余 cluster：lexical_similarity >= threshold 的 cluster head
             (代表性最高那条) + 完整 member_ids。LLM 主要做 merge。

        返回 list[dict]：每个候选含 item_id / text / salience / importance /
        confidence / access_count / created_at / cluster_member_ids /
        cluster_size / score_breakdown，按混合分数降序。
        """
        char = resolve_agent(character)
        kk = max(1, min(50, int(k or 15)))
        protected_types = tuple(
            str(value or "").strip()
            for value in (*tuple(protected_item_types or ()), "event_cluster", "topic_cluster")
            if str(value or "").strip()
        )
        protected_sql = ""
        params: list = [char]
        if protected_types:
            protected_sql = f" AND item_type NOT IN ({','.join(['?'] * len(protected_types))})"
            params.extend(protected_types)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    f"""
                    SELECT item_id, text, payload_json, salience, importance, confidence, novelty,
                           access_count, created_at, updated_at, last_accessed_at
                    FROM memory_items
                    WHERE archived=0 AND status='active'
                          AND character=? AND layer='semantic'
                          AND text != ''
                          {protected_sql}
                    ORDER BY updated_at DESC
                    LIMIT 200;
                    """,
                    tuple(params),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            return []

        items: list[dict] = []
        for r in rows:
            items.append({
                "item_id": str(r["item_id"]),
                "text": str(r["text"] or ""),
                "salience": _safe_float(r["salience"], 0.5),
                "importance": _safe_float(r["importance"], 0.5),
                "confidence": _safe_float(r["confidence"], 0.5),
                "novelty": _safe_float(r["novelty"], 0.5),
                "access_count": int(_safe_int(r["access_count"], 0)),
                "created_at": str(r["created_at"] or ""),
                "updated_at": str(r["updated_at"] or ""),
                "last_accessed_at": str(r["last_accessed_at"] or ""),
                "canonical_key": str(
                    (_json_loads(str(r["payload_json"] or "{}"), {}) or {}).get("canonical_key", "") or ""
                )[:120],
            })

        # ── A 路：高决策价值评分 ──
        # 中间 importance + 高访问 → 急需介入
        decision_scores: dict[str, float] = {}
        for it in items:
            iid = it["item_id"]
            imp = it["importance"]
            acc = it["access_count"]
            # 三角函数式中区奖励：imp ∈ [0.40, 0.75] 时奖励最高
            if 0.40 <= imp <= 0.75:
                mid_bonus = 1.0 - abs(imp - 0.575) / 0.175  # peak at 0.575
            else:
                mid_bonus = 0.0
            # access boost: 越被频繁用、纠正收益越大
            acc_boost = min(1.0, math.log1p(acc) / math.log(10.0))  # 0 at acc=0, ~1 at acc=9
            # 低 confidence 加分：本身就不确定的、LLM 修正空间大
            low_conf_boost = max(0.0, 0.55 - it["confidence"]) / 0.55
            decision_scores[iid] = round(0.55 * mid_bonus + 0.30 * acc_boost + 0.15 * low_conf_boost, 4)

        # ── B 路：高冗余 cluster 检测 ──
        # O(n^2) lexical_similarity、n=200 时 ~20k 比较、可接受
        cluster_map: dict[str, list[str]] = {}  # head_iid -> member_iids
        cluster_head_of: dict[str, str] = {}    # iid -> head_iid
        for i, it_i in enumerate(items):
            iid_i = it_i["item_id"]
            if iid_i in cluster_head_of:
                continue
            cluster_map[iid_i] = [iid_i]
            cluster_head_of[iid_i] = iid_i
            for j in range(i + 1, len(items)):
                it_j = items[j]
                iid_j = it_j["item_id"]
                if iid_j in cluster_head_of:
                    continue
                sim = _lexical_similarity(it_i["text"], it_j["text"])
                if sim >= redundancy_threshold:
                    cluster_map[iid_i].append(iid_j)
                    cluster_head_of[iid_j] = iid_i

        # cluster 评分：cluster_size >= 2 才计、size 越大分越高
        cluster_scores: dict[str, float] = {}
        for head_iid, members in cluster_map.items():
            csize = len(members)
            if csize < 2:
                continue
            # cluster score: size scaled + head importance 加权
            head_imp = next((it["importance"] for it in items if it["item_id"] == head_iid), 0.5)
            cluster_scores[head_iid] = round(0.70 * min(1.0, (csize - 1) / 4.0) + 0.30 * head_imp, 4)

        # ── C 路：与最近只读证据匹配 ──
        evidence_rows = [dict(x) for x in (evidence or []) if isinstance(x, dict)]
        evidence_match_scores: dict[str, float] = {}
        evidence_match_ids: dict[str, list[str]] = {}
        for item in items:
            best_score = 0.0
            best_ids: list[str] = []
            item_entities = set(_extract_entities_from_text(str(item.get("text", "") or "")))
            for ev in evidence_rows:
                ev_text = str(ev.get("text", "") or "")
                score = _lexical_similarity(str(item.get("text", "") or ""), ev_text)
                item_key = str(item.get("canonical_key", "") or "")
                ev_key = str(ev.get("canonical_key", "") or "")
                if item_key and ev_key and item_key == ev_key:
                    score = 1.0
                elif item_entities:
                    ev_entities = set(_extract_entities_from_text(ev_text))
                    if item_entities.intersection(ev_entities):
                        score = max(score, 0.55)
                if score > best_score:
                    best_score = score
                    best_ids = [str(ev.get("item_id", "") or "")]
                elif score == best_score and score > 0:
                    best_ids.append(str(ev.get("item_id", "") or ""))
            evidence_match_scores[str(item.get("item_id", "") or "")] = round(best_score, 4)
            evidence_match_ids[str(item.get("item_id", "") or "")] = [x for x in best_ids if x][:4]

        # ── 合并打分 ──
        # final_score = 0.55 * decision_score + 0.45 * cluster_score
        # cluster head 双倍受益、独立 item 仅 decision_score
        final_scores: dict[str, float] = {}
        for iid in {it["item_id"] for it in items}:
            ds = decision_scores.get(iid, 0.0)
            cs = cluster_scores.get(iid, 0.0)
            base_score = 0.55 * ds + 0.45 * cs
            evidence_score = evidence_match_scores.get(iid, 0.0)
            final_scores[iid] = round(
                (0.45 * base_score + 0.55 * evidence_score) if evidence_rows else base_score,
                4,
            )

        # 排序、选 top-k
        ranked = sorted(items, key=lambda it: final_scores.get(it["item_id"], 0.0), reverse=True)
        picked = ranked[:kk]

        # 输出候选 + 附 cluster 信息
        out: list[dict] = []
        seen_in_cluster: set[str] = set()
        for it in picked:
            iid = it["item_id"]
            if iid in seen_in_cluster:
                continue
            head_iid = cluster_head_of.get(iid, iid)
            cluster_members = cluster_map.get(head_iid, [iid])
            # 候选展示用 head item 的信息、但带全部 cluster member ids
            head_item = next((x for x in items if x["item_id"] == head_iid), it)
            for m_iid in cluster_members:
                seen_in_cluster.add(m_iid)
            # cluster member texts
            cluster_member_texts: list[dict] = []
            for m_iid in cluster_members:
                m_item = next((x for x in items if x["item_id"] == m_iid), None)
                if m_item:
                    cluster_member_texts.append({
                        "item_id": m_iid,
                        "text": m_item["text"],
                        "importance": m_item["importance"],
                        "access_count": m_item["access_count"],
                    })
            out.append({
                "item_id": head_item["item_id"],
                "text": head_item["text"],
                "salience": head_item["salience"],
                "importance": head_item["importance"],
                "confidence": head_item["confidence"],
                "novelty": head_item["novelty"],
                "access_count": head_item["access_count"],
                "created_at": head_item["created_at"],
                "updated_at": head_item["updated_at"],
                "last_accessed_at": head_item["last_accessed_at"],
                "cluster_member_ids": cluster_members,
                "cluster_size": len(cluster_members),
                "cluster_members": cluster_member_texts if len(cluster_members) > 1 else [],
                "score_breakdown": {
                    "decision_score": decision_scores.get(head_iid, 0.0),
                    "cluster_score": cluster_scores.get(head_iid, 0.0),
                    "evidence_match_score": evidence_match_scores.get(head_iid, 0.0),
                    "final_score": final_scores.get(head_iid, 0.0),
                },
                "evidence_match_item_ids": evidence_match_ids.get(head_iid, []),
            })
        return out

    @staticmethod
    def _rethink_cursor_key(character: str) -> str:
        return f"rethink.cursor.{str(character or '').strip()}"

    def get_rethink_cursor(self, character: str) -> dict:
        """Return the per-character transcript cursor from meta_state."""
        char = resolve_agent(character)
        value = self.get_meta_state_json(self._rethink_cursor_key(char), {})
        if not isinstance(value, dict):
            value = {}
        return {
            "character": char,
            "last_turn_fact_card_id": str(value.get("last_turn_fact_card_id", "") or ""),
            "last_turn_ts": str(value.get("last_turn_ts", "") or ""),
            "advanced_at": str(value.get("advanced_at", "") or ""),
        }

    def advance_rethink_cursor(
        self,
        character: str,
        *,
        last_turn_fact_card_id: str = "",
        last_turn_ts: str = "",
    ) -> dict:
        """Advance, but never rewind, the per-character transcript cursor."""
        char = resolve_agent(character)
        current = self.get_rethink_cursor(char)
        candidate_ts = str(last_turn_ts or "").strip()
        current_ts = str(current.get("last_turn_ts", "") or "")
        if current_ts and candidate_ts and candidate_ts < current_ts:
            candidate_ts = current_ts
        if not candidate_ts:
            candidate_ts = current_ts
        card_id = str(last_turn_fact_card_id or "").strip()
        if candidate_ts == current_ts and current_ts:
            card_id = str(current.get("last_turn_fact_card_id", "") or card_id)
        payload = {
            "character": char,
            "last_turn_fact_card_id": card_id,
            "last_turn_ts": candidate_ts,
            "advanced_at": _utc_now_iso(),
        }
        self.set_meta_state_json(self._rethink_cursor_key(char), payload)
        return payload

    @staticmethod
    def core_summary_item_id(character: str) -> str:
        return f"core_summary:{resolve_agent(character)}"

    def get_core_summary(self, character: str) -> dict | None:
        """Read the active per-character living summary without recall ranking."""
        char = resolve_agent(character)
        item_id = self.core_summary_item_id(char)
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT item_id, character, layer, memory_role, engram_type, item_type,
                           text, payload_json, salience, confidence, novelty, importance,
                           status, source, evidence_id, created_at, updated_at, archived
                    FROM memory_items
                    WHERE item_id=? AND character=? AND layer='semantic'
                          AND item_type='core_summary' AND status='active' AND archived=0
                    LIMIT 1;
                    """,
                    (item_id, char),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        item = {key: row[key] for key in row.keys()}
        item["payload"] = _json_loads(str(item.pop("payload_json", "") or "{}"), {})
        return item

    def upsert_core_summary(
        self,
        character: str,
        text: str,
        *,
        payload: dict | None = None,
        source: str = "core_summary_rebuild",
        evidence_id: str = "",
    ) -> dict:
        """Replace the single living-summary row for one character."""
        char = resolve_agent(character)
        summary = re.sub(r"\s+", " ", str(text or "")).strip()[:2000]
        if not summary:
            return {"ok": False, "error": "text_required", "character": char}
        now = _utc_now_iso()
        item_id = self.core_summary_item_id(char)
        summary_payload = dict(payload or {}) if isinstance(payload, dict) else {}
        summary_payload.update({
            "owner": "core_summary_rebuild",
            "rebuilt_at": now,
        })
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT created_at FROM memory_items WHERE item_id=? LIMIT 1;",
                    (item_id,),
                ).fetchone()
                created_at = str(existing["created_at"] or now) if existing is not None else now
                self._upsert_item(
                    {
                        "item_id": item_id,
                        "character": char,
                        "layer": "semantic",
                        "memory_role": "semantic",
                        "engram_type": "core_summary",
                        "item_type": "core_summary",
                        "text": summary,
                        "payload_json": _json_dumps(summary_payload),
                        "salience": 0.82,
                        "confidence": 0.72,
                        "novelty": 0.20,
                        "importance": 0.82,
                        "status": "active",
                        "source": str(source or "core_summary_rebuild")[:80],
                        "evidence_id": str(evidence_id or "")[:200],
                        "created_at": created_at,
                        "updated_at": now,
                        "archived": 0,
                    },
                    conn=conn,
                )
                # A rebuilt summary must never retain an embedding of its old text.
                conn.execute(
                    "UPDATE memory_items SET embedding_text='', embedding_sparse_text='', "
                    "embedding_model_id='', embedding_dim=0 WHERE item_id=?;",
                    (item_id,),
                )
                conn.commit()
            finally:
                conn.close()
        return {"ok": True, "item_id": item_id, "character": char, "text": summary}

    def _record_rethink_run_open(
        self,
        character: str,
        candidates: list[dict],
        *,
        shadow: bool,
        evidence: list[dict] | None = None,
        ooc_threshold: float = 0.45,
        protocol_version: str = "rethink_v2",
        loop_meta: dict | None = None,
    ) -> str:
        """开一条 rethink_run、返回 run_id。"""
        run_id = "reth_" + uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        status = "shadow_planned" if shadow else "planned"
        # candidates 落库前去掉冗余 cluster_members detail（节省空间、原 ids 保留）
        cand_dump = []
        for c in candidates:
            cand_dump.append({
                "item_id": c.get("item_id", ""),
                "text": c.get("text", ""),
                "importance": _safe_float(c.get("importance", 0.5), 0.5),
                "confidence": _safe_float(c.get("confidence", 0.5), 0.5),
                "access_count": int(_safe_int(c.get("access_count", 0), 0)),
                "cluster_member_ids": list(c.get("cluster_member_ids", []) or []),
                "cluster_size": int(c.get("cluster_size", 1) or 1),
                "canonical_key": str(c.get("canonical_key", "") or "")[:120],
                "evidence_match_item_ids": list(c.get("evidence_match_item_ids", []) or []),
                "score_breakdown": dict(c.get("score_breakdown", {}) or {}),
            })
        evidence_dump = []
        for item in (evidence or []):
            if not isinstance(item, dict):
                continue
            evidence_dump.append({
                "item_id": str(item.get("item_id", "") or ""),
                "text": str(item.get("text", "") or "")[:320],
                "layer": str(item.get("layer", "") or ""),
                "source": str(item.get("source", "") or ""),
                "confidence": _safe_float(item.get("confidence", 0.5), 0.5),
                "created_at": str(item.get("created_at", "") or ""),
                "canonical_key": str(item.get("canonical_key", "") or "")[:120],
                "evidence_role": str(item.get("evidence_role", "observation") or "observation"),
            })
        candidates_payload: list[dict] | dict = cand_dump
        if isinstance(loop_meta, dict):
            candidates_payload = {
                "initial_candidates": cand_dump,
                "loop_meta": dict(loop_meta),
            }
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO memory_rethink_runs
                    (run_id, character, shadow, status, candidates_count, candidates_json, evidence_json,
                     actions_proposed_count, actions_applied_count, actions_rejected_count,
                     llm_model, llm_total_tokens, llm_raw_response, summary, fallback_reason,
                     ooc_threshold, finish_confirmed, protocol_version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, '', 0, '', '', '', ?, 0, ?, ?, ?);
                    """,
                    (
                        run_id,
                        resolve_agent(character),
                        1 if shadow else 0,
                        status,
                        len(cand_dump),
                        _json_dumps(candidates_payload),
                        _json_dumps(evidence_dump),
                        float(ooc_threshold),
                        str(protocol_version or "rethink_v2")[:40],
                        now,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        return run_id

    def _record_rethink_run_complete(
        self,
        run_id: str,
        *,
        status: str,
        actions_proposed: int,
        actions_applied: int,
        actions_rejected: int,
        llm_model: str = "",
        llm_total_tokens: int = 0,
        llm_raw_response: str = "",
        summary: str = "",
        fallback_reason: str = "",
        finish_confirmed: bool = False,
    ) -> None:
        """收尾 rethink_run、写 LLM meta + 统计。"""
        rid = str(run_id or "").strip()
        if not rid:
            return
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE memory_rethink_runs
                    SET status=?, actions_proposed_count=?, actions_applied_count=?,
                        actions_rejected_count=?, llm_model=?, llm_total_tokens=?,
                        llm_raw_response=?, summary=?, fallback_reason=?, finish_confirmed=?, updated_at=?
                    WHERE run_id=?;
                    """,
                    (
                        str(status or "").strip() or "applied",
                        int(actions_proposed),
                        int(actions_applied),
                        int(actions_rejected),
                        str(llm_model or "")[:120],
                        int(llm_total_tokens),
                        str(llm_raw_response or "")[:8000],  # 防超大 response 撑爆 db
                        str(summary or "")[:500],
                        str(fallback_reason or "")[:200],
                        1 if finish_confirmed else 0,
                        now,
                        rid,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def _record_rethink_action(
        self,
        run_id: str,
        action_index: int,
        action_type: str,
        *,
        target_item_ids: list[str],
        old_texts: list[str],
        resolution_type: str = "",
        evidence_item_ids: list[str] | None = None,
        new_text: str = "",
        reason: str = "",
        ooc_score: float = 0.0,
        ooc_verdict: str = "",
        ooc_features: list[str] | None = None,
        validation_status: str = "pending",
        validation_reason: str = "",
        loop_pass: int = 0,
    ) -> int:
        """写一条 action audit、返回 action_pk。"""
        rid = str(run_id or "").strip()
        if not rid:
            return 0
        now = _utc_now_iso()
        expected_before_hash = hashlib.sha256(
            _json_dumps(list(old_texts or [])).encode("utf-8", errors="ignore")
        ).hexdigest()
        with self._lock:
            conn = self._connect()
            try:
                evidence_ids = [
                    str(value or "").strip()
                    for value in (evidence_item_ids or [])
                    if str(value or "").strip()
                ]
                evidence_before_hash = ""
                if evidence_ids:
                    placeholders = ",".join(["?"] * len(evidence_ids))
                    evidence_rows = conn.execute(
                        f"SELECT item_id, text FROM memory_items WHERE item_id IN ({placeholders});",
                        tuple(evidence_ids),
                    ).fetchall()
                    evidence_by_id = {str(row["item_id"]): str(row["text"] or "") for row in evidence_rows}
                    evidence_before_hash = hashlib.sha256(
                        _json_dumps([evidence_by_id.get(item_id, "") for item_id in evidence_ids]).encode(
                            "utf-8", errors="ignore"
                        )
                    ).hexdigest()
                cur = conn.execute(
                    """
                    INSERT INTO memory_rethink_actions
                    (run_id, action_index, action_type, target_item_ids_json, old_texts_json,
                     resolution_type, evidence_item_ids_json, expected_before_hash, evidence_before_hash, loop_pass,
                     new_text, reason, ooc_score, ooc_verdict, ooc_features_json,
                     validation_status, validation_reason, applied, applied_at,
                     reverted_at, revert_reason, before_json, after_json, apply_status, apply_error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, '', '', '', '{}', '{}', 'pending', '', ?);
                    """,
                    (
                        rid,
                        int(action_index),
                        str(action_type or "").strip().lower(),
                        _json_dumps(list(target_item_ids or [])),
                        _json_dumps(list(old_texts or [])),
                        str(resolution_type or "")[:48],
                        _json_dumps(evidence_ids),
                        expected_before_hash,
                        evidence_before_hash,
                        max(0, int(loop_pass or 0)),
                        str(new_text or "")[:2000],
                        str(reason or "")[:500],
                        float(_clamp(_safe_float(ooc_score, 0.0))),
                        str(ooc_verdict or "")[:40],
                        _json_dumps(list(ooc_features or [])),
                        str(validation_status or "pending")[:40],
                        str(validation_reason or "")[:200],
                        now,
                    ),
                )
                action_pk = int(cur.lastrowid or 0)
                conn.commit()
                return action_pk
            finally:
                conn.close()

    def apply_rethink_action(
        self,
        action_pk: int,
        *,
        shadow: bool = True,
        _conn: sqlite3.Connection | None = None,
        _commit: bool = True,
    ) -> dict:
        """执行单条 rethink action。

        shadow=True：只在 audit 表记 before/after 模拟、不动 memory_items。
        shadow=False：真改 memory_items + 记 before/after 真实快照。

        返回 dict: {ok, action_pk, action_type, applied, mutated_item_ids, error}
        """
        apk = int(action_pk or 0)
        if apk <= 0:
            return {"ok": False, "error": "invalid_action_pk"}
        now = _utc_now_iso()

        with self._lock:
            owns_connection = _conn is None
            conn = _conn if _conn is not None else self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT action_pk, run_id, action_index, action_type,
                           target_item_ids_json, old_texts_json, resolution_type,
                           evidence_item_ids_json, expected_before_hash, evidence_before_hash,
                           new_text, reason, validation_status, applied, apply_status
                    FROM memory_rethink_actions
                    WHERE action_pk=?
                    LIMIT 1;
                    """,
                    (apk,),
                ).fetchone()
                if row is None:
                    return {"ok": False, "error": "action_not_found"}
                if int(_safe_int(row["applied"], 0)) == 1 or str(row["apply_status"] or "") == "simulated":
                    return {"ok": False, "error": "already_applied"}
                if str(row["validation_status"] or "").strip() != "ok":
                    return {"ok": False, "error": f"validation_not_ok:{row['validation_status']}"}

                act_type = str(row["action_type"] or "").strip().lower()
                target_ids = [str(x or "").strip() for x in _json_loads(str(row["target_item_ids_json"] or "[]"), []) if str(x or "").strip()]
                old_texts = [str(x or "") for x in _json_loads(str(row["old_texts_json"] or "[]"), [])]
                resolution_type = str(row["resolution_type"] or "").strip()
                evidence_item_ids = [
                    str(x or "").strip()
                    for x in _json_loads(str(row["evidence_item_ids_json"] or "[]"), [])
                    if str(x or "").strip()
                ]
                new_text = str(row["new_text"] or "")
                reason = str(row["reason"] or "")
                _run_id = str(row["run_id"] or "")

                if act_type != "insert" and not target_ids:
                    return {"ok": False, "error": "no_target_ids"}
                if act_type == "insert" and target_ids:
                    return {"ok": False, "error": "insert_must_not_have_targets"}

                run_row = conn.execute(
                    "SELECT character FROM memory_rethink_runs WHERE run_id=? LIMIT 1;",
                    (_run_id,),
                ).fetchone()
                run_character = str(run_row["character"] or "") if run_row is not None else ""
                layer_rows: list[sqlite3.Row] = []
                stale_reasons: list[str] = []
                evidence_rows: list[sqlite3.Row] = []
                if act_type == "insert":
                    if not evidence_item_ids:
                        return {"ok": False, "error": "insert_requires_evidence"}
                    placeholders = ",".join(["?"] * len(evidence_item_ids))
                    evidence_rows = conn.execute(
                        f"""
                        SELECT item_id, layer, character, text, status, archived, updated_at,
                               salience, access_count, engram_type, item_type, payload_json
                               , source, confidence
                        FROM memory_items WHERE item_id IN ({placeholders});
                        """,
                        tuple(evidence_item_ids),
                    ).fetchall()
                    evidence_by_id = {str(item["item_id"]): item for item in evidence_rows}
                    if len(evidence_by_id) != len(evidence_item_ids):
                        return {"ok": False, "error": "insert_evidence_missing"}
                    current_evidence_texts = [
                        str(evidence_by_id[item_id]["text"] or "") for item_id in evidence_item_ids
                    ]
                    actual_evidence_hash = hashlib.sha256(
                        _json_dumps(current_evidence_texts).encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if (
                        str(row["evidence_before_hash"] or "")
                        and str(row["evidence_before_hash"] or "") != actual_evidence_hash
                    ):
                        stale_reasons.append("evidence_before_hash_mismatch")
                    for item_id in evidence_item_ids:
                        current = evidence_by_id[item_id]
                        if str(current["layer"] or "") not in {"working", "episodic"}:
                            stale_reasons.append(f"evidence_not_readonly_layer:{item_id}")
                        if run_character and str(current["character"] or "") != run_character:
                            stale_reasons.append(f"evidence_character_mismatch:{item_id}")
                        if str(current["status"] or "") != "active" or int(_safe_int(current["archived"], 0)) != 0:
                            stale_reasons.append(f"evidence_not_active:{item_id}")
                        payload = _json_loads(str(current["payload_json"] or "{}"), {})
                        trusted_fact = _memory_row_is_trusted_fact_observation(
                            {key: current[key] for key in current.keys()},
                            payload if isinstance(payload, dict) else {},
                        )
                        if _safe_float(current["salience"], 0.0) < 0.60:
                            stale_reasons.append(f"evidence_salience_low:{item_id}")
                        if _safe_int(current["access_count"], 0) < 1 and not trusted_fact:
                            stale_reasons.append(f"evidence_not_reused_or_trusted:{item_id}")
                else:
                    # Editable targets remain active semantic rows only.
                    layer_rows = conn.execute(
                        f"SELECT item_id, item_type, layer, character, text, status, archived, updated_at FROM memory_items WHERE item_id IN ({','.join(['?'] * len(target_ids))});",
                        tuple(target_ids),
                    ).fetchall()
                    if not layer_rows or len(layer_rows) != len(target_ids):
                        return {"ok": False, "error": "target_items_missing"}
                    if any(item["item_type"] in {"event_cluster", "topic_cluster"} for item in layer_rows):
                        return {"ok": False, "error": "cluster_requires_dedicated_rethink"}
                    non_semantic = [str(item["item_id"]) for item in layer_rows if str(item["layer"] or "") != "semantic"]
                    if non_semantic:
                        return {"ok": False, "error": f"non_semantic_target:{non_semantic[:3]}"}
                    wrong_character = [
                        str(item["item_id"])
                        for item in layer_rows
                        if run_character and str(item["character"] or "") != run_character
                    ]
                    if wrong_character:
                        return {"ok": False, "error": f"character_scope_mismatch:{wrong_character[:3]}"}
                    row_by_id = {str(item["item_id"]): item for item in layer_rows}
                    current_texts = [str(row_by_id[item_id]["text"] or "") for item_id in target_ids]
                    expected_hash = str(row["expected_before_hash"] or "")
                    actual_expected_hash = hashlib.sha256(
                        _json_dumps(old_texts).encode("utf-8", errors="ignore")
                    ).hexdigest()
                    if expected_hash and expected_hash != actual_expected_hash:
                        stale_reasons.append("proposal_hash_mismatch")
                    if len(old_texts) != len(target_ids) or current_texts != old_texts:
                        stale_reasons.append("old_text_not_exact")
                    for item_id in target_ids:
                        current = row_by_id[item_id]
                        if str(current["status"] or "") != "active" or int(_safe_int(current["archived"], 0)) != 0:
                            stale_reasons.append(f"target_not_active:{item_id}")
                if stale_reasons:
                    error = "stale_conflict:" + ",".join(stale_reasons[:4])
                    conn.execute(
                        "UPDATE memory_rethink_actions SET apply_status='stale_conflict', apply_error=? WHERE action_pk=?;",
                        (error[:240], apk),
                    )
                    if _commit:
                        conn.commit()
                    return {"ok": False, "error": error, "stale": True}

                def _attach_resolution_lineage(item_id: str) -> None:
                    current = conn.execute(
                        "SELECT payload_json FROM memory_items WHERE item_id=? LIMIT 1;", (item_id,)
                    ).fetchone()
                    if current is None:
                        return
                    payload = _json_loads(str(current["payload_json"] or "{}"), {})
                    payload = payload if isinstance(payload, dict) else {}
                    history = payload.get("memory_rethink_history", [])
                    history = history if isinstance(history, list) else []
                    history.append({
                        "run_id": _run_id,
                        "action_pk": apk,
                        "resolution_type": resolution_type,
                        "evidence_item_ids": evidence_item_ids,
                        "previous_text_hash": hashlib.sha256(
                            str(before_map.get(item_id, {}).get("row", {}).get("text", "")).encode("utf-8", errors="ignore")
                        ).hexdigest(),
                        "at": now,
                    })
                    payload["memory_rethink_history"] = history[-12:]
                    conn.execute(
                        "UPDATE memory_items SET payload_json=? WHERE item_id=?;",
                        (_json_dumps(payload), item_id),
                    )
                    for evidence_id in evidence_item_ids:
                        exists = conn.execute(
                            "SELECT 1 FROM memory_items WHERE item_id=? LIMIT 1;", (evidence_id,)
                        ).fetchone()
                        if exists is None:
                            continue
                        conn.execute(
                            """
                            INSERT OR IGNORE INTO memory_links
                            (src_item_id, dst_item_id, link_type, weight, meta_json, created_at)
                            VALUES (?, ?, 'rethink_evidence_for', 1.0, ?, ?);
                            """,
                            (evidence_id, item_id, _json_dumps({
                                "run_id": _run_id,
                                "action_pk": apk,
                                "resolution_type": resolution_type,
                            }), now),
                        )

                # 取 before 快照
                before_map = {iid: self._snapshot_item_row(conn, iid) for iid in target_ids}

                mutated_ids: list[str] = []
                after_map: dict[str, dict] = {}

                if shadow:
                    # shadow 模式：模拟 after、不真改 db
                    if act_type == "rewrite" and len(target_ids) == 1:
                        iid = target_ids[0]
                        sim_after = dict(before_map.get(iid, {}))
                        if sim_after.get("exists"):
                            sim_row = dict(sim_after.get("row", {}))
                            sim_row["text"] = new_text
                            sim_row["updated_at"] = now
                            sim_after["row"] = sim_row
                        sim_after["shadow_simulated"] = True
                        after_map[iid] = sim_after
                        mutated_ids.append(iid)
                    elif act_type == "merge":
                        for iid in target_ids:
                            sim = dict(before_map.get(iid, {}))
                            if sim.get("exists"):
                                sim_row = dict(sim.get("row", {}))
                                sim_row["status"] = "merged_by_rethink"
                                sim_row["archived"] = 1
                                sim_row["updated_at"] = now
                                sim["row"] = sim_row
                            sim["shadow_simulated"] = True
                            after_map[iid] = sim
                            mutated_ids.append(iid)
                        after_map["__new_text__"] = {"shadow_simulated": True, "new_text": new_text}
                    elif act_type == "archive":
                        for iid in target_ids:
                            sim = dict(before_map.get(iid, {}))
                            if sim.get("exists"):
                                sim_row = dict(sim.get("row", {}))
                                sim_row["status"] = "archived_by_rethink"
                                sim_row["archived"] = 1
                                sim_row["updated_at"] = now
                                sim["row"] = sim_row
                            sim["shadow_simulated"] = True
                            after_map[iid] = sim
                            mutated_ids.append(iid)
                    elif act_type == "insert":
                        after_map["__new_item__"] = {
                            "shadow_simulated": True,
                            "item_id": "reth_insert_shadow",
                            "row": {
                                "character": run_character,
                                "layer": "semantic",
                                "item_type": "rethink_insert",
                                "text": new_text,
                                "source": "memory_rethink_insert",
                            },
                        }
                        mutated_ids.append("reth_insert_shadow")
                    elif act_type == "no_change":
                        # 走到 apply 不太应该、保险记一笔
                        pass
                    else:
                        return {"ok": False, "error": f"unsupported_action_type:{act_type}"}
                else:
                    # 真改模式
                    if act_type == "rewrite" and len(target_ids) == 1:
                        iid = target_ids[0]
                        # 重写 text、清 embedding_text/sparse 让下次 retrieve 走 backfill
                        conn.execute(
                            """
                            UPDATE memory_items
                            SET text=?, embedding_text='', embedding_sparse_text='',
                                embedding_model_id='', embedding_dim=0,
                                confidence=MAX(confidence, 0.62), updated_at=?
                            WHERE item_id=?;
                            """,
                            (new_text, now, iid),
                        )
                        # 同步 FTS
                        try:
                            conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (iid,))
                            conn.execute("INSERT INTO memory_items_fts(item_id, text) VALUES (?, ?);", (iid, new_text))
                        except Exception:
                            pass
                        _attach_resolution_lineage(iid)
                        after_map[iid] = self._snapshot_item_row(conn, iid)
                        mutated_ids.append(iid)
                    elif act_type == "merge":
                        # 多条 archive + 创建 1 条新 item 承载 new_text
                        for iid in target_ids:
                            conn.execute(
                                "UPDATE memory_items SET archived=1, status='merged_by_rethink', updated_at=? WHERE item_id=?;",
                                (now, iid),
                            )
                            try:
                                conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (iid,))
                            except Exception:
                                pass
                            after_map[iid] = self._snapshot_item_row(conn, iid)
                            mutated_ids.append(iid)
                        # 创建新 item 承载合并后文本
                        # 复用第一个 target 的 character / source 等属性
                        first_row = layer_rows[0]
                        new_iid = "reth_merge_" + uuid.uuid4().hex[:12]
                        new_payload = _json_dumps({
                            "source": "memory_rethink_merge",
                            "run_id": _run_id,
                            "merged_from": list(target_ids),
                            "reason": reason,
                        })
                        new_lifecycle = _json_dumps({
                            "created_by": "memory_rethink",
                            "created_at": now,
                        })
                        new_cue = _json_dumps({})
                        conn.execute(
                            """
                            INSERT INTO memory_items
                            (item_id, character, layer, memory_role, engram_type, item_type, text,
                             payload_json, cue_json, lifecycle_json,
                             salience, confidence, novelty, importance, status, source,
                             evidence_id, created_at, updated_at, last_accessed_at, access_count, archived)
                            VALUES (?, ?, 'semantic', 'semantic', 'fact', 'rethink_merge', ?,
                                    ?, ?, ?,
                                    0.62, 0.68, 0.35, 0.66, 'active', 'memory_rethink_merge',
                                    ?, ?, ?, '', 0, 0);
                            """,
                            (
                                new_iid,
                                str(first_row["character"] or ""),
                                new_text,
                                new_payload,
                                new_cue,
                                new_lifecycle,
                                _run_id,
                                now,
                                now,
                            ),
                        )
                        try:
                            conn.execute("INSERT INTO memory_items_fts(item_id, text) VALUES (?, ?);", (new_iid, new_text))
                        except Exception:
                            pass
                        # 把每条 old → new 写 merge_into link
                        for iid in target_ids:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO memory_links(src_item_id, dst_item_id, link_type, weight, meta_json, created_at)
                                VALUES (?, ?, 'rethink_merge_into', 0.95, ?, ?);
                                """,
                                (iid, new_iid, _json_dumps({"run_id": _run_id, "reason": reason}), now),
                            )
                        _attach_resolution_lineage(new_iid)
                        after_map["__new_item__"] = self._snapshot_item_row(conn, new_iid)
                        mutated_ids.append(new_iid)
                    elif act_type == "archive":
                        for iid in target_ids:
                            conn.execute(
                                "UPDATE memory_items SET archived=1, status='archived_by_rethink', updated_at=? WHERE item_id=?;",
                                (now, iid),
                            )
                            try:
                                conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (iid,))
                            except Exception:
                                pass
                            _attach_resolution_lineage(iid)
                            after_map[iid] = self._snapshot_item_row(conn, iid)
                            mutated_ids.append(iid)
                    elif act_type == "insert":
                        if _fact_is_dialogue_residue(new_text):
                            return {"ok": False, "error": "insert_dialogue_residue"}
                        new_iid = "reth_insert_" + uuid.uuid4().hex[:12]
                        evidence_salience = [
                            _safe_float(item["salience"], 0.0) for item in evidence_rows
                        ]
                        new_salience = max(
                            0.60,
                            sum(evidence_salience) / max(1, len(evidence_salience)),
                        )
                        new_payload = _json_dumps({
                            "source": "memory_rethink_insert",
                            "run_id": _run_id,
                            "reason": reason,
                            "evidence_item_ids": list(evidence_item_ids),
                        })
                        conn.execute(
                            """
                            INSERT INTO memory_items
                            (item_id, character, layer, memory_role, engram_type, item_type, text,
                             payload_json, cue_json, lifecycle_json,
                             salience, confidence, novelty, importance, status, source,
                             evidence_id, created_at, updated_at, last_accessed_at, access_count, archived)
                            VALUES (?, ?, 'semantic', 'semantic', 'fact', 'rethink_insert', ?,
                                    ?, '{}', ?,
                                    ?, 0.68, 0.55, ?, 'active', 'memory_rethink_insert',
                                    ?, ?, ?, '', 0, 0);
                            """,
                            (
                                new_iid,
                                run_character,
                                new_text,
                                new_payload,
                                _json_dumps({"created_by": "memory_rethink", "created_at": now}),
                                float(_clamp(new_salience)),
                                float(_clamp(max(0.66, new_salience))),
                                _run_id,
                                now,
                                now,
                            ),
                        )
                        try:
                            conn.execute(
                                "INSERT INTO memory_items_fts(item_id, text) VALUES (?, ?);",
                                (new_iid, new_text),
                            )
                        except Exception:
                            pass
                        _attach_resolution_lineage(new_iid)
                        after_map["__new_item__"] = self._snapshot_item_row(conn, new_iid)
                        mutated_ids.append(new_iid)
                    elif act_type == "no_change":
                        pass
                    else:
                        return {"ok": False, "error": f"unsupported_action_type:{act_type}"}

                # 写 before/after 到 action audit row
                conn.execute(
                    """
                    UPDATE memory_rethink_actions
                    SET applied=?, applied_at=?, before_json=?, after_json=?,
                        apply_status=?, apply_error=''
                    WHERE action_pk=?;
                    """,
                    (
                        0 if shadow else 1,
                        now,
                        _json_dumps(before_map),
                        _json_dumps(after_map),
                        "simulated" if shadow else "applied",
                        apk,
                    ),
                )
                if _commit:
                    conn.commit()
            finally:
                if owns_connection:
                    conn.close()

        return {
            "ok": True,
            "action_pk": apk,
            "action_type": act_type,
            "applied": True,
            "shadow": bool(shadow),
            "mutated_item_ids": mutated_ids,
        }

    def apply_rethink_actions(self, action_pks: list[int], *, shadow: bool = True) -> dict:
        """Apply one rethink run as a single SQLite transaction.

        All actions are recorded before this method is called. Any validation,
        CAS, or mutation failure rolls the whole batch back; only audit status is
        then committed in a separate transaction.
        """
        normalized = [int(pk) for pk in action_pks if int(pk or 0) > 0]
        if not normalized:
            return {"ok": True, "shadow": bool(shadow), "applied": 0, "actions": []}
        if len(set(normalized)) != len(normalized):
            return {"ok": False, "error": "duplicate_action_pk", "applied": 0, "actions": []}

        with self._lock:
            conn = self._connect()
            results: list[dict] = []
            failed_pk = 0
            failure = ""
            try:
                conn.execute("BEGIN IMMEDIATE;")
                placeholders = ",".join(["?"] * len(normalized))
                rows = conn.execute(
                    f"""
                    SELECT action_pk, run_id, target_item_ids_json
                    FROM memory_rethink_actions
                    WHERE action_pk IN ({placeholders});
                    """,
                    tuple(normalized),
                ).fetchall()
                row_by_pk = {int(row["action_pk"]): row for row in rows}
                if len(row_by_pk) != len(normalized):
                    failure = "action_not_found"
                else:
                    run_ids = {str(row_by_pk[pk]["run_id"] or "") for pk in normalized}
                    if len(run_ids) != 1:
                        failure = "cross_run_batch"
                    seen_targets: set[str] = set()
                    for pk in normalized:
                        targets = [
                            str(value or "").strip()
                            for value in _json_loads(
                                str(row_by_pk[pk]["target_item_ids_json"] or "[]"), []
                            )
                            if str(value or "").strip()
                        ]
                        overlap = [target for target in targets if target in seen_targets]
                        if overlap:
                            failed_pk = pk
                            failure = f"duplicate_target_in_batch:{overlap[:3]}"
                            break
                        seen_targets.update(targets)

                if not failure:
                    for pk in normalized:
                        result = self.apply_rethink_action(
                            pk,
                            shadow=bool(shadow),
                            _conn=conn,
                            _commit=False,
                        )
                        results.append(result)
                        if not bool(result.get("ok", False)):
                            failed_pk = pk
                            failure = str(result.get("error", "batch_action_failed") or "batch_action_failed")
                            break

                if failure:
                    conn.rollback()
                    error_status = "stale_conflict" if failure.startswith("stale_conflict") else "batch_failed"
                    for pk in normalized:
                        status = error_status if pk == failed_pk else "batch_rolled_back"
                        conn.execute(
                            """
                            UPDATE memory_rethink_actions
                            SET applied=0, applied_at='', before_json='{}', after_json='{}',
                                apply_status=?, apply_error=?
                            WHERE action_pk=?;
                            """,
                            (status, failure[:240], pk),
                        )
                    conn.commit()
                    return {
                        "ok": False,
                        "shadow": bool(shadow),
                        "applied": 0,
                        "failed_action_pk": failed_pk,
                        "error": failure,
                        "actions": results,
                    }

                conn.commit()
                return {
                    "ok": True,
                    "shadow": bool(shadow),
                    "applied": len(results),
                    "actions": results,
                    "mutated_item_ids": [
                        item_id
                        for result in results
                        for item_id in (result.get("mutated_item_ids", []) or [])
                    ],
                }
            except Exception as exc:
                conn.rollback()
                failure = f"batch_exception:{type(exc).__name__}:{exc}"[:240]
                for pk in normalized:
                    conn.execute(
                        """
                        UPDATE memory_rethink_actions
                        SET applied=0, applied_at='', before_json='{}', after_json='{}',
                            apply_status='batch_rolled_back', apply_error=?
                        WHERE action_pk=?;
                        """,
                        (failure, pk),
                    )
                conn.commit()
                return {
                    "ok": False,
                    "shadow": bool(shadow),
                    "applied": 0,
                    "error": failure,
                    "actions": results,
                }
            finally:
                conn.close()

    def mark_rethink_actions_not_applied(
        self,
        action_pks: list[int],
        *,
        status: str,
        error: str,
    ) -> None:
        normalized = [int(pk) for pk in action_pks if int(pk or 0) > 0]
        if not normalized:
            return
        safe_status = str(status or "batch_blocked")[:40]
        safe_error = str(error or "")[:240]
        with self._lock:
            conn = self._connect()
            try:
                for pk in normalized:
                    conn.execute(
                        """
                        UPDATE memory_rethink_actions
                        SET applied=0, apply_status=?, apply_error=?
                        WHERE action_pk=? AND applied=0;
                        """,
                        (safe_status, safe_error, pk),
                    )
                conn.commit()
            finally:
                conn.close()

    def revert_rethink_run(self, run_id: str, *, reason: str = "") -> dict:
        """整批回滚一次 rethink_run。

        只有当前行仍与 apply 后快照完全一致时才恢复；任一目标已经被
        后续合法写入改变，则整批 fail closed，不覆盖新状态。
        """
        rid = str(run_id or "").strip()
        if not rid:
            return {"ok": False, "error": "missing_run_id"}
        now = _utc_now_iso()
        reverted = 0
        skipped = 0
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                rows = conn.execute(
                    """
                    SELECT action_pk, action_type, target_item_ids_json, before_json, after_json,
                           applied, reverted_at
                    FROM memory_rethink_actions
                    WHERE run_id=?
                    ORDER BY action_pk DESC;
                    """,
                    (rid,),
                ).fetchall()
                if not rows:
                    conn.rollback()
                    return {"ok": True, "run_id": rid, "reverted": 0, "skipped": 0, "status": "no_actions"}

                eligible_rows = [
                    row for row in rows
                    if int(_safe_int(row["applied"], 0)) == 1
                    and not str(row["reverted_at"] or "").strip()
                ]
                conflicts: list[str] = []
                for row in eligible_rows:
                    before_map = _json_loads(str(row["before_json"] or "{}"), {})
                    after_map = _json_loads(str(row["after_json"] or "{}"), {})
                    before_map = before_map if isinstance(before_map, dict) else {}
                    after_map = after_map if isinstance(after_map, dict) else {}
                    for item_id in before_map:
                        expected = after_map.get(item_id)
                        if not isinstance(expected, dict) or not expected.get("exists"):
                            conflicts.append(f"missing_after_snapshot:{item_id}")
                            continue
                        current = self._snapshot_item_row(conn, item_id)
                        if current != expected:
                            conflicts.append(f"after_state_changed:{item_id}")

                    action_type = str(row["action_type"] or "")
                    if action_type in {"merge", "insert"}:
                        new_item = after_map.get("__new_item__", {})
                        new_item = new_item if isinstance(new_item, dict) else {}
                        new_row = new_item.get("row", {}) if isinstance(new_item.get("row", {}), dict) else {}
                        new_item_id = str(new_row.get("item_id", "") or new_item.get("item_id", "") or "")
                        if not new_item_id:
                            conflicts.append(f"{action_type}_after_snapshot_missing")
                        else:
                            current = self._snapshot_item_row(conn, new_item_id)
                            if new_item.get("exists"):
                                if current != new_item:
                                    conflicts.append(f"after_state_changed:{new_item_id}")
                            else:
                                current_row = current.get("row", {}) if isinstance(current.get("row", {}), dict) else {}
                                if (
                                    not current.get("exists")
                                    or str(current_row.get("text", "") or "") != str(new_item.get("text", "") or "")
                                    or str(current_row.get("source", "") or "")
                                    != ("memory_rethink_insert" if action_type == "insert" else "memory_rethink_merge")
                                    or int(_safe_int(current_row.get("archived", 0), 0)) != 0
                                ):
                                    conflicts.append(f"after_state_changed:{new_item_id}")

                if conflicts:
                    conn.rollback()
                    conflict_error = "revert_conflict:" + ",".join(conflicts[:6])
                    conn.execute(
                        "UPDATE memory_rethink_runs SET status='revert_conflict', fallback_reason=?, updated_at=? WHERE run_id=?;",
                        (conflict_error[:200], now, rid),
                    )
                    for row in eligible_rows:
                        conn.execute(
                            "UPDATE memory_rethink_actions SET apply_error=? WHERE action_pk=?;",
                            (conflict_error[:240], int(_safe_int(row["action_pk"], 0))),
                        )
                    conn.commit()
                    return {
                        "ok": False,
                        "run_id": rid,
                        "reverted": 0,
                        "skipped": len(rows),
                        "status": "revert_conflict",
                        "error": conflict_error,
                    }

                for r in rows:
                    apk = int(_safe_int(r["action_pk"], 0))
                    if int(_safe_int(r["applied"], 0)) != 1:
                        skipped += 1
                        continue
                    if str(r["reverted_at"] or "").strip():
                        skipped += 1
                        continue
                    before_map = _json_loads(str(r["before_json"] or "{}"), {})
                    after_map = _json_loads(str(r["after_json"] or "{}"), {})
                    act_type = str(r["action_type"] or "")
                    try:
                        # 恢复每条原 item
                        if isinstance(before_map, dict):
                            for iid, snap in before_map.items():
                                if isinstance(snap, dict) and snap.get("exists"):
                                    self._restore_item_snapshot(conn, snap, now)
                                    restored = snap.get("row", {}) if isinstance(snap.get("row", {}), dict) else {}
                                    conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (iid,))
                                    if (
                                        int(_safe_int(restored.get("archived", 0), 0)) == 0
                                        and str(restored.get("status", "active") or "active") == "active"
                                    ):
                                        conn.execute(
                                            "INSERT INTO memory_items_fts(item_id, text) VALUES (?, ?);",
                                            (iid, str(restored.get("text", "") or "")),
                                        )
                        # merge 创建项归档；insert 是本 run 新建，回滚时删除。
                        if act_type == "merge" and isinstance(after_map, dict):
                            new_item = after_map.get("__new_item__", {})
                            new_item = new_item if isinstance(new_item, dict) else {}
                            new_row = new_item.get("row", {}) if isinstance(new_item.get("row", {}), dict) else {}
                            new_iid = str(new_row.get("item_id", "") or new_item.get("item_id", "") or "")
                            if new_iid:
                                conn.execute(
                                    "UPDATE memory_items SET archived=1, status='reverted_rethink_merge', updated_at=? WHERE item_id=?;",
                                    (now, new_iid),
                                )
                                try:
                                    conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (new_iid,))
                                except Exception:
                                    pass
                        elif act_type == "insert" and isinstance(after_map, dict):
                            new_item = after_map.get("__new_item__", {})
                            new_item = new_item if isinstance(new_item, dict) else {}
                            new_row = new_item.get("row", {}) if isinstance(new_item.get("row", {}), dict) else {}
                            new_iid = str(new_row.get("item_id", "") or new_item.get("item_id", "") or "")
                            if new_iid:
                                conn.execute("DELETE FROM memory_items_fts WHERE item_id=?;", (new_iid,))
                                conn.execute("DELETE FROM memory_items WHERE item_id=?;", (new_iid,))
                        conn.execute(
                            """
                            UPDATE memory_rethink_actions
                            SET reverted_at=?, revert_reason=?, apply_status='reverted', apply_error=''
                            WHERE action_pk=?;
                            """,
                            (now, str(reason or "")[:200], apk),
                        )
                        reverted += 1
                    except Exception as exc:
                        skipped += 1
                        conn.execute(
                            """
                            UPDATE memory_rethink_actions
                            SET revert_reason=?
                            WHERE action_pk=?;
                            """,
                            (f"revert_failed: {exc}"[:200], apk),
                        )

                conn.execute(
                    """
                    DELETE FROM memory_links
                    WHERE link_type IN ('rethink_merge_into', 'rethink_evidence_for')
                      AND json_valid(meta_json)=1
                      AND json_extract(meta_json, '$.run_id')=?;
                    """,
                    (rid,),
                )

                conn.execute(
                    """
                    UPDATE memory_rethink_runs
                    SET status='reverted', updated_at=?
                    WHERE run_id=?;
                    """,
                    (now, rid),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        return {"ok": True, "run_id": rid, "reverted": int(reverted), "skipped": int(skipped)}
