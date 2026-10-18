"""Store mixin: canon profiles, motivations, decay/forgetting, clustering, OOC and proactive policy.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import hashlib
import re
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Callable
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import (
    _emb_decode, _embedding_row_matches_current, _safe_float, _safe_int, _utc_now_iso,
    mem_policy_llm_enabled,
)
from tsuzuri.memory_core_impl.memory_rows import _lexical_similarity
from tsuzuri.memory_core_impl.explicit_recall import _forgetting_protected
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads, _parse_dt
from tsuzuri.memory_core_impl.meta_policy import (
    _CANON_CLAIM_TYPES, _CANON_SOURCE_WEIGHTS, _canon_weight_for_source, _deep_merge_dict,
    _parse_bool,
)
from tsuzuri.persona_registry import persona_regex, resolve_agent, setting, user_name

# Per-agent reply-style heuristics for the OOC audit; deployments override via
# persona_registry.setting("ooc_style_rules").
_DEFAULT_OOC_STYLE_RULES = {
    "max_len": {},          # agent -> max reply length
    "default_max_len": 80,
    "no_soft": [],          # agents for whom overly sweet replies are out of character
    "no_cold": [],          # agents for whom dismissive replies are out of character
    "banned_tone": {},      # agent -> sentence-final particles they never use
}


class _LifecycleMixin:
    """Canon profiles, motivations, decay/forgetting, clustering, OOC and proactive policy."""


    def refresh_character_canon(self, character: str, sources: list[dict] | None = None, *, shadow: bool = True) -> dict:
        char = resolve_agent(character)
        rows = [dict(x) for x in (sources or []) if isinstance(x, dict)]
        run_id = "canon_" + uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        source_rows = []
        claim_rows = []
        for src in rows:
            source_type = str(src.get("source_type", "") or "").strip() or "unknown"
            source_url = str(src.get("source_url", src.get("url", "")) or "").strip()
            if not source_url:
                continue
            title = str(src.get("title", "") or "").strip()
            snippet = str(src.get("snippet", src.get("content", "")) or "").strip()
            freshness = _clamp(_safe_float(src.get("freshness", 0.7), 0.7))
            confidence = _clamp(_safe_float(src.get("confidence", 0.7), 0.7))
            weight = round(_canon_weight_for_source(source_type) * (0.55 + 0.45 * confidence), 3)
            meta = src.get("meta", {}) if isinstance(src.get("meta", {}), dict) else {}
            source_rows.append(
                {
                    "character": char,
                    "source_type": source_type,
                    "source_url": source_url[:500],
                    "title": title[:200],
                    "snippet": snippet[:800],
                    "fetched_at": str(src.get("fetched_at", now) or now),
                    "freshness": freshness,
                    "confidence": confidence,
                    "weight": weight,
                    "meta_json": _json_dumps(meta),
                }
            )
            raw_claims = src.get("claims", []) if isinstance(src.get("claims", []), list) else []
            for claim in raw_claims:
                if isinstance(claim, str):
                    claim = {"claim_type": "speech_style", "claim_text": claim}
                if not isinstance(claim, dict):
                    continue
                claim_type = str(claim.get("claim_type", "") or "").strip()
                claim_text = str(claim.get("claim_text", claim.get("text", "")) or "").strip()
                if (not claim_text) or (claim_type not in _CANON_CLAIM_TYPES):
                    continue
                c_conf = _clamp(_safe_float(claim.get("confidence", confidence), confidence))
                c_fresh = _clamp(_safe_float(claim.get("freshness", freshness), freshness))
                c_weight = round(_canon_weight_for_source(source_type) * (0.5 + 0.5 * c_conf), 3)
                claim_rows.append(
                    {
                        "character": char,
                        "source_type": source_type,
                        "source_url": source_url[:500],
                        "claim_type": claim_type,
                        "claim_text": claim_text[:320],
                        "confidence": c_conf,
                        "freshness": c_fresh,
                        "weight": c_weight,
                        "status": "active",
                        "extracted_at": now,
                        "meta_json": _json_dumps(claim.get("meta", {}) if isinstance(claim.get("meta", {}), dict) else {}),
                    }
                )

        with self._lock:
            conn = self._connect()
            try:
                for row in source_rows:
                    conn.execute(
                        """
                        INSERT INTO canon_sources(character, source_type, source_url, title, snippet, fetched_at, freshness, confidence, weight, meta_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(character, source_type, source_url) DO UPDATE SET
                            title=excluded.title,
                            snippet=excluded.snippet,
                            fetched_at=excluded.fetched_at,
                            freshness=excluded.freshness,
                            confidence=excluded.confidence,
                            weight=excluded.weight,
                            meta_json=excluded.meta_json;
                        """,
                        (row["character"], row["source_type"], row["source_url"], row["title"], row["snippet"], row["fetched_at"], row["freshness"], row["confidence"], row["weight"], row["meta_json"]),
                    )
                for row in claim_rows:
                    conn.execute(
                        """
                        INSERT INTO canon_claims(character, source_type, source_url, claim_type, claim_text, confidence, freshness, weight, status, extracted_at, meta_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(character, source_url, claim_type, claim_text) DO UPDATE SET
                            confidence=excluded.confidence,
                            freshness=excluded.freshness,
                            weight=excluded.weight,
                            status=excluded.status,
                            extracted_at=excluded.extracted_at,
                            meta_json=excluded.meta_json;
                        """,
                        (row["character"], row["source_type"], row["source_url"], row["claim_type"], row["claim_text"], row["confidence"], row["freshness"], row["weight"], row["status"], row["extracted_at"], row["meta_json"]),
                    )
                profile = self.build_character_canon_profile(char)
                conn.execute(
                    """
                    INSERT INTO canon_calibration_runs(run_id, character, status, source_summary_json, claims_json, profile_json, shadow, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (run_id, char, "shadow_collected" if bool(shadow) else "collected", _json_dumps({"source_count": len(source_rows), "claim_count": len(claim_rows)}), _json_dumps(claim_rows[:120]), _json_dumps(profile), 1 if bool(shadow) else 0, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return {"ok": True, "run_id": run_id, "character": char, "source_count": len(source_rows), "claim_count": len(claim_rows), "shadow": bool(shadow)}

    def build_character_canon_profile(self, character: str) -> dict:
        char = resolve_agent(character)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT source_type, source_url, claim_type, claim_text, confidence, freshness, weight, extracted_at
                    FROM canon_claims
                    WHERE character=? AND status='active'
                    ORDER BY weight DESC, extracted_at DESC
                    LIMIT 160;
                    """,
                    (char,),
                ).fetchall()
            finally:
                conn.close()
        claims = [{k: r[k] for k in r.keys()} for r in rows]
        grouped = {}
        for row in claims:
            ctype = str(row.get("claim_type", "") or "").strip() or "speech_style"
            grouped.setdefault(ctype, [])
            grouped[ctype].append(row)
        anchors = {}
        for ctype, items in grouped.items():
            top = sorted(items, key=lambda x: (_safe_float(x.get("weight", 0.0), 0.0), _safe_float(x.get("confidence", 0.0), 0.0)), reverse=True)[:8]
            anchors[ctype] = [str(x.get("claim_text", "") or "").strip() for x in top if str(x.get("claim_text", "") or "").strip()]
        return {"character": char, "claims": claims[:80], "anchors": anchors, "source_weights": dict(_CANON_SOURCE_WEIGHTS), "updated_at": _utc_now_iso()}

    # ════════════════════════════════════════════════════════════
    #  Motivation Layer: persistent character goals
    # ════════════════════════════════════════════════════════════

    def upsert_motivation(self, character: str, goal_text: str, *,
                          origin: str = "reflection", priority: float = 0.5,
                          urgency: float = 0.3, decay_rate: float = 0.02,
                          meta: dict | None = None) -> str:
        """Create or update a character motivation. Returns motivation_id."""
        char = resolve_agent(character)
        text = str(goal_text or "").strip()[:120]
        if not text:
            return ""
        mid = hashlib.md5(f"{char}:{text}".encode("utf-8")).hexdigest()[:16]
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    "SELECT motivation_id, reinforcement_count FROM character_motivations WHERE motivation_id=?", (mid,)
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE character_motivations SET priority=MAX(priority,?), last_reinforced_at=?, reinforcement_count=reinforcement_count+1, status='active' WHERE motivation_id=?",
                        (priority, now, mid),
                    )
                else:
                    conn.execute(
                        """INSERT INTO character_motivations
                        (motivation_id, character, goal_text, origin, priority, urgency, created_at, last_reinforced_at, decay_rate, meta_json)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (mid, char, text, origin, min(1.0, max(0.0, priority)), min(1.0, max(0.0, urgency)),
                         now, now, max(0.001, decay_rate), _json_dumps(meta or {})),
                    )
                conn.commit()
            finally:
                conn.close()
        return mid

    def get_active_motivations(self, character: str, top_k: int = 3) -> list[dict]:
        """Get top-k active motivations for a character, sorted by effective priority (with decay)."""
        char = resolve_agent(character)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT * FROM character_motivations WHERE character=? AND status='active' ORDER BY priority DESC LIMIT ?",
                    (char, max(1, min(10, top_k * 3))),
                ).fetchall()
            finally:
                conn.close()
        now_ts = time.time()
        results = []
        for r in rows:
            row = dict(r)
            # Apply time-based decay
            last_reinf = row.get("last_reinforced_at", "")
            if last_reinf:
                try:
                    dt = datetime.fromisoformat(str(last_reinf).replace("Z", "+00:00"))
                    days_since = max(0.0, (now_ts - dt.timestamp()) / 86400.0)
                except Exception:
                    days_since = 7.0
            else:
                days_since = 30.0
            decay = float(row.get("decay_rate", 0.02) or 0.02)
            effective = float(row.get("priority", 0.5) or 0.5) * (1.0 - decay) ** days_since
            row["effective_priority"] = round(effective, 4)
            if effective >= 0.05:
                results.append(row)
        results.sort(key=lambda x: x["effective_priority"], reverse=True)
        return results[:top_k]

    def reinforce_motivation(self, motivation_id: str) -> bool:
        """Reinforce a motivation (called when conversation touches related goal)."""
        mid = str(motivation_id or "").strip()
        if not mid:
            return False
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE character_motivations SET last_reinforced_at=?, reinforcement_count=reinforcement_count+1, priority=MIN(1.0, priority+0.05) WHERE motivation_id=?",
                    (now, mid),
                )
                conn.commit()
                return True
            finally:
                conn.close()

    def decay_all_motivations(self, character: str) -> int:
        """Mark motivations with very low effective priority as dormant. Called during dream cycle."""
        char = resolve_agent(character)
        now = _utc_now_iso()
        now_ts = time.time()
        dormant_count = 0
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT motivation_id, priority, decay_rate, last_reinforced_at FROM character_motivations WHERE character=? AND status='active'",
                    (char,),
                ).fetchall()
                for r in rows:
                    last_reinf = str(r["last_reinforced_at"] or "")
                    try:
                        dt = datetime.fromisoformat(last_reinf.replace("Z", "+00:00"))
                        days = max(0.0, (now_ts - dt.timestamp()) / 86400.0)
                    except Exception:
                        days = 30.0
                    decay = float(r["decay_rate"] or 0.02)
                    eff = float(r["priority"] or 0.5) * (1.0 - decay) ** days
                    if eff < 0.05:
                        conn.execute("UPDATE character_motivations SET status='dormant', meta_json=json_set(COALESCE(meta_json,'{}'),'$.dormant_at',?) WHERE motivation_id=?",
                                     (now, r["motivation_id"]))
                        dormant_count += 1
                conn.commit()
            finally:
                conn.close()
        return dormant_count

    def decay_all_episodic(
        self,
        character: str,
        *,
        half_life_days: float = 60.0,
        archive_threshold: float = 0.10,
        salience_protect: float = 0.70,
        access_protect: int = 3,
        dry_run: bool = False,
    ) -> dict:
        """Phase O.2 (2026-05-02): 时间基 episodic 衰减、半衰期 60 天。

        对 active episodic 项：
          new_importance = importance * 0.5^(days_since / half_life_days)

        保护规则（满足任一即跳过）：
          - salience >= salience_protect (0.70)：高显著度永生
          - access_count > access_protect (3)：高频召回保留
          - layer != 'episodic'：semantic / collective / working 不衰

        archive 规则：
          new_importance < archive_threshold (0.10) → 标 archived（status='archived_decayed'）

        Args:
          character: 角色名
          half_life_days: 半衰期（天）、默认 60
          archive_threshold: archive 阈值、默认 0.10
          salience_protect: salience 保护阈值、默认 0.70
          access_protect: access_count 保护阈值、默认 3
          dry_run: True 只算不写、返 candidates list

        Returns:
          dict {
            "scanned": int 扫描数,
            "decayed": int 实际 archive 数,
            "skipped_protected": int 受保护数,
            "candidates": list[dict] dry_run 时的候选列表,
          }
        """
        char = resolve_agent(character)
        now_iso = _utc_now_iso()
        now_ts = time.time()
        scanned = 0
        decayed = 0
        skipped_protected = 0
        candidates: list[dict] = []
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT item_id, importance, salience, access_count, last_accessed_at, created_at, length(text) as tlen
                       FROM memory_items
                       WHERE character=? AND layer='episodic' AND archived=0 AND status='active'""",
                    (char,),
                ).fetchall()
                for r in rows:
                    scanned += 1
                    sal = float(r["salience"] or 0.5)
                    acc = int(r["access_count"] or 0)
                    imp = float(r["importance"] or 0.5)
                    # 保护规则
                    if sal >= salience_protect or acc > access_protect:
                        skipped_protected += 1
                        continue
                    # 计算 days_since
                    last = str(r["last_accessed_at"] or "").strip()
                    if not last:
                        last = str(r["created_at"] or "").strip()
                    try:
                        dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                        days = max(0.0, (now_ts - dt.timestamp()) / 86400.0)
                    except Exception:
                        days = 30.0  # 时间解析失败 → 默认 30 天
                    # 半衰期衰减
                    new_imp = imp * (0.5 ** (days / max(0.1, half_life_days)))
                    if new_imp >= archive_threshold:
                        continue
                    # 触发 archive
                    if dry_run:
                        candidates.append({
                            "item_id": str(r["item_id"]),
                            "importance_old": round(imp, 3),
                            "importance_new": round(new_imp, 3),
                            "salience": round(sal, 3),
                            "access_count": acc,
                            "days_since": round(days, 1),
                        })
                    else:
                        conn.execute(
                            "UPDATE memory_items SET archived=1, status='archived_decayed', updated_at=? WHERE item_id=?",
                            (now_iso, str(r["item_id"])),
                        )
                        decayed += 1
                if not dry_run:
                    conn.commit()
            finally:
                conn.close()
        return {
            "scanned": scanned,
            "decayed": decayed,
            "skipped_protected": skipped_protected,
            "candidates": candidates,
            "half_life_days": half_life_days,
            "archive_threshold": archive_threshold,
        }

    def advance_forgetting_lifecycle(
        self,
        character: str,
        *,
        frozen_min_days: float = 90.0,
        tombstone_min_days: float = 180.0,
        dry_run: bool = True,
    ) -> dict:
        """Advance archived episodic memories through frozen and tombstone states."""
        char = resolve_agent(character)
        frozen_days = max(0.0, _safe_float(frozen_min_days, 90.0))
        tombstone_days = max(frozen_days, _safe_float(tombstone_min_days, 180.0))
        now_iso = _utc_now_iso()
        now_dt = utc_now_naive()
        frozen_candidates = 0
        frozen_applied = 0
        tombstone_candidates = 0
        tombstone_applied = 0
        skipped_protected = 0
        skipped_invalid_timestamp = 0
        candidates: list[dict] = []

        def _age_days(item: dict) -> float | None:
            created = _parse_dt(str(item.get("created_at", "") or ""))
            if created is None:
                return None
            return max(0.0, (now_dt - created).total_seconds() / 86400.0)

        with self._lock:
            conn = self._connect()
            try:
                # Snapshot both pools before applying either pass so one call cannot
                # leap directly from archived to tombstone.
                archived_rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT item_id, character, layer, engram_type, item_type, text,
                               payload_json, salience, access_count, created_at, updated_at,
                               evidence_id, status
                        FROM memory_items
                        WHERE character=? AND archived=1
                          AND layer='episodic'
                          AND status IN ('archived_decayed', 'archived');
                        """,
                        (char,),
                    ).fetchall()
                ]
                frozen_rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT item_id, character, layer, engram_type, item_type, text,
                               payload_json, salience, access_count, created_at, updated_at,
                               evidence_id, status
                        FROM memory_items
                        WHERE character=? AND archived=1
                          AND layer='episodic' AND status='frozen';
                        """,
                        (char,),
                    ).fetchall()
                ]

                for item in archived_rows:
                    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
                    protected, reason = _forgetting_protected(item, payload)
                    if protected:
                        skipped_protected += 1
                        continue
                    age_days = _age_days(item)
                    if age_days is None:
                        skipped_invalid_timestamp += 1
                        continue
                    if age_days < frozen_days:
                        continue
                    frozen_candidates += 1
                    candidate = {
                        "item_id": str(item.get("item_id", "") or ""),
                        "from_status": str(item.get("status", "") or ""),
                        "to_status": "frozen",
                        "age_days": round(age_days, 1),
                    }
                    if dry_run:
                        candidates.append(candidate)
                        continue
                    update_cursor = conn.execute(
                        """
                        UPDATE memory_items
                        SET status='frozen', archived=1, embedding_text='',
                            embedding_sparse_text='', embedding_model_id='', embedding_dim=0,
                            updated_at=?
                        WHERE item_id=? AND archived=1 AND layer='episodic'
                          AND status IN ('archived_decayed', 'archived');
                        """,
                        (now_iso, candidate["item_id"]),
                    )
                    if update_cursor.rowcount > 0:
                        frozen_applied += 1

                for item in frozen_rows:
                    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
                    payload = payload if isinstance(payload, dict) else {}
                    protected, reason = _forgetting_protected(item, payload)
                    if protected:
                        skipped_protected += 1
                        continue
                    age_days = _age_days(item)
                    if age_days is None:
                        skipped_invalid_timestamp += 1
                        continue
                    if age_days < tombstone_days:
                        continue
                    tombstone_candidates += 1
                    candidate = {
                        "item_id": str(item.get("item_id", "") or ""),
                        "from_status": "frozen",
                        "to_status": "tombstone",
                        "age_days": round(age_days, 1),
                    }
                    if dry_run:
                        candidates.append(candidate)
                        continue
                    history = payload.get("forgetting_history", [])
                    if not isinstance(history, list):
                        history = []
                    history.append(
                        {
                            "at": now_iso,
                            "action": "tombstone",
                            "original_text_len": len(str(item.get("text", "") or "")),
                            "evidence_id": str(item.get("evidence_id", "") or ""),
                        }
                    )
                    payload["forgetting_history"] = history[-8:]
                    update_cursor = conn.execute(
                        """
                        UPDATE memory_items
                        SET status='tombstone', archived=1, text='[forgotten]',
                            embedding_text='', embedding_sparse_text='',
                            embedding_model_id='', embedding_dim=0,
                            payload_json=?, updated_at=?
                        WHERE item_id=? AND archived=1
                          AND layer='episodic' AND status='frozen';
                        """,
                        (_json_dumps(payload), now_iso, candidate["item_id"]),
                    )
                    if update_cursor.rowcount > 0:
                        tombstone_applied += 1

                if not dry_run:
                    if frozen_applied or tombstone_applied:
                        self._rebuild_memory_items_fts_for_active(conn)
                    conn.commit()
            finally:
                conn.close()

        return {
            "ok": True,
            "status": "dry_run" if dry_run else "applied",
            "character": char,
            "dry_run": bool(dry_run),
            "frozen_candidates": frozen_candidates,
            "frozen_applied": frozen_applied,
            "tombstone_candidates": tombstone_candidates,
            "tombstone_applied": tombstone_applied,
            "skipped_protected": skipped_protected,
            "skipped_invalid_timestamp": skipped_invalid_timestamp,
            "candidates": candidates,
            "frozen_min_days": frozen_days,
            "tombstone_min_days": tombstone_days,
        }

    def cluster_and_summarize_memories(
        self,
        character: str,
        *,
        source_layer: str = "episodic",
        summary_layer: str = "semantic",
        similarity_threshold: float = 0.62,
        min_cluster_size: int = 4,
        max_clusters_per_run: int = 3,
        llm_summarizer: "Callable[[list[dict], str], str] | None" = None,
        dry_run: bool = False,
    ) -> dict:
        """Phase Q.1.B (2026-05-03): cluster N similar episodic、合成 parent summary 节点

        借鉴 MemOS reorganizer.py：用现有 embedding_text + scipy hierarchy 算法、
        把高度相似的 episodic items 聚簇、用 LLM 提炼一句话 summary、写入新 parent
        节点（layer=semantic）、children 标 parent_id。

        参数:
          source_layer: 在哪一层做 cluster (默认 episodic)
          summary_layer: parent 写到哪一层 (默认 semantic)
          similarity_threshold: cosine sim 阈值 (默认 0.62)
          min_cluster_size: 最小 cluster 大小 (默认 4)
          max_clusters_per_run: 单次最多生成 parent 数 (默认 3、控成本)
          llm_summarizer: callable(member_items, character) -> str；不传时用默认
          dry_run: True 仅返 candidates、不写 db

        返:
          {
            "scanned": int 扫描数,
            "clusters_found": int 满足 min_size 的 cluster 数,
            "clusters_summarized": int 实际生成 parent 数,
            "parents_created": list[{item_id, member_count, summary_preview}],
            "candidates": list (dry_run only)
          }
        """
        char = resolve_agent(character)
        result = {
            "scanned": 0,
            "clusters_found": 0,
            "clusters_summarized": 0,
            "parents_created": [],
            "candidates": [],
        }

        with self._lock:
            conn = self._connect()
            try:
                # 取 source_layer × archived=0 × parent_id='' 全 items（未已被 cluster）
                # 2026-05-24 排除 cross_char_daily_digest{,_self_record}：
                # 这些 row 是「听 X 说：用户说... 我跟他...」的二手转述、视角属于 X 不属于本角色。
                # 之前 bug：cross digest 和角色自己的 episodic event 在 embedding 上相似、被聚到同一
                # cluster、_default_cluster_summarizer 用"主语 + 动词 + 宾语"格式合成出
                # 「用户与{character}约定某事」这种主体错位的 summary（实际是另一个角色
                # 和用户约定的、被 broadcast 来的）。过滤掉 cross digest 后只对角色自己经历做聚类。
                rows = conn.execute(
                    """SELECT item_id, text, importance, salience, embedding_text,
                              embedding_model_id, embedding_dim, layer
                       FROM memory_items
                       WHERE character=? AND layer=? AND archived=0 AND status='active'
                         AND parent_id=''
                         AND embedding_text != ''
                         AND source NOT LIKE 'cross_char_daily_digest%'""",
                    (char, source_layer),
                ).fetchall()
            finally:
                conn.close()

        result["scanned"] = len(rows)
        if len(rows) < min_cluster_size:
            return result  # 数据不够、跳过

        # 解 embedding 矩阵
        try:
            import numpy as _np_clust
            embs: list[_np_clust.ndarray] = []
            valid_rows: list = []
            for r in rows:
                if not _embedding_row_matches_current(dict(r)):
                    continue
                _e_text = str(r["embedding_text"] or "")
                _vec = _emb_decode(_e_text) if callable(_emb_decode) else None
                if _vec is None or len(_vec) == 0:
                    continue
                embs.append(_vec)
                valid_rows.append(r)
            if len(embs) < min_cluster_size:
                return result
            mat = _np_clust.array(embs)  # (N, D)
        except Exception as _emb_err:
            _tsz_emit(f"[ClusterSummarize] embedding decode error: {_emb_err}")
            return result

        # scipy hierarchy linkage
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import pdist
            # cosine 距离 = 1 - cosine_sim、对应 linkage method='average'
            _dist = pdist(mat, metric="cosine")
            _link = linkage(_dist, method="average")
            # threshold = 1 - similarity_threshold (距离空间)
            _labels = fcluster(_link, t=1.0 - similarity_threshold, criterion="distance")
        except Exception as _sci_err:
            _tsz_emit(f"[ClusterSummarize] scipy clustering error: {_sci_err}")
            return result

        # 按 cluster_id 分组 + 过滤 size < min
        from collections import defaultdict
        _groups: "dict[int, list]" = defaultdict(list)
        for _idx, _cid in enumerate(_labels):
            _groups[int(_cid)].append(valid_rows[_idx])
        _eligible = [(_cid, _members) for _cid, _members in _groups.items()
                     if len(_members) >= min_cluster_size]
        # 按 cluster size 倒序、优先合 size 大的
        _eligible.sort(key=lambda x: -len(x[1]))
        _eligible = _eligible[:max_clusters_per_run]
        result["clusters_found"] = len(_eligible)

        if dry_run:
            for _cid, _members in _eligible:
                result["candidates"].append({
                    "cluster_id": _cid,
                    "member_count": len(_members),
                    "member_ids": [str(m["item_id"]) for m in _members],
                    "sample_texts": [str(m["text"] or "")[:80] for m in _members[:3]],
                })
            return result

        # 默认 summarizer (lazy 用 chat_llm 调 LLM)
        if llm_summarizer is None:
            llm_summarizer = self._default_cluster_summarizer

        now_iso = _utc_now_iso()
        for _cid, _members in _eligible:
            try:
                _member_items = [
                    {
                        "item_id": str(m["item_id"]),
                        "text": str(m["text"] or ""),
                        "importance": float(m["importance"] or 0.5),
                        "salience": float(m["salience"] or 0.5),
                    }
                    for m in _members
                ]
                _summary = llm_summarizer(_member_items, char)
                if not _summary or len(_summary) < 10:
                    continue
                _summary = _summary[:300]
                # 创建 parent item
                _parent_id = uuid.uuid4().hex[:24]
                _avg_imp = sum(m["importance"] for m in _member_items) / len(_member_items)
                _avg_sal = sum(m["salience"] for m in _member_items) / len(_member_items)
                _parent_imp = min(1.0, _avg_imp + 0.10)  # parent 略高于 children 平均
                _parent_sal = min(1.0, _avg_sal + 0.10)
                _member_id_list = [m["item_id"] for m in _member_items]
                _cluster_meta = {
                    "cluster_id": _cid,
                    "clustered_at": now_iso,
                    "similarity_threshold": similarity_threshold,
                    "source_layer": source_layer,
                    "method": "scipy.hierarchy.average",
                }
                try:
                    def _write_cluster_parent(conn: sqlite3.Connection):
                        conn.execute(
                            """INSERT INTO memory_items
                            (item_id, character, layer, memory_role, item_type, text, payload_json,
                             created_at, updated_at, importance, salience, status, archived,
                             member_ids_json, cluster_meta_json)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                _parent_id, char, summary_layer, summary_layer,
                                "cluster_summary", _summary, "{}",
                                now_iso, now_iso,
                                _parent_imp, _parent_sal,
                                "active", 0,
                                _json_dumps(_member_id_list),
                                _json_dumps(_cluster_meta),
                            ),
                        )
                        for _mid in _member_id_list:
                            conn.execute(
                                "UPDATE memory_items SET parent_id=?, updated_at=? WHERE item_id=?",
                                (_parent_id, now_iso, _mid),
                            )
                        try:
                            conn.execute(
                                """INSERT INTO evolution_action_log
                                (run_id, action_index, op, item_id, before_json, after_json, status, created_at)
                                VALUES (?,?,?,?,?,?,?,?)""",
                                (
                                    f"cluster_{now_iso}_{_cid}", 0, "cluster_summarize",
                                    _parent_id,
                                    _json_dumps({"members": _member_id_list, "source_layer": source_layer}),
                                    _json_dumps({"parent_id": _parent_id, "summary": _summary[:120]}),
                                    "applied", now_iso,
                                ),
                            )
                        except Exception:
                            pass

                    self._run_write_transaction(_write_cluster_parent)
                except Exception as _write_err:
                    _tsz_emit(f"[ClusterSummarize] cluster {_cid} error: {_write_err}")
                    continue
                result["clusters_summarized"] += 1
                result["parents_created"].append({
                    "parent_id": _parent_id,
                    "member_count": len(_member_id_list),
                    "summary_preview": _summary[:80],
                })
            except Exception as _c_err:
                _tsz_emit(f"[ClusterSummarize] cluster {_cid} error: {_c_err}")
                continue

        return result

    def _default_cluster_summarizer(self, member_items: list[dict], character: str) -> str:
        """Phase Q.1.B 默认 LLM summarizer：用 character voice 提炼共性

        Falls back to heuristic concat if LLM unavailable.
        """
        def _fix_user_apology_subject(summary: str) -> str:
            _out = str(summary or "").strip()
            _char = str(character or "").strip()
            if not _out or not _char:
                return _out
            _joined = "\n".join(str(it.get("text", "") or "") for it in member_items[:8] if isinstance(it, dict))
            if not _joined:
                return _out
            _has_user_speech_marker = bool(persona_regex(r"({USER}|用户|开发者|对方|他说|她说|ta说|TA说|user).{0,8}(说|表示|称|：|:)", re.I).search(_joined))
            _addresses_char = bool(re.search(re.escape(_char) + r"[，,、：:\s]", _joined))
            _user_apologized = bool(re.search(r"(我回来了|抱歉|对不起|让你等|等我|我.*失联|我.*回来)", _joined))
            _summary_blames_char = _out.startswith(_char) and bool(re.search(r"(失联|道歉|认错|回来|让.*等)", _out))
            if _has_user_speech_marker and _addresses_char and _user_apologized and _summary_blames_char:
                if "系统故障" in _out or "失联" in _out:
                    return f"{user_name()}失联后向{_char}道歉"
                if "道歉" in _out or "认错" in _out:
                    return f"{user_name()}向{_char}道歉"
                if "回来" in _out:
                    return f"{user_name()}回来找{_char}"
            return _out

        try:
            import os as _os_cs
            _api_key = _os_cs.getenv("LLM_API_KEY", "").strip()
            _base_url = _os_cs.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1").strip()
            if not _api_key:
                # 无 LLM 不合成 parent（返回空串由 caller 跳过该簇）。旧版把
                # 「[Cluster of N] + 首条原话」当摘要写进 semantic 层（2026-10-19 审计 LIFE-016）。
                return ""
            from tsuzuri.llm_client import shared_openai_client
            _client = shared_openai_client(api_key=_api_key, base_url=_base_url, timeout=30)
            _texts = "\n".join(f"- {it.get('text', '')[:100]}" for it in member_items[:8])
            # 2026-05-11 v4: 把 summary 长度从 80-120 字直接压到 ≤15 字。
            # 老 prompt 给 LLM 100 字空间、它就会用情绪和性格判断填、产生幻觉
            # ("总是" / "每次都" / "看来你真的...")。短上限把幻觉物理空间挤掉、
            # 只剩单一主谓宾事实。长度同时由 prompt 显式 + post-process 截断双保险。
            _user = user_name()
            _prompt = (
                f"把下面 {len(member_items)} 条小事提炼成 **不超过 15 个汉字** 的"
                f"单一事实短语：\n"
                f"{_texts}\n\n"
                f"硬约束：\n"
                f"1. **总长度 ≤ 15 个汉字**\n"
                f"2. 用第三人称（'{_user}' / '{character}'）描述、不用'我'/'你'\n"
                f"3. 形式：主语 + 动词 + 宾语（例：'{_user}问{character}课表'/'{_user}关心{character}作息'）\n"
                f"4. **禁词**：总是 / 每次 / 经常 / 一直 / 老是 / 又 / 从来\n"
                f"5. 如果片段是'{_user}说：...'或用户对{character}说的话，主语必须是{_user}，不能写成{character}做了这件事\n"
                f"6. 不要评价 / 抱怨 / 推断性格\n"
                f"7. 直接输出短语、无前缀、无引号、无句号"
            )
            _model = _os_cs.getenv("LLM_MODEL", "deepseek-flash")
            _out = ""
            for _attempt, _max_tokens in enumerate((120, 240), start=1):
                _resp = _client.chat.completions.create(
                    model=_model,
                    messages=[{"role": "user", "content": _prompt}],
                    temperature=0.3, max_tokens=_max_tokens,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                # capture usage if available（lazy import）
                try:
                    from tsuzuri.llm_usage import capture_llm_usage as _cap
                    _cap(_resp, label="cluster_summarize")
                except Exception:
                    pass
                _out = str(_resp.choices[0].message.content or "").strip()
                if _out:
                    break
                _tsz_emit(
                    f"[ClusterSummarize] empty response, retry "
                    f"(attempt {_attempt}/2, max_tokens={_max_tokens})",
                    flush=True,
                )
            if not _out:
                return ""
            # post-process：去引号 / 句号、硬截断 ≤ 15 汉字（LLM 偶尔违约的兜底）
            _out = _out.strip("「」『』\"'`。.，、 ")
            # 取第一行（防 LLM 多输出）
            _out = _out.splitlines()[0].strip() if _out else ""
            if len(_out) > 15:
                _out = _out[:15]
            return _fix_user_apology_subject(_out)
        except Exception as _err:
            _tsz_emit(f"[ClusterSummarize] llm summarizer failed, cluster skipped: {_err}")
            return ""

    def seed_canon_motivations(self, character: str, canon_motivations: list[str]) -> int:
        """Seed initial motivations from character_profiles.motivation_core. Idempotent."""
        count = 0
        for goal in canon_motivations:
            mid = self.upsert_motivation(character, goal, origin="canon", priority=0.7, urgency=0.2, decay_rate=0.005)
            if mid:
                count += 1
        return count

    def audit_recent_ooc_against_canon(self, character: str, recent_replies: list[dict] | None = None, *, shadow: bool = True, run_id: str = "") -> dict:
        char = resolve_agent(character)
        rows = [dict(x) for x in (recent_replies or []) if isinstance(x, dict)]
        canon = self.build_character_canon_profile(char)
        anchors = canon.get("anchors", {}) if isinstance(canon.get("anchors", {}), dict) else {}
        evidence_hits = []
        style_deviation = {"too_long": 0.0, "too_soft": 0.0, "too_cold": 0.0, "voice_leak": 0.0, "tone_mismatch": 0.0}
        _style_rules = setting("ooc_style_rules", _DEFAULT_OOC_STYLE_RULES) or {}
        total = max(1, len(rows))
        for item in rows[:40]:
            text = str(item.get("reply_text", item.get("text", "")) or "").strip()
            if not text:
                continue
            # ── too_long: 回复过长（各角色基准不同）──
            if len(text) > int(_style_rules.get("max_len", {}).get(char, _style_rules.get("default_max_len", 80))):
                style_deviation["too_long"] += 1.0
            # ── too_soft: 过度甜腻/亲密（规则表里标记的角色不该如此）──
            _soft_markers = ("喜欢你", "抱抱", "亲亲", "最爱你", "亲死你", "抱紧", "宝贝", "好想你", "么么")
            if char in _style_rules.get("no_soft", ()) and any(w in text for w in _soft_markers):
                style_deviation["too_soft"] += 1.0
            # ── too_cold: 过度冷漠/敷衍（规则表里标记的角色不该如此）──
            _cold_markers = ("关我什么事", "懒得理", "不想聊", "滚", "无所谓", "不关心")
            if char in _style_rules.get("no_cold", ()) and any(w in text for w in _cold_markers):
                style_deviation["too_cold"] += 1.0
            # ── voice_leak: AI/助手身份泄露（所有角色）──
            _leak_markers = ("我是AI", "我是模型", "我是助手", "我是程序", "作为AI", "语言模型", "我没有感情", "我无法感受")
            if any(w in text for w in _leak_markers):
                style_deviation["voice_leak"] += 1.0
            # ── tone_mismatch: 语气词不符合角色（规则表里的禁用语气词）──
            _banned_tone = tuple(_style_rules.get("banned_tone", {}).get(char, ()))
            if _banned_tone:
                _tone_hits = sum(text.count(p) for p in _banned_tone)
                if _tone_hits > 0:
                    style_deviation["tone_mismatch"] += min(2.0, _tone_hits * 0.5)
            hit_claims = []
            for ctype, ctexts in anchors.items():
                if not isinstance(ctexts, list):
                    continue
                best = 0.0
                best_txt = ""
                for ctext in ctexts[:5]:
                    sim = _lexical_similarity(text, str(ctext or ""))
                    if sim > best:
                        best = sim
                        best_txt = str(ctext or "")
                if best >= 0.18 and best_txt:
                    hit_claims.append({"claim_type": ctype, "claim_text": best_txt[:120], "score": round(best, 3)})
            if hit_claims:
                evidence_hits.append({"reply_text": text[:180], "hits": hit_claims[:4]})
        ooc_score = round(_clamp(sum(float(v or 0.0) for v in style_deviation.values()) / float(total), 0.0, 1.0), 3)
        high_conflict = [k for k, v in style_deviation.items() if float(v or 0.0) >= max(1.0, total * 0.18)]
        penalty = {
            "persona_drift_risk_delta": round(min(0.28, ooc_score * 0.36), 3),
            "style_budget_delta": round(min(0.22, ooc_score * 0.28), 3),
            "style_deviation_axes": {k: round(min(0.2, float(v or 0.0) / float(total) * 0.2), 3) for k, v in style_deviation.items()},
            "recommendation": "tighten_internal_style_control" if ooc_score >= 0.18 else "observe_only",
        }
        audit_id = "ooc_" + uuid.uuid4().hex[:12]
        now = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO ooc_audit_runs(audit_id, character, run_id, ooc_score, style_deviation_json, evidence_hits_json, high_conflict_claims_json, penalty_json, applied, shadow, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?);
                    """,
                    (audit_id, char, str(run_id or "").strip(), ooc_score, _json_dumps(style_deviation), _json_dumps(evidence_hits[:30]), _json_dumps(high_conflict[:12]), _json_dumps(penalty), 1 if bool(shadow) else 0, now, now),
                )
                conn.commit()
            finally:
                conn.close()
        return {"audit_id": audit_id, "character": char, "ooc_score": ooc_score, "style_deviation_axes": penalty.get("style_deviation_axes", {}), "evidence_hits": evidence_hits[:20], "high_conflict_claims": high_conflict[:12], "penalty_recommendation": penalty, "shadow": bool(shadow)}

    def apply_ooc_penalty(self, character: str, audit_result: dict | None = None) -> dict:
        char = resolve_agent(character)
        audit = audit_result if isinstance(audit_result, dict) else {}
        audit_id = str(audit.get("audit_id", "") or "").strip()
        penalty = audit.get("penalty_recommendation", {}) if isinstance(audit.get("penalty_recommendation", {}), dict) else {}
        if not penalty and audit_id:
            with self._lock:
                conn = self._connect()
                try:
                    row = conn.execute("SELECT penalty_json FROM ooc_audit_runs WHERE audit_id=? LIMIT 1;", (audit_id,)).fetchone()
                finally:
                    conn.close()
            if row is not None:
                penalty = _json_loads(str(row["penalty_json"] or "{}"), {})
        axes = penalty.get("style_deviation_axes", {}) if isinstance(penalty.get("style_deviation_axes", {}), dict) else {}
        return {"ok": True, "character": char, "audit_id": audit_id, "shadow_only": True, "persona_drift_risk_delta": round(_safe_float(penalty.get("persona_drift_risk_delta", 0.0), 0.0), 3), "style_budget_delta": round(_safe_float(penalty.get("style_budget_delta", 0.0), 0.0), 3), "style_deviation_axes": {str(k): round(_safe_float(v, 0.0), 3) for k, v in axes.items()}}

    def record_proactive_event(
        self,
        *,
        kind: str,
        sent: bool,
        reason: str = "",
        session_key: str = "",
        risk_score: float = 0.0,
        dedupe_key: str = "",
        decision_bundle: dict | None = None,
    ) -> None:
        hist = self._get_meta_json("proactive_events", [])
        if not isinstance(hist, list):
            hist = []
        now = _utc_now_iso()
        hist.append(
            {
                "at": now,
                "kind": str(kind or "").strip() or "unknown",
                "sent": bool(sent),
                "reason": str(reason or "").strip()[:100],
                "session_key": str(session_key or "").strip()[:80],
                "risk_score": round(_clamp(_safe_float(risk_score, 0.0)), 3),
                "dedupe_key": str(dedupe_key or "").strip()[:180],
            }
        )
        if len(hist) > 600:
            hist = hist[-600:]
        self._set_meta_json("proactive_events", hist)
        if bool(sent) and str(dedupe_key or "").strip():
            di = self._get_meta_json("proactive_dedupe_index", {})
            if not isinstance(di, dict):
                di = {}
            di[str(dedupe_key or "").strip()[:180]] = now
            if len(di) > 2000:
                # keep newest roughly by lexical order of ts values
                pairs = sorted([(k, str(v)) for k, v in di.items()], key=lambda x: x[1], reverse=True)[:1200]
                di = {k: v for k, v in pairs}
            self._set_meta_json("proactive_dedupe_index", di)
        if isinstance(decision_bundle, dict) and decision_bundle:
            self._log_policy_decision(
                decision_kind="proactive_send",
                phase="event",
                run_id=str(session_key or "")[:40],
                input_payload={"kind": str(kind or ""), "sent": bool(sent)},
                final_payload=decision_bundle,
            )

    def _assess_proactive_budget_heuristic(
        self,
        *,
        kind: str,
        idle_sec: float,
        session_key: str = "",
        base_risk: float = 0.0,
        proactive_policy: dict | None = None,
    ) -> dict:
        ppol = proactive_policy if isinstance(proactive_policy, dict) else {}
        same_window_sec = max(60, _safe_int(ppol.get("same_session_window_sec", 20 * 60), 20 * 60))
        global_window_sec = max(300, _safe_int(ppol.get("global_window_sec", 2 * 3600), 2 * 3600))
        max_sent_same = max(1, _safe_int(ppol.get("max_sent_same_session", 1), 1))
        max_sent_global = max(1, _safe_int(ppol.get("max_sent_global", 3), 3))
        idle_followup_min_idle_sec = max(60, _safe_int(ppol.get("idle_followup_min_idle_sec", 8 * 60), 8 * 60))
        risk_threshold = _safe_float(ppol.get("risk_block_threshold", 0.66), 0.66)
        quality_threshold = _safe_float(ppol.get("quality_block_threshold", 0.36), 0.36)
        schedule_conflict_block_threshold = _safe_float(ppol.get("schedule_conflict_block_threshold", 0.60), 0.60)

        now = utc_now_naive()
        hist = self._get_meta_json("proactive_events", [])
        if not isinstance(hist, list):
            hist = []

        recent_global = []
        recent_same_session = []
        for it in hist[-300:]:
            if not isinstance(it, dict):
                continue
            dt = _parse_dt(str(it.get("at", "") or ""))
            if dt is None:
                continue
            age = (now - dt).total_seconds()
            if age <= float(global_window_sec):
                recent_global.append(it)
            if str(session_key or "").strip() and str(it.get("session_key", "") or "").strip() == str(session_key or "").strip() and age <= float(same_window_sec):
                recent_same_session.append(it)

        sent_global = len([x for x in recent_global if bool(x.get("sent", False))])
        sent_same = len([x for x in recent_same_session if bool(x.get("sent", False))])

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT score, llm_score
                    FROM retrieval_logs
                    WHERE selected=1
                    ORDER BY log_id DESC
                    LIMIT 40;
                    """
                ).fetchall()
            finally:
                conn.close()
        quality = 0.55
        if rows:
            vals = []
            for r in rows:
                vals.append(0.75 * _safe_float(r["score"], 0.0) + 0.25 * _safe_float(r["llm_score"], 0.0))
            quality = _clamp(sum(vals) / max(1.0, float(len(vals))))

        risk = _clamp(
            0.40 * _clamp(_safe_float(base_risk, 0.0))
            + 0.25 * _clamp(sent_global / max(1.0, float(max_sent_global + 1)))
            + 0.20 * _clamp(sent_same / max(1.0, float(max_sent_same + 1)))
            + 0.15 * (1.0 - quality)
        )

        reasons = []
        allow = True
        suggested_delay_sec = 0

        if sent_same >= max_sent_same:
            allow = False
            reasons.append("recent_same_session_proactive")
            suggested_delay_sec = max(suggested_delay_sec, int(max(6 * 60, same_window_sec * 0.75)))
        if sent_global >= max_sent_global:
            allow = False
            reasons.append("high_global_proactive_frequency")
            suggested_delay_sec = max(suggested_delay_sec, int(max(12 * 60, global_window_sec * 0.35)))
        if quality < quality_threshold:
            allow = False
            reasons.append("low_memory_retrieval_quality")
            suggested_delay_sec = max(suggested_delay_sec, 12 * 60)
        if _safe_float(idle_sec, 0.0) < float(idle_followup_min_idle_sec) and str(kind or "").strip().lower() == "idle_followup":
            allow = False
            reasons.append("insufficient_idle_window")
            suggested_delay_sec = max(suggested_delay_sec, int(float(idle_followup_min_idle_sec)))
        if risk >= risk_threshold:
            allow = False
            reasons.append("memory_budget_risk_high")
            suggested_delay_sec = max(suggested_delay_sec, 18 * 60)
        aux = proactive_policy.get("_aux_signals", {}) if isinstance(proactive_policy, dict) else {}
        if isinstance(aux, dict):
            schedule_conflict = _safe_float(aux.get("schedule_conflict_risk", 0.0), 0.0)
            if schedule_conflict >= schedule_conflict_block_threshold:
                allow = False
                reasons.append("schedule_conflict_risk_high")
                suggested_delay_sec = max(suggested_delay_sec, 20 * 60)

        return {
            "allow": bool(allow),
            "risk_score": round(risk, 3),
            "quality_score": round(quality, 3),
            "recent_sent_global": int(sent_global),
            "recent_sent_same_session": int(sent_same),
            "reasons": reasons[:8],
            "suggested_delay_sec": int(suggested_delay_sec),
            "policy": {
                "same_session_window_sec": int(same_window_sec),
                "global_window_sec": int(global_window_sec),
                "max_sent_same_session": int(max_sent_same),
                "max_sent_global": int(max_sent_global),
                "risk_block_threshold": float(risk_threshold),
                "quality_block_threshold": float(quality_threshold),
            },
        }

    def evaluate_proactive_policy(
        self,
        *,
        kind: str,
        idle_sec: float,
        session_key: str = "",
        base_risk: float = 0.0,
        dedupe_key: str = "",
        hard_gate: dict | None = None,
        llm_policy: Callable[[dict], dict] | None = None,
        aux_signals: dict | None = None,
    ) -> dict:
        profile = self._load_policy_profile()
        ppol = profile.get("proactive_policy", {}) if isinstance(profile.get("proactive_policy", {}), dict) else {}
        sfb = profile.get("safety_fallbacks", {}) if isinstance(profile.get("safety_fallbacks", {}), dict) else {}
        min_send_interval_sec = max(120, _safe_int(ppol.get("min_send_interval_sec", 10 * 60), 10 * 60))
        input_payload = {
            "kind": str(kind or ""),
            "idle_sec": float(max(0.0, _safe_float(idle_sec, 0.0))),
            "session_key": str(session_key or "")[:80],
            "base_risk": float(_safe_float(base_risk, 0.0)),
            "dedupe_key": str(dedupe_key or "")[:180],
            "hard_gate": hard_gate if isinstance(hard_gate, dict) else {},
            "aux_signals": aux_signals if isinstance(aux_signals, dict) else {},
        }
        reasons = []
        fallback_reason = ""

        # 1) Hard safety gate
        hg = hard_gate if isinstance(hard_gate, dict) else {}
        if hg and (not bool(hg.get("allow", True))):
            reasons.extend([str(x or "") for x in (hg.get("reasons", []) if isinstance(hg.get("reasons", []), list) else []) if str(x or "").strip()])
            final_hard = {
                "allow": False,
                "risk_score": max(float(_safe_float(base_risk, 0.0)), float(_safe_float(hg.get("risk_score", 0.0), 0.0))),
                "reasons": sorted(set(reasons))[:10],
                "suggested_delay_sec": int(_safe_int(hg.get("suggested_delay_sec", 0), 0)),
                "stage": "hard_safety",
            }
            self._log_policy_decision(
                decision_kind="proactive",
                phase="hard_safety",
                run_id=str(session_key or "")[:40],
                input_payload=input_payload,
                fallback_reason="hard_gate_block",
                final_payload=final_hard,
            )
            return final_hard

        # Dedupe safety
        if str(dedupe_key or "").strip():
            di = self._get_meta_json("proactive_dedupe_index", {})
            if isinstance(di, dict):
                seen_ts = _parse_dt(str(di.get(str(dedupe_key or "").strip(), "") or ""))
                if seen_ts is not None:
                    age = (utc_now_naive() - seen_ts).total_seconds()
                    if age < float(min_send_interval_sec):
                        final_dedupe = {
                            "allow": False,
                            "risk_score": max(float(_safe_float(base_risk, 0.0)), 0.78),
                            "reasons": ["dedupe_key_recently_sent"],
                            "suggested_delay_sec": int(max(60, float(min_send_interval_sec) - age)),
                            "stage": "hard_safety_dedupe",
                        }
                        self._log_policy_decision(
                            decision_kind="proactive",
                            phase="hard_safety",
                            run_id=str(session_key or "")[:40],
                            input_payload=input_payload,
                            fallback_reason="dedupe_block",
                            final_payload=final_dedupe,
                        )
                        return final_dedupe

        # 2) LLM policy decision (optional)
        llm_out = {}
        if bool(mem_policy_llm_enabled()) and callable(llm_policy):
            try:
                llm_out = llm_policy(
                    {
                        "kind": str(kind or ""),
                        "idle_sec": float(max(0.0, _safe_float(idle_sec, 0.0))),
                        "session_key": str(session_key or "")[:80],
                        "base_risk": float(_safe_float(base_risk, 0.0)),
                        "hard_gate": hg,
                        "aux_signals": aux_signals if isinstance(aux_signals, dict) else {},
                    }
                )
                if not isinstance(llm_out, dict):
                    llm_out = {}
                    fallback_reason = "llm_invalid_json"
            except Exception:
                llm_out = {}
                fallback_reason = "llm_timeout_or_error"

        # 3) Heuristic fallback baseline
        h = self._assess_proactive_budget_heuristic(
            kind=str(kind or ""),
            idle_sec=float(max(0.0, _safe_float(idle_sec, 0.0))),
            session_key=str(session_key or ""),
            base_risk=float(_safe_float(base_risk, 0.0)),
            proactive_policy=_deep_merge_dict(ppol, {"_aux_signals": aux_signals if isinstance(aux_signals, dict) else {}}),
        )
        allow = bool(h.get("allow", True))
        risk = float(_safe_float(h.get("risk_score", base_risk), base_risk))
        suggested_delay_sec = int(_safe_int(h.get("suggested_delay_sec", 0), 0))
        reasons.extend([str(x or "") for x in (h.get("reasons", []) if isinstance(h.get("reasons", []), list) else []) if str(x or "").strip()])

        if isinstance(llm_out, dict) and llm_out:
            llm_allow = _parse_bool(llm_out.get("allow", True), True)
            llm_risk = _safe_float(llm_out.get("risk_score", risk), risk)
            llm_delay = _safe_int(llm_out.get("suggested_delay_sec", 0), 0)
            llm_reasons = [str(x or "") for x in (llm_out.get("reasons", []) if isinstance(llm_out.get("reasons", []), list) else []) if str(x or "").strip()]
            if not llm_allow:
                allow = False
                reasons.append("llm_policy_block")
            risk = max(risk, llm_risk)
            suggested_delay_sec = max(suggested_delay_sec, llm_delay)
            reasons.extend(llm_reasons[:4])
        elif fallback_reason in ("llm_timeout_or_error", "llm_invalid_json"):
            if str(sfb.get("on_llm_error", "") or "").strip().lower() == "conservative_block":
                allow = False
                suggested_delay_sec = max(suggested_delay_sec, 10 * 60)
                reasons.append("llm_fallback_conservative_block")

        final_out = {
            "allow": bool(allow),
            "risk_score": round(_clamp(risk), 3),
            "quality_score": float(_safe_float(h.get("quality_score", 0.55), 0.55)),
            "recent_sent_global": int(_safe_int(h.get("recent_sent_global", 0), 0)),
            "recent_sent_same_session": int(_safe_int(h.get("recent_sent_same_session", 0), 0)),
            "reasons": sorted(set(reasons))[:10],
            "suggested_delay_sec": int(max(0, suggested_delay_sec)),
            "stage": "mixed_policy",
            "memory_budget": h,
        }
        self._log_policy_decision(
            decision_kind="proactive",
            phase="final",
            run_id=str(session_key or "")[:40],
            input_payload=input_payload,
            llm_payload=llm_out if isinstance(llm_out, dict) else {},
            fallback_reason=fallback_reason,
            final_payload=final_out,
        )
        return final_out

    def assess_proactive_budget(
        self,
        *,
        kind: str,
        idle_sec: float,
        session_key: str = "",
        base_risk: float = 0.0,
    ) -> dict:
        return self.evaluate_proactive_policy(
            kind=str(kind or ""),
            idle_sec=float(max(0.0, _safe_float(idle_sec, 0.0))),
            session_key=str(session_key or ""),
            base_risk=float(_safe_float(base_risk, 0.0)),
        )
