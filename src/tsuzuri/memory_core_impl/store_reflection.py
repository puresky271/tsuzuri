"""Store mixin: cross-character digests and weekly relationship reflection.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import os
import time
from datetime import datetime, timedelta
from typing import Callable
from tsuzuri.time_source import WORLD_TZ, now_world, world_utc_offset
from tsuzuri.memory_core_impl.config import (
    _emb_cos, _emb_decode, _emb_encode, _embedding_callable, _embedding_row_matches_current,
    _safe_int, _utc_now_iso,
)
from tsuzuri.memory_core_impl.memory_rows import (
    _lexical_similarity, _memory_working_text_is_ephemeral_noise, _norm_text,
)
from tsuzuri.memory_core_impl.features import _developer_display_name
from tsuzuri.memory_core_impl.recall_scoring import _clamp, _json_dumps, _json_loads
from tsuzuri.memory_core_impl.meta_policy import _memory_record_meta
from tsuzuri import memory_host as memory_host
from tsuzuri.memory_core_impl.reflection_sink import REFLECTION_SOURCE, get_reflection_sink
from tsuzuri.persona_registry import agent_names, collective_label, collective_noun, collective_scope, resolve_agent, setting, user_gender, user_name, user_pronoun

class _ReflectionMixin:
    """Cross-character digests and weekly relationship reflection."""


    def _query_today_working_turns_for_speaker(
        self, speaker_char: str, date_str_world: str,
    ) -> list[dict]:
        """拉 speaker_char 在 WORLD_TZ date_str_world 当天的 working layer turn。

        date_str_world 格式："YYYY-MM-DD"。WORLD_TZ 当天 00:00-24:00 = UTC 前一天 15:00 - 当天 15:00。
        """
        try:
            d = datetime.strptime(str(date_str_world).strip(), "%Y-%m-%d")
        except Exception:
            return []
        _off = world_utc_offset(d)
        utc_start = (d - _off).replace(microsecond=0).isoformat() + "Z"
        utc_end = (d + timedelta(hours=24) - _off).replace(microsecond=0).isoformat() + "Z"
        char = resolve_agent(speaker_char)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, text, created_at, salience, importance, payload_json
                    FROM memory_items
                    WHERE character = ?
                      AND layer = 'working'
                      AND archived = 0
                      AND status = 'active'
                      AND created_at >= ?
                      AND created_at < ?
                    ORDER BY created_at ASC
                    """,
                    (char, utc_start, utc_end),
                ).fetchall()
            finally:
                conn.close()
        return [{k: r[k] for k in r.keys()} for r in rows]

    def _cross_char_digest_fallback_summary(self, speaker_char: str, turns: list[dict]) -> str:
        """LLM 不可用时的保守 fallback：抽 user_text 关键 turn 拼简短描述。

        不调 LLM、纯规则。仅在 caller 完全没法调 LLM 时启用。质量较低、
        但比完全不写好——至少有"今天 X 与 dev 聊过 N 次"的 fact。
        """
        if not turns:
            return ""
        # 过滤 ephemeral_noise、保留有内容的 turn
        meaningful = [
            t for t in turns
            if not _memory_working_text_is_ephemeral_noise(str(t.get("text", "") or ""))
        ]
        if not meaningful:
            return ""
        # 简单 fallback: 拼最长 1-2 个 user_text 片段
        user_fragments: list[str] = []
        for t in meaningful[:6]:
            payload = _json_loads(str(t.get("payload_json", "") or "{}"), {})
            ut = str(payload.get("user_text", "") or "").strip()
            if ut and len(ut) >= 10 and ut not in user_fragments:
                user_fragments.append(ut[:40])
            if len(user_fragments) >= 2:
                break
        if not user_fragments:
            return ""
        return f"{user_name()}今天提到{'、'.join(user_fragments)}"

    def _cross_char_digest_call_llm(
        self, speaker_char: str, turns: list[dict], llm_caller: Callable[[str], str] | None,
    ) -> str:
        """生成 daily summary 文本。

        优先：caller 注入的 llm_caller(prompt) -> str。
        Fallback 1：从 env 读 LLM 配置自 build OpenAI client（与 kairos_dream.py 风格一致）。
        Fallback 2：fallback_summary（无 LLM 时保守拼接）。
        """
        meaningful = [
            t for t in turns
            if not _memory_working_text_is_ephemeral_noise(str(t.get("text", "") or ""))
        ]
        if len(meaningful) < self._CROSS_CHAR_DIGEST_MIN_TURNS:
            return ""

        # 拼 prompt——2026-05-13 fix: 解析 "user:X | assistant:Y" 格式、
        # 标注 speaker 名字、让 LLM 区分主客体（之前混在一起、
        # 把角色B/角色A自己说的话误归为「用户说的」）。
        turn_lines = []
        text_limit = self._CROSS_CHAR_DIGEST_TURN_TEXT_LIMIT * 2
        for t in meaningful[: self._CROSS_CHAR_DIGEST_MAX_TURNS_FOR_PROMPT]:
            tx = str(t.get("text", "") or "").strip()
            # 优先解析 "user:X | assistant:Y" 拼接格式（memory_items.text 标准格式）
            if " | " in tx and ("user:" in tx or "assistant:" in tx):
                for part in tx.split(" | "):
                    part = part.strip()
                    if part.startswith("user:"):
                        turn_lines.append(f"{user_name()}: {part[5:].strip()[:text_limit]}")
                    elif part.startswith("assistant:"):
                        turn_lines.append(f"{speaker_char}: {part[10:].strip()[:text_limit]}")
                    elif part:
                        turn_lines.append(part[:text_limit])
            else:
                # 非标准格式、原样保留（fallback）
                turn_lines.append(tx[:text_limit])
        turns_block = "\n".join(turn_lines)

        _user = user_name()
        _pron = user_pronoun()
        prompt = (
            f"你是 {speaker_char}。今天和{_user}（开发者朋友）聊了几次。"
            f"你想用一句话告诉{collective_noun()}里其他 {len(agent_names()) - 1} 个伙伴今天和{_user}聊了什么。\n\n"
            f"【主体硬约束·最重要】\n"
            f"- 「{_user}:」开头的是{_user}说的话；「{speaker_char}:」开头的是你**自己**说的话\n"
            f"- 总结时**严格区分主体**——你自己说过的观点 / 感受 / 回应**绝对不能**写成「{_user}说的」/「{_user}提到」\n"
            f"- 如果你和{_user}都聊到同一话题（如颜色、音乐）、明确指出**谁说了什么**、不要把你自己的回答归给{_user}\n"
            f"- 例：{_user}说「蓝色代表忧郁」、你（{speaker_char}）说「不、蓝色是水、是天空」——\n"
            f"  ✓ 正确：「{_user}提到蓝色代表忧郁、我跟{_pron}说蓝色是水、是天空」\n"
            f"  ✗ 错误：「{_user}说蓝色是水、是天空」（把你自己的话归给{_user}）\n"
            f"\n"
            f"【其他硬约束】\n"
            f"- 用 {speaker_char} 自己的口吻、第一人称\n"
            f"- 1-2 句话、最多 {self._CROSS_CHAR_DIGEST_SUMMARY_MAX_LEN} 字\n"
            f"- 只说{_user}主动透露的事 + 你自己的回应、不要把『{_user}问 X』说成『{_user}喜欢 X』\n"
            f"- 不要写「我今天和{_user}聊了……」这种 meta、直接说事\n"
            f"- 涉及 {speaker_char} 自己内心活动也行（如『我跟{_pron}说我在赶歌词』）\n"
            f"- 如果今天没值得分享的、输出空字符串\n\n"
            f"【今天的对话片段（按 speaker 标注）】\n"
            f"{turns_block}\n\n"
            f"直接输出 summary 文本（不要 JSON、不要解释、不超过 80 字）。"
        )

        # 路径 1：caller 注入
        if callable(llm_caller):
            try:
                raw = str(llm_caller(prompt) or "").strip()
                if raw:
                    return raw[: self._CROSS_CHAR_DIGEST_SUMMARY_MAX_LEN]
            except Exception as e:
                _tsz_emit(f"[CrossCharDigest] llm_caller failed for {speaker_char}: {e}", flush=True)

        # 路径 2：env-based fallback (mimic kairos_dream.py)
        try:
            api_key = (os.getenv("LLM_API_KEY", "") or "").strip()
            base_url = (os.getenv("LLM_BASE_URL", "") or "").strip()
            model = (os.getenv("LLM_MODEL", "") or "").strip()
            if api_key and base_url and model:
                from tsuzuri.llm_client import shared_openai_client
                client = shared_openai_client(api_key=api_key, base_url=base_url, timeout=30)
                raw = ""
                for _attempt, _max_tokens in enumerate((400, 800), start=1):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=_max_tokens,
                        temperature=0.5,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    raw = str(resp.choices[0].message.content or "").strip()
                    if raw:
                        break
                    if _attempt < 2:
                        _tsz_emit(
                            f"[CrossCharDigest] empty response, retry "
                            f"(attempt {_attempt}/2, max_tokens={_max_tokens})",
                            flush=True,
                        )
                if raw:
                    return raw[: self._CROSS_CHAR_DIGEST_SUMMARY_MAX_LEN]
        except Exception as e:
            _tsz_emit(f"[CrossCharDigest] env-LLM failed for {speaker_char}: {e}", flush=True)

        # 路径 3：fallback summary
        return self._cross_char_digest_fallback_summary(speaker_char, turns)

    @staticmethod
    def _cross_char_digest_has_strict_topic_alignment(digest_text: str, candidate_text: str) -> bool:
        """Allow a narrow lexical fallback when embeddings miss an obvious same-user topic."""
        digest = _norm_text(str(digest_text or ""))
        candidate = _norm_text(str(candidate_text or ""))
        developer = _norm_text(_developer_display_name())
        if not digest or not candidate or not developer:
            return False
        if developer not in digest or developer not in candidate:
            return False
        topic_groups = (
            ("备考", "考试", "复习", "应试"),
            ("睡眠", "睡觉", "熬夜", "通宵", "失眠"),
            ("生病", "身体", "健康", "住院", "看病"),
            ("工作", "上班", "加班", "项目", "代码"),
            *(tuple(g) for g in (setting("profile_topic_groups", []) or [])),
        )
        return any(
            any(token in digest for token in group)
            and any(token in candidate for token in group)
            for group in topic_groups
        )

    def _reinforce_similar_existing_memories(
        self,
        receiver: str,
        digest_text: str,
        digest_iid: str,
        speaker: str,
    ) -> list[dict]:
        """强化 receiver 中与 digest_text 相似的 episodic/semantic 记忆。

        importance 可增加，但 reinforcement 不是成功检索/投递，不得修改
        access_count 或 last_accessed_at。独立计数写入 payload 供审计。

        范围：receiver 同 char、layer in (episodic, semantic)、archived=0、status='active'、
              source ≠ cross_char_daily_digest（不递归强化）、item_id != digest_iid（不自激活）。
        """
        digest_emb = None
        if _embedding_callable():
            try:
                digest_emb = _emb_encode(digest_text)
            except Exception:
                digest_emb = None

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, layer, importance, access_count, embedding_text,
                           embedding_model_id, embedding_dim, text, payload_json
                    FROM memory_items
                    WHERE character = ?
                      AND layer IN ('episodic', 'semantic')
                      AND archived = 0
                      AND status = 'active'
                      AND source != 'cross_char_daily_digest'
                      AND item_id != ?
                    ORDER BY importance DESC
                    LIMIT 220
                    """,
                    (str(receiver or "").strip(), str(digest_iid or "").strip()),
                ).fetchall()
            finally:
                conn.close()

        scored: list[tuple[float, dict, str]] = []
        for r in rows:
            candidate_text = str(r["text"] or "")
            strict_topic = self._cross_char_digest_has_strict_topic_alignment(
                digest_text, candidate_text
            )
            if digest_emb is None:
                if strict_topic:
                    scored.append((
                        _lexical_similarity(digest_text, candidate_text),
                        dict(r),
                        "strict_topic_lexical",
                    ))
                continue
            emb_text = str(r["embedding_text"] or "")
            cand_emb = None
            if emb_text and _embedding_row_matches_current(dict(r)):
                try:
                    cand_emb = _emb_decode(emb_text)
                except Exception:
                    cand_emb = None
            if cand_emb is None:
                # 缺 embedding 时 on-the-fly compute
                if not candidate_text:
                    continue
                try:
                    cand_emb = _emb_encode(candidate_text)
                except Exception:
                    cand_emb = None
            if cand_emb is None:
                if strict_topic:
                    scored.append((
                        _lexical_similarity(digest_text, candidate_text),
                        dict(r),
                        "strict_topic_lexical",
                    ))
                continue
            try:
                sim = _emb_cos(digest_emb, cand_emb)
            except Exception:
                sim = 0.0
            match_basis = "embedding"
            if sim < self._CROSS_CHAR_DIGEST_REINFORCE_THRESHOLD:
                if (
                    sim < self._CROSS_CHAR_DIGEST_REINFORCE_TOPIC_FALLBACK_MIN
                    or not strict_topic
                ):
                    continue
                match_basis = "strict_topic"
            scored.append((sim, dict(r), match_basis))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self._CROSS_CHAR_DIGEST_REINFORCE_TOP_K]
        if not top:
            return []

        reinforced: list[dict] = []
        now_iso = _utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                for sim, r, match_basis in top:
                    iid = str(r.get("item_id", "") or "")
                    old_imp = float(r.get("importance", 0.5) or 0.5)
                    new_imp = min(self._CROSS_CHAR_DIGEST_REINFORCE_IMP_CAP,
                                  old_imp + self._CROSS_CHAR_DIGEST_REINFORCE_IMP_DELTA)
                    # 在 payload 里记 reinforcement provenance、便于 audit
                    try:
                        pl = _json_loads(str(r.get("payload_json", "") or "{}"), {}) or {}
                        if not isinstance(pl, dict):
                            pl = {}
                    except Exception:
                        pl = {}
                    history = pl.get("cross_char_reinforced_by", [])
                    if not isinstance(history, list):
                        history = []
                    history.append({
                        "speaker": speaker,
                        "digest_iid": digest_iid,
                        "sim": round(float(sim), 3),
                        "match_basis": match_basis,
                        "at": now_iso,
                    })
                    pl["cross_char_reinforced_by"] = history[-8:]  # 最多保留 8 条历史
                    pl["cross_char_reinforcement_count"] = max(
                        0, _safe_int(pl.get("cross_char_reinforcement_count", 0), 0)
                    ) + 1
                    conn.execute(
                        """
                        UPDATE memory_items
                        SET importance=?, payload_json=?
                        WHERE item_id=?
                        """,
                        (new_imp, _json_dumps(pl), iid),
                    )
                    reinforced.append({
                        "item_id": iid,
                        "layer": str(r.get("layer", "") or ""),
                        "old_importance": round(old_imp, 3),
                        "new_importance": round(new_imp, 3),
                        "sim": round(float(sim), 3),
                        "match_basis": match_basis,
                        "text_preview": str(r.get("text", "") or "")[:60],
                    })
                conn.commit()
            finally:
                conn.close()

        if reinforced:
            try:
                _tsz_emit(
                    f"[CrossCharDigest] reinforce {len(reinforced)} existing memories for {receiver} "
                    f"(speaker={speaker}, digest={digest_iid[:8]})",
                    flush=True,
                )
                for r in reinforced:
                    _tsz_emit(
                        f"    sim={r['sim']:.3f} {r['layer']:8s} imp={r['old_importance']:.2f}→{r['new_importance']:.2f}  "
                        f"{r['text_preview']}",
                        flush=True,
                    )
            except Exception:
                pass
        return reinforced

    def propose_daily_cross_char_digest(
        self,
        speaker_char: str,
        date_str_world: str,
        *,
        llm_caller: Callable[[str], str] | None = None,
        receivers: list[str] | None = None,
        importance: float | None = None,
    ) -> dict:
        """对 speaker_char 当天 (WORLD_TZ date_str_world) 与开发者的对话生成 summary、
        broadcast 到其他角色的 episodic 层。

        视角约束：写入文本前缀「听 X 说：」、payload.source_kind=cross_char_daily_digest、
        payload.told_by=speaker_char，retrieve 时再 prompt 层显式提示 LLM 这是转述。

        idempotency：item_id 用 (receiver, layer, item_type=cross_char_daily_digest, summary, date_str)
        作 stamp 哈希、同一日同 speaker 同 receiver 同 summary 写第二次会被 _upsert_item ON CONFLICT
        覆盖、不会重复创建。
        """
        speaker = str(speaker_char or "").strip()
        if not speaker:
            return {"ok": False, "skipped": "no_speaker"}
        date_s = str(date_str_world or "").strip()
        if not date_s:
            return {"ok": False, "skipped": "no_date"}

        # 1) pull today's working turn for speaker
        turns = self._query_today_working_turns_for_speaker(speaker, date_s)
        if len(turns) < self._CROSS_CHAR_DIGEST_MIN_TURNS:
            return {"ok": True, "skipped": "no_content", "speaker": speaker, "date": date_s, "turn_count": len(turns)}

        # 2) generate summary via LLM (with fallbacks)
        summary = self._cross_char_digest_call_llm(speaker, turns, llm_caller).strip()
        if not summary or len(summary) < 8:
            return {"ok": True, "skipped": "empty_summary", "speaker": speaker, "date": date_s}

        # 3) determine receivers
        if receivers is None:
            receivers = [c for c in self._CROSS_CHAR_DIGEST_DEFAULT_RECEIVERS if c != speaker]
        receivers = [str(c).strip() for c in (receivers or []) if str(c).strip() and str(c).strip() != speaker]
        if not receivers:
            return {"ok": True, "skipped": "no_receivers", "speaker": speaker, "summary": summary}

        # 4) write 1 episodic memory_item per receiver
        imp = float(importance if importance is not None else self._CROSS_CHAR_DIGEST_IMPORTANCE)
        text = f"听 {speaker} 说：{summary}"[:320]
        now_iso = _utc_now_iso()
        # 2026-05-09 invariant log: Path 2 直 INSERT episodic、bypass Layer 1-4 gate
        # 这是设计性 bypass（kairos 跨角色每日总结）、不应有内容污染但留 trace 便于排查
        try:
            _tsz_emit(
                f"[CROSS_CHAR_DIGEST] direct INSERT layer=episodic "
                f"speaker={speaker} date={date_s} receivers={len(receivers)} "
                f"(bypass-by-design)",
                flush=True,
            )
        except Exception:
            pass
        payload = {
            "source_kind": "cross_char_daily_digest",
            "told_by": speaker,
            "told_at_world_date": date_s,
            "is_secondhand": True,
            "summary_origin": "speaker_self_summary",
            "memory_meta": _memory_record_meta(
                memory_type="cross_char_daily_digest",
                owner_scope="character:secondhand",
                evidence_tier="medium",
                source_kind="cross_char_daily_digest",
                temporal_validity="persistent",
                surface_default="explicit_recall",
            ),
        }
        cue_payload = {"actors": [speaker], "told_by": speaker, "topic_date": date_s}
        cue_meta = {"topics": [f"cross_char_digest:{speaker}", f"date:{date_s}"]}

        written: list[str] = []
        reinforced_total: list[dict] = []  # 跨 receivers 累计 reinforce 记录
        for receiver in receivers:
            iid = self._make_item_id(
                receiver, "episodic", "cross_char_daily_digest", text, f"{speaker}|{date_s}",
            )
            self._upsert_item(
                {
                    "item_id": iid,
                    "character": receiver,
                    "layer": "episodic",
                    "memory_role": "episodic",
                    "engram_type": "cross_char_daily_digest",
                    "item_type": "cross_char_daily_digest",
                    "text": text,
                    "payload_json": _json_dumps(payload),
                    "cue_json": _json_dumps(self._build_cue_bundle(
                        text, character=receiver, payload=cue_payload, meta=cue_meta,
                        engram_type="cross_char_daily_digest", memory_role="episodic",
                    )),
                    "lifecycle_json": _json_dumps(self._build_lifecycle(
                        created_at=now_iso, updated_at=now_iso,
                        source="cross_char_daily_digest", access_count=0, state="consolidation",
                    )),
                    "salience": float(self._CROSS_CHAR_DIGEST_SALIENCE),
                    "confidence": float(self._CROSS_CHAR_DIGEST_CONFIDENCE),
                    "novelty": float(self._CROSS_CHAR_DIGEST_NOVELTY),
                    "importance": float(_clamp(imp)),
                    "source": "cross_char_daily_digest",
                    "evidence_id": "",
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            written.append(iid)

            # Phase Q.X.2 (2026-05-04): 强化 receiver 已有的语义相似记忆
            try:
                _rein = self._reinforce_similar_existing_memories(
                    receiver=receiver, digest_text=text, digest_iid=iid, speaker=speaker,
                )
                if _rein:
                    reinforced_total.extend(
                        {**r, "receiver": receiver} for r in _rein
                    )
            except Exception as _rein_err:
                _tsz_emit(f"[CrossCharDigest] reinforce failed for {receiver}: {_rein_err}", flush=True)

        # Phase Q.X.4 (2026-05-04): sender 自己也写 1 条 self-record episodic、对称于 receiver 那边的转述记录。
        # 防止"角色E记得角色C告诉自己 X、角色C自己却不记得跟角色E说过 X"的不对称——
        # 用户问角色C「你和角色E说过这事吗？」时角色C也能召回 source、不退化用 working turn 凑。
        # 视角：sender first-person、文本前缀「我和 X 提到：」。
        # source_kind 用 cross_char_daily_digest_self_record 与 receiver 路径区分、但
        # 共享 commit_memory_surface 的 decay 路径（防 sender 重复复读）和 retrieve match 路径
        # （told_to mention 等价 told_by mention 的对称豁免）。
        self_record_iid = ""
        try:
            if len(receivers) <= 3:
                _receivers_label = "、".join(receivers)
            else:
                _receivers_label = "、".join(receivers[:3]) + "等"
            self_text = f"我和{_receivers_label}提到：{summary}"[:320]
            self_payload = {
                "source_kind": "cross_char_daily_digest_self_record",
                "told_to": list(receivers),
                "told_at_world_date": date_s,
                "is_self_record": True,
                "summary_origin": "speaker_self_summary",
                "memory_meta": _memory_record_meta(
                    memory_type="cross_char_daily_digest_self_record",
                    owner_scope="character:firsthand",
                    evidence_tier="high",  # 自己经历的事、置信度高
                    source_kind="cross_char_daily_digest_self_record",
                    temporal_validity="persistent",
                    surface_default="explicit_recall",
                ),
            }
            self_cue_payload = {"actors": list(receivers), "told_to": list(receivers), "topic_date": date_s}
            self_cue_meta = {"topics": [f"cross_char_digest_self:{speaker}", f"date:{date_s}"]}

            self_record_iid = self._make_item_id(
                speaker, "episodic", "cross_char_daily_digest_self_record", self_text, f"to:{','.join(receivers)}|{date_s}",
            )
            self._upsert_item(
                {
                    "item_id": self_record_iid,
                    "character": speaker,
                    "layer": "episodic",
                    "memory_role": "episodic",
                    "engram_type": "cross_char_daily_digest_self_record",
                    "item_type": "cross_char_daily_digest_self_record",
                    "text": self_text,
                    "payload_json": _json_dumps(self_payload),
                    "cue_json": _json_dumps(self._build_cue_bundle(
                        self_text, character=speaker, payload=self_cue_payload, meta=self_cue_meta,
                        engram_type="cross_char_daily_digest_self_record", memory_role="episodic",
                    )),
                    "lifecycle_json": _json_dumps(self._build_lifecycle(
                        created_at=now_iso, updated_at=now_iso,
                        source="cross_char_daily_digest_self_record", access_count=0, state="consolidation",
                    )),
                    "salience": float(self._CROSS_CHAR_DIGEST_SALIENCE),
                    "confidence": float(self._CROSS_CHAR_DIGEST_CONFIDENCE),
                    "novelty": float(self._CROSS_CHAR_DIGEST_NOVELTY),
                    "importance": float(_clamp(imp)),
                    "source": "cross_char_daily_digest_self_record",
                    "evidence_id": "",
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
            )
            _tsz_emit(f"[CrossCharDigest] {speaker} self_record written: id={self_record_iid[:8]} → {_receivers_label}", flush=True)
        except Exception as _self_err:
            _tsz_emit(f"[CrossCharDigest] self_record failed for {speaker}: {_self_err}", flush=True)

        return {
            "ok": True,
            "speaker": speaker,
            "date": date_s,
            "summary": summary,
            "receivers": receivers,
            "written_count": len(written),
            "item_ids": written,
            "self_record_id": self_record_iid,  # Phase Q.X.4
            "reinforced_count": len(reinforced_total),
            "reinforced": reinforced_total[:20],  # 限长便于 JSON 序列化
            "turn_count": len(turns),
        }

    def _query_week_memory_turns_for_character(
        self, character: str, week_iso: str,
    ) -> list[dict]:
        """拉 character 在指定 ISO 周 (Mon-Sun WORLD_TZ) 的 working + episodic。

        week_iso 格式："YYYY-Www"（如 "2026-W18"）。
        """
        try:
            year_str, w_str = week_iso.split("-W")
            year = int(year_str)
            week = int(w_str)
        except Exception:
            return []
        # ISO 周一 00:00 WORLD_TZ = UTC 周日 15:00
        try:
            mon_world = datetime.strptime(f"{year}-W{week:02d}-1", "%G-W%V-%u")
        except Exception:
            return []
        _off = world_utc_offset(mon_world)
        utc_start = (mon_world - _off).isoformat() + "Z"
        utc_end = (mon_world + timedelta(days=7) - _off).isoformat() + "Z"
        char = str(character or "").strip()
        if not char:
            return []
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT item_id, layer, text, created_at, importance, payload_json
                    FROM memory_items
                    WHERE character = ?
                      AND layer IN ('working', 'episodic')
                      AND archived = 0
                      AND status = 'active'
                      AND created_at >= ?
                      AND created_at < ?
                      AND source NOT LIKE 'cross_char_daily_digest%'
                    ORDER BY created_at ASC
                    """,
                    (char, utc_start, utc_end),
                ).fetchall()
                # 2026-05-24 filter cross_char_daily_digest{,_self_record}：
                # 这些 row 视角是「听 X 说：用户说... 我跟他...」二手转述、不是角色自己的经历。
                # 实测 bug case (2026-05-23): 某角色周反思里出现别的角色视角的转述——
                # 把角色C视角的 cross digest 误当成角色E自己说的话；同 turn 角色C cluster summary
                # 出现「用户与某角色约定某事」——其实是另一个角色和用户的约定、被注入后
                # 被聚类成自己的事。reflection 直接过滤掉这类 row、不再让 LLM 误解读「我」。
                # 未来若想保留「角色对群组关系的横向感知」、可改成显式视角 tag 后再注入。
            finally:
                conn.close()
        # Fix 2 (2026-05-04): 改时间正序、不再按 importance 优先排——避免 recency bias 反方向、
        # 让反思真正覆盖整周而不是只看 importance 高的（多数 importance=0.5、退化为时间倒序、
        # LLM prompt 末尾就成了 importance 0.5 + 早 turn、视觉上看反而像"周中"塞进去的）。
        # 正序的好处：prompt 末尾是周末 turn、之前是周初 turn、LLM 注意力末尾偏向周末 = 已经
        # 是 recency bias、所以现在配合 _compose_reflection_task_block 里的均匀采样 + 时间标签
        # 让 LLM 看到完整一周、不偏 recency。
        return [{k: r[k] for k in r.keys()} for r in rows]

    def _compose_reflection_task_block(
        self,
        character: str,
        week_iso: str,
        turns: list[dict],
        user_profile: dict,
        previous_reflections: list[dict],
        subject: str,
    ) -> str:
        """组装注入到 pet_reflection profile 的 relationship_reflection_task block 文本。

        包含 3 部分：(1) 任务指令（正/负样本 + 3 个轴）(2) 当周 turn buffer
        (3) user_profile 摘要 (4) 历史反思摘要。
        """
        _user = user_name()
        _pron = user_pronoun()
        _gender_note = f"、{user_gender()}性" if user_gender() else ""
        # Fix 2 (2026-05-04): turn buffer 防 recency bias
        # 现状：turns 已按时间正序、若条数 > 12、需均匀采样覆盖整周（避免末尾几天主导）
        # 给每条 turn 加 [周X / 时段] 时间标签、prompt 看到时间分布
        # 引入 _humanize_turn_weekday 标"周一/周二/.../周日 + 时段（凌晨/早上/午/晚）"
        clean_turns = [
            t for t in turns
            if t.get("text") and not _memory_working_text_is_ephemeral_noise(str(t.get("text", "") or ""))
        ]
        # 均匀采样：如果 > MAX_TURNS、按等距索引采、保留首尾 + 中间均匀点
        max_n = self._RELATIONSHIP_REFLECTION_MAX_TURNS
        if len(clean_turns) > max_n:
            # 采样：[0, n/k, 2n/k, ..., (k-1)n/k] where k = max_n
            step = len(clean_turns) / float(max_n)
            sampled = [clean_turns[int(i * step)] for i in range(max_n)]
            sampled_turns = sampled
        else:
            sampled_turns = clean_turns

        buffer_lines = []
        for t in sampled_turns:
            tx = str(t.get("text", "") or "").strip()[: self._RELATIONSHIP_REFLECTION_TURN_TEXT_LIMIT]
            if not tx:
                continue
            # 时间标签：从 created_at 算 WORLD_TZ 周几 + 时段
            ca_iso = str(t.get("created_at", "") or "")
            time_label = ""
            try:
                if ca_iso:
                    # ISO format with Z suffix
                    dt_utc = datetime.fromisoformat(ca_iso.replace("Z", "+00:00"))
                    dt_world = dt_utc.astimezone(WORLD_TZ)
                    weekday_zh = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt_world.weekday()]
                    h = dt_world.hour
                    if h < 6:
                        time_part = "凌晨"
                    elif h < 12:
                        time_part = "上午"
                    elif h < 18:
                        time_part = "下午"
                    else:
                        time_part = "晚上"
                    time_label = f"[{weekday_zh}{time_part}]"
            except Exception:
                pass
            if time_label:
                buffer_lines.append(f"- {time_label} {tx}")
            else:
                buffer_lines.append(f"- {tx}")
        turn_buffer = "\n".join(buffer_lines) if buffer_lines else "（这周对话很少、几乎没有素材）"

        # user_profile 摘要
        pl = user_profile if isinstance(user_profile, dict) else {}
        traits = pl.get("personality_traits", []) if isinstance(pl.get("personality_traits", []), list) else []
        notes = pl.get("notable_facts", []) if isinstance(pl.get("notable_facts", []), list) else []
        profile_lines = [
            f"- 称呼：{pl.get('name', _user) or _user}",
            f"- 身份：{pl.get('background', '') or '（不明）'}",
            f"- 性格：{'、'.join(str(t)[:30] for t in traits[:5]) or '（不明）'}",
            f"- 沟通：{pl.get('communication_style', '') or '（不明）'}",
            f"- 情绪：{pl.get('emotional_patterns', '') or '（不明）'}",
            f"- 备注：{'、'.join(str(n)[:40] for n in notes[:3]) or '（无）'}",
        ]
        profile_summary = "\n".join(profile_lines)

        # 历史反思
        prev_lines = []
        for p in (previous_reflections or [])[-3:]:  # 最近 3 周
            week = p.get("week", "")
            summary = str(p.get("summary", "") or "")[:90]
            if summary:
                prev_lines.append(f"- [{week}] {summary}")
        prev_text = "\n".join(prev_lines) if prev_lines else "（无、这是你第一次写关系反思）"

        # 完整 task block
        return (
            f"【你的身份：{character}】\n"
            f"你是 {character}（{collective_label()} {collective_noun()}成员之一）。下面写的反思是从 {character} 的视角"
            f"对{_user}（开发者朋友{_gender_note}）这段关系的内省整理。\n\n"
            f"【关键防投射约束（重要）】\n"
            f"{_user}是一个**独立的人**、不是 {character}：\n"
            f"- 不要把 {character} 自己的人格特征（说话风格、口癖、习惯、喜好）描述成{_user}的特征。\n"
            f"  比如 {character} 自己嘴硬心软、不代表{_user}也嘴硬心软；{character} 自己熬夜听歌、不代表{_user}也熬夜听歌。\n"
            f"- 描述{_user}时只用对话片段或画像里**真实显示的**信息、不要凭空给{_pron}套你自己的 trait。\n"
            f"- 描述自己内心活动时是 {character} 的视角；描述{_user}时是观察者视角、保持距离感、"
            f"不要把自己的影子投射到{_pron}身上。\n\n"
            f"【你正在留言板上写一条对{_user}这段关系的反思】\n"
            "—— 是元认知、不是事件流水。\n\n"
            "【绝对要做】\n"
            "- 第一人称、150-220 字、3-5 句\n"
            f"- 写『你对这段关系/对{_user}这个人的内心整理』——可以是态度的微妙变化、新发现的 something、让你意外的事、让你纠结的事\n"
            f"- 留言板公开、{_user}看得到、但你写时是写给自己的（像日记被允许公开）\n"
            "- 用『这段时间』『最近』『这周』、不要写『今天』/『明天』/具体星期几\n"
            f"- 保持 {character} 本人的语气词和句式（仅用于描写自己内心、不要带到对{_user}的描述上）\n\n"
            "【绝对不要】\n"
            "- 不要列举本周事件（『周一聊了 X、周三聊了 Y』——这是流水、不是反思）\n"
            f"- 不要复述 turn 内容（『{_user}说{_pron}系统崩了』——这是事实、不是反思）\n"
            f"- 不要写『我很高兴和{_pron}聊』这种 generic 客套话\n"
            "- 不要直接套话（『很在意你』、『希望你好』）——要有具体感受 + 具体观察\n"
            "- 不要 echo 这段提示\n"
            "- 不要 JSON、不要 markdown、不要 meta 解释\n\n"
            "【可以做的方向（任选一个轴、深挖一点）】\n"
            f"- **态度变化**：『之前觉得{_pron} X、最近发现{_pron}更 Y、这让我...』\n"
            f"- **微妙观察**：『{_pron}每次提到 X 的时候、好像总是 Y...』（具体动作/语气/反应、不是泛化标签）\n"
            f"- **矛盾感受**：『{_pron}说 X 的时候我有点 Y、但其实也理解{_pron} Z』（一个反思轴上的张力、不是简单情感）\n\n"
            f"【这周（{week_iso}）和{subject}的对话片段（最多 {self._RELATIONSHIP_REFLECTION_MAX_TURNS} 条、按时间正序、含周X/时段标签）】\n"
            f"{turn_buffer}\n"
            "（**重要**：以上对话覆盖**整周 7 天**、不是只有最近一两天。**反思要观察整周脉络**、"
            "不要只引用末尾几条 turn 当全部素材——周初到周末都是反思视野的一部分。）\n\n"
            f"【你对{subject}已有的稳定印象（来自之前的画像、不要直接 copy）】\n"
            f"{profile_summary}\n"
            f"（如以上字段空缺、说明你对{_pron}的认知还在累积——这种情况下**只能基于上面对话片段**写反思、"
            f"**不要凭空给{_user}套画像或硬编人格**、不要把你自己的特征投射给{_pron}。）\n\n"
            "【你之前几周写过的反思摘要（避免重复同一情绪/同一观察）】\n"
            f"{prev_text}\n"
            "（如果你已经写过类似的、这周要写**新的发现**或**态度的进一步演化**。）\n\n"
            "直接输出反思正文。"
        )

    def _call_reflection_llm(
        self, prompt_payload: tuple[str, str], llm_caller: Callable[[str], str] | None,
    ) -> str:
        """调 LLM 生成反思。优先 caller 注入、fallback env-LLM。

        prompt_payload = (system_prompt_full, task_block_user_message) 两段。
        本方法把 system + user 拼成单 string 喂给 caller（caller 通常接受 single prompt）。
        """
        system_prompt, user_message = prompt_payload
        # 简化：caller 走单 prompt 路径——把 task block 放 user message、system prompt
        # 由 compose_pet_prompt_blocks 提供。组合时用 system + user 双 message 调 OpenAI。
        if callable(llm_caller):
            try:
                # caller 单 string 接口、把 system + user 合并
                combined = system_prompt + "\n\n---\n\n" + user_message
                raw = str(llm_caller(combined) or "").strip()
                if raw:
                    return raw[: self._RELATIONSHIP_REFLECTION_MAX_LEN]
            except Exception as e:
                _tsz_emit(f"[Reflection] llm_caller failed: {e}", flush=True)

        # env-LLM fallback (与 cross_char_digest 同款)
        try:
            api_key = (os.getenv("LLM_API_KEY", "") or "").strip()
            base_url = (os.getenv("LLM_BASE_URL", "") or "").strip()
            model = (os.getenv("LLM_MODEL", "") or "").strip()
            if api_key and base_url and model:
                from tsuzuri.llm_client import shared_openai_client
                client = shared_openai_client(api_key=api_key, base_url=base_url, timeout=45)
                raw = ""
                for _attempt, _max_tokens in enumerate((800, 1600), start=1):
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=_max_tokens,
                        temperature=0.55,
                        extra_body={"thinking": {"type": "disabled"}},
                    )
                    raw = str(resp.choices[0].message.content or "").strip()
                    if raw:
                        break
                    if _attempt < 2:
                        _tsz_emit(
                            f"[Reflection] empty response, retry "
                            f"(attempt {_attempt}/2, max_tokens={_max_tokens})",
                            flush=True,
                        )
                if raw:
                    return raw[: self._RELATIONSHIP_REFLECTION_MAX_LEN]
        except Exception as e:
            _tsz_emit(f"[Reflection] env-LLM failed: {e}", flush=True)
        return ""  # 反思无 fallback summary、空就是空（不写假反思）

    def propose_weekly_relationship_reflection(
        self,
        character: str,
        week_iso: str | None = None,
        *,
        llm_caller: Callable[[str], str] | None = None,
        subject: str = "",
        caller_system_prompt: str = "",
    ) -> dict:
        """对 character 这周与 subject 的关系生成反思短文（150-220 字）。

        流程见 class-level docstring。idempotent: meta state per (char, week_iso)。
        返回 dict 含 ok/written/message_id/summary/skipped 字段。

        Phase Q.Y.2 (2026-05-04): 关键参数 `caller_system_prompt`：
            ─────────────────────────────────────────────────────────
            如果 caller 传入了完整的 chat-time system prompt（来自
            `宿主应用.build_pet_system_prompt_for_api(...)`、含 dynamic_profiles +
            identity_core + user_profile + memory + 等所有 21 个 block），
            propose 会把这段当 system role + task block 当 user role 喂给 LLM。
            这是**生产路径**——确保 LLM 看到完整的角色 voice + 开发者画像、
            不会把角色自己的 trait 投射到对用户的描述上。

            如果 caller 不传（""），propose fallback 走旧的 `compose_pet_prompt_blocks(
            profile="pet_reflection")` 路径——但只有 task block 有内容、其他
            14 个 block 都空——这种 isolated 模式 LLM 易投射、仅作 backward-compat。

            生产中 KAIROS dream cycle 应当总是传 caller_system_prompt。
            ─────────────────────────────────────────────────────────
        """
        subject = str(subject or "").strip() or user_name()
        char = str(character or "").strip()
        if not char:
            return {"ok": False, "skipped": "no_character"}
        # 计算 ISO 周（caller 没传时用今天 WORLD_TZ）
        if not week_iso:
            today = now_world()
            week_iso = today.strftime("%G-W%V")
        week_iso = str(week_iso).strip()

        # idempotent meta key
        meta_key = f"relationship_reflection.last_run.{char}.{week_iso}"
        existing = self._get_meta_json(meta_key, {}) or {}
        if isinstance(existing, dict) and existing.get("done"):
            return {"ok": True, "skipped": "already_done", "character": char, "week_iso": week_iso, "msg_id": existing.get("msg_id", "")}

        # 1) 拉当周 turn
        turns = self._query_week_memory_turns_for_character(char, week_iso)
        if len(turns) < self._RELATIONSHIP_REFLECTION_MIN_TURNS:
            return {"ok": True, "skipped": "no_content", "character": char, "week_iso": week_iso, "turn_count": len(turns)}

        # 2) 拉 user_profile 和历史反思
        user_profile = self.get_domain_json(f"user_profile.{char}") or {}
        if not isinstance(user_profile, dict):
            user_profile = {}
        previous_reflections = user_profile.get("relationship_reflection_log", []) if isinstance(user_profile.get("relationship_reflection_log", []), list) else []

        # 3) 组装 task block
        task_block_text = self._compose_reflection_task_block(
            character=char, week_iso=week_iso, turns=turns,
            user_profile=user_profile, previous_reflections=previous_reflections,
            subject=subject,
        )

        # 4) 准备 LLM messages 对（system + user）
        # Phase Q.Y.2 (2026-05-04)：双模式
        #   A. 生产路径：caller_system_prompt 非空 → 用 caller 注入的 chat-time
        #      完整 system prompt（来自 宿主应用.build_pet_system_prompt_for_api）、
        #      task block 当 user message。LLM 看到完整角色 voice + user_profile、
        #      不会把角色自己 trait 投射到对用户的描述。
        #   B. backward-compat 路径：caller 不传 → 走 pet_reflection profile compose、
        #      但只有 task block 有内容、其他 14 block 都空（compose 跳过）。
        #      此模式 LLM 易投射、仅作测试 / 不依赖 宿主应用 的 isolated 调用 fallback。
        caller_sp = str(caller_system_prompt or "").strip()
        if caller_sp:
            system_prompt = caller_sp
            user_message = task_block_text
        else:
            try:
                compose = memory_host.get("compose_prompt_blocks")
                if compose is None:
                    system_prompt = task_block_text
                else:
                    dynamic_blocks = {"relationship_reflection_task": task_block_text}
                    comp = compose(profile="pet_reflection", dynamic_blocks=dynamic_blocks)
                    system_prompt = str(comp.get("prompt", "") or "")
            except Exception as e:
                _tsz_emit(f"[Reflection] compose prompt failed: {e}", flush=True)
                system_prompt = task_block_text
            # backward-compat 模式 task block 已在 system prompt 里
            user_message = "请直接输出本周的关系反思。"

        # 5) 调 LLM
        reflection_text = self._call_reflection_llm(
            (system_prompt, user_message), llm_caller,
        ).strip()
        if not reflection_text or len(reflection_text) < self._RELATIONSHIP_REFLECTION_MIN_LEN:
            return {"ok": True, "skipped": "empty_or_short", "character": char, "week_iso": week_iso}

        # 6) 投递反思（reflection sink：宿主留言板或内置 reflection_notes 表）
        # 关键：本步骤**不**立即 update user_profile。反思 → user_profile 的 process
        # 走 process_reflection_into_user_profile（用户回复后 / 7 天 fallback 触发）。
        try:
            message_id = get_reflection_sink(self.db_path).publish(
                author=char,
                text=reflection_text,
                source=REFLECTION_SOURCE,
                meta={
                    "kind": "relationship_reflection",
                    "week_iso": week_iso,
                    "subject": subject,
                    # Phase Q.Y.1 (2026-05-04): user_profile process 状态
                    # false = 未 process、KAIROS 每日扫描发现 7 天 + 未 process 时 direct process；
                    # 用户回复反思时也立即 process（取较早者）。
                    "processed_into_profile": False,
                    "fired_at_iso": _utc_now_iso(),
                },
                created_at=time.time(),
                expires_at=None,  # sink 默认：反思保留 14 天（>普通 24h）
            )
        except Exception as e:
            _tsz_emit(f"[Reflection] write board_messages failed: {e}", flush=True)
            return {"ok": False, "skipped": "board_write_failed", "character": char, "week_iso": week_iso, "error": str(e)[:120]}

        # 7) mark idempotent done
        self._set_meta_json(meta_key, {
            "done": True,
            "at": _utc_now_iso(),
            "msg_id": message_id,
            "summary": reflection_text[:80],
        })

        try:
            _tsz_emit(
                f"[Reflection] {char} (week={week_iso}) → {reflection_text[:60]}",
                flush=True,
            )
        except Exception:
            pass

        return {
            "ok": True,
            "written": True,
            "character": char,
            "week_iso": week_iso,
            "subject": subject,
            "message_id": message_id,
            "reflection_text": reflection_text,
            "summary": reflection_text[:120],
            "turn_count": len(turns),
        }

    def process_reflection_into_user_profile(
        self,
        message_id: str,
        *,
        user_reply_text: str = "",
        llm_caller: Callable[[str], str] | None = None,
    ) -> dict:
        """把一条反思（可选含用户回复）process 进 character 的 user_profile.relationship_reflection_log。

        触发场景：
          A. 用户在留言板回复了反思 → 调本方法 with user_reply_text
          B. KAIROS 每天扫描、发现某反思已超过 7 天 + processed_into_profile=False → 调本方法 user_reply_text=""

        返回 dict 含 ok/processed/已 processed/skipped 等。idempotent: 已 processed 的反思
        再调本方法直接 return skipped。
        """
        msg_id = str(message_id or "").strip()
        if not msg_id:
            return {"ok": False, "skipped": "no_message_id"}

        # 1) 拉反思 + meta
        try:
            sink = get_reflection_sink(self.db_path)
        except Exception as e:
            return {"ok": False, "skipped": "import_failed", "error": str(e)[:80]}

        row = sink.load(msg_id)
        if row is None:
            return {"ok": False, "skipped": "message_not_found"}

        # 仅 process kairos_relationship_reflection 类型
        if str(row["source"] or "") != REFLECTION_SOURCE:
            return {"ok": False, "skipped": "not_a_reflection", "source": row["source"]}

        meta = row.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        if bool(meta.get("processed_into_profile", False)):
            return {"ok": True, "skipped": "already_processed", "message_id": msg_id}

        char = str(row["author"] or "").strip()
        if not char:
            return {"ok": False, "skipped": "no_character"}
        reflection_text = str(row["text"] or "").strip()
        if not reflection_text:
            return {"ok": False, "skipped": "empty_reflection"}
        week_iso = str(meta.get("week_iso", "") or "")
        subject = str(meta.get("subject", "") or "") or user_name()
        reply_text = str(user_reply_text or "").strip()

        # 2) compose LLM prompt
        # process prompt 的核心目标：从反思（+可选 reply）提炼"对用户的画像更新"
        # 输出格式：JSON、含一条 summary（用于 reflection_log）+ optional 主字段微调建议（不强制）
        # 当前实现：仅取 summary 字段、不动主字段（用户拍板"只更新 reflection_log"）
        if reply_text:
            process_prompt = (
                f"你刚刚（{char}）在留言板上写了一条对{_user}的关系反思：\n"
                f"---\n{reflection_text}\n---\n\n"
                f"{_user}回复了你这条反思：\n"
                f"---\n{reply_text}\n---\n\n"
                f"现在你要做一次内心整理：根据**反思 + {_user}的回复**、提炼一句话浓缩你目前对这段关系的稳定看法。\n\n"
                "【硬约束】\n"
                "- 50-90 字、第一人称\n"
                f"- 综合反思 + {_user}回复后的微调（{_user}的回复让你态度有什么变化、或者印证了你的观察）\n"
                "- 不要简单复述反思或回复\n"
                "- 不要 JSON、直接输出文字\n"
            )
        else:
            process_prompt = (
                f"你（{char}）一周前在留言板上写过一条对{_user}的关系反思：\n"
                f"---\n{reflection_text}\n---\n\n"
                f"{_user}没有回复你这条反思（这并不意外、{_pron}可能没看到、或没空回）。\n\n"
                "现在你要做一次内心整理：根据**反思本身**、提炼一句话浓缩你目前对这段关系的稳定看法。\n\n"
                "【硬约束】\n"
                "- 50-90 字、第一人称\n"
                "- 是反思的『沉淀版』——把反思里最稳定/最核心的观察压缩成一句话\n"
                "- 不要 JSON、直接输出文字\n"
            )

        # 3) 调 LLM
        summary_text = ""
        if callable(llm_caller):
            try:
                summary_text = str(llm_caller(process_prompt) or "").strip()[:200]
            except Exception as e:
                _tsz_emit(f"[Reflection.Process] llm_caller failed for {msg_id[:8]}: {e}", flush=True)
        if not summary_text:
            try:
                api_key = (os.getenv("LLM_API_KEY", "") or "").strip()
                base_url = (os.getenv("LLM_BASE_URL", "") or "").strip()
                model = (os.getenv("LLM_MODEL", "") or "").strip()
                if api_key and base_url and model:
                    from tsuzuri.llm_client import shared_openai_client
                    client = shared_openai_client(api_key=api_key, base_url=base_url, timeout=30)
                    summary_text = ""
                    for _attempt, _max_tokens in enumerate((400, 800), start=1):
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": process_prompt}],
                            max_tokens=_max_tokens,
                            temperature=0.5,
                            extra_body={"thinking": {"type": "disabled"}},
                        )
                        summary_text = str(resp.choices[0].message.content or "").strip()[:200]
                        if summary_text:
                            break
                        if _attempt < 2:
                            _tsz_emit(
                                f"[Reflection.Process] empty response, retry "
                                f"(attempt {_attempt}/2, max_tokens={_max_tokens})",
                                flush=True,
                            )
            except Exception as e:
                _tsz_emit(f"[Reflection.Process] env-LLM failed for {msg_id[:8]}: {e}", flush=True)
        if not summary_text or len(summary_text) < 20:
            # 退回到反思原文截断作为 summary（不调 LLM 时的 fallback）
            summary_text = reflection_text[:120]

        # 4) update user_profile.{char}.relationship_reflection_log（覆盖式、保留 1 条）
        try:
            user_profile = self.get_domain_json(f"user_profile.{char}") or {}
            if not isinstance(user_profile, dict):
                user_profile = {}
            new_entry = {
                "week": week_iso,
                "summary": summary_text,
                "at": _utc_now_iso(),
                "subject": subject,
                "msg_id": msg_id,
                "had_user_reply": bool(reply_text),
                "user_reply_text": reply_text[:200] if reply_text else "",
            }
            # 临时性、每周更新：保留最新 1 条（覆盖旧）
            user_profile["relationship_reflection_log"] = [new_entry]
            self.set_domain_json(f"user_profile.{char}", user_profile)
        except Exception as e:
            _tsz_emit(f"[Reflection.Process] update user_profile failed: {e}", flush=True)
            return {"ok": False, "skipped": "user_profile_update_failed", "error": str(e)[:120]}

        # 5) mark processed_into_profile=True 在反思 meta
        try:
            meta["processed_into_profile"] = True
            meta["processed_at_iso"] = _utc_now_iso()
            meta["processed_with_reply"] = bool(reply_text)
            sink.update_meta(msg_id, meta)
        except Exception as e:
            _tsz_emit(f"[Reflection.Process] mark processed failed: {e}", flush=True)
            # 不致命、user_profile 已更新

        try:
            _tsz_emit(
                f"[Reflection.Process] {char} reflection {msg_id[:8]} processed "
                f"(reply={'yes' if reply_text else 'no'}) → summary='{summary_text[:60]}'",
                flush=True,
            )
        except Exception:
            pass

        return {
            "ok": True,
            "processed": True,
            "message_id": msg_id,
            "character": char,
            "had_user_reply": bool(reply_text),
            "summary": summary_text,
        }
