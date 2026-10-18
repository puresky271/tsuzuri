"""MemoryCoreStore assembly, process singleton and appointment helpers.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import os
import sqlite3
import threading
from tsuzuri.sqlite_reliability import connect_sqlite
from tsuzuri.persona_registry import agent_set_view
from tsuzuri.time_source import local_now, now_world

from tsuzuri.memory_core_impl.store_storage import _StorageMixin
from tsuzuri.memory_core_impl.store_items import _ItemsMixin
from tsuzuri.memory_core_impl.store_turn_writes import _TurnWritesMixin
from tsuzuri.memory_core_impl.store_retrieval import _RetrievalMixin
from tsuzuri.memory_core_impl.store_evolution import _EvolutionMixin
from tsuzuri.memory_core_impl.store_reflection import _ReflectionMixin
from tsuzuri.memory_core_impl.store_rethink import _RethinkMixin
from tsuzuri.memory_core_impl.store_lifecycle import _LifecycleMixin



class MemoryCoreStore(_StorageMixin, _ItemsMixin, _TurnWritesMixin, _RetrievalMixin, _EvolutionMixin, _ReflectionMixin, _RethinkMixin, _LifecycleMixin):
    def __init__(self, db_path: str, snapshot_path: str, *, bootstrap_from_snapshot: bool = True):
        self.db_path = db_path
        self.snapshot_path = snapshot_path
        self._lock = threading.RLock()
        self._health_lock = threading.Lock()
        self._connection_count = 0
        self._last_quick_check_report: dict = {}
        schema_backup = self._backup_before_schema_migration_if_needed()
        try:
            self._ensure_schema()
        except Exception as schema_error:
            if schema_backup:
                try:
                    from tsuzuri.runtime_backup import restore_sqlite

                    restore_sqlite(schema_backup, self.db_path)
                    _tsz_emit(
                        f"[MemoryCore][Schema] migration failed and database was restored "
                        f"from {schema_backup}: {schema_error}",
                        flush=True,
                    )
                except Exception as restore_error:
                    raise RuntimeError(
                        f"schema migration failed ({schema_error}); automatic restore failed "
                        f"({restore_error}); backup={schema_backup}"
                    ) from schema_error
            raise
        from tsuzuri.memory_entities import ensure_entity_schema
        ensure_entity_schema(self)
        if bootstrap_from_snapshot:
            self._bootstrap_from_snapshot_if_needed()
        self._backfill_native_mechanisms_if_needed()
        # 诊断日志年龄保留：建表以来无清理、实测已到 ~200MB，启动时节流式清过期行。
        self._maybe_prune_retrieval_logs()
        # shared_info 零投递死行回收（gossip 健康调查 §9-1，同款启动节流）。
        self._maybe_prune_stale_shared_info()
        # 2026-05-17 turn-scoped recall dedup cache
        # 单 turn 内同 (char, query, query_bundle 指纹) 的并发重复召回（双 caller / 异步路径）
        # 直接复用结果、避开 32s+ 全链路重复（bge-m3 + reranker + scoring）。
        #
        # 2026-05-23 TTL 10s → 60s：实测 web 路径第一次 retrieve 含 bge-m3 + reranker
        # cold load 可达 32s、写入 cache 时刻起算、 10s TTL 几乎立刻就 evict、
        # 第二次同 query 必 miss、跑全流程。改 60s = cover 一次完整 prompt build
        # 加 LLM 流式开始之前的 turn 窗口；过期前若 caller 又触发同 query 命中、
        # 跨 turn 污染风险低（不同 turn user query 几乎一定不同）。
        self._recall_inflight_cache: dict[tuple, tuple[float, dict]] = {}
        self._recall_inflight_lock = threading.Lock()
        self._recall_inflight_ttl_sec = 60.0

    # ── retrieval_logs 年龄保留 ──
    # 纯诊断日志（每次召回每候选一行），建表以来无任何清理、实测单表 ~200MB（≈ 6.6MB/天）。
    # 策略：年龄保留（默认 7 天，MEMORY_RETRIEVAL_LOG_RETENTION_DAYS 可调）+
    # 启动节流（默认 24h 一次，MEMORY_RETRIEVAL_LOG_PRUNE_INTERVAL_HOURS）。
    # DELETE 只释放页复用、不缩文件；确需回收磁盘用 vacuum=True（一次性、独占）。
    RETRIEVAL_LOG_PRUNE_STATE_KEY = "retrieval_logs_last_prune"

    # ── shared_info 零投递死行回收 ──
    # 2026-08-27 gossip 链路健康调查（docs/brain/memory/gossip_runtime_health_survey.md §9-1）：
    # 旧链路死行（delivered_to_json 空、超过 48h 新鲜度窗口）永远读不到也永远不会被
    # mark_delivered 打 expired 标（那只发生在送达达 max_mentions 时），持续污染
    # 频率/时效统计。策略：超窗死行置 expired=1（保留行做审计、不物理删除）+
    # 启动节流（默认 24h 一次，MEMORY_SHARED_INFO_PRUNE_INTERVAL_HOURS 可调、<=0 关闭）。
    SHARED_INFO_STALE_PRUNE_STATE_KEY = "shared_info_stale_last_prune"

    # ─────────────────────────────────────────────────────────────────
    # Phase Q.X (2026-05-04) — Cross-character daily digest broadcast
    # ─────────────────────────────────────────────────────────────────
    # 各角色之间没有共享当天与开发者的对话认知。开发者上午跟角色C聊备考、
    # 下午跟角色A聊"我累死了"——角色A不知道"累=备考"。每晚 KAIROS dream cycle
    # 跑一次、为每个 speaker 生成一句话总结、broadcast 到其他角色的
    # episodic 层、用「听 X 说」前缀显式标记 secondhand 视角。
    #
    # 复用基础设施：episodic layer / _upsert_item / retrieve 主链路、
    # 不动 gossip_queue / shared_info（它们留作高 salience 即时 channel）。
    _CROSS_CHAR_DIGEST_DEFAULT_RECEIVERS = agent_set_view()
    _CROSS_CHAR_DIGEST_IMPORTANCE = 0.70  # plan spec 3
    _CROSS_CHAR_DIGEST_SALIENCE = 0.65
    _CROSS_CHAR_DIGEST_CONFIDENCE = 0.72  # 朋友转述、可信度高于推测
    _CROSS_CHAR_DIGEST_NOVELTY = 0.55
    _CROSS_CHAR_DIGEST_MIN_TURNS = 2
    _CROSS_CHAR_DIGEST_MAX_TURNS_FOR_PROMPT = 10
    _CROSS_CHAR_DIGEST_TURN_TEXT_LIMIT = 60
    _CROSS_CHAR_DIGEST_SUMMARY_MAX_LEN = 80
    # Phase Q.X.5 (2026-05-06) — 加大"被成功应用后"的消退惩罚
    # 原 Phase Q.X.1 只把 importance 一次性 0.70→0.55，但 salience 仍 0.65 ≥
    # 当时旧阈值 promote_salience_min (0.60) + access_count >= 1 → episodic 直接 promote
    # 到 semantic，二手转告变永久 fact。修复：first delivery 同时降 imp + sal、
    # repeat delivery 累积乘性衰减、地板防止彻底归零。
    #
    # 二手转述在首次引用后降低显著性，但保留足够强度供用户主动追问来源；
    # 后续重复引用继续衰减，并由 delivered_count 的独立门槛最终归档。
    _CROSS_CHAR_DIGEST_DELIVERED_IMPORTANCE_FIRST = 0.40
    _CROSS_CHAR_DIGEST_DELIVERED_SALIENCE_FIRST = 0.40
    _CROSS_CHAR_DIGEST_DELIVERED_REPEAT_DECAY = 0.80
    _CROSS_CHAR_DIGEST_DELIVERED_FLOOR = 0.15
    # 2026-05-13 新增：delivered_count ≥ N 时自动 archive
    # 二手记忆复用 3 次后就该彻底沉默、留 active 没意义
    _CROSS_CHAR_DIGEST_DELIVERED_AUTO_ARCHIVE_COUNT = 3

    # Phase Q.X.2 (2026-05-04) — cross_char_digest 写入后、强化 receiver 自己已有的语义相似记忆。
    # 现实人脑：朋友转告"X 在备考"时、自己之前已知的"X 也跟我提过备考"那条记忆会被激活 + 强化、
    # 不是第一次听说。该机制让 receiver 自己的"亲历"记忆 importance + 0.05、access_count + 1、
    # 让它在 retrieve 排序中更靠前、并满足 episodic→semantic promote 条件（acc≥1）。
    # 2026-05-13 收紧：从 0.60 → 0.75。
    # 实测 0.60 太松——角色A「历史课明治维新」记忆 vs 角色B「蓝色诗集」digest sim=0.618 过线、
    # 但主题完全不同 → 误强化。0.75 是 sentence-transformers 中文 emb 真正"同主题"的阈值。
    _CROSS_CHAR_DIGEST_REINFORCE_THRESHOLD = 0.75
    _CROSS_CHAR_DIGEST_REINFORCE_TOPIC_FALLBACK_MIN = 0.68
    _CROSS_CHAR_DIGEST_REINFORCE_TOP_K = 3
    _CROSS_CHAR_DIGEST_REINFORCE_IMP_DELTA = 0.05
    _CROSS_CHAR_DIGEST_REINFORCE_IMP_CAP = 0.85

    # ─────────────────────────────────────────────────────────────────
    # Phase Q.Y (2026-05-04) — 留言板「关系反思」生成
    # ─────────────────────────────────────────────────────────────────
    # 与 cross_char_digest（事件总结）不同、反思是元认知：「我对这段关系的看法」。
    # 比 auto_greet 慢（每周）、比 cross_char_digest 深（不是事实而是态度）、
    # 比 gossip 元（不传 fact、写自己的内省）。
    #
    # 流程：
    #   KAIROS 周日凌晨 dream cycle → propose_weekly_relationship_reflection(char)
    #     → 拉当周 working/episodic + user_profile + 历史反思
    #     → compose_pet_prompt_blocks(profile="pet_reflection") 装 system prompt
    #     → LLM 生成 150-220 字反思
    #     → 写 board_messages (direction=character_to_user, source=kairos_relationship_reflection)
    #     → append 到 user_profile.{char}.relationship_reflection_log
    _RELATIONSHIP_REFLECTION_MIN_TURNS = 3  # 当周对话 < 3 条 → 不写反思
    _RELATIONSHIP_REFLECTION_MAX_TURNS = 12  # prompt 注入 buffer 上限
    _RELATIONSHIP_REFLECTION_TURN_TEXT_LIMIT = 80
    _RELATIONSHIP_REFLECTION_MIN_LEN = 50   # LLM 输出 < 50 字视为 fallback（角色E极简 voice canon、不强行拉长）
    _RELATIONSHIP_REFLECTION_MAX_LEN = 500  # > 500 字截断（2026-05-10：原 280 对话密度高的角色（角色A 35 turns）会撞顶被切、提到 500）
    _RELATIONSHIP_REFLECTION_LOG_KEEP = 4   # user_profile 里保留最近 4 周

    # ─────────────────────────────────────────────────────────────────
    # Phase Q.Y.1 (2026-05-04) — 反思 → user_profile 的"process"步骤
    # ─────────────────────────────────────────────────────────────────
    # 用户拍板（2026-05-04）：
    #   - 反思生成时不立即更新 user_profile、留 7 天窗口期
    #   - 用户在窗口期内**只能回复一次**；回复后立即 process
    #   - 7 天到期、用户没回复 → KAIROS 自动 fallback direct process（无 reply）
    #   - process 一次性、之后不再 update（即使有更多 reply）
    #   - 只更新 reflection_log（不动主字段）
    #   - reflection_log 仅保留**最近 1 条**（每周更新覆盖、不累积）

    _RELATIONSHIP_REFLECTION_PROCESS_DELAY_DAYS = 7  # 7 天 fallback 阈值
    _RELATIONSHIP_REFLECTION_LOG_KEEP_AFTER_PROCESS = 1  # 每周更新覆盖、保留最新 1 条


_MEMORY_CORE_SINGLETON: MemoryCoreStore | None = None
_MEMORY_CORE_LOCK = threading.RLock()


def get_memory_core(*, snapshot_path: str, db_path: str | None = None) -> MemoryCoreStore:
    global _MEMORY_CORE_SINGLETON
    with _MEMORY_CORE_LOCK:
        if _MEMORY_CORE_SINGLETON is not None:
            return _MEMORY_CORE_SINGLETON
        db = str(db_path or "").strip()
        if not db:
            db = str(os.getenv("MEM_SQLITE_PATH", "") or "").strip()
        if not db:
            base_dir = os.path.dirname(os.path.abspath(snapshot_path))
            db = os.path.join(base_dir, "memory_core.db")
        _MEMORY_CORE_SINGLETON = MemoryCoreStore(db_path=db, snapshot_path=snapshot_path)
        return _MEMORY_CORE_SINGLETON


# ══════════════════════════════════════════════════════════════════
# 约定追踪 (User Appointments)
# ══════════════════════════════════════════════════════════════════

import re as _appt_re
from datetime import datetime as _appt_dt, timedelta as _appt_td

_APPOINTMENT_PATTERN = _appt_re.compile(
    r"(?:下次|明天|后天|大后天|周[一二三四五六日末]|下[个周].{0,4}[一二三四五六日末]|以后|改天|下回|过几天|月底).{0,20}(?:一起|约|去|来|看|听|吃|玩|做|买|逛|聊)",
    _appt_re.IGNORECASE,
)

_CONFIRM_PATTERN = _appt_re.compile(
    r"(好[呀啊的嘞]|没问题|可以[呀啊]|说[定好]了|约[定好]了|一言为定|就这么[定办]|嗯[！!]|好[！!]|OK|当然|好吧)",
    _appt_re.IGNORECASE,
)

_WEEKDAY_MAP = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6, "末": 5}


def _parse_appointment_date(text: str, now: _appt_dt | None = None) -> tuple[str, str]:
    """从文本中解析约定日期。返回 (due_date_str, due_period)。"""
    if now is None:
        now = now_world()
    t = str(text or "")

    if "明天" in t:
        d = now + _appt_td(days=1)
        return d.strftime("%Y-%m-%d"), "tomorrow"
    if "后天" in t:
        d = now + _appt_td(days=2)
        return d.strftime("%Y-%m-%d"), "day_after_tomorrow"
    if "大后天" in t:
        d = now + _appt_td(days=3)
        return d.strftime("%Y-%m-%d"), "3_days_later"

    # 周X / 下周X
    for label, wd in _WEEKDAY_MAP.items():
        if _appt_re.search(rf"(?:下[个周]?)?周{label}", t):
            days_ahead = (wd - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # "周X" 默认指下一个
            if "下" in t:
                days_ahead += 7
            d = now + _appt_td(days=days_ahead)
            return d.strftime("%Y-%m-%d"), f"weekday_{label}"

    if "周末" in t:
        days_to_sat = (5 - now.weekday()) % 7
        if days_to_sat == 0:
            days_to_sat = 7
        if "下" in t:
            days_to_sat += 7
        d = now + _appt_td(days=days_to_sat)
        return d.strftime("%Y-%m-%d"), "weekend"

    if "月底" in t:
        import calendar
        last_day = calendar.monthrange(now.year, now.month)[1]
        d = now.replace(day=last_day)
        if d <= now:
            if now.month == 12:
                d = now.replace(year=now.year + 1, month=1, day=calendar.monthrange(now.year + 1, 1)[1])
            else:
                d = now.replace(month=now.month + 1, day=calendar.monthrange(now.year, now.month + 1)[1])
        return d.strftime("%Y-%m-%d"), "month_end"

    # 模糊约定
    return "", "someday"


def extract_appointment(user_msg: str, assistant_reply: str, character: str, user_id: int = 0, now: _appt_dt | None = None) -> dict | None:
    """从对话中提取约定。需要用户提出 + 角色确认。返回 dict 或 None。"""
    if not _APPOINTMENT_PATTERN.search(user_msg):
        return None
    if not _CONFIRM_PATTERN.search(assistant_reply):
        return None
    # 提取约定内容摘要
    m = _APPOINTMENT_PATTERN.search(user_msg)
    snippet = user_msg[max(0, m.start() - 5):m.end() + 10][:60]
    due_date, due_period = _parse_appointment_date(user_msg, now)
    import hashlib
    aid = hashlib.md5(f"{user_id}:{character}:{snippet}:{due_date}".encode()).hexdigest()[:16]
    return {
        "appointment_id": aid,
        "user_id": user_id,
        "character": character,
        "text": snippet.strip(),
        "due_date": due_date,
        "due_period": due_period,
    }


def save_appointment(db_path: str, appt: dict) -> bool:
    """保存约定到数据库。"""
    try:
        conn = connect_sqlite(db_path)
        conn.execute(
            "INSERT OR IGNORE INTO user_appointments (appointment_id, user_id, character, text, due_date, due_period, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (appt["appointment_id"], appt["user_id"], appt["character"], appt["text"],
             appt["due_date"], appt["due_period"], "pending", local_now().isoformat(timespec="seconds")),
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def get_due_appointments(db_path: str, today_str: str) -> list[dict]:
    """获取今天到期的约定。"""
    try:
        conn = connect_sqlite(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM user_appointments WHERE due_date=? AND status='pending'",
            (today_str,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_someday_appointments(db_path: str, character: str) -> list[dict]:
    """获取模糊约定（someday），用于低概率浮出。"""
    try:
        conn = connect_sqlite(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM user_appointments WHERE due_period='someday' AND status='pending' AND character=?",
            (character,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def mark_appointment_reminded(db_path: str, appointment_id: str) -> None:
    """标记约定已提醒。"""
    try:
        conn = connect_sqlite(db_path)
        conn.execute(
            "UPDATE user_appointments SET status='reminded', reminded_at=? WHERE appointment_id=?",
            (local_now().isoformat(timespec="seconds"), appointment_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
