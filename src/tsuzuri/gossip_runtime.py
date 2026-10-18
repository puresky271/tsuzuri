"""
gossip_runtime.py — Gossip 传播策略层（配置与传播数学的单一来源）
=================================================================

2026-08-14 链路统一重构：town_sim._GOSSIP_PROFILE 与 chat_server._GOSSIP_DISTORTION
原本是手工对齐的两份副本（chat_server.py 注释自承"对齐 town_sim"），存在漂移风险；
两者现统一改为从这里导入。

显式语义说明（与历史行为一致，未做行为变更）：
- ``distortion`` 只压低首跳 confidence 数值；用户可见的"失真"是 hearsay box
  （宿主应用._build_dev_memory_prompt Section 4）按 confidence/hop 的三档措辞 hedge，
  传播过程中文本本身逐字复制、不改写。
- chat_server._apply_gossip_distortion 的 legacy 文本级失真路径已于 2026-09-23
  删除（阈值 0.10 意味着所有角色里只有角色B 0.15 实际触发 LLM 模糊化改写，职能已被
  hearsay box 三档 hedge 取代）。
- 历史参数表里的 ``delay_ticks`` 从未被任何代码读取（死字段），已删除；
  传播延迟目前不建模。
- 2026-08-19: 写侧审计显示首跳 fanout 全 miss 偏多，上调 ``GOSSIP_PROFILE.spread_prob``
  （角色A 0.85 / 角色D 0.35 / 角色B 0.45 / 角色C 0.50 / 角色E 0.20）。

本模块只做策略与纯函数计算，不持有存储；存储 SSOT 是 memory_core.db 的
shared_info 表（由 memory_core.MemoryCore 提供事务化读写）。为可测试性，
所有掷骰点接受可注入 ``rng``（默认 ``random.random``）。

2026-08-16 审计：fanout / gate blocked / mailbox / secondary / delivered 事件
同步写入 memory_core.db 的 gossip_audit_log（fail-soft）；2026-08-18 起每次成功
提取另按 (WORLD_TZ 日, 角色) 累计 gossip_extract_audit，用于写侧频率健康审计。

2026-08-27 健康调查落地（docs/brain/memory/gossip_runtime_health_survey.md §9-2/§9-3）：
- mailbox 投影加进程内 (town_msg_id, character) 已处理缓存，重复 sync 不再逐条
  查库、不再刷 mailbox_attempt / mailbox_skip_duplicate 审计噪声；进程重启后
  DB 兜底判重命中也静默跳过（不再写 skip 审计事件）；
- secondary 候选洗牌 + 每 tick 最多 3 试 + 概率系数 0.5→0.7 + 48h 新鲜度过滤，
  修复 08-17 起二手传播停摆（固定 created_at DESC 反复撞同一条低概率候选）。

2026-10-05 全链路审计落地（同文档 §10）：
- ``get_pending_for`` 改分层配额：hop>=1 的人际传闻与 hop=0 的城镇邮箱摘要各留槽位、
  人际层优先，某层不足由另一层回填。此前摘要约 32 行/日把真八卦（约 7 行/日）挤出
  单一 created_at DESC 窗口，自 08-24 起 284 条 hop1 行只有 165 条进过提示词；
- ``record_town_mailbox`` 增内容级去重（(角色, 归一化文本) + 新鲜度窗口，进程内缓存
  与 DB 兜底双层）：town_sim 近况同步事件的 idempotency_key 含 tick id，同一内容每
  tick 换新 eid，旧的 (town_msg_id, character) 去重拦不住，实测同文本同目标 7 天落 4-6 行；
- 邮箱摘要 confidence 由 1.0 降为 ``MAILBOX_PROJECTION_CONFIDENCE``（0.70，对齐来源
  agent_message 事件），修掉"系统摘要比成员亲口转述更可信"的倒挂。
"""

from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import random

from tsuzuri.persona_registry import agent_names, live_table

# Per-agent spread probability / distortion (setting "gossip_profiles"); agents without an
# entry use _DEFAULT_PROFILE.
GOSSIP_PROFILE = live_table("gossip_profiles", {})

_DEFAULT_PROFILE = {"spread_prob": 0.3, "distortion": 0.1}

# 二手及以上传播的最大跳数（hop 从 1 起计，hop >= MAX_HOP 的条目不再被二次传播）
MAX_HOP = 3

# 二手传播概率衰减系数与每 tick 尝试上限（2026-08-27 健康调查 §9-3）：
# 旧值 0.5 且每 tick 固定按 created_at DESC 只试一条，08-17 后 secondary_written 持续为 0、
# 反复对同一条低概率掷骰。改为 0.7 × 每 tick 随机序最多试 3 条（成功即止）。
SECONDARY_PROB_MULTIPLIER = 0.7
SECONDARY_MAX_TRIES_PER_TICK = 3

# 城镇邮箱摘要投影的 hop 层（record_town_mailbox 写入值）；hop >= 1 是人际传闻
# （record_fanout 首跳 / record_secondary 二手转述）。
MAILBOX_LAYER_HOP = 0

# 邮箱摘要投影的 confidence（2026-10-05 全链路审计）：旧值 1.0 比任何真八卦首跳都高
# （first_hop_confidence 上限 ~0.90），等于把"系统摘要"排成比"成员亲口转述"更可信，
# 且摘要被二手转述时 1.0×0.7=0.70 仍留在 hearsay 第一档措辞。改为与来源 agent_message
# 事件一致（town_sim 近况同步 0.70 / 宿主应用 近期对话 0.72，取下界）：直投仍是第一档
# "{from}说："，二手转述衰减到 0.49 → 第二档 hedge。
MAILBOX_PROJECTION_CONFIDENCE = 0.70


def gossip_profile(character: str) -> dict:
    return GOSSIP_PROFILE.get(str(character or ""), _DEFAULT_PROFILE)


def distortion_level(character: str) -> float:
    return float(gossip_profile(character).get("distortion", 0.1))


def first_hop_confidence(source_char: str, target_trust: float) -> float:
    """首跳（hop=1）confidence：distortion 越低、目标对源 trust 越高，越接近 1。"""
    distortion = distortion_level(source_char)
    return round(max(0.3, 1.0 - distortion - 0.1 * (1.0 - float(target_trust))), 2)


def next_hop_confidence(confidence: float) -> float:
    """每多一手传播，confidence × 0.7。"""
    return round(float(confidence) * 0.7, 2)


def spread_probability(source_char: str, edge: dict, *, secondary: bool = False) -> float:
    """传播概率 = spread_prob × 边 info_sharing；二手传播再 × SECONDARY_PROB_MULTIPLIER。"""
    profile = gossip_profile(source_char)
    info_sharing = float((edge if isinstance(edge, dict) else {}).get("info_sharing", 0.3))
    prob = float(profile.get("spread_prob", 0.3)) * info_sharing
    if secondary:
        prob *= SECONDARY_PROB_MULTIPLIER
    return prob


def roll_spread(source_char: str, edge: dict, *, rng=None, secondary: bool = False) -> bool:
    """掷一次传播骰。rng 可注入（无参 callable 返回 [0,1)），默认 random.random。"""
    roll = (rng if callable(rng) else random.random)()
    return roll <= spread_probability(source_char, edge, secondary=secondary)


# ═══════════════════════════════════════════════════════════════════
# 存储层：shared_info 表（memory_core.db）是 gossip 的存储 SSOT
# ═══════════════════════════════════════════════════════════════════
#
# 统一前三个存储（dev_memory gossip_queue / gossip_store.json / shared_info）并存；
# 2026-08-14 起 dev 侧写入统一收敛到这里。gossip_store.json 仅保留 guest session
# 作用域条目（隔离红线，不进 dev 共享池）。
#
# 语义映射：统一前 gossip_queue 是"每目标一条 entry、各自 confidence"；统一后
# fanout 对【每个命中目标写一行、eligible_targets_json 只含该目标】，逐目标
# confidence 保真。旧行（eligible_targets_json=''）视为全员可见。

import hashlib
import json
import os
import re
from tsuzuri.persona_registry import user_name, user_pronoun
import sqlite3
import threading
import time
from datetime import datetime, timezone

from tsuzuri.sqlite_reliability import connect_sqlite

_FRESHNESS_WINDOW_HOURS = 48.0
_MAX_TEXT_LEN = 140


def _shared_info_db_path() -> str:
    """与 chat_server 历史逻辑一致：MEM_SQLITE_PATH 优先，其次项目默认。"""
    configured = str(os.environ.get("MEM_SQLITE_PATH", "") or "").strip()
    if configured:
        return configured
    try:
        from tsuzuri.project_paths import MEMORY_CORE_DB_PATH
        return str(MEMORY_CORE_DB_PATH)
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_core.db")


def _shared_info_store():
    from tsuzuri.memory_core import get_memory_core
    from tsuzuri.project_paths import DEV_MEMORY_PATH
    return get_memory_core(snapshot_path=str(DEV_MEMORY_PATH), db_path=_shared_info_db_path())


def _resolve_store(store=None):
    return store if store is not None else _shared_info_store()


def _audit_with_store(st, event_kind: str, *, source_character: str = "",
                      target_character: str = "", text: str = "", detail: dict | None = None,
                      created_at: float | None = None) -> bool:
    """写一条 gossip 审计事件；审计失败只降级为打印，不影响主链路。"""
    try:
        fn = getattr(st, "record_gossip_audit", None)
        if not callable(fn):
            return False
        return bool(fn(
            event_kind=event_kind,
            source_character=source_character,
            target_character=target_character,
            text=text,
            detail=detail if isinstance(detail, dict) else {},
            created_at=created_at,
        ))
    except Exception as e:
        _tsz_emit(f"[GossipAudit] {event_kind} error: {e}")
        return False


def _audit(event_kind: str, *, source_character: str = "", target_character: str = "",
           text: str = "", detail: dict | None = None, store=None,
           created_at: float | None = None) -> bool:
    try:
        st = _resolve_store(store)
    except Exception as e:
        _tsz_emit(f"[GossipAudit] store resolve error: {e}")
        return False
    return _audit_with_store(
        st, event_kind,
        source_character=source_character,
        target_character=target_character,
        text=text,
        detail=detail,
        created_at=created_at,
    )


def record_blocked(source_char: str, text: str, *, reason: str = "", store=None) -> bool:
    """记录 MemoryEditor / 状态门把 gossip 写入挡下的审计事件。"""
    source_char = str(source_char or "").strip()
    if not source_char:
        return False
    return _audit(
        "fanout_blocked",
        source_character=source_char,
        text=str(text or ""),
        detail={"reason": str(reason or "blocked")},
        store=store,
    )


def _record_extract_audit_with_store(st, *, character: str = "", has_gossip: bool = False,
                                     blocked_gate: bool = False, blocked_present_state: bool = False,
                                     validator_drop: bool = False, count_extraction: bool = True,
                                     created_at: float | None = None) -> bool:
    """把一次成功提取的 gossip 结果累计进 gossip_extract_audit（fail-soft）。"""
    try:
        fn = getattr(st, "record_gossip_extract_audit", None)
        if not callable(fn):
            return False
        return bool(fn(
            character=character,
            extractions=1 if count_extraction else 0,
            gossip_outputs=1 if has_gossip else 0,
            blocked_gate=1 if blocked_gate else 0,
            blocked_present_state=1 if blocked_present_state else 0,
            validator_drop=1 if validator_drop else 0,
            created_at=created_at,
        ))
    except Exception as e:
        _tsz_emit(f"[GossipAudit] extract audit error: {e}")
        return False


def record_extract_audit(character: str, *, has_gossip: bool = False, blocked_gate: bool = False,
                         blocked_present_state: bool = False, validator_drop: bool = False,
                         store=None, created_at: float | None = None) -> bool:
    """提取侧审计入口：每次成功解析后的 gossip 字段与门控结果按角色/日累计。"""
    try:
        st = _resolve_store(store)
    except Exception as e:
        _tsz_emit(f"[GossipAudit] store resolve error: {e}")
        return False
    return _record_extract_audit_with_store(
        st,
        character=str(character or "").strip(),
        has_gossip=has_gossip,
        blocked_gate=blocked_gate,
        blocked_present_state=blocked_present_state,
        validator_drop=validator_drop,
        created_at=created_at,
    )


def _dev_identity() -> tuple[str, str]:
    """(开发者显示名, 代词)，来自 persona registry。"""
    return user_name(), user_pronoun()


def _replace_pronoun_near_name(text: str, name: str, wrong: str, right: str, window: int = 12) -> str:
    """±window 字符窗口内、dev 名附近的错性别代词才替换（2026-05-17 窗口策略，
    从 chat_server._extract_and_store_gossip 迁入；旧整段全替策略已废弃）。"""
    if not text or not name or wrong not in text or name not in text:
        return text
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == wrong:
            lo, hi = max(0, i - window), min(n, i + window + 1)
            if name in text[lo:hi]:
                out.append(right)
                continue
        out.append(ch)
    return "".join(out)


def validate_gossip_text(text: str, *, dev_name: str = "", dev_pronoun: str = "") -> str:
    """gossip 落库前的 post-validator；返回清洗后文本，空串 = 丢弃。

     validator 集合从 chat_server._extract_and_store_gossip（已删除的重复提取链路）
    迁入：推断性语句丢弃、dev-as-stranger 框架丢弃、「用户/有人/某人/对方」正名、
    性别代词窗口修正、最短长度。
    """
    clean = str(text or "").strip()
    if not clean:
        return ""
    if not dev_name:
        dev_name, dev_pronoun = _dev_identity()
    # 推断性 gossip（基于一次提问推断 dev 的认知/不熟悉度）不是 fact，drop
    inference_pat = re.compile(
        rf"{re.escape(dev_name)}.{{0,20}}(?:居然|原来|竟然|不知道|没听过|不熟悉|不了解|没看过|没接触|不太懂)"
        rf"|(?:居然|原来|竟然|看来).{{0,15}}{re.escape(dev_name)}.{{0,15}}(?:不知道|没听过|不熟悉|不了解|不怎么看|不太懂|不太|没什么)"
    )
    if inference_pat.search(clean):
        return ""
    # 强制正名：禁止「用户/有人/某人/对方」
    for bad in ("用户", "有人", "某人", "对方"):
        if bad in clean:
            clean = clean.replace(bad, dev_name)
    # 防把开发者当陌生人（「遇到/认识了一个叫用户的人」类 mis-framing）→ 整条丢弃
    stranger_pat = re.compile(
        rf"(?:今天)?(?:刚才)?(?:遇到|认识|碰到|见到|遇见|结识|碰见)(?:了|过)?(?:一个|个)?(?:叫|名叫|被叫做)?{re.escape(dev_name)}(?:的人|的)?"
    )
    if stranger_pat.search(clean):
        return ""
    # 性别代词窗口修正
    if dev_pronoun == "他":
        clean = _replace_pronoun_near_name(clean, dev_name, "她", "他")
    elif dev_pronoun == "她":
        clean = _replace_pronoun_near_name(clean, dev_name, "他", "她")
    clean = clean[:_MAX_TEXT_LEN].strip()
    return clean if len(clean) > 4 else ""


def _freshness(created_at: float, *, now_ts: float | None = None) -> float:
    """48 小时线性衰减到 0（沿用 chat_server._get_gossip_freshness 语义）。"""
    now_ts = time.time() if now_ts is None else float(now_ts)
    hours = (now_ts - float(created_at or 0.0)) / 3600.0
    return max(0.0, 1.0 - hours / _FRESHNESS_WINDOW_HOURS)


def _row_visible_to(row: sqlite3.Row, character: str) -> bool:
    try:
        delivered = json.loads(str(row["delivered_to_json"] or "{}"))
    except Exception:
        delivered = {}
    if isinstance(delivered, dict) and character in delivered:
        return False
    try:
        eligible = json.loads(str(row["eligible_targets_json"] or "") or "[]")
    except Exception:
        eligible = []
    if isinstance(eligible, list) and eligible and character not in eligible:
        return False
    return True


def _pending_slots(limit: int) -> tuple[int, int]:
    """把读取配额拆成 (人际传闻槽位, 邮箱摘要槽位)。

    真八卦（hop>=1）稀缺、城镇摘要（hop=0）量大（2026-10-05 审计：摘要约 32 行/日、
    真八卦约 7 行/日），共用一个 created_at DESC 窗口时真八卦会被摘要挤出——
    自 08-24 起 284 条 hop1 行只有 165 条进过提示词。任一层不足由另一层回填。
    """
    total = max(1, int(limit))
    real = min(total, max(1, (total + 1) // 2))
    return real, total - real


def _row_to_pending(row: sqlite3.Row, fresh: float) -> dict:
    return {
        "info_id": str(row["info_id"]),
        "source_character": str(row["source_character"]),
        "original_text": str(row["original_text"]),
        "privacy_level": str(row["privacy_level"] or "public"),
        "freshness": fresh,
        "created_at": float(row["created_at"] or 0.0),
        "confidence": float(row["confidence"] if row["confidence"] is not None else 1.0),
        "hop": int(row["hop"] if row["hop"] is not None else 1),
    }


def _fetch_pending_layer(conn, character: str, *, min_hop: int, max_hop: int,
                         slots: int, now_ts: float | None) -> list[dict]:
    """取一个 hop 层内该角色当前可投递的行（层内 created_at DESC）。

    深度 = slots*3，给 eligible/新鲜度过滤留余量（沿用统一前的 overfetch 口径）。
    """
    if slots <= 0:
        return []
    rows = conn.execute(
        """
        SELECT * FROM shared_info
        WHERE source_character != ?
          AND privacy_level != 'private'
          AND expired = 0
          AND mentions_count < max_mentions
          AND COALESCE(hop, 1) >= ?
          AND COALESCE(hop, 1) <= ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (character, int(min_hop), int(max_hop), max(1, int(slots)) * 3),
    ).fetchall()
    out: list[dict] = []
    for row in rows:
        if not _row_visible_to(row, character):
            continue
        fresh = _freshness(float(row["created_at"] or 0.0), now_ts=now_ts)
        if fresh <= 0.05:
            continue
        out.append(_row_to_pending(row, fresh))
        if len(out) >= slots:
            break
    return out


def get_pending_for(character: str, limit: int = 3, *, store=None, now_ts: float | None = None) -> list[dict]:
    """读取某角色当前可投递的 gossip（统一存储 read API）。

    过滤：非本人发出、privacy != private、未过期、提及数未达上限、未投递给该角色、
    该角色在 eligible_targets 内（空 = 全员可见，兼容旧行）、48h 新鲜度 > 0.05。

    排序（2026-10-05 分层配额）：hop>=1 的人际传闻优先于 hop=0 的城镇邮箱摘要，
    各层内 created_at DESC；配额见 ``_pending_slots``，某层不足由另一层回填，
    总数不超过 limit。摘要另有 town_sim 投影 block 的独立通道，不依赖本配额。
    """
    character = str(character or "").strip()
    if not character:
        return []
    st = _resolve_store(store)
    db_path = str(getattr(st, "db_path", "") or _shared_info_db_path())
    if not os.path.isfile(db_path):
        return []
    total = max(1, int(limit))
    real_slots, digest_slots = _pending_slots(total)
    try:
        st.ensure_shared_info_schema()
        conn = connect_sqlite(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            real = _fetch_pending_layer(
                conn, character, min_hop=MAILBOX_LAYER_HOP + 1, max_hop=MAX_HOP,
                slots=total, now_ts=now_ts,
            )
            digest = _fetch_pending_layer(
                conn, character, min_hop=MAILBOX_LAYER_HOP, max_hop=MAILBOX_LAYER_HOP,
                slots=total, now_ts=now_ts,
            )
        finally:
            conn.close()
    except Exception as e:
        _tsz_emit(f"[Gossip] pending query error: {e}")
        return []
    results = [*real[:real_slots], *digest[:digest_slots]]
    if len(results) < total:
        picked = {row["info_id"] for row in results}
        for row in (*real, *digest):
            if row["info_id"] in picked:
                continue
            results.append(row)
            picked.add(row["info_id"])
            if len(results) >= total:
                break
    return results[:total]


def _audit_delivery(st, info_id: str, character: str, delivered_at: float) -> bool:
    db_path = str(getattr(st, "db_path", "") or _shared_info_db_path())
    if not os.path.isfile(db_path):
        return False
    try:
        st.ensure_shared_info_schema()
        conn = connect_sqlite(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT source_character, original_text, hop FROM shared_info WHERE info_id=?;",
                (str(info_id),),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        return _audit_with_store(
            st, "delivered",
            source_character=str(row["source_character"] or ""),
            target_character=character,
            text=str(row["original_text"] or ""),
            detail={"info_id": str(info_id), "hop": int(row["hop"] if row["hop"] is not None else 1)},
            created_at=delivered_at,
        )
    except Exception as e:
        _tsz_emit(f"[GossipAudit] delivered error: {e}")
        return False


def mark_delivered(info_id: str, character: str, *, store=None) -> None:
    if not str(info_id or "").strip() or not str(character or "").strip():
        return
    try:
        st = _resolve_store(store)
        delivered_at = time.time()
        ok = bool(st.mark_shared_info_delivered(str(info_id), str(character), delivered_at=delivered_at))
        if ok:
            _audit_delivery(st, str(info_id), str(character), delivered_at)
    except Exception as e:
        _tsz_emit(f"[Gossip] mark delivered error: {e}")


def _new_info_id() -> str:
    return f"si-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"


def _edge_from_graph(graph: dict, a: str, b: str) -> dict:
    if isinstance(graph, dict):
        edge = (graph.get(a) or {}).get(b)
        if isinstance(edge, dict):
            return edge
    return {"info_sharing": 0.3, "trust": 0.5}


def record_fanout(source_char: str, text: str, *, graph: dict | None = None,
                  privacy: str = "public", created_at: float | None = None,
                  rng=None, store=None) -> list[str]:
    """统一写入入口：validate → 对每目标掷骰（spread_prob × info_sharing）→
    每个命中目标写一行（eligible 仅含该目标，confidence 按该边 trust 计算）。
    返回写入的 info_id 列表。"""
    source_char = str(source_char or "").strip()
    clean = validate_gossip_text(text)
    st = _resolve_store(store)
    created = time.time() if created_at is None else float(created_at)
    if not source_char or not clean:
        _drop_reason = "no_source" if not source_char else "validator_drop"
        _audit_with_store(
            st, "fanout_blocked",
            source_character=source_char,
            text=str(text or ""),
            detail={"reason": _drop_reason},
            created_at=created,
        )
        if _drop_reason == "validator_drop":
            _record_extract_audit_with_store(
                st, character=source_char, validator_drop=True,
                count_extraction=False, created_at=created,
            )
        return []
    info_ids: list[str] = []
    for target in agent_names():
        if target == source_char:
            continue
        edge = _edge_from_graph(graph, source_char, target)
        prob = spread_probability(source_char, edge)
        if not roll_spread(source_char, edge, rng=rng):
            _audit_with_store(
                st, "fanout_miss",
                source_character=source_char,
                target_character=target,
                text=clean,
                detail={
                    "probability": round(prob, 4),
                    "info_sharing": edge.get("info_sharing", 0.3),
                    "trust": edge.get("trust", 0.5),
                },
                created_at=created,
            )
            continue
        info_id = _new_info_id()
        confidence = first_hop_confidence(source_char, edge.get("trust", 0.5))
        ok = st.store_shared_info(
            info_id=info_id,
            source_character=source_char,
            original_text=clean,
            privacy_level=str(privacy or "public"),
            created_at=created,
            confidence=confidence,
            hop=1,
            eligible_targets=[target],
        )
        if ok:
            info_ids.append(info_id)
            _audit_with_store(
                st, "fanout_hit",
                source_character=source_char,
                target_character=target,
                text=clean,
                detail={
                    "info_id": info_id,
                    "confidence": confidence,
                    "probability": round(prob, 4),
                },
                created_at=created,
            )
    _audit_with_store(
        st, "fanout_attempt",
        source_character=source_char,
        text=clean,
        detail={
            "target_count": len([t for t in agent_names() if t != source_char]),
            "hit_count": len(info_ids),
            "all_miss": not info_ids,
            "info_ids": info_ids,
        },
        created_at=created,
    )
    if info_ids:
        _tsz_emit(f"[Gossip] {source_char} fanout → {len(info_ids)} targets: {clean[:60]}")
    return info_ids


def _shuffled(items: list, rng=None) -> list:
    """Fisher-Yates 洗牌（rng 可注入、调用次数确定 = len-1，便于测试）。"""
    out = list(items)
    r = rng if callable(rng) else random.random
    for i in range(len(out) - 1, 0, -1):
        j = min(i, int(r() * (i + 1)))
        out[i], out[j] = out[j], out[i]
    return out


def record_secondary(a: str, b: str, *, graph: dict | None = None,
                     rng=None, store=None) -> bool:
    """tick share_news 二手传播：A 把自己**确实收到过**（delivered_to 含 A）的
    传闻转述给 B（修复旧实现转发 A 未投递条目的语义 bug）。
    hop+1、confidence×0.7、hop>=MAX_HOP 不再传播；同 (A, 文本, B) 去重。
    2026-08-27 健康调查 §9-3：候选随机洗牌（不再固定撞 created_at DESC 第一条）、
    每 tick 最多试 SECONDARY_MAX_TRIES_PER_TICK 条（成功即止）、
    过滤超 48h 新鲜度窗口的候选（转述过期传闻没有意义、写进去也永远投不出去）。"""
    a, b = str(a or "").strip(), str(b or "").strip()
    if not a or not b or a == b:
        return False
    st = _resolve_store(store)
    db_path = str(getattr(st, "db_path", "") or _shared_info_db_path())
    if not os.path.isfile(db_path):
        return False
    now = time.time()
    _audit_with_store(
        st, "secondary_attempt",
        source_character=a, target_character=b,
        detail={"outcome": "started"},
        created_at=now,
    )
    try:
        st.ensure_shared_info_schema()
        conn = connect_sqlite(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            candidates = conn.execute(
                """
                SELECT * FROM shared_info
                WHERE source_character != ?
                  AND privacy_level != 'private'
                  AND expired = 0
                  AND COALESCE(hop, 1) >= ?
                  AND hop < ?
                  AND delivered_to_json LIKE ?
                ORDER BY created_at DESC
                LIMIT 12
                """,
                (b, MAILBOX_LAYER_HOP + 1, MAX_HOP, f'%"{a}"%'),
            ).fetchall()
            existing_texts = {
                str(r[0])
                for r in conn.execute(
                    "SELECT original_text FROM shared_info WHERE source_character=? AND eligible_targets_json LIKE ?;",
                    (a, f'%"{b}"%'),
                ).fetchall()
            }
        finally:
            conn.close()
    except Exception as e:
        _tsz_emit(f"[Gossip] secondary read error: {e}")
        return False
    eligible = []
    for row in candidates:
        try:
            delivered = json.loads(str(row["delivered_to_json"] or "{}"))
        except Exception:
            delivered = {}
        if not isinstance(delivered, dict) or a not in delivered:
            continue  # A 没听说过的不传（修 semantic bug）
        if _freshness(float(row["created_at"] or 0.0), now_ts=now) <= 0.05:
            continue  # 超新鲜度窗口的不传
        text = str(row["original_text"] or "")
        if not text or text in existing_texts:
            continue
        eligible.append(row)
    if not eligible:
        _audit_with_store(
            st, "secondary_noop",
            source_character=a, target_character=b,
            detail={"outcome": "no_received_candidate"},
            created_at=now,
        )
        return False
    edge = _edge_from_graph(graph, a, b)
    written = False
    for row in _shuffled(eligible, rng)[:SECONDARY_MAX_TRIES_PER_TICK]:
        text = str(row["original_text"] or "")
        hop = int(row["hop"] if row["hop"] is not None else 1)
        if not roll_spread(a, edge, rng=rng, secondary=True):
            _audit_with_store(
                st, "secondary_miss",
                source_character=a, target_character=b, text=text,
                detail={"outcome": "roll_miss", "hop": hop},
                created_at=now,
            )
            continue
        ok = bool(st.store_shared_info(
            info_id=_new_info_id(),
            source_character=a,
            original_text=text[:_MAX_TEXT_LEN],
            privacy_level=str(row["privacy_level"] or "public"),
            confidence=next_hop_confidence(float(row["confidence"] if row["confidence"] is not None else 0.7)),
            hop=hop + 1,
            eligible_targets=[b],
        ))
        if ok:
            written = True
            _audit_with_store(
                st, "secondary_written",
                source_character=a, target_character=b, text=text,
                detail={"hop": hop + 1},
                created_at=now,
            )
            break
        _audit_with_store(
            st, "secondary_miss",
            source_character=a, target_character=b, text=text,
            detail={"outcome": "store_failed"},
            created_at=now,
        )
    return written


# 邮箱投影进程内去重缓存（2026-08-27 健康调查 §9-2）：ledger mailbox 的 delivered
# 标记只读不写，sync_projection_to_legacy 每轮把同一批 active 消息重复送进来，
# has_town_mailbox_message 每次 DB 查询 + mailbox_skip_duplicate 审计刷屏
# （08-27 实测 attempt 164 / skip 361 / 实际新增 12）。缓存已处理过的
# (town_msg_id, character)，重复调用直接静默跳过；DB 级去重保留兜底
# （进程重启后首个重复调用由 has_town_mailbox_message 拦下并回填缓存，
# 兜底命中同样静默、不再写 mailbox_skip_duplicate 审计——重启后每对
# (town_msg_id, character) 曾固定产生一条该类事件，累计成审计最大类别）。
_TOWN_MAILBOX_SETTLED: set[tuple[str, str]] = set()
_TOWN_MAILBOX_SETTLED_LOCK = threading.Lock()
_TOWN_MAILBOX_SETTLED_MAX = 20000


def _town_mailbox_settled(mid: str, character: str) -> bool:
    with _TOWN_MAILBOX_SETTLED_LOCK:
        return (mid, character) in _TOWN_MAILBOX_SETTLED


def _town_mailbox_mark_settled(mid: str, character: str) -> None:
    with _TOWN_MAILBOX_SETTLED_LOCK:
        if len(_TOWN_MAILBOX_SETTLED) > _TOWN_MAILBOX_SETTLED_MAX:
            _TOWN_MAILBOX_SETTLED.clear()
        _TOWN_MAILBOX_SETTLED.add((mid, character))


# 邮箱摘要的**内容级**去重（2026-10-05 全链路审计）：town_sim 的近况同步事件
# idempotency_key 含 tick id（town_sim.py agent_message ev3），同一 plan step 每 tick
# 都产生新 eid，(town_msg_id, character) 去重因此拦不住——实测同一句话对同一目标
# 7 天内落 4-6 行，把 get_pending_for 的读取窗口挤满、真八卦被埋（hop1 42% 零投递）。
# 这里按 (角色, 归一化文本) 在新鲜度窗口内去重：进程内缓存 + DB 兜底双层，
# 与 id 缓存同构。摘要另有 town_sim 投影 block 通道，去重不会让角色彻底看不到近况。
_TOWN_MAILBOX_TEXT_SETTLED: set[tuple[str, str]] = set()


def _mailbox_content_key(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _town_mailbox_text_settled(character: str, key: str) -> bool:
    with _TOWN_MAILBOX_SETTLED_LOCK:
        return (character, key) in _TOWN_MAILBOX_TEXT_SETTLED


def _town_mailbox_mark_text_settled(character: str, key: str) -> None:
    with _TOWN_MAILBOX_SETTLED_LOCK:
        if len(_TOWN_MAILBOX_TEXT_SETTLED) > _TOWN_MAILBOX_SETTLED_MAX:
            _TOWN_MAILBOX_TEXT_SETTLED.clear()
        _TOWN_MAILBOX_TEXT_SETTLED.add((character, key))


def _has_recent_mailbox_text(st, character: str, text: str, *, now_ts: float) -> bool:
    """DB 兜底：同一目标在新鲜度窗口内已有同文本投影行（含已投递/已过期）则不再写。"""
    key = _mailbox_content_key(text)
    if not key:
        return False
    db_path = str(getattr(st, "db_path", "") or _shared_info_db_path())
    if not os.path.isfile(db_path):
        return False
    try:
        st.ensure_shared_info_schema()
        conn = connect_sqlite(db_path, timeout=5)
        try:
            rows = conn.execute(
                """
                SELECT original_text FROM shared_info
                WHERE eligible_targets_json LIKE ?
                  AND created_at > ?
                LIMIT 200;
                """,
                (f'%"{character}"%', now_ts - _FRESHNESS_WINDOW_HOURS * 3600.0),
            ).fetchall()
        finally:
            conn.close()
    except Exception as e:
        _tsz_emit(f"[Gossip] mailbox content dedup error: {e}")
        return False
    return any(_mailbox_content_key(row[0]) == key for row in rows)


def record_town_mailbox(character: str, inbox_items: list, *, store=None) -> int:
    """TownSim 邮箱 digest 投影（替代旧 sync_projection_to_legacy 的 gossip_queue 写入）。
    每条消息一行、eligible 仅含收件角色、双层去重：
    (town_msg_id, character) 拦同一消息重投，(角色, 归一化文本) 拦同一内容换 id 重投。
    confidence=MAILBOX_PROJECTION_CONFIDENCE/hop=0（摘要可信度低于成员亲口转述，
    但直投仍落 hearsay box 第一档措辞）。"""
    character = str(character or "").strip()
    if not character or not isinstance(inbox_items, list) or not inbox_items:
        return 0
    pending_items = []
    for item in inbox_items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id", "") or "").strip()
        if not mid or _town_mailbox_settled(mid, character):
            continue
        pending_items.append(item)
    if not pending_items:
        return 0  # 全部已处理：不再写 mailbox_attempt 审计、不再查库
    st = _resolve_store(store)
    now = time.time()

    def _mailbox_text(item: dict) -> str:
        topic = str(item.get("topic", "") or "").strip()[:24]
        summ = str(item.get("summary", "") or "").strip()[:120]
        return f"{topic}：{summ}" if topic and summ else (summ or topic or "有新消息")

    fresh_items: list[tuple[dict, str]] = []
    content_dupes = 0
    batch_keys: set[str] = set()
    for item in pending_items:
        about = _mailbox_text(item)[:_MAX_TEXT_LEN]
        key = _mailbox_content_key(about)
        # 批内也要去重：同一次 sync 的 inbox_digest 可能带多条同文本消息，
        # 此时 DB 兜底还看不到彼此（都尚未写入）。
        if key and (
            key in batch_keys
            or _town_mailbox_text_settled(character, key)
            or _has_recent_mailbox_text(st, character, about, now_ts=now)
        ):
            _town_mailbox_mark_text_settled(character, key)
            # 这条消息永远不会再可投影，一并记入 id 缓存，后续 tick 不再查库。
            _town_mailbox_mark_settled(str(item.get("id", "") or "").strip(), character)
            content_dupes += 1
            continue
        if key:
            batch_keys.add(key)
        fresh_items.append((item, about))

    _audit_with_store(
        st, "mailbox_attempt",
        target_character=character,
        detail={"items": len(pending_items), "content_dedup_skipped": content_dupes},
        created_at=now,
    )
    if not fresh_items:
        return 0
    added = 0
    for item, about in fresh_items:
        mid = str(item.get("id", "") or "").strip()
        frm = str(item.get("from", "") or "成员").strip()[:12]
        try:
            if st.has_town_mailbox_message(mid, character):
                # DB 兜底判重命中（进程重启后的首个重复调用）：静默跳过、回填缓存，
                # 不再写 mailbox_skip_duplicate 审计（此前每次重启都刷一批该类事件）。
                _town_mailbox_mark_settled(mid, character)
                continue
        except Exception:
            continue
        if st.store_shared_info(
            info_id=_new_info_id(),
            source_character=frm,
            original_text=about,
            privacy_level="public",
            confidence=MAILBOX_PROJECTION_CONFIDENCE,
            hop=MAILBOX_LAYER_HOP,
            eligible_targets=[character],
            town_msg_id=mid,
        ):
            _town_mailbox_mark_settled(mid, character)
            _town_mailbox_mark_text_settled(character, _mailbox_content_key(about))
            added += 1
            _audit_with_store(
                st, "mailbox_projected",
                source_character=frm,
                target_character=character,
                text=about,
                detail={"town_msg_id": mid},
                created_at=now,
            )
    return added
