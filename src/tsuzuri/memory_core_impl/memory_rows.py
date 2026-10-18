"""Text normalization, lexical scoring and memory-row surfaceability predicates.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations

import json
import re
from tsuzuri.persona_registry import agent_names, live_pattern
from tsuzuri.secrets_loader import env_bool as _bool_env
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import _emb_sparse_dot, _emb_text_to_sparse, _safe_float
from tsuzuri.memory_core_impl.maintenance_seeds import recall_gates as _maintenance_recall_gates

def _norm_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\u4e00-\u9fff ]+", "", s)
    return s.strip()


def _bigrams(s: str) -> set[str]:
    t = _norm_text(s).replace(" ", "")
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _lexical_similarity(a: str, b: str) -> float:
    """Hybrid lexical similarity: blend char-bigram Jaccard with word-level Jaccard."""
    from tsuzuri.memory_core_impl.recall_scoring import _word_jaccard
    aa = _bigrams(a)
    bb = _bigrams(b)
    bigram_sim = 0.0
    if aa and bb:
        inter = len(aa & bb)
        union = len(aa | bb)
        if union > 0:
            bigram_sim = float(inter) / float(union)
    # Word-level Jaccard via jieba (better for Chinese synonymy)
    word_sim = _word_jaccard(a, b)
    # Blend: word-level gets higher weight (captures meaning), bigram catches exact phrases
    if word_sim > 0 or bigram_sim > 0:
        return 0.45 * bigram_sim + 0.55 * word_sim
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase B (2026-05-14): BM25 + RRF 融合工具函数
# 配合 BGE-M3 sparse embedding 实现 lexical-aware 召回
# ═══════════════════════════════════════════════════════════════════════════

def use_rrf_fusion() -> bool:
    """Feature flag: USE_RRF_FUSION=true 启用 BM25+Dense 双路 RRF 融合。"""
    return _bool_env("USE_RRF_FUSION", False)


def _bm25_score_pair(query_sparse: dict | None, item_sparse_text: str) -> float:
    """对一条 candidate 用 sparse dot product 算 BM25-like 分。

    输入：
      query_sparse: bge-m3 sparse dict[int_token_id, float_weight]
      item_sparse_text: DB 里存的 JSON 字符串
    输出：[0.0, +inf) 的 lexical-aware 相似度（实际范围通常 0-30）
    """
    if not query_sparse or not item_sparse_text:
        return 0.0
    if _emb_text_to_sparse is None or _emb_sparse_dot is None:
        return 0.0
    item_sparse = _emb_text_to_sparse(item_sparse_text)
    if not item_sparse:
        return 0.0
    return _emb_sparse_dot(query_sparse, item_sparse)


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. 业界标准做法（Cormack 2009）。

    输入：rank_lists = [[item_id1, item_id2, ...], ...] 多路独立排名
    输出：[(item_id, rrf_score), ...] desc sorted

    k=60 是经验值。RRF 优点：不需要调权重、对极端值鲁棒、对互补信号融合最稳。
    """
    if not rank_lists:
        return []
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, item_id in enumerate(ranks, start=1):
            if not item_id:
                continue
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


_RECALL_NOISE_RE = live_pattern(
    r"(你还记得|还记得|记不记得|记得吗|还想起|想起|回忆|之前|以前|上次|那次|当时|"
    r"我之前|我以前|我上次|跟你说过|和你说过|给你发过|给你看过|说过|提过|"
    r"这件事情|那件事情|这件事|那件事|这段|那段|那句话|那个|这个|什么|哪里|哪儿|哪个|地方|事情|"
    r"对了|嘛|啦|呐|诶|欸|哦|啊|{AGENTS}|吗|呀|呢|来着)"
)
_RECALL_QUOTED_RE = re.compile(r"[『「“\"']([^『』「」“”\"']{2,80})[』」”\"']")


def _recall_focus_text(query: str) -> str:
    raw = str(query or "").strip()
    if not raw:
        return ""
    quoted = [str(x or "").strip() for x in _RECALL_QUOTED_RE.findall(raw) if str(x or "").strip()]
    if quoted:
        focus = max(quoted, key=len)
        focus = re.sub(r"\s+", " ", focus).strip()
        if focus:
            return focus
    focus = _RECALL_NOISE_RE.sub(" ", raw)
    focus = re.sub(r"(?:\s*的)?\s*(?:这件事|那件事|事情|事)$", " ", focus)
    focus = re.sub(r"(?:\s*的)?\s*(?:内容|细节)$", " ", focus)
    focus = re.sub(r"\s*的$", " ", focus)
    focus = re.sub(r"[^\w\u4e00-\u9fff ]+", " ", focus)
    focus = re.sub(r"\s+", " ", focus).strip()
    # Recall scaffolding can leave one or more leading pronouns/particles:
    # "我之前说过的面试" -> "的面试" and "我的生日" -> "的生日".
    # Strip only this small, bounded prefix set; topic words remain untouched.
    for _ in range(2):
        cleaned = re.sub(r"^(?:的|我|你|他|她)\s*", "", focus).strip()
        if cleaned == focus:
            break
        focus = cleaned
    return focus or raw


def _memory_row_is_working(row: dict | None = None) -> bool:
    item = row if isinstance(row, dict) else {}
    if bool(item.get("_hybrid_promoted_layer", False)):
        return False
    return (
        str(item.get("layer", "") or "").strip().lower() == "working"
        or str(item.get("memory_role", "") or "").strip().lower() == "working"
        or str(item.get("memory_type", "") or "").strip().lower() == "working_turn"
        or str(item.get("source_kind", "") or "").strip().lower() == "turn_exchange"
        or str(item.get("engram_type", "") or "").strip().lower() == "turn_exchange"
    )


_GENERIC_MEMORY_SURFACE_EXACT = {
    "记得我的日程",
    "会主动关心我的日常",
    "乐于分享日常",
    "会分享日常琐事",
    "礼貌友好",
    "礼貌道别",
    "愿意聊天",
    "会主动关心人",
    "会主动关心他人",
    "会关心人",
    "会关心他人感受",
    "主动开启日常话题",
    "最近大家都过得怎么样呀",
}
_GENERIC_MEMORY_SURFACE_RE = re.compile(
    r"^(?:会)?(?:主动)?关心(?:人|他人|大家|朋友|.+日常|.+动态)$|"
    r"^(?:话题|topic)[:：]\s*.+$|"
    r"^(?:最近)?大家都过得怎么样[呀啊]?$|"
    r"^(?:礼貌.*|友好|解释清晰，为人着想)$",
    re.IGNORECASE,
)


def _memory_surface_text_is_generic(text: str) -> bool:
    raw = str(text or "").strip()
    norm = _norm_text(raw)
    if not raw or len(norm) < 6:
        return True
    if raw in _GENERIC_MEMORY_SURFACE_EXACT or raw in {f"关心{a}的日常" for a in agent_names()}:
        return True
    return bool(_GENERIC_MEMORY_SURFACE_RE.search(raw))


def _memory_payload_evidence_policy(payload: dict | None = None) -> dict:
    pl = payload if isinstance(payload, dict) else {}
    policy = pl.get("evidence_policy", {}) if isinstance(pl.get("evidence_policy", {}), dict) else {}
    if policy:
        return dict(policy)
    meta = pl.get("memory_meta", {}) if isinstance(pl.get("memory_meta", {}), dict) else {}
    nested = meta.get("evidence_policy", {}) if isinstance(meta.get("evidence_policy", {}), dict) else {}
    return dict(nested) if nested else {}


def _memory_row_is_explicit_recall_archive(row: dict | None = None, payload: dict | None = None) -> bool:
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    policy = _memory_payload_evidence_policy(pl if isinstance(pl, dict) else {})
    category = str(policy.get("category", "") or "").strip().lower()
    visibility = str(policy.get("visibility", "") or "").strip().lower()
    return bool(
        category == "explicit_recall_archive"
        or visibility == "explicit_recall_only"
        or str(item.get("item_type", "") or "").strip().lower() == "explicit_recall_archive"
        or str(item.get("engram_type", "") or "").strip().lower() == "explicit_recall_archive"
    )


def _memory_row_is_auto_fact_observation(row: dict | None = None, payload: dict | None = None) -> bool:
    """Phase R.W: auto-extracted facts parked in episodic before KAIROS distillation.

    These rows are useful recall material, but they are also the class most likely
    to inflate OOS false positives after the write rebalance: they carry fairly high
    recency/importance even though they are not yet stable semantic facts.  Retrieval
    should therefore require stronger query evidence before surfacing them.
    """
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    if str(item.get("layer", "") or "").strip().lower() != "episodic":
        return False
    if str(item.get("source", "") or "").strip().lower() != "turn_append":
        return False
    et = str(item.get("engram_type", "") or "").strip().lower()
    it = str(item.get("item_type", "") or "").strip().lower()
    mt = str(item.get("memory_type", "") or "").strip().lower()
    if et == "fact_observation" or it == "fact_observation":
        return True
    # character_projection auto facts are also born episodic (not semantic) in Phase R.W.
    if mt == "character_projection" and (et == "projection" or it == "projection"):
        return True
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    return bool(
        isinstance(pl, dict)
        and isinstance(pl.get("fact"), str)
        and mt in ("shared_fact", "character_projection")
    )


_DURABLE_COMMITMENT_RE = re.compile(
    r"(?:承诺|答应|保证|约定|promise|promised|commitment|agreed to)",
    re.IGNORECASE,
)
_NEGATED_DURABLE_COMMITMENT_RE = re.compile(
    r"(?:没有|并未|未曾|不是|不算|否认).{0,4}(?:承诺|答应|保证|约定)",
    re.IGNORECASE,
)


def _turn_fact_card_has_durable_commitment(card: dict | None = None) -> bool:
    """Keep broad TurnFact commitments while granting bypass only to explicit promises."""
    data = card if isinstance(card, dict) else {}
    commitments = data.get("commitments", []) if isinstance(data.get("commitments", []), list) else []
    for commitment in commitments:
        if isinstance(commitment, dict):
            kind = str(commitment.get("kind", "") or "").strip().lower()
            if bool(commitment.get("durable")) or kind in {
                "future_commitment", "promise", "mutual_agreement",
            }:
                return True
            text = str(commitment.get("text", "") or "")
        else:
            text = str(commitment or "")
        if (
            _DURABLE_COMMITMENT_RE.search(text)
            and not _NEGATED_DURABLE_COMMITMENT_RE.search(text)
        ):
            return True
    return False


def _memory_row_has_durable_turn_signal(row: dict | None = None, payload: dict | None = None) -> bool:
    """Return True for structured user-grounded facts that must not be threshold-killed.

    2026-08-02 收紧：不再把「card.fact 非空 + source_trust verified」当 durable。
    turn fact ledger（2026-07-09 起）让每个 turn 都产卡且 web 用户文本恒为
    verified，该条件退化为「全量晋升」、天气流水也会绕过 typed gate 直升
    episodic（payload-wipe bug 2026-08-01 修复后首轮才暴露）。durable 只保留
    真正有结构化纠正/显式记忆请求，或明确承诺意图的卡。TurnFact 的
    commitments 仍可保存回答/确认以维持会话连续性，但字段非空本身不再
    获得 memory evolution durable bypass。
    """
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    pl = pl if isinstance(pl, dict) else {}
    meta = pl.get("meta", {}) if isinstance(pl.get("meta", {}), dict) else {}
    card = pl.get("turn_fact_card", {}) if isinstance(pl.get("turn_fact_card", {}), dict) else {}
    if not card and isinstance(meta.get("turn_fact_card", {}), dict):
        card = meta.get("turn_fact_card", {})
    corrections = card.get("corrections", []) if isinstance(card.get("corrections", []), list) else []
    policy = pl.get("memory_policy", {}) if isinstance(pl.get("memory_policy", {}), dict) else {}
    explicit_request = str(policy.get("promotion_reason", "") or "") == "explicit_user_memory_request"
    return bool(corrections or _turn_fact_card_has_durable_commitment(card) or explicit_request)


def _memory_row_is_trusted_fact_observation(row: dict | None = None, payload: dict | None = None) -> bool:
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    return bool(
        _memory_row_is_auto_fact_observation(item, pl if isinstance(pl, dict) else {})
        and isinstance(pl, dict)
        and str(pl.get("fact", "") or "").strip()
        and _safe_float(item.get("confidence", 0.0), 0.0) >= 0.64
    )


_LOW_VALUE_MEMORY_GENERALIZATION_RE = re.compile(
    r"^(?:会|愿意|乐于|善于|喜欢|重视|关心|观察力|工作认真|能理解|记得日期|专注于|常需|会分享).{0,48}$|"
    r"^(?:对.+有(?:想法|兴趣|偏好)|需要(?:休息|放松|鼓励)|主动.+)$",
    re.IGNORECASE,
)


_EVOLUTION_SEMANTIC_PROMOTION_BLOCKED_TYPES = frozenset({
    "group_session_event",
    "collective_user_fact",
    "cross_char_daily_digest",
    "cross_char_daily_digest_self_record",
    "school_timeline_anchor",
})


def _memory_row_allows_evolution_semantic_promotion(
    row: dict | None = None,
    payload: dict | None = None,
) -> bool:
    """Exclude secondhand/state-projection records from generic semantic promotion."""
    from tsuzuri.memory_core_impl.recall_scoring import _json_loads
    item = row if isinstance(row, dict) else {}
    pl = payload if isinstance(payload, dict) else _json_loads(str(item.get("payload_json", "") or "{}"), {})
    pl = pl if isinstance(pl, dict) else {}
    source = str(item.get("source", "") or "").strip().lower()
    source_kind = str(pl.get("source_kind", "") or "").strip().lower()
    item_types = {
        str(item.get("engram_type", "") or "").strip().lower(),
        str(item.get("item_type", "") or "").strip().lower(),
        source_kind,
    }
    return not (
        source.startswith("cross_char_daily_digest")
        or bool(item_types.intersection(_EVOLUTION_SEMANTIC_PROMOTION_BLOCKED_TYPES))
    )
_EPHEMERAL_MEMORY_TEXT_RE = re.compile(
    r"(/reset|/clear|今天|现在|刚刚|刚才|这会|目前|正在|天气|下雨|带伞|睡觉|晚安|早安|晚上好|在吗|在家|在旁边|路上|被窝|几点)",
    re.IGNORECASE,
)
_WORKING_EPHEMERAL_NOISE_RE = re.compile(
    r"(/reset|/clear|天气|下雨|带伞|外面|阴天|晴天|几点(?:了|呢|啊)?|什么声音|"
    r"在哪|在哪里|在旁边|旁边|在家|家里|路上|回家|教室|活动室感觉|"
    r"在睡觉|睡着|刚醒|刚才在睡|晚安|早安|晚上好|被窝|床上|"
    r"只是.*话题|提到.*这个话题|话题先放|"
    # 2026-05-04 Tier 1 扩列：greeting 全集 + 询问角色当前状态/动作变体
    # （「晚上好」「晚安」「早安」原 regex 已含、此处补全 morning/midday/afternoon
    # 的对称形式 + 「晚上好啊」等口语尾词。re.search 不锚定、子串匹配均生效。）
    r"早上好|早啊|下午好|中午好|晚上好啊|你好啊?|"
    r"在做什么|在做啥|在干嘛|在干什么|在忙什么|忙什么|忙啥|忙不忙|"
    r"还在线|这个点|这会儿|正在做|做什么呢|做啥呢|"
    r"还没睡|这么晚还|这么早就|起来了吗|醒了吗)",
    re.IGNORECASE,
)
# 通用起源问法；部署专属的起源锚词（具体经历关键词）由 maintenance seeds 的 recall_gates 补充。
_SYSTEM_ORIGIN_RECALL_ANCHOR_RE = re.compile(
    r"(最开始|一开始|来历|起源|怎么来|怎么来的|从哪来|为什么创造|为啥创造|"
    r"创造这个系统|建立这个系统)"
)
_SYSTEM_META_TOPIC_RE = re.compile(
    r"(系统指令|系统提示|提示词|system\s*prompt|developer\s*(?:message|prompt|instruction)|"
    r"开发者指令|开发者提示|指令这个话题|提示词这个话题|话题[:：]?\s*(?:系统|指令|提示词))",
    re.IGNORECASE,
)
_VAGUE_META_TOPIC_RE = re.compile(
    r"^\s*(?:你还记得|还记得|记得|想起)?\s*(?:我(?:们)?(?:说|聊|提到|提过)?)?\s*"
    r"(?:这个|那个|这|那)(?:话题|段|件事|事)\s*(?:吗|么|嘛|[?？])?\s*$"
)
_SYSTEM_INSTRUCTION_TOPIC_RE = re.compile(r"(系统指令|话题[:：]?\s*系统|系统这个话题|系统指令这个话题)")
_EXPLICIT_ARCHIVE_META_RE = re.compile(r"(系统|QQ|qq|账号|封号|故障|bug|崩溃)")
_SHARED_PROFILE_META_RE = re.compile(r"(系统|创造|QQ|qq|账号|封号)")


def _memory_text_is_low_value_generalization(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    norm = _norm_text(raw)
    if _memory_surface_text_is_generic(raw):
        return True
    if len(norm) <= 10 and _EPHEMERAL_MEMORY_TEXT_RE.search(raw):
        return True
    return bool(_LOW_VALUE_MEMORY_GENERALIZATION_RE.search(raw))


def _memory_working_text_is_ephemeral_noise(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return True
    return bool(_WORKING_EPHEMERAL_NOISE_RE.search(raw))


def _memory_row_age_hours(row: dict | None = None) -> float:
    from tsuzuri.memory_core_impl.recall_scoring import _parse_dt
    item = row if isinstance(row, dict) else {}
    dt = _parse_dt(str(item.get("updated_at", "") or item.get("created_at", "") or ""))
    if dt is None:
        return 999999.0
    now = utc_now_naive()
    return max(0.0, (now - dt).total_seconds() / 3600.0)


def _memory_row_is_surfaceable_working(row: dict | None = None) -> bool:
    item = row if isinstance(row, dict) else {}
    return bool(
        _memory_row_is_working(item)
        and (
            item.get("working_continuity_lane")
            or item.get("working_direct_recall_lane")
        )
    )


def _memory_query_hits_removed_recall_tombstone(query: str) -> bool:
    norm = _norm_text(str(query or ""))
    if not ("记忆系统" in norm or "回忆系统" in norm):
        return False
    if not ("坏" in norm or "出问题" in norm):
        return False
    return bool("找回" in norm or "那次" in norm or "那件" in norm or "之前" in norm or "回忆" in norm)


def _text_mentions_system_origin(text: str) -> bool:
    raw = str(text or "")
    if _SYSTEM_ORIGIN_RECALL_ANCHOR_RE.search(raw):
        return True
    extra = _maintenance_recall_gates().get("system_origin_anchor_re")
    return bool(extra is not None and extra.search(raw))


def _memory_query_allows_system_origin(query: str) -> bool:
    return _text_mentions_system_origin(query)


def _memory_query_is_meta_instruction_topic(query: str) -> bool:
    raw = str(query or "").strip()
    if not raw:
        return False
    if _memory_query_allows_system_origin(raw):
        return False
    return bool(_SYSTEM_META_TOPIC_RE.search(raw) or _VAGUE_META_TOPIC_RE.search(raw))


def _memory_row_blocked_by_meta_topic(row: dict | None = None) -> bool:
    item = row if isinstance(row, dict) else {}
    item_id = str(item.get("item_id", "") or "")
    text = str(item.get("text", "") or "")
    character = str(item.get("character", "") or "").strip()
    item_type = str(item.get("item_type", "") or "").strip().lower()
    engram_type = str(item.get("engram_type", "") or "").strip().lower()
    source_kind = str(item.get("source_kind", "") or "").strip().lower()
    gates = _maintenance_recall_gates()
    for archive in gates.get("explicit_archives", ()):
        marker = archive.get("item_id_marker", "")
        if marker and marker in item_id:
            return True
    if item_type == "explicit_recall_archive" or engram_type == "explicit_recall_archive":
        return bool(_EXPLICIT_ARCHIVE_META_RE.search(text + item_id))
    if character == "*" and (item_type in ("profile_fact", "key_event") or source_kind in ("profile", "key_event")):
        if _SHARED_PROFILE_META_RE.search(text + item_id):
            return True
        extra = gates.get("meta_topic_profile_re")
        return bool(extra is not None and extra.search(text + item_id))
    return False


# 2026-05-11 加：collective 层 3rd-person 自指 filter
# 防止 "我向某角色坦白一件私事" 这种私聊内容（在角色A chat 写入 collective）
# 又被 角色B chat 的 collective recall 拽出来——视角错位 canon 灾难
def _collective_row_third_person_self_referent(row: dict | None = None, character: str = "") -> bool:
    """检测 collective row 是否含对当前 character 的 3rd-person 自指。

    返回 True 表示这条 collective 记录从当前角色视角读起来是 3rd-person 描述
    （e.g. 角色B chat 中读到 "我向角色B坦白" → 角色B 看到自己被第三人称指代、视角错位）。
    这种 row 应该从 character 的 recall pool 里 skip。

    只对 memory_role='collective' 的 row 有效——working / episodic 不参与判断。
    """
    if not isinstance(row, dict):
        return False
    role = str(row.get("memory_role", "") or "").strip().lower()
    if role != "collective":
        return False
    char = str(character or "").strip()
    if not char:
        return False
    text = str(row.get("text", "") or "")
    if not text:
        return False
    # 3rd-person 自指 patterns（私下行为指向 char、char 作为说话主体被引述）
    patterns = (
        f"向{char}",      # "我向角色B坦白"
        f"跟{char}",      # "我跟角色B说" / "我跟角色B聊"
        f"和{char}聊",    # "和角色B聊过"
        f"和{char}说",    # "和角色B说"
        f"对{char}说",    # "对角色B说"
        f"给{char}讲",    # "给角色B讲"
        f"{char}说",      # "角色B说没关系"
        f"{char}告诉",    # "角色B告诉我"
        f"{char}问",      # "角色B问我"
        f"{char}回答",    # "角色B回答"
        f"{char}回复",    # "角色B回复"
    )
    return any(p in text for p in patterns)


def _memory_explicit_archive_matches_query(row: dict | None = None, query: str = "", recall_focus: str = "") -> bool:
    item = row if isinstance(row, dict) else {}
    item_id = str(item.get("item_id", "") or "")
    text = str(item.get("text", "") or "")
    probe = f"{query} {recall_focus}".strip()
    # 每条运维整理的显式回忆归档由 seeds 声明自己的问法锚点；无 seeds 时不额外限制。
    for archive in _maintenance_recall_gates().get("explicit_archives", ()):
        id_marker = archive.get("item_id_marker", "")
        text_marker = archive.get("text_marker", "")
        if not ((id_marker and id_marker in item_id) or (text_marker and text_marker in text)):
            continue
        anchor = archive.get("anchor")
        if anchor is None or not anchor.search(probe):
            return False
        if archive.get("reject_system_instruction_topic") and _SYSTEM_INSTRUCTION_TOPIC_RE.search(probe):
            return False
        return True
    return True


def _memory_snapshot_text_index(snapshot: dict | None = None) -> str:
    try:
        return _norm_text(json.dumps(snapshot if isinstance(snapshot, dict) else {}, ensure_ascii=False))
    except Exception:
        return ""


_FOCUS_GENERIC_CHUNKS = {"你们", "我们", "他们", "她们", "这个", "那个", "这件", "那件", "事情", "的事"}
_TOPIC_ONLY_FACT_RE = re.compile(r"^\s*(?:话题|topic)[:：]\s*(.{1,48})\s*$", re.IGNORECASE)
_EXPLICIT_MEMORY_WRITE_RE = re.compile(
    r"(记住|帮我记|你要记得|以后|下次|我喜欢|我不喜欢|我讨厌|我希望|我想要|"
    r"我叫|叫我|一直都很爱你们|很爱你们|爱你们)"
)
_COLLECTIVE_ADDRESS_RE = live_pattern(r"(你们|大家|{COLLECTIVE}|{COLLECTIVE_LOWER}|全员|{COLLECTIVE_NOUN}|我们几个|你们几个)")
# 用户明确要求保密 / 别外传的原话（2026-10-19 外部审计 COMP-004）：这类句子
# 不得升 collective、不得进跨角色 digest、不得 gossip 扩散——否则「先别告诉某成员」
# 恰好被共享给那位成员。宁可漏写一条集体记忆，也不把秘密写进共享层。
_USER_CONFIDENTIAL_RE = re.compile(
    r"(?:别|不要|不许|不能|千万别|先别|暂时别|还别|先不要|暂时不要)"
    r"(?:告诉|透露|说出去|讲出去|外传|声张|张扬|让[^，。！？!?\n]{1,12}?知道|"
    r"(?:跟|和|对|向|给)[^，。！？!?\n]{1,12}?(?:说|讲|提|透露))"
    r"|保密|替我瞒|帮我瞒|先瞒着|瞒着[^，。！？!?\n]{1,12}|你知我知"
    r"|只(?:跟|和|对)你(?:一个人)?(?:说|讲)|只告诉你|就告诉你一个"
    r"|(?:这|这个|这件事)是(?:个|我的)?秘密|是秘密|秘密哦|秘密喔|秘密啦|偷偷告诉你"
)


def _user_text_requests_confidentiality(text: str) -> bool:
    return bool(_USER_CONFIDENTIAL_RE.search(str(text or "")))
_ASSISTANT_ONLY_FACT_RE = re.compile(r"^(我觉得|我想|我会|我能|我可以|我应该|我记得|角色|她|她们)")


# ═════════════════════════════════════════════════════════════════════════
# Memory Feature Extraction (Phase Q.W2 / Option E Layer 1, 2026-05-04)
# ═════════════════════════════════════════════════════════════════════════
# Cheap content features at write time. 8 features → typed_score for Layer 2 state machine.
# Design principle (用户 plan 2026-05-04)：regex 是召回候选的门、heuristic 结构是排序依据。
# specificity = surface feature（句子层）；user_fact_density = semantic feature（关于用户）；
# emotional_weight = 价值绑定情绪、不是短暂 reactive；
# ephemeral 强档 → 直接 stale_archive、不进 candidate。

# 1. EXPLICIT_MEMORY_INTENT — 用户显式表达"记住这个" + 自我事实声明 + 关系提及
_EXPLICIT_MEMORY_INTENT_RE = re.compile(
    r"(记住|帮我记|你要记得|你记得|别忘了|不要忘|"
    r"以后|下次|每次|今后|"
    r"我叫|叫我|我家在|我住在|我从小|我以前|我之前|"
    r"我喜欢|我不喜欢|我讨厌|我害怕|我希望|我想要|我最在意|我特别|"
    r"我家(?:人|里)?|我妈|我爸|我哥|我姐|我弟|我妹|我老板|我同学|我老师|我朋友)"
)

# 2. COLD_START_FAST_TRACK — 强信号身份/偏好声明、绕过 reuse gate
# 这类不能等 reuse、否则用户感觉"说了等于白说"
# 允许「我从小就特别讨厌」「我最讨厌」中间有副词
_COLD_START_IDENTITY_RE = re.compile(
    r"(我叫\s*[\w一-鿿]{1,8}|"
    r"叫我\s*[\w一-鿿]{1,8}|"
    r"我家(?:在|住|是)|我住在|我老家|"
    r"我生日是|我生日在|我\s*\d+\s*岁|我今年\s*\d+|"
    r"我特别(?:讨厌|喜欢|怕|在意|害怕)|"
    r"我最(?:讨厌|喜欢|怕|在意|害怕)|"
    r"我从小[就开始一直]*[一-鿿]{0,4}(?:讨厌|喜欢|怕|害怕|在意))"
)

# 3. SPECIFICITY — 表面：含具体时间/地点/人名/数字/活动锚点
# 2026-05-04 audit fix：原 hardcoded 短列表草率、用形态学 catch-all + 系统化分类列表重做
#
# (a) TIME — 具体日期 / 时间锚 / 学龄阶段 / 模糊时间 / 时段
# 顺序非常重要：specific patterns 必须先于 generic（例如 \d 年代 必须在 \d 年 之前）
_SPECIFICITY_TIME_RE = re.compile(
    # === 日期格式（最 specific、灵活分隔符）===
    r"\d{4}\s*[年./\-]\s*\d{1,2}\s*[月./\-]\s*\d{1,2}|"
    # === 时段命名 ===
    r"(?:早晨|清晨|早上|上午|中午|下午|傍晚|黄昏|晚上|夜里|深夜|凌晨|半夜|午夜)|"
    # === 周/星期 ===
    r"(?:周|星期|礼拜)\s*[一二三四五六日天](?![一-鿿])|"
    # === 学龄（specific 单位）===
    r"(?:幼儿园|学前班|小学|初中|高中|大学|大学院|研究生|博士|本科|硕士)\s*(?:[一二三四五六七八九十\d]+\s*年级)?|"
    r"\d+\s*年级|"
    # === 年代（必须在 \d 年 / \d+ 年 之前）===
    r"\d{2,4}\s*年代|"
    # === 4-digit 年（必须在 \d+ 年 之前、且不能后跟"级"）===
    r"\d{4}\s*年(?!\s*级)|"
    # === 月日 ===
    r"\d{1,2}\s*月\s*\d{1,2}|"
    # === 时刻（\d点 / \d点半 / \d:\d）===
    r"\d{1,2}\s*[:：]\s*\d{2}|"
    r"\d{1,2}\s*点(?:半|多|左右|整|钟)?|"
    # === 年龄 ===
    r"\d+\s*多?\s*岁|"
    # === 时长（数字+单位+前后内/以前/以后/期间）— 必须有时间锚后缀、避免 generic over-match ===
    r"\d+\s*(?:天|周|月|年|小时|分钟|秒|分|刻钟)\s*(?:[前后内]|以[前后内]|期间|之[前后内])|"
    r"\d+\s*(?:天|周|月|年)(?=\b|前|后|内)|"  # bare \d+年 仅当后面是 词边界 / 前 / 后 / 内
    # === 季节 ===
    r"(?:春|夏|秋|冬)(?:季|天)(?![一-鿿])|"
    # === 相对时间锚（确定）===
    r"昨天|今天|明天|前天|后天|大前天|大后天|"
    r"上周|这周|下周|上上周|下下周|本周|"
    r"上个月|这个月|下个月|本月|"
    r"去年|今年|明年|前年|后年|"
    r"上次|下次|这次|那次|每次|有时|有时候|"
    # === 模糊时间锚 ===
    r"刚才|刚刚|刚好|前阵子|那阵子|那时候|这时候|那会儿|这会儿|"
    r"以前|之前|从前|过去|曾经|当时|"
    r"以后|之后|未来|将来|后来|后续|后面|"
    r"很久(?:以前|以后)?|早就|早早|"
    r"这些年|这几年|那些年|那几年|从小|长大后|长大以后|小时候|"
    # === 节假日锚（非定义性、但 specific）===
    r"春节|元旦|国庆|中秋|端午|清明|劳动节|"
    r"圣诞|新年|跨年|除夕|大年初[一二三四五六]|"
    r"暑假|寒假|国庆假|春假|黄金周|GW"
)

# (b) PLACE — 形态学 catch-all + 极少量高频地名锚点
# 2026-05-09 重构：删除具体 entity 枚举（具名地名/地标无穷无尽不可维护），
# 只保留两类：
#   1. 闭集地点类型词（咖啡馆/医院/车站等场所类型词，不是 entity）
#   2. 形态学后缀模式（…寺/…神社/…大学，配 prefix 限制 + 动词排除）
# 真实 entity（西町/北町/群组 地标/海外地名）改靠 embedding_novelty 信号识别。
_SPECIFICITY_PLACE_RE = re.compile(
    # === 通用泛指地点类型（闭集、不是 entity）===
    r"咖啡店|咖啡馆|奶茶店|甜品店|蛋糕店|餐厅|食堂|料理店"
    r"|酒店|宾馆|旅馆|青旅|民宿"
    r"|车站|地铁站|公交站|机场|港|码头|火车站"
    r"|商场|超市|便利店|药店"
    r"|图书馆|博物馆|美术馆|水族馆|动物园|植物园|游乐园|体育馆"
    r"|音乐厅|歌剧院|剧场|剧院|电影院|展览馆|纪念馆"
    r"|医院|诊所|银行|邮局|教室|宿舍"
    r"|公园|广场|大道"
    # === 形态学 catch-all（高价值、低 over-match 风险）===
    # 寺庙 / 神社 / 神宫 / 道观：prefix 限 2 字、起始非代词/动词
    r"|(?!我|你|他|她|它|去|来|到|走|从|上|找|看|玩|逛|吃|喝|做|带|陪|想|要|正|在|有|没|不|和|跟|与|刚|过)[一-鿿]{2}(?:神宫|神社|清真寺|道观)"
    r"|(?!我|你|他|她|它|去|来|到|走|从|上|找|看|玩|逛|吃|喝|做|带|陪|想|要|正|在|有|没|不|和|跟|与|刚|过)[一-鿿]{2,3}寺(?![庙院子家里])"
    # 大学 / 学院（命中 4 字以内的标准格式：临川大学、明治大学、北京大学）
    r"|(?!我|你|他|她|它|去|来|到|走|从|上|找|看|玩|逛|吃|喝|做|带|陪|想|要|正|在|有|没|不|和|跟|与|刚|过)[一-鿿]{2}(?:大学|学院)"
    # === 极少数高频锚点（< 30 项；用户母语圈 + 主要日本城市）===
    r"|临川|臨川|大阪|京都|北京|上海|广州|深圳|杭州|南京|成都|重庆|香港|台北"
    r"|纽约|伦敦|巴黎|首尔|曼谷"
    r"|中国|日本|美国|英国|法国"
    # === 活动类型（闭集）===
    r"|演唱会|演奏会|展览|画展|漫展|读书会|分享会|发布会"
)

# (c) PROPER — 已删除（2026-05-09）
# 历史曾枚举 ~120 hardcoded 项（编程语言/AI/游戏/品牌/群组 canon 等）。
# 用户判定 entity 集合"无穷无尽且持续更新、根本照顾不完"——hardcoded list 不可维护。
# 该信号的 quota（原 typed_score 0.20 中的一部分）改由：
#   - embedding_novelty (0.06)：与最近 N turn 平均 cosine 反向
#   - digit_density (0.02)：日期/价格/年龄/ID
#   - rare_char_ratio (0.02)：片假名/CJK Ext/大写英文 token
# 共同承担。
_SPECIFICITY_PROPER_RE = None  # 占位、保留名字以防外部 import；任何 .findall 调用应先 if 判空

# 4. USER_FACT_DENSITY — 语义：是否描述用户自己的稳定属性
# 主语+谓语关系：「我 + (副词)? + 状态/属性动词」、允许中间有副词修饰
_USER_FACT_PATTERN_RE = re.compile(
    r"我[一-鿿]{0,3}(?:是|不是|在|从小|以前|之前|一直|平时|经常|总是|"
    r"喜欢|不喜欢|讨厌|害怕|想|希望|觉得|认为|愿意|"
    r"住|生日|年纪|岁|"
    r"做|学|学过|玩|玩过|看|读|写|画|更想|更喜欢|更看重)"
    r"|我家|我妈|我爸|我哥|我姐|我弟|我妹|我老板|我同学|我老师"
)
# Negative: assistant 自语 / 第三人称（不是 user fact）
_USER_FACT_NEGATIVE_RE = re.compile(
    r"^(她|他|它|她们|他们|你|你们|刚才|这|那|这个|那个)"
)

# 5. STANCE_OR_VALUE — 态度立场陈述
_STANCE_RE = re.compile(
    r"(其实|说实话|实际上|老实说|坦白说|"
    r"比起.*[更也还]|反而|宁可|宁愿|不如|与其|"
    r"我更想|我更喜欢|我更看重|我更在意|"
    r"在我看来|对我来说|对我而言|我个人|我自己觉得|私心|"
    r"重要的是|关键是|核心是)"
)

# 6. EMOTIONAL_WEIGHT — 价值绑定情绪（不是短暂 reactive）
# Reactive (NOT counted)：独立成句的情绪词
_EMOTIONAL_REACTIVE_RE = re.compile(
    r"^[^。.，,！!]*(好难过|气死了|好烦|郁闷死了|烦死了|开心死了|笑死|乐死|"
    r"嗨|爽|爽爆|无语|无奈|崩溃|绝)[^。.，,！!]*$"
)
# 2026-05-11 拓宽（用户拍板）：情感类 turn 都该被识别保护、不只 disclosure→reassurance
# 词典分 4 类、覆盖更全：
#   · vulnerability：自感的脆弱情绪（用户对自己的评价 / 内核流露）
#   · connection：关系性 / 价值绑定情绪（人对人）
#   · seeking：意义 / 价值 / 目标 / 思考层情绪（也是 canon 关键）
#   · preference：长期偏好（喜欢 / 讨厌 / 在意）
_EMOTION_WORDS_VULNERABILITY = (
    "难过|失落|失望|后悔|内疚|心虚|不安|害怕|担心|忧心|紧张|焦虑|纠结|委屈|自责|"
    "羞愧|惭愧|自卑|脆弱|崩溃|挫败|无力|疲惫|累|寂寞|孤独|迷茫|空虚|抑郁|"
    "不好意思|抱歉|对不起|歉意|愧疚|"
    # 自我评价类（隐性脆弱）
    "没用|不够好|做不好|配不上|做不到|不行|"
    # 情绪反思（含程度副词）
    "有时候.*?觉得|总觉得.*?自己"
)
_EMOTION_WORDS_CONNECTION = (
    "心疼|感动|怀念|思念|想念|想你|牵挂|挂念|"
    "感激|感谢|谢谢|多谢|"
    "温暖|安心|被理解|被接住|被看见|"
    "依赖|需要你|离不开|信任|放心|信赖|"
    "陪伴|陪着|相处|默契|不孤单|有你"
)
_EMOTION_WORDS_SEEKING = (
    "意义|值得|价值|目的|目标|理想|希望|愿望|信念|"
    "为什么|凭什么|凭借什么|"
    "努力|坚持|放弃|失败|成功|"
    "改变|不一样|重要|要紧|放在心上"
)
_EMOTION_WORDS_PREFERENCE = (
    "讨厌|喜欢|爱|讨厌过|不喜欢|很怕|在意|向往|羡慕|嫉妒|期待"
)

_EMOTION_TEMPORAL_ANCHOR = (
    "因为|所以|让我|使我|让你|对.*[来说而言]|"
    "一直|总是|这些年|很久|从小|长大后|每次|那时候|"
    "现在|后来|以前|曾经|那段|当时|从那以后|从此"
)

# Value-bound（任一组合即命中）：
# A) 情绪词 + 时间/因果锚（双命中、强）
# B) 情绪词 + 「你 / 我们 / 一起」式关系第二人称（中、关系性情绪本身就有价值绑定）
# C) **disclosure 形态**：「坦白/告诉你/告诉我/说实话/坦白说/告诉过你」+ 任意情绪词（弱锚也命中）
_EMOTIONAL_VALUE_BOUND_RE = re.compile(
    # Pattern A1: 情绪词 → 时间/因果锚
    rf"(?:{_EMOTION_WORDS_VULNERABILITY}|{_EMOTION_WORDS_CONNECTION}|"
    rf"{_EMOTION_WORDS_SEEKING}|{_EMOTION_WORDS_PREFERENCE})"
    rf".*?(?:{_EMOTION_TEMPORAL_ANCHOR})"
    # Pattern A2: 时间/因果锚 → 情绪词
    rf"|(?:{_EMOTION_TEMPORAL_ANCHOR}).*?"
    rf"(?:{_EMOTION_WORDS_VULNERABILITY}|{_EMOTION_WORDS_CONNECTION}|"
    rf"{_EMOTION_WORDS_SEEKING}|{_EMOTION_WORDS_PREFERENCE})"
    # Pattern B: 情绪词 + 第二人称 / 共同体（关系绑定）
    rf"|(?:{_EMOTION_WORDS_VULNERABILITY}|{_EMOTION_WORDS_CONNECTION}|"
    rf"{_EMOTION_WORDS_SEEKING}|{_EMOTION_WORDS_PREFERENCE})"
    rf".*?(?:你|你们|我们|一起|互相)"
    rf"|(?:你|你们|我们|一起|互相).*?"
    rf"(?:{_EMOTION_WORDS_VULNERABILITY}|{_EMOTION_WORDS_CONNECTION}|"
    rf"{_EMOTION_WORDS_SEEKING}|{_EMOTION_WORDS_PREFERENCE})"
    # Pattern C: disclosure 标记词 + 任意情绪词（自我披露本身就是情感事件）
    rf"|(?:坦白|告诉你|告诉过你|告诉我|说实话|实话说|向你说|对你说|"
    rf"和你说|跟你说|我承认|不瞒你|有件事)"
    rf".*?(?:{_EMOTION_WORDS_VULNERABILITY}|{_EMOTION_WORDS_CONNECTION}|"
    rf"{_EMOTION_WORDS_SEEKING}|{_EMOTION_WORDS_PREFERENCE})"
)

# 2026-05-11 新增：关系性 / 偏好情感词单独命中（不需时间锚）
# 这类情感本质上就是 value-bound（你不会 reactive 地"想你 5 秒"）、单独出现也算长期情感
_EMOTION_CONNECTION_OR_PREFERENCE_ALONE_RE = re.compile(
    rf"(?:{_EMOTION_WORDS_CONNECTION}|{_EMOTION_WORDS_PREFERENCE})"
)

# 2026-05-11 新增：disclosure_acknowledgment 双向 pattern（user 披露 + assistant 安抚）
# 这是 canon 上最关键的 turn 类——角色对用户的脆弱接住、必须长期记忆
# 命中条件：user 含 disclosure 或 vulnerability、assistant 含 reassurance 词
_DISCLOSURE_USER_RE = re.compile(
    rf"(?:坦白|告诉你|告诉过你|告诉我|说实话|实话说|向你说|对你说|"
    rf"和你说|跟你说|我承认|不瞒你|有件事|"
    rf"{_EMOTION_WORDS_VULNERABILITY}|"
    rf"会让你失望|让你觉得|你会不会觉得|你会不会嫌|你会不会怪)"
)
_REASSURANCE_ASSISTANT_RE = re.compile(
    r"(?:不会|没关系|没事|都好|不要紧|无所谓|我懂|理解|"
    r"现在.*?就好|现在.*?够了|此时.*?就好|这样就好|够了|"
    r"我会.*?在|我会.*?陪|不要走|别走|别担心|放心|"
    r"喜欢这样|这样很好|挺好的|挺好|"
    r"接住|抱抱|拍拍|"
    r"我也|我们也|我们一样)"
)

# 7. EPHEMERAL_PENALTY — 强规则、命中即 archive、不进 candidate
# (a) 短问候/招呼/反应词
_EPHEMERAL_GREETING_RE = re.compile(
    r"^[^。.！!？?]{0,8}(?:晚安|早安|早上好|午安|你好|拜拜|再见|回见|"
    r"嗯|啊|哦|哇|噗|哈哈+|嘿嘿|呵呵|"
    r"好的|好啊|可以|没事|没问题|没关系|那行|这样|这样啊)[^。.！!？?]{0,8}$"
)
# (b) 操作指令 / slash command
_EPHEMERAL_COMMAND_RE = re.compile(
    r"^[/!@#$%^&*]"
    r"|^(刷新|重启|清屏|清除|reload|reset|debug|test|status|help)\b"
)
# (c) 短暂状态/天气/位置（不是深度内容）
_EPHEMERAL_TRANSIENT_RE = re.compile(
    r"^.{0,15}(?:刚醒|刚到家|刚起床|要睡了|准备睡|快睡了|"
    r"在.{0,4}旁边|今天天气|刚下雨|正下雨|今天好热|今天好冷).{0,15}$"
)

# 8. GENERIC_PENALTY — 礼貌问候 / 日常关心 / 短泛化、弱罚
_GENERIC_RE = re.compile(
    r"^[^。.！!？?]{0,12}(?:吃了吗|忙什么|在干嘛|累不累|"
    r"加油|辛苦了|注意身体|早点睡|别熬夜|多保重|休息一下)[^。.！!？?]{0,12}$"
)


# Path 4 chitchat fact filter (2026-05-09 + 2026-05-09 收紧)
# 拦"长 chitchat 告辞 / 应答口头禅"：不属于 stable fact、不应作为 semantic 落库。
# 比 _EPHEMERAL_*RE 覆盖范围更宽（不限制句长）、专门针对 fact 候选文本。
# 命中任一即视为 chitchat fact、suppressed。
#
# 注：原 v1 包含 `^[嗯啊哦][，,]` / `^好[，,]` 类宽松 starter 模式、误杀
# "啊，我最近还没法过去..."（用户的真实陈述句）。v2 收紧：starter 必须是
# 显著的应答 marker（是吗/真的吗/原来/可太好了 等）、单纯感叹词开头放行。
# 2026-05-12: 一次性"承诺/允诺/同意/认知 ack"模式拦截
# 拦截 "X 承诺/答应/保证 + 下次..." / "X 同意了..." / "X 知道 Y 不喜欢/讨厌..." 这类
# turn-specific 事件描述（不是 stable user fact），避免它们落 semantic 层。
# 例:
#   "用户知道角色A不喜欢被这样问，承诺下次不再问" → 拦截（混合两个 turn-level ack）
#   "用户答应下次早点睡"                            → 拦截
#   "用户喜欢熬夜调代码"                            → 放行（stable trait）
_PATH4_ONESHOT_PROMISE_RE = live_pattern(
    r"(承诺|答应|保证|发誓)(?:下次|以后|再也|不再|绝不)"          # 承诺/答应 + 时间指示
    r"|(?:下次|以后)(?:不(?:再|会)|再也不|绝不)"                # 下次不再 / 以后不会
    r"|知道(?:{AGENTS}|{USER}|开发者|你|他|她|TA|对方)(?:不喜欢|不愿意|讨厌|介意|烦)"
    r"|向(?:{AGENTS}|{USER}|开发者)(?:道歉|认错)"
    r"|(?:同意|接受)了(?:{AGENTS}|{USER}|开发者)的"
)

_PATH4_CHITCHAT_FACT_RE = re.compile(
    # 1) 显著应答 / 反问 marker 开头（这些几乎不会出现在 user 主动陈述里）
    r"^是[吗呀啊](?:[，,。.！!？?\s]|$)"              # 是吗 / 是吗，
    r"|^真的[吗呀啊](?:[，,。.！!？?\s]|$)"          # 真的吗
    r"|^原来[如这是](?:[此样]|是这|是那)?"            # 原来如此/原来这样
    r"|^[哈嘿呵嘻]{2,}"                                # 哈哈/嘿嘿/呵呵 (2+ 字)
    r"|^行(?:吧)?(?:[，,。.~!！]|$)"                  # 行 / 行吧 + 标点或结尾
    r"|^可以(?:[，,。.~!！]|$)"                        # 可以 (单独成句)
    # 2) 告辞 / 收尾短语（任意位置、强 chitchat 信号）
    r"|下次再[聊说见]"
    r"|下回再[聊说见]"
    r"|改天再[聊说见]"
    r"|那(?:我|就)?先[去走休][^。.！!？?]{0,8}[了着]"  # 那我先去...了/那就先休息了
    r"|先去[看读吃睡][^。.！!？?]{0,4}[了着]"          # 先去看书了/先去睡了
    # 3) 应答 chunk（assistant 闲聊回复的 fingerprint）
    r"|可太好了"
    r"|挺好的(?:[，,。.！!]|$)"
    r"|那挺好(?:[，,。.！!]|$)"
    r"|我懂[~～]?(?:[，,。.！!]|$)"
)

# 2026-05-15: transient daily activity reports — not durable semantic facts
# "吃了饭/看了视频/喝了咖啡" 是一次性日常行为、不应进入永久语义记忆
_PATH4_TRANSIENT_DAILY_RE = re.compile(
    # 一次性日常行为报告 — "吃了饭/看了视频/喝了奶茶" 不是 durable semantic fact
    # 设计原则: 了/过 + 日常动词 + 日常宾语 = transient activity
    # 不加 "买/逛/玩" → 这些可能承载偏好信息（"买了吉他""逛乐器店"）
    r"(?:刚|刚才|才|已经|今天|昨天|刚刚)"
    r"(?:.{0,6})(?:吃[了过]|喝[了过]|看[了过]|做[了过])"
    r"|(?:吃[了过].{0,3}(?:饭|早[饭餐]|午[饭餐]|晚[饭餐]|夜宵|东西|零食|面|拉面|便当|外卖))"
    r"|(?:喝[了过].{0,3}(?:水|茶|咖啡|奶茶|饮料|酒|牛奶|汤|果汁|可乐|奶昔))"
    r"|(?:看[了过].{0,10}(?:视频|更新|直播|动漫|剧|电影|综艺|番|频道))"
    r"|(?:做[了过].{0,3}(?:饭|菜|作业|任务|工作|事|家务|卫生))"
    r"|(?:刚[吃完喝看做好].{0,4}(?:[了，。]|$))"
)


# ═════════════════════════════════════════════════════════════════════════
# 2026-05-09 重构：替代 hardcoded PROPER 信号的统计/语义特征
# ═════════════════════════════════════════════════════════════════════════
# 哲学：entity 集合无穷无尽不可枚举，靠"统计形态 + 语义稀有度"代替关键词命中。
# 三个新 feature 总占 typed_score 0.10（specificity 从 0.20 缩为 0.10、剩 0.10 给三者）。

# 大写英文 token（长度 ≥ 2）：可捕获专名 / 缩写（GPT, OpenAI, 群组 等）
_RARE_UPPER_TOKEN_RE = re.compile(r"[A-Z][A-Za-z]{2,}")
