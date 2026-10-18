"""Query expansion, ACT-R recency, co-occurrence/spreading activation and JSON helpers.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import json
import math
import os
import re
from tsuzuri.persona_registry import agent_names, is_agent, text_variants, user_name
import time
from datetime import datetime, timezone
from tsuzuri.time_source import utc_now_naive
from tsuzuri.memory_core_impl.config import _safe_float, _utc_now_iso
from tsuzuri.memory_core_impl.memory_rows import _bigrams, _lexical_similarity, _norm_text
from tsuzuri.memory_core_impl.features import _developer_display_name, _TOPIC_LABEL_KEYWORDS
from tsuzuri.memory_core_impl.explicit_recall import _query_requires_first_person_fact_ownership

try:
    import jieba as _jieba_mod
    _jieba_mod.setLogLevel(20)  # Suppress jieba debug logs
    _HAS_JIEBA = True

    def _word_jaccard(a: str, b: str) -> float:
        wa = set(_jieba_mod.cut(_norm_text(a)))
        wb = set(_jieba_mod.cut(_norm_text(b)))
        wa.discard(""); wb.discard("")
        if not wa or not wb:
            return 0.0
        inter = len(wa & wb)
        union = len(wa | wb)
        return float(inter) / float(union) if union > 0 else 0.0
except ImportError:
    _HAS_JIEBA = False
    # 一次性 loud 告警：jieba 缺失会让 word-jaccard 永久返 0、_lexical_similarity 退化为
    # char-bigram-only，biomimetic gate 的 relevance 信号塌掉、召回退回 embedding-only
    # （实测 canon 召回 90%→80%、关键 keyword-overlap 记忆被 gate 掉）。这类 silent
    # fallback 是历史血泪点（参见 sentence_transformers 被 uninstall 教训）、必须 loud。
    _tsz_emit(
        "[MemoryCore] WARNING: jieba 不可用 — 记忆召回 lexical 通道(word-jaccard)已禁用、"
        "召回会退化为 embedding-only。请 `pip install jieba` 修复。",
        flush=True,
    )

    def _word_jaccard(a: str, b: str) -> float:
        return 0.0  # fallback: no jieba, word-level disabled


def _parse_dt(iso_value: str | None) -> datetime | None:
    s = str(iso_value or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1]
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is not None:
        try:
            dt = dt.replace(tzinfo=None)
        except Exception:
            pass
    return dt


# ── A. 语义查询扩展（深度语义联想机制）────────────────────────
# 论文参考: Carpineto & Romano 2012《Survey of Automatic Query Expansion in IR》
# Voorhees 1994 (synonym-based QE)
#
# 核心思路：将单 query 扩展为 2-3 语义等价变体、对每个变体单独 embedding、
# 候选评分时取 cosine max，让深层语义联想（"怕被忘记" → "担心失联"/"渴望记住"/"孤独"）
# 能召回字面不重合但语义相关的 episodic canon。
#
# Phase A1: 规则版（character-specific topic dict + 同义词扩展）
# Phase A2: LLM 版（v4-flash 自调、温度 0.5、变体上限 3）

# Character-specific topic clusters: 每个主题词激活时附带 expansion bag
# 2026-05-17 大幅精简：原表 16 条、绝大多数因 `replace(topic, exp[0])` 字面替换
# 不分上下文、容易语义漂移。审查后只保留 5 条「双向自然且不易搭配错误」的：
#   - 删 "忘记":["记住"...] — 反义词、"我忘记了" → "我记住了" 完全反转
#   - 删 "等":["等待"...] — 单字太通用、"等下" → "等待下" 不通
#   - 删 "怕"/"孤独"/"认识"/"记得"/"想你"/"陪我" — 单字命中导致语法/时态/主宾漂移
#   - 删 "向阳"/"羁绊" — 群组 主题词的意象不可降级到泛指
#   - 删 "系统":["项目"...] — 宿主应用 上下文里"系统"是特定产品（这个 AI chat 系统）、不是泛指
_TOPIC_EXPANSIONS = {
    # 群组 canon 词（user 主动说才命中、且基本同义）
    "解散": ["分开", "结束", "最后"],
    # 开发者/系统层（弱同义、不破语法）
    "记忆": ["回忆", "片段", "存档"],
    "崩溃": ["坏掉", "出错", "故障"],
    # 2026-05-17 删 "封号":["被封"...] — "账号被封号了" → "账号被被封了" 重复"被被"。
    # 2026-08-04 [QueryExpand 规则版切分]：补充安全同义词对。
    # 只加整句替换不破语法的项（"活动室"→"活动室"、"期末考试"→"期末测验" 都成立）；
    # 禁加会误匹配子串的英文词（如 "live" 会命中 "hall" → 产出 "演唱会house"）。
    "集训": ["练习", "合训"],
    "演唱会": ["live", "演出"],
    "考试": ["测验", "备考"],
    "打工": ["兼职"],
    "水族馆": ["aquarium"],
    "作业": ["homework", "功课"],
    "樱花": ["桜"],
}


def _rule_expand_query(query: str, character: str = "") -> list[str]:
    """规则版同义替换 query expansion：基于 _TOPIC_EXPANSIONS 字典扫描 query 关键词、
    对每个匹配主题生成所有同义变体（最多 2 个变体、不含原 query）。
    [2026-08-04] 从 expansions[:1] 改为全部——"演唱会"→["live","演出"] 之前只出 "live"，
    "演出" 同义桥接（匹配含"演出"的记忆）完全失效。"""
    variants: list[str] = []
    if not query or len(query) < 4:
        return variants
    seen_topics = set()
    for topic, expansions in _TOPIC_EXPANSIONS.items():
        if topic in query and topic not in seen_topics:
            seen_topics.add(topic)
            for exp in expansions:
                variant = query.replace(topic, exp)
                if variant != query and variant not in variants:
                    variants.append(variant)
                    if len(variants) >= 2:
                        return variants
    return variants


def _semantic_query_expand_enabled() -> bool:
    """SEMANTIC_QUERY_EXPAND 总闸（'0'/'false'/'no' → 关闭）。"""
    import os as _os
    return not (_os.getenv("SEMANTIC_QUERY_EXPAND", "1") or "1").strip().lower() in ("0", "false", "no")


# ── [QueryExpand] 规则版语句切分（2026-08-04）────────────────────────────
# 之前的 query expansion 只有 4 词条同义替换、且不拆子句、变体不喂候选池——
# 实测 10 条 realistic query 里 7 条 0 变体、等于死代码。
# 新机制拆两层：
#   1) _split_query_clauses — 按标点 + 时间连词把复合句拆成语义子句
#      （"上次集训吵架之后角色D说了什么" → ["上次集训吵架", "角色D说了什么"]），
#      解决 BGE-M3 对长复合句 mean-pooling 稀释各概念的问题。
#   2) _extract_query_keywords — 抽短高精度关键词（主角名/领域词/jieba），
#      短 query 变体 embed 精度远高于整句（"晴天伴走" 这类实体）。
# 子句 + 关键词随后回喂 _query_items 的 FTS/LIKE 候选池（见 retrieve_memory_context）。
_CLAUSE_PUNCT_RE = re.compile(r"[，,。．.？?！!、；;：:…—…~～\n]+")
# 时间/话语连词只放安全的边界词——"和/或/与" 会误拆 "猫和老鼠"，不放。
_CLAUSE_CONJUNCTIONS = ("然后", "后来", "接着", "另外", "还有", "之后", "随后", "再说")
_CLAUSE_LEAD_RE = re.compile(r"^(?:然后|之后|后来|接着|另外|还有|随后|再说)[，,：:\s]*")
_CLAUSE_CONJUNCTION_MIN_SIDE_LEN = 4
_CLAUSE_CONJUNCTION_IDIOMS = {"后来": ("后来居上",)}


def _split_query_clauses(query: str, max_clauses: int = 4) -> list[str]:
    """按标点 + 时间连词把复合 query 拆成语义子句（保序、去空、≥2 字、上限 max_clauses）。"""
    raw = str(query or "").strip()
    if not raw:
        return []
    segments = [s.strip() for s in _CLAUSE_PUNCT_RE.split(raw) if s and s.strip()]
    out: list[str] = []
    for seg in segments:
        pending = [seg]
        # 逐连词切分。两侧都要有足够信息，避免把“我们之后再说吧”拆成泛词“我们”；
        # 固定表达也不切，避免“角色D后来居上了吗”产生错误人物变体。
        for conj in _CLAUSE_CONJUNCTIONS:
            nxt: list[str] = []
            for part in pending:
                idioms = _CLAUSE_CONJUNCTION_IDIOMS.get(conj, ())
                if conj in part and not any(idiom in part for idiom in idioms):
                    head, _, tail = part.partition(conj)
                    head = head.strip()
                    tail = tail.strip()
                    if (len(head) >= _CLAUSE_CONJUNCTION_MIN_SIDE_LEN
                            and len(tail) >= _CLAUSE_CONJUNCTION_MIN_SIDE_LEN):
                        nxt.append(head)
                        nxt.append((conj + tail).strip())
                    else:
                        nxt.append(part)
                else:
                    nxt.append(part)
            pending = nxt
        for part in pending:
            stripped = _CLAUSE_LEAD_RE.sub("", part).strip()
            # “然后我们去了 活动场地”可剥连词；“还有珠吗”保留原句，避免只剩低信息“珠吗”。
            c = stripped if len(stripped) >= _CLAUSE_CONJUNCTION_MIN_SIDE_LEN else part.strip()
            if len(c) >= 2 and c not in out:
                out.append(c)
            if len(out) >= max_clauses:
                return out
    return out


_QUERY_KEYWORD_MIN_LEN = 2
_QUERY_KEYWORD_MAX_LEN = 12
# 领域词表里过于泛用的词（作独立 LIKE query 会拉进大量无关行、但本身不具判别力）
_QUERY_KEYWORD_STOP = frozenset({
    "发生", "消息", "事件", "事情", "感觉", "关系", "问题",
    "地方", "时候", "东西", "最近", "现在", "昨天", "今天", "明天", "早上", "晚上", "下午",
})


def _extract_query_keywords(query: str, character: str = "", max_keywords: int = 4) -> list[str]:
    """从 query 抽短高精度关键词变体：领域词（_TOPIC_LABEL_KEYWORDS）+ 主角名/开发者名 + jieba 名词。

    只抽 len∈[2,12] 的子串、去重、按长度降序（更具体优先）、上限 max_keywords。
    jieba 名词层覆盖固定字典外的具体实体（水族馆/樱花/企鹅 等），
    让"记忆提到同名词、但 query 整句词法稀释"的场景也能被词法门禁命中。
    """
    from tsuzuri.memory_core_impl.meta_policy import _CHARACTER_NAME_ALIASES
    raw = str(query or "").strip()
    if not raw or len(raw) < 2:
        return []
    # 2026-09-14: 第一人称事实问询（"角色A，你还记得我最喜欢的花是哪种"）里，
    # 角色名/开发者名只是称呼语、对"我 X 是什么"零判别力；裸名变体会灌大角色
    # 档案事实、把真正答案挤出 top-k（郁金香 2026-09-14 失败案例）。此时名字
    # 不进关键词变体；以角色为主题的问句（"角色A的生日"）没有第一人称标记，
    # 不受影响。
    _first_person_topic_query = _query_requires_first_person_fact_ownership(raw)
    hits: list[str] = []
    raw_l = raw.lower()
    for _kws in _TOPIC_LABEL_KEYWORDS.values():
        for _kw in _kws:
            kw = str(_kw or "").strip()
            if (_QUERY_KEYWORD_MIN_LEN <= len(kw) <= _QUERY_KEYWORD_MAX_LEN
                    and kw not in _QUERY_KEYWORD_STOP
                    and kw in raw and kw != raw):
                hits.append(kw)
    for _aliases in _CHARACTER_NAME_ALIASES.values():
        for _a in _aliases:
            a = str(_a or "").strip().lower()
            if _QUERY_KEYWORD_MIN_LEN <= len(a) <= 6 and a in raw_l and a != raw_l:
                hits.append(a)
    dev = _developer_display_name()
    if _QUERY_KEYWORD_MIN_LEN <= len(dev) <= 6 and dev in raw and dev != raw:
        hits.append(dev)
    # jieba 名词/专名/外文 token（覆盖 dict 外实体）；"角色D" 会被 jieba 拆成 素+世 两个单字、
    # 由上面的别名层兜住，这里不会误抽
    try:
        import jieba.posseg as _pseg
        for _w in _pseg.cut(raw):
            _tok = str(_w.word or "").strip()
            _flag = str(_w.flag or "").strip()
            if (_QUERY_KEYWORD_MIN_LEN <= len(_tok) <= 6
                    and (_flag.startswith("n") or _flag == "eng")
                    and _tok not in _QUERY_KEYWORD_STOP
                    and _tok in raw and _tok != raw):
                hits.append(_tok)
    except Exception:
        pass
    # 名字过滤放在 jieba 之后：称呼语位置的 "角色A" 会被 jieba 当名词抽出来，
    # 只在上游过滤会漏。
    if _first_person_topic_query:
        _names = {str(a).strip().lower() for _als in _CHARACTER_NAME_ALIASES.values() for a in _als}
        if dev:
            _names.add(str(dev).strip().lower())
        hits = [h for h in hits if str(h).strip().lower() not in _names]
    seen: set[str] = set()
    out: list[str] = []
    for kw in sorted(hits, key=lambda x: (-len(x), x)):
        if kw not in seen:
            seen.add(kw)
            out.append(kw)
        if len(out) >= max_keywords:
            break
    return out


def _split_query_semantics(query: str, character: str = "") -> dict:
    """规则版 query 语义切分主入口 → {"clauses", "keywords", "variants"}。

    variants = clauses + keywords 合并去重（≤3），供 embedding 变体；
    clauses + keywords 供候选池 FTS/LIKE 扩展。SEMANTIC_QUERY_EXPAND=0 时全空。
    """
    if not _semantic_query_expand_enabled():
        return {"clauses": [], "keywords": [], "variants": []}
    clauses = _split_query_clauses(query)
    keywords = _extract_query_keywords(query, character)
    variants: list[str] = []
    for v in clauses + keywords:
        v = str(v or "").strip()
        if len(v) >= 2 and v != query and v not in variants:
            variants.append(v)
        if len(variants) >= 3:
            break
    return {"clauses": clauses, "keywords": keywords, "variants": variants}


def _llm_expand_query(query: str, character: str = "") -> list[str]:
    """LLM 版 query expansion：调用 LLM 生成 2 个语义等价变体。失败返空列表 fallback 到规则版。"""
    if not query or len(query) < 6:
        return []
    try:
        import os as _os
        api_key = (_os.getenv("LLM_API_KEY", "") or "").strip()
        base_url = (_os.getenv("LLM_BASE_URL", "") or "").strip()
        model = (_os.getenv("LLM_MODEL", "") or "").strip()
        if not (api_key and base_url and model):
            return []
        from tsuzuri.llm_client import shared_openai_client
        client = shared_openai_client(api_key=api_key, base_url=base_url)
        char_hint = f"角色：{character}。" if character else ""
        prompt = (
            f"{char_hint}请把下面的对话 query 改写成 2 个语义等价但用词不同的变体（保留同样的意图、不要加新事实）、"
            f"每行一个变体、不要解释、不要编号。\n\n"
            f"原 query：{query}"
        )
        text = ""
        for _attempt, _max_tokens in enumerate((200, 400), start=1):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_max_tokens,
                temperature=0.5,
                extra_body={"thinking": {"type": "disabled"}},
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                break
            if _attempt < 2:
                _tsz_emit(
                    f"[QueryExpand] empty response, retry "
                    f"(attempt {_attempt}/2, max_tokens={_max_tokens})",
                    flush=True,
                )
        variants: list[str] = []
        for line in text.splitlines():
            v = line.strip().lstrip("-•*0123456789. ").strip()
            if v and v != query and len(v) >= 4 and v not in variants:
                variants.append(v)
                if len(variants) >= 2:
                    break
        return variants
    except Exception:
        return []


def _expand_query_semantically(query: str, character: str = "", max_variants: int = 3) -> list[str]:
    """主入口：返回 [original_query, variant1, variant2, ...]、长度 1-3。
    env SEMANTIC_QUERY_EXPAND=0 关闭、返单元素。
    SEMANTIC_QUERY_EXPAND_LLM=1 启用 LLM 版（默认规则版）。
    [2026-08-04 规则版切分] 默认变体来源从『4 词条同义替换』升级为
    『子句拆分 + 关键词抽取』（_split_query_semantics），同义替换降级为兜底补充。
    """
    if not _semantic_query_expand_enabled():
        return [query]
    out = [query]
    use_llm = (os.getenv("SEMANTIC_QUERY_EXPAND_LLM", "0") or "0").strip().lower() in ("1", "true", "yes")
    if use_llm:
        variants = _llm_expand_query(query, character)
        if not variants:
            variants = _split_query_semantics(query, character)["variants"]
    else:
        variants = _split_query_semantics(query, character)["variants"]
        # 同义替换作兜底补充（子句/关键词不足时才占位）
        for v in _rule_expand_query(query, character):
            if v not in variants:
                variants.append(v)
    for v in variants[: max(0, max_variants - 1)]:
        if v not in out:
            out.append(v)
    return out


def _recency_score(ts: str, half_life_hours: float = 96.0) -> float:
    dt = _parse_dt(ts)
    if dt is None:
        return 0.25
    delta_h = max(0.0, (utc_now_naive() - dt).total_seconds() / 3600.0)
    hl = max(1.0, float(half_life_hours))
    return float(math.exp(-delta_h / hl))


def _memory_recency_timestamp(row: dict | None = None) -> str:
    """Return the timestamp that represents when a memory happened."""
    item = row if isinstance(row, dict) else {}
    payload = _json_loads(str(item.get("payload_json", "") or "{}"), {})
    repair = str(payload.get("repair", "") or "") if isinstance(payload, dict) else ""
    if repair.startswith("splice_cleanup"):
        return str(item.get("created_at", "") or item.get("updated_at", "") or "")
    return str(item.get("updated_at", "") or item.get("created_at", "") or "")


# ── ACT-R 幂律激活 ─────────────────────────────────────────────
# B_i = ln(Σ t_j^{-d}),  P(recall) = sigmoid((B_i - θ) / σ)
# d=0.5, θ=-1.5, σ=0.25  (标准 ACT-R 参数, 与 cognitive_sim.py 一致)
_ACT_R_DECAY = 0.5
_ACT_R_NOISE = 0.25
_ACT_R_THRESHOLD = -1.5


def _actr_recency_score(created_at: str, lifecycle_json_str: str, access_count: int) -> float:
    """ACT-R base-level activation → retrieval probability (0.0~1.0).

    Uses real access timestamps from lifecycle_json when available;
    falls back to evenly-spaced interpolation using access_count.
    """
    now_ts = time.time()

    # --- build access time list ---
    access_times_ts: list[float] = []

    # 1) created_at = first access
    dt0 = _parse_dt(created_at)
    if dt0 is not None:
        access_times_ts.append(dt0.replace(tzinfo=timezone.utc).timestamp() if dt0.tzinfo is None else dt0.timestamp())

    # 2) real access timestamps from lifecycle_json
    lifecycle = _json_loads(lifecycle_json_str if isinstance(lifecycle_json_str, str) else "{}", {})
    raw_times = lifecycle.get("access_times", []) if isinstance(lifecycle, dict) else []
    if isinstance(raw_times, list):
        for rt in raw_times[-20:]:
            rdt = _parse_dt(str(rt or ""))
            if rdt is not None:
                ts_val = rdt.replace(tzinfo=timezone.utc).timestamp() if rdt.tzinfo is None else rdt.timestamp()
                access_times_ts.append(ts_val)

    # 3) interpolation: 补足真实 timestamp 不足 access_count 的部分
    # 2026-05-09 fix: 旧逻辑只在 len <= 1 时 fallback、但实际 commit_memory_surface 只在
    # LLM 真 callback 时 append、所以 access_count >> len(access_times) 是常态、
    # 旧逻辑导致 access_times=1 比 access_times=0 召回分暴跌 80x（fallback 不触发）。
    # 新逻辑：保证 access_times_ts 长度 ≈ access_count + 1（含 created_at）、
    # 真实 timestamp 优先、不足部分用内插补、保持 ACT-R 信号连续。
    ac = max(0, int(access_count or 0))
    if ac > 0 and access_times_ts:
        target_len = ac + 1  # +1 for created_at
        deficit = target_len - len(access_times_ts)
        if deficit > 0:
            first = access_times_ts[0]
            # 内插范围：[first, last]（如有真 timestamp）or [first, now]（无真）
            last = access_times_ts[-1] if len(access_times_ts) > 1 else now_ts
            span = max(0.0, last - first)
            if span > 0:
                # 在 [first, last] 之间均匀补 deficit 个 fake timestamp
                interval = span / max(1, deficit + 1)
                # 收集已有 timestamps、避免重复
                existing = set(access_times_ts)
                added = 0
                for i in range(1, deficit + 1):
                    candidate = first + interval * i
                    if candidate not in existing:
                        access_times_ts.append(candidate)
                        added += 1
                        if added >= deficit:
                            break
            # span=0（只 created_at、无真 access_times）→ 内插到 now
            # 这是真实"长期未被记录但 access_count 累计"的 case
            elif len(access_times_ts) == 1:
                span2 = now_ts - first
                if span2 > 0:
                    interval = span2 / (ac + 1)
                    for i in range(1, min(ac, 20) + 1):
                        access_times_ts.append(first + interval * i)
        # cap 总长度 50（防 access_count 极大时算力）
        if len(access_times_ts) > 50:
            access_times_ts = access_times_ts[:50]

    if not access_times_ts:
        return 0.25  # no temporal data at all

    # --- ACT-R base-level activation ---
    total = 0.0
    for at in access_times_ts:
        delta_h = max(0.001, (now_ts - at) / 3600.0)
        total += delta_h ** (-_ACT_R_DECAY)

    if total <= 0:
        activation = _ACT_R_THRESHOLD
    else:
        activation = math.log(total)

    # --- retrieval probability (sigmoid) ---
    x = (activation - _ACT_R_THRESHOLD) / _ACT_R_NOISE
    x = max(-10.0, min(10.0, x))
    return round(1.0 / (1.0 + math.exp(-x)), 4)


# ── 共现检测 + 扩散激活 + 侧抑制 + 动机层 ──────────────────────

# Hebbian co-occurrence strength cache + sqlite persistence (2026-05-09)
# 旧版仅 in-memory、重启清空、长期 Hebbian 学习失效。
# 新版：cache 作快读、write-through 到 sqlite co_occurrence_strength 表。
_CO_STRENGTH_CACHE: dict[tuple[str, str], float] = {}
_CO_STRENGTH_LOADED = False  # lazy load flag、首次访问时从 sqlite 加载
_CO_STRENGTH_LR = 0.15
_CO_STRENGTH_DEFAULT = 0.10
_CO_STRENGTH_MAX_ENTRIES = 2000


def _co_strength_key(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _get_co_store():
    """获取当前 MemoryCoreStore 单例（用于 hebbian sqlite 读写）。"""
    from tsuzuri.memory_core_impl import store as _m_store
    try:
        return _m_store._MEMORY_CORE_SINGLETON
    except Exception:
        return None


def _co_strength_load_from_sqlite() -> None:
    """启动时从 sqlite 加载 top-N co_occurrence rows 到 cache。"""
    global _CO_STRENGTH_LOADED
    if _CO_STRENGTH_LOADED:
        return
    store = _get_co_store()
    if store is None:
        return
    try:
        with store._lock:
            conn = store._connect()
            try:
                rows = conn.execute(
                    """SELECT item_a, item_b, strength FROM co_occurrence_strength
                       ORDER BY last_updated DESC LIMIT ?;""",
                    (_CO_STRENGTH_MAX_ENTRIES,),
                ).fetchall()
                for r in rows:
                    a, b = str(r[0] or ""), str(r[1] or "")
                    if a and b:
                        _CO_STRENGTH_CACHE[(a, b)] = float(r[2] or _CO_STRENGTH_DEFAULT)
            finally:
                conn.close()
        _CO_STRENGTH_LOADED = True
        _tsz_emit(
            f"[Hebbian] loaded {len(_CO_STRENGTH_CACHE)} co_strength entries from sqlite",
            flush=True,
        )
    except Exception as _hl_err:
        _tsz_emit(f"[Hebbian] sqlite load failed: {_hl_err}", flush=True)


def _co_strength_persist(a: str, b: str, strength: float) -> None:
    """write-through 到 sqlite。失败时不阻塞、只 log。"""
    store = _get_co_store()
    if store is None:
        return
    try:
        now_iso = _utc_now_iso()
        with store._lock:
            conn = store._connect()
            try:
                conn.execute(
                    """INSERT INTO co_occurrence_strength(item_a, item_b, strength, last_updated, update_count)
                       VALUES (?, ?, ?, ?, 1)
                       ON CONFLICT(item_a, item_b) DO UPDATE SET
                         strength=excluded.strength,
                         last_updated=excluded.last_updated,
                         update_count=update_count+1;""",
                    (a, b, strength, now_iso),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass  # 静默降级、不影响主流程


def _get_co_strength(a: str, b: str) -> float:
    if not _CO_STRENGTH_LOADED:
        _co_strength_load_from_sqlite()
    return _CO_STRENGTH_CACHE.get(_co_strength_key(a, b), _CO_STRENGTH_DEFAULT)


def _hebbian_update(a: str, b: str) -> None:
    if not a or not b or a == b:
        return
    if not _CO_STRENGTH_LOADED:
        _co_strength_load_from_sqlite()
    key = _co_strength_key(a, b)
    old = _CO_STRENGTH_CACHE.get(key, _CO_STRENGTH_DEFAULT)
    new = min(0.5, old + _CO_STRENGTH_LR * (1.0 - old))
    _CO_STRENGTH_CACHE[key] = new
    # write-through 到 sqlite（失败静默）
    _co_strength_persist(key[0], key[1], new)
    # LRU eviction
    if len(_CO_STRENGTH_CACHE) > _CO_STRENGTH_MAX_ENTRIES:
        # Remove ~10% oldest entries (insertion-order in Python 3.7+)
        to_remove = list(_CO_STRENGTH_CACHE.keys())[: _CO_STRENGTH_MAX_ENTRIES // 10]
        for k in to_remove:
            _CO_STRENGTH_CACHE.pop(k, None)


def _find_co_occurring(top_items: list[dict], all_candidates: list[dict]) -> dict[str, set[str]]:
    """Find co-occurring candidates for each top item.

    Co-occurrence heuristics:
      1) Entity overlap: cue_json.entities share 2+ entities
      2) Temporal proximity: created_at within same day + 2-hour window (same session)
    """
    result: dict[str, set[str]] = {}

    # Pre-parse top items
    top_data = []
    for t in top_items:
        tid = str(t.get("item_id", "") or "")
        if not tid:
            continue
        t_cue = _json_loads(str(t.get("cue_json", "") or "{}"), {})
        t_entities = set()
        if isinstance(t_cue, dict):
            raw = t_cue.get("entities", [])
            if isinstance(raw, list):
                t_entities = {str(e).strip().lower() for e in raw if str(e).strip()}
        t_dt = _parse_dt(str(t.get("created_at", "") or ""))
        top_data.append((tid, t_entities, t_dt))
        result[tid] = set()

    if not top_data:
        return result

    # 2026-05-09 fix: 过滤掉过于 generic 的 entity（user/character names、否则
    # 几乎所有 episodic 都互相"共现"、扩散激活变成全员 boost、失去差异化）
    _GENERIC_COOC_ENTITIES = {
        "user", user_name(), "开发者",
        *agent_names(), *(v for v, c in text_variants().items() if is_agent(c) and re.fullmatch(r"[\u4e00-\u9fff]+", v)),
        # 通用代词
        "你", "我", "他", "她", "它", "我们", "你们", "她们", "他们",
    }

    def _filter_generic(ent_set: set[str]) -> set[str]:
        return {e for e in ent_set if e not in _GENERIC_COOC_ENTITIES}

    for c in all_candidates:
        cid = str(c.get("item_id", "") or "")
        if not cid:
            continue
        # Skip if candidate is one of the top items
        if any(cid == td[0] for td in top_data):
            continue

        c_cue = _json_loads(str(c.get("cue_json", "") or "{}"), {})
        c_entities = set()
        if isinstance(c_cue, dict):
            raw = c_cue.get("entities", [])
            if isinstance(raw, list):
                c_entities = {str(e).strip().lower() for e in raw if str(e).strip()}
        c_entities_specific = _filter_generic(c_entities)
        c_dt = _parse_dt(str(c.get("created_at", "") or ""))

        for tid, t_entities, t_dt in top_data:
            co = False
            # Check 1: 过滤后 entity overlap >= 1（specific entity 命中即足）
            # 旧逻辑要求 >= 2、但平均 entity 数 1.69、几乎不命中；
            # 新逻辑过滤 generic（user/角色名）后、specific 重合 ≥ 1 才算真共现
            t_entities_specific = _filter_generic(t_entities)
            if (t_entities_specific and c_entities_specific
                    and len(t_entities_specific & c_entities_specific) >= 1):
                co = True
            # Check 2: temporal proximity (same day, within 2h)
            if not co and t_dt is not None and c_dt is not None:
                delta = abs((t_dt - c_dt).total_seconds())
                if delta <= 7200 and t_dt.date() == c_dt.date():
                    co = True
            if co:
                result[tid].add(cid)

    return result


def _apply_spreading_activation(candidates: list[dict], boost_factor: float = 0.10) -> None:
    """Spreading activation: boost candidates co-occurring with top-3.

    Mutates candidate scores in-place. Sets _spread_activated=True on boosted items.
    """
    if len(candidates) < 3:
        return

    sorted_c = sorted(candidates, key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    top3 = sorted_c[:3]
    co_map = _find_co_occurring(top3, candidates)

    # Build reverse lookup: candidate_id → set of top_ids it co-occurs with
    activated: dict[str, set[str]] = {}  # cid → {top_ids}
    for tid, co_set in co_map.items():
        for cid in co_set:
            activated.setdefault(cid, set()).add(tid)

    if not activated:
        return

    for c in candidates:
        cid = str(c.get("item_id", "") or "")
        if cid in activated:
            top_ids = activated[cid]
            boost = max((_get_co_strength(cid, tid) for tid in top_ids), default=boost_factor)
            c["score"] = float(c.get("score", 0.0) or 0.0) + boost
            c["_spread_activated"] = True
            # Hebbian learning: strengthen co-retrieval links
            for tid in top_ids:
                _hebbian_update(cid, tid)


_LATERAL_INHIBIT_SIM_THRESHOLD = 0.55


def _bigram_jaccard(a: str, b: str) -> float:
    """Pure character-bigram Jaccard for lateral inhibition.
    Doesn't depend on jieba, so it's robust across deployments.
    """
    aa = _bigrams(a)
    bb = _bigrams(b)
    if not aa or not bb:
        return 0.0
    union = len(aa | bb)
    if union == 0:
        return 0.0
    return float(len(aa & bb)) / float(union)


def _apply_lateral_inhibition(candidates: list[dict], inhibit_factor: float = 0.06) -> None:
    """Lateral inhibition: suppress near-duplicate candidates NOT co-activated.

    Mutates candidate scores in-place. Must be called AFTER spreading activation.
    Uses bigram Jaccard (jieba-free) to stay robust to environment.
    """
    if len(candidates) < 3:
        return

    sorted_c = sorted(candidates, key=lambda x: float(x.get("score", 0) or 0), reverse=True)
    top_texts = []
    top_ids = set()
    for c in sorted_c[:3]:
        t = str(c.get("text", "") or "").strip()
        if t:
            top_texts.append(t)
            top_ids.add(str(c.get("item_id", "") or ""))

    if not top_texts:
        return

    for c in candidates:
        cid = str(c.get("item_id", "") or "")
        if cid in top_ids:
            continue
        c_text = str(c.get("text", "") or "").strip()
        if not c_text:
            continue

        max_sim = max((_bigram_jaccard(c_text, tt) for tt in top_texts), default=0.0)
        if max_sim > _LATERAL_INHIBIT_SIM_THRESHOLD and not c.get("_spread_activated"):
            c["score"] = max(0.0, float(c.get("score", 0.0) or 0.0) - inhibit_factor * max_sim)


# Phase G.S2 (2026-05-02): goal_text 词颗粒化 helper（替代 char 迭代 bug）
# 之前 `{c.lower() for c in goal if len(c.strip()) > 1}` 把字符串当 char 序列迭代、
# 每个 c 是单字符、`c.strip()` 长度永远 ≤ 1、过滤永远 False → 集合恒空、
# entity_overlap 永远 0、动机层退化为只剩 lex_overlap。
# 新版优先 jieba 词级分词、fallback 到 char-bigram。
def _tokenize_goal_words(goal: str, min_len: int = 2) -> set[str]:
    raw = str(goal or "").strip()
    if not raw:
        return set()
    norm = _norm_text(raw)
    # 优先用 jieba（与 _word_jaccard 同款依赖）
    try:
        if "_jieba_mod" in globals():
            return {tok.strip().lower() for tok in _jieba_mod.cut(norm)
                    if len(tok.strip()) >= min_len}
    except Exception:
        pass
    # Fallback: char-bigram 集合（连续 2 字符窗口）
    if len(norm) < min_len:
        return set()
    return {norm[i:i + 2] for i in range(len(norm) - 1)}


def _apply_affect_memory_boost(score: float, emotional_weight: float, affect_state: dict | None) -> float:
    """情绪-记忆双向耦合：根据当前角色情绪态对记忆分数做乘法调整。

    情绪塑造记忆：负面情绪态下高情感权重记忆更容易浮现（仿海马体-杏仁核耦合）；
    高唤起态下任何情感强度较高的记忆都会被激活。

    纯函数，可独立单测。
    score:            base 分（已含所有其他维度）
    emotional_weight: 记忆在入库时锚定的情绪权重 0-1
    affect_state:     来自 cognitive_state_v2 的情绪状态 dict，含 valence/arousal
    返回: 调整后的 score（不超过 1.8）
    """
    if not affect_state or not isinstance(affect_state, dict):
        return score
    valence = float(affect_state.get("valence", 0.0) or 0.0)
    arousal = float(affect_state.get("arousal", 0.0) or 0.0)
    ew = max(0.0, min(1.0, float(emotional_weight)))

    if valence < -0.3:  # 负面情绪态
        if ew >= 0.6:
            score = min(1.8, score * 1.40)   # 高情感记忆在悲伤/焦虑时更清晰
        elif ew <= 0.2:
            score = min(1.8, score * 0.75)   # 轻松愉快记忆在负面情绪时反而难想起
    elif arousal > 0.7:  # 高唤起态（激动/兴奋/紧张）
        if ew >= 0.5:
            score = min(1.8, score * 1.25)
    elif valence > 0.5 and arousal > 0.4:  # 正面高唤起
        if ew >= 0.7:
            score = min(1.8, score * 1.15)

    return score


def _extract_payload_emotional_weight(payload: dict | None) -> float:
    """从 memory_policy.memory_features.emotional_weight 安全提取情感权重。"""
    if not isinstance(payload, dict):
        return 0.0
    policy = payload.get("memory_policy", {})
    if not isinstance(policy, dict):
        return 0.0
    features = policy.get("memory_features", {})
    if not isinstance(features, dict):
        return 0.0
    return max(0.0, min(1.0, float(features.get("emotional_weight", 0.0) or 0.0)))


def _compute_motivation_boost(text: str, cue_json_str: str, motivations: list[dict]) -> float:
    """Compute motivation boost: how well a memory aligns with active character goals.

    Returns 0.0~0.15 boost score.
    """
    if not motivations or not text:
        return 0.0

    cue = _json_loads(cue_json_str if isinstance(cue_json_str, str) else "{}", {})
    cue_entities = set()
    if isinstance(cue, dict):
        raw = cue.get("entities", [])
        if isinstance(raw, list):
            cue_entities = {str(e).strip().lower() for e in raw if str(e).strip()}

    max_score = 0.0
    for mot in motivations:
        if not isinstance(mot, dict):
            continue
        goal = str(mot.get("goal_text", "") or "").strip()
        if not goal:
            continue
        eff_pri = _safe_float(mot.get("effective_priority", 0.5), 0.5)

        # Lexical overlap between goal and memory text
        lex_overlap = _lexical_similarity(goal, text)

        # Phase G.S2 (2026-05-02): goal_text 走 jieba/bigram 分词、不再 char 迭代
        # 修复 entity_overlap 死代码 bug（详见审计 S2）
        goal_words = _tokenize_goal_words(goal)
        entity_overlap = 0.0
        if cue_entities and goal_words:
            shared = sum(1 for e in cue_entities if any(gw in e for gw in goal_words))
            entity_overlap = min(1.0, shared / max(1, len(goal_words)))

        combined = max(lex_overlap, entity_overlap)
        score = combined * eff_pri
        max_score = max(max_score, score)

    return min(0.15, max_score)


def _source_prior(layer: str) -> float:
    return {
        "working": 0.72,
        "episodic": 0.68,
        "semantic": 0.74,
        "procedural": 0.63,
    }.get(str(layer or "").strip().lower(), 0.55)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _json_dumps(v) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except Exception as _dump_err:
        # 2026-07-31: 不再静默吞错。此前 meta 混入 callable（fact_rewrite_llm_caller）
        # 导致整条 payload 落成 "{}"、memory_features/typed_score/turn_fact_card 全丢、
        # 晋升链断流 2.5 个月无察觉。序列化失败必须在日志留痕。
        try:
            _tsz_emit(
                f"[MemoryCore] _json_dumps failed ({type(_dump_err).__name__}: {_dump_err}); "
                f"value_type={type(v).__name__} — returning '{{}}'",
                flush=True,
            )
        except Exception:
            pass
        return "{}"


def _json_safe_meta(meta, *, _depth: int = 0):
    """Return a JSON-serializable copy of turn meta for persisting inside payload_json.

    chat_server/宿主应用 pass runtime-only objects in meta (e.g. fact_rewrite_llm_caller,
    an LLM callable).  Embedding raw meta made ``json.dumps`` raise and ``_json_dumps``
    silently persist "{}" — wiping memory_features / typed_score / turn_fact_card /
    memory_meta from every affected row (2026-05-09 ~ 2026-07-31).  This helper keeps
    plain JSON values (dicts/lists/scalars) and drops anything else (callables,
    clients, open handles).  Dict keys are stringified; unserializable values are
    omitted; tuples become lists; nesting is depth-capped.
    """
    if _depth > 6:
        return None
    if meta is None or isinstance(meta, (str, int, float, bool)):
        return meta
    if isinstance(meta, dict):
        out: dict = {}
        for k, v in meta.items():
            key = str(k or "").strip()
            if not key:
                continue
            safe = _json_safe_meta(v, _depth=_depth + 1)
            if safe is not None or v is None:
                out[key] = safe
        return out
    if isinstance(meta, (list, tuple)):
        out_list: list = []
        for item in meta:
            safe = _json_safe_meta(item, _depth=_depth + 1)
            if safe is not None or item is None:
                out_list.append(safe)
        return out_list
    return None


def _json_loads(s: str, default):
    try:
        out = json.loads(str(s or ""))
        if out is None:
            return default
        return out
    except Exception:
        return default


def _payload_marks_non_retrievable_present_state(payload: dict | None = None) -> bool:
    row = payload if isinstance(payload, dict) else {}
    if bool(row.get("non_retrievable_present_state", False)):
        return True
    flags = row.get("reply_fact_flags", {}) if isinstance(row.get("reply_fact_flags", {}), dict) else {}
    if bool(flags.get("contains_current_self_state", False)):
        return True
    meta = row.get("meta", {}) if isinstance(row.get("meta", {}), dict) else {}
    if bool(meta.get("non_retrievable_present_state", False)):
        return True
    meta_flags = meta.get("reply_fact_flags", {}) if isinstance(meta.get("reply_fact_flags", {}), dict) else {}
    return bool(meta_flags.get("contains_current_self_state", False))
