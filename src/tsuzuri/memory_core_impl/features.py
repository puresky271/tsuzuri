"""Memory feature extraction, typed scoring, entities/topics, consolidation gates and fact rewriting.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import os
import re
from tsuzuri.persona_registry import agent_names, live_pattern, live_table, persona_regex, setting, user_name, user_pronoun
from typing import Callable
from tsuzuri.memory_core_impl.config import _emb_cos, _emb_encode, _embedding_callable
from tsuzuri.memory_core_impl.memory_rows import (
    _ASSISTANT_ONLY_FACT_RE, _COLD_START_IDENTITY_RE, _COLLECTIVE_ADDRESS_RE,
    _DISCLOSURE_USER_RE, _EMOTION_CONNECTION_OR_PREFERENCE_ALONE_RE, _EMOTIONAL_REACTIVE_RE,
    _EMOTIONAL_VALUE_BOUND_RE, _EPHEMERAL_COMMAND_RE, _EPHEMERAL_GREETING_RE,
    _EPHEMERAL_TRANSIENT_RE, _EXPLICIT_MEMORY_INTENT_RE, _EXPLICIT_MEMORY_WRITE_RE,
    _FOCUS_GENERIC_CHUNKS, _GENERIC_RE, _norm_text, _RARE_UPPER_TOKEN_RE,
    _REASSURANCE_ASSISTANT_RE, _SPECIFICITY_PLACE_RE, _SPECIFICITY_TIME_RE, _STANCE_RE,
    _TOPIC_ONLY_FACT_RE, _USER_FACT_NEGATIVE_RE, _USER_FACT_PATTERN_RE,
)

def _compute_digit_density(text: str) -> float:
    """数字密度：日期/价格/年龄/ID/版本号都自带 specific 信号。
    返 0.0-1.0；权重 0.02、单独触发不了 candidate 阈值。"""
    if not isinstance(text, str) or not text:
        return 0.0
    digit = sum(1 for c in text if c.isdigit())
    if digit == 0:
        return 0.0
    # 5 倍放大、最长 10 字符 mention 即可饱和
    return min(1.0, digit / max(len(text), 10) * 5.0)


def _compute_rare_char_ratio(text: str) -> float:
    """罕见字符 / 大写英文 token 密度：
    - 片假名（多用于外来语、专名）
    - CJK 扩展 A 区罕用汉字
    - 长度 ≥ 2 大写英文 token（专名 / 缩写、不重叠 SPECIFICITY 文字）
    返 0.0-1.0；权重 0.02。"""
    if not isinstance(text, str) or not text:
        return 0.0
    rare = 0
    for c in text:
        # 片假名块: U+30A0-U+30FF
        if "゠" <= c <= "ヿ":
            rare += 1
        # CJK Unified Ideographs Extension A: U+3400-U+4DBF
        elif "㐀" <= c <= "䶿":
            rare += 1
    rare += sum(len(m.group(0)) for m in _RARE_UPPER_TOKEN_RE.finditer(text))
    if rare == 0:
        return 0.0
    return min(1.0, rare / max(len(text), 10) * 4.0)


def _compute_embedding_novelty(
    text: str,
    *,
    character: str = "",
    store=None,
    recent_limit: int = 30,
) -> float:
    """与该角色最近 N working turn 平均 cosine 的反向（1 - avg_sim）。
    - embedding 不可用 → 返 0.5（neutral，不阻断也不偏袒）
    - 短句（< 6 char）→ 返 0.5（embedding 区分度太低）
    - 冷启动（无 recent）→ 返 0.7（中高、给冷启动一点奖励但不过头）
    异常 → 0.5。
    权重 0.06、是 specificity 退役后语义稀有度的主代偿。"""
    if not isinstance(text, str) or len(text.strip()) < 6:
        return 0.5
    if not _embedding_callable():
        return 0.5
    if store is None or not character:
        return 0.5
    try:
        emb_q = _emb_encode(text)
        if emb_q is None:
            return 0.5
        recent = store.get_recent_working_items(
            character, max(1, int(recent_limit or 30)), True
        )
        if not recent:
            return 0.7  # 冷启动小奖励
        sims: list[float] = []
        for r in recent[: int(recent_limit or 30)]:
            r_text = str(r.get("text") or "")
            if not r_text or len(r_text) < 4:
                continue
            try:
                emb_r = _emb_encode(r_text)
                if emb_r is None:
                    continue
                s = _emb_cos(emb_q, emb_r)
                if isinstance(s, (int, float)):
                    sims.append(float(s))
            except Exception:
                continue
        if not sims:
            return 0.7
        avg = sum(sims) / len(sims)
        # 与最近相似 → 重复 → 低 novelty；与最近不相似 → novel → 高
        return max(0.0, min(1.0, 1.0 - avg))
    except Exception:
        return 0.5


def _extract_memory_features(
    user_text: str,
    assistant_reply: str = "",
    meta: dict | None = None,
    *,
    character: str = "",
    store=None,
) -> dict:
    """Compute Layer 1 cheap content features for a turn (Phase Q.W2 / Option E).

    All features are 0.0-1.0 floats:
      - explicit_memory_intent: 用户显式记忆意图 (binary)
      - cold_start_fast_track: 强信号身份/偏好 (binary、bypasses reuse gate)
      - specificity: 表面具体度（TIME + PLACE 形态 — 缩减后版本）
      - embedding_novelty: 与最近 working turn 平均相似度的反向（语义稀有度）
      - digit_density: 数字密度
      - rare_char_ratio: 片假名 / CJK Ext / 大写英文 token 密度
      - user_fact_density: 关于用户的稳定属性（semantic）
      - stance_or_value: 态度立场陈述
      - emotional_weight: 价值绑定情绪（NOT reactive）
      - ephemeral_penalty: 强档 archive 信号（binary、命中即直接 stale）
      - generic_penalty: 弱档 dampener

    新增 character / store 关键字参数（embedding_novelty 需要、可省略 → fallback 0.5）。
    """
    user = str(user_text or "").strip()
    asst = str(assistant_reply or "").strip()
    combined = f"{user} {asst}".strip()
    user_msg_only = user

    feats = {
        "explicit_memory_intent": 0.0,
        "cold_start_fast_track": 0.0,
        "specificity": 0.0,
        "embedding_novelty": 0.5,
        "digit_density": 0.0,
        "rare_char_ratio": 0.0,
        "user_fact_density": 0.0,
        "stance_or_value": 0.0,
        "emotional_weight": 0.0,
        "disclosure_acknowledgment": 0.0,  # 2026-05-11 新加：disclosure→reassurance 双向 pattern
        "ephemeral_penalty": 0.0,
        "generic_penalty": 0.0,
    }

    if not user:
        return feats

    # 1. explicit_memory_intent (binary)
    if _EXPLICIT_MEMORY_INTENT_RE.search(user_msg_only):
        feats["explicit_memory_intent"] = 1.0

    # 2. cold_start_fast_track (binary、very strong signal)
    if _COLD_START_IDENTITY_RE.search(user_msg_only):
        feats["cold_start_fast_track"] = 1.0

    # 3. specificity (only TIME + PLACE 形态后缀 + 极少锚点；PROPER 已退役)
    spec_count = (
        len(_SPECIFICITY_TIME_RE.findall(combined))
        + len(_SPECIFICITY_PLACE_RE.findall(combined))
    )
    # density: 1 entity per 30 chars = saturated to 1.0
    if spec_count > 0:
        feats["specificity"] = min(1.0, spec_count * 30.0 / max(len(combined), 30))

    # 3b. embedding_novelty (语义稀有度、PROPER 主代偿)
    feats["embedding_novelty"] = _compute_embedding_novelty(
        combined, character=character, store=store
    )

    # 3c. digit_density (日期/价格/ID 等)
    feats["digit_density"] = _compute_digit_density(combined)

    # 3d. rare_char_ratio (片假名/CJK Ext/大写英文 token)
    feats["rare_char_ratio"] = _compute_rare_char_ratio(combined)

    # 4. user_fact_density (about user、stable)
    if _USER_FACT_NEGATIVE_RE.match(user_msg_only):
        # Starts with assistant/third-person — not user_fact
        feats["user_fact_density"] = 0.0
    elif _USER_FACT_PATTERN_RE.search(user_msg_only):
        # Has user-attribute pattern — boost by length
        feats["user_fact_density"] = min(1.0, 0.5 + (len(user_msg_only) / 200.0))

    # 5. stance_or_value
    if _STANCE_RE.search(user_msg_only):
        feats["stance_or_value"] = min(1.0, 0.5 + (len(user_msg_only) / 200.0))

    # 6. emotional_weight (value-bound、NOT transient reactive)
    # 2026-05-11 拓宽：覆盖 vulnerability/connection/seeking/preference 4 大类、disclosure 标记词单独可命中
    if _EMOTIONAL_REACTIVE_RE.match(user_msg_only):
        # Pure reactive — does not count as long-term emotion
        feats["emotional_weight"] = 0.0
    elif _EMOTIONAL_VALUE_BOUND_RE.search(user_msg_only):
        feats["emotional_weight"] = min(1.0, 0.6 + (len(user_msg_only) / 200.0))
    elif _EMOTION_CONNECTION_OR_PREFERENCE_ALONE_RE.search(user_msg_only):
        # 关系性 / 偏好情感词单独命中（"想你"/"喜欢你"/"谢谢"）
        # 这类情感本身就是 value-bound、不需要时间锚也算
        feats["emotional_weight"] = min(0.7, 0.4 + (len(user_msg_only) / 250.0))

    # 6b. disclosure_acknowledgment (2026-05-11 新加)
    # 双向检测：user 自我披露 / 脆弱情绪 + assistant 接住 / 安抚 → canon-key 情感事件
    # 这类 turn 即使其他 feature 都低、也应高 score 通过 promotion gate（不进 stale_archive）
    if asst:  # 必须有 assistant reply 才能判断接住
        _user_discloses = bool(_DISCLOSURE_USER_RE.search(user_msg_only))
        _asst_reassures = bool(_REASSURANCE_ASSISTANT_RE.search(asst))
        if _user_discloses and _asst_reassures:
            # 双向命中、强信号、按对话长度加权
            feats["disclosure_acknowledgment"] = min(1.0, 0.7 + (len(combined) / 300.0))
        elif _user_discloses or _asst_reassures:
            # 单向命中、中等信号（可能 user 在披露但 assistant 还没明确接住、或反过来）
            feats["disclosure_acknowledgment"] = min(0.5, 0.3 + (len(combined) / 400.0))

    # 7. ephemeral_penalty (strong kill — direct stale_archive)
    if (_EPHEMERAL_GREETING_RE.match(user_msg_only)
            or _EPHEMERAL_COMMAND_RE.search(user_msg_only)
            or _EPHEMERAL_TRANSIENT_RE.match(user_msg_only)):
        feats["ephemeral_penalty"] = 1.0

    # 8. generic_penalty (weak)
    if _GENERIC_RE.match(user_msg_only):
        feats["generic_penalty"] = 0.5

    return feats


def _compute_typed_score(features: dict) -> float:
    """Aggregate Layer 1 features into typed_score (0-1) for Layer 2 state machine.

    Weights per 2026-05-09 SPECIFICITY 去 hardcode 重构：
      Positive (base sum 1.00):
        explicit_memory_intent:  0.25  (用户显式 = 强意图)
        user_fact_density:       0.25  (关于用户稳定属性)
        stance_or_value:         0.15  (态度立场)
        emotional_weight:        0.15  (价值绑定情绪)
        specificity:             0.10  (TIME + PLACE 形态后缀；2026-05-09 从 0.20 缩半)
        embedding_novelty:       0.06  (语义稀有度；新；PROPER 主代偿)
        digit_density:           0.02  (新)
        rare_char_ratio:         0.02  (新)
      Combo bonus (audit calibration 2026-05-04 — 多 feature 互相印证)：
        2 个 positive feature 各 >= 0.3 → +0.10
        3+ 个 positive feature 各 >= 0.3 → +0.20
      Penalty:
        ephemeral_penalty:      -0.50  (强档 kill)
        generic_penalty:        -0.30  (弱档 dampener)

    embedding_novelty 默认 0.5（neutral fallback），减去 0.5 后再计权（>0.5 加分、<0.5 减分），
    避免 embedding 关闭时给所有 turn 平白多 0.03。

    Returns 0.0-1.0 clamped。Threshold guidelines (audit calibrated):
      score >= 0.85: cold-start / 强信号 fast-track 候选
      score >= 0.70: 标准 promotion 候选 + reuse gate 后晋升
      score >= 0.30: 进入 candidate 状态、等 reuse 信号
      score <  0.30: 直接 stale_archive
    """
    if not isinstance(features, dict):
        return 0.0

    f = features
    # embedding_novelty 折成 (-0.5, +0.5)、再 × 0.06；neutral 不偏袒
    novelty_centered = max(-0.5, min(0.5, float(f.get("embedding_novelty", 0.5)) - 0.5))

    base = (
        0.25 * float(f.get("explicit_memory_intent", 0.0))
        + 0.25 * float(f.get("user_fact_density", 0.0))
        + 0.15 * float(f.get("stance_or_value", 0.0))
        + 0.15 * float(f.get("emotional_weight", 0.0))
        + 0.10 * float(f.get("specificity", 0.0))
        + 0.06 * (novelty_centered + 0.5)  # 还原 0.0-1.0 维度后乘 0.06
        + 0.02 * float(f.get("digit_density", 0.0))
        + 0.02 * float(f.get("rare_char_ratio", 0.0))
        # 2026-05-11 新加：disclosure_acknowledgment（独立维度、不挤占其他 weight 上限）
        # 权重 0.30——单独命中 0.7+ 即可把 score 推到 0.21+、配合 emotional_weight 必过 0.45 门槛
        + 0.30 * float(f.get("disclosure_acknowledgment", 0.0))
    )

    # combo bonus 仍只看主信号、不把统计 feature 计入（避免数字密度刷分）
    # 2026-05-11：disclosure_acknowledgment 也加入 combo signal pool
    positive_count = sum(
        1 for k in ("explicit_memory_intent", "user_fact_density",
                    "specificity", "stance_or_value", "emotional_weight",
                    "disclosure_acknowledgment")
        if float(f.get(k, 0.0)) >= 0.3
    )
    combo_bonus = 0.0
    if positive_count >= 3:
        combo_bonus = 0.20
    elif positive_count >= 2:
        combo_bonus = 0.10

    penalty = (
        0.50 * float(f.get("ephemeral_penalty", 0.0))
        + 0.30 * float(f.get("generic_penalty", 0.0))
    )

    score = base + combo_bonus - penalty

    # 2026-05-11 加：情感类 turn 的 score floor（用户拍板「只要是情感类的都应该这样」）
    # 单纯情感表达（无其它 feature 配合）也应被保护、避免 score-killed → stale_archive
    # Floor 触发条件（任一）：
    #   · emotional_weight >= 0.60（中等以上情感强度）
    #   · disclosure_acknowledgment >= 0.50（明确披露 / 接住事件）
    #   · 二者之和 >= 0.70（轻微情感 + 轻微披露互补）
    # 但 ephemeral_penalty 仍能 override（保留 "好烦" / "晚安" 这类纯短反应不被强升）
    _emo = float(f.get("emotional_weight", 0.0))
    _discl = float(f.get("disclosure_acknowledgment", 0.0))
    _eph = float(f.get("ephemeral_penalty", 0.0))
    if _eph < 0.5 and (_emo >= 0.60 or _discl >= 0.50 or (_emo + _discl) >= 0.70):
        # 情感 floor = 0.50（足够过 CONSOLIDATION_SCORE_CANDIDATE_MIN=0.30 且贴近 promote 门槛）
        score = max(score, 0.50)

    return max(0.0, min(1.0, score))


def _is_cold_start_fast_track(features: dict) -> bool:
    """Cold-start fast-track gate (bypasses reuse signal requirement)."""
    if not isinstance(features, dict):
        return False
    return float(features.get("cold_start_fast_track", 0.0)) >= 1.0


# ═════════════════════════════════════════════════════════════════════════
# Phase Q.W2 / Option E Layer 3: Entity extraction + reuse detection helpers
# ═════════════════════════════════════════════════════════════════════════
# 用于 chat_server 在 N+1 turn 检测是否复用了 working_N 的 entities

# 2026-05-09 重构：PROPER 已退役、entity 提取只复用 PLACE + TIME
# Layer 3 entity overlap = Layer 1 specificity 命中（PLACE 形态 + TIME） + family / food 专属
_ENTITY_EXTRACT_RES = [
    # 复用 Layer 1 SPECIFICITY 两个 RE（PROPER 2026-05-09 已删）
    _SPECIFICITY_PLACE_RE,
    _SPECIFICITY_TIME_RE,
    # === Layer 3 专属：family relationships（含外祖父母/家族泛词）===
    re.compile(
        r"我妈妈?|我爸爸?|我哥哥?|我姐姐?|我弟弟?|我妹妹?|我女儿|我儿子|"
        r"我老板|我同学|我老师|我朋友|我同事|我室友|我邻居|"
        r"外婆|外公|奶奶|爷爷|阿姨|姨妈|姨夫|舅舅|舅妈|叔叔|婶婶|表哥|表姐|表弟|表妹|"
        r"老婆|老公|男朋友|女朋友|前男友|前女友"
    ),
    # === 食物 specifics（宿主应用 / 日常 / 中餐 / 西餐）===
    re.compile(
        # 日式
        r"拉面|拉麵|寿司|壽司|抹茶|和菓子|关东煮|關東煮|寿喜烧|涮涮锅|"
        r"可丽饼|大阪烧|章鱼烧|烤鸡串|天妇罗|親子丼|牛丼|海鮮丼|"
        r"パフェ|タピオカ|オムライス|ハヤシライス|どら焼き|たこ焼き|お好み焼き|"
        # 中式
        r"火锅|麻辣烫|串串|烤肉|烧烤|烤串|"
        r"饺子|包子|馒头|烧麦|小笼包|烤鸭|"
        r"米饭|炒饭|盖饭|拌饭|炒粉|炒面|盖面|"
        r"川菜|粤菜|湘菜|鲁菜|苏菜|浙菜|徽菜|闽菜|"
        r"螺蛳粉|肠粉|早茶|点心|"
        # 西式
        r"披萨|意面|意大利面|汉堡|薯条|炸鸡|"
        r"沙拉|三明治|墨西哥卷|"
        r"咖啡|拿铁|美式|卡布奇诺|玛奇朵|摩卡|手冲|冷萃|"
        r"奶茶|抹茶拿铁|奶盖|柠檬茶|果茶|气泡水|苏打|"
        # 甜点
        r"蛋糕|布丁|冰淇淋|甜筒|马卡龙|提拉米苏|"
        r"巧克力|曲奇|饼干|甜甜圈|"
        # 水果
        r"草莓|蓝莓|樱桃|苹果|香蕉|橘子|橙子|柚子|葡萄|西瓜|哈密瓜|柿子|桃子|梨子|芒果"
    ),
    # === 中文常见姓名前缀（XX 哥/姐/老师/医生）形态学 ===
    re.compile(
        r"[一-鿿]{1,3}(?:老师|医生|教练|经理|主任|队长|班长|师傅|老板)|"
        r"小[一-鿿]{1,3}(?!时|心|说|时候|学|说话)|"  # 小 X 称呼但非常用词
        r"老[一-鿿]{1,3}(?!师|板|实|是|了|早|说|好|样|样子)"  # 老 X 称呼但非常用词
    ),
]

# Negation patterns (用于 assistant_reference_rejected 检测)
_NEGATION_RE = re.compile(
    r"(不是|不对|错了|没说过|没说|没那个意思|没这意思|"
    r"我没|我不|你搞错|你记错|你弄错|"
    r"才不是|才没|哪有|怎么会)"
)


def _extract_entities_from_text(text: str, max_entities: int = 8) -> list[str]:
    """Extract distinct named entities from a turn text (Layer 3 reuse detection).

    Uses regex-based pattern matching across 7 entity classes (place / brand /
    time / song / activity / food / family).
    Returns deduplicated list (preserves order). Cap at max_entities to avoid
    pathological large turn texts.
    """
    if not isinstance(text, str) or not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for pat in _ENTITY_EXTRACT_RES:
        for m in pat.finditer(text):
            ent = str(m.group(0) or "").strip()
            # 2026-05-04 audit fix: normalize internal whitespace
            # "2025 年" → "2025年"、便于跨 turn entity matching
            ent = re.sub(r"\s+", "", ent)
            if ent and ent not in seen and len(ent) >= 2:
                seen.add(ent)
                out.append(ent)
                if len(out) >= max_entities:
                    return out
    return out


def _detect_negation(user_text: str) -> bool:
    """Cheap negation detection for assistant_reference_rejected signal."""
    if not isinstance(user_text, str) or not user_text:
        return False
    return bool(_NEGATION_RE.search(user_text))


# ═════════════════════════════════════════════════════════════════════════
# Phase Q.W2 / Option E P2 (2026-05-04): Topic-level reuse detection
# ═════════════════════════════════════════════════════════════════════════
# Audit feedback：entity-only 对"没有实体的价值观/偏好复用"无感。
# 加 topic categorization：8 类、其中 stance_value / life_value_meaning 是
# 高信号（用户表达 attitude / 价值观、跨 turn 复用 = 真 reuse）；其他类如
# schedule / emotion 是状态类（recurrence 不等于 reuse）、低信号。
#
# 设计原则：
#   - entity 重叠 = strong signal (confidence 0.7)、所有 topic 类都触发
#   - topic 重叠 (无 entity) = weak signal (confidence 0.5)、仅 stance_value /
#     life_value_meaning 类触发（避免 "我累 vs 我也累" 假复用）

# Stance / value：用户表达稳定态度立场（强复用价值）
_TOPIC_STANCE_VALUE_RE = re.compile(
    r"(其实|说实话|实际上|老实说|"
    r"比起.*[更也还]|反而|宁可|宁愿|不如|与其|"
    r"我更想|我更喜欢|我更看重|我更在意|"
    r"在我看来|对我来说|对我而言|我个人|我自己觉得|"
    r"重要的是|关键是|核心是|本质上|说到底)"
)

# Life value / meaning：人生意义 / 目标 / 平凡 / 价值观 主题
_TOPIC_LIFE_VALUE_RE = re.compile(
    r"(意义|目标|价值|平凡|人生|未来|"
    r"努力|拼|奋斗|"
    r"为什么.*要|怎么办|应该|"
    r"不想.*平凡|不想.*普通|不想.*这样|害怕.*变成|"
    r"一开始|初心|本来想|当初)"
)

# Schedule / lifestyle (作息/节奏、recurrence 类、低信号)
_TOPIC_SCHEDULE_RE = re.compile(
    r"(熬夜|作息|早睡|晚睡|睡眠|休息|"
    r"通宵|凌晨|半夜|"
    r"几点睡|几点起|"
    r"早起|起晚|睡懒觉|"
    r"累|疲惫|疲倦|疲劳)"
)

# Work / study (工作学习、recurrence 类、低信号)
_TOPIC_WORK_STUDY_RE = re.compile(
    r"(工作|学习|上学|复习|考试|作业|"
    r"项目|开发|代码|"
    r"加班|周末加班|加班费|"
    r"老板|同事|"
    r"图书馆|教室|课|备考)"
)

# Hobby / interest (兴趣、可能复用)
_TOPIC_HOBBY_RE = live_pattern(
    r"(游戏|抽卡|手游|二次元|"
    r"音乐|{SETTING:hobby_terms}|演唱会|"
    r"看书|读书|小说|漫画|动画|"
    r"运动|健身|跑步|爬山)"
)

# Emotion state (情绪状态、recurrence 类、低信号)
_TOPIC_EMOTION_RE = re.compile(
    r"^[^。.！!？?]*(开心|高兴|难过|失落|烦|累|焦虑|担心|害怕|紧张|"
    r"郁闷|无奈|崩溃|绝望)[^。.！!？?]*$"
)

# Relationship self (用户自己的家庭/朋友关系)
_TOPIC_RELATIONSHIP_SELF_RE = re.compile(
    r"我家|我妈|我爸|我哥|我姐|我弟|我妹|我老板|我同学|我老师|我朋友"
)

# Self identity / development (用户身份/开发者/项目相关)
_TOPIC_DEVELOPER_RE = re.compile(
    r"开发|系统|功能|bug|代码|项目|架构|"
    r"我做|我写|我搭|我建|"
    r"我在.*做|我在.*开发|我在.*研究"
)


# Topic categories with reuse signal strength
_TOPIC_CATEGORIES = [
    ("stance_value", _TOPIC_STANCE_VALUE_RE, "high"),
    ("life_value_meaning", _TOPIC_LIFE_VALUE_RE, "high"),
    ("developer_self", _TOPIC_DEVELOPER_RE, "medium"),
    ("hobby_interest", _TOPIC_HOBBY_RE, "medium"),
    ("relationship_self", _TOPIC_RELATIONSHIP_SELF_RE, "medium"),
    ("schedule_lifestyle", _TOPIC_SCHEDULE_RE, "low"),
    ("work_study", _TOPIC_WORK_STUDY_RE, "low"),
    ("emotion_state", _TOPIC_EMOTION_RE, "low"),
]


def _extract_topic_tags(text: str) -> list[tuple[str, str]]:
    """Extract topic tags from a turn text。

    Returns list of (category, strength) tuples。strength in ('high', 'medium', 'low')。
    high = stance / value 类（reuse 价值高）
    medium = personal / hobby（中等）
    low = state / recurrence（reuse 信号弱、容易假阳性）

    一个 turn 可有多个 tag、因为话题可重叠（如"我妈最近身体不好我有点担心"
    同时含 relationship_self + emotion_state）。
    """
    if not isinstance(text, str) or not text:
        return []
    out: list[tuple[str, str]] = []
    for category, pattern, strength in _TOPIC_CATEGORIES:
        if pattern.search(text):
            out.append((category, strength))
    return out


# ── Topic label keywords (入库细粒度话题标签推断) ──
# key = label slug（与 TurnFactCard 词表一致），value = 触发该 label 的关键词列表
_BASE_TOPIC_LABEL_KEYWORDS: dict[str, list[str]] = {
    "school_exam":     ["考试", "备考", "复习", "テスト", "受験", "期末", "期中", "模拟考"],
    "homework":        ["作业", "宿題", "写作业"],
    "class_school":    ["上课", "放学", "下课", "授業", "課", "学校", "早自习"],
    "graduation":      ["毕业", "卒業", "考大学", "升学", "高考"],
    "student_life":    ["同学", "校园", "班级", "学园"],
    "food_cooking":    ["吃", "做饭", "料理", "烹饪", "食べ", "ご飯", "餐", "饭", "菜"],
    "shopping":        ["购物", "逛街", "买东西", "ショッピング", "买了"],
    "outing":          ["出门", "出去", "外出", "散步", "游玩", "逛"],
    "home_life":       ["在家", "家里", "家事", "房间", "自室", "部屋"],
    "sleep_rest":      ["睡觉", "午睡", "休息", "寝る", "疲れ", "疲惫", "累"],
    "family":          ["妈妈", "爸爸", "家人", "父母", "お母さん", "お父さん", "家族", "兄弟", "姐姐"],
    "friendship":      ["朋友", "友达", "友情", "友人"],
    "romance":         ["恋爱", "喜欢", "心动", "告白", "恋", "好きな人", "表白"],
    "part_time_job":   ["兼职", "打工", "バイト", "アルバイト", "做兼职"],
    "conflict":        ["吵架", "矛盾", "冲突", "误解", "ケンカ", "闹矛盾"],
    "apology":         ["道歉", "抱歉", "对不起", "ごめん", "すまない", "认错"],
    "anxiety":         ["焦虑", "担心", "紧张", "不安", "心配", "忧虑"],
    "loneliness":      ["孤独", "寂寞", "孤单", "ひとり", "寂しい"],
    "joy":             ["开心", "高兴", "快乐", "嬉しい", "楽しい", "喜悦"],
    "worry":           ["担忧", "烦恼", "烦", "悩み", "困扰", "发愁"],
    "future_plan":     ["计划", "将来", "未来", "打算", "夢", "目标", "志愿", "规划"],
    "creative_project":["创作", "作品", "创意", "项目", "企划"],
    "hobby_games":     ["游戏", "ゲーム", "手游", "玩游戏"],
    "hobby_anime":     ["动漫", "アニメ", "番剧", "二次元", "动画"],
    "hobby_books":     ["读书", "看书", "小说", "漫画", "本", "阅读"],
    "exercise":        ["运动", "跑步", "健身", "锻炼", "体育"],
    "health":          ["生病", "身体", "健康", "发烧", "病気", "薬", "吃药"],
    "dev_project":     ["开发", "代码", "功能", "bug", "架构", "系统", "编程", "项目"],
    "travel":          ["旅行", "旅游", "出游", "旅", "trip"],
    "money":           ["钱", "零花钱", "工资", "お金", "缺钱"],
    "news_event":      ["新闻", "消息", "事件", "发生", "ニュース", "热点"],
}
# Deployment vocabulary (setting "topic_label_keywords") is consulted first, then the built-in table.
_TOPIC_LABEL_KEYWORDS = live_table("topic_label_keywords", _BASE_TOPIC_LABEL_KEYWORDS)


def _infer_topic_labels_from_query(text: str) -> list[str]:
    """从查询文本纯规则推断匹配的 topic_labels（不调用 LLM，检索路径零延迟）。"""
    if not text:
        return []
    out: list[str] = []
    for label, keywords in _TOPIC_LABEL_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            out.append(label)
    return out[:8]


def _topic_overlap(tags_a: list[tuple[str, str]], tags_b: list[tuple[str, str]]) -> tuple[set[str], str]:
    """Compute topic overlap between two turn's topic tags。

    Returns (overlapping_categories, max_strength)。
    max_strength = max strength of overlapping categories ('high'|'medium'|'low'|'none')
    """
    if not tags_a or not tags_b:
        return (set(), "none")
    cats_a = {c for c, _ in tags_a}
    cats_b = {c for c, _ in tags_b}
    overlap = cats_a & cats_b
    if not overlap:
        return (set(), "none")
    # Find max strength among overlapping categories
    strength_map_a = {c: s for c, s in tags_a}
    strength_map_b = {c: s for c, s in tags_b}
    strengths = []
    for cat in overlap:
        sa = strength_map_a.get(cat, "low")
        sb = strength_map_b.get(cat, "low")
        # Take max (assume same category has same strength on both sides; safety)
        max_s = sa if sa == "high" or sb == "high" else (
            "high" if "high" in (sa, sb)
            else "medium" if "medium" in (sa, sb)
            else "low"
        )
        strengths.append(max_s)
    if "high" in strengths:
        max_strength = "high"
    elif "medium" in strengths:
        max_strength = "medium"
    else:
        max_strength = "low"
    return (overlap, max_strength)


def _is_reuse_worthy_topic_overlap(overlap: set[str], max_strength: str) -> bool:
    """Topic overlap 是否值得记 reuse 信号。

    high: 强 stance/value 重叠 → reuse 信号
    medium: 至少 2 个类重叠 → reuse 信号（多 topic 同时命中 = 强相关）
    low: 单 low 类 → 不计（容易假阳性、如"累 vs 累"）
    """
    if not overlap:
        return False
    if max_strength == "high":
        return True
    if max_strength == "medium" and len(overlap) >= 2:
        return True
    return False


# ═════════════════════════════════════════════════════════════════════════
# Layer 2 State Machine constants (Option E、2026-05-04)
# ═════════════════════════════════════════════════════════════════════════
CONSOLIDATION_STATE_TRANSIENT = "transient"
CONSOLIDATION_STATE_CANDIDATE = "candidate"
CONSOLIDATION_STATE_PROFILE_PROMOTED = "profile_promoted"
CONSOLIDATION_STATE_KEY_EVENT_PROMOTED = "key_event_promoted"
CONSOLIDATION_STATE_RECALL_ARCHIVE = "explicit_recall_archive"
CONSOLIDATION_STATE_STALE_ARCHIVE = "stale_archive"
CONSOLIDATION_STATE_ARCHIVE_CANDIDATE = "archive_candidate"

CONSOLIDATION_VALID_STATES = {
    CONSOLIDATION_STATE_TRANSIENT,
    CONSOLIDATION_STATE_CANDIDATE,
    CONSOLIDATION_STATE_PROFILE_PROMOTED,
    CONSOLIDATION_STATE_KEY_EVENT_PROMOTED,
    CONSOLIDATION_STATE_RECALL_ARCHIVE,
    CONSOLIDATION_STATE_STALE_ARCHIVE,
    CONSOLIDATION_STATE_ARCHIVE_CANDIDATE,
}

# Threshold constants (calibrate after Phase 1 data)
CONSOLIDATION_SCORE_PROMOTE = 0.70   # >= → candidate for promotion (with reuse gate)
CONSOLIDATION_SCORE_FAST_TRACK = 0.85  # >= → bypass reuse gate
# 2026-05-04 audit calibration: 0.45 → 0.30 (含金 chitchat 进 candidate)
# 实测 chitchat 单 feature 命中约 0.14-0.15、attitude 表达 0.36 → 0.30 阈值能区分
CONSOLIDATION_SCORE_CANDIDATE_MIN = 0.30  # >= → enter candidate state

# 2026-08-02: 重大情感话题直写 semantic 的触发标准（用户拍板）
# 标签组 = ledger 受控词表的情感子集；阈值来自 876 条历史 keep 记忆初筛（命中 ~13%、
# 抽查近零误伤；生产端用 ledger 卡片自带 salience、与初筛标尺对齐良好）。
_EMOTIONAL_DIRECT_SEMANTIC_LABELS = frozenset({
    "anxiety", "loneliness", "worry", "conflict", "apology", "romance", "joy",
})
_EMOTIONAL_DIRECT_SEMANTIC_MIN_SALIENCE = 0.60


def _typed_gate_decision(features: dict, score: float, reuse_count: int) -> tuple[str, str, str]:
    """Layer 4 typed gate: decide promotion target based on features + reuse.

    Returns (target_layer, engram_type, decision_reason).
    target_layer in ("semantic", "episodic", None).
    None → no promotion (stay candidate or other state).

    Logic（2026-05-09 SPECIFICITY 去 hardcode 重构）：
      - cold_start fast-track: 绕过 reuse、看 user_fact / specificity 分类
      - score >= 0.85: 同 fast-track（强信号）
      - score 0.70-0.85: 需要 confirmed_reuse >= 1
      - score < 0.70: not promotable
      Type 决策（specificity 改为 specific_signal = max(specificity, novelty*0.85)）：
        user_fact_density >= 0.6 + stance >= 0.4 → profile (developer_profile)
        specific_signal >= 0.6 + emotional >= 0.4 → key_event
        else                                     → recall_archive
      novelty*0.85 折算：novel 但无 hardcoded entity 命中的 turn（如"我换工作了"
      embedding_novelty=0.7、specificity=0）能过 specific_signal=0.595 ≈ 0.6 阈值。
    """
    if not isinstance(features, dict):
        return (None, None, "no_features")

    cold_start = _is_cold_start_fast_track(features)
    user_fact = float(features.get("user_fact_density", 0.0))
    specificity = float(features.get("specificity", 0.0))
    novelty = float(features.get("embedding_novelty", 0.5))
    emotional = float(features.get("emotional_weight", 0.0))
    stance = float(features.get("stance_or_value", 0.0))

    # specific_signal: hardcoded 形态 specificity OR 语义 novelty（85% 折算）
    # novelty 是 0.0-1.0 (0.5 neutral)、需要 > 0.5 才真正贡献
    # 折算公式：max(0, novelty - 0.5) * 1.7  → novelty=0.85 → 0.595（约等 specificity 弱阈值 0.5）
    novelty_signal = max(0.0, novelty - 0.5) * 1.7
    specific_signal = max(specificity, novelty_signal)

    # Decide if promotion-eligible
    if cold_start:
        eligible = True
        reason_prefix = "cold_start"
    elif score >= CONSOLIDATION_SCORE_FAST_TRACK:
        eligible = True
        reason_prefix = f"score>={CONSOLIDATION_SCORE_FAST_TRACK}"
    elif score >= CONSOLIDATION_SCORE_PROMOTE and reuse_count >= 1:
        eligible = True
        reason_prefix = f"score>={CONSOLIDATION_SCORE_PROMOTE}+reuse>={reuse_count}"
    else:
        return (None, None,
                f"not_eligible (cold={cold_start}, score={score:.2f}, reuse={reuse_count})")

    # Typed selection
    if user_fact >= 0.6 and stance >= 0.4:
        return ("semantic", "user_profile_fact", f"{reason_prefix}+profile_gate")
    elif specific_signal >= 0.6 and emotional >= 0.4:
        return ("episodic", "key_event", f"{reason_prefix}+key_event_gate")
    elif user_fact >= 0.6:
        # 2026-05-15 fix: stance 不足时不再直接升 semantic
        # 旧行为: semantic/user_profile_fact — "吃了饭看了视频" 也成永久画像 fact
        # 新行为: episodic/key_event — 保留在情节记忆中、自然衰减、不进画像
        return ("episodic", "key_event", f"{reason_prefix}+profile_weak_gate")
    elif specific_signal >= 0.5:
        return ("episodic", "key_event", f"{reason_prefix}+key_event_weak_gate")
    else:
        # 有信号但都不强 → recall_archive (有事件价值、但不进画像)
        return ("episodic", "recall_archive", f"{reason_prefix}+recall_archive_gate")


def _developer_display_name() -> str:
    return user_name()


def _canonicalize_developer_fact(fact: str, *, character: str = "") -> str:
    tx = str(fact or "").strip()
    if not tx:
        return ""
    name = _developer_display_name()
    char = str(character or "").strip()

    # user_observation 偶尔保留开发者原话中的第二人称。不能把
    # "用户你以后……" 机械替换成 "用户你以后……"，否则改写器会把
    # "你" 的未来错误归到用户身上。当前角色明确时保留说话关系。
    dialogue_body = re.sub(
        rf"^(?:用户|对方|开发者|{re.escape(name)})[，,：:\s]*",
        "",
        tx,
        count=1,
    )
    if char and dialogue_body.startswith("你"):
        return f"{name}对{char}说：{dialogue_body}"

    tx = re.sub(r"^(?:用户|对方|开发者)", name, tx)
    tx = tx.replace("用户一直都很爱你们", f"{name}一直都很爱你们")
    tx = tx.replace("用户喜欢", f"{name}喜欢")
    tx = tx.replace("用户希望", f"{name}希望")
    return tx


# 2026-06-02: 对话残留前置拦截。
# 上游 raw_facts 抽取偶尔把 user 的提问 / 安慰 / 口头语原样当成 fact。
# _canonicalize_developer_fact 会先把「嗯，话说…」加主语成「用户嗯，话说…」、
# 架空 _rewrite_fact_via_llm 里 ^口头语 的引语拦截（^嗯 再也匹配不到）。
# 这里在 canonicalize **之前**、对原文判定（内部先剥任何已有主语前缀）。
_FACT_RESIDUE_SUBJECT_PREFIX_RE = live_pattern(r"^(?:{USER}|开发者|用户|对方)[，,：: ]*")
_FACT_RESIDUE_LEAD_FILLER_RE = re.compile(
    r"^(?:嗯+|啊+|哦+|呃+|唉+|诶+|哇+|喂+|唔+|呀+)"          # 语气词开头
    r"|^(?:哈{2,}|嘿{2,}|呵{2,}|嘻{2,})"                     # 哈哈/嘿嘿/呵呵
    r"|^(?:好吧|那好吧|话说|说起来|对了|是吗|真的吗|原来)"    # 口头语 / 应答开头
    r"|^当然(?:可以|行|好|是|啊)"                            # 当然可以啊…
    r"|^(?:可以啊|没事(?:儿|啦|的)?[，,。]|好的[，,。]|行吧)"  # 应答短语
)
# user 把疑问句抛给角色、被误抽成「关于 user 的 fact」（你…吗/呢/吧？）
_FACT_RESIDUE_QUESTION_RE = re.compile(r"你[^。！？!?]*(?:吗|呢|吧)[？?。.]?$|[？?]$")


def _fact_is_dialogue_residue(raw_fact: str) -> bool:
    """raw_fact 是否为对话残留（口头语 / 应答开头、或抛给角色的疑问句），
    这类既非「关于 user 的稳定事实」、LLM 改写也救不回、直接 suppress。"""
    s = re.sub(r"\s+", "", str(raw_fact or "").strip())
    if not s:
        return False
    body = _FACT_RESIDUE_SUBJECT_PREFIX_RE.sub("", s, count=1)
    if not body:
        return False
    if _FACT_RESIDUE_LEAD_FILLER_RE.match(body):
        return True
    if "你" in body and _FACT_RESIDUE_QUESTION_RE.search(body):
        return True
    return False


# 2026-05-09: LLM rewrite to concise callback-able fact form
# semantic 层 fact 不应是 verbatim 原话、应是抽象后的稳定 proposition
# （短、portable、能 inline 进 light_callback）

_FACT_REWRITE_PROMPT_TEMPLATE = """你正在帮助 AI 角色「{character}」把它和开发者「{name}」的对话沉淀成一条**简短的稳定事实**。

【对话主体】
- 开发者本名：{name}（{agent_count} 个角色都习惯叫{pron}{name}）
- 当前角色：{character}（正在听 {name} 说话的角色）

【任务】把下面这句开发者说过的话或关于开发者的观察，抽象成一条简短的、portable 的事实陈述，能在以后被角色自然地 callback。

【硬规则】
1. 控制在 8-30 字、单句、句号结尾
2. **先区分说话者与命题主语**：开发者自身的事实用「{name}」作主语；形如「{name}说：……」时引语里的「我」= {name}、「你」= {character}（听者），命题主语仍是 {name}；形如「{name}对{character}说：你……」时，「你」指 {character}，不能把 {character} 的处境、未来或选择改写成 {name} 的事实
3. **保留言语行为的确定性边界**：祝愿、安慰、建议、猜测、预测只能写成「{name}希望/鼓励/建议/认为……」，不能把尚未发生的内容写成确定事实
4. **角色简称必须从原话中保留**：如果原话明确出现 {agent_count} 角色简称（{agent_list}）作为对象/听者，或由「{name}对{character}说」明确了听者，重写必须保留该简称；不得凭空添加其他角色
   - 原话有"{agent_a}" → 重写必须出现"{agent_a}"
   - 原话在「{name}对{character}说」之后出现"你" → 重写写"{character}"
   - **严禁 hallucinate**：原话没出现"{agent_list}"中任一个 → 重写就不许出现该简称
5. 抽象掉口语化外壳（语气词"的啦/嘛/呀"/感叹"啊嗯哦"/连接词"所以/不过"）
6. 抽象掉具体一次性事件细节、保留稳定属性 / 偏好 / 立场 / 长期愿望
7. 不要加猜测内容、不要扩充原话没有的信息
8. 直接输出事实、不要任何前缀（如"事实："）、不要解释、不要引号

{examples}现在轮到你：

原话：{raw_fact}
事实："""

# Few-shot block for the rewrite prompt; deployments override via
# persona_registry.setting("fact_rewrite_examples"). Placeholders: {name} {character} {agent_a} {agent_b}.
_DEFAULT_FACT_REWRITE_EXAMPLES = """【正例】
原话：等我忙完这阵子，我们约个周末一起去海边吧。现在{agent_a}先照顾好自己就好
事实：{name}希望忙完后和{agent_a}周末一起去海边。   ← 保留了"{agent_a}"

原话：我以后还是想换个安静点的地方住，现在合租太吵了，我本来就很怕吵
事实：{name}怕吵，以后想搬到安静的地方住。   ← 原话无角色简称、不要硬加

原话：我小时候特别怕打雷，因为有一次停电只有我一个人在家
事实：{name}从小怕打雷，因为小时候停电时独自在家过。

原话：{agent_b}你别熬太晚了、我会担心的
事实：{name}担心{agent_b}熬夜。   ← 保留"{agent_b}"

原话：{name}对{character}说：你以后一定能遇到愿意陪你逛书店的朋友的，到时候一起去就好
事实：{name}鼓励{character}以后和朋友一起去逛书店。   ← 「你」是 {character}，且祝愿不能写成确定未来

原话：{name}说：虽然我们隔得很远，但我会常常想起你画画的样子
事实：{name}常常想起{character}画画的样子。   ← 「{name}说：」引语里「我」是 {name}、「你」是 {character}，别颠倒

【反例（不要这样写）】
- "用户说他忙完要去海边"  ← 用了"用户"
- "{character}常常想起{name}画画的样子"（原话是「{name}说：…我会常常想起你画画的样子…」）  ← 主客颠倒：引语里的「我」是 {name}、不是 {character} ❌
- "{name}希望忙完后一起去海边"  ← 原话明确提了"{agent_a}"、却被丢掉了 ❌
- "{name}希望忙完后和你们一起去海边"  ← 用了"你们"代替简称 ❌
- "{name}表示自己很想去海边然后还提到{agent_a}照顾好自己就好"  ← 太长 + 流水
- "{name}可能性格内向"  ← 凭空猜测
- "事实：{name}..."  ← 加了前缀

"""



def _rewrite_fact_via_llm(
    raw_fact: str,
    *,
    llm_caller: Callable[[str], str] | None = None,
    character: str = "",
) -> str:
    """LLM 重写 fact 为简短 callback-able 形态。

    - llm_caller None / 失败 / 输出违规 → 返原 raw_fact（fail-safe、不阻断写入）
    - 仅 raw_fact > 25 字时才调（短句不需要 rewrite、节省 latency / cost）
    - LLM 输出做 sanity check：长度 ≤ 60、不含 "用户/对方" / "事实：" 前缀
    """
    from tsuzuri.memory_core_impl.meta_policy import _COLLECTIVE_SCOPE_MEMBERS
    s = (raw_fact or "").strip()
    if not s:
        return ""
    # 2026-05-12: 短句虽然不调 LLM、但要拒掉明显的"用户原话引语" pattern
    # （省略号 / 语气词 / 引语开头）—— 之前 "好吧...你下次可以换个说法"
    # 这种 21 字的引语 verbatim 进了 semantic 层。
    _quote_pattern = re.compile(r"^(好吧|嗯|啊|哦|呃|唉|诶|哇|喂)|\.\.\.|…|[嘛吧呢呀啦呗哦哎]$")
    quote_like = bool(_quote_pattern.search(s))
    if quote_like and (len(s) <= 25 or not callable(llm_caller)):
        _tsz_emit(f"[FactRewrite] REJECT quote-like fact: {s[:60]!r}", flush=True)
        return ""
    # 短句直接放行、不调 LLM（节省 80% 调用）；但含对话信号（第二人称「你」/
    # 问句 / 第一人称对话口吻）的短句即使 ≤25 字也强制走 LLM 抽象——
    # 2026-06-02：这类是对话原话漏进 semantic 的主源（119/121 条 fact ≤25 字、
    # 旧逻辑全部跳过改写、baseline 只加主语不改人称）。
    _has_dialogue_signal = bool(re.search(r"你|[？?]|我也|我不希望|我希望你", s))
    if len(s) <= 25 and not _has_dialogue_signal:
        return s
    if not callable(llm_caller):
        return s  # 无 LLM 可用时降级、保留原话（其他 filter 仍跑）
    name = _developer_display_name()
    # 2026-05-17 C: character 参数 wire 进 prompt 模板。之前 dead arg、LLM 不知道当前角色、
    # bug case 把"角色C的 fact"重写成"角色A的 fact"——纯 hallucination。
    _char_display = str(character or "").strip() or "角色"
    try:
        _agents = list(agent_names())
        _agent_a = _agents[0] if _agents else _char_display
        _agent_b = _agents[2] if len(_agents) > 2 else (_agents[-1] if _agents else _char_display)
        _examples = str(setting("fact_rewrite_examples", _DEFAULT_FACT_REWRITE_EXAMPLES) or "").format(
            name=name, character=_char_display, agent_a=_agent_a, agent_b=_agent_b,
        )
        prompt = _FACT_REWRITE_PROMPT_TEMPLATE.format(
            name=name, character=_char_display, raw_fact=s, examples=_examples,
            agent_count=len(_agents), agent_list="/".join(_agents), agent_a=_agent_a, pron=user_pronoun(),
        )
        out = (llm_caller(prompt) or "").strip()
        # 取第一行（防 LLM 多输出几行）
        out = out.splitlines()[0].strip() if out else ""
        # sanity: 去 quoting / "事实：" 前缀
        out = re.sub(r"^[「『\"\'`]+|[」』\"\'`]+$", "", out)
        out = re.sub(r"^(事实|fact)\s*[:：]\s*", "", out, flags=re.IGNORECASE)
        out = out.strip()
        if not out or len(out) > 60 or "用户" in out or "对方" in out:
            _tsz_emit(f"[FactRewrite] BAD output len={len(out)} text={out[:60]!r}, fallback", flush=True)
            return s
        # 2026-09-07 C: 主语方向校验。输入以 {name} 开头（命题主语是开发者，
        # 含「{name}说：…」/「{name}对X说：…」形态）时，若输出以角色简称开头，
        # 判定为主客颠倒（如 "用户说：…我会想起你…" → "角色会想起用户…"）、
        # fallback 原句。宁可保留啰嗦原句、也不写归因错误的事实。
        if s.startswith(name) and any(
            out.startswith(cn) for cn in _COLLECTIVE_SCOPE_MEMBERS if cn
        ):
            _tsz_emit(f"[FactRewrite] BAD subject-swap: in={s[:40]!r} out={out[:40]!r}, fallback", flush=True)
            return s
        # 末尾补句号
        if out and out[-1] not in "。.！!？?":
            out = out + "。"
        _tsz_emit(f"[FactRewrite] {len(s)}->{len(out)}: {s[:40]!r} -> {out[:40]!r}", flush=True)
        return out
    except Exception as e:
        _tsz_emit(f"[FactRewrite] FAILED ({type(e).__name__}: {e}), fallback to raw", flush=True)
        return s


def _recall_focus_keyword_score(focus: str, text: str) -> float:
    from tsuzuri.memory_core_impl.recall_scoring import _clamp
    focus_norm = _norm_text(str(focus or "")).lower()
    text_norm = _norm_text(str(text or "")).lower()
    if not focus_norm or not text_norm:
        return 0.0
    if focus_norm in text_norm:
        return 1.0
    chunks: list[str] = []
    for seq in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", focus_norm):
        if re.fullmatch(r"[a-z0-9]+", seq):
            if len(seq) >= 2:
                chunks.append(seq)
            continue
        if len(seq) == 1:
            continue
        if len(seq) == 2:
            chunks.append(seq)
        else:
            chunks.extend(seq[i:i + 2] for i in range(0, len(seq) - 1))
    filtered = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk in _FOCUS_GENERIC_CHUNKS or chunk in seen:
            continue
        seen.add(chunk)
        filtered.append(chunk)
    if not filtered:
        return 0.0
    hits = sum(1 for chunk in filtered if chunk in text_norm)
    return _clamp(hits / max(1, len(filtered)))


def _candidate_is_memory_probe_text(text: str) -> bool:
    from tsuzuri.memory_core_impl.meta_policy import _EXPLICIT_RECALL_RE
    raw = str(text or "")
    return bool(_EXPLICIT_RECALL_RE.search(raw) and re.search(r"[?？吗]", raw))


def _topic_from_semantic_fact(text: str) -> str:
    m = _TOPIC_ONLY_FACT_RE.match(str(text or "").strip())
    if not m:
        return ""
    topic = re.sub(r"\s+", " ", str(m.group(1) or "").strip())
    return topic[:24]


def _dedupe_short_strings(values: list[str] | tuple[str, ...] | None = None, *, limit: int = 12, max_len: int = 32) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values if isinstance(values, (list, tuple)) else []:
        text = str(value or "").strip()
        if not text:
            continue
        key = _norm_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text[: max(1, int(max_len or 32))])
        if len(out) >= max(1, int(limit or 1)):
            break
    return out


def _is_collective_memory_text(text: str, *, character: str = "") -> bool:
    from tsuzuri.memory_core_impl.meta_policy import _COLLECTIVE_SCOPE_MEMBERS
    raw = str(text or "").strip()
    if not raw:
        return False
    char = str(character or "").strip()
    mentioned_members = [m for m in _COLLECTIVE_SCOPE_MEMBERS if m in raw and m != char]
    return bool(_COLLECTIVE_ADDRESS_RE.search(raw) or len(mentioned_members) >= 1)


def _fact_is_durable_from_user(fact: str, user_text: str, *, explicit_request: bool = False) -> bool:
    tx = _canonicalize_developer_fact(fact)
    user = str(user_text or "").strip()
    if not tx or _topic_from_semantic_fact(tx):
        return False
    if explicit_request or _EXPLICIT_MEMORY_WRITE_RE.search(user):
        return True
    if _ASSISTANT_ONLY_FACT_RE.search(tx) and not persona_regex(r"(用户|开发者|对方|{USER}|你)").search(tx):
        return False
    if persona_regex(r"(用户|开发者|对方|{USER}|我|我的|喜欢|不喜欢|讨厌|希望|想要|正在|准备|在忙|压力|紧张|爱你们)").search(tx):
        return True
    return False


def _baseline_subject_normalize(text: str, name: str) -> str:
    """semantic fact 的轻 baseline normalize：
      - 缺主语 → 前置 name
      - "我..." 开头 → 把首字"我"替成 name（不做切片所有格手术）
      - 含「你」或语气/呼格开头的对话原话 → 引语形态「{name}说：…」
      - 缺句末标点 → 补"。"
    """
    s = (text or "").strip()
    if not s:
        return s
    # 已有 name 主语或系统级前缀 → 不动主语
    if s.startswith(name) or s.startswith("系统"):
        pass
    # "我..." 开头 → 直接替换为 name + 剩余（不区分亲属/所有格、复杂场景留给 LLM）
    elif s.startswith("我"):
        s = name + s[1:]
    # 含第二人称单数「你」（排除「你们」小组称呼）或语气/呼格开头的文本是对话原话：
    # 裸拼主语会产生呼格歧义（"你知道吗，虽然我们隔得很远，但我会想起你…"
    # 会被读成对用户说话、rewrite LLM 主客颠倒，实测事故 2026-09-07）。
    # 改用引语形态「{name}说：…」保留原始人称，由下游 rewrite 按「我=name / 你=听者」归因。
    elif re.search(r"你(?!们)", s) or _FACT_RESIDUE_LEAD_FILLER_RE.match(s):
        s = f"{name}说：{s}"
    # 其他无主语 → 前置 name
    else:
        s = name + s
    # 末尾标点
    if s and s[-1] not in "。.！!？?":
        s += "。"
    return s


def _derive_semantic_facts_from_user_text(user_text: str) -> list[str]:
    """显式记忆请求路径的 fact 抽取（2026-05-09 轻量化重构）。

    职责仅 3 项：
      1. 识别显式记忆意图（_EXPLICIT_MEMORY_WRITE_RE 或 "爱你们" hardcoded）
      2. 剥离引导词（"请帮我记住"/"麻烦记一下" 等）
      3. baseline 主语 + 句末标点 normalize

    刻意**不做**的事（之前 P2/P3/P5 bug 来源）：
      - "我家" → "用户的家" 亲属/所有格切片（错位多、留给 LLM）
      - "以后叫我XX" → "希望" 拼接（内部"我"歧义、留给 LLM）
      - "下次" → "以后" 替换（语义损失）

    长句（> 25 字）下游 _rewrite_fact_via_llm 会用 LLM 抽象、覆盖这些。
    短句保留 baseline、虽朴素但语法稳定。
    """
    user = re.sub(r"\s+", " ", str(user_text or "").strip())
    if not user:
        return []
    name = _developer_display_name()

    # 特殊 hardcoded：爱你们
    if "爱你们" in user or "很爱你们" in user:
        return [_baseline_subject_normalize("一直都很爱你们", name)]

    # 没显式请求 → 不产 fact（其他 user fact 走 raw_facts 路径、由 LLM 抽取）
    if not _EXPLICIT_MEMORY_WRITE_RE.search(user):
        return []

    # 剥离引导词
    cleaned = re.sub(
        r"^(?:请|麻烦)?(?:你)?(?:帮我)?(?:记住|记一下|要记得|你要记得)\s*[:：,，]?\s*",
        "", user,
    )
    cleaned = cleaned.strip(" ，,。.!！?？")
    if len(_norm_text(cleaned)) < 4:
        return []

    # baseline normalize（轻、确定）
    fact = _baseline_subject_normalize(cleaned, name)
    return [fact[:120]]
