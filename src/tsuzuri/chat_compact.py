"""每日 chat session 压缩、v1 衰减回顾与 DailyContinuityCapsule v2 facade。

设计：
  · 上游：Phase 2 archive 写入 chat_sessions/archive/{YYYY-MM-DD}/{sid}.json
  · 本模块：
      (1) rule-based strip — 去除角色B式 ······-only / 嗯-only bubble、proactive auto_greet 等噪声
      (2) session map — 每个 archive 生成兼容的 5 段摘要
      (3) character/day reduce — 跨 session 合并为唯一 5 段固定回顾
      (4) 衰减 merge — 昨日 / 2-7 天 / 1-2 周 三档分别加载、合并
  · 下游：宿主应用 在 build prompt 时调 load_historical_recap → 注入 `historical_chat_recap` block

v2（2026-07-25）以独立 `.continuity.v2.json` sidecar 生成。有效记录经 TurnFact ledger
resolve → FactWorkspace selective projection，与固定的角色日级回顾并行；两者不再按日期互斥。

5 段 schema（宿主应用 专属、不是 Claude Code 的 9 段）：
  1. 重要事件
  2. 关系状态变化
  3. ongoing 话题
  4. 用户透露的事实
  5. 角色情绪 / 承诺

衰减规则（2026-05-21 用户拍板）：
  · 昨日（1 天前）            → 900 字角色日级聚合总结、独立 block 段
  · 2-7 天前                  → 每日 80 字、合并 1 段
  · 1-2 周前                  → 每周 80 字、合并 1 段
  · > 2 周                    → 不再以"昨日聊过"形式注入、走 memory_core 长期记忆

2026-09-11 决定（原记为待决问题）：900 字是**注入侧**约束、不是生成侧硬门禁。
  `_daily_recap_quality_issues` 只对五段 schema 与解释性措辞判失败；超预算照写，
  payload 记 `over_budget` / `summary_chars`，由 `_truncate_recap_summary` 注入时
  按五段权重截断。理由：模型实测不服从 900 字上限（常 1300–1800 字），repair 也压
  不下来时旧逻辑整段不落盘，导致该日丢跨 session 聚合、退化为单 session 摘要、
  offline smoke 里甚至整天消失；而注入侧本来就会截断，hard fail 净损失。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from tsuzuri.persona_registry import user_name
import time
from pathlib import Path
from typing import Any, Callable

from tsuzuri.time_source import now_world as _ts_now_world


_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_CHAT_SESSIONS_DIR = os.path.join(_PROJECT_DIR, "chat_sessions")
_CHAT_SESSIONS_ARCHIVE_DIR = os.path.join(_CHAT_SESSIONS_DIR, "archive")


def _developer_history_view(history) -> list[dict]:
    """Select rows eligible for developer historical recap generation."""
    if not isinstance(history, list):
        return []
    # Legacy rows predate audience metadata and are developer-owned. Guest rows
    # stay in the raw archive for UI/audit but must never become tomorrow's dev
    # recap through the compaction channel.
    return [
        dict(row) for row in history
        if isinstance(row, dict)
        and str(row.get("audience_scope", "") or "").strip() in ("", "developer")
    ]


def _now_world():
    # DEBUG_MOCK_DATE-aware via time_source SSOT.
    return _ts_now_world()


def _today_world_date_str() -> str:
    return (_now_world() - datetime.timedelta(hours=2, minutes=30)).strftime("%Y-%m-%d")


def _date_str_add_days(date_str: str, days: int) -> str:
    d = datetime.date.fromisoformat(date_str)
    return (d + datetime.timedelta(days=days)).isoformat()


def _archive_dir_for_date(date_str: str) -> str:
    safe = "".join(c if c.isalnum() or c == "-" else "_" for c in str(date_str or ""))
    return os.path.join(_CHAT_SESSIONS_ARCHIVE_DIR, safe)


def _archive_file_path(session_id: str, date_str: str) -> str:
    """2026-05-22 hardening (defense-in-depth)：与 chat_server._chat_session_file_path
    对齐、禁 `.` 防 `..` 路径穿越、纯下划线/横线 sentinel 兜底为 anon。
    chat_server caller 已 sanitize、本层兜底防未来 caller 跳过 chat_server 直接传 raw input。
    """
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id or ""))
    if not safe or set(safe) <= {"_", "-"}:
        safe = "anon"
    return os.path.join(_archive_dir_for_date(date_str), f"{safe}.json")


def _summary_file_path(session_id: str, date_str: str) -> str:
    """同 _archive_file_path 的 sanitize 规则。"""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id or ""))
    if not safe or set(safe) <= {"_", "-"}:
        safe = "anon"
    return os.path.join(_archive_dir_for_date(date_str), f"{safe}.summary.json")


def _character_daily_summary_file_path(character: str, date_str: str) -> str:
    """Stable cross-session recap path without exposing Unicode in filenames."""
    digest = hashlib.sha1(
        str(character or "").strip().encode("utf-8", errors="ignore")
    ).hexdigest()[:12]
    return os.path.join(
        _archive_dir_for_date(date_str),
        f"character-{digest}.daily.summary.json",
    )


def _env_enabled(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw not in ("0", "false", "off", "no", "n")


def daily_continuity_v2_enabled() -> bool:
    """Generate v2 sidecars while keeping the established v1 recap path."""
    return _env_enabled("DAILY_CONTINUITY_V2_ENABLED", True)


def daily_continuity_v2_projection_enabled() -> bool:
    """Project validated v2 items alongside the fixed daily recap."""
    return (
        daily_continuity_v2_enabled()
        and _env_enabled("DAILY_CONTINUITY_V2_READ_ENABLED", True)
        and not _env_enabled("DAILY_CONTINUITY_V2_SHADOW_ONLY", False)
    )


def _continuity_v2_file_path(session_id: str, date_str: str) -> str:
    from tsuzuri.daily_continuity import capsule_path_for_archive

    return str(capsule_path_for_archive(_archive_file_path(session_id, date_str)))


# ── Rule-based strip ─────────────────────────────────────────
_ELLIPSIS_ONLY_RE = re.compile(r"^[·\.…—\s]*[·\.…—]{3,}[·\.…—\s？！?!]*$")
_HUM_ONLY_RE = re.compile(r"^[\s]*(嗯|啊|哦|诶|唉|呃)[\s，。．、！？!?]*$")


def _rule_based_strip(history: list[dict]) -> list[dict]:
    """去除低信号 turn：

    · 角色B式纯 ······ / ············ bubble
    · 嗯-only / 啊-only / 哦-only 短回应
    · proactive 标记的 auto_greet（信号低于用户主动 turn）
    · timestamp 缺失 / role 异常的脏数据
    """
    out = []
    for m in history:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content", "") or "").strip()
        if not content:
            continue
        # 角色B式纯省略号 / 助词 only
        if role == "assistant":
            if _ELLIPSIS_ONLY_RE.match(content) or _HUM_ONLY_RE.match(content):
                continue
            # proactive auto_greet（来自 _web_auto_greet_loop）信号低于用户响应、跳过
            if m.get("proactive") and m.get("source") == "web_auto_greet":
                continue
            if m.get("auto_greet"):
                continue
        out.append(m)
    return out


# ── LLM summarize（5 段 schema）──────────────────────────────
_COMPACT_PROMPT_TEMPLATE = """TASK=CHARACTER_DAILY_RECAP_MAP_V1
你的任务：把下面这一天（{date_world}）里，{character} 和"{user}"的聊天对话浓缩成一份**结构化摘要**。

⚠️ 重要规则：
- **只输出文本**、严禁调用任何工具。
- 这是日级事实叙事层，不是角色心理分析或关系裁判。只写可核对的发言、行动、明确状态和承诺，不替任何人判断内心、动机或关系性质；二次关系定性只能归入开发者专属印象，由 KAIROS 每周关系反思维护。
- 严禁逐字照抄原话进 summary（用户名称除外）；用你自己的话提炼。
- 严禁编造对话里没发生过的事件。
- 遇到“不是 X、是 Y”“刚才理解错了”一类明确纠正，只保留纠正后的 Y；被否定的 X 不得作为事实残留。
- 关系变化必须有双方直接表达或明确确认；普通关心、玩笑、称赞、没有否认都不能推断为暧昧或关系升级。
- 禁止用“未拒绝也未接受”“进入暧昧/边界模糊状态”“朋友以上”替代原话作关系定性；若原话是“作为朋友的喜欢，还有一点别的”，只如实记录这一表达。
- 可以概括当时可观察的对话气氛或语气变化，但必须写成“对话气氛显得…… / {character}语气从……变为……”，不能把语气推断成内心结论。
- 严格区分{user}、{character} 和第三人的主体。邀请、祝愿、提问、假设不能改写成对方的事实或计划。
- 承诺必须写清责任人、条件和时间；“如果……就……”是条件式承诺，不是当天日程，也不能猜测为已经完成。
- 兼顾关键与日常：纠正、关系、承诺、待续优先保全，同时保留能支撑次日自然连续性的具体活动、偏好和环境细节。
- 不确定是哪一天发生的就不提日期。
- 这是过去某一天的摘要，所有临时状态必须写成「当时/那天」语气；不要写「目前」「现在」
  「当前」「正在」来暗示它仍然发生在读摘要的此刻。

对话内容（按时序）：
————————————————————————————————
{conversation_text}
————————————————————————————————

请按以下格式输出：

<analysis>
（在这里梳理你的思路、对话里发生的事、你打算放进每个 section 的要点。这部分不会进入最终 summary、只是给你想清楚的草稿区。）
</analysis>

<summary>
1. 重要事件：（这一天发生过的具体事件、行动、决定。每条 1-2 句、不要罗列鸡毛蒜皮。如果没有显著事件、写"无显著事件"。）

2. 关系状态变化：（只记录双方直接说出的关系相关表达、明确误解纠正或公开确认；不要总结关系性质、亲密度、信任度或替角色下判断。没有直接表达就写"无明显变化"。）

3. ongoing 话题：（聊到一半没说完的、明显还会继续的话题。每条 1 句。无则写"无"。）

4. {user}透露的事实：（{user}提到的自己的事 — 喜好、状态、计划、过去经历等。每条 1 句。无则写"无"。）

5. {character} 的情绪 / 承诺：（{character} 当天的情绪底色、对{user}说要做的事、答应过的承诺等。每条 1 句。无则写"无"。）
</summary>

⚠️ 全文必须使用简体中文。整个 <summary> 输出控制在 {target_char_budget} 字以内（含标题）。
"""


_COMPACT_MAX_TURNS = 150  # 单日 prompt 最多塞 150 条 turn（防 prompt 爆模型 context）


def _build_compact_prompt(character: str, date_world: str, history: list[dict], target_char_budget: int) -> str:
    """构造给 LLM 的 compact prompt。

    2026-05-21 hardening：history > _COMPACT_MAX_TURNS 时只取最后 150 条、加前缀提示
    LLM 较早内容已省略（保留近期更重要的对话、压缩这一天的"尾声"）。
    """
    char_label = str(character or "角色")
    skipped_marker = ""
    if len(history) > _COMPACT_MAX_TURNS:
        _skipped_count = len(history) - _COMPACT_MAX_TURNS
        history = history[-_COMPACT_MAX_TURNS:]
        skipped_marker = f"[注：当天较早的 {_skipped_count} 条对话已省略、下面是最后 {_COMPACT_MAX_TURNS} 条]\n"
    _lines = []
    for m in history:
        role = str(m.get("role", "") or "").strip().lower()
        content = str(m.get("content", "") or "").strip()
        if not content:
            continue
        if role == "user":
            _lines.append(f"{user_name()}：{content}")
        else:
            _lines.append(f"{char_label}：{content}")
    conversation_text = skipped_marker + "\n".join(_lines)
    return _COMPACT_PROMPT_TEMPLATE.format(
        user=user_name(),
        date_world=str(date_world),
        character=char_label,
        conversation_text=conversation_text,
        target_char_budget=int(target_char_budget),
    )


_ANALYSIS_RE = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>", re.IGNORECASE)


def _extract_summary_content(raw: str) -> str:
    """从 LLM 输出抽 <summary> 内容、strip 掉 <analysis> 草稿区。"""
    if not raw:
        return ""
    s = str(raw)
    # 严格解 <summary>...</summary>
    m = _SUMMARY_RE.search(s)
    if m:
        return m.group(1).strip()
    # 兜底：strip <analysis>、剩下当 summary
    s = _ANALYSIS_RE.sub("", s).strip()
    # strip 最外层 tags
    s = re.sub(r"^<summary>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</summary>$", "", s, flags=re.IGNORECASE)
    return s.strip()


def compact_session_for_date(
    session_id: str,
    date_world: str,
    *,
    character: str = "",
    llm_caller: Callable[[str], str] | None = None,
    target_char_budget: int = 500,
    force: bool = False,
) -> dict[str, Any]:
    """对某天的 archive 做语义压缩、写 summary.json。

    Args:
        session_id: 会话 id
        date_world: YYYY-MM-DD（被压缩的日期）
        character: 角色名（archive 内已有、传值为可选 override）
        llm_caller: 同步函数 prompt → response 文本
        target_char_budget: summary 目标字数上限（默认 500 字，含 5 个段标题）
        force: True 时即使 summary 已存在也重新生成

    Returns:
        {"ok": bool, "skipped": bool, "summary_path": str, "reason": str}
    """
    archive_path = _archive_file_path(session_id, date_world)
    if not os.path.exists(archive_path):
        return {"ok": False, "skipped": True, "summary_path": "", "reason": "archive_missing"}
    summary_path = _summary_file_path(session_id, date_world)
    if os.path.exists(summary_path) and not force:
        return {"ok": True, "skipped": True, "summary_path": summary_path, "reason": "already_exists"}
    try:
        with open(archive_path, "r", encoding="utf-8") as _f:
            archive = json.load(_f)
    except Exception as _e:
        return {"ok": False, "skipped": False, "summary_path": "", "reason": f"archive_read_fail:{_e}"}
    if not isinstance(archive, dict):
        return {"ok": False, "skipped": False, "summary_path": "", "reason": "archive_invalid"}
    history = archive.get("history", [])
    if not isinstance(history, list):
        history = []
    history = _developer_history_view(history)
    char = str(character or archive.get("character", "") or "")
    stripped = _rule_based_strip(history)
    if not stripped:
        # 空对话日：写一个 empty summary 标记日期已处理、避免下次反复尝试
        payload = {
            "session_id": session_id,
            "character": char,
            "date_world": date_world,
            "compacted_at_ms": int(time.time() * 1000),
            "method": "rule_based_empty",
            "summary": "",
            "raw_message_count": len(history),
            "stripped_message_count": 0,
        }
        try:
            os.makedirs(_archive_dir_for_date(date_world), exist_ok=True)
            tmp = summary_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as _f:
                json.dump(payload, _f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, summary_path)
            return {"ok": True, "skipped": False, "summary_path": summary_path, "reason": "empty_after_strip"}
        except Exception as _e:
            return {"ok": False, "skipped": False, "summary_path": "", "reason": f"write_fail:{_e}"}
    if not callable(llm_caller):
        return {"ok": False, "skipped": False, "summary_path": "", "reason": "no_llm_caller"}
    prompt = _build_compact_prompt(char, date_world, stripped, target_char_budget)
    try:
        raw = llm_caller(prompt) or ""
    except Exception as _e:
        return {"ok": False, "skipped": False, "summary_path": "", "reason": f"llm_call_fail:{_e}"}
    summary_text = _extract_summary_content(raw)
    if not summary_text:
        return {"ok": False, "skipped": False, "summary_path": "", "reason": "empty_summary"}
    payload = {
        "session_id": session_id,
        "character": char,
        "date_world": date_world,
        "compacted_at_ms": int(time.time() * 1000),
        "method": "rule_strip_then_llm_5section",
        "summary": summary_text,
        "raw_message_count": len(history),
        "stripped_message_count": len(stripped),
        "target_char_budget": target_char_budget,
    }
    try:
        os.makedirs(_archive_dir_for_date(date_world), exist_ok=True)
        tmp = summary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as _f:
            json.dump(payload, _f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, summary_path)
        return {"ok": True, "skipped": False, "summary_path": summary_path, "reason": "ok"}
    except Exception as _e:
        return {"ok": False, "skipped": False, "summary_path": "", "reason": f"write_fail:{_e}"}


_CHARACTER_DAILY_RECAP_PROMPT = """TASK=CHARACTER_DAILY_RECAP_AGGREGATE_V1
把 {date_world} 这一天里，{character} 与{user}分散在多个 chat session 的摘要，归并为一份角色日级对话回顾。

这些输入是同一天不同窗口的 map 摘要。最终输出必须是单一信息源，严禁按“长会话 / 日常会话 / session”拆分；保留既有 v1 五段标签，不另造分类体系。

裁决规则：
1. 合并语义重复项；明确的“不是 X、是 Y”纠正优先于旧说法，旧值不得残留。
2. 只保留输入直接支持的事实。祝愿、提问、假设、沉默或未否认不能建立计划、关系或承诺。
   本摘要是事实叙事层，不是角色心理分析或关系裁判；主观二次关系定性只能归入开发者专属印象，由 KAIROS 每周关系反思维护，不在这里生成或写回事实层。
3. 关系和情感变化必须有直接表达；普通关心、玩笑、亲近语气不能推断成关系升级。
   禁止归纳成“未拒绝也未接受”“进入暧昧/边界模糊状态”“朋友以上”；只写双方各自直接表达了什么。
   可以描述可观察的对话气氛和语气变化，但不能据此断言内心、动机、信任或亲密程度。
4. 承诺必须保留责任人、条件和时间。条件式承诺不能写成当天日程；计划不能猜测为已完成。
   玩笑、假设和泛泛的“以后再聊”不是承诺或 ongoing 事项。
5. 区分{user}与 {character} 的事实，禁止交换主体。临时活动统一用“那天 / 当时”，不能暗示此刻仍在持续。
6. 在准确的前提下提高信息密度：关系、纠正、承诺和待续不因日常细节被截断，具体日常脉络也不能完全丢失。
7. 同一事实只放入最合适的一段，不在“重要事件 / 关系 / ongoing / 承诺”间重复改写。ongoing 只保留有具体未决问题、条件或后续动作的事项。

输入摘要：
————————————————————————————————
{summary_text}
————————————————————————————————

仅按以下格式输出：

<analysis>
（先跨 session 去重、裁决纠正、核对主体和承诺条件；此段不会进入角色 prompt。）
</analysis>

<summary>
1. 重要事件：（跨全部 session 合并当天具体事件、行动和明确决定；没有则写“无显著事件”。）

2. 关系状态变化：（只记录双方直接说出的关系相关表达、明确误解纠正或公开确认；禁止总结关系性质、亲密度或信任度。没有直接表达则写“无明显变化”。）

3. ongoing 话题：（未说完、尚待确认或明确会继续的话题；没有则写“无”。）

4. {user}透露的事实：（喜好、状态、计划、经历等，保留必要时间和条件；没有则写“无”。）

5. {character} 的情绪 / 承诺：（当天情绪底色、明确承诺及其条件/时间；没有则写“无”。）
</summary>

全文使用简体中文，<summary> 必须控制在 {target_char_budget} 字以内，超过即视为失败。不要输出 session 名、证据 ID、分析过程或额外标题。
"""


_CHARACTER_DAILY_RECAP_REPAIR_PROMPT = """TASK=CHARACTER_DAILY_RECAP_REPAIR_V1
下面这份角色日级回顾没有通过事实叙事质量门，请在不增加任何新信息的前提下修复。

失败项：{quality_issues}

待修复摘要：
————————————————————————————————
{draft_summary}
————————————————————————————————

硬规则：
- 只保留可核对的发言、行动、明确状态与承诺；删除对内心、动机、关系性质、亲密度和信任度的解释。
- “未拒绝也未接受”“朋友以上”“边界模糊”等二次关系定性只能归入开发者专属印象，由 KAIROS 每周关系反思维护；本摘要必须删除。
- 删除“暗示/体现/反映出/说明了/可以看出”等解释性转述，改为原事实或直接删除。
- 请求、建议、玩笑、自然结果和泛泛的“以后再聊”不是承诺。
- 保留既有五段标签，每个事实只出现一次；不得增加原摘要没有的事实。
- 可保留“对话气氛显得…… / 语气从……变为……”这类可观察氛围描述。

仅输出 `<summary>...</summary>`，其中必须依次包含：
1. 重要事件：
2. 关系状态变化：
3. ongoing 话题：
4. {user}透露的事实：
5. {character} 的情绪 / 承诺：

<summary> 必须不超过 {target_char_budget} 字，超过即视为失败。
"""


_DAILY_RECAP_INTERPRETATION_PATTERNS = (
    re.compile(r"未拒绝也未接受|朋友以上|边界模糊|亲密与信任"),
    re.compile(r"(?:关系|双方).{0,12}(?:进入|变得|处于).{0,12}(?:状态|阶段|关系)"),
    re.compile(r"(?:暗示|体现|反映出|可以看出|由此可见)"),
    re.compile(r"属于.{0,10}(?:承诺|关系|接受|拒绝)"),
)


def _daily_recap_char_budget(target_char_budget: int) -> int:
    """生效的目标字数（仍是生成侧目标、不是失败门禁，见模块 docstring 2026-09-11 决定）。"""
    return max(400, int(target_char_budget or 900))


def _daily_recap_quality_issues(summary: str, *, target_char_budget: int) -> list[str]:
    """返回**致命**质量问题：五段 schema 缺失、解释性措辞。

    长度不再列入致命项（2026-09-11 决定）：超预算由注入侧 `_truncate_recap_summary`
    按五段权重截断，生成侧只在 payload 里记 `over_budget`；`target_char_budget`
    参数保留以兼容既有调用点。
    """
    text = str(summary or "").strip()
    issues: list[str] = []
    required_sections = (
        "1. 重要事件",
        "2. 关系状态变化",
        "3. ongoing 话题",
        f"4. {user_name()}透露的事实",
        "5.",
    )
    if any(label not in text for label in required_sections):
        issues.append("missing_five_section_schema")
    for pattern in _DAILY_RECAP_INTERPRETATION_PATTERNS:
        match = pattern.search(text)
        if match:
            issues.append(f"interpretive_language:{match.group(0)[:40]}")
    return issues


def _iter_character_day_summary_maps(character: str, date_world: str) -> tuple[list[str], int, list[str]]:
    """Return completed per-session map summaries for one character/day."""
    target = str(character or "").strip()
    archive_dir = Path(_archive_dir_for_date(date_world))
    if not target or not archive_dir.exists():
        return [], 0, []
    summaries: list[str] = []
    source_count = 0
    missing: list[str] = []
    for archive_path in sorted(archive_dir.glob("*.json")):
        if archive_path.name.endswith(
            (".summary.json", ".continuity.v2.json", ".retry.json", ".checkpoint.json")
        ):
            continue
        # Work sessions are event-sourced in work_agent_runtime SQLite and never
        # participate in chat compact accounting (audit/backfill skip them too);
        # a legacy -work- archive without summary must not block the recap gate.
        if "-work-" in archive_path.name:
            continue
        # QQ 群会话每天用完即丢：不进角色级聚合（群的压缩记录=1v1 同款角色级 recap），
        # 也避免群快照缺 summary 时卡住 recap gate。
        if archive_path.name.startswith("qq-group-"):
            continue
        try:
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(archive, dict):
            continue
        if str(archive.get("character", "") or "").strip() != target:
            continue
        source_count += 1
        summary_path = archive_path.with_name(f"{archive_path.stem}.summary.json")
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            missing.append(archive_path.stem)
            continue
        summary = str(payload.get("summary", "") or "").strip() if isinstance(payload, dict) else ""
        if summary:
            summaries.append(summary)
    return summaries, source_count, missing


def compact_character_daily_recap(
    character: str,
    date_world: str,
    *,
    llm_caller: Callable[[str], str] | None = None,
    target_char_budget: int = 900,
    force: bool = False,
) -> dict[str, Any]:
    """Reduce all per-session map summaries into one character/day recap."""
    target = str(character or "").strip()
    output_path = _character_daily_summary_file_path(target, date_world)
    summaries, source_count, missing = _iter_character_day_summary_maps(target, date_world)
    if not target or source_count <= 0:
        return {"ok": False, "skipped": True, "summary_path": "", "reason": "no_character_archives"}
    if missing:
        return {
            "ok": False,
            "skipped": True,
            "summary_path": output_path,
            "reason": "incomplete_session_summaries",
            "missing_session_ids": missing,
        }
    fingerprint_payload = json.dumps(
        {"character": target, "date_world": date_world, "summaries": summaries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    source_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    if os.path.exists(output_path) and not force:
        try:
            existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
            if existing.get("source_fingerprint") == source_fingerprint:
                return {
                    "ok": True,
                    "skipped": True,
                    "summary_path": output_path,
                    "reason": "idempotent",
                }
        except Exception:
            pass
    if not summaries:
        summary_text = ""
        method = "character_day_empty"
        model_call_count = 0
    else:
        if not callable(llm_caller):
            return {"ok": False, "skipped": False, "summary_path": "", "reason": "no_llm_caller"}
        budget = _daily_recap_char_budget(target_char_budget)
        prompt = _CHARACTER_DAILY_RECAP_PROMPT.format(
            user=user_name(),
            character=target,
            date_world=str(date_world),
            summary_text="\n\n".join(
                f"[对话片段 {index}]\n{summary}"
                for index, summary in enumerate(summaries, start=1)
            ),
            target_char_budget=budget,
        )
        try:
            summary_text = _extract_summary_content(llm_caller(prompt) or "")
        except Exception as exc:
            return {
                "ok": False,
                "skipped": False,
                "summary_path": "",
                "reason": f"llm_call_fail:{exc}",
            }
        if not summary_text:
            return {"ok": False, "skipped": False, "summary_path": "", "reason": "empty_summary"}
        method = "character_day_session_summary_reduce_v1"
        model_call_count = 1
        quality_issues = _daily_recap_quality_issues(
            summary_text,
            target_char_budget=target_char_budget,
        )
        if quality_issues:
            repair_prompt = _CHARACTER_DAILY_RECAP_REPAIR_PROMPT.format(
                user=user_name(),
                quality_issues="；".join(quality_issues),
                draft_summary=summary_text,
                character=target,
                target_char_budget=budget,
            )
            try:
                repaired = _extract_summary_content(llm_caller(repair_prompt) or "")
            except Exception as exc:
                return {
                    "ok": False,
                    "skipped": False,
                    "summary_path": output_path,
                    "reason": f"repair_call_fail:{exc}",
                }
            model_call_count += 1
            remaining_issues = _daily_recap_quality_issues(
                repaired,
                target_char_budget=target_char_budget,
            )
            if remaining_issues:
                return {
                    "ok": False,
                    "skipped": False,
                    "summary_path": output_path,
                    "reason": "quality_validation_failed",
                    "quality_issues": remaining_issues,
                    "summary_chars": len(str(repaired or "")),
                }
            summary_text = repaired
            method = "character_day_session_summary_reduce_repair_v1"
    summary_chars = len(summary_text)
    char_budget = _daily_recap_char_budget(target_char_budget)
    over_budget = summary_chars > char_budget
    payload = {
        "character": target,
        "date_world": str(date_world),
        "compacted_at_ms": int(time.time() * 1000),
        "method": method,
        "summary": summary_text,
        "summary_chars": summary_chars,
        "over_budget": over_budget,
        "source_session_count": source_count,
        "source_summary_count": len(summaries),
        "source_fingerprint": source_fingerprint,
        "target_char_budget": char_budget,
        "model_call_count": model_call_count,
    }
    try:
        os.makedirs(_archive_dir_for_date(date_world), exist_ok=True)
        tmp_path = output_path + ".tmp"
        Path(tmp_path).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(tmp_path, output_path)
    except Exception as exc:
        return {"ok": False, "skipped": False, "summary_path": "", "reason": f"write_fail:{exc}"}
    return {
        "ok": True,
        "skipped": False,
        "summary_path": output_path,
        "reason": "ok",
        "summary_chars": summary_chars,
        "over_budget": over_budget,
    }


def compact_session_for_date_v2(
    session_id: str,
    date_world: str,
    *,
    llm_caller: Callable[[str], str] | None = None,
    force: bool = False,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Build a versioned continuity sidecar without changing the v1 summary."""
    if not daily_continuity_v2_enabled():
        return {"ok": True, "skipped": True, "capsule_path": "", "reason": "v2_disabled"}
    archive_path = _archive_file_path(session_id, date_world)
    if not os.path.exists(archive_path):
        return {"ok": False, "skipped": True, "capsule_path": "", "reason": "archive_missing"}
    if not callable(llm_caller):
        return {"ok": False, "skipped": False, "capsule_path": "", "reason": "no_llm_caller"}
    try:
        with open(archive_path, "r", encoding="utf-8") as handle:
            archive = json.load(handle)
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "capsule_path": "",
            "reason": f"archive_read_fail:{exc}",
        }
    if not isinstance(archive, dict):
        return {"ok": False, "skipped": False, "capsule_path": "", "reason": "archive_invalid"}

    from tsuzuri.daily_continuity import (
        ContinuityStorage,
        DailyContinuityCompactor,
        retry_path_for_capsule,
    )
    from tsuzuri.daily_continuity.storage import compute_archive_sha256

    capsule_path = _continuity_v2_file_path(session_id, date_world)
    retry_path = str(retry_path_for_capsule(capsule_path))
    storage = ContinuityStorage()
    archive_sha = compute_archive_sha256(archive)
    existing = storage.load_matching_capsule(
        capsule_path,
        archive_sha,
    ) if not force else None
    if existing is None and not force and not storage.retry_is_due(
        retry_path,
        now_ms=now_ms,
        archive_sha256=archive_sha,
    ):
        return {
            "ok": False,
            "skipped": True,
            "capsule_path": capsule_path,
            "reason": "retry_deferred",
            "retry_state": storage.load_retry_state(retry_path),
        }
    result = DailyContinuityCompactor(storage=storage).compact_archive(
        archive,
        llm_caller=llm_caller,
        capsule_path=capsule_path,
        retry_state_path=retry_path,
        force=force,
        now_ms=now_ms,
    )
    capsule = result.capsule
    return {
        "ok": bool(result.ok),
        "skipped": result.status == "idempotent",
        "capsule_path": capsule_path,
        "reason": result.status if result.ok else result.reason,
        "archive_sha256": result.archive_sha256,
        "retry_state": dict(result.retry_state),
        "quality": dict(capsule.quality) if capsule is not None else {},
        "item_count": len(capsule.items) if capsule is not None else 0,
    }


def _load_matching_continuity_capsule(session_id: str, date_world: str) -> dict[str, Any] | None:
    archive_path = _archive_file_path(session_id, date_world)
    if not os.path.exists(archive_path):
        return None
    try:
        with open(archive_path, "r", encoding="utf-8") as handle:
            archive = json.load(handle)
        if not isinstance(archive, dict):
            return None
        from tsuzuri.daily_continuity import ContinuityStorage
        from tsuzuri.daily_continuity.storage import compute_archive_sha256

        capsule = ContinuityStorage().load_matching_capsule(
            _continuity_v2_file_path(session_id, date_world),
            compute_archive_sha256(archive),
        )
        return capsule.to_dict() if capsule is not None else None
    except Exception:
        return None


def load_daily_continuity_capsules(
    session_id: str,
    *,
    today_date_world: str = "",
    min_age_days: int = 1,
    max_age_days: int = 14,
) -> list[dict[str, Any]]:
    """Load only capsules whose schema and source archive SHA still match."""
    sid = str(session_id or "").strip()
    if not sid:
        return []
    today = str(today_date_world).strip() or _today_world_date_str()
    start = max(1, int(min_age_days or 1))
    end = max(start, int(max_age_days or start))
    capsules: list[dict[str, Any]] = []
    for age in range(start, end + 1):
        date_world = _date_str_add_days(today, -age)
        capsule = _load_matching_continuity_capsule(sid, date_world)
        if capsule is not None:
            capsules.append(capsule)
    return capsules


def load_daily_continuity_capsules_for_character(
    character: str,
    *,
    today_date_world: str = "",
    min_age_days: int = 15,
    max_age_days: int = 45,
) -> list[dict[str, Any]]:
    """Load matching capsules across every session owned by one character."""
    target = str(character or "").strip()
    if not target:
        return []
    today = str(today_date_world).strip() or _today_world_date_str()
    start = max(0, int(min_age_days if min_age_days is not None else 15))
    end = max(start, int(max_age_days or start))
    from tsuzuri.daily_continuity import ContinuityStorage
    from tsuzuri.daily_continuity.storage import capsule_path_for_archive, compute_archive_sha256

    storage = ContinuityStorage()
    capsules: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for age in range(start, end + 1):
        date_world = _date_str_add_days(today, -age)
        archive_dir = Path(_archive_dir_for_date(date_world))
        if not archive_dir.exists():
            continue
        for archive_path in sorted(archive_dir.glob("*.json")):
            if archive_path.name.endswith(
                (".summary.json", ".continuity.v2.json", ".retry.json")
            ):
                continue
            # QQ 群会话用完即丢：群 capsule 不进角色级胶囊集（与 1v1 保持同一来源）
            if archive_path.name.startswith("qq-group-"):
                continue
            try:
                with archive_path.open("r", encoding="utf-8") as handle:
                    archive = json.load(handle)
                if not isinstance(archive, dict):
                    continue
                if str(archive.get("character", "") or "").strip() != target:
                    continue
                source_sha = compute_archive_sha256(archive)
                if source_sha in seen_sources:
                    continue
                capsule = storage.load_matching_capsule(
                    capsule_path_for_archive(archive_path),
                    source_sha,
                )
                if capsule is None:
                    continue
                seen_sources.add(source_sha)
                capsules.append(capsule.to_dict())
            except Exception:
                continue
    return capsules


def list_due_daily_continuity_retries(
    *,
    now_ms: int | None = None,
    max_jobs: int = 4,
) -> list[dict[str, str]]:
    """Discover persisted retry jobs without relying on archive catch-up state."""
    if not daily_continuity_v2_enabled():
        return []
    from tsuzuri.daily_continuity import ContinuityStorage

    storage = ContinuityStorage()
    jobs: list[dict[str, str]] = []
    root = Path(_CHAT_SESSIONS_ARCHIVE_DIR)
    if not root.exists():
        return jobs
    for retry_path in sorted(root.glob("*/*.continuity.v2.json.retry.json")):
        retry_state = storage.load_retry_state(retry_path)
        if retry_state.get("status") != "retryable_failure":
            continue
        archive_name = retry_path.name.removesuffix(".continuity.v2.json.retry.json") + ".json"
        archive_path = retry_path.with_name(archive_name)
        try:
            with archive_path.open("r", encoding="utf-8") as handle:
                archive = json.load(handle)
        except Exception:
            continue
        if not isinstance(archive, dict):
            continue
        from tsuzuri.daily_continuity.storage import compute_archive_sha256

        if not storage.retry_is_due(
            retry_path,
            now_ms=now_ms,
            archive_sha256=compute_archive_sha256(archive),
        ):
            continue
        session_id = str(archive.get("session_id", "") or "").strip()
        date_world = str(archive.get("date_world", "") or retry_path.parent.name).strip()
        character = str(archive.get("character", "") or "").strip()
        if session_id and date_world:
            jobs.append({"session_id": session_id, "date_world": date_world, "character": character})
        if len(jobs) >= max(1, int(max_jobs or 1)):
            break
    return jobs


# ── 衰减 merge：load_historical_recap ─────────────────────────
def _load_summary_for_date(session_id: str, date_world: str) -> dict[str, Any] | None:
    path = _summary_file_path(session_id, date_world)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as _f:
            data = json.load(_f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _load_recap_source_for_date(
    session_id: str,
    date_world: str,
    *,
    character: str = "",
) -> dict[str, Any] | None:
    """Prefer the cross-session character/day recap, then fail back to v1 map output."""
    target = str(character or "").strip()
    if target:
        path = _character_daily_summary_file_path(target, date_world)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and str(data.get("character", "") or "").strip() == target:
                return data
        except Exception:
            pass
    # QQ 群会话没有自有压缩记录（每天用完即丢），只读上面的角色级聚合
    if str(session_id or "").startswith("qq-group-"):
        return None
    return _load_summary_for_date(session_id, date_world)


def _truncate_text(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if len(s) <= max_chars:
        return s
    if max_chars <= 6:
        return s[:max_chars]
    # 在 max_chars 内找最后一个段落 / 句末标点
    _slice = s[: max_chars - 1]
    for sep in ("\n\n", "\n", "。", "！", "？", "."):
        idx = _slice.rfind(sep)
        if idx >= int(max_chars * 0.5):
            return _slice[: idx + len(sep)].rstrip() + "…"
    return _slice.rstrip() + "…"


_RECAP_SECTION_RE = re.compile(r"(?m)^\s*([1-5])\.\s*([^\n：:]*?)[：:]\s*")
_RECAP_SECTION_LABELS = {
    "1": "1.事件：",
    "2": "2.关系：",
    "3": "3.待续：",
    "4": "4.{user}事实：",
    "5": "5.情绪/承诺：",
}
_RECAP_SECTION_WEIGHTS = {"1": 0.20, "2": 0.12, "3": 0.18, "4": 0.25, "5": 0.25}


def _recap_section_label(number: str) -> str:
    return _RECAP_SECTION_LABELS[number].replace("{user}", user_name())


def _truncate_recap_summary(text: str, max_chars: int) -> str:
    """Keep all five legacy fields represented instead of prefix-starving later fields."""
    source = str(text or "").strip()
    matches = list(_RECAP_SECTION_RE.finditer(source))
    if len(matches) < 2:
        return _truncate_text(source, max_chars)
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        sections.append((match.group(1), source[match.end():end].strip()))
    labels_cost = sum(len(_recap_section_label(number)) for number, _ in sections)
    labels_cost += max(0, len(sections) - 1)
    if labels_cost >= max_chars:
        return _truncate_text(source, max_chars)
    body_budget = max_chars - labels_cost
    total_weight = sum(_RECAP_SECTION_WEIGHTS.get(number, 0.2) for number, _ in sections)
    lines: list[str] = []
    remaining = body_budget
    for index, (number, body) in enumerate(sections):
        if index == len(sections) - 1:
            allocation = remaining
        else:
            allocation = max(
                1,
                int(body_budget * _RECAP_SECTION_WEIGHTS.get(number, 0.2) / total_weight),
            )
            remaining -= allocation
        lines.append(
            _recap_section_label(number) + _truncate_text(body or "无", allocation)
        )
    return "\n".join(lines)[:max_chars]


def _normalize_recap_temporal_language(text: str) -> str:
    """Reduce stale-current wording in archived summaries before prompt injection."""
    s = str(text or "")
    if not s:
        return ""
    _user = user_name()
    replacements = {
        f"{_user}目前": f"{_user}当时",
        f"{_user}现在": f"{_user}当时",
        f"{_user}当前": f"{_user}当时",
        "用户目前": "用户当时",
        "用户现在": "用户当时",
        "用户当前": "用户当时",
        "开发者目前": "开发者当时",
        "开发者现在": "开发者当时",
        "开发者当前": "开发者当时",
        "目前处于": "当时处于",
        "现在处于": "当时处于",
        "当前处于": "当时处于",
        "目前在": "当时在",
        "现在在": "当时在",
        "当前在": "当时在",
        "目前只是": "当时只是",
        "现在只是": "当时只是",
        "当前只是": "当时只是",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def load_historical_recap(
    session_id: str,
    *,
    today_date_world: str = "",
    character: str = "",
    yesterday_budget: int = 900,
    midrange_budget_per_day: int = 1200,
    weekly_budget: int = 1800,
    exclude_dates: set[str] | None = None,
) -> str:
    """构造 historical_chat_recap block 内容。

    衰减规则（2026-05-21 用户拍板）：
      · 昨日（today - 1）：角色日级聚合 summary、cap 900 字
      · 2-7 天前：6 天 × 1200 字、合并 1 个 sub-section
      · 1-2 周前（8-14 天）：按周聚合 1800 字、合并 1 个 sub-section
      · > 14 天：不注入

    若所有 sub-section 都为空、整体返 ""（caller 的 budget mechanism 会跳过）。
    """
    sid = str(session_id or "").strip()
    if not sid:
        return ""
    today = str(today_date_world).strip() or _today_world_date_str()
    excluded = {str(value) for value in (exclude_dates or set()) if str(value)}
    sections: list[str] = []

    # ─ 昨日 ─
    yesterday = _date_str_add_days(today, -1)
    y_data = None if yesterday in excluded else _load_recap_source_for_date(
        sid, yesterday, character=character
    )
    if y_data and isinstance(y_data, dict):
        y_text = _normalize_recap_temporal_language(str(y_data.get("summary", "") or "").strip())
        if y_text:
            sections.append(
                f"━━ 昨天（{yesterday}）的对话回顾 ━━\n{_truncate_recap_summary(y_text, yesterday_budget)}"
            )

    # ─ 2-7 天前（合并 1 段）─
    midrange_lines: list[str] = []
    for n in range(2, 8):
        d = _date_str_add_days(today, -n)
        if d in excluded:
            continue
        data = _load_recap_source_for_date(sid, d, character=character)
        if not (data and isinstance(data, dict)):
            continue
        text = _normalize_recap_temporal_language(str(data.get("summary", "") or "").strip())
        if not text:
            continue
        snippet = _truncate_recap_summary(text, midrange_budget_per_day)
        midrange_lines.append(f"· {d}（{n} 天前）：{snippet}")
    if midrange_lines:
        sections.append("━━ 最近一周的对话要点 ━━\n" + "\n".join(midrange_lines))

    # ─ 1-2 周前（按周聚合）─
    week_lines: list[str] = []
    week_buckets: dict[int, list[str]] = {1: [], 2: []}  # 1 = 一周前，2 = 两周前
    for n in range(8, 15):
        d = _date_str_add_days(today, -n)
        if d in excluded:
            continue
        data = _load_recap_source_for_date(sid, d, character=character)
        if not (data and isinstance(data, dict)):
            continue
        text = _normalize_recap_temporal_language(str(data.get("summary", "") or "").strip())
        if not text:
            continue
        bucket = 1 if n <= 10 else 2  # 8-10 天 = "一周多前"，11-14 天 = "两周前"
        snippet = _truncate_recap_summary(text, midrange_budget_per_day)
        week_buckets[bucket].append(f"· {d}：{snippet}")
    if week_buckets[1] or week_buckets[2]:
        if week_buckets[1]:
            week_lines.append("[一周多前]")
            week_lines.append(_truncate_text("\n".join(week_buckets[1]), weekly_budget))
        if week_buckets[2]:
            week_lines.append("[两周前]")
            week_lines.append(_truncate_text("\n".join(week_buckets[2]), weekly_budget))
        sections.append("━━ 更早的对话碎片 ━━\n" + "\n".join(week_lines))

    if not sections:
        return ""

    # 2026-05-21 强 fence（迁移自旧 daily_state「昨天对话回顾」block 的 4 条铁律）：
    header = (
        "【历史对话回顾（只读、context 参考、不是你今天该说出来的话）】\n"
        f"以下是过去几天和{user_name()}聊过的内容**浓缩摘要**——保持人格连续度的 context。\n"
        "硬规则（绝对遵守、违反 = 出戏 / 幻觉）：\n"
        "  · ⚠️ 绝对禁止编造下面摘要里没出现过的事件——『昨天你抱怨过 X』『前几天聊过 Y』、\n"
        "    X/Y 必须在下文摘要里找得到才能引用、否则就是幻觉。\n"
        "  · ⚠️ 不是今天发生的事——下面任何内容**不要**说成『今天 / 现在 / 刚才』正在发生、\n"
        "    要带时间锚就用『昨天』『前几天』『上周』、且必须真的对应下文日期标签。\n"
        "  · ⚠️ 不确定是哪天发生的话题、宁可**不带时间锚**直接接当下话题、也不要硬安一个『昨天』。\n"
        "  · ⚠️ 严禁把摘要里的文字逐字复制进回复——这是 LLM 视角的 context、不是你的台词。\n"
        "    要延续话题就用全新的措辞、像本来就记得这件事一样自然带过去。\n"
    )
    return header + "\n\n".join(sections)


def load_historical_context_bundle(
    session_id: str,
    *,
    today_date_world: str = "",
    character: str = "",
) -> dict[str, Any]:
    """Keep one fixed character/day recap while also exposing v2 for selection."""
    today = str(today_date_world).strip() or _today_world_date_str()
    if not daily_continuity_v2_projection_enabled():
        return {
            "legacy_recap": load_historical_recap(
                session_id, today_date_world=today, character=character
            ),
            "capsules": [],
            "covered_dates": [],
            "projection_enabled": False,
        }
    session_capsules = load_daily_continuity_capsules(
        session_id,
        today_date_world=today,
        min_age_days=1,
        max_age_days=45,
    )
    capsules = (
        load_daily_continuity_capsules_for_character(
            character,
            today_date_world=today,
            min_age_days=1,
            max_age_days=45,
        )
        if str(character or "").strip()
        else session_capsules
    )
    covered_dates = {
        str(capsule.get("date_world", "") or "")
        for capsule in capsules
        if str(capsule.get("date_world", "") or "")
    }
    return {
        "legacy_recap": load_historical_recap(
            session_id,
            today_date_world=today,
            character=character,
        ),
        "capsules": capsules,
        "covered_dates": sorted(covered_dates),
        "projection_enabled": True,
    }


__all__ = [
    "_rule_based_strip",
    "_build_compact_prompt",
    "_extract_summary_content",
    "compact_character_daily_recap",
    "compact_session_for_date",
    "compact_session_for_date_v2",
    "daily_continuity_v2_enabled",
    "daily_continuity_v2_projection_enabled",
    "load_daily_continuity_capsules",
    "load_daily_continuity_capsules_for_character",
    "list_due_daily_continuity_retries",
    "load_historical_context_bundle",
    "load_historical_recap",
]
