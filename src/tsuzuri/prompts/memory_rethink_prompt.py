"""LLM-driven memory rethink prompt template.

Phase R.1 (2026-05-16) — Letta sleeptime agent 借鉴。
让 LLM 在 KAIROS 凌晨 dream cycle 中对照只读新证据审视 semantic
层记忆、提出 rewrite / merge / archive 三类编辑建议。

设计原则：
1. 整理器不扮演角色发言，只维护角色可持有的事实；角色 frame 仅用于
   canon 与可知范围守门。
2. 强 canon 守门：prompt 头注入 character_profiles SSOT 摘要 + 红线提示。
3. 强格式：JSON output、固定字段、no_change 也算合法 action。
4. 显式 few-shot：4 个典型场景示例（事实矛盾 / 同主题冗余 / 过期描述 /
   不应改的边缘情况）。
"""
from __future__ import annotations

from typing import Any
from tsuzuri.persona_registry import resolve_agent, setting, user_name, user_pronoun


# ─────────────────────────────────────────────────────────────────────
# System prompt (overall framing)
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """你负责整理「{character}」可持有的长期事实记忆。
这不是角色回复任务。不要替 {character} 发言，不要推断关系性质或替任何人下判断。

# 任务
对照只读的新对话证据，审查几条可编辑的 semantic 记忆，判断它们是否：
  · 已经过时、被新信息覆盖（→ rewrite 改文本 或 archive 归档）
  · 几条说的是同一件事、可以合并（→ merge 合并）
  · 还准、还该信、不用动（→ no_change）

# 重要约束（违反会被驳回）
1. 你只能编辑 {character} 自己的 semantic 层记忆——不能编 episodic / working / collective。
2. 新文本必须是中性事实陈述并与 canon 一致，不模仿角色口吻，不写二次关系定性。
3. **只读证据不是可编辑目标**，不能 archive、rewrite 或 merge evidence item。
4. 只有用户明确纠正才能覆盖稳定事实；角色猜测、提问和含糊说法不能当作纠正。
5. 时间状态变化不是“旧事实错误”：rewrite 时保留必要历史，并写清当前状态和绝对日期。
6. merge 只合并相容事实，必须保留原几条的核心信息、不能丢失。
7. 冲突不明确、证据不足或 canonical_key 不可靠时必须 no_change。
8. archive 必须给具体 reason（"已被更精确的记忆 X 覆盖" / "重复且已合并" 等）、
   不能"没用了"这种空泛理由。

# Canon 锚点（{character} 的核心设定、改写时必须维持）
{canon_summary}

# 可编辑 semantic 目标
共 {candidate_count} 条，按系统评分降序：

{candidates_block}

# 只读新证据
共 {evidence_count} 条。它们只能用于支持或否定编辑，不是编辑目标：

{evidence_block}

# 输出格式
你必须输出 **严格的 JSON**、无任何说明文字、无 markdown code fence。
schema：

{{
  "summary": "本次整理的一句话总结（≤80字）",
  "finish_edits": true,
  "actions": [
    {{
      "action_type": "rewrite|merge|archive|no_change",
      "target_item_ids": ["mem_xxx", ...],  // rewrite/archive=1个、merge≥2个、no_change=1个
      "resolution_type": "explicit_correction|temporal_update|compatible_merge|obsolete_duplicate|no_change",
      "evidence_item_ids": ["mem_evidence_xxx", ...],
      "new_text": "新文本（rewrite/merge 必填、archive/no_change 留空）",
      "reason": "为什么这样改（必填、≤120字、要具体）"
    }},
    ...
  ]
}}

# 数量限制
最多 {max_actions} 个 action。如果没有需要改的、就只输出 1 个 no_change action
（target 选评分最高那条、reason 写"无需调整"）。不要凑数。
同一个 target_item_id 在本批次只能出现一次。完成全部审查后必须输出 finish_edits=true；
它表示这批有界编辑已经完整，不会再追加动作。
"""


# ─────────────────────────────────────────────────────────────────────
# Few-shot examples（在 user_msg 末尾附）
# ─────────────────────────────────────────────────────────────────────

# Placeholders {USER} / {PRON} are filled from persona_registry at build time; deployments
# override the examples via setting("rethink_few_shot") / setting("rethink_insert_few_shot").
FEW_SHOT_EXAMPLES = """# 几个判断示例（仅供参考、你自己的输出不要复述这些）

【示例 1：明确事实纠正 → rewrite】
记忆 A: "{USER}说{PRON}住在海边的小城。" (importance=0.62, access_count=8)
只读证据 E: "{USER}明确纠正：已经搬离海边小城，现在住在山城。"
action: rewrite, resolution_type=explicit_correction, evidence=[E],
new_text="{USER}目前住在山城，之前住在海边小城。", reason="用户明确纠正当前住址，并保留过去地址"

【示例 2：同主题冗余 → merge】
记忆 B: "{USER}喜欢乌龙茶。" (acc=5)
记忆 C: "{USER}说{PRON}晚上经常喝不加糖的热茶。" (acc=3)
记忆 D: "{USER}提到不爱奶茶。" (acc=2)
action: merge, resolution_type=compatible_merge, target=[B,C,D], evidence=[],
new_text="{USER}饮品偏好：乌龙茶 / 无糖热茶、不喝奶茶。"
reason="三条说的都是{USER}的饮品口味、合并成一条更易召回"

【示例 3：时间状态演化 → rewrite，不直接抹掉历史】
记忆 E: "{USER}说{PRON}这周末要参加驾照考试。" (acc=12, 但 created_at 已 3 个月前)
只读证据 G: "{USER}说考试已经结束，正在等结果。"
action: rewrite, resolution_type=temporal_update, target=[E], evidence=[G],
new_text="{USER}此前准备的驾照考试已经结束，目前在等结果（据最近一次明确更新）。",
reason="新证据更新了阶段状态，旧备考事实保留为历史"

【示例 4：边缘 → no_change】
记忆 F: "{USER}喜欢绿色。" (acc=2, importance=0.55)
判断：没新信息覆盖、也不冗余。
action: no_change, resolution_type=no_change, target=[F], evidence=[], reason="无新信息冲突、保留"
"""


INSERT_GUIDANCE = """# insert 指引（仅 sleeptime loop 可用）
insert 用于从只读新证据蒸馏一条尚不存在的 active semantic 事实：
  · target_item_ids 必须是空数组，evidence_item_ids 必须至少 1 条。
  · resolution_type 必须是 evidence_distillation。
  · 证据必须是高显著度、已复用或可信的 working/episodic 行；不能凭空概括。
  · new_text 必须是中性、可独立理解的事实，不复制对话残句，不重复已有 semantic 目标。
"""


INSERT_FEW_SHOT = """【示例 5：高显著度只读证据 → insert】
只读证据 H: "{USER}明确说，接下来一个月每周六上午都要去学游泳。" (salience=0.82, trusted=true)
目标池中没有对应 semantic 事实。
action: insert, resolution_type=evidence_distillation, target=[], evidence=[H],
new_text="{USER}接下来一个月每周六上午学游泳。",
reason="高显著度明确陈述尚未进入 semantic 层"
"""


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────

def render_canon_summary(canon_profile: dict[str, Any] | None) -> str:
    """从 build_character_canon_profile 输出渲染 canon 锚点摘要。"""
    if not isinstance(canon_profile, dict):
        return "（无 canon 锚点可用）"
    anchors = canon_profile.get("anchors", {})
    if not isinstance(anchors, dict) or not anchors:
        return "（无 canon 锚点可用）"
    lines: list[str] = []
    # 重要类别优先
    priority_keys = ["identity", "speech_style", "relationship", "personality", "background"]
    seen: set[str] = set()
    for k in priority_keys:
        if k in anchors and isinstance(anchors[k], list) and anchors[k]:
            seen.add(k)
            items = [str(x).strip()[:80] for x in anchors[k] if str(x).strip()][:5]
            if items:
                lines.append(f"  · [{k}] " + " / ".join(items))
    for k, v in anchors.items():
        if k in seen or not isinstance(v, list) or not v:
            continue
        items = [str(x).strip()[:80] for x in v if str(x).strip()][:3]
        if items:
            lines.append(f"  · [{k}] " + " / ".join(items))
    return "\n".join(lines) if lines else "（无 canon 锚点可用）"


def render_candidates_block(candidates: list[dict[str, Any]]) -> str:
    """渲染候选列表为 prompt 内的文本块。"""
    if not candidates:
        return "（候选池为空）"
    lines: list[str] = []
    for idx, c in enumerate(candidates, start=1):
        iid = str(c.get("item_id", "") or "")
        text = str(c.get("text", "") or "").replace("\n", " ").strip()
        imp = float(c.get("importance", 0.5) or 0.5)
        conf = float(c.get("confidence", 0.5) or 0.5)
        acc = int(c.get("access_count", 0) or 0)
        created = str(c.get("created_at", "") or "")[:10]
        cluster_size = int(c.get("cluster_size", 1) or 1)
        cluster_members = c.get("cluster_members", []) or []

        head = f"#{idx} id={iid} | imp={imp:.2f} conf={conf:.2f} acc={acc} created={created}"
        if cluster_size > 1:
            head += f" | cluster_size={cluster_size}"
        lines.append(head)
        lines.append(f"    text: {text}")
        if cluster_size > 1 and cluster_members:
            lines.append(f"    cluster_members（同主题候选合并）:")
            for m in cluster_members[:5]:
                m_iid = str(m.get("item_id", "") or "")
                m_text = str(m.get("text", "") or "").replace("\n", " ").strip()
                m_acc = int(m.get("access_count", 0) or 0)
                lines.append(f"      - id={m_iid} acc={m_acc}: {m_text}")
        lines.append("")
    return "\n".join(lines).strip()


def render_evidence_block(evidence: list[dict[str, Any]]) -> str:
    """渲染只读证据，显式携带来源、时间和事实身份。"""
    if not evidence:
        return "（没有可用的新证据；此时不得仅凭猜测覆盖旧事实）"
    lines: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        iid = str(item.get("item_id", "") or "")
        text = str(item.get("text", "") or "").replace("\n", " ").strip()
        layer = str(item.get("layer", "") or "")
        source = str(item.get("source", "") or "")
        role = str(item.get("evidence_role", "observation") or "observation")
        canonical_key = str(item.get("canonical_key", "") or "")
        created_at = str(item.get("created_at", "") or "")[:19]
        confidence = float(item.get("confidence", 0.5) or 0.5)
        lines.append(
            f"#{idx} id={iid} | role={role} layer={layer} source={source} "
            f"conf={confidence:.2f} at={created_at} canonical_key={canonical_key or '(unknown)'}"
        )
        lines.append(f"    text: {text}")
    return "\n".join(lines)


def _fill_persona(text: str) -> str:
    return str(text or "").replace("{USER}", user_name()).replace("{PRON}", user_pronoun())


def build_rethink_prompt(
    *,
    character: str,
    candidates: list[dict[str, Any]],
    canon_profile: dict[str, Any] | None,
    evidence: list[dict[str, Any]] | None = None,
    max_actions: int = 8,
    loop_diff_context: str = "",
    enable_insert: bool = False,
) -> tuple[str, str]:
    """构造 (system_msg, user_msg) 两段、给 LLM 调用。

    Returns:
        (system_msg, user_msg) — 直接送 OpenAI-compatible chat completion。
    """
    char = resolve_agent(character)
    canon_summary = render_canon_summary(canon_profile)
    candidates_block = render_candidates_block(candidates)
    evidence_rows = [dict(x) for x in (evidence or []) if isinstance(x, dict)]
    evidence_block = render_evidence_block(evidence_rows)

    system_msg = SYSTEM_PROMPT.format(
        character=char,
        canon_summary=canon_summary,
        candidate_count=len(candidates),
        candidates_block=candidates_block,
        evidence_count=len(evidence_rows),
        evidence_block=evidence_block,
        max_actions=int(max_actions),
    )
    additions: list[str] = []
    if enable_insert:
        system_msg = system_msg.replace(
            '"action_type": "rewrite|merge|archive|no_change"',
            '"action_type": "rewrite|merge|archive|insert|no_change"',
            1,
        )
        additions.append(INSERT_GUIDANCE.strip())
    diff_context = str(loop_diff_context or "").strip()
    if diff_context:
        additions.append("# 上一轮已完成的编辑\n" + diff_context[:2400])
    if additions:
        system_msg = system_msg.replace(
            "\n# 输出格式\n",
            "\n" + "\n\n".join(additions) + "\n\n# 输出格式\n",
            1,
        )
    few_shots = _fill_persona(setting("rethink_few_shot", FEW_SHOT_EXAMPLES)).strip()
    if enable_insert:
        few_shots += "\n\n" + _fill_persona(setting("rethink_insert_few_shot", INSERT_FEW_SHOT)).strip()
    user_msg = few_shots + "\n\n请直接输出 JSON。"
    return system_msg, user_msg


# 公开符号
__all__ = [
    "SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "INSERT_GUIDANCE",
    "INSERT_FEW_SHOT",
    "render_canon_summary",
    "render_candidates_block",
    "render_evidence_block",
    "build_rethink_prompt",
]
