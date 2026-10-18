import json
import os
import tempfile
import asyncio
import threading
import time
import unittest
from unittest import mock


class TurnFactLedgerTests(unittest.TestCase):
    def test_normalize_rejects_future_plan_as_character_start_action(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        card = normalize_turn_fact_card(
            {
                "fact": "张三计划明天整理路线。",
                "character_action_updates": [{
                    "transition": "start",
                    "event_text": "明天整理路线",
                    "evidence_span": "我明天会整理路线",
                }],
            },
            "张三",
            {"id": "u-plan", "content": "路线先记一下。"},
            {"id": "a-plan", "content": "我明天会整理路线。"},
        )
        self.assertEqual(card["character_action_updates"], [])

    def test_normalize_requires_user_evidence_for_developer_state(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        card = normalize_turn_fact_card(
            {
                "fact": "阿澈已经放暑假。",
                "developer_state_updates": [{
                    "predicate": "academic.phase",
                    "value": "暑假",
                    "operation": "set",
                    "temporal_status": "active",
                    "evidence_span": "张三说阿澈放暑假",
                }],
            },
            "张三",
            {"id": "u-state", "content": "最近还在上课。"},
            {"id": "a-state", "content": "那你辛苦了。"},
        )
        self.assertEqual(card["developer_state_updates"], [])

    def test_normalize_preserves_mixed_completed_and_future_action_semantics(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        card = normalize_turn_fact_card(
            {
                "fact": "张三已经整理好路线，并答应明天发送截图。",
                "character_action_updates": [
                    {"transition": "complete", "event_text": "整理路线", "evidence_span": "我已经整理好路线"},
                    {"transition": "start", "event_text": "明天发送截图", "evidence_span": "我明天会发送截图"},
                ],
            },
            "张三",
            {"id": "u-mixed", "content": "路线呢？"},
            {"id": "a-mixed", "content": "我已经整理好路线，明天会发送截图。"},
        )
        transitions = [row["transition"] for row in card["character_action_updates"]]
        self.assertIn("complete", transitions)
        self.assertNotIn("start", transitions)
    def test_extraction_prompt_teaches_quote_is_evidence_not_new_user_fact(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "张三",
            {"role": "user", "content": '“你明天会帮我看路线。”这句话是真的吗？'},
            {"role": "assistant", "content": "那是我之前的计划，不代表已经做完。"},
        )
        self.assertIn("引用语义", prompt)
        self.assertIn("不等于阿澈当前重新确认", prompt)
        self.assertIn("只有用户在引用之外明确确认、否定、追问、改写", prompt)

    def test_turnfact_canonical_key_accepts_stable_claim_identity_only(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card

        base = {"fact": "阿澈和张三确认花展地点。", "canonical_key": "shared:花展:location"}
        card = normalize_turn_fact_card(base, "张三", {"id": "u1", "content": "花展在哪？"}, {"id": "a1", "content": "冲绳。"})
        self.assertEqual(card["canonical_key"], "shared:花展:location")
        invalid = normalize_turn_fact_card({"fact": "事实", "canonical_key": "花展地点"}, "张三", {"id": "u2", "content": "事实"}, {"id": "a2", "content": "事实"})
        self.assertEqual(invalid["canonical_key"], "")

    def test_fact_card_inherits_audience_scope_from_exchange(self):
        from tsuzuri.turn_fact_ledger import (
            build_fact_extraction_prompt,
            normalize_turn_fact_card,
            sanitize_fact_cards,
        )

        card = normalize_turn_fact_card(
            {"fact": "访客提到一场花展。"},
            "王五",
            {"id": "u-guest", "audience_scope": "guest:scope-1"},
            {"id": "a-guest", "audience_scope": "guest:scope-1"},
        )

        self.assertEqual(card["audience_scope"], "guest:scope-1")
        self.assertEqual(sanitize_fact_cards([card])[0]["audience_scope"], "guest:scope-1")
        prompt = build_fact_extraction_prompt(
            "王五",
            {"content": "花展怎么样？", "audience_scope": "guest:scope-1"},
            {"content": "还不错。", "audience_scope": "guest:scope-1"},
        )
        self.assertIn("访客：花展怎么样？", prompt)
        self.assertIn("不知道访客的姓名、性别和所在地", prompt)
        self.assertNotIn("阿澈在中国", prompt)

    def test_fact_prompt_preserves_event_time_separately_from_source_time(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "李四",
            self._entry("user", "今晚我要睡觉了。", "u-time", 1_700_000_000_000),
            self._entry("assistant", "晚安。", "a-time", 1_700_000_001_000),
        )

        self.assertIn("不得猜绝对日期", prompt)
        self.assertIn("事实发生时间与消息来源时间的区别", prompt)
        self.assertIn('"event_time"', prompt)
        self.assertNotIn("必须换算成", prompt)

    def _entry(self, role, content, msg_id, ts):
        return {
            "role": role,
            "content": content,
            "id": msg_id,
            "timestamp": ts,
        }

    def test_extract_card_uses_primary_json_and_clamps_scores(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry("user", "我担心历史数组会让角色复述。", "u-1", 1000)
        assistant = self._entry("assistant", "可以把旧对话改成事实流。", "a-1", 1001)
        calls = []

        def primary(prompt):
            calls.append(("primary", prompt))
            return json.dumps(
                {
                    "near_summary": "阿澈担心历史数组诱发复述，李四建议改成事实流。",
                    "fact": "阿澈担心旧聊天原文进入 prompt 会诱发复述。",
                    "state_delta": "讨论方向转向事实流上下文。",
                    "open_threads": ["如何设计保真度衰减"],
                    "commitments": ["李四会继续讨论事实流方案"],
                    "corrections": [],
                    "salience": 1.5,
                    "confidence": -0.2,
                },
                ensure_ascii=False,
            )

        def fallback(prompt):
            calls.append(("fallback", prompt))
            return "{}"

        card = extract_turn_fact_card(
            "李四",
            user,
            assistant,
            llm_callers=[("qwen3.7-max", primary), ("deepseek-v4-flash", fallback)],
        )

        self.assertEqual([c[0] for c in calls], ["primary"])
        self.assertEqual(card["model"], "qwen3.7-max")
        self.assertFalse(card["fallback_used"])
        self.assertEqual(card["user_id"], "u-1")
        self.assertEqual(card["assistant_id"], "a-1")
        self.assertLessEqual(card["salience"], 1.0)
        self.assertGreaterEqual(card["confidence"], 0.0)
        self.assertIn("事实流", card["near_summary"])

    def test_extract_card_falls_back_to_second_model(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry("user", "别硬切窗口。", "u-2", 2000)
        assistant = self._entry("assistant", "可以渐进式压缩。", "a-2", 2001)
        calls = []

        def primary(prompt):
            calls.append("primary")
            return "not json"

        def fallback(prompt):
            calls.append("fallback")
            return json.dumps(
                {
                    "near_summary": "阿澈不想硬切窗口，李四建议渐进式压缩。",
                    "fact": "阿澈希望历史上下文按距离渐进压缩。",
                    "state_delta": "方案从滑动窗口推进到保真度衰减。",
                    "open_threads": [],
                    "commitments": [],
                    "corrections": [],
                    "salience": 0.7,
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card(
            "李四",
            user,
            assistant,
            llm_callers=[("qwen3.7-max", primary), ("deepseek-v4-flash", fallback)],
        )

        self.assertEqual(calls, ["primary", "fallback"])
        self.assertEqual(card["model"], "deepseek-v4-flash")
        self.assertTrue(card["fallback_used"])
        self.assertIn("渐进压缩", card["fact"])

    def test_extract_card_uses_rule_fallback_when_models_fail(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry("user", "我想让事实提取失败时也不中断聊天。", "u-3", 3000)
        assistant = self._entry("assistant", "那就用规则 fallback。", "a-3", 3001)

        card = extract_turn_fact_card(
            "李四",
            user,
            assistant,
            llm_callers=[
                ("qwen3.7-max", lambda prompt: ""),
                ("deepseek-v4-flash", lambda prompt: (_ for _ in ()).throw(RuntimeError("boom"))),
            ],
        )

        self.assertEqual(card["model"], "rule_fallback")
        self.assertTrue(card["fallback_used"])
        self.assertIn("阿澈对李四说", card["fact"])
        self.assertIn("不中断聊天", card["fact"])

    def test_extract_card_rewrites_dev_pronoun_drift(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry("user", "你还记得我要考电路我就已经很开心了。", "u-pronoun", 3100)
        assistant = self._entry("assistant", "撑不住的时候别硬扛，你那边又不是只有一个人。", "a-pronoun", 3101)

        def primary(prompt):
            return json.dumps(
                {
                    "near_summary": "阿澈感谢王五记得电路考试，王五提醒她并非孤身一人。",
                    "fact": "王五承诺在她撑不住时不会让她独自承担。",
                    "state_delta": "双方就她的电路考试达成自主与后备支援的共识。",
                    "open_threads": [],
                    "commitments": ["王五承诺在她撑不住时提供支援"],
                    "corrections": [],
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card("王五", user, assistant, llm_callers=[("qwen3.7-max", primary)])

        merged = "\n".join(
            [
                card["near_summary"],
                card["fact"],
                card["state_delta"],
                "\n".join(card["commitments"]),
            ]
        )
        self.assertIn("提醒阿澈并非孤身一人", merged)
        self.assertIn("阿澈撑不住", merged)
        self.assertIn("不会让阿澈独自承担", merged)
        self.assertIn("阿澈的电路考试", merged)
        self.assertNotIn("她", merged)

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    def test_kinship_third_party_blocks_pronoun_rewrite(self):
        """家人不在 NPC 花名册里，靠称谓词兜住。"""
        from tsuzuri.turn_fact_ledger import _normalize_actor_references

        text = "阿澈和王五聊到李四的姐姐，她这几天在准备考试"
        self.assertEqual(
            _normalize_actor_references(text, "王五", limit=180), text
        )

    def test_lamp_noun_does_not_count_as_character_name(self):
        """「李四」是角色名也是普通名词，灯具语境不得误判成第三方在场。"""
        from tsuzuri.turn_fact_ledger import _text_mentions_third_party

        self.assertFalse(_text_mentions_third_party("阿澈说路灯下等她", "王五"))
        self.assertFalse(_text_mentions_third_party("灯光太亮了", "王五"))
        self.assertTrue(_text_mentions_third_party("王五说李四今天很安静", "王五"))
        # 李四本人是当前角色时，自己不算第三方。
        self.assertFalse(_text_mentions_third_party("李四说她有点累", "李四"))

    def test_developer_pronoun_repair_still_fires_without_third_party(self):
        """guard 不能误伤原有的开发者代词修复。"""
        from tsuzuri.turn_fact_ledger import _normalize_actor_references, _normalize_actor_list

        self.assertNotIn(
            "她",
            _normalize_actor_references("王五提醒她并非孤身一人", "王五", limit=180),
        )
        self.assertEqual(
            _normalize_actor_list(["王五劝她别硬扛"], "王五"),
            ["王五劝阿澈别硬扛"],
        )


    def test_proactive_turn_without_user_message_reaches_llm(self):
        """auto_greet 等无用户发言的主动轮次不触发代词 guard，_background_facts 通道保持可用。"""
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        assistant = self._entry(
            "assistant",
            "今天放学和三三一起走。她没问我那件事，我也没说。",
            "a-greet",
            3_700,
        )
        calls = []

        def primary(prompt):
            calls.append(prompt)
            return json.dumps(
                {
                    "near_summary": "李四提到今天放学和三三一起走。",
                    "fact": "李四今天放学和三三一起走。",
                    "_background_facts": ["李四今天放学和三三一起走。"],
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card("李四", {}, assistant, llm_callers=[("qwen3.7-max", primary)])

        self.assertTrue(calls)
        self.assertEqual(card["model"], "qwen3.7-max")
        self.assertFalse(card["fallback_used"])
        self.assertEqual(card["_background_facts"], ["李四今天放学和三三一起走。"])

    pass  # upstream-only test removed
    def test_fact_prompt_defines_third_party_and_unknown_referent_exit(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "王五",
            self._entry("user", "她最近还好吗？", "u-tp", 3_600),
            self._entry("assistant", "还行吧。", "a-tp", 3_601),
        )

        self.assertIn("第三方必须实名化", prompt)
        self.assertIn("（对象不明）", prompt)
        self.assertIn("不得被压成", prompt)

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    def test_antecedent_context_renders_prior_turns_bounded(self):
        from tsuzuri.turn_fact_ledger import build_antecedent_context, build_fact_extraction_prompt

        cards = [
            {"near_summary": "阿澈问起孙八在做什么，王五说她在准备新小组演出"},
            {"near_summary": "王五提到孙八最近集训到很晚"},
            {"near_summary": "阿澈说自己在赶实验报告"},
        ]
        block = build_antecedent_context(cards)
        # 只取最近 2 轮，且保持时间顺序。
        self.assertNotIn("在准备新小组演出", block)
        self.assertLess(block.index("集训到很晚"), block.index("赶实验报告"))

        prompt = build_fact_extraction_prompt(
            "王五",
            self._entry("user", "她还好吗？", "u-ant", 3_700),
            self._entry("assistant", "还行。", "a-ant", 3_701),
            prior_cards=cards,
        )
        self.assertIn("【前文】", prompt)
        self.assertIn("【本轮】", prompt)
        self.assertIn("不要把前文内容当成本轮发生的事", prompt)

    def test_antecedent_context_skips_archived_and_empty(self):
        from tsuzuri.turn_fact_ledger import build_antecedent_context

        self.assertEqual(build_antecedent_context([]), "")
        self.assertEqual(build_antecedent_context(None), "")
        self.assertEqual(
            build_antecedent_context([{"near_summary": "旧事", "archived": True}]), ""
        )

    def test_extract_card_repairs_remote_meal_actor_drift(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry("user", "知道了，我还不至于做出什么傻事，放心吧", "u-meal", 3_200)
        assistant = self._entry(
            "assistant",
            "切，知道了还在这边答得这么老实。不过你既然累了，总归要吃饭吧。"
            "我这边刚系上围裙，晚上打算炒个蛋饭",
            "a-meal",
            3_201,
        )

        def drifting_primary(prompt):
            return json.dumps(
                {
                    "near_summary": "阿澈保证不会做傻事，王五主动提出为疲惫的阿澈做蛋炒饭。",
                    "fact": "王五已系上围裙准备炒蛋饭给阿澈吃，阿澈承诺不会做傻事。",
                    "state_delta": "王五以实际做饭行动照顾阿澈。",
                    "open_threads": [],
                    "commitments": ["王五承诺晚上为阿澈炒蛋饭", "阿澈承诺不会做傻事"],
                    "corrections": [],
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card(
            "王五",
            user,
            assistant,
            llm_callers=[("qwen3.7-max", drifting_primary)],
        )
        merged = "\n".join(
            [
                card["near_summary"],
                card["fact"],
                card["state_delta"],
                "\n".join(card["commitments"]),
            ]
        )

        self.assertIn("王五自己准备炒个蛋饭", merged)
        self.assertNotIn("给阿澈", merged)
        self.assertNotIn("为阿澈", merged)
        self.assertNotIn("做饭行动照顾阿澈", merged)
        self.assertNotIn("王五承诺晚上", merged)
        self.assertIn("阿澈承诺不会做傻事", merged)

    def test_extract_card_preserves_salient_user_details_when_model_is_too_generic(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry(
            "user",
            "刚翻了翻手机，看到冲绳那边有个夏季花展的推荐，向日葵从七月上旬开到八月下旬。"
            "······你那边今天忙完了？累不累",
            "u-sunflower",
            3500,
        )
        assistant = self._entry("assistant", "累，当然累，根本就复习不完", "a-sunflower", 3501)

        def generic_primary(prompt):
            return json.dumps(
                {
                    "near_summary": "阿澈看到夏季花展推荐后关心王五是否忙完、累不累。",
                    "fact": "阿澈提到夏季花展推荐，并关心王五今天是否累。",
                    "state_delta": "话题从花展推荐进入王五的疲惫和复习压力。",
                    "open_threads": [],
                    "commitments": [],
                    "corrections": [],
                    "salience": 0.7,
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card(
            "王五",
            user,
            assistant,
            llm_callers=[("qwen3.7-max", generic_primary)],
        )
        merged = " ".join(
            [
                card.get("near_summary", ""),
                card.get("fact", ""),
                card.get("state_delta", ""),
                " ".join(card.get("open_threads", [])),
            ]
        )

        self.assertIn("冲绳", merged)
        self.assertIn("向日葵", merged)
        self.assertIn("七月上旬", merged)
        self.assertIn("八月下旬", merged)

    def test_extract_card_does_not_add_noisy_details_when_model_preserved_them(self):
        from tsuzuri.turn_fact_ledger import extract_turn_fact_card

        user = self._entry(
            "user",
            "刚翻了翻手机，看到冲绳那边有个夏季花展的推荐，向日葵从七月上旬开到八月下旬。"
            "······你那边今天忙完了？累不累",
            "u-sunflower-ok",
            3600,
        )
        assistant = self._entry("assistant", "累，当然累，根本就复习不完", "a-sunflower-ok", 3601)

        def preserving_primary(prompt):
            return json.dumps(
                {
                    "near_summary": "阿澈分享冲绳向日葵花展（七月上旬至八月下旬）并关心王五是否累。",
                    "fact": "阿澈提到冲绳向日葵花展（七月上旬至八月下旬），王五表示复习很累。",
                    "state_delta": "花展线索之后，话题进入王五复习疲惫状态。",
                    "open_threads": [],
                    "commitments": [],
                    "corrections": [],
                    "salience": 0.7,
                    "confidence": 0.8,
                },
                ensure_ascii=False,
            )

        card = extract_turn_fact_card(
            "王五",
            user,
            assistant,
            llm_callers=[("qwen3.7-max", preserving_primary)],
        )
        merged = " ".join([card.get("near_summary", ""), card.get("fact", "")])

        self.assertIn("冲绳", merged)
        self.assertIn("向日葵", merged)
        self.assertNotIn("具体线索", merged)
        self.assertNotIn("看到冲绳", merged)
        self.assertNotIn("······你", merged)

    def test_render_progressive_messages_keeps_short_continuous_dialogue_verbatim(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        history = [
            self._entry("user", "第零轮用户原话", "u-0", 500),
            self._entry("assistant", "第零轮角色原话不该再出现", "a-0", 501),
            self._entry("user", "第一轮用户原话", "u-1", 1000),
            self._entry("assistant", "第一轮角色原话不该再出现", "a-1", 1001),
            self._entry("user", "第二轮用户原话", "u-2", 2000),
            self._entry("assistant", "第二轮角色原话也不该再出现", "a-2", 2001),
            self._entry("user", "上一轮用户原话", "u-3", 3000),
            self._entry("assistant", "上一轮角色原话应该保留", "a-3", 3001),
            self._entry("user", "当前用户原话", "u-4", 4000),
        ]
        cards = [
            {
                "turn_id": "t-0",
                "user_id": "u-0",
                "assistant_id": "a-0",
                "near_summary": "近端摘要零",
                "fact": "中端事实零",
                "state_delta": "中端状态零",
                "commitments": [],
                "corrections": [],
            },
            {
                "turn_id": "t-1",
                "user_id": "u-1",
                "assistant_id": "a-1",
                "near_summary": "近端摘要一",
                "fact": "远端事实一",
                "state_delta": "远端状态一",
                "commitments": [],
                "corrections": [],
            },
            {
                "turn_id": "t-2",
                "user_id": "u-2",
                "assistant_id": "a-2",
                "near_summary": "近端摘要二",
                "fact": "中端事实二",
                "state_delta": "中端状态二",
                "commitments": [],
                "corrections": [],
            },
            {
                "turn_id": "t-3",
                "user_id": "u-3",
                "assistant_id": "a-3",
                "near_summary": "上一轮摘要不用替代原文",
                "fact": "上一轮事实",
                "state_delta": "",
                "commitments": [],
                "corrections": [],
            },
        ]

        messages = render_fact_ledger_messages(history, cards, current_character="李四")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("第零轮角色原话不该再出现", merged)
        self.assertIn("第一轮角色原话不该再出现", merged)
        self.assertIn("第二轮角色原话也不该再出现", merged)
        self.assertNotIn("中端事实零", merged)
        self.assertNotIn("近端摘要二", merged)
        self.assertIn("上一轮角色原话应该保留", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("当前用户原话", messages[-1]["content"])

    def test_render_time_gap_state_guard_suppresses_stale_last_assistant_raw(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        base = 1_609_459_200_000  # 2021-01-01 09:00 WORLD_TZ
        history = [
            self._entry("user", "昨天晚上还在聊集训结束后的状态", "u-old", base),
            self._entry("assistant", "上一时间段的角色原话不应在长间隔后继续裸露", "a-old", base + 60_000),
            self._entry("user", "今天我们改聊事实流状态提示怎么收口。", "u-now", base + 7_400_000),
        ]
        cards = [
            {
                "turn_id": "t-old",
                "user_id": "u-old",
                "assistant_id": "a-old",
                "near_summary": "阿澈和李四上一时间段聊过集训后的状态。",
                "fact": "阿澈和李四上一时间段聊过集训后的状态。",
                "state_delta": "该话题属于上一时间段。",
                "commitments": [],
                "corrections": [],
            }
        ]

        messages = render_fact_ledger_messages(history, cards, current_character="李四")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("【对话状态提示】", merged)
        self.assertIn("不要把当前轮当成紧跟上一轮的承接", merged)
        self.assertIn("阿澈和李四上一时间段聊过集训后的状态", merged)
        self.assertIn("来源：2021-01-01 09:00-09:01 WORLD_TZ", merged)
        self.assertNotIn("上一时间段的角色原话不应在长间隔后继续裸露", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("今天我们改聊", messages[-1]["content"])

    def test_render_half_hour_gap_uses_same_boundary_for_guard_and_raw_exchange(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        base = 1_609_459_200_000
        history = [
            self._entry("user", "上一段在聊电路复习。", "u-old", base),
            self._entry("assistant", "半小时前的角色原话不应继续裸露。", "a-old", base + 60_000),
            self._entry("user", "电路复习这件事后来怎么样了？", "u-now", base + 31 * 60_000),
        ]
        cards = [
            {
                "turn_id": "t-old",
                "user_id": "u-old",
                "assistant_id": "a-old",
                "created_at_ms": base + 60_000,
                "fact": "阿澈和王五讨论过电路复习。",
            }
        ]

        messages = render_fact_ledger_messages(history, cards, current_character="王五")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("不要把当前轮当成紧跟上一轮的承接", merged)
        self.assertNotIn("半小时前的角色原话不应继续裸露", merged)
        self.assertIn("阿澈和王五讨论过电路复习", merged)

    def test_inline_state_hints_expire_after_the_segment_they_guarded(self):
        from tsuzuri.turn_fact_ledger import collect_conversation_state_hints

        history = [
            self._entry("user", "第一段用户消息", "u-0", 1_000),
            self._entry("assistant", "第一段回复", "a-0", 2_000),
            {
                "role": "system",
                "content": "【当前轮次优先】旧断层提示只约束紧随其后的那一轮。",
                "id": "s-old",
                "timestamp": 2_500,
            },
            self._entry("user", "第二段用户消息", "u-1", 3_000),
            self._entry("assistant", "第二段回复", "a-1", 4_000),
            self._entry("user", "当前用户消息", "u-now", 5_000),
        ]

        hints = collect_conversation_state_hints(history)

        self.assertFalse(any("旧断层提示" in hint for hint in hints))

    def test_uncovered_assistant_fact_scan_keeps_the_most_recent_candidates(self):
        from tsuzuri.turn_fact_ledger import collect_uncovered_history_fact_records

        history = [
            self._entry(
                "assistant",
                f"地点{idx}那边的向日葵从七月上旬开到八月下旬。",
                f"a-{idx}",
                1_000 + idx,
            )
            for idx in range(6)
        ]
        history.append(self._entry("user", "向日葵花期呢？", "u-now", 2_000_000))

        records = collect_uncovered_history_fact_records(
            history,
            [],
            current_character="王五",
        )
        ids = [str(row.get("source_ref_id", "")) for row in records]

        self.assertIn("a-4", ids)
        self.assertNotIn("a-0", ids)

    def test_render_marks_fact_source_periods_for_multiple_segments(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        base = 1_609_459_200_000  # 2021-01-01 09:00 WORLD_TZ
        history = [
            self._entry("user", "早上聊过集训安排", "u-morning", base),
            self._entry("assistant", "早上的角色原文不应裸露", "a-morning", base + 60_000),
            self._entry("user", "下午聊过吃饭安排", "u-afternoon", base + 7_200_000),
            self._entry("assistant", "下午的角色原文不应裸露", "a-afternoon", base + 7_260_000),
            self._entry("user", "现在换个问题。", "u-now", base + 14_400_000),
        ]
        cards = [
            {
                "turn_id": "t-morning",
                "user_id": "u-morning",
                "assistant_id": "a-morning",
                "near_summary": "阿澈和李四早上确认集训安排。",
                "fact": "阿澈和李四早上确认集训安排。",
                "state_delta": "集训安排已确认。",
                "commitments": ["李四会按早上确认的集训安排行动。"],
                "corrections": [],
            },
            {
                "turn_id": "t-afternoon",
                "user_id": "u-afternoon",
                "assistant_id": "a-afternoon",
                "near_summary": "阿澈和灯下午确认吃饭安排。",
                "fact": "阿澈和灯下午确认吃饭安排。",
                "state_delta": "吃饭安排已确认。",
                "commitments": [],
                "corrections": [],
            },
        ]

        messages = render_fact_ledger_messages(
            history,
            cards,
            current_character="李四",
            near_summary_cards=1,
        )
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("来源：2021-01-01 09:00-09:01 WORLD_TZ", merged)
        self.assertIn("阿澈和李四早上确认集训安排", merged)
        self.assertIn("来源：2021-01-01 11:00-11:01 WORLD_TZ", merged)
        self.assertIn("阿澈和灯下午确认吃饭安排", merged)
        self.assertNotIn("早上的角色原文不应裸露", merged)
        self.assertNotIn("下午的角色原文不应裸露", merged)

    def test_render_meal_callback_is_not_marked_as_topic_shift(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        history = [
            self._entry("user", "那我先出门吃饭去了，你今天考完试也辛苦了，忙你的去吧", "u-1", 1_000),
            self._entry("assistant", "嗯，我去弄饭了。你路上小心，别边走边看手机", "a-1", 2_000),
            self._entry("user", "我吃完回来了，路上没怎么看手机。你蛋饭弄好了？", "u-2", 3_000),
        ]

        messages = render_fact_ledger_messages(history, [], current_character="王五")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertNotIn("切换了新话题", merged)
        self.assertNotIn("除非用户明确要求继续旧话题", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("蛋饭弄好了", messages[-1]["content"])

    def test_render_preserves_old_assistant_salient_fact_with_correct_subject(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        history = [
            self._entry(
                "assistant",
                "刚翻了翻手机，看到冲绳那边有个夏季花展的推荐，向日葵从七月上旬开到八月下旬。"
                "······你那边今天忙完了？累不累",
                "a-flower",
                1_000,
            ),
            self._entry("user", "累，当然累，根本就复习不完", "u-1", 2_000_000),
            self._entry(
                "assistant",
                "那当然的啊，考试内容又不会自己少一半。要帮忙？\n"
                "我这刚到家，一时半会也跑不了\n"
                "你那边比我这边晚一个小时吧，问句不好听的······你准备熬到几点",
                "a-1",
                2_001_000,
            ),
            self._entry("user", "你要怎么帮啊......我学的东西你也看不懂，你照顾好自己就好了", "u-2", 2_002_000),
            self._entry(
                "assistant",
                "说什么傻话。我确实看不懂你那堆电路，但这跟帮不帮得上忙是两回事\n"
                "你那口气叹得我都听见了\n"
                "办法总比困难多，自己先把人稳住再说",
                "a-2",
                2_003_000,
            ),
            self._entry("user", "我吃完回来了，路上没怎么看手机。你蛋饭弄好了？", "u-current", 2_004_000),
        ]

        cards = [
            {
                "turn_id": "t-1",
                "user_id": "u-1",
                "assistant_id": "a-1",
                "near_summary": "阿澈说自己复习很累，王五提出帮忙并关心熬夜计划。",
                "fact": "阿澈复习压力很大，王五提出帮忙并关心阿澈准备熬到几点。",
                "state_delta": "话题进入复习压力和熬夜安排。",
                "commitments": [],
                "corrections": [],
            },
            {
                "turn_id": "t-2",
                "user_id": "u-2",
                "assistant_id": "a-2",
                "near_summary": "阿澈觉得电路内容王五看不懂，王五把帮助转向情绪和生活支撑。",
                "fact": "王五说明看不懂电路不代表帮不上忙，并提醒阿澈先稳住自己。",
                "state_delta": "王五把帮助定义为支撑阿澈状态。",
                "commitments": ["王五提醒阿澈先把人稳住。"],
                "corrections": [],
            },
        ]

        messages = render_fact_ledger_messages(history, cards, current_character="王五")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("王五早些时候提到", merged)
        self.assertIn("冲绳", merged)
        self.assertIn("向日葵", merged)
        self.assertIn("七月上旬", merged)
        self.assertIn("八月下旬", merged)
        self.assertNotIn("阿澈提到冲绳", merged)
        self.assertNotIn("阿澈分享冲绳", merged)

    def test_render_inline_system_hints_as_state_guard(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        history = [
            self._entry("user", "上一轮用户原话", "u-1", 1_000),
            self._entry("assistant", "上一轮角色原话应该保留", "a-1", 2_000),
            {
                "role": "system",
                "content": "【当前轮次优先】用户刚切换了一个新话题。优先直接回答最后一条用户消息，除非用户明确追问。",
                "id": "sys-1",
                "timestamp": 2_500,
            },
            self._entry("user", "当前用户原话", "u-2", 3_000),
        ]

        messages = render_fact_ledger_messages(history, [], current_character="李四")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("【对话状态提示】", merged)
        self.assertIn("用户刚切换了一个新话题", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("当前用户原话", messages[-1]["content"])

    def test_render_delayed_reply_state_guard_keeps_current_user_last(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        history = [
            self._entry("user", "上一轮用户原话", "u-1", 1_000),
            self._entry("assistant", "上一轮角色原话应该保留", "a-1", 2_000),
            self._entry("user", "当前这条需要即时偷回", "u-2", 3_000),
        ]
        pending = [
            {
                "id": "u-pending",
                "content": "上一条进入延迟回复队列的用户消息",
                "timestamp_ms": 2_500,
                "reason": "in_class",
            }
        ]

        messages = render_fact_ledger_messages(
            history,
            [],
            current_character="李四",
            pending_user_messages=pending,
            delay_sneak_prompt=(
                "（系统提示：你刚才其实已经看到了对方在这条当前消息之前发来的消息，"
                "只是当时没有回。）"
            ),
        )
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("【对话状态提示】", merged)
        self.assertIn("延迟回复状态", merged)
        self.assertIn("上一条进入延迟回复队列的用户消息", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("当前这条需要即时偷回", messages[-1]["content"])
        self.assertNotIn("上一条进入延迟回复队列", messages[-1]["content"])

    def test_render_old_image_uses_compact_visual_fact_without_full_description(self):
        from tsuzuri.turn_fact_ledger import render_fact_ledger_messages

        long_image_description = (
            "[李四 看到一张图片：画面里有一整段非常长的识图原文，包含光线、背景、姿势、"
            "衣服纹理、桌面物品、截图文字和许多不该在远端 prompt 继续完整保留的细节。]"
        )
        history = [
            {
                "role": "user",
                "content": long_image_description,
                "id": "u-img",
                "timestamp": 1_000,
                "vision_meta": {
                    "compact": "[李四 看到一张图片：阿澈发过一张活动室桌面照片。]",
                    "tokens": ["活动室", "桌面"],
                },
            },
            self._entry("assistant", "这张图我看到了。", "a-img", 2_000),
            self._entry("user", "后来我们已经聊别的了。", "u-1", 3_000),
            self._entry("assistant", "嗯，先聊别的。", "a-1", 4_000),
            self._entry("user", "当前用户原话", "u-2", 5_000),
        ]

        messages = render_fact_ledger_messages(history, [], current_character="李四")
        merged = "\n".join(str(m.get("content", "")) for m in messages)

        self.assertIn("阿澈发过一张活动室桌面照片", merged)
        self.assertNotIn("画面里有一整段非常长的识图原文", merged)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("当前用户原话", messages[-1]["content"])

    def test_remove_cards_for_undone_message_ids(self):
        from tsuzuri.turn_fact_ledger import remove_fact_cards_for_message_ids

        cards = [
            {"turn_id": "t-1", "user_id": "u-1", "assistant_id": "a-1"},
            {"turn_id": "t-2", "user_id": "u-2", "assistant_id": "a-2"},
        ]

        remaining = remove_fact_cards_for_message_ids(cards, {"u-1"})

        self.assertEqual([c["turn_id"] for c in remaining], ["t-2"])

    def test_undo_preserves_lifecycle_reserve(self):
        from tsuzuri.turn_fact_ledger import remove_fact_cards_for_message_ids

        cards = [
            {
                "turn_id": f"t-reserve-{idx}",
                "user_id": f"u-reserve-{idx}",
                "assistant_id": f"a-reserve-{idx}",
                "commitments": [f"王五承诺处理事项 {idx}"],
            }
            for idx in range(24)
        ]
        cards.extend(
            {
                "turn_id": f"t-recent-{idx}",
                "user_id": f"u-recent-{idx}",
                "assistant_id": f"a-recent-{idx}",
                "fact": f"近期事实 {idx}",
            }
            for idx in range(80)
        )

        remaining = remove_fact_cards_for_message_ids(cards, {"u-recent-40"})

        ids = {card["turn_id"] for card in remaining}
        self.assertEqual(len(remaining), 103)
        self.assertIn("t-reserve-0", ids)
        self.assertIn("t-reserve-23", ids)
        self.assertNotIn("t-recent-40", ids)

    def test_sanitize_fact_cards_counts_evictions(self):
        from tsuzuri.turn_fact_ledger import get_fact_card_eviction_counts, sanitize_fact_cards

        before = get_fact_card_eviction_counts()
        cards = [{"fact": f"有效事实 {idx}"} for idx in range(85)]
        cards.extend(["not-a-dict", {"turn_id": "t-empty"}])

        out = sanitize_fact_cards(cards, limit=80)

        self.assertEqual(len(out), 80)
        after = get_fact_card_eviction_counts()
        self.assertEqual(after["sanitize_over_limit"] - before["sanitize_over_limit"], 5)
        self.assertEqual(after["sanitize_invalid_row"] - before["sanitize_invalid_row"], 1)
        self.assertEqual(after["sanitize_empty_card"] - before["sanitize_empty_card"], 1)

    def test_lifecycle_card_window_counts_aged_out_and_reserve_overflow(self):
        from tsuzuri.turn_fact_ledger import get_fact_card_eviction_counts, lifecycle_card_window

        before = get_fact_card_eviction_counts()
        aged_plain = [{"fact": f"旧事实 {idx}"} for idx in range(10)]
        lifecycle = [
            {"fact": f"承诺事实 {idx}", "commitments": [f"承诺 {idx}"]}
            for idx in range(30)
        ]
        recent = [{"fact": f"近期事实 {idx}"} for idx in range(80)]

        windowed = lifecycle_card_window([*aged_plain, *lifecycle, *recent], recent=80, reserve=24)

        self.assertEqual(len(windowed), 104)
        after = get_fact_card_eviction_counts()
        self.assertEqual(after["window_aged_out"] - before["window_aged_out"], 10)
        self.assertEqual(
            after["window_reserve_overflow"] - before["window_reserve_overflow"], 6
        )

    def test_fact_state_summary_covers_newest_card_beyond_default_cap(self):
        from tsuzuri.turn_fact_ledger import build_fact_state_summary

        cards = [{"fact": f"旧事实 {idx}"} for idx in range(90)]
        cards.append({"fact": "最新事实", "state_delta": "最新状态标记XYZ"})

        summary = build_fact_state_summary(cards)

        self.assertIn("最新状态标记XYZ", summary)

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    def test_fact_card_user_ids_participate_in_undo_removal(self):
        from tsuzuri.turn_fact_ledger import normalize_turn_fact_card, remove_fact_cards_for_message_ids

        card = normalize_turn_fact_card(
            {"fact": "阿澈告诉王五自己仍在复习，之后准备先吃饭。"},
            "王五",
            {
                "id": "u-2",
                "source_user_ids": ["u-1", "u-2"],
                "content": "第一条复习；第二条吃饭",
                "timestamp": 2_000,
            },
            {"id": "a-1", "content": "先去吃饭。", "timestamp": 3_000},
        )

        self.assertEqual(card["user_ids"], ["u-1", "u-2"])
        self.assertEqual(remove_fact_cards_for_message_ids([card], ["u-1"]), [])

    def test_uncovered_group_assistant_fact_keeps_actual_speaker(self):
        from tsuzuri.turn_fact_ledger import collect_uncovered_history_fact_records

        history = [
            {
                "role": "assistant",
                "character": "张三",
                "content": "冲绳夏季花展的向日葵从七月上旬开到八月下旬。",
                "id": "a-old-zhangsan",
                "timestamp": 1_000,
            },
            self._entry("user", "上一轮用户", "u-prev", 2_000_000),
            self._entry("assistant", "上一轮王五回复", "a-prev", 2_001_000),
            self._entry("user", "那花展呢？", "u-now", 2_002_000),
        ]

        records = collect_uncovered_history_fact_records(
            history,
            [],
            current_character="王五",
        )

        flower = next(row for row in records if "向日葵" in row["text"])
        self.assertEqual(flower["source_character"], "张三")
        self.assertTrue(flower["text"].startswith("张三早些时候提到"))

    pass  # upstream-only test removed

    def test_fact_prompt_does_not_define_supportive_advice_as_commitment(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "王五",
            self._entry("user", "剩下的我自己面对。", "u-1", 1_000),
            self._entry(
                "assistant",
                "实在撑不住的时候别硬扛，你那边又不是只有一个人。",
                "a-1",
                2_000,
            ),
        )

        self.assertIn("提醒、建议、安慰", prompt)
        self.assertIn("只有角色明确说自己之后会做什么", prompt)

    def test_proactive_fact_prompt_has_no_fake_user_utterance(self):
        from tsuzuri.turn_fact_ledger import build_fact_extraction_prompt

        prompt = build_fact_extraction_prompt(
            "王五",
            {},
            {
                "role": "assistant",
                "content": "饭吃完了吗，别又只顾着看电路。",
                "id": "a-proactive",
                "timestamp": 2_000,
                "proactive": True,
                "proactive_kind": "idle_followup",
            },
        )

        self.assertIn("阿澈本轮没有发送新消息", prompt)
        self.assertIn("王五主动接续对话", prompt)
        self.assertNotIn("阿澈：\n", prompt)

    pass  # upstream-only test removed

if __name__ == "__main__":
    unittest.main()
