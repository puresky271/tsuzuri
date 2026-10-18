import os
import tempfile
import unittest

from tsuzuri.memory_core import MemoryCoreStore, _canonicalize_developer_fact, _rewrite_fact_via_llm


class FactRewriteSubjectTests(unittest.TestCase):
    def test_second_person_quote_preserves_speaker_and_addressee(self):
        raw = "用户你以后会遇到那样的人的，到时候让对方带你一起去看花火就是了，额，我当然指的是"

        normalized = _canonicalize_developer_fact(raw, character="李四")

        self.assertEqual(
            normalized,
            "阿澈对李四说：你以后会遇到那样的人的，到时候让对方带你一起去看花火就是了，额，我当然指的是",
        )

    def test_legacy_concatenated_subject_is_repaired(self):
        raw = "阿澈你以后会遇到那样的人的，到时候让对方带你一起去看花火就是了"

        normalized = _canonicalize_developer_fact(raw, character="李四")

        self.assertEqual(
            normalized,
            "阿澈对李四说：你以后会遇到那样的人的，到时候让对方带你一起去看花火就是了",
        )

    def test_rewrite_prompt_keeps_prediction_as_speech_act(self):
        captured = {}

        def caller(prompt: str) -> str:
            captured["prompt"] = prompt
            return "阿澈鼓励李四以后和喜欢的人一起看花火。"

        raw = "阿澈对李四说：你以后会遇到那样的人的，到时候让对方带你一起去看花火就是了"
        rewritten = _rewrite_fact_via_llm(raw, llm_caller=caller, character="李四")

        self.assertEqual(rewritten, "阿澈鼓励李四以后和喜欢的人一起看花火。")
        self.assertIn(f"原话：{raw}", captured["prompt"])
        self.assertIn("先区分说话者与命题主语", captured["prompt"])
        self.assertIn("不能把尚未发生的内容写成确定事实", captured["prompt"])

    pass  # upstream-only test removed


class TurnFactCardReuseTests(unittest.TestCase):
    """方案 D (2026-09-07): 显式记忆请求优先复用 turn_fact_card.fact。

    事故背景：user 说「我喜欢去想象你在窗边看书的样子…」命中
    _EXPLICIT_MEMORY_WRITE_RE 的裸「我喜欢」，整句原话被 _baseline_subject_normalize
    裸拼「阿澈」前缀成呼格歧义句，rewrite LLM 主客颠倒成「张三喜欢想象阿澈…」
    并直写 semantic。修复后显式路径用当轮 TurnFact card（已正确归因）作事实源，
    且不再过二次 rewrite。
    """

    _USER_TEXT = "知道吗，虽然我们隔得很远，但我喜欢去想象你在窗边看书的样子。那会让我开心好久好久"

    def _run_plan(self, tmp, meta):
        core = MemoryCoreStore(
            db_path=os.path.join(tmp, "memory.db"),
            snapshot_path=os.path.join(tmp, "snapshot.json"),
        )
        return core.plan_turn_memory_writes(
            "张三",
            user_text=self._USER_TEXT,
            assistant_reply="真的吗…其实我也是。",
            event_nodes=[],
            meta=meta,
        )

    def test_explicit_request_reuses_card_fact_and_skips_rewrite(self):
        card_fact = "阿澈喜欢想象张三在窗边看书的样子，这让他开心好久。"

        def caller(prompt: str) -> str:
            raise AssertionError(f"card-sourced fact 不应再过 rewrite: {prompt[:80]}")

        with tempfile.TemporaryDirectory() as tmp:
            plan = self._run_plan(tmp, {
                "latest_user_msg": self._USER_TEXT,
                "turn_fact_card": {"fact": card_fact},
                "fact_rewrite_llm_caller": caller,
            })

        sem_texts = [r.get("text", "") for r in plan.get("records", []) if r.get("layer") == "semantic"]
        self.assertIn(card_fact, sem_texts, f"semantic 应直写 card fact、actual: {sem_texts}")
        self.assertTrue(
            all("知道吗" not in t for t in sem_texts),
            f"不应再写入原话 verbatim: {sem_texts}",
        )

    def test_explicit_request_falls_back_when_card_missing(self):
        prompts = []

        def caller(prompt: str) -> str:
            prompts.append(prompt)
            return "阿澈喜欢想象张三在窗边看书的样子。"

        with tempfile.TemporaryDirectory() as tmp:
            plan = self._run_plan(tmp, {
                "latest_user_msg": self._USER_TEXT,
                "fact_rewrite_llm_caller": caller,
            })

        self.assertTrue(prompts, "card 缺失时应回退 verbatim+rewrite 路径")
        self.assertTrue(any("原话：阿澈说：知道吗" in p for p in prompts), prompts)
        sem_texts = [r.get("text", "") for r in plan.get("records", []) if r.get("layer") == "semantic"]
        self.assertIn("阿澈喜欢想象张三在窗边看书的样子。", sem_texts)

    def test_explicit_request_falls_back_when_card_fact_is_residue(self):
        prompts = []

        def caller(prompt: str) -> str:
            prompts.append(prompt)
            return "阿澈喜欢想象张三在窗边看书的样子。"

        with tempfile.TemporaryDirectory() as tmp:
            plan = self._run_plan(tmp, {
                "latest_user_msg": self._USER_TEXT,
                "turn_fact_card": {"fact": "那好吧，我喜欢你。"},
                "fact_rewrite_llm_caller": caller,
            })

        self.assertTrue(prompts, "card fact 是对话残留时应回退 verbatim+rewrite 路径")
        sem_texts = [r.get("text", "") for r in plan.get("records", []) if r.get("layer") == "semantic"]
        self.assertNotIn("那好吧，我喜欢你。", sem_texts)


class FactRewriteSubjectGuardTests(unittest.TestCase):
    """FactRewrite 修复 (2026-09-07): 引语形态归因 + 主客颠倒输出拦截。"""

    def test_rewrite_rejects_subject_swapped_output(self):
        """输入主语是阿澈（含「阿澈说：…」引语形态）、输出却以角色开头 → 判主客颠倒、fallback 原句。"""

        def caller(prompt: str) -> str:
            return "张三喜欢想象阿澈在窗边看书的样子。"

        raw = "阿澈说：知道吗，虽然我们隔得很远，但我喜欢去想象你在窗边看书的样子。那会让我开心好久好久。"
        rewritten = _rewrite_fact_via_llm(raw, llm_caller=caller, character="张三")

        self.assertEqual(rewritten, raw)

    def test_rewrite_accepts_character_subject_when_input_not_developer_led(self):
        """输入不以阿澈开头时，角色开头的输出不受主语校验拦截。"""

        def caller(prompt: str) -> str:
            return "李四的歌声被阿澈称赞。"

        raw = "李四的歌声真好听，阿澈从第一次集训开始就一直夸个不停，今天见面又说了一次"
        rewritten = _rewrite_fact_via_llm(raw, llm_caller=caller, character="李四")

        self.assertEqual(rewritten, "李四的歌声被阿澈称赞。")

    def test_rewrite_prompt_covers_quote_form_attribution(self):
        captured = {}

        def caller(prompt: str) -> str:
            captured["prompt"] = prompt
            return "阿澈喜欢想象张三在窗边看书的样子。"

        raw = "阿澈说：知道吗，虽然我们隔得很远，但我喜欢去想象你在窗边看书的样子。"
        rewritten = _rewrite_fact_via_llm(raw, llm_caller=caller, character="张三")

        self.assertEqual(rewritten, "阿澈喜欢想象张三在窗边看书的样子。")
        self.assertIn("引语里的「我」= 阿澈", captured["prompt"])
        self.assertIn("主客颠倒", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
