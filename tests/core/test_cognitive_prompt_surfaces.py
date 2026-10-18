import unittest
from unittest.mock import patch


pass  # upstream-only test removed


class CognitivePromptSurfaceTests(unittest.TestCase):
    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    def test_workspace_message_precedes_recent_dialogue_and_guidance_is_last(self):
        from tsuzuri.turn_context_workspace import assemble_workspace_messages, compile_turn_execution_packet

        packet = compile_turn_execution_packet(
            "张三",
            "我一直很想你。",
            reply_guidance="直接回应阿澈表达的想念，说出你真实的感受。",
        )
        messages = assemble_workspace_messages(
            persona_prompt="PERSONA",
            current_state_frame="STATE",
            on_demand_evidence="EVIDENCE",
            cognitive_workspace='<cognitive-workspace transient="true">\nINNER\n</cognitive-workspace>',
            recent_dialogue_guard="RECENT_GUARD",
            conversation_messages=[{"role": "user", "content": "我一直很想你。"}],
            turn_execution_packet=packet["text"],
        )
        contents = [row["content"] for row in messages]
        self.assertLess(contents.index("EVIDENCE"), next(i for i, text in enumerate(contents) if "<cognitive-workspace" in text))
        self.assertLess(next(i for i, text in enumerate(contents) if "<cognitive-workspace" in text), contents.index("RECENT_GUARD"))
        self.assertTrue(messages[-1]["content"].rstrip().endswith("</turn-reply-guidance>"))
        self.assertIn("直接回应阿澈表达的想念", messages[-1]["content"])
        self.assertNotIn("【回复目标】", messages[-1]["content"])


pass  # upstream-only test removed


pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
