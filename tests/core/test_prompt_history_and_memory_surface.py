import os
import unittest
from datetime import datetime, timezone, timedelta


_WORLD_TZ = timezone(timedelta(hours=9))


def _entry(role, content, *, world_dt, character=""):
    row = {
        "role": role,
        "content": content,
        "timestamp": int(world_dt.timestamp() * 1000),
    }
    if character:
        row["character"] = character
    return row


class PromptHistoryAndMemorySurfaceTests(unittest.TestCase):
    def setUp(self):
        self._orig_mock = os.environ.get("DEBUG_MOCK_DATE")
        os.environ["DEBUG_MOCK_DATE"] = "2026-06-29T18:00:00+09:00"

    def tearDown(self):
        if self._orig_mock is None:
            os.environ.pop("DEBUG_MOCK_DATE", None)
        else:
            os.environ["DEBUG_MOCK_DATE"] = self._orig_mock

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    pass  # upstream-only test removed

    def test_historical_recap_normalizes_stale_current_wording(self):
        from tsuzuri.chat_compact import _normalize_recap_temporal_language

        text = "阿澈目前在外散步；用户现在处于比较空闲的状态。"
        normalized = _normalize_recap_temporal_language(text)
        self.assertIn("阿澈当时在外散步", normalized)
        self.assertIn("用户当时处于比较空闲的状态", normalized)
        self.assertNotIn("目前", normalized)
        self.assertNotIn("现在", normalized)

    pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
