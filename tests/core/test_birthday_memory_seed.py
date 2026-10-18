"""Phase 5 单元测试：生日 memory 预播种。

测试 town_sim._seed_birthday_today_memories：
  · 非生日日 → 0 written
  · 生日日 → 5 char × 寿星 = N written
  · 寿星自身 memory 含 SSOT seed_hint
  · peer memory 含 Phase 1 _BIRTHDAY_PEER_TONE 个性化文案
  · idempotent: 同日重复调用不产生 duplicate
"""
from __future__ import annotations
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

_WORLD_TZ = timezone(timedelta(hours=9))


pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main(verbosity=2)
