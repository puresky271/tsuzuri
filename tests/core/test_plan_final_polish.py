# -*- coding: utf-8 -*-
"""Stage P polish / Stage R repeated-title repair / publish side-effect tests."""
from __future__ import annotations

import copy
import json
import os
import tempfile
import types
import unittest
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo


WORLD_TZ = ZoneInfo("UTC")


def _fake_client(content: str | list[str]):
    calls = []
    contents = list(content) if isinstance(content, list) else [content]

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            index = min(len(calls) - 1, len(contents) - 1)
            message = types.SimpleNamespace(content=contents[index])
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice], usage=None)

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()),
    ), calls


def _step(title: str, location: str = "自宅", topic: str = "日常") -> dict:
    return {"title": title, "location": location, "topic": topic}


def _raw_batch() -> dict:
    batch = {}
    for char in ("张三", "赵六", "李四", "王五", "钱七"):
        batch[char] = [_step(f"{char}活动{i}") for i in range(16)]
    return batch


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


pass  # upstream-only test removed


if __name__ == "__main__":
    unittest.main()
