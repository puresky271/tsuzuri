"""Engine-level tests run against a fixed test cast (tests/fixtures/test_cast.json)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Engine suites assert wall-clock dates in a fixed UTC+9 world clock; set it before tsuzuri imports.
os.environ.setdefault("WORLD_TIMEZONE", "Asia/Seoul")
os.environ.setdefault("WORLD_TIMEZONE_LABEL", "WORLD_TZ")

_CAST = json.loads((Path(__file__).resolve().parent.parent / "fixtures" / "test_cast.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _test_cast(tmp_path, monkeypatch):
    from tsuzuri import memory_host, persona_registry

    monkeypatch.setenv("TSUZURI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EMBEDDING_ENABLED", os.environ.get("EMBEDDING_ENABLED", "0"))
    monkeypatch.setenv("MEMORY_HOST_MODULE", "none")
    memory_host.reset(autoload=False)
    persona_registry.configure_personas(_CAST, reset=True)
    yield
    persona_registry.configure_personas(reset=True)
    memory_host.reset(autoload=False)
