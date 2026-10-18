from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own data directory and the built-in demo cast."""
    from tsuzuri import memory_host, persona_registry
    from tsuzuri.memory_core_impl import store as store_module

    monkeypatch.setenv("TSUZURI_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("EMBEDDING_ENABLED", os.environ.get("EMBEDDING_ENABLED", "0"))
    monkeypatch.setenv("MEMORY_HOST_MODULE", "none")
    monkeypatch.setattr(store_module, "_MEMORY_CORE_SINGLETON", None)
    memory_host.reset(autoload=False)
    persona_registry.configure_personas(reset=True)
    yield
    memory_host.reset(autoload=False)
