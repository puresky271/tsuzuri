"""Implementation package for ``memory_core`` (SQLite four-layer memory store).

``MemoryCoreStore`` is composed from per-domain mixins; ``memory_core`` re-exports every name.
"""
import os as _os

FACADE_FILE = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "memory_core.py")
