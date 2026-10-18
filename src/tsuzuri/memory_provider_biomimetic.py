"""BiomimeticMemoryProvider — wraps the existing memory_core retrieval.

Phase C.4 (2026-05-01) — default provider. Pure wrapper、行为与直接 call
memory_core 完全等价。registry 默认激活的就是这个。
"""
from __future__ import annotations

from typing import Any

from tsuzuri.memory_provider import MemoryProvider


class BiomimeticMemoryProvider(MemoryProvider):
    """Default — 仿生召回 + KAIROS、对应现有 memory_core.MemoryCoreStore。"""

    @property
    def name(self) -> str:
        return "biomimetic"

    def is_available(self) -> bool:
        return True

    def build_memory_context_pack(
        self,
        character: str,
        query_text: str,
        query_bundle: dict | None = None,
        max_items: int = 6,
        *,
        llm_selector: Any = None,
        world_signals: dict | None = None,
        risk_bundle: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        from tsuzuri.memory_core import get_memory_core
        from tsuzuri.project_paths import DEV_MEMORY_PATH, MEMORY_CORE_DB_PATH

        store = get_memory_core(snapshot_path=DEV_MEMORY_PATH, db_path=MEMORY_CORE_DB_PATH)
        result = store.build_memory_context_pack(
            character,
            query_text,
            query_bundle=query_bundle,
            max_items=max_items,
            llm_selector=llm_selector,
            world_signals=world_signals,
            risk_bundle=risk_bundle,
        )
        return dict(result) if isinstance(result, dict) else {}


__all__ = ["BiomimeticMemoryProvider"]
