"""Memory provider ABC for 群组 — Phase C.4 (2026-05-01).

灵感来自 hermes-agent agent/memory_provider.py L43-281。

设计目标：
- 把 memory_core.build_memory_context_pack 这个主要召回入口包成 ABC、未来可
  plug 替代实现（SSOT-only fallback / FTS5-only / 不同 embedding 模型）做 A/B
  对照、不需要每次 fork memory_core 7000+ 行单文件。
- 第一阶段只抽象 chat reply 路径上的 build_memory_context_pack。其他方法
  （commit_memory_surface 等）保持现有直接 call、避免一次性 refactor 太多。
- 选择性激活：默认 biomimetic、env TSUZURI_MEMORY_PROVIDER=ssot_only 切对照组。

主要 API（ABC）:
    name: str
    is_available() -> bool
    initialize(session_id, **kwargs) -> None  (默认 no-op)
    build_memory_context_pack(query_text, character, scope="normal", **kwargs) -> dict
    sync_turn(user_content, assistant_content, **kwargs) -> None  (默认 no-op)
    on_session_end(messages) -> None  (默认 no-op)
    shutdown() -> None  (默认 no-op)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryProvider(ABC):
    """Abstract base for memory backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'biomimetic', 'ssot_only', 'mock')."""
        ...

    def is_available(self) -> bool:
        """Provider can be activated. Default: always True for built-ins."""
        return True

    def initialize(self, session_id: str = "", **kwargs: Any) -> None:
        """Lazy init hook (open DB, warm cache, etc.). Default: no-op."""
        return

    @abstractmethod
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
        """Main retrieval entry — signature matches memory_core.MemoryCoreStore.build_memory_context_pack
        for drop-in compatibility. Returns dict with:
          - memory_rows: list[dict]
          - memory_read_bundle: dict (含 query_analysis 等)
          - memory_surface_plan: dict
          - guest_safe_profile_view: dict
        """
        ...

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        character: str = "",
        **kwargs: Any,
    ) -> None:
        """Persist turn after reply. Default: no-op (caller still writes memory_items)."""
        return

    def on_session_end(self, messages: list) -> None:
        """End-of-session hook. Default: no-op."""
        return

    def shutdown(self) -> None:
        """Clean shutdown. Default: no-op."""
        return


__all__ = ["MemoryProvider"]

