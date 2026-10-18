"""tsuzuri — long-term conversational memory and a context compiler for multi-agent character chat.

Quick start::

    from tsuzuri import Tsuzuri

    memory = Tsuzuri("./.tsuzuri")              # offline: no API key needed
    memory.chat("林栖", "我下周三要去面试，有点紧张")
    turn = memory.chat("林栖", "你还记得我下周有什么安排吗？")
    print(turn.reply)

See ``docs/`` for the architecture, configuration and hosting hooks.
"""
from __future__ import annotations

__version__ = "0.1.0"

from tsuzuri.engine import Tsuzuri, TurnResult
from tsuzuri.offline import offline_reply, openai_compatible
from tsuzuri.persona_registry import configure_personas

__all__ = ["Tsuzuri", "TurnResult", "configure_personas", "offline_reply", "openai_compatible", "__version__"]
