"""Diagnostics go to the ``tsuzuri`` logger instead of stdout.

Engine modules report progress with ``emit(...)`` (print-compatible signature). Routine
messages are DEBUG; messages that mention a failure are WARNING. Enable them with
``logging.getLogger("tsuzuri").setLevel(logging.DEBUG)`` plus a handler.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("tsuzuri")
_PROBLEM_WORDS = ("fail", "error", "exception", "traceback", "invalid", "corrupt")


def emit(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    del end, file, flush
    message = sep.join(str(a) for a in args)
    lowered = message.lower()
    level = logging.WARNING if any(word in lowered for word in _PROBLEM_WORDS) else logging.DEBUG
    logger.log(level, message)


__all__ = ["emit", "logger"]
