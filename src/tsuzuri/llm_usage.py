"""Shared LLM usage / cache-hit observability.

This module was extracted from chat_server so lower-level modules such as
宿主应用, memory_core, and cluster_memory can record usage without importing the
web server.
"""
from tsuzuri._log import emit as _tsz_emit

import threading
import time
from typing import Any

llm_usage_stats_lock = threading.Lock()
llm_usage_stats: dict[str, Any] = {
    "total_calls": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_cached_tokens": 0,
    "by_label": {},   # label -> {calls, prompt, completion, cached}
    "recent": [],     # most recent 100 entries (ts, label, prompt, cached, hit_rate)
}


def record_llm_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    *,
    label: str = "",
) -> None:
    """Record one LLM call's token usage into the module-level stats."""
    try:
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        cached_tokens = max(0, int(cached_tokens or 0))
        label = str(label or "unknown").strip() or "unknown"
        with llm_usage_stats_lock:
            llm_usage_stats["total_calls"] += 1
            llm_usage_stats["total_prompt_tokens"] += prompt_tokens
            llm_usage_stats["total_completion_tokens"] += completion_tokens
            llm_usage_stats["total_cached_tokens"] += cached_tokens
            by_label = llm_usage_stats["by_label"].setdefault(
                label, {"calls": 0, "prompt": 0, "completion": 0, "cached": 0}
            )
            by_label["calls"] += 1
            by_label["prompt"] += prompt_tokens
            by_label["completion"] += completion_tokens
            by_label["cached"] += cached_tokens
            hit = (cached_tokens / prompt_tokens) if prompt_tokens > 0 else 0.0
            llm_usage_stats["recent"].append({
                "ts": int(time.time() * 1000),
                "label": label,
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "cached": cached_tokens,
                "hit_rate": round(hit, 3),
            })
            if len(llm_usage_stats["recent"]) > 100:
                llm_usage_stats["recent"] = llm_usage_stats["recent"][-100:]
    except Exception:
        pass


def capture_llm_usage(resp, *, label: str = "") -> None:
    """Extract OpenAI / DeepSeek usage fields, log, and record them."""
    try:
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        comp = int(getattr(usage, "completion_tokens", 0) or 0)
        # OpenAI path: usage.prompt_tokens_details.cached_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        cached = 0
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        # DeepSeek path: usage.prompt_cache_hit_tokens
        if not cached:
            cached = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        hit = (cached / prompt) if prompt > 0 else 0.0
        _tsz_emit(
            f"[LLM][usage] prompt={prompt} completion={comp} cached={cached} "
            f"({hit:.0%}) label={label or 'unknown'}",
            flush=True,
        )
        record_llm_usage(prompt, comp, cached, label=label)
    except Exception as err:
        _tsz_emit(f"[LLM][usage] capture error: {err}", flush=True)
