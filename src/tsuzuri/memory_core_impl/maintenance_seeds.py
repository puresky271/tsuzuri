"""Operator-supplied seeds for ``maintain_memory_evidence_pool`` and their recall gates.

The evidence-pool maintenance used to hard-code a handful of hand-curated
developer facts, key events and explicit-recall archives (plus the source row
ids they consolidate). Those are deployment data, not engine logic, so they now
live in a local JSON file that is never committed:

- ``MEM_MAINTENANCE_SEEDS_PATH`` (env) if set, else
- ``<repo root>/memory_maintenance_seeds.local.json``.

A missing or malformed file means "no seeds": maintenance still runs its
generic TTL / profile-coverage / low-value passes, and the recall gates below
fall back to their generic terms only.

File shape::

    {
      "profile_candidates": [{"text", "source_item_ids", "target_path"}],
      "key_events": [{"date", "event", "importance", "source_item_ids"}],
      "consolidated_items": [{"item_id", "character", "layer", ..., "source_item_ids"}],
      "exclude_source_ids": {"<item_id>": "<reason>"},
      "recall_gates": {
        "system_origin_anchor_terms": ["..."],
        "meta_topic_profile_terms": ["..."],
        "explicit_archives": [
          {"item_id_marker": "...", "text_marker": "...", "anchor": "<regex>",
           "reject_system_instruction_topic": false}
        ]
      }
    }
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading

_DEFAULT_FILENAME = "memory_maintenance_seeds.local.json"
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_cache_lock = threading.Lock()
_cache: dict[str, object] = {"key": None, "seeds": None}


def maintenance_seeds_path() -> str:
    override = str(os.getenv("MEM_MAINTENANCE_SEEDS_PATH", "") or "").strip()
    return override or os.path.join(_REPO_ROOT, _DEFAULT_FILENAME)


def _empty_seeds() -> dict:
    return {
        "profile_candidates": [],
        "key_events": [],
        "consolidated_items": [],
        "exclude_source_ids": {},
        "recall_gates": {
            "system_origin_anchor_terms": [],
            "meta_topic_profile_terms": [],
            "explicit_archives": [],
        },
    }


def _dict_rows(value) -> list[dict]:
    return [dict(x) for x in value if isinstance(x, dict)] if isinstance(value, list) else []


def _str_terms(value) -> list[str]:
    return [str(x).strip() for x in value if str(x or "").strip()] if isinstance(value, list) else []


def _normalize(raw) -> dict:
    seeds = _empty_seeds()
    if not isinstance(raw, dict):
        return seeds
    for key in ("profile_candidates", "key_events", "consolidated_items"):
        seeds[key] = _dict_rows(raw.get(key))
    excludes = raw.get("exclude_source_ids")
    if isinstance(excludes, dict):
        seeds["exclude_source_ids"] = {
            str(k).strip(): str(v or "") for k, v in excludes.items() if str(k or "").strip()
        }
    gates = raw.get("recall_gates")
    if isinstance(gates, dict):
        archives = []
        for row in _dict_rows(gates.get("explicit_archives")):
            try:
                anchor = re.compile(str(row.get("anchor", "") or "")) if row.get("anchor") else None
            except re.error:
                anchor = None
            archives.append({
                "item_id_marker": str(row.get("item_id_marker", "") or "").strip(),
                "text_marker": str(row.get("text_marker", "") or "").strip(),
                "anchor": anchor,
                "reject_system_instruction_topic": bool(row.get("reject_system_instruction_topic", False)),
            })
        seeds["recall_gates"] = {
            "system_origin_anchor_terms": _str_terms(gates.get("system_origin_anchor_terms")),
            "meta_topic_profile_terms": _str_terms(gates.get("meta_topic_profile_terms")),
            "explicit_archives": archives,
        }
    return seeds


def _load_cached() -> dict:
    """Shared, read-only normalized seeds; re-parsed when the path, mtime or size changes."""
    path = maintenance_seeds_path()
    try:
        st = os.stat(path)
        mtime = (st.st_mtime_ns, st.st_size)  # size too: coarse timestamps (e.g. Windows) can repeat
    except OSError:
        mtime = None
    key = (path, mtime)
    with _cache_lock:
        if _cache["key"] == key and isinstance(_cache["seeds"], dict):
            return _cache["seeds"]  # type: ignore[return-value]
    seeds = _empty_seeds()
    if mtime is not None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                seeds = _normalize(json.load(fh))
        except (OSError, ValueError):
            seeds = _empty_seeds()
    gates = seeds["recall_gates"]
    gates["system_origin_anchor_re"] = _terms_regex(gates["system_origin_anchor_terms"])
    gates["meta_topic_profile_re"] = _terms_regex(gates["meta_topic_profile_terms"])
    with _cache_lock:
        _cache["key"] = key
        _cache["seeds"] = seeds
    return seeds


def load_maintenance_seeds() -> dict:
    """Return a private (mutable) copy of the normalized seeds. Never raises."""
    return copy.deepcopy(_load_cached())


def recall_gates() -> dict:
    """Read-only gate view for the recall hot path (do not mutate)."""
    return _load_cached()["recall_gates"]  # type: ignore[index]


def _terms_regex(terms: list[str]) -> re.Pattern | None:
    cleaned = [re.escape(t) for t in terms if t]
    return re.compile("(" + "|".join(cleaned) + ")") if cleaned else None


__all__ = ["maintenance_seeds_path", "load_maintenance_seeds", "recall_gates"]
