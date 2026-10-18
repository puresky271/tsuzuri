"""Imports, embedding engine probe and memory feature flags.

Part of the ``memory_core`` package split; ``memory_core`` re-exports every name.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from tsuzuri.reliable_io import interprocess_file_lock
from tsuzuri.sqlite_reliability import connect_sqlite, is_sqlite_busy_error
from tsuzuri.secrets_loader import env_bool as _bool_env
from tsuzuri.time_source import local_now, now_world, utc_now, utc_now_naive


# Phase G.M2 (2026-05-02): embedding 模块 module-level 单次 import + cache
# 之前三处独立 from embedding_engine import ... + 各自外层 except: pass、
# silent fail 无日志无 metric。现在 module 顶部一次 import、降级状态可观测。
_embedding_engine_available: bool = False
_embedding_import_error: str = ""
_emb_is_enabled = None
_emb_encode = None
_emb_decode = None  # text_to_embedding
_emb_cos = None     # cosine_similarity
_emb_to_text = None  # embedding_to_text
# Phase A (2026-05-14): BGE-M3 sparse 双输出
_emb_is_m3_mode = None
_emb_encode_dense = None
_emb_encode_sparse = None
_emb_encode_both = None
_emb_sparse_to_text = None
_emb_text_to_sparse = None
_emb_sparse_dot = None
_emb_declared_model_id = None
_emb_declared_dim = None
try:
    from tsuzuri.embedding_engine import (
        is_embedding_enabled as _emb_is_enabled,
        encode_dense as _emb_encode,    # Phase A: M3-aware unified encoder (bge-m3 1024 or bge-small 512)
        text_to_embedding as _emb_decode,
        cosine_similarity as _emb_cos,
        embedding_to_text as _emb_to_text,
        # Phase A: M3
        is_m3_mode as _emb_is_m3_mode,
        encode_sparse as _emb_encode_sparse,
        encode_both as _emb_encode_both,
        sparse_to_text as _emb_sparse_to_text,
        text_to_sparse as _emb_text_to_sparse,
        sparse_dot_product as _emb_sparse_dot,
        declared_embedding_model_id as _emb_declared_model_id,
        declared_embedding_dimension as _emb_declared_dim,
    )
    _embedding_engine_available = True
except Exception as _emb_imp_err:
    _embedding_import_error = f"{type(_emb_imp_err).__name__}: {_emb_imp_err}"


def _embedding_callable() -> bool:
    """检查 embedding 是否真正可用（既导入成功又 enabled flag）。"""
    if not _embedding_engine_available:
        return False
    try:
        explicit = os.getenv("EMBEDDING_ENABLED", "").strip().lower()
        if explicit in ("false", "0", "no", "off"):
            return False
        if callable(_emb_is_m3_mode) and bool(_emb_is_m3_mode()):
            return True
        return bool(_emb_is_enabled()) if callable(_emb_is_enabled) else False
    except Exception:
        return False


def _current_embedding_spec() -> tuple[str, int]:
    try:
        model_id = str(_emb_declared_model_id() if callable(_emb_declared_model_id) else "")
        dimension = int(_emb_declared_dim() if callable(_emb_declared_dim) else 0)
        return model_id, dimension
    except Exception:
        return "", 0


def _embedding_row_matches_current(row: dict) -> bool:
    model_id, dimension = _current_embedding_spec()
    return bool(
        model_id
        and dimension > 0
        and str(row.get("embedding_model_id", "") or "") == model_id
        and int(_safe_int(row.get("embedding_dim", 0), 0)) == dimension
    )


def _utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def mem_sqlite_enabled() -> bool:
    return _bool_env("MEM_SQLITE_ENABLE", True)


def mem_evolve_shadow_only() -> bool:
    return _bool_env("MEM_EVOLVE_SHADOW_ONLY", False)


def mem_llm_selector_enabled() -> bool:
    return _bool_env("MEM_LLM_SELECTOR_ENABLE", True)


def mem_policy_llm_enabled() -> bool:
    return _bool_env("MEM_POLICY_LLM_ENABLE", True)


def mem_actr_recency_enabled() -> bool:
    return _bool_env("MEM_ACTR_RECENCY_ENABLE", True)


def mem_spreading_activation_enabled() -> bool:
    return _bool_env("MEM_SPREADING_ACTIVATION_ENABLE", True)


def mem_lateral_inhibition_enabled() -> bool:
    return _bool_env("MEM_LATERAL_INHIBITION_ENABLE", True)


def mem_motivation_boost_enabled() -> bool:
    return _bool_env("MEM_MOTIVATION_BOOST_ENABLE", True)


def mem_context_dual_read_enabled() -> bool:
    return _bool_env("MEM_CONTEXT_DUAL_READ_ENABLE", True)


def mem_context_primary_enabled() -> bool:
    return _bool_env("MEM_CONTEXT_PRIMARY_ENABLE", False)


def mem_json_online_export_enabled() -> bool:
    return _bool_env("MEM_JSON_ONLINE_EXPORT_ENABLE", False)


def mem_json_bootstrap_fallback_enabled() -> bool:
    return _bool_env("MEM_JSON_BOOTSTRAP_FALLBACK_ENABLE", True)


def mem_snapshot_backup_enabled() -> bool:
    return _bool_env("MEM_SNAPSHOT_BACKUP_ENABLE", True)


def mem_snapshot_backup_interval_hours() -> int:
    return max(1, _safe_int(os.getenv("MEM_SNAPSHOT_BACKUP_INTERVAL_HOURS", "24"), 24))


def mem_snapshot_backup_keep() -> int:
    return max(1, _safe_int(os.getenv("MEM_SNAPSHOT_BACKUP_KEEP", "30"), 30))


def mem_evolve_min_shadow_days() -> int:
    return max(0, _safe_int(os.getenv("MEM_EVOLVE_MIN_SHADOW_DAYS", "7"), 7))


# ─────────────────────────────────────────────────────────────
# Phase R.2 (2026-07-25) — evidence-bound LLM memory rethink flags
# 借鉴 Letta sleeptime agent，让 LLM 在 KAIROS dream cycle 主动
# rewrite/merge/archive semantic 层记忆。默认启用并 live apply；
# canon/OOC、证据校验和原子 CAS 始终生效，shadow 可由环境变量显式恢复。
# ─────────────────────────────────────────────────────────────

def mem_rethink_enabled() -> bool:
    """是否启用 LLM rethink。默认开启。"""
    return _bool_env("MEM_RETHINK_ENABLED", True)


def mem_rethink_shadow_only() -> bool:
    """是否仅 shadow 模式（只写 audit 不真改）。默认关闭，可显式回退。"""
    return _bool_env("MEM_RETHINK_SHADOW_ONLY", False)


def mem_rethink_min_shadow_days() -> int:
    """shadow 强制窗口天数。默认 0；可显式设置观察期。"""
    return max(0, _safe_int(os.getenv("MEM_RETHINK_MIN_SHADOW_DAYS", "0"), 0))


def mem_rethink_k() -> int:
    """单次 rethink 处理的候选数。"""
    return max(1, min(50, _safe_int(os.getenv("MEM_RETHINK_K", "15"), 15)))


def mem_rethink_max_actions() -> int:
    """单次 rethink LLM 可产出的最大 action 数。"""
    return max(1, min(20, _safe_int(os.getenv("MEM_RETHINK_MAX_ACTIONS", "8"), 8)))


def mem_rethink_max_iterations() -> int:
    """一次 KAIROS rethink loop 的最大 pass 数。"""
    return max(1, min(5, _safe_int(os.getenv("MEM_RETHINK_MAX_ITERATIONS", "3"), 3)))


def mem_rethink_intraday_enabled() -> bool:
    """P2 预留的日间轻 pass 开关；本阶段不接入调度。"""
    return _bool_env("MEM_RETHINK_INTRADAY", False)


def mem_core_summary_enabled() -> bool:
    """是否允许 KAIROS 重建 per-character living summary。"""
    return _bool_env("MEM_CORE_SUMMARY_ENABLE", False)


def mem_core_summary_render_enabled() -> bool:
    """是否把已存在的 living summary 渲染进 CURRENT_STATE_FRAME。"""
    return _bool_env("MEM_CORE_SUMMARY_RENDER", False)


def mem_forgetting_enabled() -> bool:
    """Whether KAIROS may scan archived memories for Phase E/P4."""
    return _bool_env("MEM_FORGETTING_ENABLED", False)


def mem_forgetting_apply_enabled() -> bool:
    """Whether Phase E/P4 may persist frozen/tombstone transitions."""
    return _bool_env("MEM_FORGETTING_APPLY", False)


def mem_rethink_ooc_threshold() -> float:
    """OOC 二审阈值。new_text 的 ooc.confidence > 此值 → reject。"""
    raw = os.getenv("MEM_RETHINK_OOC_THRESHOLD", "0.45")
    return _safe_float(raw, 0.45)
