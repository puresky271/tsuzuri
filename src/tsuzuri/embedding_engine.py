"""Embedding Engine: Local embedding wrapper for semantic memory recall.

Two backends supported:
  · Default (legacy): BAAI/bge-small-zh-v1.5 via sentence_transformers (~95MB, 512-dim)
  · M3 mode (Phase A 2026-05-14): BAAI/bge-m3 via FlagEmbedding (~2.3GB, 1024-dim
    dense + sparse 双输出、sparse 替代传统 BM25、自带 IDF + 学习权重)

Lazy-loads model on first use. Graceful fallback if not installed or disabled.

Environment variables:
  EMBEDDING_ENABLED=true        — enable embedding (default: false)
  EMBEDDING_MODEL=...           — override legacy model name (default: BAAI/bge-small-zh-v1.5)
  EMBEDDING_MODEL_M3=true       — Phase A: 切换到 BGE-M3（dense+sparse 双输出）
  EMBEDDING_M3_PATH=...         — M3 模型路径（默认 BAAI/bge-m3、可指本地 cache）
  EMBEDDING_CPU_RESERVE=n       — 给前台预留的逻辑核数（0 = 不限制；默认 max(2, cpu//4)）
  EMBEDDING_TORCH_THREADS=n     — 直接指定 torch 推理线程数（优先于 RESERVE）
  EMBEDDING_MAX_CONCURRENCY=n   — 全局 encode 并发上限（默认 1 = 严格串行排队）
"""
from __future__ import annotations
from tsuzuri._log import emit as _tsz_emit

import base64
import json
import os
import time
import threading
import warnings
from typing import Optional

import numpy as np

_EMBEDDING_ENABLED_KEY = "EMBEDDING_ENABLED"
_EMBEDDING_M3_KEY = "EMBEDDING_MODEL_M3"
_EMBEDDING_M3_PATH_KEY = "EMBEDDING_M3_PATH"
_DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
_DEFAULT_M3_MODEL = "BAAI/bge-m3"
_EMBEDDING_DIM = 512        # legacy bge-small dim
_EMBEDDING_DIM_M3 = 1024    # bge-m3 dim

_model_lock = threading.Lock()
_model_instance = None      # legacy SentenceTransformer
_m3_model_instance = None   # BGEM3FlagModel
_model_last_fail: float = 0.0  # monotonic timestamp of last failure
_m3_last_fail: float = 0.0
_MODEL_RETRY_COOLDOWN = 60.0   # retry after 60s on failure


# ═══════════════════════════════════════════════════════════════════════════
# CPU 让步层 (2026-09-17)
# ═══════════════════════════════════════════════════════════════════════════
# PyTorch CPU 版默认 torch.set_num_threads(os.cpu_count())。本机实测：
#   os.cpu_count() = 16 / torch.get_num_threads() = 16（torch 2.11.0+cpu、无 GPU）
# 一次批量 embedding（如 turn_semantic_scorer 冷缓存回填的 56 个 batch）会把 16 个
# 逻辑核全部跑满。同机播放的过载模式音源是 Hi-Res FLAC（44 首平均 88MB、最大
# 190MB、96kHz/24bit、纯软件解码），音频解码线程与 Electron 渲染线程同时被饿死，
# 表现为「声音和画面一起卡」。
# 这里给推理留出前台预留核；代价是批量编码变慢，属后台任务可接受范围。
_CPU_POLITENESS_APPLIED = False


def cpu_thread_budget() -> tuple[int, int]:
    """返回 (推理线程数, 总逻辑核数)。

    EMBEDDING_TORCH_THREADS 显式指定时优先（>0）；EMBEDDING_CPU_RESERVE 显式指定
    前台预留核数（0 = 不限制、保持 PyTorch 默认）；都未设时自动预留 max(2, total//4)。
    """
    total = os.cpu_count() or 1
    explicit_threads = os.getenv("EMBEDDING_TORCH_THREADS", "").strip()
    if explicit_threads:
        try:
            n = int(explicit_threads)
            if n > 0:
                return min(n, total), total
        except ValueError:
            pass
    raw_reserve = os.getenv("EMBEDDING_CPU_RESERVE", "").strip()
    reserve = max(2, total // 4)
    if raw_reserve:
        try:
            reserve = max(0, int(raw_reserve))
        except ValueError:
            pass
    if reserve == 0:
        return total, total
    return max(1, total - reserve), total


def apply_cpu_politeness() -> None:
    """幂等：把 torch 推理线程数压到给前台留出核心的水平。

    在模型构造前（或首次推理前）调用即可；torch 未安装时静默跳过。
    显式设 EMBEDDING_CPU_RESERVE=0 可退回 PyTorch 默认的全核行为。
    interop 线程数同步压到同一预算（PyTorch 规定其只能在首次并行工作前
    设置一次、重复或过晚调用会抛错，故单独 try/except 保护并记日志）。
    """
    global _CPU_POLITENESS_APPLIED
    if _CPU_POLITENESS_APPLIED:
        return
    _CPU_POLITENESS_APPLIED = True  # 先置位：失败不重试、不刷屏
    try:
        import torch
    except Exception:
        return
    threads, total = cpu_thread_budget()
    if threads >= total:
        return
    try:
        torch.set_num_threads(threads)
    except Exception as exc:
        _tsz_emit(f"[Embedding] CPU politeness failed (non-fatal): {exc}", flush=True)
        return
    interop_note = ""
    try:
        torch.set_num_interop_threads(threads)
        interop_note = f", interop={threads}"
    except Exception as exc:
        # 已被其他模块先行设置 / 并行工作已启动：跳过即可，不影响主线程设置
        interop_note = f", interop=skipped ({exc})"
    _tsz_emit(
        f"[Embedding] CPU politeness: torch threads {total} → {threads}{interop_note} "
        f"(reserved {total - threads} logical cores for foreground)",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 全局 encode 串行化 (2026-09-21)
# ═══════════════════════════════════════════════════════════════════════════
# bge-m3 CPU fp32 单次 forward 内部已按 torch 线程预算并行；recall / scorer /
# 写入回填等多路并发 encode 会互相超订 CPU、饿死前台（音频解码）。用进程级
# 信号量把实际模型推理串行排队（默认 1，env EMBEDDING_MAX_CONCURRENCY 可调）。
_ENCODE_WAIT_LOG_SEC = 2.0
_encode_semaphore: Optional[threading.BoundedSemaphore] = None
_encode_semaphore_init_lock = threading.Lock()


def encode_max_concurrency() -> int:
    """EMBEDDING_MAX_CONCURRENCY=n 覆盖全局 encode 并发上限（默认 1 = 严格串行）。"""
    raw = os.getenv("EMBEDDING_MAX_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 1


def _get_encode_semaphore() -> threading.BoundedSemaphore:
    global _encode_semaphore
    if _encode_semaphore is None:
        with _encode_semaphore_init_lock:
            if _encode_semaphore is None:
                _encode_semaphore = threading.BoundedSemaphore(encode_max_concurrency())
    return _encode_semaphore


def _encode_with_slot(fn, *args, **kwargs):
    """在全局信号量下执行一次模型 forward。

    只包 model.encode 调用本身（不含前置 get_model / 后置 numpy 转换），
    公开函数间的嵌套（encode_both legacy fallback → encode_text）不会自死锁。
    排队超 _ENCODE_WAIT_LOG_SEC 打一条日志后继续等（排队是设计行为、不超时失败）。
    """
    sem = _get_encode_semaphore()
    if not sem.acquire(timeout=_ENCODE_WAIT_LOG_SEC):
        _tsz_emit(f"[Embedding] encode slot busy > {_ENCODE_WAIT_LOG_SEC:.0f}s; queued", flush=True)
        sem.acquire()
    try:
        return fn(*args, **kwargs)
    finally:
        sem.release()


_flag_embedding_importable: Optional[bool] = None
_fe_import_error: str = ""


# Phase G.M1 (2026-05-02): 双 gate 防 silent fail —
# 之前只看 cache 目录、sentence_transformers 被 uninstall 时 cache 还在、误判 enabled、
# 全链路 silent bypass（参见 [Memory Embedding Refix 2026-04-25] 历史教训）。
# 新版同时检查 import 可用 + cache 在场。
_sentence_transformers_importable: bool | None = None  # 三态：None=未检测 / True / False
_st_import_error: str = ""
_cosine_warning_keys: set[tuple[int, int]] = set()
_cosine_warning_lock = threading.Lock()


def _check_sentence_transformers() -> bool:
    """One-shot import probe (cached). 启动 / force_retry() 时重置。"""
    global _sentence_transformers_importable, _st_import_error
    if _sentence_transformers_importable is not None:
        return _sentence_transformers_importable
    try:
        import importlib
        importlib.import_module("sentence_transformers")
        _sentence_transformers_importable = True
        _st_import_error = ""
    except Exception as _err:
        _sentence_transformers_importable = False
        _st_import_error = f"{type(_err).__name__}: {_err}"
        # 一次性告警：让管理员知道为什么 embedding 被禁用
        _tsz_emit(f"[Embedding] sentence_transformers unavailable: {_st_import_error}", flush=True)
    return _sentence_transformers_importable


def is_embedding_enabled() -> bool:
    """Check if embedding is enabled.

    Phase G.M1 (2026-05-02) 双 gate：必须同时满足
      (a) sentence_transformers 模块可 import
      (b) HuggingFace cache 目录存在 OR EMBEDDING_ENABLED=true 强制
    单独 cache 在场不够、防 silent fail（package 被 uninstall 时 cache 残留）
    """
    explicit = os.getenv(_EMBEDDING_ENABLED_KEY, "").strip().lower()
    if explicit in ("false", "0", "no"):
        return False
    # Gate (a): 模块必须可 import — 这是 silent fail 主防线
    if not _check_sentence_transformers():
        return False
    if explicit in ("true", "1", "yes"):
        return True
    # Gate (b): 自动开启需要 cache 已下载
    try:
        _cache = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5")
        return os.path.isdir(_cache)
    except Exception:
        return False


def get_model():
    """Lazy-load the SentenceTransformer model. Returns None on failure.
    Retries after cooldown instead of giving up permanently."""
    global _model_instance, _model_last_fail
    apply_cpu_politeness()
    if _model_instance is not None:
        return _model_instance
    # Still in cooldown from last failure — skip
    if _model_last_fail > 0 and (time.monotonic() - _model_last_fail) < _MODEL_RETRY_COOLDOWN:
        return None
    with _model_lock:
        if _model_instance is not None:
            return _model_instance
        if _model_last_fail > 0 and (time.monotonic() - _model_last_fail) < _MODEL_RETRY_COOLDOWN:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
            _model_instance = SentenceTransformer(model_name)
            _model_last_fail = 0.0  # clear failure state on success
            _tsz_emit(f"[Embedding] loaded {model_name} (dim={_EMBEDDING_DIM})")
            return _model_instance
        except Exception as e:
            _model_last_fail = time.monotonic()
            _tsz_emit(f"[Embedding] model load failed (retry in {_MODEL_RETRY_COOLDOWN:.0f}s): {e}")
            return None


def force_retry() -> bool:
    """Clear failure state and force model reload. Returns True if model loaded."""
    global _model_instance, _model_last_fail, _sentence_transformers_importable
    global _m3_model_instance, _m3_last_fail, _flag_embedding_importable
    with _model_lock:
        _model_last_fail = 0.0
        _model_instance = None
        # Phase G.M1: 同时清掉 import probe、允许 user 装回 sentence_transformers 后重试
        _sentence_transformers_importable = None
        _m3_model_instance = None
        _m3_last_fail = 0.0
        _flag_embedding_importable = None
    if is_m3_mode():
        return get_m3_model() is not None
    return get_model() is not None


# ═══════════════════════════════════════════════════════════════════════════
# Phase A (2026-05-14): BGE-M3 backend — dense + sparse 双输出
# ═══════════════════════════════════════════════════════════════════════════

def _check_flag_embedding() -> bool:
    """Import probe for FlagEmbedding package."""
    global _flag_embedding_importable, _fe_import_error
    if _flag_embedding_importable is not None:
        return _flag_embedding_importable
    try:
        import importlib
        importlib.import_module("FlagEmbedding")
        _flag_embedding_importable = True
        _fe_import_error = ""
    except Exception as _err:
        _flag_embedding_importable = False
        _fe_import_error = f"{type(_err).__name__}: {_err}"
        _tsz_emit(f"[Embedding-M3] FlagEmbedding unavailable: {_fe_import_error}", flush=True)
    return _flag_embedding_importable


def is_m3_mode() -> bool:
    """Check if Phase A M3 mode is enabled (EMBEDDING_MODEL_M3=true + FlagEmbedding 可 import)."""
    explicit = os.getenv(_EMBEDDING_M3_KEY, "").strip().lower()
    if explicit not in ("true", "1", "yes"):
        return False
    return _check_flag_embedding()


def get_m3_model():
    """Lazy-load BGEM3FlagModel. Returns None on failure (with cooldown)."""
    global _m3_model_instance, _m3_last_fail
    apply_cpu_politeness()
    if _m3_model_instance is not None:
        return _m3_model_instance
    if _m3_last_fail > 0 and (time.monotonic() - _m3_last_fail) < _MODEL_RETRY_COOLDOWN:
        return None
    with _model_lock:
        if _m3_model_instance is not None:
            return _m3_model_instance
        if _m3_last_fail > 0 and (time.monotonic() - _m3_last_fail) < _MODEL_RETRY_COOLDOWN:
            return None
        try:
            from FlagEmbedding import BGEM3FlagModel
            model_path = os.getenv(_EMBEDDING_M3_PATH_KEY, _DEFAULT_M3_MODEL).strip() or _DEFAULT_M3_MODEL
            # CPU 默认 fp32（fp16 在 CPU 会报错）；有 GPU 时可以 use_fp16=True 加速
            _m3_model_instance = BGEM3FlagModel(model_path, use_fp16=False)
            _m3_last_fail = 0.0
            _tsz_emit(f"[Embedding-M3] loaded {model_path} (dense={_EMBEDDING_DIM_M3}, sparse=lexical-weights)", flush=True)
            return _m3_model_instance
        except Exception as e:
            _m3_last_fail = time.monotonic()
            _tsz_emit(f"[Embedding-M3] model load failed (retry in {_MODEL_RETRY_COOLDOWN:.0f}s): {e}", flush=True)
            return None


def encode_dense(text: str) -> Optional[np.ndarray]:
    """Encode dense embedding. M3 mode → 1024 dim、legacy → 512 dim。统一接口。"""
    t = str(text or "").strip()
    if not t:
        return None
    if is_m3_mode():
        model = get_m3_model()
        if model is None:
            return None
        try:
            out = _encode_with_slot(model.encode, [t[:8000]], return_dense=True, return_sparse=False, return_colbert_vecs=False)
            dense = out['dense_vecs'][0]
            return np.asarray(dense, dtype=np.float32)
        except Exception as e:
            _tsz_emit(f"[Embedding-M3] encode_dense failed: {e}", flush=True)
            return None
    # Legacy bge-small path
    return encode_text(t)


def encode_sparse(text: str) -> Optional[dict]:
    """Encode sparse embedding (M3 only). Returns dict[int_token_id, float_weight]."""
    if not is_m3_mode():
        return None
    t = str(text or "").strip()
    if not t:
        return None
    model = get_m3_model()
    if model is None:
        return None
    try:
        out = _encode_with_slot(model.encode, [t[:8000]], return_dense=False, return_sparse=True, return_colbert_vecs=False)
        lex = out['lexical_weights'][0]  # dict[token_id, weight]
        return {int(k): float(v) for k, v in lex.items()}
    except Exception as e:
        _tsz_emit(f"[Embedding-M3] encode_sparse failed: {e}", flush=True)
        return None


def encode_both(text: str) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """One-shot dense + sparse encode. 比分别调用快 ~40%（一次 forward）。

    Legacy mode 下 sparse 总是 None。
    """
    t = str(text or "").strip()
    if not t:
        return None, None
    if is_m3_mode():
        model = get_m3_model()
        if model is None:
            return None, None
        try:
            out = _encode_with_slot(model.encode, [t[:8000]], return_dense=True, return_sparse=True, return_colbert_vecs=False)
            dense = np.asarray(out['dense_vecs'][0], dtype=np.float32)
            sparse = {int(k): float(v) for k, v in out['lexical_weights'][0].items()}
            return dense, sparse
        except Exception as e:
            _tsz_emit(f"[Embedding-M3] encode_both failed: {e}", flush=True)
            return None, None
    return encode_text(t), None


def encode_both_batch(texts: list[str]) -> list[tuple[Optional[np.ndarray], Optional[dict]]]:
    """Batch encode dense + sparse (M3 only)、给 backfill 用。

    Legacy mode 下退化到 encode_texts + None sparse。
    """
    if not texts:
        return []
    if is_m3_mode():
        model = get_m3_model()
        if model is None:
            return [(None, None)] * len(texts)
        try:
            cleaned = [str(t or "").strip()[:8000] for t in texts]
            non_empty = [(i, t) for i, t in enumerate(cleaned) if t]
            if not non_empty:
                return [(None, None)] * len(texts)
            batch_texts = [t for _, t in non_empty]
            out = _encode_with_slot(
                model.encode,
                batch_texts,
                batch_size=8,  # CPU 8 是中庸值
                return_dense=True, return_sparse=True, return_colbert_vecs=False,
            )
            result: list[tuple[Optional[np.ndarray], Optional[dict]]] = [(None, None)] * len(texts)
            for j, (idx, _) in enumerate(non_empty):
                dense = np.asarray(out['dense_vecs'][j], dtype=np.float32)
                sparse = {int(k): float(v) for k, v in out['lexical_weights'][j].items()}
                result[idx] = (dense, sparse)
            return result
        except Exception as e:
            _tsz_emit(f"[Embedding-M3] encode_both_batch failed: {e}", flush=True)
            return [(None, None)] * len(texts)
    # legacy fallback: encode dense only
    dense_list = encode_texts(texts)
    return [(d, None) for d in dense_list]


def sparse_to_text(sparse: Optional[dict]) -> str:
    """Serialize sparse dict to JSON string. None → empty string."""
    if not sparse:
        return ""
    try:
        return json.dumps(sparse, ensure_ascii=True, separators=(',', ':'))
    except Exception:
        return ""


def text_to_sparse(encoded: str) -> Optional[dict]:
    """Deserialize JSON string back to sparse dict. Empty/None/invalid → None."""
    if not encoded:
        return None
    try:
        d = json.loads(encoded)
        if not isinstance(d, dict):
            return None
        return {int(k): float(v) for k, v in d.items()}
    except Exception:
        return None


def sparse_dot_product(a: Optional[dict], b: Optional[dict]) -> float:
    """Compute sparse dot product (BM25-like score)。

    bge-m3 sparse 每个 token 的 weight 是已经学习好的 IDF-like 值、
    所以单纯点积 = 词项贡献分总和。
    """
    if not a or not b:
        return 0.0
    # iterate over the smaller dict for speed
    if len(a) > len(b):
        a, b = b, a
    return float(sum(weight * b.get(tok, 0.0) for tok, weight in a.items()))


# Backward-compat aliases / shortcuts
encode_text_dense = encode_dense  # alias, callers may use either


def encode_text(text: str) -> Optional[np.ndarray]:
    """Encode a single text into a 512-dim float32 numpy array. Returns None on failure."""
    if not is_embedding_enabled():
        return None
    model = get_model()
    if model is None:
        return None
    try:
        t = str(text or "").strip()[:500]  # Truncate to avoid excessive compute
        if not t:
            return None
        arr = _encode_with_slot(model.encode, t, normalize_embeddings=True)
        return arr.astype(np.float32)
    except Exception:
        return None


def encode_texts(texts: list[str]) -> list[Optional[np.ndarray]]:
    """Batch-encode multiple texts. Returns list of arrays (or None per-item)."""
    if not is_embedding_enabled():
        return [None] * len(texts)
    model = get_model()
    if model is None:
        return [None] * len(texts)
    try:
        cleaned = [str(t or "").strip()[:500] for t in texts]
        non_empty_idx = [i for i, t in enumerate(cleaned) if t]
        if not non_empty_idx:
            return [None] * len(texts)
        batch = [cleaned[i] for i in non_empty_idx]
        embeddings = _encode_with_slot(model.encode, batch, normalize_embeddings=True, batch_size=64)
        result: list[Optional[np.ndarray]] = [None] * len(texts)
        for j, idx in enumerate(non_empty_idx):
            result[idx] = embeddings[j].astype(np.float32)
        return result
    except Exception:
        return [None] * len(texts)


def embedding_to_text(arr: np.ndarray) -> str:
    """Serialize numpy array to base64-encoded string for TEXT column storage."""
    try:
        return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")
    except Exception:
        return ""


def declared_embedding_dimension() -> int:
    return _EMBEDDING_DIM_M3 if is_m3_mode() else _EMBEDDING_DIM


def declared_embedding_model_id() -> str:
    if is_m3_mode():
        model = os.getenv(_EMBEDDING_M3_PATH_KEY, _DEFAULT_M3_MODEL).strip() or _DEFAULT_M3_MODEL
        return f"bge-m3:{model}"
    model = os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    return f"sentence-transformers:{model}"


def declared_embedding_spec() -> tuple[str, int]:
    return declared_embedding_model_id(), declared_embedding_dimension()


def text_to_embedding(encoded: str, *, expected_dim: int | None = None) -> Optional[np.ndarray]:
    """Deserialize base64-encoded string back to numpy array."""
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded, validate=True)
        vector = np.frombuffer(raw, dtype=np.float32).copy()
        dimension = declared_embedding_dimension() if expected_dim is None else int(expected_dim)
        if dimension > 0 and int(vector.size) != dimension:
            warnings.warn(
                f"embedding dimension mismatch: stored={int(vector.size)} expected={dimension}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        return vector
    except Exception as exc:
        warnings.warn(f"embedding decode failed: {exc}", RuntimeWarning, stacklevel=2)
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Compute cosine similarity, using Python math if NumPy cannot do so."""
    try:
        dim_a = len(a)
        dim_b = len(b)
    except Exception as exc:
        warnings.warn(f"embedding shape unavailable: {exc}", RuntimeWarning, stacklevel=2)
        return None
    if dim_a != dim_b:
        key = (int(dim_a), int(dim_b))
        with _cosine_warning_lock:
            first_warning = key not in _cosine_warning_keys
            _cosine_warning_keys.add(key)
        if first_warning:
            warnings.warn(
                f"embedding dimension mismatch: left={dim_a} right={dim_b}; lexical fallback required",
                RuntimeWarning,
                stacklevel=2,
            )
        return None
    try:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception as exc:
        try:
            left = [float(value) for value in a]
            right = [float(value) for value in b]
            dot = sum(x * y for x, y in zip(left, right))
            norm_a = sum(x * x for x in left) ** 0.5
            norm_b = sum(y * y for y in right) ** 0.5
            if norm_a < 1e-8 or norm_b < 1e-8:
                return 0.0
            warnings.warn(f"NumPy cosine failed; used Python fallback: {exc}", RuntimeWarning, stacklevel=2)
            return float(dot / (norm_a * norm_b))
        except Exception as fallback_exc:
            warnings.warn(f"embedding cosine failed: {fallback_exc}", RuntimeWarning, stacklevel=2)
            return None
