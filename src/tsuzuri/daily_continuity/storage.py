"""Versioned sidecar storage and retry state for continuity capsules."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import DailyContinuityCapsule, GENERATOR_VERSION, SCHEMA_VERSION
from tsuzuri.reliable_io import atomic_write_json


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_archive_sha256(archive: Mapping[str, Any]) -> str:
    if not isinstance(archive, Mapping):
        raise TypeError("archive must be a mapping")
    return hashlib.sha256(canonical_json_bytes(archive)).hexdigest()


def capsule_path_for_archive(archive_path: str | os.PathLike[str]) -> Path:
    path = Path(archive_path)
    return path.with_name(f"{path.stem}.continuity.v2.json")


def retry_path_for_capsule(capsule_path: str | os.PathLike[str]) -> Path:
    path = Path(capsule_path)
    return path.with_name(f"{path.name}.retry.json")


def checkpoint_path_for_capsule(capsule_path: str | os.PathLike[str]) -> Path:
    path = Path(capsule_path)
    return path.with_name(f"{path.name}.checkpoint.json")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload, indent=None, sort_keys=True)


class ContinuityStorage:
    def load_matching_capsule(
        self, capsule_path: str | os.PathLike[str], archive_sha256: str
    ) -> DailyContinuityCapsule | None:
        payload = _read_json(Path(capsule_path))
        if not payload:
            return None
        try:
            capsule = DailyContinuityCapsule.from_dict(payload)
        except Exception:
            return None
        if capsule.source_archive_sha256 != archive_sha256:
            return None
        if capsule.schema_version != SCHEMA_VERSION or capsule.generator_version != GENERATOR_VERSION:
            return None
        return capsule

    def write_capsule(
        self, capsule_path: str | os.PathLike[str], capsule: DailyContinuityCapsule
    ) -> None:
        _atomic_write_json(Path(capsule_path), capsule.to_dict())

    def load_retry_state(self, retry_path: str | os.PathLike[str]) -> dict[str, Any]:
        return _read_json(Path(retry_path)) or {}

    def load_map_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        archive_sha256: str,
    ) -> dict[str, Any]:
        payload = _read_json(Path(checkpoint_path)) or {}
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("generator_version") != GENERATOR_VERSION
            or payload.get("archive_sha256") != archive_sha256
            or payload.get("status") != "in_progress"
            or not isinstance(payload.get("segments"), dict)
        ):
            return {}
        return payload

    def write_map_checkpoint(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        archive_sha256: str,
        segments: Mapping[str, Any],
        model_call_count: int,
        now_ms: int | None = None,
    ) -> None:
        _atomic_write_json(
            Path(checkpoint_path),
            {
                "schema_version": SCHEMA_VERSION,
                "generator_version": GENERATOR_VERSION,
                "status": "in_progress",
                "archive_sha256": archive_sha256,
                "segments": dict(segments),
                "model_call_count": max(0, int(model_call_count or 0)),
                "updated_at_ms": int(now_ms if now_ms is not None else time.time() * 1000),
            },
        )

    def clear_map_checkpoint(self, checkpoint_path: str | os.PathLike[str]) -> None:
        path = Path(checkpoint_path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def record_retryable_failure(
        self,
        retry_path: str | os.PathLike[str],
        *,
        archive_sha256: str,
        error: str,
        now_ms: int | None = None,
        base_delay_ms: int = 60_000,
        max_delay_ms: int = 6 * 60 * 60 * 1000,
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        previous = self.load_retry_state(retry_path)
        attempts = (
            int(previous.get("attempts", 0) or 0) + 1
            if previous.get("archive_sha256") == archive_sha256
            else 1
        )
        delay = min(max_delay_ms, base_delay_ms * (2 ** min(attempts - 1, 10)))
        quarantined = attempts >= max(1, int(max_attempts or 1))
        state = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "status": "quarantined" if quarantined else "retryable_failure",
            "archive_sha256": archive_sha256,
            "attempts": attempts,
            "last_error": str(error or "unknown_error")[:1000],
            "last_attempt_at_ms": now,
            "next_retry_at_ms": 0 if quarantined else now + delay,
        }
        _atomic_write_json(Path(retry_path), state)
        return state

    def record_success(
        self,
        retry_path: str | os.PathLike[str],
        *,
        archive_sha256: str,
        now_ms: int | None = None,
    ) -> dict[str, Any]:
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        previous = self.load_retry_state(retry_path)
        state = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "status": "succeeded",
            "archive_sha256": archive_sha256,
            "attempts": int(previous.get("attempts", 0) or 0),
            "last_error": "",
            "last_attempt_at_ms": now,
            "next_retry_at_ms": 0,
        }
        _atomic_write_json(Path(retry_path), state)
        return state

    def retry_is_due(
        self,
        retry_path: str | os.PathLike[str],
        *,
        now_ms: int | None = None,
        archive_sha256: str = "",
    ) -> bool:
        state = self.load_retry_state(retry_path)
        if archive_sha256 and state.get("archive_sha256") != archive_sha256:
            return True
        if state.get("status") != "retryable_failure":
            return state.get("status") != "quarantined"
        now = int(now_ms if now_ms is not None else time.time() * 1000)
        return now >= int(state.get("next_retry_at_ms", 0) or 0)


__all__ = [
    "ContinuityStorage",
    "canonical_json_bytes",
    "capsule_path_for_archive",
    "checkpoint_path_for_capsule",
    "compute_archive_sha256",
    "retry_path_for_capsule",
]
