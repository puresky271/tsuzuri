"""Offline core for versioned cross-day conversation continuity capsules."""

from .compactor import CompactorConfig, DailyContinuityCompactor
from .contracts import (
    CapsuleItem,
    CompactionResult,
    DailyContinuityCapsule,
    SchemaValidationError,
    SegmentProvenance,
)
from .storage import ContinuityStorage, capsule_path_for_archive, retry_path_for_capsule

__all__ = [
    "CapsuleItem",
    "CompactionResult",
    "CompactorConfig",
    "DailyContinuityCapsule",
    "DailyContinuityCompactor",
    "ContinuityStorage",
    "SchemaValidationError",
    "SegmentProvenance",
    "capsule_path_for_archive",
    "retry_path_for_capsule",
]
