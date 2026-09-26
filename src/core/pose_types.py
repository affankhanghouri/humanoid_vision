"""Immutable image-coordinate output from a pose estimator."""
from dataclasses import dataclass
from core.types import BBox


@dataclass(frozen=True)
class PosePoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PosePerson:
    bbox: BBox
    confidence: float
    keypoints: tuple[PosePoint, ...]
