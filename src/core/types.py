"""Backend-independent packets. Published frames are read-only by convention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracking.motion import MotionSnapshot
import numpy as np

BBox = tuple[float, float, float, float]

@dataclass(frozen=True)
class Detection:
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float

@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    motion: "MotionSnapshot | None" = None
