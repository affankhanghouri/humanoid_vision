"""Backend-independent packets. Published frames are read-only by convention."""
from dataclasses import dataclass
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

@dataclass(frozen=True)
class RenderTrack:
    track_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    width: float
    height: float

@dataclass(frozen=True)
class TrackingResult:
    frame_id: int
    timestamp: float
    tracks: tuple[RenderTrack, ...]
    inference_ms: float
    detector_hz: float
    average_ms: float
    p95_ms: float
