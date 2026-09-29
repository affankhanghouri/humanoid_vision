"""Immutable tracking, pose and road snapshots with independent source timestamps."""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ObservationMeta:
    source_frame_id: int
    source_timestamp: float
    produced_timestamp: float

    @property
    def processing_latency_ms(self) -> float:
        return max(0.0, (self.produced_timestamp - self.source_timestamp) * 1000.0)

    def age_at(self, timestamp: float) -> float:
        return timestamp - self.source_timestamp

    def is_valid_for(self, frame_id: int, timestamp: float, max_age: float) -> bool:
        return self.source_frame_id <= frame_id and 0 <= self.age_at(timestamp) <= max_age


@dataclass(frozen=True)
class RoadSegObservation(ObservationMeta):
    inference_ms: float
    drivable_mask: np.ndarray = field(repr=False, compare=False)
    # Content rectangle within the padded native mask: left, top, width, height.
    content_rect: tuple[int, int, int, int]
    source_size: tuple[int, int]
    preprocessing_ms: float = 0.0
    postprocessing_ms: float = 0.0

    def __post_init__(self):
        mask = np.asarray(self.drivable_mask)
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError('Road mask must be a 2D uint8 array')
        # Bytes-backed storage cannot be made writable by a renderer/consumer.
        immutable = np.frombuffer(mask.tobytes(), dtype=np.uint8).reshape(mask.shape)
        object.__setattr__(self, 'drivable_mask', immutable)


@dataclass(frozen=True)
class PoseKeypoint:
    # Coordinates relative to the pose model's person box, not image pixels.
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class PoseObservation:
    meta: ObservationMeta
    keypoints: tuple[PoseKeypoint, ...]
    confidence: float


@dataclass(frozen=True)
class PerceptionEntity:
    entity_id: int
    class_name: str
    confidence: float
    center_x: float
    center_y: float
    velocity_x: float
    velocity_y: float
    width: float
    height: float
    pose: PoseObservation | None = None
    last_observed_timestamp: float | None = None
    misses: int = 0

    def visible_at(self, timestamp: float, max_coast_age: float) -> bool:
        return (self.misses == 0 or (self.last_observed_timestamp is not None
                and 0 <= timestamp - self.last_observed_timestamp <= max_coast_age))


@dataclass(frozen=True)
class PerceptionState:
    # These source fields describe tracking only; pose never refreshes them.
    source_frame_id: int
    source_timestamp: float
    produced_timestamp: float
    entities: tuple[PerceptionEntity, ...]
    inference_ms: float
    detector_hz: float
    average_ms: float
    p95_ms: float
    pose_meta: ObservationMeta | None = None
    pose_inference_ms: float = 0.0
    pose_hz: float = 0.0
    pose_average_ms: float = 0.0
    pose_p95_ms: float = 0.0
    road: RoadSegObservation | None = None

    @property
    def tracking_meta(self) -> ObservationMeta:
        return ObservationMeta(self.source_frame_id, self.source_timestamp, self.produced_timestamp)

    @property
    def processing_latency_ms(self) -> float:
        return self.tracking_meta.processing_latency_ms

    def age_at(self, timestamp: float) -> float:
        return self.tracking_meta.age_at(timestamp)

    def is_valid_for(self, frame_id: int, timestamp: float, max_age: float) -> bool:
        return self.tracking_meta.is_valid_for(frame_id, timestamp, max_age)
