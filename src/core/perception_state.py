"""Immutable tracking and pose snapshots with independent source timestamps."""
from dataclasses import dataclass


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
