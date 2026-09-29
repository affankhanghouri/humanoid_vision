"""Immutable outputs for road-relative tracked-object state."""

from dataclasses import dataclass
from enum import Enum


class PathRelationState(str, Enum):
    UNKNOWN = "UNKNOWN"
    OFF_DRIVABLE = "OFF_DRIVABLE"
    ON_DRIVABLE_OUTSIDE_CORRIDOR = "ON_DRIVABLE_OUTSIDE_CORRIDOR"
    ENTERING_CORRIDOR = "ENTERING_CORRIDOR"
    IN_CORRIDOR = "IN_CORRIDOR"
    CROSSING_CORRIDOR = "CROSSING_CORRIDOR"
    LEAVING_CORRIDOR = "LEAVING_CORRIDOR"


class RoadSupportKind(str, Enum):
    ROAD_VISIBLE_AND_SUPPORTED = "ROAD_VISIBLE_AND_SUPPORTED"
    ROAD_INFERRED_FROM_SURROUNDINGS = "ROAD_INFERRED_FROM_SURROUNDINGS"
    ROAD_NOT_SUPPORTED = "ROAD_NOT_SUPPORTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PathRelationObservationV2:
    track_id: int
    tracking_source_frame_id: int
    tracking_source_timestamp: float
    road_source_frame_id: int | None
    road_source_timestamp: float | None
    road_transform_timestamp: float | None
    produced_timestamp: float
    contact_point: tuple[float, float] | None
    predicted_contact_point: tuple[float, float] | None
    probe_fractions: tuple[tuple[str, float | None], ...]
    road_support_score: float | None
    road_support_kind: RoadSupportKind
    corridor_overlap: float | None
    state: PathRelationState
    confidence: float
    motion_reliable: bool
    road_reliable: bool
    road_spatially_projected: bool
    corridor_quality: float
    evidence_count: int


@dataclass(frozen=True)
class PathRelationObservation:
    track_id: int
    tracking_source_frame_id: int
    tracking_source_timestamp: float
    road_source_frame_id: int | None
    road_source_timestamp: float | None
    produced_timestamp: float
    contact_point: tuple[float, float] | None
    predicted_contact_point: tuple[float, float] | None
    drivable_fraction: float | None
    corridor_overlap: float | None
    state: PathRelationState
    confidence: float
    motion_reliable: bool
    road_reliable: bool
    corridor_quality: float
    evidence_count: int

