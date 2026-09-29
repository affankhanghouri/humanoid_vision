"""Stateful, conservative path-relation reasoning over tracks and road masks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np

from rejected_path_relation.geometry import (
    CorridorGeometry, box_patch_radii, contact_point, construct_forward_corridor,
    corridor_distance_and_side, image_to_mask, integral_mask, integral_patch_fraction,
)
from rejected_path_relation.types import PathRelationObservation, PathRelationState


SUPPORTED_CLASSES = frozenset({"person", "bicycle", "motorcycle", "car", "truck", "bus"})


@dataclass
class _Sample:
    frame_id: int
    timestamp: float
    point: tuple[float, float]
    road_member: bool | None
    corridor_member: bool | None
    corridor_side: int | None


@dataclass
class _TrackHistory:
    samples: deque = field(default_factory=lambda: deque(maxlen=8))
    state: PathRelationState = PathRelationState.UNKNOWN
    pending: PathRelationState | None = None
    pending_count: int = 0
    last_seen: float = 0.0


class PathRelationEngine:
    """No inference: native-mask sampling, short track projection and hysteresis."""

    def __init__(self, *, road_max_age: float = .55, tracking_max_age: float = .75,
                 track_expiry: float = 2.0, prediction_horizon: float = .4,
                 on_road_threshold: float = .60, off_road_threshold: float = .25,
                 in_corridor_threshold: float = .45,
                 out_corridor_threshold: float = .15,
                 transition_persistence: int = 2):
        self.road_max_age = road_max_age
        self.tracking_max_age = tracking_max_age
        self.track_expiry = track_expiry
        self.prediction_horizon = prediction_horizon
        self.on_road_threshold = on_road_threshold
        self.off_road_threshold = off_road_threshold
        self.in_corridor_threshold = in_corridor_threshold
        self.out_corridor_threshold = out_corridor_threshold
        self.transition_persistence = transition_persistence
        self._tracks: dict[int, _TrackHistory] = {}
        self._corridor_key = None
        self._corridor: CorridorGeometry | None = None
        self._road_integral = None
        self._corridor_integral = None

    def _corridor_for(self, road) -> CorridorGeometry | None:
        key = None if road is None else (road.source_frame_id, road.source_timestamp,
                                         id(road.drivable_mask))
        if key != self._corridor_key:
            self._corridor_key = key
            self._corridor = (None if road is None else construct_forward_corridor(
                road.drivable_mask, road.content_rect))
            self._road_integral = None if road is None else integral_mask(road.drivable_mask)
            self._corridor_integral = (None if self._corridor is None
                                       else integral_mask(self._corridor.mask))
        return self._corridor

    @staticmethod
    def _membership(value: float | None, previous: bool | None,
                    on_threshold: float, off_threshold: float) -> bool | None:
        if value is None:
            return None
        if value >= on_threshold:
            return True
        if value <= off_threshold:
            return False
        return previous

    def _motion_reliable(self, entity, history: _TrackHistory,
                         timestamp: float, frame_size: tuple[int, int]) -> bool:
        if entity.misses or entity.last_observed_timestamp is None:
            return False
        age = timestamp - entity.last_observed_timestamp
        if not 0 <= age <= min(self.tracking_max_age, .45):
            return False
        if len(history.samples) < 3:
            return False
        recent = list(history.samples)[-3:]
        span = recent[-1].timestamp - recent[0].timestamp
        if span < .18 or span > 1.0:
            return False
        speed = math.hypot(entity.velocity_x, entity.velocity_y)
        return (math.isfinite(speed)
                and speed <= math.hypot(*frame_size) * 2.0)

    def _candidate(self, road_member, corridor_member, current_relation, predicted_relation,
                   history, entity, motion_reliable, frame_width):
        if road_member is False:
            return PathRelationState.OFF_DRIVABLE
        if road_member is not True or corridor_member is None:
            return PathRelationState.UNKNOWN
        if not motion_reliable:
            return (PathRelationState.IN_CORRIDOR if corridor_member
                    else PathRelationState.ON_DRIVABLE_OUTSIDE_CORRIDOR)

        current_distance, side = current_relation or (None, None)
        predicted_distance, predicted_side = predicted_relation or (None, None)
        vx = entity.velocity_x
        lateral = abs(vx) >= frame_width * .025
        recent = list(history.samples)
        recent_sides = [sample.corridor_side for sample in recent[-4:]
                        if sample.corridor_side is not None]
        was_inside = any(sample.corridor_member is True for sample in recent[-4:-1])
        prior_outside_sides = [value for value in recent_sides[:-1] if value]
        crossed_side = bool(prior_outside_sides and predicted_side
                            and prior_outside_sides[-1] != predicted_side)
        moving_through = (lateral and corridor_member and prior_outside_sides
                          and predicted_side in (0, -prior_outside_sides[-1]))
        if crossed_side or moving_through:
            return PathRelationState.CROSSING_CORRIDOR

        if corridor_member:
            if (was_inside and predicted_distance is not None
                    and predicted_distance >= frame_width * .025):
                return PathRelationState.LEAVING_CORRIDOR
            return PathRelationState.IN_CORRIDOR

        predicted_near = predicted_distance is not None and predicted_distance <= frame_width * .035
        toward = (side in (-1, 1) and predicted_side in (0, side) and predicted_near
                  and current_distance is not None
                  and predicted_distance <= current_distance - frame_width * .012)
        if toward:
            return PathRelationState.ENTERING_CORRIDOR
        if was_inside and predicted_side == side and predicted_distance is not None:
            return PathRelationState.LEAVING_CORRIDOR
        return PathRelationState.ON_DRIVABLE_OUTSIDE_CORRIDOR

    def _stabilize(self, history: _TrackHistory, candidate: PathRelationState,
                   new_evidence: bool) -> PathRelationState:
        important = {PathRelationState.ENTERING_CORRIDOR,
                     PathRelationState.CROSSING_CORRIDOR,
                     PathRelationState.LEAVING_CORRIDOR}
        if not new_evidence:
            return history.state
        required = self.transition_persistence if candidate in important else 1
        if candidate == history.state:
            history.pending = None
            history.pending_count = 0
            return history.state
        if history.pending == candidate:
            history.pending_count += 1
        else:
            history.pending = candidate
            history.pending_count = 1
        if history.pending_count >= required:
            history.state = candidate
            history.pending = None
            history.pending_count = 0
        return history.state

    @staticmethod
    def _confidence(road_fraction, corridor_overlap, corridor_quality,
                    road_age, road_max_age, track_age, tracking_max_age,
                    persistence, motion_reliable, state):
        if road_fraction is None or corridor_overlap is None or state == PathRelationState.UNKNOWN:
            return 0.0
        road_decisiveness = min(1., abs(road_fraction - .425) / .425)
        corridor_decisiveness = min(1., abs(corridor_overlap - .30) / .30)
        freshness = min(max(0., 1 - road_age / road_max_age),
                        max(0., 1 - track_age / tracking_max_age))
        temporal = min(1., persistence / 4)
        motion = 1. if motion_reliable else .35
        return float(np.clip(.30 * road_decisiveness + .25 * corridor_decisiveness
                             + .20 * corridor_quality + .15 * freshness
                             + .06 * temporal + .04 * motion, 0., 1.))

    def update(self, state, road, *, frame_id: int, timestamp: float,
               frame_size: tuple[int, int]) -> tuple[PathRelationObservation, ...]:
        """Derive states at display time while preserving both source timestamps."""
        now = time.perf_counter()
        width, height = frame_size
        active = {entity.entity_id for entity in state.entities} if state is not None else set()
        for track_id in list(self._tracks):
            if track_id not in active and timestamp - self._tracks[track_id].last_seen > self.track_expiry:
                del self._tracks[track_id]
        if state is None:
            return ()

        tracking_age = timestamp - state.source_timestamp
        tracking_reliable = (state.source_frame_id <= frame_id
                             and 0 <= tracking_age <= self.tracking_max_age)
        road_age = math.inf if road is None else timestamp - road.source_timestamp
        road_fresh = (road is not None and road.source_frame_id <= frame_id
                      and 0 <= road_age <= self.road_max_age)
        corridor = self._corridor_for(road) if road_fresh else None
        road_reliable = road_fresh and corridor is not None
        output = []
        for entity in state.entities:
            history = self._tracks.setdefault(entity.entity_id, _TrackHistory())
            history.last_seen = timestamp
            supported = entity.class_name.lower() in SUPPORTED_CLASSES
            age = max(0., tracking_age) if tracking_reliable else 0.
            box = (entity.center_x + entity.velocity_x * age - entity.width / 2,
                   entity.center_y + entity.velocity_y * age - entity.height / 2,
                   entity.center_x + entity.velocity_x * age + entity.width / 2,
                   entity.center_y + entity.velocity_y * age + entity.height / 2)
            contact = contact_point(box, frame_size)
            new_evidence = not history.samples or history.samples[-1].frame_id != state.source_frame_id
            if not (supported and tracking_reliable and road_reliable and contact is not None):
                if new_evidence:
                    history.state = PathRelationState.UNKNOWN
                    history.pending = None
                    history.pending_count = 0
                output.append(PathRelationObservation(
                    entity.entity_id, state.source_frame_id, state.source_timestamp,
                    None if road is None else road.source_frame_id,
                    None if road is None else road.source_timestamp, now, contact, None,
                    None, None, PathRelationState.UNKNOWN, 0., False,
                    bool(road_reliable), 0. if corridor is None else corridor.quality,
                    len(history.samples)))
                continue

            native = image_to_mask(contact, frame_size, road.content_rect,
                                   road.drivable_mask.shape)
            radii = box_patch_radii(box, frame_size, road.content_rect)
            road_fraction = integral_patch_fraction(
                self._road_integral, native, radii, road.drivable_mask.shape)
            corridor_overlap = integral_patch_fraction(
                self._corridor_integral, native, radii, corridor.mask.shape)
            previous = history.samples[-1] if history.samples else None
            road_member = self._membership(road_fraction,
                                           None if previous is None else previous.road_member,
                                           self.on_road_threshold, self.off_road_threshold)
            corridor_member = self._membership(corridor_overlap,
                                               None if previous is None else previous.corridor_member,
                                               self.in_corridor_threshold,
                                               self.out_corridor_threshold)
            relation = corridor_distance_and_side(native, corridor, width,
                                                   road.content_rect[2])
            side = None if relation is None else relation[1]
            if new_evidence:
                history.samples.append(_Sample(state.source_frame_id, state.source_timestamp,
                                               contact, road_member, corridor_member, side))
            motion_reliable = self._motion_reliable(entity, history, timestamp, frame_size)
            predicted = None
            predicted_relation = None
            if motion_reliable:
                predicted = (float(np.clip(contact[0] + entity.velocity_x * self.prediction_horizon,
                                           0, width - 1)),
                             float(np.clip(contact[1] + entity.velocity_y * self.prediction_horizon,
                                           0, height - 1)))
                predicted_native = image_to_mask(predicted, frame_size, road.content_rect,
                                                 road.drivable_mask.shape)
                predicted_relation = corridor_distance_and_side(
                    predicted_native, corridor, width, road.content_rect[2])
            candidate = self._candidate(road_member, corridor_member, relation,
                                        predicted_relation, history, entity,
                                        motion_reliable, width)
            relation_state = self._stabilize(history, candidate, new_evidence)
            track_age = timestamp - (entity.last_observed_timestamp
                                     if entity.last_observed_timestamp is not None
                                     else state.source_timestamp)
            confidence = self._confidence(
                road_fraction, corridor_overlap, corridor.quality, road_age,
                self.road_max_age, max(0., track_age), self.tracking_max_age,
                len(history.samples), motion_reliable, relation_state)
            output.append(PathRelationObservation(
                entity.entity_id, state.source_frame_id, state.source_timestamp,
                road.source_frame_id, road.source_timestamp, now, contact, predicted,
                road_fraction, corridor_overlap, relation_state, confidence,
                motion_reliable, True, corridor.quality, len(history.samples)))
        return tuple(output)

    @property
    def corridor(self) -> CorridorGeometry | None:
        return self._corridor

    @property
    def tracked_ids(self) -> frozenset[int]:
        return frozenset(self._tracks)
