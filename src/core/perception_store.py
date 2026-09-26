"""Atomically merge independent tracking and pose updates by entity ID."""
from dataclasses import replace
from threading import Lock
from core.perception_state import ObservationMeta, PerceptionState, PoseObservation


class PerceptionStore:
    def __init__(self):
        self._lock = Lock()
        self._state: PerceptionState | None = None

    def publish(self, state: PerceptionState) -> None:
        """Compatibility entry point for tracking publishers."""
        self.publish_tracking(state)

    def publish_tracking(self, state: PerceptionState) -> None:
        with self._lock:
            old = self._state
            if old is not None:
                if (state.source_frame_id, state.produced_timestamp) < (old.source_frame_id, old.produced_timestamp):
                    return
                poses = {entity.entity_id: entity.pose for entity in old.entities
                         if entity.class_name == 'person'}
                state = replace(
                    state,
                    entities=tuple(replace(entity, pose=poses.get(entity.entity_id))
                                   if entity.class_name == 'person' else entity
                                   for entity in state.entities),
                    pose_meta=old.pose_meta, pose_inference_ms=old.pose_inference_ms,
                    pose_hz=old.pose_hz, pose_average_ms=old.pose_average_ms,
                    pose_p95_ms=old.pose_p95_ms,
                )
            self._state = state

    def publish_pose(self, observations: dict[int, PoseObservation], meta: ObservationMeta,
                     inference_ms: float, pose_hz: float, average_ms: float, p95_ms: float) -> None:
        with self._lock:
            state = self._state
            if state is None:
                return
            old_meta = state.pose_meta
            if old_meta is not None and (meta.source_frame_id, meta.produced_timestamp) <= (old_meta.source_frame_id, old_meta.produced_timestamp):
                return
            # A newer pose pass clears skeletons for people it did not match.
            self._state = replace(
                state,
                entities=tuple(replace(entity, pose=observations.get(entity.entity_id)
                                       if entity.class_name == 'person' else None)
                               for entity in state.entities),
                pose_meta=meta, pose_inference_ms=inference_ms, pose_hz=pose_hz,
                pose_average_ms=average_ms, pose_p95_ms=p95_ms,
            )

    def get_latest(self) -> PerceptionState | None:
        with self._lock:
            return self._state
