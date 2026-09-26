"""Associate pose people one-to-one with fresh tracked person IDs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import VisionConfig
from core.perception_state import ObservationMeta, PerceptionState, PoseKeypoint, PoseObservation
from core.pose_types import PosePerson
from tracking.tracker import calculate_iou


def has_fresh_person(state: PerceptionState | None, frame_id: int,
                     timestamp: float, max_age: float) -> bool:
    return (state is not None and state.is_valid_for(frame_id, timestamp, max_age)
            and any(entity.class_name == 'person' and entity.misses == 0 for entity in state.entities))


def entity_box_at(entity, state: PerceptionState, timestamp: float, max_prediction_age: float):
    age = min(state.age_at(timestamp), max_prediction_age)
    cx = entity.center_x + entity.velocity_x * age
    cy = entity.center_y + entity.velocity_y * age
    return (cx - entity.width/2, cy - entity.height/2,
            cx + entity.width/2, cy + entity.height/2)


def normalized_center_distance(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    center_ax = (ax1 + ax2) / 2.0
    center_ay = (ay1 + ay2) / 2.0
    center_bx = (bx1 + bx2) / 2.0
    center_by = (by1 + by2) / 2.0

    distance = ((center_ax - center_bx) ** 2 + (center_ay - center_by) ** 2) ** 0.5

    width = max(1.0, bx2 - bx1)
    height = max(1.0, by2 - by1)
    diagonal = (width ** 2 + height ** 2) ** 0.5

    return distance / diagonal


def match_pose_to_entities(people: tuple[PosePerson, ...], state: PerceptionState,
                           meta: ObservationMeta, config: VisionConfig) -> dict[int, PoseObservation]:
    if not has_fresh_person(state, meta.source_frame_id, meta.source_timestamp, config.max_render_age):
        return {}

    entities = [entity for entity in state.entities
                if entity.class_name == 'person' and entity.misses == 0]
    people = [person for person in people if np.isfinite(person.bbox).all()
              and person.bbox[2] > person.bbox[0] and person.bbox[3] > person.bbox[1]]
    if not people or not entities:
        return {}

    rows = len(people)
    columns = len(entities)
    invalid_cost = 10000.0
    cost_matrix = np.full((rows, columns), invalid_cost, dtype=np.float32)

    # ------------------------------------------
    # Build matching cost
    # ------------------------------------------
    for pose_index, person in enumerate(people):
        for entity_index, entity in enumerate(entities):
            track_box = entity_box_at(entity, state, meta.source_timestamp, config.max_prediction_age)

            overlap = calculate_iou(person.bbox, track_box)
            distance = normalized_center_distance(person.bbox, track_box)

            # ----------------------------------
            # Matching gate
            #
            # Accept when either:
            # boxes overlap
            # OR centers are close.
            # ----------------------------------
            valid = overlap >= config.pose_match_min_iou or distance <= config.pose_match_max_distance
            if not valid:
                continue

            distance_cost = min(distance, 1.0)
            cost = (config.pose_match_iou_weight * (1.0 - overlap)
                    + config.pose_match_distance_weight * distance_cost)
            cost_matrix[pose_index, entity_index] = cost

    # ------------------------------------------
    # Hungarian assignment
    # ------------------------------------------
    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    matched: dict[int, PoseObservation] = {}
    for pose_index, entity_index in zip(row_indices, column_indices):
        cost = cost_matrix[pose_index, entity_index]
        if cost >= invalid_cost:
            continue
        if cost > config.pose_match_max_cost:
            continue

        entity = entities[entity_index]
        person = people[pose_index]
        x1, y1, x2, y2 = person.bbox
        matched[entity.entity_id] = PoseObservation(
            meta=meta, confidence=person.confidence,
            keypoints=tuple(PoseKeypoint((point.x-x1)/(x2-x1), (point.y-y1)/(y2-y1), point.confidence)
                            for point in person.keypoints),
        )
    return matched