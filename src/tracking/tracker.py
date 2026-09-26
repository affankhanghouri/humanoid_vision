"""Confidence-aware association with bounded, explicit track lifetimes."""
import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import VisionConfig
from core.types import Detection
from core.perception_state import PerceptionEntity

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    width = max(0, ix2 - ix1)
    height = max(0, iy2 - iy1)
    intersection = width * height
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union

def normalized_center_distance(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2
    distance = np.hypot(acx - bcx, acy - bcy)
    width_a = max(1, ax2 - ax1)
    height_a = max(1, ay2 - ay1)
    width_b = max(1, bx2 - bx1)
    height_b = max(1, by2 - by1)
    scale_a = np.hypot(width_a, height_a)
    scale_b = np.hypot(width_b, height_b)
    scale = max(scale_a, scale_b, 1.0)
    return distance / scale

class Track:
    next_id = 1

    def __init__(self, bbox, class_name, confidence, timestamp, config: VisionConfig):
        self.config = config
        self.id = Track.next_id
        Track.next_id += 1
        self.class_name = class_name
        self.confidence = confidence
        self.hits = 1
        self.misses = 0
        self.last_state_time = timestamp
        self.last_observed_timestamp = timestamp
        x1, y1, x2, y2 = bbox
        self.width = x2 - x1
        self.height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.config.measurement_noise
        self.kalman.processNoiseCov = np.array(np.diag(self.config.process_noise), dtype=np.float32)
        self.kalman.errorCovPost = np.diag(self.config.initial_covariance).astype(np.float32)
        state = np.array([[center_x], [center_y], [0.0], [0.0]], dtype=np.float32)
        self.kalman.statePost = state.copy()
        self.kalman.statePre = state.copy()
        self.bbox = bbox

    def predict_to(self, timestamp):
        dt = timestamp - self.last_state_time
        if dt <= 0:
            return self.bbox
        self.kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        prediction = self.kalman.predict()
        self.last_state_time = timestamp
        center_x = float(prediction[0, 0])
        center_y = float(prediction[1, 0])
        self.bbox = self._make_bbox(center_x, center_y)
        return self.bbox

    def update(self, bbox, confidence):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        corrected = self.kalman.correct(measurement)
        self.width = (1 - self.config.track_size_alpha) * self.width + self.config.track_size_alpha * width
        self.height = (1 - self.config.track_size_alpha) * self.height + self.config.track_size_alpha * height
        self.confidence = confidence
        self.last_observed_timestamp = self.last_state_time
        self.hits += 1
        self.misses = 0
        corrected_x = float(corrected[0, 0])
        corrected_y = float(corrected[1, 0])
        self.bbox = self._make_bbox(corrected_x, corrected_y)

    def _make_bbox(self, center_x, center_y):
        return (center_x - self.width / 2, center_y - self.height / 2, center_x + self.width / 2, center_y + self.height / 2)

    def snapshot(self) -> PerceptionEntity:
        state = self.kalman.statePost
        return PerceptionEntity(entity_id=self.id, class_name=self.class_name, confidence=self.confidence, center_x=float(state[0, 0]), center_y=float(state[1, 0]), velocity_x=float(state[2, 0]), velocity_y=float(state[3, 0]), width=self.width, height=self.height, last_observed_timestamp=self.last_observed_timestamp, misses=self.misses)

VEHICLE_CLASSES = frozenset(("car", "truck", "bus"))


class MultiObjectTracker:
    def __init__(self, config: VisionConfig = VisionConfig()):
        self.config = config
        self.tracks = []
        self.last_timestamp = None

    def update(self, detections: list[Detection], timestamp: float) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("Tracking timestamp must be finite")
        # Repeated/out-of-order observations must not confirm or age a track.
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return
        self.last_timestamp = timestamp
        self.tracks = [t for t in self.tracks
                       if timestamp - t.last_observed_timestamp <= self.config.max_track_age]
        candidates = sorted((d for d in detections if self._valid_detection(d)),
                            key=lambda d: d.confidence, reverse=True)
        detections = []
        for candidate in candidates:
            # End-to-end outputs can contain two subtype labels for one vehicle.
            duplicate = any(candidate.class_name != other.class_name
                            and candidate.class_name in VEHICLE_CLASSES
                            and other.class_name in VEHICLE_CLASSES
                            and calculate_iou(candidate.bbox, other.bbox) >= .85
                            for other in detections)
            if not duplicate:
                detections.append(candidate)
        for track in self.tracks:
            track.predict_to(timestamp)

        high = [d for d in detections if d.confidence >= self.config.confidence]
        low = [d for d in detections if self.config.track_low_confidence <= d.confidence
               < self.config.confidence]
        matched, used = self._associate(list(range(len(self.tracks))), high)
        # Recovery is deliberately stricter and only available to confirmed IDs.
        remaining = [i for i, t in enumerate(self.tracks)
                     if i not in matched and t.hits >= self.config.min_hits]
        recovered, _ = self._associate(remaining, low, recovery=True)
        matched.update(recovered)
        for index, track in enumerate(self.tracks):
            if index not in matched:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses
                       and (t.hits >= self.config.min_hits or t.misses == 0)]
        for index, detection in enumerate(high):
            if index not in used:
                self.tracks.append(Track(detection.bbox, detection.class_name,
                                         detection.confidence, timestamp, self.config))

    @staticmethod
    def _valid_detection(detection):
        box = detection.bbox
        return (len(box) == 4 and all(math.isfinite(v) for v in box)
                and box[2] > box[0] and box[3] > box[1]
                and math.isfinite(detection.confidence) and 0 <= detection.confidence <= 1)

    def _associate(self, indices, detections, recovery=False):
        if not indices or not detections:
            return set(), set()
        invalid = self.config.invalid_match_cost
        costs = np.full((len(indices), len(detections)), invalid, dtype=np.float32)
        for row, index in enumerate(indices):
            track = self.tracks[index]
            for col, detection in enumerate(detections):
                same_vehicle = (track.class_name in VEHICLE_CLASSES
                                and detection.class_name in VEHICLE_CLASSES)
                if track.class_name != detection.class_name and not same_vehicle:
                    continue
                width = detection.bbox[2] - detection.bbox[0]
                height = detection.bbox[3] - detection.bbox[1]
                ratio = max(width / track.width, track.width / width,
                            height / track.height, track.height / height)
                if ratio > self.config.match_max_size_ratio:
                    continue
                iou = calculate_iou(track.bbox, detection.bbox)
                distance = normalized_center_distance(track.bbox, detection.bbox)
                if (recovery or track.class_name != detection.class_name) and iou < self.config.recovery_min_iou:
                    continue
                if iou < self.config.match_min_iou and distance > self.config.match_max_distance:
                    continue
                cost = (self.config.match_iou_weight * (1.0 - iou)
                        + self.config.match_distance_weight
                        * min(distance / self.config.match_max_distance, 1.0))
                # Gate BEFORE assignment so an invalid pair cannot steal a valid one.
                if cost <= self.config.match_max_cost:
                    costs[row, col] = cost
        rows, cols = linear_sum_assignment(costs)
        matched, used = set(), set()
        for row, col in zip(rows, cols):
            if costs[row, col] >= invalid:
                continue
            index = indices[row]
            self.tracks[index].update(detections[col].bbox, detections[col].confidence)
            matched.add(index)
            used.add(col)
        return matched, used

    def snapshots(self) -> tuple[PerceptionEntity, ...]:
        return tuple(t.snapshot() for t in self.tracks
                     if t.hits >= self.config.min_hits
                     and t.misses <= self.config.max_snapshot_misses
                     and (t.misses == 0 or self.last_timestamp - t.last_observed_timestamp
                          <= self.config.max_coast_age))
