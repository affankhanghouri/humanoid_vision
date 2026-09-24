"""Original Kalman + Hungarian tracker; tuning lives in VisionConfig."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from config import VisionConfig
from core.types import Detection, RenderTrack

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
        self.hits += 1
        self.misses = 0
        corrected_x = float(corrected[0, 0])
        corrected_y = float(corrected[1, 0])
        self.bbox = self._make_bbox(corrected_x, corrected_y)

    def _make_bbox(self, center_x, center_y):
        return (center_x - self.width / 2, center_y - self.height / 2, center_x + self.width / 2, center_y + self.height / 2)

    def snapshot(self):
        state = self.kalman.statePost
        return RenderTrack(track_id=self.id, class_name=self.class_name, confidence=self.confidence, center_x=float(state[0, 0]), center_y=float(state[1, 0]), velocity_x=float(state[2, 0]), velocity_y=float(state[3, 0]), width=self.width, height=self.height)

class MultiObjectTracker:

    def __init__(self, config: VisionConfig = VisionConfig()):
        self.config = config
        self.tracks = []

    def update(self, detections: list[Detection], timestamp: float) -> None:
        for track in self.tracks:
            track.predict_to(timestamp)
        if not self.tracks:
            for detection in detections:
                self._create_track(detection, timestamp)
            return
        if not detections:
            for track in self.tracks:
                track.misses += 1
            self._remove_dead_tracks()
            return
        rows = len(self.tracks)
        cols = len(detections)
        INVALID = self.config.invalid_match_cost
        cost_matrix = np.full((rows, cols), INVALID, dtype=np.float32)
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue
                iou = calculate_iou(track.bbox, detection.bbox)
                distance = normalized_center_distance(track.bbox, detection.bbox)
                if iou < self.config.match_min_iou and distance > self.config.match_max_distance:
                    continue
                iou_cost = 1.0 - iou
                distance_cost = min(distance / self.config.match_max_distance, 1.0)
                cost = self.config.match_iou_weight * iou_cost + self.config.match_distance_weight * distance_cost
                cost_matrix[track_index, detection_index] = cost
        row_ids, detection_ids = linear_sum_assignment(cost_matrix)
        matched_tracks = set()
        matched_detections = set()
        for track_index, detection_index in zip(row_ids, detection_ids):
            cost = cost_matrix[track_index, detection_index]
            if cost >= INVALID:
                continue
            if cost > self.config.match_max_cost:
                continue
            track = self.tracks[track_index]
            detection = detections[detection_index]
            track.update(detection.bbox, detection.confidence)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.misses += 1
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            self._create_track(detection, timestamp)
        self._remove_dead_tracks()

    def _create_track(self, detection, timestamp):
        self.tracks.append(Track(bbox=detection.bbox, class_name=detection.class_name, confidence=detection.confidence, timestamp=timestamp, config=self.config))

    def _remove_dead_tracks(self):
        self.tracks = [track for track in self.tracks if track.misses <= self.config.max_misses]

    def snapshots(self) -> tuple[RenderTrack, ...]:
        snapshots = []
        for track in self.tracks:
            if track.hits < self.config.min_hits:
                continue
            if track.misses > self.config.max_snapshot_misses:
                continue
            snapshots.append(track.snapshot())
        return tuple(snapshots)
