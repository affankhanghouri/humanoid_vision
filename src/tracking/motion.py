"""Carry source-frame boxes through measured image motion, without new inference.

One sparse flow field is shared by all entities. History is bounded by tracking
freshness; a broken chain falls back to the caller's Kalman prediction. This is
an image-space estimate, not a new detector observation or world-space velocity.
"""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from config import VisionConfig


@dataclass(frozen=True)
class MotionStep:
    start: float
    end: float
    before: np.ndarray
    after: np.ndarray
    affine: np.ndarray | None

    def move(self, box, fraction=1.0):
        if self.affine is None:
            return None
        x1, y1, x2, y2 = box
        matrix = self.affine.copy()
        inside = ((self.before[:, 0] >= x1) & (self.before[:, 0] <= x2)
                  & (self.before[:, 1] >= y1) & (self.before[:, 1] <= y2))
        before, after = self.before[inside], self.after[inside]
        if len(before) >= 3:
            # Local residual handles object motion/parallax relative to the camera.
            expected = before @ matrix[:, :2].T + matrix[:, 2]
            matrix[:, 2] += np.median(after - expected, axis=0)
        if fraction != 1:
            identity = np.eye(2, 3)
            matrix = identity + fraction * (matrix - identity)
        corners = np.array(((x1, y1), (x2, y1), (x2, y2), (x1, y2)))
        moved = corners @ matrix[:, :2].T + matrix[:, 2]
        return np.concatenate((moved.min(axis=0), moved.max(axis=0)))


@dataclass(frozen=True)
class MotionSnapshot:
    steps: tuple[MotionStep, ...]
    timestamp: float
    frame_id: int
    scale: tuple[float, float]


class BoxMotionHistory:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.steps = deque(maxlen=120)
        self.previous = None
        self.timestamp = None
        self.display_timestamp = None
        self.frame_id = None
        self.scale = None
        self.cache = {}
        self.cache_source = None

    def snapshot(self):
        if self.timestamp is None:
            return None
        return MotionSnapshot(tuple(self.steps), self.timestamp, self.frame_id, tuple(self.scale))

    def update(self, packet):
        self.display_timestamp = packet.timestamp
        if packet.motion is not None:
            motion = packet.motion
            if (self.timestamp is not None and (motion.timestamp < self.timestamp
                    or motion.timestamp - self.timestamp > self.config.motion_max_frame_gap)):
                self.cache.clear()
            self.steps = deque(motion.steps, maxlen=120)
            self.timestamp, self.frame_id = motion.timestamp, motion.frame_id
            self.scale = np.asarray(motion.scale)
            return
        if not self.config.motion_enabled or packet.frame_id == self.frame_id:
            return
        if (self.timestamp is not None
                and 0 < packet.timestamp - self.timestamp < self.config.motion_min_interval
                and packet.frame_id > self.frame_id
                and packet.frame.shape[:2] == getattr(self, 'source_shape', None)):
            return
        self.source_shape = packet.frame.shape[:2]
        h, w = self.source_shape
        width = min(w, self.config.motion_image_width)
        height = max(1, round(h * width / w))
        small = cv2.resize(packet.frame, (width, height), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        scale = np.array((width / w, height / h), dtype=np.float64)
        contiguous = (self.previous is not None and self.previous.shape == gray.shape
                      and np.array_equal(self.scale, scale)
                      and 0 < packet.timestamp - self.timestamp <= self.config.motion_max_frame_gap
                      and packet.frame_id > self.frame_id)
        if contiguous:
            self.steps.append(self._estimate(self.previous, gray, self.timestamp, packet.timestamp))
        else:
            self.steps.clear()
            self.cache.clear()
        self.previous, self.timestamp = gray, packet.timestamp
        self.frame_id, self.scale = packet.frame_id, scale
        while self.steps and packet.timestamp - self.steps[0].end > self.config.max_render_age:
            self.steps.popleft()

    def _estimate(self, previous, gray, start, end):
        empty = np.empty((0, 2), np.float32)
        failed = MotionStep(start, end, empty, empty, None)
        points = cv2.goodFeaturesToTrack(previous, self.config.motion_max_points,
                                        qualityLevel=.015, minDistance=6, blockSize=3)
        if points is None or len(points) < self.config.motion_min_points:
            return failed
        options = dict(winSize=(11, 11), maxLevel=1,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, .03))
        moved, status, error = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
        if moved is None:
            return failed
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, moved, None, **options)
        if back is None:
            return failed
        valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
                 & (error.ravel() < 25)
                 & (np.linalg.norm(points - back, axis=2).ravel() < 1.0)
                 & np.isfinite(moved).all(axis=(1, 2)))
        before, after = points[valid, 0], moved[valid, 0]
        if len(before) < self.config.motion_min_points:
            return failed
        affine, inliers = cv2.estimateAffinePartial2D(
            before, after, method=cv2.RANSAC, ransacReprojThreshold=2,
            maxIters=100, confidence=.95, refineIters=3)
        if affine is None or inliers.mean() < .45:
            return failed
        zoom = np.linalg.norm(affine[:, 0])
        if not .85 <= zoom <= 1.18:
            return failed
        before.setflags(write=False)
        after.setflags(write=False)
        affine.setflags(write=False)
        return MotionStep(start, end, before, after, affine)

    def project(self, entity_id, box, source_timestamp):
        """Return a current-frame box or None when the motion chain is incomplete."""
        if not self.config.motion_enabled or self.timestamp is None or self.scale is None:
            return None
        if source_timestamp != self.cache_source:
            self.cache.clear()
            self.cache_source = source_timestamp
        cached = self.cache.get(entity_id)
        if cached is None:
            current = np.asarray(box, dtype=np.float64) * np.tile(self.scale, 2)
            at = source_timestamp
        else:
            current, at = cached
        if at > self.timestamp or self.display_timestamp - source_timestamp > self.config.max_render_age:
            return None
        for step in self.steps:
            if step.end <= at:
                continue
            if step.start > at + 1e-6:
                return None
            fraction = (step.end - at) / (step.end - step.start)
            current = step.move(current, fraction)
            if current is None:
                return None
            at = step.end
        if abs(at - self.timestamp) > 1e-6:
            return None
        self.cache[entity_id] = (current, at)
        # Between flow passes, extrapolate only the short unmeasured interval.
        # Do not put extrapolated geometry back into the measured history/cache.
        extra = self.display_timestamp - self.timestamp
        if extra > 0:
            if not self.steps:
                return None
            step = self.steps[-1]
            if extra > self.config.motion_max_extrapolation or step.affine is None:
                return None
            projected = step.move(current, extra / (step.end - step.start))
            if projected is not None:
                current = projected
        return tuple(current / np.tile(self.scale, 2))
