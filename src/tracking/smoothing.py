"""Adaptive smoothing for display only."""
import numpy as np
from config import VisionConfig

class DisplayBoxSmoother:

    def __init__(self, config: VisionConfig = VisionConfig()):
        self.config = config
        self.boxes = {}

    def smooth(self, track_id, target_box):
        x1, y1, x2, y2 = target_box
        target = np.array([x1, y1, x2, y2], dtype=np.float32)
        if track_id not in self.boxes:
            self.boxes[track_id] = target
            return target
        old = self.boxes[track_id]
        old_center_x = (old[0] + old[2]) / 2
        old_center_y = (old[1] + old[3]) / 2
        new_center_x = (target[0] + target[2]) / 2
        new_center_y = (target[1] + target[3]) / 2
        movement = np.hypot(new_center_x - old_center_x, new_center_y - old_center_y)
        width = max(1.0, target[2] - target[0])
        height = max(1.0, target[3] - target[1])
        box_size = np.hypot(width, height)
        movement_ratio = movement / box_size
        if movement_ratio < self.config.smoothing_thresholds[0]:
            alpha_position = self.config.smoothing_alphas[0]
        elif movement_ratio < self.config.smoothing_thresholds[1]:
            alpha_position = self.config.smoothing_alphas[1]
        elif movement_ratio < self.config.smoothing_thresholds[2]:
            alpha_position = self.config.smoothing_alphas[2]
        else:
            alpha_position = self.config.smoothing_alphas[3]
        alpha_size = self.config.display_size_alpha
        old_width = old[2] - old[0]
        old_height = old[3] - old[1]
        target_width = target[2] - target[0]
        target_height = target[3] - target[1]
        smooth_center_x = old_center_x + alpha_position * (new_center_x - old_center_x)
        smooth_center_y = old_center_y + alpha_position * (new_center_y - old_center_y)
        smooth_width = old_width + alpha_size * (target_width - old_width)
        smooth_height = old_height + alpha_size * (target_height - old_height)
        smoothed = np.array([smooth_center_x - smooth_width / 2, smooth_center_y - smooth_height / 2, smooth_center_x + smooth_width / 2, smooth_center_y + smooth_height / 2], dtype=np.float32)
        self.boxes[track_id] = smoothed
        return smoothed

    def keep_only(self, active_ids):
        active_ids = set(active_ids)
        dead_ids = [track_id for track_id in self.boxes if track_id not in active_ids]
        for track_id in dead_ids:
            del self.boxes[track_id]
