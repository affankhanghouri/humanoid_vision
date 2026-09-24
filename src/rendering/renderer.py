"""Prediction, smoothing and drawing; never changes tracking state."""
import cv2
from config import VisionConfig
from core.types import FramePacket, TrackingResult
from tracking.smoothing import DisplayBoxSmoother

def draw_track(frame, track, age, smoother, config):
    height, width = frame.shape[:2]
    prediction_age = min(age, config.max_prediction_age)
    predicted_center_x = track.center_x + track.velocity_x * prediction_age
    predicted_center_y = track.center_y + track.velocity_y * prediction_age
    raw_box = (predicted_center_x - track.width / 2, predicted_center_y - track.height / 2, predicted_center_x + track.width / 2, predicted_center_y + track.height / 2)
    x1, y1, x2, y2 = smoother.smooth(track.track_id, raw_box)
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width - 1))
    y2 = max(0, min(int(y2), height - 1))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f'ID {track.track_id} {track.class_name} {track.confidence:.2f}'
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


class Renderer:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.smoother = DisplayBoxSmoother(config)

    def render(self, packet: FramePacket, result: TrackingResult | None, display_fps: float):
        frame = packet.frame.copy()
        age = None
        inference_ms = detector_hz = 0.0
        if result is not None:
            age = max(0.0, packet.timestamp - result.timestamp)
            inference_ms, detector_hz = result.inference_ms, result.detector_hz
            if age <= self.config.max_render_age:
                for track in result.tracks:
                    draw_track(frame, track, age, self.smoother, self.config)
                self.smoother.keep_only(t.track_id for t in result.tracks)
        lines = [f"Display: {display_fps:.1f} FPS", f"Detector: {detector_hz:.1f} Hz", f"YOLO: {inference_ms:.1f} ms"]
        if age is not None:
            lines.append(f"Detection age: {age * 1000:.0f} ms")
        for index, text in enumerate(lines):
            cv2.putText(frame, text, (20, 40 + index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
