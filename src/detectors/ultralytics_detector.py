"""The only application module that handles Ultralytics objects."""
import numpy as np
from ultralytics import YOLO
from core.types import Detection
from detectors.base import Detector


class UltralyticsDetector(Detector):
    def __init__(self, model_path: str, image_size: int, confidence: float,
                 device: str, warmup_runs: int):
        self.image_size = image_size
        self.confidence = confidence
        self.device = device
        self.warmup_runs = warmup_runs
        self.model = YOLO(model_path)

    def _predict(self, frame):
        return self.model.predict(source=frame, imgsz=self.image_size,
                                  conf=self.confidence, device=self.device,
                                  verbose=False)[0]

    def warmup(self) -> None:
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(self.warmup_runs):
            self._predict(dummy)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._predict(frame)
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            detections.append(Detection(
                bbox=tuple(box.xyxy[0].tolist()), class_id=class_id,
                class_name=result.names[class_id], confidence=float(box.conf[0]),
            ))
        return detections
