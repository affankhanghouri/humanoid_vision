"""The only application module that handles Ultralytics objects."""
import numpy as np
from ultralytics import YOLO
from core.types import Detection
from core.onnx_runtime import configure_cpu_onnx
from detectors.base import Detector


class UltralyticsDetector(Detector):
    def __init__(self, model_path: str, image_size: int, confidence: float,
                 device: str, warmup_runs: int, onnx_threads: int = 2):
        self.image_size = image_size
        self.confidence = confidence
        self.device = device
        self.warmup_runs = warmup_runs
        self.model = YOLO(model_path)
        if device == "cpu":
            configure_cpu_onnx(self.model, model_path, onnx_threads)

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
        boxes = result.boxes
        if boxes is None:
            return []
        # Transfer the arrays once instead of slicing PyTorch wrappers per box.
        coordinates = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        return [Detection(
            bbox=tuple(float(value) for value in bbox), class_id=int(class_id),
            class_name=result.names[int(class_id)], confidence=float(confidence),
        ) for bbox, class_id, confidence in zip(coordinates, classes, confidences)]
