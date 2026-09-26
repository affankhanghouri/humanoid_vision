"""YOLO pose estimation using the fixed-size ONNX model on CPU."""
import numpy as np
from ultralytics import YOLO
from core.pose_types import PosePerson, PosePoint
from pose.base import PoseEstimator
from core.onnx_runtime import configure_cpu_onnx


class UltralyticsPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str, image_size: int, confidence: float,
                 warmup_runs: int = 3, onnx_threads: int = 2):
        self.image_size = image_size
        self.confidence = confidence
        self.warmup_runs = warmup_runs
        self.model = YOLO(model_path, task="pose")
        configure_cpu_onnx(self.model, model_path, onnx_threads)

    def _predict(self, frame):
        return self.model.predict(source=frame, imgsz=self.image_size,
                                  conf=self.confidence, device="cpu", verbose=False)[0]

    def warmup(self) -> None:
        frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        for _ in range(self.warmup_runs):
            self._predict(frame)

    def estimate(self, frame: np.ndarray) -> tuple[PosePerson, ...]:
        result = self._predict(frame)
        if result.boxes is None or result.keypoints is None:
            return ()
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        points = result.keypoints.data.cpu().numpy()
        people = []
        for box, score, joints in zip(boxes, scores, points):
            people.append(PosePerson(
                bbox=tuple(float(value) for value in box),
                confidence=float(score),
                keypoints=tuple(PosePoint(float(p[0]), float(p[1]),
                                         float(p[2]) if len(p) > 2 else 1.0)
                                for p in joints),
            ))
        return tuple(people)
