from config import VisionConfig
from detectors.base import Detector


def create_detector(config: VisionConfig) -> Detector:
    if config.detector_backend not in {"openvino", "pytorch", "onnx"}:
        raise ValueError(f"Unknown detector backend: {config.detector_backend}")
    from detectors.ultralytics_detector import UltralyticsDetector
    return UltralyticsDetector(config.model_path, config.image_size,
                               min(config.confidence, config.track_low_confidence), config.inference_device,
                               config.warmup_runs, config.onnx_intra_op_threads)
