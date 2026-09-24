"""Runtime settings and original tracking/smoothing tuning."""
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VisionConfig:
    video_path: str = str(PROJECT_ROOT / "test_data" / "test_video.mp4")
    detector_backend: str = "openvino"
    model_path: str = str(PROJECT_ROOT / "models" / "yolo26n_openvino_model")
    inference_device: str = "intel:cpu"
    image_size: int = 512
    confidence: float = 0.5
    min_hits: int = 2
    max_misses: int = 2
    max_snapshot_misses: int = 1
    measurement_noise: float = 1.0
    process_noise: tuple[float, ...] = (0.1, 0.1, 5.0, 5.0)
    initial_covariance: tuple[float, ...] = (10.0, 10.0, 100.0, 100.0)
    track_size_alpha: float = 0.25
    invalid_match_cost: float = 10000.0
    match_min_iou: float = 0.01
    match_max_distance: float = 2.5
    match_iou_weight: float = 0.70
    match_distance_weight: float = 0.30
    match_max_cost: float = 0.90
    smoothing_thresholds: tuple[float, ...] = (0.02, 0.08, 0.25)
    smoothing_alphas: tuple[float, ...] = (0.12, 0.30, 0.55, 0.80)
    display_size_alpha: float = 0.18
    max_render_age: float = 0.75
    max_prediction_age: float = 0.30
    opencv_threads: int = 1
    warmup_runs: int = 2
    metrics_window: int = 100
    fallback_source_fps: float = 24.0
    empty_poll_seconds: float = 0.002
    repeat_poll_seconds: float = 0.001
    window_title: str = "Real-Time Vision Pipeline"
