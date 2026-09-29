"""Central configuration for the vision system."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VisionConfig:

    # ==================================================
    # SOURCE
    # ==================================================

    video_path: str | int = str(
        PROJECT_ROOT
        / "test_data"
        / "test_video.mp4"
    )

    fallback_source_fps: float = 24.0

    # ==================================================
    # OBJECT DETECTION
    # ==================================================

    detector_backend: str = "onnx"

    model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "yolo26n.onnx"
    )

    inference_device: str = "cpu"

    image_size: int = 512
    confidence: float = 0.5
    # Weak detections maintain confirmed tracks; they never create new ones.
    track_low_confidence: float = 0.25

    warmup_runs: int = 5

    # ==================================================
    # TRACKING
    # ==================================================

    min_hits: int = 2
    max_misses: int = 2

    max_snapshot_misses: int = 1
    max_track_age: float = 1.0
    max_coast_age: float = 0.45

    measurement_noise: float = 0.1

    process_noise: tuple[float, ...] = (
        0.1,
        0.1,
        5.0,
        5.0,
    )

    initial_covariance: tuple[float, ...] = (
        10.0,
        10.0,
        100.0,
        100.0,
    )

    track_size_alpha: float = 0.80

    invalid_match_cost: float = 10000.0

    match_min_iou: float = 0.01
    match_max_distance: float = 0.65
    match_max_size_ratio: float = 3.0
    recovery_min_iou: float = 0.20

    match_iou_weight: float = 0.70
    match_distance_weight: float = 0.30

    match_max_cost: float = 0.90

    # ==================================================
    # BOX SMOOTHING
    # ==================================================

    smoothing_thresholds: tuple[float, ...] = (
        0.02,
        0.08,
        0.25,
    )

    smoothing_alphas: tuple[float, ...] = (
        0.65,
        0.75,
        0.85,
        1.0,
    )

    display_size_alpha: float = 0.80

    # ==================================================
    # RENDER PREDICTION
    # ==================================================

    max_render_age: float = 0.75
    max_prediction_age: float = 0.75

    # Shared sparse optical flow carries delayed boxes through camera motion.
    motion_enabled: bool = True
    motion_image_width: int = 192
    motion_max_points: int = 64
    motion_min_points: int = 6
    motion_min_interval: float = 0.15
    motion_max_extrapolation: float = 0.20
    motion_max_frame_gap: float = 0.25

    # ==================================================
    # DETECTION SCHEDULING
    # ==================================================

    detection_min_interval: float = 0.25

    detection_max_input_age: float = 0.15

    # ==================================================
    # POSE MODEL
    # ==================================================

    pose_model_path: str = str(
        PROJECT_ROOT
        / "models"
        / "yolo26n-pose-320.onnx"
    )

    pose_image_size: int = 320

    pose_confidence: float = 0.25

    pose_warmup_runs: int = 3

    # ==================================================
    # POSE SCHEDULING
    # ==================================================

    pose_min_interval: float = 0.50

    pose_max_input_age: float = 0.15

    pose_max_age: float = 0.85

    pose_min_keypoint_confidence: float = 0.30

    # ==================================================
    # POSE ↔ TRACK MATCHING
    # ==================================================

    pose_match_min_iou: float = 0.03

    pose_match_max_distance: float = 0.75

    pose_match_iou_weight: float = 0.75

    pose_match_distance_weight: float = 0.25

    pose_match_max_cost: float = 0.95

    # ==================================================
    # POSE DISPLAY SMOOTHING
    # ==================================================

    pose_smoothing_alpha: float = 0.35

    # ==================================================
    # DEMO / RENDER MODE
    # ==================================================

    # "demo" = existing LinkedIn UI; "demo_risk" adds the demo-only
    # perception priority map. Normal defaults remain unchanged.
    # "debug" = old simple renderer
    render_mode: str = "demo"
    risk_heatmap_enabled: bool = False
    road_demo_enabled: bool = False
    road_model_path: str = str(PROJECT_ROOT / "models" / "road_nano_640.onnx")
    road_request_interval: float = 1.0
    road_max_age: float = 1.35
    road_overlay_alpha: float = 0.06
    risk_heatmap_alpha: float = 0.36
    risk_heatmap_scale: int = 4
    record_output_path: str | None = None
    record_fps: float = 24.0

    # ==================================================
    # RUNTIME
    # ==================================================

    opencv_threads: int = 1

    # Detection and pose share the CPU; idle ONNX pools must not spin.
    onnx_intra_op_threads: int = 2

    metrics_window: int = 100

    empty_poll_seconds: float = 0.002

    repeat_poll_seconds: float = 0.001

    window_title: str = (
        "Real-Time Vision Pipeline"
    )
