"""Command-line entry point for the road-scene perception pipeline."""
import argparse
import logging
from dataclasses import replace
from pathlib import Path

from config import VisionConfig
from pipeline.vision_pipeline import VisionPipeline


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CPU road-scene perception using newest-frame scheduling.",
    )
    parser.add_argument('--risk-demo', action='store_true',
                        help='enable the demo-only Perception Priority Map and road overlay')
    parser.add_argument('--no-road-overlay', action='store_true',
                        help='run the risk map without the optional road model')
    parser.add_argument('--video', help='override the configured video source')
    parser.add_argument('--record', metavar='OUTPUT.mp4',
                        help='write clean rendered frames without desktop/window chrome')
    parser.add_argument('--verbose', action='store_true',
                        help='show per-inference diagnostic logging')
    return parser.parse_args(argv)


def validate_config(config: VisionConfig) -> None:
    """Fail before model initialization with actionable asset errors."""
    required = {
        "detector model": config.model_path,
        "pose model": config.pose_model_path,
    }
    if config.road_demo_enabled:
        required["optional road model"] = config.road_model_path
    for label, value in required.items():
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if isinstance(config.video_path, str):
        video_path = Path(config.video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing input video: {video_path}")

    if config.render_mode not in {"demo", "demo_risk", "debug"}:
        raise ValueError(f"Unsupported render mode: {config.render_mode}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    config = VisionConfig()
    updates = {}
    if args.risk_demo:
        updates.update(render_mode='demo_risk', risk_heatmap_enabled=True,
                       road_demo_enabled=not args.no_road_overlay,
                       window_title='Perception Priority Map Demo')
    if args.video:
        updates['video_path'] = args.video
    if args.record:
        updates['record_output_path'] = args.record
    if updates:
        config = replace(config, **updates)
    validate_config(config)
    logging.getLogger(__name__).info(
        "Vision pipeline starting | detector=%s | device=%s | input=%s | "
        "tracking=enabled | pose=enabled | motion=%s | road=%s | mode=%s",
        Path(config.model_path).name, config.inference_device.upper(),
        config.video_path, "enabled" if config.motion_enabled else "disabled",
        "enabled" if config.road_demo_enabled else "disabled", config.render_mode,
    )
    VisionPipeline(config).run()


if __name__ == '__main__':
    main()
