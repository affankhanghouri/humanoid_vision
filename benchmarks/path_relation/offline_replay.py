#!/usr/bin/env python3
"""Offline path-relation replay using existing detector, tracker and road model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from config import VisionConfig
from core.perception_state import PerceptionState
from detectors.factory import create_detector
from rejected_path_relation import PathRelationEngine, PathRelationEngineV2
from rejected_path_relation.types import PathRelationState
from road.segmentation import RoadSegmenter
from tracking.tracker import MultiObjectTracker, Track
from tracking.motion import BoxMotionHistory
from core.types import FramePacket


COLORS = {
    PathRelationState.UNKNOWN: (145, 145, 145),
    PathRelationState.OFF_DRIVABLE: (255, 110, 40),
    PathRelationState.ON_DRIVABLE_OUTSIDE_CORRIDOR: (60, 205, 70),
    PathRelationState.ENTERING_CORRIDOR: (0, 225, 255),
    PathRelationState.IN_CORRIDOR: (40, 40, 235),
    PathRelationState.CROSSING_CORRIDOR: (0, 135, 255),
    PathRelationState.LEAVING_CORRIDOR: (190, 80, 210),
}


def stats(values):
    if not values:
        return {}
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def frame_mask(native, content_rect, frame_size):
    left, top, width, height = content_rect
    content = native[top:top + height, left:left + width]
    return cv2.resize(content, frame_size, interpolation=cv2.INTER_NEAREST)


def render(frame, observations, engine, road):
    output = frame.copy()
    frame_size = (frame.shape[1], frame.shape[0])
    if road is not None:
        road_full = frame_mask(road.drivable_mask, road.content_rect, frame_size)
        tint = np.zeros_like(output)
        tint[:] = (45, 120, 45)
        selected = road_full > 0
        output[selected] = cv2.addWeighted(output, .86, tint, .14, 0)[selected]
    if road is not None and engine.corridor is not None:
        corridor_full = frame_mask(engine.corridor.mask, road.content_rect, frame_size)
        tint = np.zeros_like(output)
        tint[:] = (45, 45, 210)
        selected = corridor_full > 0
        output[selected] = cv2.addWeighted(output, .68, tint, .32, 0)[selected]
    for item in observations:
        color = COLORS[item.state]
        if item.contact_point is not None:
            point = tuple(int(round(value)) for value in item.contact_point)
            cv2.circle(output, point, 6, color, -1, cv2.LINE_AA)
            label = f"#{item.track_id} {item.state.value} {item.confidence:.2f}"
            cv2.putText(output, label, (point[0] + 8, max(20, point[1] - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
            if item.predicted_contact_point is not None:
                predicted = tuple(int(round(value)) for value in item.predicted_contact_point)
                cv2.arrowedLine(output, point, predicted, color, 2, cv2.LINE_AA, tipLength=.2)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "test_data/test_video.mp4")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "output/replay")
    parser.add_argument("--sample-interval", type=float, default=.5)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics = args.output / "diagnostics"
    diagnostics.mkdir(exist_ok=True)

    config = replace(VisionConfig(), video_path=str(args.video), warmup_runs=2,
                     road_demo_enabled=True, onnx_intra_op_threads=2)
    detector = create_detector(config)
    road_model = RoadSegmenter(config.road_model_path, threads=2)
    detector.warmup()
    road_model.warmup(2)
    Track.next_id = 1
    tracker = MultiObjectTracker(config)
    engine = (PathRelationEngineV2(tracking_max_age=config.max_render_age) if args.version == "v2" else
              PathRelationEngine(road_max_age=.55, tracking_max_age=config.max_render_age))
    motion_history = BoxMotionHistory(replace(config, motion_min_interval=0))

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or config.fallback_source_fps
    detection_interval = max(1, round(fps * config.detection_min_interval))
    road_interval = max(1, round(fps * config.road_request_interval))
    sample_interval = max(1, round(fps * args.sample_interval))
    frame_id = 0
    state_value = None
    road_value = None
    rows = []
    path_times = []
    detection_times = []
    road_times = []
    state_counts = Counter()
    transitions = Counter()
    previous_states = {}
    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_id / fps
        motion_history.update(FramePacket(frame_id, timestamp, frame))
        motion_snapshot = motion_history.snapshot()
        if frame_id % detection_interval == 0:
            tick = time.perf_counter()
            detections = detector.detect(frame)
            detection_ms = (time.perf_counter() - tick) * 1000
            detection_times.append(detection_ms)
            tracker.update(detections, timestamp)
            state_value = PerceptionState(
                frame_id, timestamp, timestamp + detection_ms / 1000,
                tracker.snapshots(), detection_ms,
                len(detection_times) / max(timestamp + 1 / fps, 1 / fps),
                float(np.mean(detection_times)), float(np.percentile(detection_times, 95)))
        if frame_id % road_interval == 0:
            tick = time.perf_counter()
            observed = road_model.observe(frame, frame_id, timestamp)
            road_ms = (time.perf_counter() - tick) * 1000
            road_times.append(road_ms)
            road_value = replace(observed, produced_timestamp=timestamp + road_ms / 1000)

        tick = time.perf_counter()
        observations = engine.update(
            state_value, road_value, frame_id=frame_id, timestamp=timestamp,
            frame_size=(frame.shape[1], frame.shape[0]), **({"motion": motion_snapshot} if args.version == "v2" else {}))
        path_times.append((time.perf_counter() - tick) * 1000)
        if state_value is not None and frame_id % detection_interval == 0:
            for item in observations:
                state_counts[item.state.value] += 1
                previous = previous_states.get(item.track_id)
                if previous is not None and previous != item.state.value:
                    transitions[(previous, item.state.value)] += 1
                previous_states[item.track_id] = item.state.value
                row = asdict(item)
                row["state"] = item.state.value
                if "road_support_kind" in row:
                    row["road_support_kind"] = item.road_support_kind.value
                row["display_frame_id"] = frame_id
                row["display_timestamp"] = timestamp
                row["class_name"] = next(
                    entity.class_name for entity in state_value.entities
                    if entity.entity_id == item.track_id)
                rows.append(row)
        if frame_id % sample_interval == 0:
            visual = render(frame, observations, engine, road_value)
            cv2.imwrite(str(diagnostics / f"frame_{frame_id:05d}.jpg"), visual,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            if args.version == "v2" and road_value is not None and engine.cleaned_mask is not None and engine.corridor is not None:
                panels = []
                for label, native in (("RAW ROAD MASK", road_value.drivable_mask), ("CLEANED ROAD MASK", engine.cleaned_mask), ("FORWARD CORRIDOR", engine.corridor.mask)):
                    panel = cv2.cvtColor(frame_mask(native, road_value.content_rect, (frame.shape[1], frame.shape[0])) * 255, cv2.COLOR_GRAY2BGR)
                    cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2, cv2.LINE_AA)
                    panels.append(panel)
                cv2.imwrite(str(diagnostics / f"masks_{frame_id:05d}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
        frame_id += 1
    cap.release()
    wall = time.perf_counter() - start
    summary = {
        "video": str(args.video), "frames": frame_id, "fps": fps,
        "duration_seconds": frame_id / fps, "wall_seconds": wall,
        "detection_interval_frames": detection_interval,
        "road_interval_frames": road_interval,
        "path_compute": stats(path_times),
        "detection": stats(detection_times), "road": stats(road_times),
        "state_counts_on_unique_tracking_observations": dict(state_counts),
        "state_transitions": {f"{a}->{b}": count for (a, b), count in transitions.items()},
        "observations": len(rows),
    }
    (args.output / "observations.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
