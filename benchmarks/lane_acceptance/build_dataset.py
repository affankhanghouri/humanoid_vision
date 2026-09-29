#!/usr/bin/env python3
"""Extract the fixed, manually reviewed 100-frame acceptance-set selection."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2

from common import SCENE_TYPES, atomic_write_json


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "test_data/test_video.mp4"
DEFAULT_DIRECTORY = ROOT / "benchmarks/lane_acceptance"

# Frames were selected from 10-frame contact sheets, with slow/stationary spans
# sampled more sparsely. Primary strata were assigned by manual visual review;
# scene tags can be refined during boundary annotation.
STRATA = {
    "intersection": (
        0, 40, 80, 120, 160, 200, 240, 550, 590, 630, 940,
        1870, 1910, 1950, 1990,
    ),
    "crosswalk_stop_line": (20, 60, 100, 140, 180, 220, 260, 570, 610, 1890),
    "parked_cars_curb": (
        270, 290, 310, 330, 350, 370, 390, 410, 430, 450,
        650, 710, 770, 830, 890,
    ),
    "clear_normal_lane": (
        460, 480, 500, 520, 540, 640, 660, 680, 700, 720,
        740, 760, 780, 800, 820, 840, 860, 880, 900, 920,
    ),
    "occlusion": (970, 1010, 1050, 1090, 1130, 1170, 1210, 1250, 1310, 1370),
    "faded_lane": (1430, 1470, 1510, 1550, 1590, 1630, 1670, 1720, 1780, 1840),
    "unmarked_road": (1450, 1490, 1530, 1570, 1610, 1650, 1690, 1740, 1800, 1820),
    "turning_curved_road": (1410, 1700, 1760, 1860, 1880, 1900, 1920, 1940, 1960, 1980),
}


def scene_tags(frame_id: int, stratum: str) -> list[str]:
    tags = [stratum]
    if stratum == "crosswalk_stop_line":
        tags.append("intersection")
    elif stratum == "intersection" and frame_id <= 600:
        tags.append("crosswalk_stop_line")
    elif stratum == "turning_curved_road" and frame_id >= 1860:
        tags.extend(("intersection", "unmarked_road"))
    return sorted(set(tags), key=SCENE_TYPES.index)


def selection() -> list[tuple[int, str]]:
    selected = [(frame_id, stratum) for stratum, ids in STRATA.items() for frame_id in ids]
    selected.sort()
    frame_ids = [frame_id for frame_id, _ in selected]
    counts = Counter(stratum for _, stratum in selected)
    assert len(selected) == 100
    assert len(set(frame_ids)) == 100
    assert min(b - a for a, b in zip(frame_ids, frame_ids[1:])) >= 10
    assert counts == {
        "clear_normal_lane": 20,
        "intersection": 15,
        "parked_cars_curb": 15,
        "crosswalk_stop_line": 10,
        "faded_lane": 10,
        "unmarked_road": 10,
        "turning_curved_road": 10,
        "occlusion": 10,
    }
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.output / "dataset.json"
    if dataset_path.exists() and not args.force:
        raise FileExistsError(
            f"{dataset_path} already exists; refusing to overwrite annotations. Use --force explicitly."
        )
    selected = selection()
    selected_by_id = dict(selected)
    frames_directory = args.output / "frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = {}
    frame_id = 0
    while frame_id <= selected[-1][0]:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id in selected_by_id:
            relative_image = Path("frames") / f"frame_{frame_id:04d}.jpg"
            destination = args.output / relative_image
            if not cv2.imwrite(
                str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            ):
                raise RuntimeError(f"Could not write {destination}")
            extracted[frame_id] = str(relative_image)
        frame_id += 1
    capture.release()
    missing = sorted(set(selected_by_id) - set(extracted))
    if missing:
        raise RuntimeError(f"Could not extract selected frames: {missing}")
    expected_names = {Path(path).name for path in extracted.values()}
    for stale_image in frames_directory.glob("frame_*.jpg"):
        if stale_image.name not in expected_names:
            stale_image.unlink()
    counts = Counter(stratum for _, stratum in selected)
    annotations = []
    for frame_id, stratum in selected:
        annotations.append(
            {
                "frame_id": frame_id,
                "timestamp": round(frame_id / fps, 6),
                "image": extracted[frame_id],
                "selection_stratum": stratum,
                "scene_type": scene_tags(frame_id, stratum),
                "review_status": "pending",
                "ego_lane_status": None,
                "annotation_flag": None,
                "left_boundary": {"status": None, "points": []},
                "right_boundary": {"status": None, "points": []},
                "notes": "",
            }
        )
    dataset = {
        "schema_version": 1,
        "name": "humanoid_vision_lane_acceptance_v1",
        "video": {
            "path": str(args.video.relative_to(ROOT)),
            "fps": fps,
            "width": width,
            "height": height,
            "reported_frame_count": reported_frames,
            "last_selected_frame_id": selected[-1][0],
        },
        "selection": {
            "method": "manual contact-sheet review with near-duplicate rejection",
            "minimum_gap_frames": min(
                b - a for (a, _), (b, _) in zip(selected, selected[1:])
            ),
            "minimum_gap_seconds": min(
                b - a for (a, _), (b, _) in zip(selected, selected[1:])
            ) / fps,
            "primary_stratum_distribution": dict(sorted(counts.items())),
        },
        "annotation_policy": {
            "human_review_required": True,
            "automatic_model_annotation_allowed": False,
            "point_order": "click order; bottom-to-top is recommended",
            "invalid_marker": "NO_VALID_EGO_LANE",
        },
        "progress": {"last_frame_id": None, "completed": 0},
        "annotations": annotations,
    }
    atomic_write_json(dataset_path, dataset)
    print(f"Extracted {len(annotations)} frames to {frames_directory}")
    print(f"Wrote pending human annotations to {dataset_path}")
    print(dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
