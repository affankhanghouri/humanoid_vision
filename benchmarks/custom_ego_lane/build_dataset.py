#!/usr/bin/env python3
"""Extract leakage-safe clip-split frames for custom ego-lane annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "benchmarks/lane_acceptance/dataset.json"
OUTPUT = ROOT / "benchmarks/custom_ego_lane/data/dataset.json"


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def dhash(image: np.ndarray) -> np.ndarray:
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (17, 16))
    return (gray[:, 1:] > gray[:, :-1]).reshape(-1)


def hamming(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def sampling_stride(frame_count: int, fps: float, interval: float, limit: int) -> int:
    """Cover the full clip even when a per-clip sample cap is active."""
    return max(1, int(round(interval * fps)), int(np.ceil(frame_count / limit)))


def parse_source(value: str) -> tuple[Path, str, str]:
    try:
        path, clip, split = value.rsplit(":", 2)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected PATH:CLIP_ID:train|validation") from error
    if split not in {"train", "validation"}:
        raise argparse.ArgumentTypeError("split must be train or validation")
    return Path(path).resolve(), clip, split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--interval", type=float, default=0.5, help="minimum seconds between candidates")
    parser.add_argument("--train-max-per-clip", type=int, default=180)
    parser.add_argument("--validation-max-per-clip", type=int, default=110)
    parser.add_argument("--dedup-distance", type=int, default=10)
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    frozen = json.loads(FROZEN.read_text())
    frozen_video = (ROOT / frozen["video"]["path"]).resolve()
    if any(path == frozen_video for path, _, _ in args.source):
        raise SystemExit(
            "REFUSED: the frozen acceptance source video is quarantined in full; neighboring frames would leak final-test scenes"
        )
    clips = [clip for _, clip, _ in args.source]
    if len(clips) != len(set(clips)):
        raise SystemExit("clip IDs must be unique")
    if {split for _, _, split in args.source} != {"train", "validation"}:
        raise SystemExit("at least one independent train clip and one validation clip are required")

    output = args.output.resolve()
    frames_dir = output.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    sources = []
    item_id = 0
    for path, clip, split in args.source:
        if not path.is_file():
            raise FileNotFoundError(path)
        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_limit = args.train_max_per_clip if split == "train" else args.validation_max_per_clip
        stride = sampling_stride(frame_count, fps, args.interval, clip_limit)
        accepted_hashes: list[np.ndarray] = []
        source_hash = fingerprint(path)
        taken = 0
        candidate_count = 0
        duplicate_rejections = 0
        frame_id = -1
        while taken < clip_limit:
            ok, image = capture.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % stride:
                continue
            candidate_count += 1
            signature = dhash(image)
            if accepted_hashes and min(hamming(signature, old) for old in accepted_hashes[-20:]) < args.dedup_distance:
                duplicate_rejections += 1
                continue
            accepted_hashes.append(signature)
            image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
            relative = Path("frames") / f"{split}_{clip}_{frame_id:06d}.jpg"
            cv2.imwrite(str(output.parent / relative), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            records.append({
                "item_id": item_id, "frame_id": frame_id, "timestamp": frame_id / fps,
                "image": str(relative), "source_video": str(path), "source_sha256": source_hash,
                "clip_id": clip, "split": split, "selection_stratum": "unassigned",
                "scene_type": ["unmarked_road"], "review_status": "pending",
                "label_origin": "UNREVIEWED", "ego_lane_status": None, "annotation_flag": None,
                "left_boundary": {"status": None, "points": []},
                "right_boundary": {"status": None, "points": []}, "notes": "",
            })
            item_id += 1
            taken += 1
        capture.release()
        sources.append({"path": str(path), "sha256": source_hash, "clip_id": clip, "split": split, "frames": taken, "sampling_stride_frames": stride, "candidate_slots_considered": candidate_count, "near_duplicate_rejections": duplicate_rejections})
    payload = {
        "schema_version": 1, "name": "humanoid_custom_ego_lane_training_v1",
        "video": {"width": args.width, "height": args.height}, "sources": sources,
        "split_policy": "source video or non-overlapping contiguous clip; frozen acceptance source quarantined",
        "progress": {"last_frame_id": None, "completed": 0}, "annotations": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "records": len(records), "sources": sources}, indent=2))


if __name__ == "__main__":
    main()
