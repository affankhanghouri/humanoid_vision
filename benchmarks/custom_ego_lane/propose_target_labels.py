#!/usr/bin/env python3
"""Generate conservative TwinLiteNet lane-mask proposals for target-camera review.

The output is never ground truth: every record remains pending and keeps explicit
AUTO_* provenance until a reviewer accepts or rejects its overlay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = ROOT / "benchmarks/test_road_geometry.py"


def load_geometry_module():
    spec = importlib.util.spec_from_file_location("target_geometry", GEOMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def boundary(fit, height: int, width: int) -> dict:
    if fit is None:
        return {"status": "not_visible", "points": []}
    ys = np.linspace(height * .42, height * .95, 12)
    xs = np.polyval(fit.coefficients, ys)
    keep = (xs >= 0) & (xs < width)
    points = [[round(float(x), 2), round(float(y), 2)] for x, y in zip(xs[keep], ys[keep])]
    return {"status": "visible", "points": points} if len(points) >= 4 else {"status": "not_visible", "points": []}


def horizontal_marking_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=image.shape[1] // 8, maxLineGap=15)
    if lines is None:
        return 0.0
    score = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if y1 < image.shape[0] * .42 or y2 < image.shape[0] * .42:
            continue
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > 5 * max(dy, 1):
            score += dx / image.shape[1]
    return score


def scene_tags(item: dict, image: np.ndarray, geometry, horizontal_score: float) -> list[str]:
    tags: list[str] = []
    timestamp = float(item["timestamp"])
    if item["clip_id"] == "target_train" and 252 <= timestamp < 396:
        tags.append("unmarked_road")
    elif geometry.lane_left is not None or geometry.lane_right is not None:
        tags.append("clear_normal_lane")
    else:
        tags.append("faded_lane")
    if horizontal_score >= .45:
        tags.extend(("intersection", "crosswalk_stop_line"))
    if item["clip_id"] == "target_validation" or timestamp >= 584:
        tags.append("parked_cars_curb")
    if geometry.lane_left is not None and abs(float(geometry.lane_left.coefficients[0])) > .001:
        tags.append("turning_curved_road")
    # Large foreground objects and the motorcycle cluster are explicit occlusion cases.
    if (item["clip_id"] == "target_train" and timestamp >= 584):
        tags.append("occlusion")
    return list(dict.fromkeys(tags))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models/twinlitenetplus_nano.onnx")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    if any(item["split"] not in {"train", "validation"} for item in payload["annotations"]):
        raise SystemExit("REFUSED: target proposal manifest may contain train/validation only")
    geometry_module = load_geometry_module()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    input_spec = session.get_inputs()[0]
    _, _, model_h, model_w = input_spec.shape
    road_name, lane_name = geometry_module.find_output_names(session)
    counts: dict[str, int] = {}
    for number, item in enumerate(payload["annotations"], 1):
        if item.get("label_origin") != "UNREVIEWED":
            continue
        image = cv2.imread(str(args.dataset.parent / item["image"]))
        if image is None:
            raise FileNotFoundError(item["image"])
        height, width = image.shape[:2]
        tensor, crop = geometry_module.preprocess(image, model_h, model_w)
        road_output, lane_output = session.run([road_name, lane_name], {input_spec.name: tensor})
        road_mask = geometry_module.decode_mask(road_output, crop, width, height)
        lane_mask = geometry_module.decode_mask(lane_output, crop, width, height)
        geometry, _, _, _ = geometry_module.build_geometry(lane_mask, road_mask)
        left = boundary(geometry.lane_left, height, width)
        right = boundary(geometry.lane_right, height, width)
        visible = sum(side["status"] == "visible" for side in (left, right))
        confidences = [geometry.left_lane_confidence, geometry.right_lane_confidence]
        horizontal_score = horizontal_marking_score(image)
        known_unmarked = item["clip_id"] == "target_train" and 252 <= float(item["timestamp"]) < 396
        if known_unmarked:
            status, origin = "invalid", "AUTO_HIGH_CONFIDENCE"
            left = right = {"status": "not_visible", "points": []}
            flag = "NO_VALID_EGO_LANE"
            rationale = "audited contiguous unmarked-road segment"
        elif geometry.ego_lane_valid and min(confidences) >= .42 and abs(geometry.normalized_offset or 0) <= 1.1:
            status, origin, flag = "valid", "AUTO_HIGH_CONFIDENCE", None
            rationale = "two supported lane-mask fits with plausible width and camera offset"
        elif visible == 1 and max(confidences) >= .55 and horizontal_score < .45:
            status, origin, flag = "valid", "AUTO_HIGH_CONFIDENCE", None
            rationale = "one supported lane-mask fit; other boundary not invented"
        elif horizontal_score >= .45:
            status, origin, flag = "invalid", "AUTO_HIGH_CONFIDENCE", "NO_VALID_EGO_LANE"
            left = right = {"status": "not_visible", "points": []}
            rationale = "dominant horizontal intersection/crosswalk markings"
        else:
            status, origin, flag = "ambiguous", "AUTO_LOW_CONFIDENCE", None
            left = left if left["status"] == "visible" else {"status": "ambiguous", "points": []}
            right = right if right["status"] == "visible" else {"status": "ambiguous", "points": []}
            rationale = "insufficient safe geometry; review required"
        item.update(
            ego_lane_status=status,
            annotation_flag=flag,
            left_boundary=left,
            right_boundary=right,
            scene_type=scene_tags(item, image, geometry, horizontal_score),
            label_origin=origin,
            review_status="pending",
            notes=(f"TwinLiteNet lane-mask proposal: {rationale}; "
                   f"L={geometry.left_lane_confidence:.3f} R={geometry.right_lane_confidence:.3f} "
                   f"horizontal={horizontal_score:.3f}"),
        )
        key = f"{item['split']}|{status}|{origin}"
        counts[key] = counts.get(key, 0) + 1
        if number % 100 == 0:
            print(f"proposed {number}/{len(payload['annotations'])}", flush=True)
    payload["proposal_generator"] = {
        "model": str(args.model),
        "method": "TwinLiteNet lane mask + conservative geometry; proposals are not ground truth",
    }
    args.dataset.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"counts": counts}, indent=2))


if __name__ == "__main__":
    main()
