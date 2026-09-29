#!/usr/bin/env python3
"""Build reporting-only scene summaries and visual sheets from saved predictions."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent / "target_camera_v1"


def scene_breakdown(manifest_path: Path, metrics_path: Path, output_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    anns = {a["item_id"]: a for a in manifest["annotations"]}
    rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "frames": 0,
            "valid": 0,
            "invalid_or_ambiguous": 0,
            "detected_valid": 0,
            "missed_valid": 0,
            "false_lane": 0,
            "correct_abstain": 0,
        }
    )
    for pred in metrics["predictions"]:
        ann = anns[pred["item_id"]]
        truth_valid = ann["ego_lane_status"] == "valid"
        predicted_valid = pred["predicted"] == "valid"
        tags = ann.get("scene_type") or ["untagged"]
        for tag in tags:
            row = rows[tag]
            row["frames"] += 1
            row["valid" if truth_valid else "invalid_or_ambiguous"] += 1
            if truth_valid and predicted_valid:
                row["detected_valid"] += 1
            elif truth_valid:
                row["missed_valid"] += 1
            elif predicted_valid:
                row["false_lane"] += 1
            else:
                row["correct_abstain"] += 1

    result = {}
    for tag, row in sorted(rows.items()):
        valid = row["valid"]
        negative = row["invalid_or_ambiguous"]
        result[tag] = {
            **row,
            "valid_lane_detection_rate": row["detected_valid"] / valid if valid else None,
            "false_lane_rate": row["false_lane"] / negative if negative else None,
            "correct_abstain_rate": row["correct_abstain"] / negative if negative else None,
        }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def evenly(items: list, count: int) -> list:
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[i] for i in indices]


def draw_polyline(image, boundary, color) -> None:
    points = boundary.get("points", [])
    if len(points) >= 2:
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], False, color, 4, cv2.LINE_AA)


def make_sheet(name: str, predictions: list[dict], anns: dict[int, dict], count: int = 12) -> None:
    selected = evenly(predictions, count)
    tiles = []
    for pred in selected:
        ann = anns[pred["item_id"]]
        image = cv2.imread(str(ROOT / "final_holdout" / ann["image"]))
        if image is None:
            raise FileNotFoundError(ann["image"])
        draw_polyline(image, ann["left_boundary"], (0, 255, 0))
        draw_polyline(image, ann["right_boundary"], (0, 255, 255))
        cv2.putText(
            image,
            f"t={ann['timestamp']:.2f}s truth={ann['ego_lane_status']} pred={pred['predicted']} p={pred['valid_probability']:.3f}",
            (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 255), 2, cv2.LINE_AA,
        )
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        tiles.append(image)
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    out = ROOT / "final_holdout" / "frozen_eval" / name
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    validation = scene_breakdown(
        ROOT / "dataset.json",
        ROOT / "validation_final" / "validation_metrics.json",
        ROOT / "validation_final" / "scene_breakdown.json",
    )
    holdout = scene_breakdown(
        ROOT / "final_holdout" / "dataset.json",
        ROOT / "final_holdout" / "frozen_eval" / "validation_metrics.json",
        ROOT / "final_holdout" / "frozen_eval" / "scene_breakdown.json",
    )

    manifest = json.loads((ROOT / "final_holdout" / "dataset.json").read_text())
    metrics = json.loads((ROOT / "final_holdout" / "frozen_eval" / "validation_metrics.json").read_text())
    anns = {a["item_id"]: a for a in manifest["annotations"]}
    misses = [p for p in metrics["predictions"] if p["truth"] == "valid" and p["predicted"] != "valid"]
    abstains = [p for p in metrics["predictions"] if p["truth"] != "valid" and p["predicted"] != "valid"]
    make_sheet("final_missed_valid_examples.jpg", misses, anns)
    make_sheet("final_correct_abstention_examples.jpg", abstains, anns)
    print("validation scenes", len(validation))
    print("holdout scenes", len(holdout))
    print("misses", len(misses), "correct abstentions", len(abstains))


if __name__ == "__main__":
    main()
