#!/usr/bin/env python3
"""Prepare a stratified, pending human-review set from saved replay diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
REPLAY = ROOT / "output/replay_v1"
OUTPUT = ROOT / "acceptance_v1"


def spaced(items, count):
    if len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in indices]


def main():
    rows = json.loads((REPLAY / "observations.json").read_text())
    supported = {"person", "bicycle", "motorcycle", "car", "truck", "bus"}
    rows = [row for row in rows if row["display_frame_id"] % 12 == 0
            and row["class_name"] in supported]
    event_keys = {
        (track_id, frame_id)
        for track_id, frames in {
            11: (192, 204, 216, 228, 240, 252),
            17: (216, 228, 240, 252, 264, 276, 288),
            38: (360, 384, 408, 432),
            142: (1752, 1764, 1776, 1812),
        }.items()
        for frame_id in frames
    }
    selected = [row for row in rows
                if (row["track_id"], row["display_frame_id"]) in event_keys]
    selected_keys = {(row["track_id"], row["display_frame_id"]) for row in selected}
    targets = {
        "OFF_DRIVABLE": 25,
        "ON_DRIVABLE_OUTSIDE_CORRIDOR": 20,
        "UNKNOWN": 15,
    }
    for state, count in targets.items():
        pool = [row for row in rows if row["state"] == state
                and (row["track_id"], row["display_frame_id"]) not in selected_keys]
        for row in spaced(pool, count):
            selected.append(row)
            selected_keys.add((row["track_id"], row["display_frame_id"]))
    selected = sorted(selected, key=lambda row: (row["display_timestamp"], row["track_id"]))
    if len(selected) > 80:
        # Preserve all event cases, then keep uniformly distributed remainder.
        events = [row for row in selected
                  if (row["track_id"], row["display_frame_id"]) in event_keys]
        general = [row for row in selected
                   if (row["track_id"], row["display_frame_id"]) not in event_keys]
        selected = sorted(events + spaced(general, 80 - len(events)),
                          key=lambda row: (row["display_timestamp"], row["track_id"]))

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sheets = OUTPUT / "review_sheets"
    sheets.mkdir(exist_ok=True)
    annotations = []
    tiles = []
    for index, row in enumerate(selected):
        item = {
            "item_id": index,
            "frame_id": row["display_frame_id"],
            "timestamp": row["display_timestamp"],
            "track_id": row["track_id"],
            "class_name": row["class_name"],
            "proposal_state": row["state"],
            "expected_state": None,
            "review_status": "PENDING",
            "review_notes": "",
            "tracking_source_frame_id": row["tracking_source_frame_id"],
            "tracking_source_timestamp": row["tracking_source_timestamp"],
            "road_source_frame_id": row["road_source_frame_id"],
            "road_source_timestamp": row["road_source_timestamp"],
        }
        annotations.append(item)
        path = REPLAY / "diagnostics" / f"frame_{row['display_frame_id']:05d}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        point = tuple(int(round(value)) for value in row["contact_point"])
        cv2.circle(image, point, 18, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.rectangle(image, (0, 0), (image.shape[1], 48), (0, 0, 0), -1)
        label = (f"ITEM {index:02d}  TARGET #{row['track_id']} {row['class_name'].upper()}  "
                 f"PROPOSAL={row['state']}  t={row['display_timestamp']:.2f}s")
        cv2.putText(image, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA))

    for start in range(0, len(tiles), 12):
        page = tiles[start:start + 12]
        while len(page) < 12:
            page.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(page[row:row + 3]) for row in range(0, 12, 3)])
        cv2.imwrite(str(sheets / f"review_{start // 12:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    manifest = {
        "schema_version": 1,
        "video": "test_data/test_video.mp4",
        "ground_truth_policy": "Human review only; proposal_state is not ground truth.",
        "annotations": annotations,
    }
    (OUTPUT / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("items", len(annotations))
    print("proposals", Counter(item["proposal_state"] for item in annotations))
    print("classes", Counter(item["class_name"] for item in annotations))


if __name__ == "__main__":
    main()
