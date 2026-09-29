#!/usr/bin/env python3
"""Build a compact truth-versus-prediction sheet from sealed acceptance items."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "acceptance_v1/dataset.json").read_text())
REPLAY = ROOT / "output/replay_v1/diagnostics"
SELECTED = [12, 17, 19, 1, 2, 24, 36, 39, 44, 23, 31, 57]


def main():
    by_id = {item["item_id"]: item for item in DATA["annotations"]}
    tiles = []
    for item_id in SELECTED:
        item = by_id[item_id]
        image = cv2.imread(str(REPLAY / f"frame_{item['frame_id']:05d}.jpg"))
        if image is None:
            raise FileNotFoundError(item["frame_id"])
        color = (60, 220, 60) if item["expected_state"] == item["proposal_state"] else (30, 40, 235)
        cv2.rectangle(image, (0, 0), (image.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(image, f"ITEM {item_id:02d}  #{item['track_id']} {item['class_name'].upper()}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"GT={item['expected_state']}  PRED={item['proposal_state']}",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, .54, color, 2, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA))
    sheet = np.vstack([np.hstack(tiles[index:index + 3]) for index in range(0, len(tiles), 3)])
    cv2.imwrite(str(ROOT / "acceptance_v1/visual_examples.jpg"), sheet,
                [cv2.IMWRITE_JPEG_QUALITY, 91])


if __name__ == "__main__":
    main()
