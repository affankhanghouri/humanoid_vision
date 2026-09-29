#!/usr/bin/env python3
"""Audit cached target-camera samples and build compact chronological contact sheets.

This intentionally operates on the train/validation cache produced by build_dataset.py.
It never accepts a final-holdout source.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np


def image_stats(image: np.ndarray) -> tuple[float, float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-sheet", type=int, default=40)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    if any(item["split"] not in {"train", "validation"} for item in payload["annotations"]):
        raise SystemExit("REFUSED: audit cache must contain train/validation only")
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"method": "chronological cached samples from full sequential decode"}
    for split in ("train", "validation"):
        items = [item for item in payload["annotations"] if item["split"] == split]
        means, contrasts, sharpness = [], [], []
        for page, start in enumerate(range(0, len(items), args.per_sheet)):
            tiles = []
            for item in items[start : start + args.per_sheet]:
                image = cv2.imread(str(args.dataset.parent / item["image"]))
                if image is None:
                    raise FileNotFoundError(item["image"])
                mean, contrast, sharp = image_stats(image)
                means.append(mean)
                contrasts.append(contrast)
                sharpness.append(sharp)
                tile = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (320, 25), (0, 0, 0), -1)
                cv2.putText(tile, f"#{item['item_id']} {item['timestamp']:.1f}s", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            while len(tiles) < args.per_sheet:
                tiles.append(np.zeros((180, 320, 3), np.uint8))
            columns = 5
            sheet = np.vstack([np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)])
            cv2.imwrite(str(args.output / f"{split}_audit_{page:02d}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 84])
        source = next(source for source in payload["sources"] if source["split"] == split)
        report[split] = {
            "source": source,
            "cached_samples": len(items),
            "first_timestamp_s": items[0]["timestamp"] if items else None,
            "last_timestamp_s": items[-1]["timestamp"] if items else None,
            "luminance": {"min": min(means), "median": float(np.median(means)), "max": max(means)},
            "contrast": {"min": min(contrasts), "median": float(np.median(contrasts)), "max": max(contrasts)},
            "laplacian_variance": {"min": min(sharpness), "median": float(np.median(sharpness)), "max": max(sharpness)},
            "contact_sheets": (len(items) + args.per_sheet - 1) // args.per_sheet,
        }
    (args.output / "audit_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
