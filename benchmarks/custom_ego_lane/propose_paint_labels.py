#!/usr/bin/env python3
"""Paint-supported ego-boundary proposals for human overlay review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def paint_mask(image: np.ndarray) -> np.ndarray:
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, lightness, saturation = cv2.split(hls)
    white = ((lightness >= 155) & (saturation <= 145)).astype(np.uint8) * 255
    yellow = cv2.inRange(hsv, np.array((10, 55, 65)), np.array((42, 255, 255)))
    mask = cv2.bitwise_or(white, yellow)
    roi = np.zeros_like(mask)
    h, w = mask.shape
    cv2.fillPoly(roi, [np.array([(0, h - 1), (int(.27*w), int(.43*h)),
                                (int(.73*w), int(.43*h)), (w - 1, h - 1)])], 255)
    return cv2.bitwise_and(mask, roi)


def candidates(image: np.ndarray) -> list[dict]:
    h, w = image.shape[:2]
    mask = paint_mask(image)
    edges = cv2.Canny(mask, 45, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 16, minLineLength=16, maxLineGap=45)
    output = []
    if lines is None:
        return output
    y_eval = .92 * h
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(y2-y1) < 8:
            continue
        slope = (x2-x1)/(y2-y1)
        if abs(slope) < .12 or abs(slope) > 3.2:
            continue
        bottom = x1 + slope * (y_eval-y1)
        top = x1 + slope * (.45*h-y1)
        length = float(np.hypot(x2-x1, y2-y1))
        if not (-.08*w <= bottom <= 1.08*w and .12*w <= top <= .88*w):
            continue
        side = "left" if bottom < .5*w else "right"
        if side == "left" and slope >= 0:
            continue
        if side == "right" and slope <= 0:
            continue
        output.append({"side": side, "bottom": bottom, "points": ((x1,y1),(x2,y2)), "length": length})
    return output


def fit_side(lines: list[dict], side: str, width: int, height: int) -> dict:
    choices = [line for line in lines if line["side"] == side]
    if not choices:
        return {"status": "not_visible", "points": []}
    # Closest supported boundary to camera center; cluster by extrapolated bottom x.
    choices.sort(key=lambda line: abs(line["bottom"] - .5*width))
    seed = choices[0]["bottom"]
    cluster = [line for line in choices if abs(line["bottom"]-seed) <= .09*width]
    points = np.asarray([point for line in cluster for point in line["points"]], np.float32)
    if len(points) < 4 or np.ptp(points[:,1]) < .12*height or sum(line["length"] for line in cluster) < .16*height:
        return {"status": "not_visible", "points": []}
    coefficient = np.polyfit(points[:,1], points[:,0], 1)
    residual = np.abs(points[:,0] - np.polyval(coefficient, points[:,1]))
    keep = residual <= max(12, .018*width)
    if keep.sum() < 4:
        return {"status": "not_visible", "points": []}
    coefficient = np.polyfit(points[keep,1], points[keep,0], 1)
    low, high = max(.45*height, float(points[keep,1].min())), min(.95*height, float(points[keep,1].max()))
    if high-low < .12*height:
        return {"status": "not_visible", "points": []}
    ys = np.linspace(low, high, 10)
    xs = np.polyval(coefficient, ys)
    if np.any(xs < 0) or np.any(xs >= width):
        return {"status": "not_visible", "points": []}
    return {"status": "visible", "points": [[round(float(x),2),round(float(y),2)] for x,y in zip(xs,ys)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    counts = {}
    for item in payload["annotations"]:
        if item.get("clip_id") != "target_train_focus":
            continue
        image = cv2.imread(str(args.dataset.parent/item["image"]))
        h, w = image.shape[:2]
        lines = candidates(image)
        left, right = fit_side(lines,"left",w,h), fit_side(lines,"right",w,h)
        visible = sum(x["status"] == "visible" for x in (left,right))
        t = float(item["timestamp"])
        known_unmarked = item["clip_id"] == "target_train" and 252 <= t < 396
        if known_unmarked:
            status, origin, flag = "invalid", "AUTO_HIGH_CONFIDENCE", "NO_VALID_EGO_LANE"
            left = right = {"status":"not_visible","points":[]}
        elif visible:
            status, origin, flag = "valid", "AUTO_HIGH_CONFIDENCE", None
        else:
            status, origin, flag = "ambiguous", "AUTO_LOW_CONFIDENCE", None
            left = right = {"status":"ambiguous","points":[]}
        item.update(ego_lane_status=status, annotation_flag=flag, left_boundary=left,
                    right_boundary=right, label_origin=origin, review_status="pending",
                    notes="white/yellow paint-mask Hough proposal; requires overlay review")
        key=f"{item['split']}|{status}|{origin}"; counts[key]=counts.get(key,0)+1
    payload["proposal_generator"]={"method":"white/yellow paint mask + closest-boundary Hough fit; proposals are not ground truth"}
    args.dataset.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"counts":counts},indent=2))


if __name__ == "__main__":
    main()
