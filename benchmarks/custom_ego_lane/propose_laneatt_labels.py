#!/usr/bin/env python3
"""Use the installed LaneATT model only as a target-label proposal generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/lane_study"))
import benchmark_laneatt as laneatt  # noqa: E402


def sample_boundary(points: np.ndarray, width: int, height: int) -> dict:
    supported = points[(points[:, 1] >= .35) & (points[:, 1] <= .96)]
    if len(supported) < 3 or float(np.ptp(supported[:, 1])) < .16:
        return {"status": "not_visible", "points": []}
    indices = np.unique(np.linspace(0, len(supported)-1, min(12, len(supported)), dtype=int))
    selected = supported[indices]
    return {"status": "visible", "points": [[round(float(x*width),2), round(float(y*height),2)] for x,y in selected]}


def x_at(points: np.ndarray, y: float = .90) -> float | None:
    order = np.argsort(points[:,1])
    pts = points[order]
    if y < pts[0,1] or y > pts[-1,1]:
        return None
    return float(np.interp(y, pts[:,1], pts[:,0]))


def audited_negative(item: dict) -> bool:
    t = float(item["timestamp"])
    if item["clip_id"] == "target_train":
        return (252 <= t < 396) or (92 <= t < 119) or (584 <= t < 611) or (684 <= t < 704)
    return (0 <= t < 9.5) or (28 <= t < 42) or (43.5 <= t < 48) or (60 <= t < 66)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT/"models/laneatt_r18_culane.onnx")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    runner = laneatt.make_ort_runner(args.model, args.threads)
    counts: dict[str,int] = {}
    for number,item in enumerate(payload["annotations"],1):
        if item.get("label_origin") != "UNREVIEWED":
            continue
        image = cv2.imread(str(args.dataset.parent/item["image"]))
        h,w = image.shape[:2]
        lanes = laneatt.postprocess(runner(laneatt.preprocess(image)))
        located = [(x_at(points),points) for points in lanes]
        located = [(x,points) for x,points in located if x is not None]
        lefts = [(x,p) for x,p in located if x < .5]
        rights = [(x,p) for x,p in located if x > .5]
        left_points = max(lefts,key=lambda pair:pair[0])[1] if lefts else None
        right_points = min(rights,key=lambda pair:pair[0])[1] if rights else None
        left = sample_boundary(left_points,w,h) if left_points is not None else {"status":"not_visible","points":[]}
        right = sample_boundary(right_points,w,h) if right_points is not None else {"status":"not_visible","points":[]}
        visible = sum(side["status"]=="visible" for side in (left,right))
        if audited_negative(item):
            status,origin,flag="invalid","AUTO_HIGH_CONFIDENCE","NO_VALID_EGO_LANE"
            left=right={"status":"not_visible","points":[]}
        elif visible == 2:
            lx=x_at(left_points); rx=x_at(right_points); width=(rx-lx) if lx is not None and rx is not None else 0
            if .14 <= width <= .78:
                status,origin,flag="valid","AUTO_HIGH_CONFIDENCE",None
            else:
                status,origin,flag="ambiguous","AUTO_LOW_CONFIDENCE",None
                left=right={"status":"ambiguous","points":[]}
        elif visible == 1:
            status,origin,flag="valid","AUTO_HIGH_CONFIDENCE",None
        else:
            status,origin,flag="ambiguous","AUTO_LOW_CONFIDENCE",None
            left=right={"status":"ambiguous","points":[]}
        item.update(ego_lane_status=status,annotation_flag=flag,left_boundary=left,right_boundary=right,
                    label_origin=origin,review_status="pending",notes="LaneATT proposal only; nearest supported boundaries; requires overlay review")
        item["proposal_origin"]="LANEATT_AUTO_PROPOSAL"
        key=f"{item['split']}|{status}|{origin}";counts[key]=counts.get(key,0)+1
        if number%50==0: print(f"proposed {number}/{len(payload['annotations'])}",flush=True)
    payload["proposal_generator"]={"model":str(args.model),"method":"LaneATT proposal only; nearest supported markings; audited negative windows; human review required"}
    args.dataset.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({"counts":counts},indent=2))


if __name__=="__main__":
    main()
