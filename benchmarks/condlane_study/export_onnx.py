#!/usr/bin/env python3
"""Export the official CondLaneNet CULane-small weights to a fixed ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from model import load_model, mlp_weights, preprocess  # noqa: E402


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/condlanenet_culane_small.pth")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models/condlanenet_culane_small.onnx")
    parser.add_argument("--post", type=Path, default=ROOT / "models/condlanenet_culane_small_post.npz")
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = load_model(args.checkpoint)
    sample = torch.zeros((1, 3, 320, 800), dtype=torch.float32)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        args.onnx,
        input_names=["images"],
        output_names=["heatmap_logits", "dynamic_parameters", "mask_features"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.onnx))
    np.savez(args.post, **dict(zip(("w0", "b0", "w1", "b1"), mlp_weights(model))))

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    dataset = json.loads((ROOT / "benchmarks/lane_acceptance/dataset.json").read_text())
    ids = [dataset["annotations"][index]["frame_id"] for index in (0, 49, 99)]
    comparisons = []
    for frame_id in ids:
        image = cv2.imread(str(ROOT / f"benchmarks/lane_acceptance/frames/frame_{frame_id:04d}.jpg"))
        array, _ = preprocess(image)
        with torch.inference_mode():
            expected = [item.numpy() for item in model(torch.from_numpy(array))]
        actual = session.run(None, {"images": array})
        differences = [np.abs(left - right) for left, right in zip(expected, actual)]
        comparisons.append({
            "frame_id": frame_id,
            "max_abs_error": max(float(item.max()) for item in differences),
            "mean_abs_error": float(np.mean([item.mean() for item in differences])),
        })
    report = {
        "checkpoint_sha256": digest(args.checkpoint),
        "onnx_sha256": digest(args.onnx),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "onnx_bytes": args.onnx.stat().st_size,
        "onnx_checker": "passed",
        "opset": 17,
        "comparisons": comparisons,
    }
    report_path = HERE / "output/export_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
