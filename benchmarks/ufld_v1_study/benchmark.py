#!/usr/bin/env python3
"""Bounded CPU benchmark for the UFLD v1 ResNet-18 CULane model."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models/ufld_v1_culane_r18.onnx"
DEFAULT_DATASET = ROOT / "benchmarks/lane_acceptance/dataset.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/ufld_v1_study/timing.json"
MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


def preprocess(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(image, (800, 288)), cv2.COLOR_BGR2RGB)
    normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def postprocess(logits: np.ndarray) -> list[list[tuple[float, float]]]:
    """Decode the four UFLD lane slots at native 800x288 coordinates."""
    scores = np.asarray(logits).reshape(201, 18, 4)[:, ::-1, :]
    lane_logits = scores[:-1]
    lane_logits -= lane_logits.max(axis=0, keepdims=True)
    probability = np.exp(lane_logits)
    probability /= probability.sum(axis=0, keepdims=True)
    locations = (probability * np.arange(1, 201, dtype=np.float32)[:, None, None]).sum(axis=0)
    locations[scores.argmax(axis=0) == 200] = 0
    anchors = np.asarray(
        [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287],
        dtype=np.float32,
    )[::-1]
    scale = 799.0 / 199.0
    lanes: list[list[tuple[float, float]]] = []
    for lane_index in range(4):
        points = [
            (float((locations[row, lane_index] - 1.0) * scale), float(anchors[row]))
            for row in range(18)
            if locations[row, lane_index] > 0
        ]
        lanes.append(points if len(points) > 2 else [])
    return lanes


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
    }


def load_images(dataset_path: Path, count: int) -> list[np.ndarray]:
    payload = json.loads(dataset_path.read_text())
    annotations = payload["annotations"]
    images = []
    for record in annotations[:count]:
        image = cv2.imread(str(dataset_path.parent / record["image"]))
        if image is None:
            raise FileNotFoundError(record["image"])
        images.append(image)
    if len(images) < count:
        raise ValueError(f"dataset contains only {len(images)} images; requested {count}")
    return images


def make_ort(model: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return lambda tensor: session.run(None, {input_name: tensor})[0]


def make_openvino(model: Path, threads: int):
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(
        str(model),
        "CPU",
        {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": "1"},
    )
    request = compiled.create_infer_request()
    input_port, output_port = compiled.input(0), compiled.output(0)

    def infer(tensor: np.ndarray) -> np.ndarray:
        request.infer({input_port: tensor})
        return np.array(request.get_tensor(output_port).data, copy=True)

    return infer


def run_backend(name: str, factory, model: Path, images: list[np.ndarray], warmup: int) -> dict:
    infer = factory(model)
    tensors = [preprocess(image) for image in images]
    for index in range(warmup):
        infer(tensors[index % len(tensors)])
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    total_ms: list[float] = []
    for image in images:
        start = time.perf_counter_ns()
        tensor = preprocess(image)
        after_pre = time.perf_counter_ns()
        output = infer(tensor)
        after_infer = time.perf_counter_ns()
        postprocess(output)
        end = time.perf_counter_ns()
        preprocess_ms.append((after_pre - start) / 1e6)
        inference_ms.append((after_infer - after_pre) / 1e6)
        postprocess_ms.append((end - after_infer) / 1e6)
        total_ms.append((end - start) / 1e6)
    result = {
        "backend": name,
        "samples": len(images),
        "warmup": warmup,
        "preprocess": summary(preprocess_ms),
        "inference": summary(inference_ms),
        "postprocess": summary(postprocess_ms),
        "total": summary(total_ms),
    }
    print(f"{name}: total mean={result['total']['mean_ms']:.2f} ms p95={result['total']['p95_ms']:.2f} ms")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    images = load_images(args.dataset, args.frames)
    results = []
    for threads in (1, 2):
        results.append(run_backend(f"onnxruntime_fp32_{threads}t", lambda p, t=threads: make_ort(p, t), args.model, images, args.warmup))
    for threads in (1, 2):
        results.append(run_backend(f"openvino_fp32_{threads}t", lambda p, t=threads: make_openvino(p, t), args.model, images, args.warmup))
    payload = {"model": str(args.model), "input": [1, 3, 288, 800], "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
