#!/usr/bin/env python3
"""Benchmark CondLaneNet and evaluate its conservative ego-lane predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
from openvino import Core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks/lane_acceptance"))
from model import decode_lanes, finish_decode, preprocess  # noqa: E402
from common import _polyline_error, evaluate_predictions, report_markdown  # noqa: E402


def summary(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(data.mean()),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


class ORTRunner:
    def __init__(self, path: Path, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        return tuple(self.session.run(None, {"images": image}))


class OpenVINORunner:
    def __init__(self, path: Path, threads: int):
        core = Core()
        model = core.read_model(path)
        self.compiled = core.compile_model(model, "CPU", {
            "INFERENCE_NUM_THREADS": threads,
            "NUM_STREAMS": "1",
            "PERFORMANCE_HINT": "LATENCY",
        })
        self.output_ports = list(self.compiled.outputs)

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        result = self.compiled({"images": image})
        return tuple(np.asarray(result[port]) for port in self.output_ports)


def interpolate_x(points: list[list[float]], y: float) -> float | None:
    array = np.asarray(points, dtype=np.float32)
    order = np.argsort(array[:, 1])
    array = array[order]
    unique_y = np.unique(array[:, 1])
    if len(unique_y) < 2 or y < unique_y[0] or y > unique_y[-1]:
        return None
    xs = np.asarray([array[array[:, 1] == value, 0].mean() for value in unique_y])
    return float(np.interp(y, unique_y, xs))


def lane_candidate(lane: dict[str, object], width: int, height: int) -> dict[str, object] | None:
    points = [[float(x), float(y)] for x, y in lane["points"] if 0 <= x < width and 0 <= y < height]
    if len(points) < 8:
        return None
    points.sort(key=lambda point: point[1])
    if points[-1][1] - points[0][1] < 0.18 * height:
        return None
    reference_y = 0.84 * height
    x_reference = interpolate_x(points, reference_y)
    if x_reference is None:
        return None
    return {"score": lane["score"], "points": points, "x_reference": x_reference}


def adapt_ego(lanes: list[dict[str, object]], width: int, height: int) -> dict[str, object]:
    candidates = [item for lane in lanes if (item := lane_candidate(lane, width, height))]
    center = width / 2
    lefts = sorted((item for item in candidates if item["x_reference"] < center), key=lambda item: center - item["x_reference"])
    rights = sorted((item for item in candidates if item["x_reference"] > center), key=lambda item: item["x_reference"] - center)
    left = lefts[0] if lefts else None
    right = rights[0] if rights else None
    valid_pair = False
    if left and right:
        lane_width = right["x_reference"] - left["x_reference"]
        lane_center = (right["x_reference"] + left["x_reference"]) / 2
        low = max(left["points"][0][1], right["points"][0][1])
        high = min(left["points"][-1][1], right["points"][-1][1])
        if high - low >= 0.12 * height:
            ys = np.linspace(low, high, 12)
            widths = np.asarray([
                interpolate_x(right["points"], y) - interpolate_x(left["points"], y)
                for y in ys
            ])
            valid_pair = bool(
                0.18 * width <= lane_width <= 0.75 * width
                and abs(lane_center - center) <= 0.18 * width
                and np.all(widths > 0.025 * width)
                and widths[0] <= widths[-1] * 1.35
            )

    def boundary(item: dict[str, object] | None) -> dict[str, object]:
        if item is None:
            return {"status": "not_visible", "points": []}
        points = item["points"][::2]
        if points[-1] != item["points"][-1]:
            points.append(item["points"][-1])
        return {"status": "visible", "points": [[round(x, 3), round(y, 3)] for x, y in points]}

    if valid_pair:
        status = "valid"
    elif left or right:
        status = "ambiguous"
    else:
        status = "invalid"
    return {
        "ego_lane_status": status,
        "left_boundary": boundary(left),
        "right_boundary": boundary(right),
    }


def run_backend(name: str, runner, annotations: list[dict], post_weights: tuple[np.ndarray, ...],
                make_predictions: bool) -> tuple[dict, list[dict]]:
    first_image = cv2.imread(str(ROOT / "benchmarks/lane_acceptance" / annotations[0]["image"]))
    warmup, _ = preprocess(first_image)
    for _ in range(3):
        runner(warmup)
    timings = {key: [] for key in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")}
    predictions = []
    for annotation in annotations:
        started = time.perf_counter_ns()
        image = cv2.imread(str(ROOT / "benchmarks/lane_acceptance" / annotation["image"]))
        array, transform = preprocess(image)
        after_preprocess = time.perf_counter_ns()
        output = runner(array)
        after_inference = time.perf_counter_ns()
        raw = decode_lanes(output, transform, threshold=0.5)
        lanes = finish_decode(raw, post_weights, transform)
        adapted = adapt_ego(lanes, transform.original_width, transform.original_height)
        ended = time.perf_counter_ns()
        timings["preprocess_ms"].append((after_preprocess - started) / 1e6)
        timings["inference_ms"].append((after_inference - after_preprocess) / 1e6)
        timings["postprocess_ms"].append((ended - after_inference) / 1e6)
        timings["total_ms"].append((ended - started) / 1e6)
        if make_predictions:
            predictions.append({"frame_id": annotation["frame_id"], **adapted})
    return {"backend": name, "frames": len(annotations), **{key: summary(value) for key, value in timings.items()}}, predictions


def draw_polyline(image: np.ndarray, boundary: dict, color: tuple[int, int, int], thickness: int) -> None:
    if boundary["status"] != "visible" or len(boundary["points"]) < 2:
        return
    points = np.asarray(boundary["points"], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], False, color, thickness, cv2.LINE_AA)


def failure_ids(dataset: dict, predictions: list[dict]) -> list[tuple[int, str]]:
    by_id = {item["frame_id"]: item for item in predictions}
    failures: list[tuple[float, int, str]] = []
    for truth in dataset["annotations"]:
        prediction = by_id[truth["frame_id"]]
        truth_valid = truth["ego_lane_status"] == "valid"
        predicted_valid = prediction["ego_lane_status"] == "valid"
        if predicted_valid and not truth_valid:
            failures.append((10000, truth["frame_id"], "FALSE EGO LANE"))
            continue
        if truth_valid and not predicted_valid:
            failures.append((5000, truth["frame_id"], "MISSED VALID LANE"))
            continue
        errors = []
        for side in ("left", "right"):
            gt = truth[f"{side}_boundary"]
            pred = prediction[f"{side}_boundary"]
            if gt["status"] == pred["status"] == "visible":
                value, _ = _polyline_error(gt["points"], pred["points"], dataset["video"]["width"])
                errors.append(float(value.mean()))
        if errors:
            failures.append((max(errors), truth["frame_id"], f"BOUNDARY ERROR {max(errors):.0f}px"))
    failures.sort(reverse=True)
    selected = []
    categories: dict[str, int] = {}
    for _, frame_id, reason in failures:
        category = reason.split()[0]
        if categories.get(category, 0) >= 4:
            continue
        selected.append((frame_id, reason))
        categories[category] = categories.get(category, 0) + 1
        if len(selected) == 12:
            break
    return selected


def save_failure_visuals(dataset: dict, predictions: list[dict], output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    gt_by_id = {item["frame_id"]: item for item in dataset["annotations"]}
    pred_by_id = {item["frame_id"]: item for item in predictions}
    records, tiles = [], []
    for frame_id, reason in failure_ids(dataset, predictions):
        truth, prediction = gt_by_id[frame_id], pred_by_id[frame_id]
        image = cv2.imread(str(ROOT / "benchmarks/lane_acceptance" / truth["image"]))
        for side in ("left", "right"):
            draw_polyline(image, truth[f"{side}_boundary"], (0, 220, 0), 5)
            draw_polyline(image, prediction[f"{side}_boundary"], (0, 0, 255), 3)
        label = f"frame {frame_id} | {reason} | GT {truth['ego_lane_status']} PRED {prediction['ego_lane_status']}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(image, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        path = output / f"frame_{frame_id:04d}.jpg"
        cv2.imwrite(str(path), image)
        tile = cv2.resize(image, (576, 360), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
        records.append({"frame_id": frame_id, "reason": reason, "file": str(path.relative_to(ROOT))})
    if tiles:
        rows = []
        for index in range(0, len(tiles), 3):
            row = tiles[index:index + 3]
            while len(row) < 3:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        cv2.imwrite(str(output / "failure_contact_sheet.jpg"), np.vstack(rows))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=ROOT / "models/condlanenet_culane_small.onnx")
    parser.add_argument("--post", type=Path, default=ROOT / "models/condlanenet_culane_small_post.npz")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    dataset_path = ROOT / "benchmarks/lane_acceptance/dataset.json"
    dataset = json.loads(dataset_path.read_text())
    annotations = dataset["annotations"]
    saved = np.load(args.post)
    post_weights = tuple(saved[key] for key in ("w0", "b0", "w1", "b1"))

    configs = [
        ("onnxruntime_fp32_1_thread", ORTRunner(args.onnx, 1)),
        ("onnxruntime_fp32_2_threads", ORTRunner(args.onnx, 2)),
        ("openvino_fp32_1_thread", OpenVINORunner(args.onnx, 1)),
        ("openvino_fp32_2_threads", OpenVINORunner(args.onnx, 2)),
    ]
    timing, predictions = [], []
    for index, (name, runner) in enumerate(configs):
        print(f"Running {name} ...", flush=True)
        result, produced = run_backend(name, runner, annotations, post_weights, index == len(configs) - 1)
        timing.append(result)
        if produced:
            predictions = produced
        print(json.dumps(result, indent=2), flush=True)

    prediction_document = {
        "schema_version": 1,
        "model": {
            "name": "condlanenet_culane_small_conservative",
            "version": "official-culane-small-threshold-0.5",
            "backend": "openvino_fp32_2_threads",
            "adapter": "two-sided center-bracketing; questionable/partial lanes abstain",
        },
        "predictions": predictions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.json"
    prediction_path.write_text(json.dumps(prediction_document, indent=2) + "\n")
    acceptance = evaluate_predictions(dataset, prediction_document)
    (args.output / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    (args.output / "acceptance.md").write_text(report_markdown(acceptance))
    failures = save_failure_visuals(dataset, predictions, args.output / "visuals")
    report = {
        "model": prediction_document["model"],
        "machine": {"processor": platform.processor(), "platform": platform.platform()},
        "frames": len(annotations),
        "warmups_per_backend": 3,
        "timing_ms": timing,
        "acceptance_metrics": acceptance["metrics"],
        "acceptance_counts": acceptance["counts"],
        "scene_false_positive_rates": acceptance["scene_false_positive_rates"],
        "failure_visuals": failures,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
