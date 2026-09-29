#!/usr/bin/env python3
"""Export and benchmark LaneATT ResNet-18/CULane without production integration.

The official CUDA NMS is deliberately kept outside the ONNX graph.  The graph
emits the 1,000 structured lane proposals and this script applies the same
average horizontal-distance suppression rule on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/tmp/LaneATT-main")
DEFAULT_CHECKPOINT = ROOT / "models/laneatt_r18_culane.pt"
DEFAULT_ONNX = ROOT / "models/laneatt_r18_culane.onnx"
DEFAULT_VIDEO = ROOT / "test_data/test_video.mp4"
DEFAULT_OUTPUT = ROOT / "benchmarks/lane_study/output"
IMAGE_HEIGHT = 360
IMAGE_WIDTH = 640
N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 50.0
MAX_LANES = 4


@dataclass(frozen=True)
class StageSample:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    lane_count: int


def percentile_summary(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Match official CULane evaluation: resize BGR, scale to [0, 1], CHW."""
    resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _install_nms_import_stub() -> None:
    # LaneATT imports its CUDA extension at module import time. The exported
    # neural core ends before NMS, so a stub is sufficient and cannot be called.
    module = types.ModuleType("nms")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("CUDA NMS is unavailable in the CPU lane study")

    module.nms = unavailable
    sys.modules["nms"] = module


def load_official_model(source: Path, checkpoint: Path):
    import torch

    source = source.resolve()
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "lib"))
    _install_nms_import_stub()
    from lib.models.laneatt import LaneATT

    # The official constructor loads the tiny anchor-frequency tensor itself.
    # Force restricted tensor-only loading for that external artifact.
    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    torch.load = safe_load
    try:
        model = LaneATT(
            backbone="resnet18",
            pretrained_backbone=False,
            S=N_OFFSETS,
            img_w=IMAGE_WIDTH,
            img_h=IMAGE_HEIGHT,
            anchors_freq_path=str(source / "data/culane_anchors_freq.pt"),
            topk_anchors=1000,
        )
    finally:
        torch.load = original_load

    state = original_load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_raw_model(model):
    import torch
    import torch.nn as nn

    class RawLaneATT(nn.Module):
        def __init__(self, lane_model):
            super().__init__()
            self.model = lane_model
            count = len(lane_model.anchors)
            mask = ~torch.eye(count, dtype=torch.bool)
            self.register_buffer("attention_mask", mask)

        def forward(self, image):
            lane_model = self.model
            features = lane_model.conv1(lane_model.feature_extractor(image))
            anchor_features = lane_model.cut_anchor_features(features)
            batch = image.shape[0]
            count = len(lane_model.anchors)
            flat = anchor_features.view(-1, lane_model.anchor_feat_channels * lane_model.fmap_h)
            attention = torch.softmax(lane_model.attention_layer(flat), dim=1)
            attention = attention.reshape(batch, count, count - 1)
            matrix = torch.zeros(
                (batch, count, count), dtype=attention.dtype, device=attention.device
            )
            matrix = matrix.masked_scatter(
                self.attention_mask.unsqueeze(0).expand(batch, -1, -1), attention
            )
            base = flat.reshape(batch, count, -1)
            context = torch.bmm(base.transpose(1, 2), matrix.transpose(1, 2)).transpose(1, 2)
            joined = torch.cat((context, base), dim=2).reshape(batch * count, -1)
            logits = lane_model.cls_layer(joined).reshape(batch, count, 2)
            regression = lane_model.reg_layer(joined).reshape(batch, count, N_OFFSETS + 1)
            anchors = lane_model.anchors.unsqueeze(0).expand(batch, -1, -1)
            return torch.cat(
                (
                    logits,
                    anchors[:, :, 2:4],
                    anchors[:, :, 4:] + regression,
                ),
                dim=2,
            )

    return RawLaneATT(model).eval()


def export_onnx(raw_model, output: Path) -> None:
    import onnx
    import torch

    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            raw_model,
            example,
            str(output),
            input_names=["image"],
            output_names=["proposals"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(output)))


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values[:, 1] / np.sum(values, axis=1)


def lane_nms(
    proposals: np.ndarray,
    confidence: float = CONFIDENCE_THRESHOLD,
    threshold: float = NMS_THRESHOLD,
    top_k: int = MAX_LANES,
) -> np.ndarray:
    scores = _softmax_positive(proposals[:, :2])
    candidates = np.flatnonzero(scores > confidence)
    candidates = candidates[np.argsort(-scores[candidates], kind="stable")]
    keep: list[int] = []
    for index in candidates:
        current = proposals[index]
        start_a = int(current[2] * N_STRIPS + 0.5)
        length_a = current[4]
        end_a = int(start_a + length_a - 1 + 0.5 - ((length_a - 1) < 0))
        suppressed = False
        for kept_index in keep:
            other = proposals[kept_index]
            start_b = int(other[2] * N_STRIPS + 0.5)
            length_b = other[4]
            end_b = int(start_b + length_b - 1 + 0.5 - ((length_b - 1) < 0))
            start = max(start_a, start_b)
            end = min(end_a, end_b, N_OFFSETS - 1)
            if end >= start:
                distance = np.abs(
                    current[5 + start : 5 + end + 1]
                    - other[5 + start : 5 + end + 1]
                ).sum()
                if distance < threshold * (end - start + 1):
                    suppressed = True
                    break
        if not suppressed:
            keep.append(int(index))
            if len(keep) == top_k:
                break
    selected = proposals[keep].copy()
    if len(selected):
        selected[:, :2] = np.stack((1.0 - scores[keep], scores[keep]), axis=1)
        selected[:, 4] = np.round(selected[:, 4])
    return selected


def proposal_points(lane: np.ndarray) -> np.ndarray | None:
    lane_xs = lane[5:] / IMAGE_WIDTH
    start = int(round(float(lane[2]) * N_STRIPS))
    length = int(round(float(lane[4])))
    end = min(start + length - 1, N_OFFSETS - 1)
    if start > 0:
        valid_below = ((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))[::-1]
        invalid = ~(np.cumprod(valid_below.astype(np.uint8))[::-1].astype(bool))
        lane_xs[:start][invalid] = -2.0
    lane_xs[end + 1 :] = -2.0
    lane_ys = np.linspace(1.0, 0.0, N_OFFSETS, dtype=np.float32)
    valid = lane_xs >= 0.0
    xs = lane_xs[valid][::-1]
    ys = lane_ys[valid][::-1]
    if len(xs) <= 1:
        return None
    return np.column_stack((xs, ys))


def postprocess(proposals: np.ndarray) -> list[np.ndarray]:
    lanes = []
    for proposal in lane_nms(proposals[0]):
        points = proposal_points(proposal)
        if points is not None:
            lanes.append(points)
    return lanes


def video_frames(path: Path, count: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total - 3), count, dtype=np.int64)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {index} from {path}")
        frames.append((int(index), frame))
    capture.release()
    return frames


def verify_onnx(raw_model, onnx_path: Path, frames: list[tuple[int, np.ndarray]]) -> dict:
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    comparisons = []
    with torch.inference_mode():
        for frame_id, frame in frames[:3]:
            tensor = preprocess(frame)
            reference = raw_model(torch.from_numpy(tensor)).numpy()
            actual = session.run(None, {"image": tensor})[0]
            difference = np.abs(reference - actual)
            comparisons.append(
                {
                    "frame_id": frame_id,
                    "max_abs": float(difference.max()),
                    "mean_abs": float(difference.mean()),
                    "reference_max": float(np.abs(reference).max()),
                }
            )
    if max(item["max_abs"] for item in comparisons) > 2e-3:
        raise RuntimeError(f"ONNX numerical mismatch: {comparisons}")
    return {"frames": comparisons, "max_abs_limit": 2e-3}


def make_ort_runner(path: Path, threads: int) -> Callable[[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return lambda tensor: session.run(None, {"image": tensor})[0]


def make_openvino_runner(path: Path, threads: int) -> Callable[[np.ndarray], np.ndarray]:
    import openvino as ov

    core = ov.Core()
    model = core.read_model(str(path))
    compiled = core.compile_model(
        model,
        "CPU",
        {
            "INFERENCE_NUM_THREADS": threads,
            "NUM_STREAMS": "1",
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
        },
    )
    request = compiled.create_infer_request()
    output = compiled.output(0)
    return lambda tensor: np.asarray(request.infer({0: tensor})[output])


def benchmark_runner(
    runner: Callable[[np.ndarray], np.ndarray], frames: list[tuple[int, np.ndarray]]
) -> tuple[dict, list[list[np.ndarray]]]:
    warm = preprocess(frames[0][1])
    for _ in range(3):
        postprocess(runner(warm))
    samples: list[StageSample] = []
    outputs: list[list[np.ndarray]] = []
    for _, frame in frames:
        started = time.perf_counter_ns()
        tensor = preprocess(frame)
        preprocessed = time.perf_counter_ns()
        proposals = runner(tensor)
        inferred = time.perf_counter_ns()
        lanes = postprocess(proposals)
        finished = time.perf_counter_ns()
        samples.append(
            StageSample(
                preprocess_ms=(preprocessed - started) / 1e6,
                inference_ms=(inferred - preprocessed) / 1e6,
                postprocess_ms=(finished - inferred) / 1e6,
                total_ms=(finished - started) / 1e6,
                lane_count=len(lanes),
            )
        )
        outputs.append(lanes)
    return (
        {
            "samples": len(samples),
            "preprocess_ms": percentile_summary(x.preprocess_ms for x in samples),
            "inference_ms": percentile_summary(x.inference_ms for x in samples),
            "postprocess_ms": percentile_summary(x.postprocess_ms for x in samples),
            "total_ms": percentile_summary(x.total_ms for x in samples),
            "lane_count": percentile_summary(x.lane_count for x in samples),
        },
        outputs,
    )


def draw_lanes(frame: np.ndarray, lanes: list[np.ndarray]) -> np.ndarray:
    rendered = frame.copy()
    height, width = rendered.shape[:2]
    colors = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 255, 0)]
    for index, lane in enumerate(lanes):
        points = np.column_stack((lane[:, 0] * width, lane[:, 1] * height)).round().astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(rendered, [points], False, colors[index % len(colors)], 4, cv2.LINE_AA)
    return rendered


def save_visuals(
    frames: list[tuple[int, np.ndarray]], outputs: list[list[np.ndarray]], directory: Path
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    picks = np.linspace(0, len(frames) - 1, 12, dtype=np.int64)
    paths = []
    thumbs = []
    for ordinal, sample_index in enumerate(picks, 1):
        frame_id, frame = frames[int(sample_index)]
        rendered = draw_lanes(frame, outputs[int(sample_index)])
        path = directory / f"sample_{ordinal:02d}_frame_{frame_id:04d}.jpg"
        cv2.imwrite(str(path), rendered)
        paths.append(str(path.relative_to(ROOT)))
        thumb = cv2.resize(rendered, (480, 270))
        cv2.putText(
            thumb,
            f"frame {frame_id} | lanes {len(outputs[int(sample_index)])}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbs.append(thumb)
    rows = [np.hstack(thumbs[i : i + 3]) for i in range(0, 12, 3)]
    cv2.imwrite(str(directory / "contact_sheet.jpg"), np.vstack(rows))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=70)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames < 70:
        raise ValueError("The lane study requires at least 70 frames")
    model = load_official_model(args.source, args.checkpoint)
    raw_model = make_raw_model(model)
    if not args.skip_export:
        export_onnx(raw_model, args.onnx)
    frames = video_frames(args.video, args.frames)
    verification = verify_onnx(raw_model, args.onnx, frames)
    benchmarks = {}
    visual_outputs = None
    for backend in ("ort", "openvino"):
        for threads in (1, 2):
            runner = (
                make_ort_runner(args.onnx, threads)
                if backend == "ort"
                else make_openvino_runner(args.onnx, threads)
            )
            result, outputs = benchmark_runner(runner, frames)
            benchmarks[f"{backend}_{threads}_thread"] = result
            print(backend, threads, json.dumps(result), flush=True)
            if backend == "openvino" and threads == 2:
                visual_outputs = outputs
    assert visual_outputs is not None
    visuals = save_visuals(frames, visual_outputs, args.output / "visuals")
    result = {
        "candidate": "LaneATT ResNet-18 CULane",
        "input": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "color": "BGR", "scale": "1/255"},
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "onnx": str(args.onnx.relative_to(ROOT)),
        "onnx_verification": verification,
        "benchmarks": benchmarks,
        "visuals": visuals,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_json = args.output / "results.json"
    output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
