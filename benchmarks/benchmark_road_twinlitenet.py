"""
Standalone CPU benchmark for TwinLiteNet / TwinLiteNet+ ONNX models.

Supports both output naming styles:

Old TwinLiteNet:
    da
    ll

TwinLiteNet+ export:
    drivable_area
    lane_line

The model input size is read directly from ONNX.
No changes are made to the live perception pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(
    frame: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Convert BGR OpenCV frame to:
        RGB
        float32
        NCHW
        range [0, 1]
    """

    tensor = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(width, height),
        swapRB=True,
        crop=False,
    )

    return tensor


# ============================================================
# OUTPUT NAME DETECTION
# ============================================================

def find_output_names(
    session: ort.InferenceSession,
) -> tuple[str, str]:

    available = [
        output.name
        for output in session.get_outputs()
    ]

    output_set = set(available)

    # TwinLiteNet+
    if {
        "drivable_area",
        "lane_line",
    }.issubset(output_set):

        return (
            "drivable_area",
            "lane_line",
        )

    # Original TwinLiteNet
    if {
        "da",
        "ll",
    }.issubset(output_set):

        return (
            "da",
            "ll",
        )

    raise ValueError(
        "Could not find road segmentation outputs.\n"
        f"Available outputs: {available}\n"
        "Expected either:\n"
        "  drivable_area + lane_line\n"
        "or:\n"
        "  da + ll"
    )


# ============================================================
# DECODE
# ============================================================

def decode(
    outputs: list[np.ndarray],
    source_size: tuple[int, int],
    height: int,
    width: int,
) -> list[np.ndarray]:

    masks = []

    names = (
        "drivable_area",
        "lane_line",
    )

    for name, output in zip(
        names,
        outputs,
    ):

        expected_shape = (
            1,
            2,
            height,
            width,
        )

        if output.shape != expected_shape:
            raise ValueError(
                f"{name}: expected "
                f"{expected_shape}, "
                f"got {output.shape}"
            )

        if not np.isfinite(
            output
        ).all():

            raise ValueError(
                f"{name}: model produced "
                "non-finite values"
            )

        # output shape:
        # [1, 2, H, W]
        #
        # channel 0 = background
        # channel 1 = foreground

        mask = np.argmax(
            output[0],
            axis=0,
        ).astype(np.uint8)

        # Resize mask back to original video size.
        mask = cv2.resize(
            mask,
            source_size,
            interpolation=cv2.INTER_NEAREST,
        )

        masks.append(mask)

    return masks


# ============================================================
# STATISTICS
# ============================================================

def stats(
    values: np.ndarray,
) -> dict:

    return {
        "average_ms": float(
            np.mean(values)
        ),
        "median_ms": float(
            np.median(values)
        ),
        "p95_ms": float(
            np.percentile(values, 95)
        ),
        "min_ms": float(
            np.min(values)
        ),
        "max_ms": float(
            np.max(values)
        ),
    }


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    args,
) -> None:

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found:\n"
            f"{args.model}"
        )

    if not args.video.is_file():
        raise FileNotFoundError(
            f"Video not found:\n"
            f"{args.video}"
        )

    # Keep OpenCV from creating many CPU threads.
    cv2.setNumThreads(1)

    # --------------------------------------------------------
    # ONNX Runtime configuration
    # --------------------------------------------------------

    options = ort.SessionOptions()

    options.intra_op_num_threads = (
        args.threads
    )

    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    options.add_session_config_entry(
        "session.intra_op.allow_spinning",
        "0",
    )

    options.add_session_config_entry(
        "session.inter_op.allow_spinning",
        "0",
    )

    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    inputs = session.get_inputs()

    if len(inputs) != 1:
        raise ValueError(
            "Expected exactly one model input, "
            f"found {len(inputs)}"
        )

    spec = inputs[0]

    if spec.type != "tensor(float)":
        raise ValueError(
            f"Expected float32 input, "
            f"got {spec.type}"
        )

    if len(spec.shape) != 4:
        raise ValueError(
            f"Expected 4D NCHW input, "
            f"got {spec.shape}"
        )

    if spec.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1, "
            f"got {spec.shape}"
        )

    if spec.shape[1] != 3:
        raise ValueError(
            f"Expected 3 image channels, "
            f"got {spec.shape}"
        )

    if not all(
        isinstance(value, int)
        and value > 0
        for value in spec.shape
    ):
        raise ValueError(
            "Benchmark requires a static "
            f"input shape, got {spec.shape}"
        )

    height = spec.shape[2]
    width = spec.shape[3]

    # --------------------------------------------------------
    # Model outputs
    # --------------------------------------------------------

    road_output_name, lane_output_name = (
        find_output_names(session)
    )

    print()
    print("=" * 65)
    print("ROAD SEGMENTATION CPU BENCHMARK")
    print("=" * 65)

    print(
        f"Model:   {args.model}"
    )

    print(
        f"Video:   {args.video}"
    )

    print(
        f"Threads: {args.threads}"
    )

    print(
        f"Input:   {width}x{height}"
    )

    print(
        f"Input name: {spec.name}"
    )

    print(
        f"Road output: {road_output_name}"
    )

    print(
        f"Lane output: {lane_output_name}"
    )

    print(
        "Provider:",
        session.get_providers(),
    )

    print("=" * 65)
    print()

    # --------------------------------------------------------
    # Inference function
    # --------------------------------------------------------

    def predict(
        frame: np.ndarray,
    ):

        total_start = (
            time.perf_counter()
        )

        # ----------------------------
        # Preprocess
        # ----------------------------

        preprocess_start = (
            time.perf_counter()
        )

        tensor = preprocess(
            frame,
            height,
            width,
        )

        preprocess_end = (
            time.perf_counter()
        )

        # ----------------------------
        # ONNX inference
        # ----------------------------

        outputs = session.run(
            [
                road_output_name,
                lane_output_name,
            ],
            {
                spec.name: tensor
            },
        )

        inference_end = (
            time.perf_counter()
        )

        # ----------------------------
        # Mask decoding
        # ----------------------------

        masks = decode(
            outputs,
            frame.shape[1::-1],
            height,
            width,
        )

        postprocess_end = (
            time.perf_counter()
        )

        preprocess_ms = (
            preprocess_end
            - preprocess_start
        ) * 1000.0

        inference_ms = (
            inference_end
            - preprocess_end
        ) * 1000.0

        postprocess_ms = (
            postprocess_end
            - inference_end
        ) * 1000.0

        total_ms = (
            postprocess_end
            - total_start
        ) * 1000.0

        timings = (
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            total_ms,
        )

        return masks, timings

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(args.video)
    )

    timings = []
    coverage = []

    last_frame = None
    last_masks = None

    try:

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video:\n"
                f"{args.video}"
            )

        if args.start_frame > 0:

            success = cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                args.start_frame,
            )

            if not success:
                raise RuntimeError(
                    "Could not seek to "
                    f"frame {args.start_frame}"
                )

        ok, first = cap.read()

        if not ok:
            raise RuntimeError(
                "Could not read the first "
                "requested video frame."
            )

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        print(
            f"Warming up "
            f"{args.warmup} runs..."
        )

        for _ in range(
            args.warmup
        ):
            predict(first)

        print(
            "Warmup complete."
        )

        print(
            "Measuring frames..."
        )

        print()

        # ----------------------------------------------------
        # Benchmark frames
        # ----------------------------------------------------

        frame = first

        for index in range(
            args.frames
        ):

            if index > 0:

                ok, frame = cap.read()

                if not ok:

                    print(
                        f"Video ended after "
                        f"{index} measured frames."
                    )

                    break

            masks, measured = predict(
                frame
            )

            timings.append(
                measured
            )

            road_mask, lane_mask = masks

            coverage.append(
                [
                    float(
                        road_mask.mean()
                    ) * 100.0,

                    float(
                        lane_mask.mean()
                    ) * 100.0,
                ]
            )

            last_frame = frame.copy()

            last_masks = (
                road_mask,
                lane_mask,
            )

            if (
                index == 0
                or (index + 1) % 25 == 0
                or index + 1 == args.frames
            ):

                print(
                    f"{index + 1}/"
                    f"{args.frames}: "
                    f"total="
                    f"{measured[3]:.1f} ms | "
                    f"inference="
                    f"{measured[1]:.1f} ms"
                )

    finally:

        cap.release()

    # --------------------------------------------------------
    # Validate benchmark
    # --------------------------------------------------------

    if not timings:

        raise RuntimeError(
            "No frames were measured."
        )

    values = np.asarray(
        timings,
        dtype=np.float64,
    )

    coverage_values = np.asarray(
        coverage,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Model hash
    # --------------------------------------------------------

    with args.model.open(
        "rb"
    ) as stream:

        model_hash = (
            hashlib.file_digest(
                stream,
                "sha256",
            ).hexdigest()
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "model": str(
            args.model.resolve()
        ),

        "model_sha256": model_hash,

        "model_size_bytes": (
            args.model.stat().st_size
        ),

        "video": str(
            args.video.resolve()
        ),

        "start_frame": (
            args.start_frame
        ),

        "requested_frames": (
            args.frames
        ),

        "measured_frames": (
            len(timings)
        ),

        "warmup_runs": (
            args.warmup
        ),

        "threads": (
            args.threads
        ),

        "providers": (
            session.get_providers()
        ),

        "input_name": (
            spec.name
        ),

        "input_shape": (
            spec.shape
        ),

        "output_names": [
            road_output_name,
            lane_output_name,
        ],

        "preprocess": stats(
            values[:, 0]
        ),

        "inference": stats(
            values[:, 1]
        ),

        "postprocess": stats(
            values[:, 2]
        ),

        "total": stats(
            values[:, 3]
        ),

        "approximate_processing_fps": (
            1000.0
            / float(
                values[:, 3].mean()
            )
        ),

        "mean_drivable_area_percent": (
            float(
                coverage_values[
                    :,
                    0,
                ].mean()
            )
        ),

        "mean_lane_percent": (
            float(
                coverage_values[
                    :,
                    1,
                ].mean()
            )
        ),

        "versions": {
            "python": (
                platform.python_version()
            ),

            "opencv": (
                cv2.__version__
            ),

            "numpy": (
                np.__version__
            ),

            "onnxruntime": (
                ort.__version__
            ),
        },

        "note": (
            "Standalone road-model processing speed. "
            "This is not live display FPS and does not "
            "include YOLO, pose, tracking, scheduler or "
            "renderer load. Mask coverage is not accuracy."
        ),
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    # --------------------------------------------------------
    # Save preview image
    # --------------------------------------------------------

    preview_path = (
        args.output.with_suffix(
            ".jpg"
        )
    )

    preview = (
        last_frame.copy()
    )

    road_mask, lane_mask = (
        last_masks
    )

    # Drivable area = green
    road_selected = (
        road_mask.astype(bool)
    )

    road_color = np.array(
        [0, 200, 0],
        dtype=np.float32,
    )

    preview[
        road_selected
    ] = (
        0.55
        * preview[
            road_selected
        ]
        + 0.45
        * road_color
    ).astype(
        np.uint8
    )

    # Lane line = red
    lane_selected = (
        lane_mask.astype(bool)
    )

    lane_color = np.array(
        [0, 0, 255],
        dtype=np.float32,
    )

    preview[
        lane_selected
    ] = (
        0.35
        * preview[
            lane_selected
        ]
        + 0.65
        * lane_color
    ).astype(
        np.uint8
    )

    cv2.putText(
        preview,
        "green = drivable | red = lane",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(
        str(preview_path),
        preview,
    )

    if not success:
        raise RuntimeError(
            f"Could not write preview:\n"
            f"{preview_path}"
        )

    report[
        "preview"
    ] = str(
        preview_path.resolve()
    )

    # Rewrite report so preview path is included.
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n"
    )

    # --------------------------------------------------------
    # Console results
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(
        f"Measured frames: "
        f"{len(timings)}"
    )

    print()

    print(
        "Preprocess:"
    )

    print(
        f"  average = "
        f"{report['preprocess']['average_ms']:.2f} ms"
    )

    print(
        f"  p95     = "
        f"{report['preprocess']['p95_ms']:.2f} ms"
    )

    print()

    print(
        "Inference:"
    )

    print(
        f"  average = "
        f"{report['inference']['average_ms']:.2f} ms"
    )

    print(
        f"  median  = "
        f"{report['inference']['median_ms']:.2f} ms"
    )

    print(
        f"  p95     = "
        f"{report['inference']['p95_ms']:.2f} ms"
    )

    print()

    print(
        "Postprocess:"
    )

    print(
        f"  average = "
        f"{report['postprocess']['average_ms']:.2f} ms"
    )

    print(
        f"  p95     = "
        f"{report['postprocess']['p95_ms']:.2f} ms"
    )

    print()

    print(
        "TOTAL:"
    )

    print(
        f"  average = "
        f"{report['total']['average_ms']:.2f} ms"
    )

    print(
        f"  median  = "
        f"{report['total']['median_ms']:.2f} ms"
    )

    print(
        f"  p95     = "
        f"{report['total']['p95_ms']:.2f} ms"
    )

    print()

    print(
        "Standalone road throughput:"
    )

    print(
        f"  "
        f"{report['approximate_processing_fps']:.2f} FPS"
    )

    print()

    print(
        "Mask coverage:"
    )

    print(
        f"  drivable = "
        f"{report['mean_drivable_area_percent']:.2f}%"
    )

    print(
        f"  lane     = "
        f"{report['mean_lane_percent']:.2f}%"
    )

    print()

    print(
        f"Report:  "
        f"{args.output}"
    )

    print(
        f"Preview: "
        f"{preview_path}"
    )

    print("=" * 65)


# ============================================================
# CLI
# ============================================================

def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            ROOT
            / "models"
            / "twinlitenetplus_nano.onnx"
        ),
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=(
            ROOT
            / "test_data"
            / "test_video.mp4"
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "validation"
            / "road_twinlitenetplus_nano.json"
        ),
    )

    args = parser.parse_args()

    if args.threads < 1:
        parser.error(
            "--threads must be >= 1"
        )

    if args.frames < 1:
        parser.error(
            "--frames must be >= 1"
        )

    if args.warmup < 0:
        parser.error(
            "--warmup must be >= 0"
        )

    if args.start_frame < 0:
        parser.error(
            "--start-frame must be >= 0"
        )

    try:

        benchmark(args)

    except (
        OSError,
        RuntimeError,
        ValueError,
        ort.capi.onnxruntime_pybind11_state.Fail,
        ort.capi.onnxruntime_pybind11_state.InvalidProtobuf,
    ) as exc:

        print(
            f"Benchmark failed: {exc}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())