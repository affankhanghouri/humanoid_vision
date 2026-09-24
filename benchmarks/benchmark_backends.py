import sys
from pathlib import Path
import time
import statistics

import cv2
import numpy as np
import torch

from ultralytics import YOLO


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROJECT_ROOT, VisionConfig

CONFIG = VisionConfig()
VIDEO_PATH = CONFIG.video_path

IMAGE_SIZE = CONFIG.image_size
CONFIDENCE = CONFIG.confidence

TEST_FRAMES = 60
WARMUP_RUNS = 5


BACKENDS = [
    {
        "name": "PyTorch",
        "model": str(PROJECT_ROOT / "models" / "yolo26n.pt"),
        "device": "cpu",
    },
    {
        "name": "OpenVINO",
        "model": CONFIG.model_path,
        "device": "intel:cpu",
    },
    {
        "name": "ONNX",
        "model": str(PROJECT_ROOT / "models" / "yolo26n.onnx"),
        "device": "cpu",
    },
]


torch.set_num_threads(4)


def load_test_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    frames = []

    while len(frames) < TEST_FRAMES:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

    cap.release()

    print(f"Loaded {len(frames)} frames")

    return frames


def benchmark_backend(config, frames):
    print("\n" + "=" * 55)
    print(f"BACKEND: {config['name']}")
    print("=" * 55)

    model = YOLO(config["model"])

    # ---------------------------------------
    # Warm-up
    # ---------------------------------------

    print("Warming up...")

    warmup_frame = frames[0]

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=warmup_frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=config["device"],
            verbose=False,
        )

    # ---------------------------------------
    # Benchmark
    # ---------------------------------------

    times = []
    detection_counts = []

    for index, frame in enumerate(frames):
        start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=config["device"],
            verbose=False,
        )[0]

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        times.append(elapsed_ms)
        detection_counts.append(
            len(result.boxes)
        )

        print(
            f"{index + 1:02d} | "
            f"{elapsed_ms:7.2f} ms | "
            f"objects: {len(result.boxes)}"
        )

    # ---------------------------------------
    # Statistics
    # ---------------------------------------

    sorted_times = sorted(times)

    average = statistics.mean(times)
    median = statistics.median(times)

    p95_index = int(
        len(sorted_times) * 0.95
    ) - 1

    p95 = sorted_times[
        max(0, p95_index)
    ]

    fastest = min(times)
    slowest = max(times)

    approx_fps = 1000 / average

    avg_detections = statistics.mean(
        detection_counts
    )

    print("\nRESULT")
    print(f"Backend:       {config['name']}")
    print(f"Average:       {average:.2f} ms")
    print(f"Median:        {median:.2f} ms")
    print(f"P95:           {p95:.2f} ms")
    print(f"Fastest:       {fastest:.2f} ms")
    print(f"Slowest:       {slowest:.2f} ms")
    print(f"Approx FPS:    {approx_fps:.2f}")
    print(f"Avg detections:{avg_detections:.2f}")

    return {
        "backend": config["name"],
        "average": average,
        "median": median,
        "p95": p95,
        "fps": approx_fps,
        "detections": avg_detections,
    }


def main():
    frames = load_test_frames()

    results = []

    for backend in BACKENDS:
        result = benchmark_backend(
            backend,
            frames,
        )

        results.append(result)

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for result in results:
        print(
            f"{result['backend']:10s} | "
            f"avg {result['average']:7.2f} ms | "
            f"median {result['median']:7.2f} ms | "
            f"P95 {result['p95']:7.2f} ms | "
            f"{result['fps']:5.2f} FPS | "
            f"objects {result['detections']:.2f}"
        )


if __name__ == "__main__":
    main()