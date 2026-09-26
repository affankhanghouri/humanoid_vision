"""Benchmark YOLO26n Pose ONNX on CPU."""

import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    PROJECT_ROOT
    / "test_data"
    / "test_video.mp4"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "yolo26n-pose-512.onnx"
)

IMAGE_SIZE = 512
CONFIDENCE = 0.5

NUM_FRAMES = 60
WARMUP_RUNS = 5


def load_frames(
    video_path: Path,
    count: int,
):
    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    frames = []

    while len(frames) < count:

        ok, frame = cap.read()

        if not ok:
            break

        frames.append(frame)

    cap.release()

    return frames


def main():

    print(
        f"Loading {NUM_FRAMES} frames..."
    )

    frames = load_frames(
        VIDEO_PATH,
        NUM_FRAMES,
    )

    if not frames:
        raise RuntimeError(
            "No frames loaded."
        )

    print(
        f"Loaded {len(frames)} frames"
    )

    print()
    print(
        "Loading YOLO26n Pose ONNX..."
    )

    model = YOLO(
        str(MODEL_PATH)
    )

    # ------------------------------------------
    # Warm-up
    # ------------------------------------------

    print(
        f"Warming up "
        f"({WARMUP_RUNS} runs)..."
    )

    for _ in range(WARMUP_RUNS):

        model.predict(
            source=frames[0],
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

    print("Warm-up complete.")

    # ------------------------------------------
    # Benchmark
    # ------------------------------------------

    times = []

    total_people = 0
    total_keypoints = 0

    print()
    print("=" * 70)
    print("POSE BENCHMARK")
    print("=" * 70)

    for index, frame in enumerate(
        frames,
        start=1,
    ):

        start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )[0]

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        times.append(
            elapsed_ms
        )

        # --------------------------------------
        # People
        # --------------------------------------

        people = 0

        if result.boxes is not None:
            people = len(
                result.boxes
            )

        total_people += people

        # --------------------------------------
        # Keypoints
        # --------------------------------------

        keypoint_count = 0

        if (
            result.keypoints is not None
            and result.keypoints.xy is not None
        ):

            xy = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            if len(xy) > 0:

                # Shape normally:
                # people x 17 x 2
                keypoint_count = (
                    xy.shape[0]
                    * xy.shape[1]
                )

        total_keypoints += (
            keypoint_count
        )

        print(
            f"{index:02d} | "
            f"{elapsed_ms:8.2f} ms | "
            f"people: {people:2d} | "
            f"keypoints: {keypoint_count:3d}"
        )

    # ------------------------------------------
    # Statistics
    # ------------------------------------------

    values = np.asarray(
        times,
        dtype=np.float32,
    )

    average = float(
        np.mean(values)
    )

    median = float(
        np.median(values)
    )

    p95 = float(
        np.percentile(
            values,
            95,
        )
    )

    fastest = float(
        np.min(values)
    )

    slowest = float(
        np.max(values)
    )

    approx_hz = (
        1000.0 / average
        if average > 0
        else 0.0
    )

    avg_people = (
        total_people
        / len(frames)
    )

    avg_keypoints = (
        total_keypoints
        / len(frames)
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "Model:          YOLO26n-Pose ONNX"
    )

    print(
        f"Image size:     {IMAGE_SIZE}"
    )

    print(
        f"Average:        {average:.2f} ms"
    )

    print(
        f"Median:         {median:.2f} ms"
    )

    print(
        f"P95:            {p95:.2f} ms"
    )

    print(
        f"Fastest:        {fastest:.2f} ms"
    )

    print(
        f"Slowest:        {slowest:.2f} ms"
    )

    print(
        f"Approx FPS:     {approx_hz:.2f}"
    )

    print(
        f"Avg people:     {avg_people:.2f}"
    )

    print(
        f"Avg keypoints:  {avg_keypoints:.2f}"
    )


if __name__ == "__main__":
    main()
