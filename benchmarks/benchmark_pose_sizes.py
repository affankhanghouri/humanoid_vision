"""Compare fixed-size YOLO26n Pose ONNX exports on identical webcam frames."""

import gc
import time
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    512: PROJECT_ROOT / "models" / "yolo26n-pose-512.onnx",
    384: PROJECT_ROOT / "models" / "yolo26n-pose-384.onnx",
    320: PROJECT_ROOT / "models" / "yolo26n-pose-320.onnx",
    256: PROJECT_ROOT / "models" / "yolo26n-pose-256.onnx",
}

FRAME_COUNT = 30

IMAGE_SIZES = (
    512,
    384,
    320,
    256,
)

CONFIDENCE = 0.25


def capture_frames():
    print("Opening webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam."
        )

    frames = []

    print(
        "Stand in front of the camera."
    )

    print(
        f"Capturing {FRAME_COUNT} frames..."
    )

    while len(frames) < FRAME_COUNT:

        ok, frame = cap.read()

        if not ok:
            continue

        frames.append(frame)

        preview = frame.copy()

        cv2.putText(
            preview,
            f"Capturing {len(frames)}/{FRAME_COUNT}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Pose Benchmark Capture",
            preview,
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frames


def benchmark_size(
    model,
    frames,
    image_size,
):

    print()
    print("=" * 60)
    print(
        f"IMAGE SIZE: {image_size}"
    )
    print("=" * 60)

    # Warm up
    for _ in range(3):

        model.predict(
            source=frames[0],
            imgsz=image_size,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False,
        )

    times = []

    detected_people = 0
    detected_keypoints = 0

    for index, frame in enumerate(
        frames,
        start=1,
    ):

        start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=image_size,
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

        people = (
            len(result.boxes)
            if result.boxes is not None
            else 0
        )

        detected_people += people

        keypoints = 0

        if (
            result.keypoints is not None
            and result.keypoints.xy is not None
        ):

            data = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            if len(data) > 0:

                keypoints = (
                    data.shape[0]
                    * data.shape[1]
                )

        detected_keypoints += (
            keypoints
        )

        print(
            f"{index:02d} | "
            f"{elapsed_ms:7.2f} ms | "
            f"people={people} | "
            f"keypoints={keypoints}"
        )

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

    hz = (
        1000.0 / average
        if average > 0
        else 0.0
    )

    print()
    print("RESULT")
    print(
        f"Image size:     {image_size}"
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
        f"Approx Hz:      {hz:.2f}"
    )
    print(
        f"Avg people:     "
        f"{detected_people / len(frames):.2f}"
    )
    print(
        f"Avg keypoints:  "
        f"{detected_keypoints / len(frames):.2f}"
    )

    return (
        image_size,
        average,
        p95,
        hz,
    )


def main():

    for image_size in IMAGE_SIZES:
        if not MODEL_PATHS[image_size].is_file():
            raise FileNotFoundError(
                f"Missing fixed-size pose model: {MODEL_PATHS[image_size]}"
            )

    frames = capture_frames()

    if not frames:
        raise RuntimeError(
            "No webcam frames captured."
        )

    results = []

    for image_size in IMAGE_SIZES:

        print()
        print(f"Loading {image_size} model...")
        model = YOLO(str(MODEL_PATHS[image_size]), task="pose")

        result = benchmark_size(
            model,
            frames,
            image_size,
        )

        results.append(
            result
        )

        # Release the previous ONNX session before loading the next model.
        del model
        gc.collect()

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for (
        size,
        average,
        p95,
        hz,
    ) in results:

        print(
            f"{size:3d} | "
            f"avg {average:7.2f} ms | "
            f"p95 {p95:7.2f} ms | "
            f"{hz:5.2f} Hz"
        )


if __name__ == "__main__":
    main()
