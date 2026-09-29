"""
Inspect TwinLiteNet+ road masks across a full video.

This script:
- samples frames across the whole video
- uses correct aspect-ratio letterbox preprocessing
- runs TwinLiteNet+ ONNX
- removes letterbox padding from masks
- resizes masks back to the original frame
- saves individual preview images
- creates one montage image

No live pipeline changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(
    frame: np.ndarray,
    target_height: int,
    target_width: int,
):
    """
    Resize while keeping aspect ratio.

    Remaining space is padded with 114,114,114.

    Returns:
        padded image
        crop area inside padded image
    """

    original_height, original_width = frame.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = int(
        round(original_width * scale)
    )

    resized_height = int(
        round(original_height * scale)
    )

    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height

    left = int(
        round(pad_width / 2 - 0.1)
    )

    right = pad_width - left

    top = int(
        round(pad_height / 2 - 0.1)
    )

    bottom = pad_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    crop = (
        left,
        top,
        left + resized_width,
        top + resized_height,
    )

    return padded, crop


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(
    frame: np.ndarray,
    height: int,
    width: int,
):
    padded, crop = letterbox(
        frame,
        height,
        width,
    )

    # BGR -> RGB
    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB,
    )

    # HWC -> CHW
    tensor = rgb.transpose(
        2,
        0,
        1,
    )

    tensor = tensor.astype(
        np.float32
    )

    tensor /= 255.0

    # Add batch dimension
    tensor = np.expand_dims(
        tensor,
        axis=0,
    )

    tensor = np.ascontiguousarray(
        tensor
    )

    return tensor, crop


# ============================================================
# FIND OUTPUT NAMES
# ============================================================

def find_output_names(
    session: ort.InferenceSession,
):
    names = [
        output.name
        for output in session.get_outputs()
    ]

    names_set = set(names)

    # TwinLiteNet+
    if {
        "drivable_area",
        "lane_line",
    }.issubset(names_set):

        return (
            "drivable_area",
            "lane_line",
        )

    # Original TwinLiteNet
    if {
        "da",
        "ll",
    }.issubset(names_set):

        return (
            "da",
            "ll",
        )

    raise RuntimeError(
        f"Unknown model outputs: {names}"
    )


# ============================================================
# DECODE
# ============================================================

def decode_mask(
    output: np.ndarray,
    crop,
    original_width: int,
    original_height: int,
):
    """
    Convert model output into a binary mask.

    Then remove the letterbox padding and return
    the mask at original frame resolution.
    """

    if output.ndim != 4:
        raise RuntimeError(
            f"Unexpected output shape: {output.shape}"
        )

    if output.shape[0] != 1:
        raise RuntimeError(
            f"Expected batch size 1: {output.shape}"
        )

    if output.shape[1] != 2:
        raise RuntimeError(
            f"Expected 2 classes: {output.shape}"
        )

    mask = np.argmax(
        output[0],
        axis=0,
    ).astype(
        np.uint8
    )

    left, top, right, bottom = crop

    mask = mask[
        top:bottom,
        left:right,
    ]

    mask = cv2.resize(
        mask,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


# ============================================================
# OVERLAY
# ============================================================

def make_overlay(
    frame: np.ndarray,
    road_mask: np.ndarray,
    lane_mask: np.ndarray,
    frame_number: int,
):
    output = frame.copy()

    # --------------------------------------------------------
    # Drivable area = green
    # --------------------------------------------------------

    road_pixels = (
        road_mask > 0
    )

    green = np.zeros_like(
        output
    )

    green[:, :] = (
        0,
        220,
        0,
    )

    output[
        road_pixels
    ] = cv2.addWeighted(
        output[
            road_pixels
        ],
        0.55,
        green[
            road_pixels
        ],
        0.45,
        0,
    )

    # --------------------------------------------------------
    # Lane = red
    # --------------------------------------------------------

    lane_pixels = (
        lane_mask > 0
    )

    red = np.zeros_like(
        output
    )

    red[:, :] = (
        0,
        0,
        255,
    )

    output[
        lane_pixels
    ] = cv2.addWeighted(
        output[
            lane_pixels
        ],
        0.25,
        red[
            lane_pixels
        ],
        0.75,
        0,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        output,
        f"frame {frame_number}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        "green=drivable | red=lane",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# MONTAGE
# ============================================================

def make_montage(
    images: list[np.ndarray],
    columns: int = 3,
):
    if not images:
        raise RuntimeError(
            "No images available for montage."
        )

    tile_width = 480

    tiles = []

    for image in images:

        h, w = image.shape[:2]

        scale = (
            tile_width / w
        )

        new_height = int(
            round(h * scale)
        )

        tile = cv2.resize(
            image,
            (
                tile_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tiles.append(tile)

    # Make all tiles same height
    tile_height = max(
        tile.shape[0]
        for tile in tiles
    )

    normalized = []

    for tile in tiles:

        if tile.shape[0] < tile_height:

            padding = (
                tile_height
                - tile.shape[0]
            )

            tile = cv2.copyMakeBorder(
                tile,
                0,
                padding,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        normalized.append(tile)

    rows = []

    for start in range(
        0,
        len(normalized),
        columns,
    ):

        row = normalized[
            start:start + columns
        ]

        # Fill missing tiles
        while len(row) < columns:

            blank = np.zeros_like(
                normalized[0]
            )

            row.append(blank)

        rows.append(
            cv2.hconcat(row)
        )

    montage = cv2.vconcat(
        rows
    )

    return montage


# ============================================================
# MAIN
# ============================================================

def inspect(args):
    if not args.model.is_file():

        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    if not args.video.is_file():

        raise FileNotFoundError(
            f"Video not found: {args.video}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.setNumThreads(1)

    # --------------------------------------------------------
    # ONNX
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

    input_spec = (
        session.get_inputs()[0]
    )

    _, channels, height, width = (
        input_spec.shape
    )

    if channels != 3:
        raise RuntimeError(
            f"Expected 3 channels: {input_spec.shape}"
        )

    road_output, lane_output = (
        find_output_names(
            session
        )
    )

    print()
    print("=" * 60)
    print("ROAD MASK INSPECTION")
    print("=" * 60)

    print(
        f"Model: {args.model}"
    )

    print(
        f"Input: {width}x{height}"
    )

    print(
        f"Threads: {args.threads}"
    )

    print(
        f"Samples: {args.samples}"
    )

    print()

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(args.video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {args.video}"
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:

        cap.release()

        raise RuntimeError(
            "Video frame count unavailable."
        )

    print(
        f"Video frames: {total_frames}"
    )

    # Avoid exact first / last frame
    sample_positions = np.linspace(
        0.05,
        0.95,
        args.samples,
    )

    frame_numbers = [
        int(
            position
            * (total_frames - 1)
        )
        for position in sample_positions
    ]

    previews = []

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    for sample_index, frame_number in enumerate(
        frame_numbers,
        start=1,
    ):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        ok, frame = cap.read()

        if not ok:

            print(
                f"Could not read frame "
                f"{frame_number}"
            )

            continue

        original_height, original_width = (
            frame.shape[:2]
        )

        tensor, crop = preprocess(
            frame,
            height,
            width,
        )

        outputs = session.run(
            [
                road_output,
                lane_output,
            ],
            {
                input_spec.name: tensor
            },
        )

        road_mask = decode_mask(
            outputs[0],
            crop,
            original_width,
            original_height,
        )

        lane_mask = decode_mask(
            outputs[1],
            crop,
            original_width,
            original_height,
        )

        preview = make_overlay(
            frame,
            road_mask,
            lane_mask,
            frame_number,
        )

        previews.append(
            preview
        )

        output_path = (
            args.output_dir
            / f"frame_{frame_number:06d}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            preview,
        )

        road_coverage = (
            road_mask.mean()
            * 100.0
        )

        lane_coverage = (
            lane_mask.mean()
            * 100.0
        )

        print(
            f"[{sample_index:02d}/"
            f"{args.samples:02d}] "
            f"frame={frame_number:<6d} "
            f"road={road_coverage:5.1f}% "
            f"lane={lane_coverage:5.1f}%"
        )

    cap.release()

    # --------------------------------------------------------
    # Montage
    # --------------------------------------------------------

    montage = make_montage(
        previews,
        columns=3,
    )

    montage_path = (
        args.output_dir
        / "montage.jpg"
    )

    cv2.imwrite(
        str(montage_path),
        montage,
    )

    print()
    print("=" * 60)

    print(
        f"Saved {len(previews)} previews."
    )

    print(
        f"Montage: {montage_path}"
    )

    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

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
        "--samples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "validation"
            / "road_nano_inspection"
        ),
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error(
            "--samples must be >= 1"
        )

    if args.threads < 1:
        parser.error(
            "--threads must be >= 1"
        )

    inspect(args)


if __name__ == "__main__":
    main()