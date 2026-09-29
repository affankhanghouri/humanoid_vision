"""
Road Geometry V2.

Neural network:
    TwinLiteNet+ Nano
        ↓
    drivable mask + lane mask

Geometry:
    drivable mask
        ↓
    road corridor boundaries

    lane mask
        ↓
    remove horizontal markings
        ↓
    line segments
        ↓
    group possible lane boundaries
        ↓
    left/right ego lane boundaries

Important:
Road corridor and ego lane are treated separately.

A road corridor may be valid even when the ego lane
cannot be found safely.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# TYPES
# ============================================================

@dataclass
class CurveFit:
    coefficients: np.ndarray
    confidence: float
    residual_px: float
    point_count: int


@dataclass
class GeometryResult:
    road_left: CurveFit | None
    road_right: CurveFit | None

    lane_left: CurveFit | None
    lane_right: CurveFit | None

    road_valid: bool
    ego_lane_valid: bool

    normalized_offset: float | None

    left_lane_confidence: float
    right_lane_confidence: float


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(
    frame: np.ndarray,
    target_height: int,
    target_width: int,
):
    h, w = frame.shape[:2]

    scale = min(
        target_width / w,
        target_height / h,
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    dw = target_width - new_w
    dh = target_height - new_h

    left = int(round(dw / 2 - 0.1))
    top = int(round(dh / 2 - 0.1))

    right_pad = dw - left
    bottom_pad = dh - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom_pad,
        left,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    crop = (
        left,
        top,
        left + new_w,
        top + new_h,
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
    image, crop = letterbox(
        frame,
        height,
        width,
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    tensor = image.transpose(
        2,
        0,
        1,
    ).astype(np.float32)

    tensor /= 255.0

    tensor = tensor[None]

    return np.ascontiguousarray(tensor), crop


# ============================================================
# OUTPUT NAMES
# ============================================================

def find_output_names(
    session: ort.InferenceSession,
):
    names = {
        output.name
        for output in session.get_outputs()
    }

    if {
        "drivable_area",
        "lane_line",
    }.issubset(names):

        return (
            "drivable_area",
            "lane_line",
        )

    if {
        "da",
        "ll",
    }.issubset(names):

        return (
            "da",
            "ll",
        )

    raise RuntimeError(
        f"Unknown outputs: {sorted(names)}"
    )


# ============================================================
# MASK DECODING
# ============================================================

def decode_mask(
    output: np.ndarray,
    crop,
    source_width: int,
    source_height: int,
):
    mask = np.argmax(
        output[0],
        axis=0,
    ).astype(np.uint8)

    left, top, right, bottom = crop

    mask = mask[
        top:bottom,
        left:right,
    ]

    mask = cv2.resize(
        mask,
        (
            source_width,
            source_height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask


# ============================================================
# RUN DETECTION
# ============================================================

def find_runs(
    row: np.ndarray,
):
    positions = np.flatnonzero(
        row
    )

    if len(positions) == 0:
        return []

    splits = np.where(
        np.diff(positions) > 1
    )[0] + 1

    groups = np.split(
        positions,
        splits,
    )

    output = []

    for group in groups:

        if len(group) == 0:
            continue

        output.append(
            (
                int(group[0]),
                int(group[-1]),
            )
        )

    return output


# ============================================================
# ROBUST CURVE FIT
# ============================================================

def robust_curve_fit(
    points: np.ndarray,
    image_width: int,
    min_points: int = 8,
):
    if points is None:
        return None

    if len(points) < min_points:
        return None

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    xs = points[:, 0]
    ys = points[:, 1]

    y_span = (
        ys.max()
        - ys.min()
    )

    if y_span < 25:
        return None

    # Do not use a strong quadratic curve unless
    # we have enough vertical information.
    degree = (
        2
        if y_span >= 100
        and len(points) >= 12
        else 1
    )

    keep = np.ones(
        len(points),
        dtype=bool,
    )

    residual_limit = max(
        14.0,
        image_width * 0.018,
    )

    coefficients = None

    for _ in range(5):

        if keep.sum() < min_points:
            return None

        coefficients = np.polyfit(
            ys[keep],
            xs[keep],
            degree,
        )

        prediction = np.polyval(
            coefficients,
            ys,
        )

        residuals = np.abs(
            xs - prediction
        )

        new_keep = (
            residuals
            <= residual_limit
        )

        if np.array_equal(
            keep,
            new_keep,
        ):
            break

        keep = new_keep

    if coefficients is None:
        return None

    filtered = points[
        keep
    ]

    if len(filtered) < min_points:
        return None

    prediction = np.polyval(
        coefficients,
        filtered[:, 1],
    )

    residual = float(
        np.mean(
            np.abs(
                filtered[:, 0]
                - prediction
            )
        )
    )

    y_span = float(
        filtered[:, 1].max()
        - filtered[:, 1].min()
    )

    point_score = min(
        len(filtered) / 24.0,
        1.0,
    )

    span_score = min(
        y_span / 250.0,
        1.0,
    )

    residual_score = max(
        0.0,
        1.0
        - residual / residual_limit,
    )

    confidence = (
        0.35 * point_score
        + 0.40 * span_score
        + 0.25 * residual_score
    )

    return CurveFit(
        coefficients=coefficients,
        confidence=float(confidence),
        residual_px=residual,
        point_count=len(filtered),
    )


# ============================================================
# ROAD CORRIDOR FROM DRIVABLE MASK
# ============================================================

def extract_road_boundaries(
    road_mask: np.ndarray,
):
    h, w = road_mask.shape

    center_x = w / 2.0

    y_start = int(
        h * 0.42
    )

    y_end = int(
        h * 0.97
    )

    # Fill small holes in road segmentation.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            max(9, int(w * 0.025)),
            max(7, int(h * 0.018)),
        ),
    )

    clean = cv2.morphologyEx(
        road_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
    )

    left_points = []
    right_points = []

    step = max(
        6,
        int(h * 0.012),
    )

    for y in range(
        y_end,
        y_start,
        -step,
    ):
        band_top = max(
            0,
            y - 2,
        )

        band_bottom = min(
            h,
            y + 3,
        )

        row = clean[
            band_top:band_bottom
        ].max(axis=0)

        runs = find_runs(
            row
        )

        if not runs:
            continue

        chosen = None

        # Prefer a road section containing
        # the camera center.
        for left, right in runs:

            if left <= center_x <= right:

                if (
                    chosen is None
                    or right - left
                    > chosen[1] - chosen[0]
                ):
                    chosen = (
                        left,
                        right,
                    )

        # If camera center is temporarily blocked,
        # use the nearest large road region.
        if chosen is None:

            candidates = []

            for left, right in runs:

                run_width = (
                    right - left
                )

                if run_width < w * 0.12:
                    continue

                run_center = (
                    left + right
                ) / 2.0

                distance = abs(
                    run_center
                    - center_x
                )

                candidates.append(
                    (
                        distance,
                        left,
                        right,
                    )
                )

            if candidates:

                candidates.sort(
                    key=lambda item: item[0]
                )

                _, left, right = (
                    candidates[0]
                )

                chosen = (
                    left,
                    right,
                )

        if chosen is None:
            continue

        left, right = chosen

        if (
            right - left
            < w * 0.15
        ):
            continue

        left_points.append(
            (left, y)
        )

        right_points.append(
            (right, y)
        )

    left_fit = robust_curve_fit(
        np.asarray(left_points),
        w,
        min_points=10,
    )

    right_fit = robust_curve_fit(
        np.asarray(right_points),
        w,
        min_points=10,
    )

    return (
        left_fit,
        right_fit,
        y_start,
        y_end,
    )


# ============================================================
# REMOVE BAD LANE COMPONENTS
# ============================================================

def clean_lane_mask(
    lane_mask: np.ndarray,
    road_mask: np.ndarray,
):
    h, w = lane_mask.shape

    output = np.zeros_like(
        lane_mask,
        dtype=np.uint8,
    )

    # Slightly expand road mask so road-edge
    # lane lines are not removed.
    kernel_size = max(
        7,
        int(w * 0.012),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    road_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    expanded_road = cv2.dilate(
        road_mask.astype(np.uint8),
        road_kernel,
    )

    candidate = (
        (lane_mask > 0)
        & (expanded_road > 0)
    ).astype(np.uint8)

    # Ignore upper part.
    candidate[
        :int(h * 0.38)
    ] = 0

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            candidate,
            connectivity=8,
        )
    )

    for component in range(
        1,
        count,
    ):
        x = stats[
            component,
            cv2.CC_STAT_LEFT,
        ]

        y = stats[
            component,
            cv2.CC_STAT_TOP,
        ]

        width = stats[
            component,
            cv2.CC_STAT_WIDTH,
        ]

        height = stats[
            component,
            cv2.CC_STAT_HEIGHT,
        ]

        area = stats[
            component,
            cv2.CC_STAT_AREA,
        ]

        if area < 12:
            continue

        # Obvious crosswalk / stop-bar type piece.
        if (
            width > height * 3.0
            and width > w * 0.08
        ):
            continue

        coords_y, coords_x = np.where(
            labels == component
        )

        if len(coords_x) < 10:
            continue

        coords = np.column_stack(
            [
                coords_x,
                coords_y,
            ]
        ).astype(np.float64)

        centered = (
            coords
            - coords.mean(
                axis=0
            )
        )

        covariance = np.cov(
            centered.T
        )

        try:
            values, vectors = np.linalg.eigh(
                covariance
            )
        except np.linalg.LinAlgError:
            continue

        principal = vectors[
            :,
            np.argmax(values)
        ]

        verticalness = abs(
            principal[1]
        )

        # Reject strongly horizontal components.
        if (
            verticalness < 0.28
            and width > height
        ):
            continue

        output[
            labels == component
        ] = 1

    return output


# ============================================================
# HOUGH LANE SEGMENTS
# ============================================================

def get_lane_segments(
    clean_lane: np.ndarray,
):
    h, w = clean_lane.shape

    image = (
        clean_lane
        * 255
    ).astype(np.uint8)

    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180.0,
        threshold=18,
        minLineLength=max(
            18,
            int(h * 0.035),
        ),
        maxLineGap=max(
            15,
            int(h * 0.035),
        ),
    )

    if lines is None:
        return []

    # Normalize both OpenCV layouts: [N, 1, 4] and [N, 4].
    lines = np.asarray(lines).reshape(-1, 4)

    evaluation_y = (
        h * 0.92
    )

    segments = []

    for raw in lines:

        x1, y1, x2, y2 = map(float, raw)

        dx = (
            x2 - x1
        )

        dy = (
            y2 - y1
        )

        length = float(
            np.hypot(
                dx,
                dy,
            )
        )

        if length < 15:
            continue

        # Reject horizontal lines.
        vertical_ratio = (
            abs(dy)
            / max(length, 1.0)
        )

        if vertical_ratio < 0.30:
            continue

        if abs(dy) < 4:
            continue

        # Extrapolate segment toward bottom.
        slope_xy = (
            dx / dy
        )

        x_bottom = (
            x1
            + (
                evaluation_y - y1
            )
            * slope_xy
        )

        if (
            x_bottom < -0.25 * w
            or x_bottom > 1.25 * w
        ):
            continue

        segments.append(
            {
                "x_bottom": float(
                    x_bottom
                ),
                "length": length,
                "points": [
                    (
                        float(x1),
                        float(y1),
                    ),
                    (
                        float(x2),
                        float(y2),
                    ),
                ],
            }
        )

    return segments


# ============================================================
# CLUSTER SEGMENTS
# ============================================================

def cluster_segments(
    segments,
    image_width: int,
):
    if not segments:
        return []

    ordered = sorted(
        segments,
        key=lambda segment: (
            segment["x_bottom"]
        ),
    )

    max_gap = (
        image_width * 0.10
    )

    clusters = []

    current = [
        ordered[0]
    ]

    for segment in ordered[1:]:

        current_center = float(
            np.median(
                [
                    item["x_bottom"]
                    for item in current
                ]
            )
        )

        if abs(
            segment["x_bottom"]
            - current_center
        ) <= max_gap:

            current.append(
                segment
            )

        else:

            clusters.append(
                current
            )

            current = [
                segment
            ]

    clusters.append(
        current
    )

    return clusters


# ============================================================
# FIT ONE LANE CLUSTER
# ============================================================

def fit_lane_cluster(
    cluster,
    width: int,
):
    points = []

    for segment in cluster:

        points.extend(
            segment["points"]
        )

        # Add midpoint too.
        p1, p2 = (
            segment["points"]
        )

        points.append(
            (
                (
                    p1[0]
                    + p2[0]
                )
                / 2.0,
                (
                    p1[1]
                    + p2[1]
                )
                / 2.0,
            )
        )

    return robust_curve_fit(
        np.asarray(points),
        width,
        min_points=6,
    )


# ============================================================
# EGO LANE FROM NN LANE MASK
# ============================================================

def extract_ego_lane(
    lane_mask: np.ndarray,
    road_mask: np.ndarray,
    road_left: CurveFit | None,
    road_right: CurveFit | None,
):
    h, w = lane_mask.shape

    camera_x = (
        w / 2.0
    )

    evaluation_y = (
        h * 0.92
    )

    clean = clean_lane_mask(
        lane_mask,
        road_mask,
    )

    segments = get_lane_segments(
        clean
    )

    clusters = cluster_segments(
        segments,
        w,
    )

    left_candidates = []
    right_candidates = []

    # Road envelope at bottom.
    road_left_x = None
    road_right_x = None

    if road_left is not None:

        road_left_x = float(
            np.polyval(
                road_left.coefficients,
                evaluation_y,
            )
        )

    if road_right is not None:

        road_right_x = float(
            np.polyval(
                road_right.coefficients,
                evaluation_y,
            )
        )

    for cluster in clusters:

        bottom_values = [
            segment["x_bottom"]
            for segment in cluster
        ]

        bottom_x = float(
            np.median(
                bottom_values
            )
        )

        total_length = sum(
            segment["length"]
            for segment in cluster
        )

        # Must be roughly inside the road.
        if (
            road_left_x is not None
            and bottom_x
            < road_left_x
            - w * 0.08
        ):
            continue

        if (
            road_right_x is not None
            and bottom_x
            > road_right_x
            + w * 0.08
        ):
            continue

        fit = fit_lane_cluster(
            cluster,
            w,
        )

        if fit is None:
            continue

        candidate = (
            bottom_x,
            total_length,
            fit,
        )

        dead_zone = (
            w * 0.025
        )

        if bottom_x < (
            camera_x
            - dead_zone
        ):

            left_candidates.append(
                candidate
            )

        elif bottom_x > (
            camera_x
            + dead_zone
        ):

            right_candidates.append(
                candidate
            )

    # Pick nearest lane boundary to camera center.
    left_fit = None
    right_fit = None

    if left_candidates:

        left_candidates.sort(
            key=lambda item: (
                abs(
                    camera_x
                    - item[0]
                )
            )
        )

        left_fit = (
            left_candidates[0][2]
        )

    if right_candidates:

        right_candidates.sort(
            key=lambda item: (
                abs(
                    item[0]
                    - camera_x
                )
            )
        )

        right_fit = (
            right_candidates[0][2]
        )

    return (
        left_fit,
        right_fit,
        clean,
    )


# ============================================================
# BUILD GEOMETRY
# ============================================================

def build_geometry(
    lane_mask: np.ndarray,
    road_mask: np.ndarray,
):
    h, w = lane_mask.shape

    (
        road_left,
        road_right,
        y_start,
        y_end,
    ) = extract_road_boundaries(
        road_mask
    )

    road_valid = False

    evaluation_y = (
        h * 0.92
    )

    if (
        road_left is not None
        and road_right is not None
    ):
        left_x = float(
            np.polyval(
                road_left.coefficients,
                evaluation_y,
            )
        )

        right_x = float(
            np.polyval(
                road_right.coefficients,
                evaluation_y,
            )
        )

        road_width = (
            right_x
            - left_x
        )

        if (
            road_width
            > w * 0.20
            and road_width
            < w * 1.10
        ):
            road_valid = True

    (
        lane_left,
        lane_right,
        cleaned_lane,
    ) = extract_ego_lane(
        lane_mask,
        road_mask,
        road_left,
        road_right,
    )

    ego_lane_valid = False
    normalized_offset = None

    if (
        lane_left is not None
        and lane_right is not None
    ):

        sample_y = np.linspace(
            h * 0.48,
            h * 0.94,
            20,
        )

        left_x = np.polyval(
            lane_left.coefficients,
            sample_y,
        )

        right_x = np.polyval(
            lane_right.coefficients,
            sample_y,
        )

        lane_widths = (
            right_x
            - left_x
        )

        valid_widths = (
            lane_widths
            > w * 0.08
        )

        if (
            valid_widths.sum()
            >= 16
        ):

            bottom_left = float(
                np.polyval(
                    lane_left.coefficients,
                    evaluation_y,
                )
            )

            bottom_right = float(
                np.polyval(
                    lane_right.coefficients,
                    evaluation_y,
                )
            )

            lane_width = (
                bottom_right
                - bottom_left
            )

            if (
                lane_width
                > w * 0.12
                and lane_width
                < w * 0.75
            ):

                lane_center = (
                    bottom_left
                    + bottom_right
                ) / 2.0

                camera_center = (
                    w / 2.0
                )

                normalized_offset = (
                    camera_center
                    - lane_center
                ) / (
                    lane_width / 2.0
                )

                ego_lane_valid = True

    result = GeometryResult(
        road_left=road_left,
        road_right=road_right,

        lane_left=lane_left,
        lane_right=lane_right,

        road_valid=road_valid,
        ego_lane_valid=ego_lane_valid,

        normalized_offset=(
            float(normalized_offset)
            if normalized_offset is not None
            else None
        ),

        left_lane_confidence=(
            lane_left.confidence
            if lane_left is not None
            else 0.0
        ),

        right_lane_confidence=(
            lane_right.confidence
            if lane_right is not None
            else 0.0
        ),
    )

    return (
        result,
        cleaned_lane,
        y_start,
        y_end,
    )


# ============================================================
# CURVE POINTS
# ============================================================

def curve_points(
    fit: CurveFit,
    y_start: float,
    y_end: float,
    width: int,
):
    ys = np.linspace(
        y_start,
        y_end,
        100,
    )

    xs = np.polyval(
        fit.coefficients,
        ys,
    )

    valid = (
        (xs >= 0)
        & (xs < width)
    )

    points = np.stack(
        [
            xs[valid],
            ys[valid],
        ],
        axis=1,
    )

    return np.round(
        points
    ).astype(np.int32)


# ============================================================
# RENDER
# ============================================================

def render(
    frame,
    road_mask,
    cleaned_lane,
    geometry,
    y_start,
    y_end,
    frame_number,
):
    output = frame.copy()

    h, w = output.shape[:2]

    # --------------------------------------------------------
    # Faint neural drivable area
    # --------------------------------------------------------

    road_pixels = (
        road_mask > 0
    )

    green = np.zeros_like(
        output
    )

    green[:] = (
        0,
        170,
        0,
    )

    output[
        road_pixels
    ] = cv2.addWeighted(
        output[
            road_pixels
        ],
        0.83,
        green[
            road_pixels
        ],
        0.17,
        0,
    )

    # --------------------------------------------------------
    # Clean lane pixels
    # --------------------------------------------------------

    lane_pixels = (
        cleaned_lane > 0
    )

    output[
        lane_pixels
    ] = (
        80,
        80,
        160,
    )

    # --------------------------------------------------------
    # Road corridor
    # --------------------------------------------------------

    road_left_pts = None
    road_right_pts = None

    if geometry.road_left is not None:

        road_left_pts = curve_points(
            geometry.road_left,
            y_start,
            y_end,
            w,
        )

        if len(road_left_pts) > 1:

            cv2.polylines(
                output,
                [road_left_pts],
                False,
                (255, 100, 0),
                3,
                cv2.LINE_AA,
            )

    if geometry.road_right is not None:

        road_right_pts = curve_points(
            geometry.road_right,
            y_start,
            y_end,
            w,
        )

        if len(road_right_pts) > 1:

            cv2.polylines(
                output,
                [road_right_pts],
                False,
                (255, 100, 0),
                3,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Ego lane fits
    # --------------------------------------------------------

    lane_y_start = (
        h * 0.42
    )

    lane_y_end = (
        h * 0.96
    )

    if geometry.lane_left is not None:

        points = curve_points(
            geometry.lane_left,
            lane_y_start,
            lane_y_end,
            w,
        )

        if len(points) > 1:

            cv2.polylines(
                output,
                [points],
                False,
                (255, 255, 0),
                5,
                cv2.LINE_AA,
            )

    if geometry.lane_right is not None:

        points = curve_points(
            geometry.lane_right,
            lane_y_start,
            lane_y_end,
            w,
        )

        if len(points) > 1:

            cv2.polylines(
                output,
                [points],
                False,
                (0, 255, 255),
                5,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Ego lane centerline
    # --------------------------------------------------------

    if geometry.ego_lane_valid:

        ys = np.linspace(
            lane_y_start,
            lane_y_end,
            100,
        )

        left_x = np.polyval(
            geometry.lane_left.coefficients,
            ys,
        )

        right_x = np.polyval(
            geometry.lane_right.coefficients,
            ys,
        )

        valid = (
            (left_x >= 0)
            & (left_x < w)
            & (right_x >= 0)
            & (right_x < w)
            & (right_x > left_x)
        )

        center_x = (
            left_x[valid]
            + right_x[valid]
        ) / 2.0

        center_y = (
            ys[valid]
        )

        if len(center_y) > 1:

            center_points = np.stack(
                [
                    center_x,
                    center_y,
                ],
                axis=1,
            )

            center_points = np.round(
                center_points
            ).astype(np.int32)

            cv2.polylines(
                output,
                [center_points],
                False,
                (255, 0, 255),
                4,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # Camera center
    # --------------------------------------------------------

    center = int(
        w / 2
    )

    cv2.line(
        output,
        (
            center,
            int(h * 0.82),
        ),
        (
            center,
            h - 1,
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    road_status = (
        "VALID"
        if geometry.road_valid
        else "NOT VALID"
    )

    lane_status = (
        "VALID"
        if geometry.ego_lane_valid
        else "NOT VALID"
    )

    cv2.putText(
        output,
        (
            f"frame {frame_number} "
            f"| road={road_status}"
        ),
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        output,
        (
            f"ego lane={lane_status} "
            f"| L={geometry.left_lane_confidence:.2f} "
            f"R={geometry.right_lane_confidence:.2f}"
        ),
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if geometry.normalized_offset is not None:

        cv2.putText(
            output,
            (
                "offset="
                f"{geometry.normalized_offset:+.2f}"
            ),
            (12, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# MONTAGE
# ============================================================

def montage(
    images,
    columns=3,
):
    tile_width = 480

    tiles = []

    for image in images:

        h, w = image.shape[:2]

        scale = (
            tile_width / w
        )

        new_height = int(
            round(
                h * scale
            )
        )

        tile = cv2.resize(
            image,
            (
                tile_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        tiles.append(
            tile
        )

    if not tiles:
        raise RuntimeError(
            "No preview images generated."
        )

    tile_height = max(
        tile.shape[0]
        for tile in tiles
    )

    normalized = []

    for tile in tiles:

        missing = (
            tile_height
            - tile.shape[0]
        )

        if missing > 0:

            tile = cv2.copyMakeBorder(
                tile,
                0,
                missing,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        normalized.append(
            tile
        )

    rows = []

    for index in range(
        0,
        len(normalized),
        columns,
    ):

        row = normalized[
            index:index + columns
        ]

        while len(row) < columns:

            row.append(
                np.zeros_like(
                    normalized[0]
                )
            )

        rows.append(
            cv2.hconcat(row)
        )

    return cv2.vconcat(
        rows
    )


# ============================================================
# RUN
# ============================================================

def run(args):
    if not args.model.is_file():
        raise FileNotFoundError(
            args.model
        )

    if not args.video.is_file():
        raise FileNotFoundError(
            args.video
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.setNumThreads(1)

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

    (
        _,
        channels,
        model_h,
        model_w,
    ) = input_spec.shape

    if channels != 3:
        raise RuntimeError(
            "Expected RGB model."
        )

    road_output, lane_output = (
        find_output_names(
            session
        )
    )

    cap = cv2.VideoCapture(
        str(args.video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open video."
        )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    positions = np.linspace(
        0.05,
        0.95,
        args.samples,
    )

    frame_numbers = [
        int(
            p
            * (total_frames - 1)
        )
        for p in positions
    ]

    previews = []

    road_valid_count = 0
    lane_valid_count = 0

    print()
    print("=" * 70)
    print("ROAD GEOMETRY V2")
    print("=" * 70)

    print(
        f"Model input: "
        f"{model_w}x{model_h}"
    )

    print()

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
            continue

        h, w = frame.shape[:2]

        tensor, crop = preprocess(
            frame,
            model_h,
            model_w,
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
            w,
            h,
        )

        lane_mask = decode_mask(
            outputs[1],
            crop,
            w,
            h,
        )

        (
            geometry,
            cleaned_lane,
            y_start,
            y_end,
        ) = build_geometry(
            lane_mask,
            road_mask,
        )

        if geometry.road_valid:
            road_valid_count += 1

        if geometry.ego_lane_valid:
            lane_valid_count += 1

        preview = render(
            frame,
            road_mask,
            cleaned_lane,
            geometry,
            y_start,
            y_end,
            frame_number,
        )

        previews.append(
            preview
        )

        output_path = (
            args.output_dir
            / (
                f"geometry_"
                f"{frame_number:06d}.jpg"
            )
        )

        cv2.imwrite(
            str(output_path),
            preview,
        )

        offset = (
            f"{geometry.normalized_offset:+.2f}"
            if geometry.normalized_offset is not None
            else "N/A"
        )

        print(
            f"[{sample_index:02d}/{args.samples:02d}] "
            f"frame={frame_number:<5d} "
            f"road={str(geometry.road_valid):<5} "
            f"ego={str(geometry.ego_lane_valid):<5} "
            f"L={geometry.left_lane_confidence:.2f} "
            f"R={geometry.right_lane_confidence:.2f} "
            f"offset={offset}"
        )

    cap.release()

    output_montage = montage(
        previews,
        columns=3,
    )

    montage_path = (
        args.output_dir
        / "geometry_v2_montage.jpg"
    )

    cv2.imwrite(
        str(montage_path),
        output_montage,
    )

    print()
    print("=" * 70)

    print(
        f"Road corridor valid: "
        f"{road_valid_count}/"
        f"{len(previews)}"
    )

    print(
        f"Ego lane valid:      "
        f"{lane_valid_count}/"
        f"{len(previews)}"
    )

    print(
        f"Montage: "
        f"{montage_path}"
    )

    print("=" * 70)


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
        "--threads",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "validation"
            / "road_geometry_v2"
        ),
    )

    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()