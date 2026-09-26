"""Reusable drawing functions for demo mode."""

import math

import cv2
import numpy as np


COCO_SKELETON = (
    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (0, 1),
    (0, 2),

    (1, 3),
    (2, 4),
)


def dark_panel(frame, x, y, width, height, alpha=0.78, border_color=None,
               background=(7, 13, 20)):
    """Blend only the visible panel, never a full-frame copy per label."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + width)), min(h, int(y + height))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    fill = np.empty_like(roi)
    cv2.rectangle(fill, (0, 0), (x2-x1, y2-y1), background, -1)
    cv2.addWeighted(roi, 1.0-alpha, fill, alpha, 0, dst=roi)
    if border_color is not None:
        cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), border_color, 1, cv2.LINE_AA)


def draw_corner_box(frame, box, color, thickness=2):
    x1, y1, x2, y2 = box
    corner = max(8, min(30, int(min(max(1, x2-x1), max(1, y2-y1)) * .25)))
    # Four connected corners use one array conversion and two draw calls.
    paths = np.asarray((
        ((x1+corner, y1), (x1, y1), (x1, y1+corner)),
        ((x2-corner, y1), (x2, y1), (x2, y1+corner)),
        ((x1+corner, y2), (x1, y2), (x1, y2-corner)),
        ((x2-corner, y2), (x2, y2), (x2, y2-corner)),
    ), dtype=np.int32)
    glow = tuple(int(channel * .30) for channel in color)
    cv2.polylines(frame, paths, False, glow, thickness+4, cv2.LINE_AA)
    cv2.polylines(frame, paths, False, color, thickness, cv2.LINE_AA)


def draw_entity_label(
    frame,
    box,
    entity,
    color,
    pose_ready,
):

    x1, y1, _, _ = box

    title = (
        f"{entity.class_name.upper()} "
        f"#{entity.entity_id}  "
        f"{entity.confidence:.2f}"
    )

    if pose_ready:

        status = (
            "TRACKED  |  POSE READY"
        )

    else:

        status = "TRACKED"

    if entity.misses:
        status = "PREDICTED"

    text_size = cv2.getTextSize(
        title,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        1,
    )[0]

    panel_width = max(
        155,
        text_size[0] + 18,
    )

    panel_height = 46

    panel_y = max(
        0,
        y1 - panel_height - 4,
    )

    dark_panel(
        frame,
        x1,
        panel_y,
        panel_width,
        panel_height,
        alpha=0.82,
        border_color=color,
    )

    cv2.putText(
        frame,
        title,
        (
            x1 + 7,
            panel_y + 18,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status,
        (
            x1 + 7,
            panel_y + 36,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (
            220,
            230,
            235,
        ),
        1,
        cv2.LINE_AA,
    )


def draw_motion_trail(frame, trail, color):
    if len(trail) < 2:
        return
    points = np.asarray(trail, dtype=np.int32)
    # Batch the fading trail into four bands instead of a Python/OpenCV call
    # for each historical segment, retaining the same path and recent markers.
    segments = len(points) - 1
    bands = min(4, segments)
    for band in range(bands):
        start = band * segments // bands
        end = (band + 1) * segments // bands
        strength = ((start + end + 1) / 2) / len(points)
        faded = tuple(int(channel * (.20 + strength * .80)) for channel in color)
        cv2.polylines(frame, [points[start:end+1]], False, faded, 2, cv2.LINE_AA)
    for point in set(map(tuple, points[-5:])):
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)


def draw_velocity_arrow(
    frame,
    center,
    velocity_x,
    velocity_y,
    color,
):

    speed = math.hypot(
        velocity_x,
        velocity_y,
    )

    if speed < 10.0:
        return

    scale = (
        65.0
        / max(
            speed,
            1.0,
        )
    )

    end_x = int(
        center[0]
        + velocity_x
        * scale
    )

    end_y = int(
        center[1]
        + velocity_y
        * scale
    )

    cv2.arrowedLine(
        frame,
        center,
        (
            end_x,
            end_y,
        ),
        color,
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def draw_pose(
    frame,
    entity,
    box,
    timestamp,
    pose_smoother,
    config,
    color,
    frame_id,
):

    pose = entity.pose

    if pose is None:
        return False

    if not pose.meta.is_valid_for(frame_id, timestamp, config.pose_max_age):
        return False

    x1, y1, x2, y2 = box

    box_width = max(
        1,
        x2 - x1,
    )

    box_height = max(
        1,
        y2 - y1,
    )

    relative_points = (
        pose_smoother.smooth(
            entity.entity_id,
            pose.keypoints,
        )
    )

    points = {}

    for (
        index,
        relative_point,
    ) in relative_points.items():

        relative_x, relative_y = (
            relative_point
        )

        x = int(
            x1
            + relative_x
            * box_width
        )

        y = int(
            y1
            + relative_y
            * box_height
        )

        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            points[index] = (x, y)

    # ------------------------------
    # skeleton
    # ------------------------------

    for (
        start,
        end,
    ) in COCO_SKELETON:

        if (
            start not in points
            or end not in points
        ):
            continue

        cv2.line(
            frame,
            points[start],
            points[end],
            (
                int(
                    color[0]
                    * 0.30
                ),
                int(
                    color[1]
                    * 0.30
                ),
                int(
                    color[2]
                    * 0.30
                ),
            ),
            6,
            cv2.LINE_AA,
        )

        cv2.line(
            frame,
            points[start],
            points[end],
            color,
            2,
            cv2.LINE_AA,
        )

    # ------------------------------
    # joints
    # ------------------------------

    for point in points.values():

        cv2.circle(
            frame,
            point,
            6,
            (
                int(
                    color[0]
                    * 0.25
                ),
                int(
                    color[1]
                    * 0.25
                ),
                int(
                    color[2]
                    * 0.25
                ),
            ),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            frame,
            point,
            3,
            color,
            -1,
            cv2.LINE_AA,
        )

    return True
