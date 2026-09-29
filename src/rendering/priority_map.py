"""Demo-only perception priority visualization from real tracked state.

This is an engineering relevance display, not a human-attention estimate and
not a calibrated collision-risk predictor.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

PEDESTRIANS = {'person'}
CYCLISTS = {'bicycle', 'motorcycle'}
VEHICLES = {'car', 'truck', 'bus', 'motorcycle'}


def fresh_road_mask(observation, frame_id: int, timestamp: float,
                    frame_size: tuple[int, int], max_age: float):
    """Return a frame-sized mask only when source metadata is still fresh."""
    if observation is None or not observation.is_valid_for(frame_id, timestamp, max_age):
        return None
    frame_width, frame_height = frame_size
    left, top, width, height = observation.content_rect
    native = observation.drivable_mask
    if width <= 0 or height <= 0:
        return None
    content = native[top:top + height, left:left + width]
    if content.size == 0:
        return None
    return cv2.resize(content, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)


def corridor_center(mask: np.ndarray | None, y: int, fallback: float) -> float:
    if mask is None:
        return fallback
    row = mask[max(0, min(mask.shape[0] - 1, int(y)))] > 0
    xs = np.flatnonzero(row)
    if xs.size < 4:
        return fallback
    # Use the connected drivable run containing the image center when possible.
    splits = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    center = mask.shape[1] / 2
    run = min(splits, key=lambda values: 0 if values[0] <= center <= values[-1]
              else min(abs(values[0] - center), abs(values[-1] - center)))
    return float((run[0] + run[-1]) * .5)


def entity_priority(entity, box, frame_size: tuple[int, int], path_center_x: float):
    """Return a bounded score and transparent component breakdown.

    Velocity is image-space track velocity. Positive vertical motion and lateral
    motion toward the current corridor center raise priority; neither is treated
    as physical speed or time-to-collision.
    """
    width, height = frame_size
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    area_fraction = max(0., x2 - x1) * max(0., y2 - y1) / max(1., width * height)
    class_name = entity.class_name.lower()
    class_relevance = (1.0 if class_name in PEDESTRIANS else
                       .88 if class_name in CYCLISTS else
                       .68 if class_name in VEHICLES else .38)
    size_closeness = min(1., math.sqrt(area_fraction / .08))
    vertical_closeness = float(np.clip((cy / max(height, 1) - .25) / .65, 0., 1.))
    closeness = .55 * size_closeness + .45 * vertical_closeness
    normalized_offset = abs(cx - path_center_x) / max(width * .24, 1.)
    path_relevance = math.exp(-.5 * normalized_offset * normalized_offset)
    direction = 1. if path_center_x >= cx else -1.
    lateral_toward = float(np.clip(entity.velocity_x * direction / max(width * .14, 1.), 0., 1.))
    approaching = float(np.clip(entity.velocity_y / max(height * .18, 1.), 0., 1.))
    motion_toward_path = .65 * approaching + .35 * lateral_toward
    confidence = float(np.clip(entity.confidence, 0., 1.))
    score = (.04 + .28 * class_relevance + .23 * closeness +
             .25 * path_relevance + .15 * motion_toward_path + .05 * confidence)
    if entity.misses:
        score *= .72
    components = {
        'class_relevance': class_relevance,
        'closeness': closeness,
        'path_relevance': path_relevance,
        'motion_toward_path': motion_toward_path,
        'confidence': confidence,
    }
    return float(np.clip(score, 0., 1.)), components


def _add_gaussian(canvas, center, sigma_x, sigma_y, strength):
    height, width = canvas.shape
    cx, cy = center
    radius_x, radius_y = max(2, int(3 * sigma_x)), max(2, int(3 * sigma_y))
    x1, x2 = max(0, cx - radius_x), min(width, cx + radius_x + 1)
    y1, y2 = max(0, cy - radius_y), min(height, cy + radius_y + 1)
    if x2 <= x1 or y2 <= y1:
        return
    xs = (np.arange(x1, x2, dtype=np.float32) - cx) / max(sigma_x, 1.)
    ys = (np.arange(y1, y2, dtype=np.float32) - cy) / max(sigma_y, 1.)
    blob = np.exp(-.5 * (ys[:, None] ** 2 + xs[None, :] ** 2)) * strength
    np.maximum(canvas[y1:y2, x1:x2], blob, out=canvas[y1:y2, x1:x2])


def render_priority_overlay(frame: np.ndarray, views, road_mask: np.ndarray | None,
                            alpha: float = .30, scale: int = 4):
    """Blend soft green/yellow/red blobs and return per-entity scores."""
    height, width = frame.shape[:2]
    small_width = max(1, width // scale)
    small_height = max(1, height // scale)
    heat = np.zeros((small_height, small_width), np.float32)
    scored = []
    for view in views:
        x1, y1, x2, y2 = view.box
        center_x = corridor_center(road_mask, y2, width * .5)
        score, components = entity_priority(view.entity, view.box, (width, height), center_x)
        cx = int(((x1 + x2) * .5) / scale)
        cy = int((y1 * .35 + y2 * .65) / scale)
        box_width = max(1., x2 - x1) / scale
        box_height = max(1., y2 - y1) / scale
        _add_gaussian(heat, (cx, cy), max(7., box_width * .75),
                      max(6., box_height * .48), score)
        # Real tracker velocity supplies a short, softer predicted focus lobe.
        future_x = int(cx + view.entity.velocity_x * .35 / scale)
        future_y = int(cy + view.entity.velocity_y * .35 / scale)
        _add_gaussian(heat, (future_x, future_y), max(6., box_width * .55),
                      max(5., box_height * .38), score * .58)
        scored.append((view.entity.entity_id, score, components))
    if not np.any(heat > .025):
        return scored
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=2.2, sigmaY=2.2)
    heat = np.clip(heat, 0., 1.)
    # Low=green, medium=yellow, high=red in OpenCV BGR order.
    color = np.zeros((small_height, small_width, 3), np.float32)
    low = heat <= .55
    ratio = np.clip(heat / .55, 0., 1.)
    color[..., 0] = np.where(low, 45 * (1 - ratio), 0)
    color[..., 1] = np.where(low, 190 + 45 * ratio, 235 * (1 - (heat - .55) / .45))
    color[..., 2] = np.where(low, 35 + 220 * ratio, 255)
    color = cv2.resize(color.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    intensity = cv2.resize((heat * 255).astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_LINEAR)
    # One native blend keeps the demo real-time. Gaussian color/intensity still
    # produces a soft green/yellow/red field; a low cutoff confines the overlay.
    mask = cv2.compare(intensity, 5, cv2.CMP_GE)
    points = cv2.findNonZero(mask)
    if points is not None:
        x, y, roi_width, roi_height = cv2.boundingRect(points)
        frame_roi = frame[y:y+roi_height, x:x+roi_width]
        color_roi = color[y:y+roi_height, x:x+roi_width]
        mask_roi = mask[y:y+roi_height, x:x+roi_width]
        amount = alpha * .72
        blended = cv2.addWeighted(frame_roi, 1.0 - amount, color_roi, amount, 0)
        cv2.copyTo(blended, mask_roi, frame_roi)
    return scored


def render_road_overlay(frame: np.ndarray, road_mask: np.ndarray | None, alpha: float = .10):
    if road_mask is None:
        return
    mask = cv2.compare(road_mask, 0, cv2.CMP_GT)
    points = cv2.findNonZero(mask)
    if points is None:
        return
    x, y, roi_width, roi_height = cv2.boundingRect(points)
    frame_roi = frame[y:y+roi_height, x:x+roi_width]
    mask_roi = mask[y:y+roi_height, x:x+roi_width]
    tint = np.empty_like(frame_roi)
    tint[:] = (55, 145, 45)
    blended = cv2.addWeighted(frame_roi, 1.0 - alpha, tint, alpha, 0)
    cv2.copyTo(blended, mask_roi, frame_roi)
