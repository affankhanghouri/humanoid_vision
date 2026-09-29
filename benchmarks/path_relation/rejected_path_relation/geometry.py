"""Native-mask geometry for road support and a conservative forward corridor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CorridorGeometry:
    mask: np.ndarray
    quality: float
    row_bounds: tuple[tuple[int, int] | None, ...]


def image_to_mask(point: tuple[float, float], image_size: tuple[int, int],
                  content_rect: tuple[int, int, int, int],
                  mask_shape: tuple[int, int]) -> tuple[int, int] | None:
    """Map an image point into the padded native road-mask coordinates."""
    x, y = point
    image_width, image_height = image_size
    left, top, width, height = content_rect
    mask_height, mask_width = mask_shape
    if (image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0
            or mask_width <= 0 or mask_height <= 0
            or not all(math.isfinite(value) for value in (x, y))):
        return None
    x = min(max(x, 0.0), image_width - 1.0)
    y = min(max(y, 0.0), image_height - 1.0)
    mx = left + x / max(image_width - 1.0, 1.0) * max(width - 1, 0)
    my = top + y / max(image_height - 1.0, 1.0) * max(height - 1, 0)
    return (int(round(min(max(mx, 0.0), mask_width - 1.0))),
            int(round(min(max(my, 0.0), mask_height - 1.0))))


def contact_point(box: tuple[float, float, float, float],
                  image_size: tuple[int, int]) -> tuple[float, float] | None:
    """Return the safely clamped bottom-center of a valid image-space box."""
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        return None
    x1, y1, x2, y2 = box
    width, height = image_size
    if width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
        return None
    return (float(np.clip((x1 + x2) * .5, 0, width - 1)),
            float(np.clip(y2, 0, height - 1)))


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    groups = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    return [(int(group[0]), int(group[-1])) for group in groups if group.size]


def construct_forward_corridor(drivable_mask: np.ndarray,
                               content_rect: tuple[int, int, int, int]) -> CorridorGeometry | None:
    """Select the lower-middle connected road and keep its conservative center band."""
    mask = (np.asarray(drivable_mask) > 0).astype(np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        return None
    left, top, width, height = content_rect
    if width <= 0 or height <= 0:
        return None
    content = mask[top:top + height, left:left + width]
    if content.shape != (height, width) or not np.any(content):
        return None

    seed = np.zeros_like(content, dtype=bool)
    seed_y1, seed_y2 = int(height * .72), max(int(height * .96), 1)
    seed_x1, seed_x2 = int(width * .42), max(int(width * .58), 1)
    seed[seed_y1:seed_y2, seed_x1:seed_x2] = True
    candidates = np.argwhere((content > 0) & seed)
    if candidates.size == 0:
        return None
    anchor = np.asarray((height * .84, width * .5))
    sy, sx = candidates[np.argmin(np.sum((candidates - anchor) ** 2, axis=1))]
    filled = content.copy()
    cv2.floodFill(filled, np.zeros((height + 2, width + 2), np.uint8),
                  (int(sx), int(sy)), 2, flags=8)
    component = filled == 2
    best_overlap = int(np.count_nonzero(component & seed))
    min_seed = max(6, int(seed.sum() * .008))
    if best_overlap < min_seed:
        return None
    area_fraction = float(component.mean())
    if area_fraction < .025:
        return None

    # Vectorized row envelopes avoid hundreds of tiny per-row allocations.
    row_has = np.any(component, axis=1)
    row_min = np.argmax(component, axis=1).astype(np.int32)
    row_max = (width - 1 - np.argmax(component[:, ::-1], axis=1)).astype(np.int32)
    run_width = np.where(row_has, row_max - row_min + 1, 0)
    usable = run_width >= 6
    centers = (row_min + row_max) * .5
    half = np.maximum(1, (run_width * .18).astype(np.int32))
    grid_x = np.arange(width, dtype=np.int32)[None, :]
    rounded = np.rint(centers).astype(np.int32)
    corridor_content = (component & usable[:, None]
                        & (grid_x >= (rounded - half)[:, None])
                        & (grid_x <= (rounded + half)[:, None])).astype(np.uint8)
    corridor_has = np.any(corridor_content, axis=1)
    corridor_min = np.argmax(corridor_content, axis=1).astype(np.int32)
    corridor_max = (width - 1 - np.argmax(corridor_content[:, ::-1], axis=1)).astype(np.int32)
    bounds: list[tuple[int, int] | None] = [None] * mask.shape[0]
    valid_rows = int(np.count_nonzero(corridor_has))
    for cy in np.flatnonzero(corridor_has):
        bounds[top + int(cy)] = (left + int(corridor_min[cy]),
                                 left + int(corridor_max[cy]))
    row_coverage = valid_rows / max(height, 1)
    if row_coverage < .25 or not np.any(corridor_content):
        return None
    output = np.zeros_like(mask)
    output[top:top + height, left:left + width] = corridor_content
    output.setflags(write=False)
    seed_density = best_overlap / max(int(seed.sum()), 1)
    quality = float(np.clip(.50 * min(row_coverage / .65, 1.)
                            + .30 * min(area_fraction / .30, 1.)
                            + .20 * min(seed_density / .45, 1.), 0., 1.))
    return CorridorGeometry(output, quality, tuple(bounds))


def patch_fraction(mask: np.ndarray, center: tuple[int, int], radii: tuple[int, int]) -> float | None:
    """Measure binary support in a clipped ellipse without resizing the mask."""
    height, width = mask.shape
    cx, cy = center
    rx, ry = max(1, int(radii[0])), max(1, int(radii[1]))
    x1, x2 = max(0, cx - rx), min(width - 1, cx + rx)
    y1, y2 = max(0, cy - ry), min(height - 1, cy + ry)
    if x2 < x1 or y2 < y1:
        return None
    yy, xx = np.ogrid[y1:y2 + 1, x1:x2 + 1]
    ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    if not np.any(ellipse):
        return None
    return float(np.count_nonzero((mask[y1:y2 + 1, x1:x2 + 1] > 0) & ellipse)
                 / np.count_nonzero(ellipse))


def integral_mask(mask: np.ndarray) -> np.ndarray:
    """Build a summed-area table for repeated constant-time foot-patch queries."""
    return cv2.integral((np.asarray(mask) > 0).astype(np.uint8), sdepth=cv2.CV_32S)


def integral_patch_fraction(integral: np.ndarray, center: tuple[int, int],
                            radii: tuple[int, int],
                            mask_shape: tuple[int, int]) -> float | None:
    """Measure a clipped rectangular patch from a cached summed-area table."""
    height, width = mask_shape
    cx, cy = center
    rx, ry = max(1, int(radii[0])), max(1, int(radii[1]))
    x1, x2 = max(0, cx - rx), min(width - 1, cx + rx)
    y1, y2 = max(0, cy - ry), min(height - 1, cy + ry)
    if x2 < x1 or y2 < y1:
        return None
    total = (integral[y2 + 1, x2 + 1] - integral[y1, x2 + 1]
             - integral[y2 + 1, x1] + integral[y1, x1])
    return float(total / ((x2 - x1 + 1) * (y2 - y1 + 1)))


def box_patch_radii(box: tuple[float, float, float, float], image_size: tuple[int, int],
                    content_rect: tuple[int, int, int, int]) -> tuple[int, int]:
    image_width, image_height = image_size
    _, _, content_width, content_height = content_rect
    box_width = max(1., box[2] - box[0])
    box_height = max(1., box[3] - box[1])
    return (max(2, int(round(box_width * .18 * content_width / max(image_width, 1)))),
            max(2, int(round(box_height * .06 * content_height / max(image_height, 1)))))


def corridor_distance_and_side(point: tuple[int, int], corridor: CorridorGeometry,
                               image_width: int, content_width: int) -> tuple[float, int] | None:
    x, y = point
    if y < 0 or y >= len(corridor.row_bounds):
        return None
    bounds = corridor.row_bounds[y]
    if bounds is None:
        return None
    left, right = bounds
    scale = image_width / max(content_width, 1)
    if x < left:
        return (float((left - x) * scale), -1)
    if x > right:
        return (float((x - right) * scale), 1)
    return (0.0, 0)
