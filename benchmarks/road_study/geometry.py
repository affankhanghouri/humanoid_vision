"""Conservative image-space geometry for standalone LSTR experiments only.

Curve representation follows https://github.com/liuruijin17/LSTR .
No road edge is substituted for a lane. Confidence is a model score, not calibration.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Lane:
    query: int
    confidence: float
    points: np.ndarray  # normalized x,y, sorted by y


def decode_lstr(logits, curves, threshold=0.7):
    logits, curves = np.asarray(logits), np.asarray(curves)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] != 2:
        raise ValueError(f"Unexpected LSTR logits: {logits.shape}")
    if curves.shape != (*logits.shape[:2], 8):
        raise ValueError(f"Unexpected LSTR curves: {curves.shape}")
    if not np.isfinite(logits).all() or not np.isfinite(curves).all():
        return []
    scores = np.exp(logits[0] - logits[0].max(axis=-1, keepdims=True))
    scores = scores[:, 1] / scores.sum(axis=-1)
    lanes = []
    for query, (score, curve) in enumerate(zip(scores, curves[0])):
        if score < threshold:
            continue
        lower, upper, k, f, m, n, b, c = curve
        lo, hi = max(0., float(lower)), min(1., float(upper))
        if hi - lo < 0.15 or lo <= f <= hi:
            continue
        y = np.linspace(lo, hi, 64)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = k / (y - f)**2 + m / (y - f) + n + b*y - c
        valid = np.isfinite(x) & (x >= 0) & (x <= 1)
        # Keep only the longest continuous visible part; never bridge a pole.
        runs = np.split(np.flatnonzero(valid), np.flatnonzero(np.diff(np.flatnonzero(valid)) > 1) + 1)
        indices = max(runs, key=len)
        if len(indices) >= 8 and y[indices[-1]] - y[indices[0]] >= .15:
            lanes.append(Lane(query, float(score), np.column_stack((x[indices], y[indices]))))
    return lanes


def select_ego(lanes, road_mask=None, reference_y=.9):
    """Return geometry only with two observed, plausible boundaries.

    road_mask is unpadded drivable support at any resolution. Without that
    independent observation, the result remains a candidate, not validated.
    """
    candidates = [(float(np.interp(reference_y, p.points[:, 1], p.points[:, 0])), p)
                  for p in lanes if p.points[0, 1] <= reference_y <= p.points[-1, 1]]
    left = [(x, p) for x, p in candidates if x < .5]
    right = [(x, p) for x, p in candidates if x > .5]
    if not left or not right:
        return {"valid": False, "reason": "missing_boundary"}
    lx, l = max(left, key=lambda t: t[0])
    rx, r = min(right, key=lambda t: t[0])
    ymin = max(.55, l.points[0, 1], r.points[0, 1])
    if reference_y - ymin < .2:
        return {"valid": False, "reason": "insufficient_overlap"}
    y = np.linspace(ymin, reference_y, 24)
    xs = [np.interp(y, p.points[:, 1], p.points[:, 0]) for p in (l, r)]
    width = xs[1] - xs[0]
    if not .15 <= rx-lx <= .85 or np.any(width < .025):
        return {"valid": False, "reason": "width_or_crossing"}
    if any(np.max(np.abs(np.diff(x, n=2))) > .025 for x in xs):
        return {"valid": False, "reason": "curvature"}
    result = {"valid": False, "reason": "road_support_missing", "left_query": l.query,
              "right_query": r.query, "confidence": min(l.confidence, r.confidence),
              "normalized_offset": float((.5 - (lx+rx)/2) / ((rx-lx)/2)),
              "left": np.column_stack((xs[0], y)).tolist(),
              "right": np.column_stack((xs[1], y)).tolist()}
    if road_mask is not None:
        h, w = road_mask.shape
        # Interior support, not lane-marking pixels or the full road's outer edges.
        xx = xs[0][:, None] + width[:, None] * np.linspace(.15, .85, 7)
        yy = np.broadcast_to(y[:, None], xx.shape)
        support = road_mask[np.clip((yy*h).astype(int), 0, h-1),
                            np.clip((xx*w).astype(int), 0, w-1)].mean()
        result.update(valid=bool(support >= .85), reason="accepted" if support >= .85 else "road_support",
                      road_support=float(support))
    return result
