"""Schema validation and metric logic for the human lane acceptance set."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SCENE_TYPES = (
    "clear_normal_lane",
    "intersection",
    "parked_cars_curb",
    "crosswalk_stop_line",
    "faded_lane",
    "unmarked_road",
    "turning_curved_road",
    "occlusion",
)
EGO_STATUSES = ("valid", "invalid", "ambiguous")
BOUNDARY_STATUSES = ("visible", "not_visible", "ambiguous")
NO_VALID_EGO_LANE = "NO_VALID_EGO_LANE"
FALSE_LANE_COST = 3.0
BOUNDARY_SAMPLE_COUNT = 50


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace JSON atomically so an interrupted annotation cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_points(points: Any, width: int, height: int, name: str) -> list[str]:
    errors = []
    if not isinstance(points, list):
        return [f"{name}.points must be a list"]
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            errors.append(f"{name}.points[{index}] must be [x, y]")
            continue
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            errors.append(f"{name}.points[{index}] coordinates must be numeric")
        elif not (0 <= x < width and 0 <= y < height):
            errors.append(f"{name}.points[{index}] is outside {width}x{height}")
    return errors


def validate_annotation(
    annotation: dict[str, Any], width: int, height: int, require_complete: bool = True
) -> list[str]:
    errors = []
    frame_id = annotation.get("frame_id", "?")
    prefix = f"frame {frame_id}: "
    tags = annotation.get("scene_type")
    if not isinstance(tags, list) or not tags:
        errors.append(prefix + "scene_type must contain at least one tag")
    elif unknown := sorted(set(tags) - set(SCENE_TYPES)):
        errors.append(prefix + f"unknown scene tags: {unknown}")
    if not require_complete and annotation.get("review_status") != "complete":
        return errors
    if annotation.get("review_status") != "complete":
        errors.append(prefix + "review_status is not complete")
    ego_status = annotation.get("ego_lane_status")
    if ego_status not in EGO_STATUSES:
        errors.append(prefix + f"ego_lane_status must be one of {EGO_STATUSES}")
    boundaries = []
    for side in ("left_boundary", "right_boundary"):
        boundary = annotation.get(side)
        if not isinstance(boundary, dict):
            errors.append(prefix + f"{side} must be an object")
            continue
        boundaries.append(boundary)
        status = boundary.get("status")
        points = boundary.get("points")
        if status not in BOUNDARY_STATUSES:
            errors.append(prefix + f"{side}.status must be one of {BOUNDARY_STATUSES}")
        errors.extend(prefix + error for error in _validate_points(points, width, height, side))
        if status == "visible" and isinstance(points, list) and len(points) < 2:
            errors.append(prefix + f"{side} needs at least two points when visible")
        if status == "not_visible" and points:
            errors.append(prefix + f"{side} must have no points when not_visible")
        if status == "ambiguous" and points and len(points) < 2:
            errors.append(prefix + f"{side} needs zero or at least two points when ambiguous")
    marker = annotation.get("annotation_flag")
    if ego_status == "invalid":
        if marker != NO_VALID_EGO_LANE:
            errors.append(prefix + f"invalid ego lane must set {NO_VALID_EGO_LANE}")
        for side, boundary in zip(("left_boundary", "right_boundary"), boundaries):
            if boundary.get("status") != "not_visible" or boundary.get("points"):
                errors.append(prefix + f"invalid ego lane requires empty not_visible {side}")
    elif marker is not None:
        errors.append(prefix + "annotation_flag is only allowed for an invalid ego lane")
    if ego_status == "valid" and boundaries:
        if not any(boundary.get("status") == "visible" for boundary in boundaries):
            errors.append(prefix + "valid ego lane needs at least one visible boundary")
    return errors


def validate_dataset(dataset: dict[str, Any], require_complete: bool = True) -> list[str]:
    errors = []
    video = dataset.get("video", {})
    width, height = video.get("width"), video.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return ["dataset video width and height must be integers"]
    annotations = dataset.get("annotations")
    if not isinstance(annotations, list):
        return ["dataset annotations must be a list"]
    if len(annotations) != 100:
        errors.append(f"dataset must contain exactly 100 annotations, found {len(annotations)}")
    frame_ids = [item.get("frame_id") for item in annotations]
    if len(set(frame_ids)) != len(frame_ids):
        errors.append("dataset frame_id values must be unique")
    for annotation in annotations:
        errors.extend(validate_annotation(annotation, width, height, require_complete))
    return errors


def _prediction_for_missing(frame_id: int) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "ego_lane_status": "invalid",
        "left_boundary": {"status": "not_visible", "points": []},
        "right_boundary": {"status": "not_visible", "points": []},
        "_missing": True,
    }


def _prediction_errors(prediction: dict[str, Any], width: int, height: int) -> list[str]:
    errors = []
    frame_id = prediction.get("frame_id", "?")
    ego_status = prediction.get("ego_lane_status")
    if ego_status not in EGO_STATUSES:
        errors.append(f"prediction frame {frame_id}: invalid ego_lane_status")
    boundaries = []
    for side in ("left_boundary", "right_boundary"):
        boundary = prediction.get(side)
        if not isinstance(boundary, dict):
            errors.append(f"prediction frame {frame_id}: missing {side}")
            continue
        boundaries.append(boundary)
        if boundary.get("status") not in BOUNDARY_STATUSES:
            errors.append(f"prediction frame {frame_id}: invalid {side}.status")
        errors.extend(
            f"prediction frame {frame_id}: {error}"
            for error in _validate_points(boundary.get("points"), width, height, side)
        )
        if boundary.get("status") == "visible" and len(boundary.get("points", [])) < 2:
            errors.append(f"prediction frame {frame_id}: visible {side} needs two points")
        if boundary.get("status") == "not_visible" and boundary.get("points"):
            errors.append(f"prediction frame {frame_id}: not_visible {side} must have no points")
        if boundary.get("status") == "ambiguous" and len(boundary.get("points", [])) == 1:
            errors.append(f"prediction frame {frame_id}: ambiguous {side} needs zero or two points")
    if ego_status == "invalid" and boundaries:
        if any(item.get("status") != "not_visible" or item.get("points") for item in boundaries):
            errors.append(f"prediction frame {frame_id}: invalid ego lane must have empty boundaries")
    if ego_status == "valid" and boundaries:
        if not any(item.get("status") == "visible" for item in boundaries):
            errors.append(f"prediction frame {frame_id}: valid ego lane needs a visible boundary")
    return errors


def _polyline_error(
    truth_points: list[list[float]], prediction_points: list[list[float]], width: int
) -> tuple[np.ndarray, bool]:
    """Return x errors at 50 common image y values and whether y ranges overlap."""

    def prepare(points: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(points, dtype=np.float64)
        order = np.argsort(array[:, 1], kind="stable")
        array = array[order]
        unique_y = np.unique(array[:, 1])
        x = np.asarray([array[array[:, 1] == y, 0].mean() for y in unique_y])
        return unique_y, x

    truth_y, truth_x = prepare(truth_points)
    prediction_y, prediction_x = prepare(prediction_points)
    if len(truth_y) < 2 or len(prediction_y) < 2:
        return np.full(BOUNDARY_SAMPLE_COUNT, width, dtype=np.float64), False
    low = max(float(truth_y[0]), float(prediction_y[0]))
    high = min(float(truth_y[-1]), float(prediction_y[-1]))
    if high <= low:
        return np.full(BOUNDARY_SAMPLE_COUNT, width, dtype=np.float64), False
    sample_y = np.linspace(low, high, BOUNDARY_SAMPLE_COUNT)
    truth_sample_x = np.interp(sample_y, truth_y, truth_x)
    prediction_sample_x = np.interp(sample_y, prediction_y, prediction_x)
    return np.abs(truth_sample_x - prediction_sample_x), True


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _error_summary(values: list[float], width: int, matched: int, no_overlap: int) -> dict[str, Any]:
    if not values:
        return {
            "matched_boundaries": matched,
            "no_y_overlap_boundaries": no_overlap,
            "sample_count": 0,
            "pixels": None,
            "normalized_by_image_width": None,
        }
    data = np.asarray(values, dtype=np.float64)
    pixels = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "max": float(np.max(data)),
    }
    return {
        "matched_boundaries": matched,
        "no_y_overlap_boundaries": no_overlap,
        "sample_count": int(len(data)),
        "pixels": pixels,
        "normalized_by_image_width": {key: value / width for key, value in pixels.items()},
    }


def evaluate_predictions(dataset: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    dataset_errors = validate_dataset(dataset, require_complete=True)
    if dataset_errors:
        raise ValueError("Ground truth is not evaluation-ready:\n" + "\n".join(dataset_errors))
    width = dataset["video"]["width"]
    height = dataset["video"]["height"]
    prediction_items = predictions.get("predictions")
    if not isinstance(prediction_items, list):
        raise ValueError("predictions must contain a predictions list")
    prediction_by_id = {}
    errors = []
    for prediction in prediction_items:
        frame_id = prediction.get("frame_id")
        if frame_id in prediction_by_id:
            errors.append(f"duplicate prediction for frame {frame_id}")
        prediction_by_id[frame_id] = prediction
        errors.extend(_prediction_errors(prediction, width, height))
    if errors:
        raise ValueError("Invalid predictions:\n" + "\n".join(errors))

    counts = {
        "frames": len(dataset["annotations"]),
        "ground_truth_valid": 0,
        "ground_truth_no_valid_lane": 0,
        "valid_lane_detected": 0,
        "false_lane": 0,
        "missed_lane": 0,
        "abstained": 0,
        "correct_abstain": 0,
        "missing_predictions": 0,
    }
    boundary = {
        side: {"truth_visible": 0, "detected": 0, "errors": [], "matched": 0, "no_overlap": 0}
        for side in ("left", "right")
    }
    scene_counts = {
        scene: {"eligible_no_valid_lane": 0, "false_lane": 0}
        for scene in (
            "intersection",
            "unmarked_road",
            "crosswalk_stop_line",
            "parked_cars_curb",
        )
    }
    ground_truth_ids = set()
    for truth in dataset["annotations"]:
        frame_id = truth["frame_id"]
        ground_truth_ids.add(frame_id)
        prediction = prediction_by_id.get(frame_id)
        if prediction is None:
            prediction = _prediction_for_missing(frame_id)
            counts["missing_predictions"] += 1
        truth_valid = truth["ego_lane_status"] == "valid"
        prediction_valid = prediction["ego_lane_status"] == "valid"
        if truth_valid:
            counts["ground_truth_valid"] += 1
            if prediction_valid:
                counts["valid_lane_detected"] += 1
            else:
                counts["missed_lane"] += 1
        else:
            counts["ground_truth_no_valid_lane"] += 1
            if prediction_valid:
                counts["false_lane"] += 1
            else:
                counts["correct_abstain"] += 1
        if not prediction_valid:
            counts["abstained"] += 1
        for scene, scene_metric in scene_counts.items():
            if scene in truth["scene_type"] and not truth_valid:
                scene_metric["eligible_no_valid_lane"] += 1
                if prediction_valid:
                    scene_metric["false_lane"] += 1
        for side in ("left", "right"):
            truth_boundary = truth[f"{side}_boundary"]
            predicted_boundary = prediction[f"{side}_boundary"]
            state = boundary[side]
            if truth_boundary["status"] != "visible":
                continue
            state["truth_visible"] += 1
            if predicted_boundary["status"] != "visible":
                continue
            state["detected"] += 1
            values, overlaps = _polyline_error(
                truth_boundary["points"], predicted_boundary["points"], width
            )
            state["matched"] += 1
            if not overlaps:
                state["no_overlap"] += 1
            state["errors"].extend(values.tolist())

    false_weighted = FALSE_LANE_COST * counts["false_lane"]
    weighted_denominator = (
        FALSE_LANE_COST * counts["ground_truth_no_valid_lane"]
        + counts["ground_truth_valid"]
    )
    metrics = {
        "valid_ego_lane_detection_rate": _safe_rate(
            counts["valid_lane_detected"], counts["ground_truth_valid"]
        ),
        "false_ego_lane_rate": _safe_rate(
            counts["false_lane"], counts["ground_truth_no_valid_lane"]
        ),
        "missed_ego_lane_rate": _safe_rate(
            counts["missed_lane"], counts["ground_truth_valid"]
        ),
        "abstain_rate": _safe_rate(counts["abstained"], counts["frames"]),
        "correct_abstain_rate": _safe_rate(
            counts["correct_abstain"], counts["ground_truth_no_valid_lane"]
        ),
        "safety_weighted_error_rate": (
            (false_weighted + counts["missed_lane"]) / weighted_denominator
            if weighted_denominator
            else None
        ),
        "false_lane_cost_weight": FALSE_LANE_COST,
    }
    for side in ("left", "right"):
        state = boundary[side]
        metrics[f"{side}_boundary_detection_rate"] = _safe_rate(
            state["detected"], state["truth_visible"]
        )
        metrics[f"{side}_boundary_error"] = _error_summary(
            state["errors"], width, state["matched"], state["no_overlap"]
        )
    for scene, scene_metric in scene_counts.items():
        scene_metric["false_positive_rate"] = _safe_rate(
            scene_metric["false_lane"], scene_metric["eligible_no_valid_lane"]
        )
    extra_ids = sorted(set(prediction_by_id) - ground_truth_ids)
    return {
        "model": predictions.get("model", {"name": "unnamed"}),
        "dataset": dataset.get("name"),
        "counts": counts,
        "metrics": metrics,
        "scene_false_positive_rates": scene_counts,
        "extra_prediction_frame_ids": extra_ids,
        "policy": {
            "ambiguous_ground_truth_requires_abstention": True,
            "predicted_invalid_or_ambiguous_counts_as_abstention": True,
            "false_lane_cost_relative_to_missed_lane": FALSE_LANE_COST,
            "boundary_samples_per_match": BOUNDARY_SAMPLE_COUNT,
            "boundary_units": "image pixels and fraction of image width",
        },
    }


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.2f}%"


def format_number(value: float | None, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:.3f}{suffix}"


def report_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    counts = result["counts"]
    model_name = result.get("model", {}).get("name", "unnamed")
    lines = [
        f"# Lane acceptance report: {model_name}",
        "",
        "False ego-lane predictions carry three times the cost of missed lanes.",
        "Ambiguous ground truth is expected to produce abstention.",
        "",
        "## Ego-lane decision metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Valid ego-lane detection rate | {format_rate(metrics['valid_ego_lane_detection_rate'])} |",
        f"| False ego-lane rate | {format_rate(metrics['false_ego_lane_rate'])} |",
        f"| Missed ego-lane rate | {format_rate(metrics['missed_ego_lane_rate'])} |",
        f"| Abstain rate | {format_rate(metrics['abstain_rate'])} |",
        f"| Correct abstain rate | {format_rate(metrics['correct_abstain_rate'])} |",
        f"| Safety-weighted error rate | {format_rate(metrics['safety_weighted_error_rate'])} |",
        "",
        f"Evaluated {counts['frames']} frames: {counts['ground_truth_valid']} valid and "
        f"{counts['ground_truth_no_valid_lane']} invalid/ambiguous. Missing predictions "
        f"treated as abstentions: {counts['missing_predictions']}.",
        "",
        "## Boundary metrics",
        "",
        "| Boundary | Detection | Mean error | P95 error | Mean / width | P95 / width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for side in ("left", "right"):
        error = metrics[f"{side}_boundary_error"]
        pixels = error["pixels"] or {}
        normalized = error["normalized_by_image_width"] or {}
        lines.append(
            f"| {side.title()} | {format_rate(metrics[f'{side}_boundary_detection_rate'])} | "
            f"{format_number(pixels.get('mean'), ' px')} | {format_number(pixels.get('p95'), ' px')} | "
            f"{format_number(normalized.get('mean'))} | {format_number(normalized.get('p95'))} |"
        )
    lines.extend(
        [
            "",
            "Errors are sampled at common image y positions. They are image-space values, not meters.",
            "",
            "## False positives by difficult scene",
            "",
            "| Scene | False positives | Eligible invalid/ambiguous | Rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for scene, item in result["scene_false_positive_rates"].items():
        lines.append(
            f"| {scene} | {item['false_lane']} | {item['eligible_no_valid_lane']} | "
            f"{format_rate(item['false_positive_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Raw decision counts",
            "",
            "```json",
            json.dumps(counts, indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)
