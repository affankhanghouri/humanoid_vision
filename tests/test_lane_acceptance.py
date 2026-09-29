"""Safety-weighted decision and image-space lane metric tests."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/lane_acceptance"))
from common import NO_VALID_EGO_LANE, evaluate_predictions, validate_annotation


WIDTH = 1152
HEIGHT = 720


def boundary(status="visible", offset=0, y_values=(700, 500, 300)):
    points = [[300 + offset, y] for y in y_values] if status == "visible" else []
    return {"status": status, "points": points}


def annotation(frame_id, status="valid", scene_type=None):
    valid = status == "valid"
    return {
        "frame_id": frame_id,
        "timestamp": frame_id / 24,
        "image": f"frames/frame_{frame_id:04d}.jpg",
        "selection_stratum": "clear_normal_lane",
        "scene_type": scene_type or ["clear_normal_lane"],
        "review_status": "complete",
        "ego_lane_status": status,
        "annotation_flag": None if valid or status == "ambiguous" else NO_VALID_EGO_LANE,
        "left_boundary": boundary("visible" if valid else "not_visible"),
        "right_boundary": boundary("visible" if valid else "not_visible", 400),
        "notes": "",
    }


def prediction(frame_id, status="valid", offset=0, y_values=(700, 500, 300)):
    valid = status == "valid"
    return {
        "frame_id": frame_id,
        "ego_lane_status": status,
        "left_boundary": boundary("visible" if valid else "not_visible", offset, y_values),
        "right_boundary": boundary("visible" if valid else "not_visible", 400 + offset, y_values),
    }


def dataset(statuses=None, tags=None):
    statuses = statuses or ["valid"] * 100
    annotations = [
        annotation(frame_id, statuses[frame_id], (tags or {}).get(frame_id))
        for frame_id in range(100)
    ]
    return {
        "schema_version": 1,
        "name": "test",
        "video": {"width": WIDTH, "height": HEIGHT},
        "annotations": annotations,
    }


def prediction_file(items):
    return {"schema_version": 1, "model": {"name": "test-model"}, "predictions": items}


class LaneAcceptanceMetricTests(unittest.TestCase):
    def test_perfect_predictions_and_correct_abstentions(self):
        statuses = ["valid"] * 50 + ["invalid"] * 25 + ["ambiguous"] * 25
        truth = dataset(statuses)
        predictions = [prediction(i, "valid" if i < 50 else "invalid") for i in range(100)]
        result = evaluate_predictions(truth, prediction_file(predictions))
        metrics = result["metrics"]
        self.assertEqual(metrics["valid_ego_lane_detection_rate"], 1.0)
        self.assertEqual(metrics["false_ego_lane_rate"], 0.0)
        self.assertEqual(metrics["missed_ego_lane_rate"], 0.0)
        self.assertEqual(metrics["correct_abstain_rate"], 1.0)
        self.assertEqual(metrics["safety_weighted_error_rate"], 0.0)
        self.assertEqual(metrics["left_boundary_error"]["pixels"]["mean"], 0.0)

    def test_false_lane_costs_three_times_a_miss(self):
        statuses = ["valid"] * 50 + ["invalid"] * 50
        truth = dataset(statuses)
        predictions = [prediction(i, statuses[i]) for i in range(100)]
        predictions[0] = prediction(0, "invalid")
        predictions[50] = prediction(50, "valid")
        result = evaluate_predictions(truth, prediction_file(predictions))
        self.assertEqual(result["counts"]["missed_lane"], 1)
        self.assertEqual(result["counts"]["false_lane"], 1)
        self.assertAlmostEqual(result["metrics"]["missed_ego_lane_rate"], 1 / 50)
        self.assertAlmostEqual(result["metrics"]["false_ego_lane_rate"], 1 / 50)
        self.assertAlmostEqual(result["metrics"]["safety_weighted_error_rate"], 4 / 200)

    def test_boundary_interpolation_reports_pixels_and_width_fraction(self):
        truth = dataset()
        predictions = [prediction(i, offset=10) for i in range(100)]
        result = evaluate_predictions(truth, prediction_file(predictions))
        for side in ("left", "right"):
            error = result["metrics"][f"{side}_boundary_error"]
            self.assertAlmostEqual(error["pixels"]["mean"], 10.0)
            self.assertAlmostEqual(error["pixels"]["p95"], 10.0)
            self.assertAlmostEqual(error["normalized_by_image_width"]["mean"], 10 / WIDTH)
            self.assertEqual(error["sample_count"], 5000)

    def test_no_vertical_overlap_receives_image_width_error(self):
        truth = dataset()
        predictions = [prediction(i) for i in range(100)]
        predictions[0] = prediction(0, y_values=(250, 150, 50))
        result = evaluate_predictions(truth, prediction_file(predictions))
        error = result["metrics"]["left_boundary_error"]
        self.assertEqual(error["no_y_overlap_boundaries"], 1)
        self.assertAlmostEqual(error["pixels"]["max"], WIDTH)

    def test_missing_predictions_are_abstentions_and_scene_false_rates_are_safe(self):
        tags = {
            i: ["intersection", "unmarked_road", "crosswalk_stop_line", "parked_cars_curb"]
            for i in range(100)
        }
        truth = dataset(["invalid"] * 100, tags)
        result = evaluate_predictions(truth, prediction_file([]))
        self.assertEqual(result["counts"]["missing_predictions"], 100)
        self.assertEqual(result["metrics"]["abstain_rate"], 1.0)
        self.assertEqual(result["metrics"]["correct_abstain_rate"], 1.0)
        for item in result["scene_false_positive_rates"].values():
            self.assertEqual(item["eligible_no_valid_lane"], 100)
            self.assertEqual(item["false_positive_rate"], 0.0)

    def test_invalid_annotation_requires_explicit_marker_and_empty_boundaries(self):
        item = annotation(5, "invalid")
        self.assertEqual(validate_annotation(item, WIDTH, HEIGHT), [])
        item["annotation_flag"] = None
        errors = validate_annotation(item, WIDTH, HEIGHT)
        self.assertTrue(any(NO_VALID_EGO_LANE in error for error in errors))


if __name__ == "__main__":
    unittest.main()
