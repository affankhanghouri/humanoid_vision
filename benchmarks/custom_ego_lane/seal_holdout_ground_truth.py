#!/usr/bin/env python3
"""Seal already-reviewed holdout labels before frozen-model inference."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "target_camera_v1" / "final_holdout"
DATASET = ROOT / "dataset.json"


def main() -> None:
    data = json.loads(DATASET.read_text())
    annotations = data["annotations"]
    assert len(annotations) == 168

    for ann in annotations:
        status = ann["ego_lane_status"]
        assert status in {"valid", "invalid", "ambiguous"}
        if status == "valid":
            visible_points = sum(
                len(ann[name].get("points", []))
                for name in ("left_boundary", "right_boundary")
                if ann[name].get("status") == "visible"
            )
            assert visible_points >= 2, ann["item_id"]
        for name in ("left_boundary", "right_boundary"):
            points = ann[name].get("points", [])
            # Proposal rendering can land a terminal point just beyond the image.
            # Clip to the last valid pixel; do not extrapolate or add geometry.
            ann[name]["points"] = [
                [min(1279.0, max(0.0, x)), min(719.0, max(0.0, y))]
                for x, y in points
            ]
            for x, y in ann[name]["points"]:
                assert 0 <= x < 1280 and 0 <= y < 720, (ann["item_id"], x, y)

        ann["proposal_origin"] = ann["label_origin"]
        ann["label_origin"] = "HUMAN_REVIEWED"
        ann["review_status"] = "complete"
        ann["review_method"] = "full_chronological_contact_sheet_overlay_review"

    distribution = Counter(a["ego_lane_status"] for a in annotations)
    assert distribution == {"valid": 23, "ambiguous": 8, "invalid": 137}
    data["ground_truth_review"] = {
        "method": "full_chronological_contact_sheet_and_proposal_overlay_review",
        "policy": (
            "Only confidently visible ego-lane paint is valid. The prolonged "
            "signalized-intersection/crosswalk segment is invalid; transition "
            "frames without defensible geometry are ambiguous."
        ),
        "distribution": dict(distribution),
        "frozen_before_model_inference": True,
    }

    DATASET.write_text(json.dumps(data, indent=2) + "\n")
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    (ROOT / "GROUND_TRUTH_SHA256.txt").write_text(digest + "  dataset.json\n")

    evaluation = copy.deepcopy(data)
    for ann in evaluation["annotations"]:
        ann["split"] = "validation"
    for source in evaluation.get("sources", []):
        source["split"] = "validation"
    (ROOT / "evaluation_manifest.json").write_text(
        json.dumps(evaluation, indent=2) + "\n"
    )
    print(dict(distribution))
    print(digest)


if __name__ == "__main__":
    main()
