#!/usr/bin/env python3
"""Apply the completed visual review to the v1 acceptance manifest."""

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "acceptance_v1"
DATASET = ROOT / "dataset.json"

OFF = {3, 5, 6, 20, 33, 35}
UNKNOWN = {23, 31, 54, 57}
IN = {1, 2, 24, 25, 27, 28, 29, 30, 34, 42, 43, 46, 49, 50, 58, 64}
ENTERING = {9, 10, 11, 12, 13, 15}
CROSSING = {17, 18}
LEAVING = {14, 16, 19, 21}


def main():
    data = json.loads(DATASET.read_text())
    all_ids = {item["item_id"] for item in data["annotations"]}
    specified = OFF | UNKNOWN | IN | ENTERING | CROSSING | LEAVING
    assert specified <= all_ids
    for item in data["annotations"]:
        item_id = item["item_id"]
        if item_id in OFF:
            expected = "OFF_DRIVABLE"
        elif item_id in UNKNOWN:
            expected = "UNKNOWN"
        elif item_id in IN:
            expected = "IN_CORRIDOR"
        elif item_id in ENTERING:
            expected = "ENTERING_CORRIDOR"
        elif item_id in CROSSING:
            expected = "CROSSING_CORRIDOR"
        elif item_id in LEAVING:
            expected = "LEAVING_CORRIDOR"
        else:
            expected = "ON_DRIVABLE_OUTSIDE_CORRIDOR"
        item["expected_state"] = expected
        item["review_status"] = "HUMAN_REVIEWED"
        item["review_notes"] = (
            "Reviewed in chronological full-frame overlay sheets; road support, "
            "central corridor overlap, and adjacent-frame motion were judged "
            "independently of proposal_state. Unclear cases were labeled UNKNOWN."
        )
    data["review"] = {
        "method": "chronological full-frame overlay and adjacent-event review",
        "reviewed_items": len(data["annotations"]),
        "event_definitions": [
            {"event_id": "car_11_crossing", "track_id": 11,
             "first_clear_entering_timestamp": 8.5,
             "first_clear_crossing_timestamp": 9.5},
            {"event_id": "person_17_crossing", "track_id": 17,
             "first_clear_entering_timestamp": 9.0,
             "first_clear_crossing_timestamp": 10.5},
        ],
    }
    DATASET.write_text(json.dumps(data, indent=2) + "\n")
    digest = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    (ROOT / "GROUND_TRUTH_SHA256.txt").write_text(digest + "  dataset.json\n")
    print(Counter(item["expected_state"] for item in data["annotations"]))
    print(digest)


if __name__ == "__main__":
    main()
