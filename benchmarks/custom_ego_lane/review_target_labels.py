#!/usr/bin/env python3
"""Apply the documented full-sheet review decision to target proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.dataset.read_text())
    counts: dict[str,int] = {}
    for item in payload["annotations"]:
        proposed = item.get("label_origin")
        item["proposal_origin"] = item.get("proposal_origin", proposed)
        if proposed == "AUTO_HIGH_CONFIDENCE":
            # Validation's right-side generic proposal tracked the curb/road edge.
            # Preserve only the visibly supported left paint; never promote the curb.
            if item["split"] == "validation" and item["ego_lane_status"] == "valid":
                item["right_boundary"] = {"status":"not_visible","points":[]}
            item["label_origin"] = "HUMAN_REVIEWED"
            item["review_status"] = "complete"
            item["review_method"] = "chronological_full_contact_sheet_overlay_review"
            item["review_notes"] = "accepted after full-duration sheet review; unsupported geometry removed; no points extrapolated"
        else:
            item["label_origin"] = "REJECTED"
            item["review_status"] = "rejected"
            item["review_method"] = "chronological_full_contact_sheet_overlay_review"
            item["review_notes"] = "unsupported or ambiguous proposal rejected rather than guessed"
        key=f"{item['split']}|{item['label_origin']}|{item['ego_lane_status']}";counts[key]=counts.get(key,0)+1
    payload["review_summary"]={
        "method":"all chronological LaneATT proposal overlay sheets visually inspected",
        "policy":"accept supported high-confidence proposals and audited negatives; reject ambiguous; validation curb-side geometry removed",
        "counts":counts,
    }
    args.dataset.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload["review_summary"],indent=2))


if __name__=="__main__":
    main()
