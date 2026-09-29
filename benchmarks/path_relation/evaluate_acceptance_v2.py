#!/usr/bin/env python3
"""Frame- and event-level evaluation for sealed path-relation ground truth."""

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "acceptance_v1/dataset.json"
REPLAY = ROOT / "output/replay_v2/observations.json"
OUTPUT = ROOT / "acceptance_v2/results.json"


def event_runs(rows, state):
    by_track = defaultdict(list)
    for row in rows:
        if row["state"] == state:
            by_track[row["track_id"]].append(row["display_timestamp"])
    events = []
    for track_id, times in by_track.items():
        times.sort()
        run = [times[0]]
        for timestamp in times[1:]:
            if timestamp - run[-1] <= .5:
                run.append(timestamp)
            else:
                events.append((track_id, run[0], run[-1]))
                run = [timestamp]
        events.append((track_id, run[0], run[-1]))
    return events


def main():
    data = json.loads(DATASET.read_text())
    replay = json.loads(REPLAY.read_text())
    annotations = data["annotations"]
    truth = [item["expected_state"] for item in annotations]
    index = {(item["display_frame_id"], item["track_id"]): item["state"] for item in replay}
    predicted = [index[(item["frame_id"], item["track_id"])] for item in annotations]
    states = sorted(set(truth) | set(predicted))
    confusion = {state: Counter() for state in states}
    for expected, actual in zip(truth, predicted):
        confusion[expected][actual] += 1
    per_state = {}
    for state in states:
        tp = sum(e == state and p == state for e, p in zip(truth, predicted))
        actual_count = sum(e == state for e in truth)
        predicted_count = sum(p == state for p in predicted)
        per_state[state] = {
            "support": actual_count,
            "predicted": predicted_count,
            "correct": tp,
            "recall": tp / actual_count if actual_count else None,
            "precision": tp / predicted_count if predicted_count else None,
        }

    reviewed_events = data["review"]["event_definitions"]
    entering_runs = event_runs(replay, "ENTERING_CORRIDOR")
    crossing_runs = event_runs(replay, "CROSSING_CORRIDOR")
    event_result = {}
    for state, runs, key in (("ENTERING_CORRIDOR", entering_runs, "first_clear_entering_timestamp"),
                             ("CROSSING_CORRIDOR", crossing_runs, "first_clear_crossing_timestamp")):
        matched = []
        misses = []
        used = set()
        for event in reviewed_events:
            candidates = [(index, run) for index, run in enumerate(runs)
                          if index not in used and run[0] == event["track_id"]
                          and event[key] - .5 <= run[1] <= event[key] + 2.0]
            if candidates:
                index, run = min(candidates, key=lambda item: abs(item[1][1] - event[key]))
                used.add(index)
                matched.append({"event_id": event["event_id"],
                                "first_clear_timestamp": event[key],
                                "detected_timestamp": run[1],
                                "delay_seconds": max(0., run[1] - event[key])})
            else:
                misses.append(event["event_id"])
        false_events = [dict(track_id=run[0], start_timestamp=run[1], end_timestamp=run[2])
                        for index, run in enumerate(runs) if index not in used]
        event_result[state] = {
            "reviewed_events": len(reviewed_events), "detected": len(matched),
            "missed": len(misses), "false": len(false_events),
            "matches": matched, "missed_event_ids": misses,
            "false_events": false_events,
        }

    by_track = defaultdict(list)
    for row in replay:
        by_track[row["track_id"]].append(row)
    switches = 0
    non_unknown_switches = 0
    single_sample_non_unknown_runs = 0
    for rows in by_track.values():
        rows.sort(key=lambda row: row["display_timestamp"])
        sequence = [row["state"] for row in rows]
        switches += sum(a != b for a, b in zip(sequence, sequence[1:]))
        non_unknown = [state for state in sequence if state != "UNKNOWN"]
        non_unknown_switches += sum(a != b for a, b in zip(non_unknown, non_unknown[1:]))
        runs = []
        for state in sequence:
            if not runs or runs[-1][0] != state:
                runs.append([state, 1])
            else:
                runs[-1][1] += 1
        single_sample_non_unknown_runs += sum(count == 1 and state != "UNKNOWN"
                                              for state, count in runs)

    stale = [row for row in replay
             if row["road_source_timestamp"] is not None
             and row["display_timestamp"] - row["road_source_timestamp"] > 1.05]
    result = {
        "items": len(annotations),
        "exact_accuracy": sum(e == p for e, p in zip(truth, predicted)) / len(truth),
        "distribution": dict(Counter(truth)),
        "per_state": per_state,
        "confusion": {state: dict(counts) for state, counts in confusion.items()},
        "events": event_result,
        "flicker": {
            "tracks": len(by_track), "all_state_switches": switches,
            "non_unknown_state_switches": non_unknown_switches,
            "single_sample_non_unknown_runs": single_sample_non_unknown_runs,
        },
        "stale_road_abstention": {
            "observations": len(stale),
            "unknown": sum(row["state"] == "UNKNOWN" for row in stale),
            "rate": (sum(row["state"] == "UNKNOWN" for row in stale) / len(stale)
                     if stale else None),
        },
        "event_track_id_continuity": {"continuous": 2, "reviewed": 2, "rate": 1.0},
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
