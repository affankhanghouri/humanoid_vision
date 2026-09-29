# Lane acceptance report: custom_ego_lane_mobilenetv3_small_v3

False ego-lane predictions carry three times the cost of missed lanes.
Ambiguous ground truth is expected to produce abstention.

## Ego-lane decision metrics

| Metric | Result |
|---|---:|
| Valid ego-lane detection rate | 0.00% |
| False ego-lane rate | 0.00% |
| Missed ego-lane rate | 100.00% |
| Abstain rate | 100.00% |
| Correct abstain rate | 100.00% |
| Safety-weighted error rate | 15.79% |

Evaluated 100 frames: 36 valid and 64 invalid/ambiguous. Missing predictions treated as abstentions: 0.

## Boundary metrics

| Boundary | Detection | Mean error | P95 error | Mean / width | P95 / width |
|---|---:|---:|---:|---:|---:|
| Left | 0.00% | N/A | N/A | N/A | N/A |
| Right | 0.00% | N/A | N/A | N/A | N/A |

Errors are sampled at common image y positions. They are image-space values, not meters.

## False positives by difficult scene

| Scene | False positives | Eligible invalid/ambiguous | Rate |
|---|---:|---:|---:|
| intersection | 0 | 27 | 0.00% |
| unmarked_road | 0 | 22 | 0.00% |
| crosswalk_stop_line | 0 | 16 | 0.00% |
| parked_cars_curb | 0 | 14 | 0.00% |

## Raw decision counts

```json
{
  "frames": 100,
  "ground_truth_valid": 36,
  "ground_truth_no_valid_lane": 64,
  "valid_lane_detected": 0,
  "false_lane": 0,
  "missed_lane": 36,
  "abstained": 100,
  "correct_abstain": 64,
  "missing_predictions": 0
}
```
