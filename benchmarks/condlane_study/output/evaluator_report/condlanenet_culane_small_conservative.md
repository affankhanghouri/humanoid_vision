# Lane acceptance report: condlanenet_culane_small_conservative

False ego-lane predictions carry three times the cost of missed lanes.
Ambiguous ground truth is expected to produce abstention.

## Ego-lane decision metrics

| Metric | Result |
|---|---:|
| Valid ego-lane detection rate | 13.89% |
| False ego-lane rate | 1.56% |
| Missed ego-lane rate | 86.11% |
| Abstain rate | 94.00% |
| Correct abstain rate | 98.44% |
| Safety-weighted error rate | 14.91% |

Evaluated 100 frames: 36 valid and 64 invalid/ambiguous. Missing predictions treated as abstentions: 0.

## Boundary metrics

| Boundary | Detection | Mean error | P95 error | Mean / width | P95 / width |
|---|---:|---:|---:|---:|---:|
| Left | 90.24% | 107.957 px | 138.321 px | 0.094 | 0.120 |
| Right | 45.45% | 198.678 px | 292.993 px | 0.172 | 0.254 |

Errors are sampled at common image y positions. They are image-space values, not meters.

## False positives by difficult scene

| Scene | False positives | Eligible invalid/ambiguous | Rate |
|---|---:|---:|---:|
| intersection | 0 | 27 | 0.00% |
| unmarked_road | 1 | 22 | 4.55% |
| crosswalk_stop_line | 0 | 16 | 0.00% |
| parked_cars_curb | 1 | 14 | 7.14% |

## Raw decision counts

```json
{
  "frames": 100,
  "ground_truth_valid": 36,
  "ground_truth_no_valid_lane": 64,
  "valid_lane_detected": 5,
  "false_lane": 1,
  "missed_lane": 31,
  "abstained": 94,
  "correct_abstain": 63,
  "missing_predictions": 0
}
```
