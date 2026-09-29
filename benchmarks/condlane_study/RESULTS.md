# CondLaneNet CPU and lane acceptance results

## Decision

Reject CondLaneNet Small/CULane for this system. It fails both the fixed
100-frame quality gate and the CPU budget. It was not integrated into the live
pipeline and no production default changed.

## Ground-truth validation

The fixed dataset contains exactly 100 unique records. All 100 have
`review_status: complete`; zero are pending. The schema validator reports zero
errors. Visible boundaries have at least two in-image points. Every
`not_visible` and `ambiguous` boundary has zero points. All 47 invalid ego-lane
records, and only those records, use `NO_VALID_EGO_LANE` with empty
`not_visible` boundaries.

Ground truth contains 36 valid, 47 invalid, and 17 ambiguous ego-lane frames.
Left boundary statuses are 41 visible, 53 not visible, and 6 ambiguous. Right
boundary statuses are 11 visible, 80 not visible, and 9 ambiguous. Human labels
were not changed.

## Model and export

The measured candidate is the official Apache-2.0 CondLaneNet Small checkpoint:
ResNet-18, trained on CULane, 800x320 input, structured lane instances, and
10.2 reported GFLOPs. The ONNX fixed graph is 47,700,083 bytes. ONNX Checker
passed. Across acceptance frames 0, 860, and 1990, ONNX Runtime versus PyTorch
had maximum absolute error `9.54e-6` and mean absolute error below `7.7e-7`.
OpenVINO versus ONNX Runtime had maximum absolute error `1.05e-5` in the smoke
comparison.

## Intel i5-6300U timing

Each backend processed all 100 frames after three warmups. Values are
mean / median / P95 / P99 in milliseconds.

| Backend | Threads | Preprocess | Inference | Postprocess | Total |
|---|---:|---:|---:|---:|---:|
| ONNX Runtime FP32 | 1 | 16.72 / 15.58 / 20.80 / 31.18 | 402.84 / 376.85 / 511.43 / 653.23 | 3.53 / 1.57 / 17.76 / 24.31 | 423.09 / 395.83 / 533.63 / 673.67 |
| ONNX Runtime FP32 | 2 | 18.31 / 17.43 / 23.19 / 30.55 | 259.14 / 223.39 / 415.46 / 460.91 | 4.70 / 1.67 / 25.40 / 43.02 | 282.14 / 244.09 / 450.42 / 495.17 |
| OpenVINO FP32 | 1 | 15.02 / 14.47 / 18.94 / 20.18 | 412.07 / 403.98 / 470.48 / 615.92 | 3.05 / 1.47 / 16.83 / 20.85 | 430.14 / 421.60 / 486.97 / 637.77 |
| OpenVINO FP32 | 2 | 14.84 / 14.65 / 17.31 / 18.10 | 237.19 / 234.21 / 258.90 / 281.90 | 3.24 / 1.39 / 16.74 / 27.56 | 255.28 / 250.60 / 286.21 / 318.76 |

OpenVINO with two threads is fastest, but its 237 ms mean and 259 ms P95
inference cost is too large to fit safely between the current detector's
freshness deadlines on the serial heavy worker.

## Fixed acceptance gate

The report uses OpenVINO FP32 with two threads and the conservative adapter
described in the README.

| Metric | Result |
|---|---:|
| Valid ego-lane detection | 5/36 (13.89%) |
| False ego-lane rate | 1/64 (1.56%) |
| Missed ego-lane rate | 31/36 (86.11%) |
| Abstain rate | 94/100 (94.00%) |
| Correct abstain rate | 63/64 (98.44%) |
| Safety-weighted error | 14.91% |
| Left boundary detection | 37/41 (90.24%) |
| Right boundary detection | 5/11 (45.45%) |
| Left error mean / P95 | 107.96 / 138.32 px |
| Left normalized mean / P95 | 0.0937 / 0.1201 image widths |
| Right error mean / P95 | 198.68 / 292.99 px |
| Right normalized mean / P95 | 0.1725 / 0.2543 image widths |

False-positive rates among invalid/ambiguous ground truth are 0/27 at
intersections, 1/22 on unmarked roads, 0/16 at crosswalks/stop lines, and 1/14
in parked-car/curb scenes. The single false ego lane is frame 390, tagged
`parked_cars_curb` and `unmarked_road`. This is a critical failure category.

The main visual failures saved in `output/visuals` include:

- frame 390: false ego lane on an invalid parked-car/unmarked-road scene;
- frames 1250, 1310, and 1370: abstention on valid occluded or parked-car scenes;
- frame 1450: missed valid lane on an unmarked-road scene;
- frames 500, 550, 630, and 640: accepted or partial boundaries with roughly
  160-231 pixel worst-side mean errors.

## Exact next recommendation

Do not add CondLaneNet or temporal lane tracking to production. Keep this
100-frame set frozen as the test gate. The next work should create a separate
local-domain training/validation set from additional urban videos, then train
or fine-tune a much smaller structured model with explicit no-lane negatives.
The deployment target should be below 100 ms P95 with two CPU threads before a
full acceptance run. UFLD v1 ResNet-18/CULane is the next reasonable
off-the-shelf CPU screen because its row-structured output is lighter, but it
should be rejected at preflight if its measured graph cannot approach that
timing target; it must not be tuned on these 100 acceptance labels.

