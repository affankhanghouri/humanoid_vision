# Road integration and urban lane study

## Decision

The road-only TwinLiteNet+ task can coexist with the current detector at about
1 Hz on this machine. Across the bracketing A/B/B/A full-video replays it did
not reduce detector freshness, display rate, or stale-tracking rate. Road stays
disabled by default until the application has an explicit product use for its
native-resolution drivable mask.

LaneATT ResNet-18/CULane is rejected. Its structured output is preferable to a
semantic lane mask, but its single-frame visual results are not reliable on
this video and its best measured CPU backend is too expensive for the shared
worker. No lane task was added to production and no temporal lane tracking was
implemented.

## Phase 1: shared-worker road integration

All runs replayed the complete decodable portion of
`test_data/test_video.mp4`. B runs used TwinLiteNet+ 640x384, OpenVINO FP32,
two inference threads, one stream, and a one-second request interval.

| Run | Road | Detector Hz | Road Hz | Display FPS | Stale tracking | Worker util. | Process CPU, whole machine |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | off | 3.979 | 0 | 24.007 | 0.00% | 55.91% | 35.77% |
| B3 | on | 3.979 | 0.998 | 24.016 | 0.00% | 57.43% | 38.46% |
| B4 | on | 3.978 | 0.997 | 24.011 | 0.00% | 57.98% | 38.76% |
| A2 | off | 3.991 | 0 | 24.019 | 0.00% | 56.61% | 41.81% |
| A mean | off | 3.985 | 0 | 24.013 | 0.00% | 56.26% | 38.79% |
| B mean | on | 3.978 | 0.998 | 24.014 | 0.00% | 57.70% | 38.61% |

Road changed detector rate by -0.007 Hz (-0.18%), display rate by +0.0002
FPS, and shared-worker utilization by +1.45 percentage points. Process CPU as a
share of the whole four-core machine varied more between the two baseline runs
than between A and B; the paired mean changed by -0.18 percentage points.
System CPU changed from 56.86% to 57.37% (+0.51 points).

Detector source age at render time did not regress. The A mean of the two run
means was 281.84 ms, compared with 272.52 ms for B. Mean P95 was 416.61 ms for
A and 398.64 ms for B. This small improvement is run-to-run variation, so the
supported conclusion is no measurable freshness harm rather than a road-driven
improvement.

| Detector source age at display | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| A1 | 278.40 ms | 291.28 ms | 416.63 ms | 460.02 ms |
| B3 | 271.56 ms | 255.07 ms | 381.29 ms | 420.09 ms |
| B4 | 273.48 ms | 289.92 ms | 415.99 ms | 419.91 ms |
| A2 | 285.28 ms | 292.53 ms | 416.59 ms | 418.46 ms |

| Detector inference | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| A1 | 131.68 ms | 128.18 ms | 177.64 ms | 246.41 ms |
| B3 | 124.38 ms | 118.82 ms | 169.65 ms | 190.56 ms |
| B4 | 126.70 ms | 124.81 ms | 168.93 ms | 197.64 ms |
| A2 | 135.14 ms | 142.20 ms | 166.97 ms | 176.57 ms |

| Road stage, B3/B4 average | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| Preprocess | 3.94 ms | 3.30 ms | 6.94 ms | 9.65 ms |
| Inference | 42.00 ms | 36.58 ms | 73.62 ms | 107.44 ms |
| Postprocess | 0.66 ms | 0.55 ms | 1.24 ms | 2.47 ms |
| Full road worker task | 46.73 ms | 40.85 ms | 79.74 ms | 115.09 ms |
| Capture to road result | 68.23 ms | 62.65 ms | 106.59 ms | 142.75 ms |
| Road source age at display | 569.93 ms | 582.25 ms | 1030.48 ms | 1066.43 ms |

The roughly 570 ms display-age mean is expected for a result refreshed at 1
Hz; capture-to-result latency is the appropriate measure of processing delay.
Each B run offered 2,000 road frames, executed 83 newest-frame tasks, replaced
1,916 pending frames, deferred once for detector budget, and had zero stale
drops and zero explicit skips. Detection executed 331/332 tasks with zero stale
drops. Instrumentation found no overlapping heavy-compute intervals.

The implementation keeps one serial heavy worker, preserves source frame ID,
source timestamp, and produced timestamp, and stores an immutable native
384x640 `RoadSegObservation` independently of tracking and pose. Renderer reads
do not refresh timestamps. The generalized optional-work admission check uses
measured task cost and the next detection freshness deadline; detection retains
priority and stale pending road work is replaced rather than queued.

Detailed machine-readable evidence is in
[`road_study/shared_worker_summary.json`](road_study/shared_worker_summary.json).

## Phase 2: standalone lane candidate

UFLDv2 ResNet-18/CULane was screened first because of its structured row/column
lane output. It was not promoted to the measured candidate: safe checkpoint
metadata inspection showed 787.01 MiB of FP32 parameters, including a single
712.69 MiB dense classification weight. That is an unsuitable starting point
for this CPU budget, so no visual verdict is claimed for UFLDv2.

The bounded candidate was LaneATT ResNet-18 trained on CULane. CULane is a
closer domain match than highway-only lane datasets, and LaneATT produces
separate lane proposals with 72 sampled curve offsets. The official 640x360
preprocessing and checkpoint were used. The exported graph emits proposals;
CPU postprocessing reproduces the official confidence filter, lane-distance
NMS, and top-four limit.

ONNX Checker passed. Across three real video frames, ONNX Runtime versus
PyTorch had maximum absolute error `1.220703125e-4` and mean absolute error
between `2.97e-6` and `4.35e-6`, below the study limit of `2e-3`.

| Backend | Threads | Preprocess mean / P95 / P99 | Inference mean / P95 / P99 | Postprocess mean / P95 / P99 | Total mean / P95 / P99 |
|---|---:|---:|---:|---:|---:|
| ONNX Runtime | 1 | 2.85 / 3.38 / 3.54 ms | 466.48 / 498.82 / 527.50 ms | 0.45 / 0.79 / 1.09 ms | 469.78 / 502.74 / 531.09 ms |
| ONNX Runtime | 2 | 3.20 / 4.01 / 5.10 ms | 336.89 / 471.39 / 553.04 ms | 0.52 / 0.87 / 1.80 ms | 340.60 / 477.40 / 557.85 ms |
| OpenVINO FP32 | 1 | 3.02 / 3.75 / 5.25 ms | 424.62 / 474.34 / 557.85 ms | 0.47 / 0.96 / 1.13 ms | 428.11 / 477.32 / 561.40 ms |
| OpenVINO FP32 | 2 | 3.27 / 3.91 / 5.34 ms | 264.23 / 297.60 / 385.16 ms | 0.50 / 0.93 / 1.25 ms | 268.01 / 301.12 / 388.89 ms |

These figures cover 70 evenly spaced frames after three warmups. OpenVINO with
two threads is the fastest configuration, but a 264 ms mean and 301 ms P95 is
too large for this serial worker's detector-freshness budget.

Visual review also rejects the model. The median sampled frame produced no
lane. Frames 1071 and 1245 correctly avoid inventing lanes in a crowded,
ambiguous intersection, and frame 1999 correctly stays silent on an unmarked
road. However, frame 347 projects a curve through the parked-pickup region,
frames 898 and 1796 follow road edges/curbs as lane boundaries, and frame 1448
misses visible lane structure. The output is too sparse and the positive
predictions are not trustworthy enough for an ego-lane boundary.

Review the [`12-frame contact sheet`](lane_study/output/visuals/contact_sheet.jpg),
the individual images under [`lane_study/output/visuals`](lane_study/output/visuals),
and the full [`lane timing JSON`](lane_study/output/results.json). Source,
revision, preprocessing, and hashes are recorded in
[`lane_study/SOURCES.md`](lane_study/SOURCES.md).

## Recommended architecture and next step

Keep detection, pose, and the validated road mask on the existing newest-frame
scheduler and single serial heavy worker. Enable road only through explicit
configuration at about 1 Hz. Keep lane perception as an offline standalone
experiment until a single-frame model passes visual acceptance and timing; do
not add temporal tracking around the rejected predictions.

The biggest remaining risk is domain mismatch without labeled acceptance data:
a model can look plausible while selecting a curb, parking boundary, or wrong
road partition. The next exact step is to annotate a stratified 100-frame lane
acceptance set from this video, including explicit no-lane frames, before
testing another checkpoint. That creates a repeatable gate for false lanes,
missed intersections, parked-vehicle crossings, and usable ego boundaries.
