# Custom ego-lane experiment: final result

## Decision

**Reject this checkpoint and do not integrate it into production.** The model
passed the development validation and CPU gates, but failed the one-time frozen
100-frame quality gate by abstaining on every frame. This is safe with respect
to invented lanes, but it is not useful: it missed all 36 valid ego lanes.

The frozen set was evaluated once after architecture and thresholds were fixed.
Its JSON hash remained
`6ffe9309c1882a09b6bcf542c7f9fd940d3cf3c6435732f8d1b079ba504ace0d`
before and after the run.

## Data

The unchanged train split contains 455 reviewed records: 211 valid and 244
invalid. The completed validation split contains 90 reviewed records: 70 valid
and 20 invalid. The new `validatoion_video.mp4` contributes the 20 independent
urban negatives. Twenty-eight unsupported source-level proposals were rejected
rather than accepted as ground truth.

## Training and development validation

The selected model remains MobileNetV3-Small at 320x192 with 24 row anchors,
left/right x and visibility outputs, and a three-state global head. The first
head averaged away horizontal feature position and produced 10.1%/15.2% mean
left/right errors. The corrected head preserves ten horizontal feature bins.
The final checkpoint is epoch 2 of a low-learning-rate refinement from the
frozen-backbone run.

Frozen development thresholds:

- valid probability: `0.9734095335006714`
- boundary visibility: `0.5`

Training-set diagnostic at the frozen thresholds (455 reviewed frames): 75.36% valid-lane detection, 0.00% false lanes, 24.64% misses, and 100.00% correct abstention. These are in-sample diagnostics, not acceptance evidence.

Validation results (90 reviewed frames):

| Metric | Result |
|---|---:|
| Valid-lane detection | 82.86% (58/70) |
| False ego-lane rate | 0.00% (0/20) |
| Missed valid-lane rate | 17.14% (12/70) |
| Correct abstention | 100.00% (20/20) |
| Left boundary detection | 84.62% |
| Right boundary detection | 82.35% |
| Left error mean / P95 | 54.30 / 155.47 px (4.71% / 13.50% width) |
| Right error mean / P95 | 49.22 / 109.20 px (4.27% / 9.48% width) |

The main validation failures are conservative misses under the highway
overpass, plus boundary drift on wide multi-lane highway frames. Visual sheets
are in `validation_v3/`.

## ONNX and CPU

ONNX checker passed. PyTorch and ONNX Runtime agreed across one random tensor
and ten real validation frames; maximum absolute error was below `9e-6` on all
outputs.

FP32 total time over 80 validation images:

| Backend | Threads | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| ONNX Runtime | 1 | 8.94 ms | 8.43 ms | 12.30 ms | 12.58 ms |
| ONNX Runtime | 2 | 8.61 ms | 8.36 ms | 10.75 ms | 15.00 ms |
| OpenVINO | 1 | 7.85 ms | 7.71 ms | 9.74 ms | 11.22 ms |
| OpenVINO | 2 | 6.71 ms | 6.42 ms | 8.74 ms | 9.25 ms |

The CPU requirement passes with ample margin.

## Frozen 100-frame result

| Metric | Result |
|---|---:|
| Valid-lane detection | 0.00% (0/36) |
| False ego-lane rate | 0.00% (0/64) |
| Missed valid-lane rate | 100.00% (36/36) |
| Abstain rate | 100.00% (100/100) |
| Correct abstention | 100.00% (64/64) |
| Left/right boundary detection | 0.00% / 0.00% |

No boundary error exists because the model emitted no valid boundaries. See
`frozen_100_report/missed_valid_examples.jpg` and the JSON/Markdown report in
the same directory.

## Exact next step

Do not lower the threshold using the frozen results and do not rerun that set.
Collect and human-review training and development-validation frames from the
same camera/domain as the frozen urban dashcam clip, especially valid faded,
one-sided, intersection-adjacent, and partially occluded lanes. Keep the frozen
100 frames sealed. Retrain and choose thresholds only on a new domain-matched
validation split; use a new untouched final set for the next honest acceptance
decision, because this frozen set has now been consumed once.
