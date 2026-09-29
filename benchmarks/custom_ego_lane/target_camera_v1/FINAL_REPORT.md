# Target-camera ego-lane experiment — final report

**Decision: REJECT.** The frozen model is fast and safely abstains, but it is not useful: target validation recall is 41.24%, and the single frozen holdout run detected 0 of 23 valid-lane frames. No production integration was performed.

## 1. Train video audit

- Source: `training_data/lane/target_camera/train/train_drive.mp4`
- SHA-256: `66ba3789108b60aea7f4597efe21773fa831c49c06f3a80ea552449d93bba718`
- 762.566667 s, 1280×720, 30 FPS, 22,877 video frames.
- Audit used a full sequential decode, 848 cached samples spanning 0.0–762.3 s, and 22 chronological contact sheets. Median luminance was 102.21/255; observed range was 55.07–143.56.
- Camera position is **not consistent**. The file is a stitched compilation: roughly 0–252 s is an elevated expressway dashcam; 252–396 s is a narrow tropical rural/residential dashcam; 396–584 s is another divided rural-highway view; 584–762 s is a motorcycle camera with the instrument cluster occluding the lower center.
- Conditions include daylight, bright open highway, deep vegetation shade, traffic, and partial vehicle/instrument occlusion.
- Roads include marked divided expressways, toll/plaza approaches, narrow unmarked roads, rural curves, urban streets, intersections and crosswalks.
- Valid examples include strong and faded paint, one-sided boundaries, curves, and partially occluded lanes. Negatives include genuinely unmarked roads, intersection/crosswalk/stop-line areas, curbs/road edges, parked-car edges and ambiguous motorcycle-view geometry.
- This diversity is useful, but the stitched cameras substantially weaken the meaning of “target-camera” training.

## 2. Validation video audit

- Source: `training_data/lane/target_camera/validation/validation_drive.mp4`
- SHA-256: `d28fd69703561b5b45c32929a3fc531cbee64594abe8d88c2c2b459d91654ecd`
- 83.950533 s, 1280×720, 30000/1001 FPS, 2,516 frames.
- Audit used a full sequential decode, 168 cached samples spanning 0.0–83.5835 s, and 5 chronological contact sheets. Median luminance was 119.64/255; observed range was 87.14–130.72.
- Camera position is consistent: a hood-mounted urban view, but it differs from all major train segments.
- Daylight footage contains wide urban roads, turns/curves, traffic, parked vehicles, curbs, intersections, crosswalks, stop bars and occlusion.
- Only the left boundary could be labeled safely in positive frames. Apparent right-side candidates were usually curb/road edge and were rejected rather than promoted to ground truth.

## 3. Sealed final-video SHA-256

- Pre-development seal and post-freeze verification: `b980049c0492a1bc259bdc55632cedf39a1e1bc8003cb3e60bfaa7e044a241da`.
- Final source metadata: 96.966667 s, 1280×720, 60 FPS, 5,818 frames.
- The hash matched immediately before holdout decoding and again after evaluation.

## 4. Number of reviewed train frames

- 840 human-reviewed frames used: 620 initial full-duration samples plus 220 TRAIN-only active-learning samples.
- The complete train candidate pool contained 1,068 frames; 228 initial proposals were rejected.
- All retained model inputs were cached at 320×192, so epochs did not repeatedly decode video.

## 5. Train label distribution

| State/boundaries | Count |
|---|---:|
| Valid | 449 |
| Invalid | 243 |
| Ambiguous | 148 |
| Both visible | 219 |
| Left only | 187 |
| Right only | 43 |
| Neither | 391 |

Reviewed scene tags (non-exclusive): clear lane 377, curve/turn 346, intersection 83, crosswalk/stop 83, occlusion 354, unmarked 160, parked/curb 221, faded 221.

## 6. Number of reviewed validation frames

- 165 human-reviewed frames were retained; 3 were rejected.
- The set contains both positives and negatives and exceeds the requested 60 positive / 40 negative preference.

## 7. Validation label distribution

| State/boundaries | Count |
|---|---:|
| Valid | 97 |
| Invalid | 68 |
| Ambiguous | 0 |
| Both visible | 0 |
| Left only | 97 |
| Right only | 0 |
| Neither | 68 |

Reviewed scene tags (non-exclusive): clear lane 97, curve/turn 59, intersection 40, crosswalk/stop 40, occlusion 88, parked/curb 125, faded 28. The absence of right-boundary-positive validation labels is a real source limitation, not fabricated balance.

## 8. Label and review method

- Train and validation were decoded sequentially across their full durations and cached. Chronological contact sheets were reviewed instead of seeking isolated compressed frames.
- Sampling favored temporal coverage. The active-learning expansion used perceptual dHash filtering (`min distance = 8`) and added samples only from the TRAIN video.
- Existing TwinLiteNet masks, paint/Hough geometry, and the already-installed LaneATT were used only as proposal generators. LaneATT was not considered or trained as a candidate model.
- Every retained annotation was manually reviewed. Provenance remained explicit; weak proposals were rejected or marked ambiguous. Supported points were retained only where visible; active-learning geometry below 53% image height was removed to avoid the motorcycle instrument cluster, with no extrapolation.
- One controlled active-learning loop added 220 TRAIN-only frames focused on the motorcycle/weak-paint domain: 72 partial valid labels and 148 ambiguous frames.

## 9. Candidate A result

Candidate A fine-tuned the prior corrected custom checkpoint on target train data (batch 32, learning rate 1e-4, early-stopping patience 4). It stopped after 6 epochs; epoch 2 was selected.

| Metric | Result |
|---|---:|
| Valid-lane detection | 37.11% (36/97) |
| False-lane rate | 2.94% (2/68) |
| Missed valid lanes | 62.89% |
| Correct abstention | 97.06% |
| Mean left error | 4.70% image width |
| Selected threshold | 0.9964549541 |

## 10. Candidate B result

Candidate B started from ImageNet MobileNetV3-Small initialization (batch 32, learning rate 2e-4, early-stopping patience 4). It stopped after 6 epochs; epoch 2 was selected.

| Metric | Result |
|---|---:|
| Valid-lane detection | 27.84% (27/97) |
| False-lane rate | 4.41% (3/68) |
| Missed valid lanes | 72.16% |
| Correct abstention | 95.59% |
| Mean left error | 7.90% image width |
| Selected threshold | 0.6923961043 |

## 11. Selected model and why

Candidate A was selected because it had higher useful recall, lower false-lane rate, better abstention and materially better geometry than Candidate B. The one permitted TRAIN-only active-learning retrain of A improved validation recall from 37.11% to 41.24% while retaining 2.94% false lanes and 97.06% correct abstention. That retrained checkpoint, `candidate_a_active.pt`, was frozen.

## 12. Architecture and parameter count

- `TinyEgoLaneNet`, MobileNetV3-Small backbone, 993,799 parameters.
- Input: RGB 320×192 with ImageNet mean/std normalization.
- Corrected head projects each feature row while preserving 10 horizontal cells; it does not average away horizontal position.
- 24 normalized vertical anchors from 0.35 through 0.95.
- Outputs: left/right normalized x, left/right visibility logits, and valid/invalid/ambiguous state logits.

## 13. Training history

| Run | Epochs run | Best epoch | Initial → final train loss | Best validation outcome |
|---|---:|---:|---:|---|
| Candidate A | 6 | 2 | 1.1883 → 0.3812 | 37.11% recall, 2.94% false lanes |
| Candidate B | 6 | 2 | 3.1021 → 0.4032 | 27.84% recall, 4.41% false lanes |
| A + active learning | 5 | 1 | 1.1503 → 0.6066 | 41.24% recall, 2.94% false lanes |

All runs stopped early after four non-improving epochs. Later loss reduction did not translate into safer validation utility. The training history files did not record epoch wall time (`epoch_seconds` is absent); this is an instrumentation gap, though the capped runs and early stopping constrained compute.

## 14. Validation metrics

Frozen validation set: 165 frames, 97 valid and 68 invalid.

| Metric | Result | Development target | Gate |
|---|---:|---:|---|
| Valid-lane detection | 41.24% (40/97) | ≥80% | **FAIL** |
| False ego-lane rate | 2.94% (2/68) | ≤5% | PASS |
| Missed valid-lane rate | 58.76% (57/97) | — | Poor |
| Correct abstention | 97.06% (66/68) | ≥95% | PASS |
| Left-boundary detection | 41.24% | — | Poor |
| Mean left error | 60.92 px / 4.76% width | ≤5% | PASS |
| P95 left error | 171.15 px / 13.37% width | ≈≤12% | **FAIL** |
| Right-boundary detection | N/A | — | Not measurable |
| Right false visibility | 24.24% | — | Safety concern |

Median left error was 48.06 px / 3.75% width; P99 was 276.37 px / 21.59%. Right detection/error cannot be estimated because validation has no safely labeled right-boundary positives.

## 15. Validation failure cases

- 57 valid lanes were missed, versus 40 detected.
- All 20 positives tagged with occlusion were missed.
- Curved-road recall was 42.37% (25/59).
- The two false ego-lane decisions were intersection/crosswalk frames: 5.0% within that 40-frame category.
- Although frame-level false lanes stayed under 5%, a right boundary was spuriously visible on 24.24% of no-right-ground-truth frames.
- The validation score distributions overlap heavily. Lowering the threshold is not a safe remedy: at threshold 0.90, validation recall would be 98.97% but false lanes would be 88.24%.
- Visual diagnostics: [`validation_final`](validation_final/).

## 16. Frozen thresholds and postprocessing

- Valid-lane threshold: `0.9956316947937012`.
- Boundary visibility threshold: `0.7`.
- Valid decision: `softmax(state_logits)[valid] >= valid_threshold`.
- Emit a boundary only when the frame is valid and at least two anchors have sigmoid visibility ≥0.7.
- Coordinates are sigmoid-normalized x values at supported anchors only.
- No extrapolation and no temporal smoothing.
- Freeze record: [`FROZEN_CONFIG.json`](FROZEN_CONFIG.json), SHA-256 `866f2e0800f52e16e4c8c7b9dc84693872250da78f28cb2995f62245e84cd8a7`.

## 17. ONNX numerical verification

- ONNX Checker: passed.
- PyTorch vs ONNX, 11 samples, all outputs `allclose`.
- Maximum absolute errors: coordinates `6.5565e-7`; visibility logits `1.02818e-5`; state logits `4.52995e-6`.
- The exporter’s verification JSON retains a default `visibility_threshold: 0.5` metadata field, but the ONNX graph emits raw logits. The actual frozen postprocessor is unambiguously 0.7 in `FROZEN_CONFIG.json` and was used for validation and holdout evaluation.

## 18. ORT/OpenVINO CPU results

80 validation images, 10 warmups, FP32. Times are end-to-end milliseconds.

| Backend | Mean | Median | P95 | P99 | Pre / infer / post P95 |
|---|---:|---:|---:|---:|---:|
| ORT 1 thread | 6.667 | 6.623 | 7.538 | 7.776 | 2.163 / 5.562 / 0.076 |
| ORT 2 threads | 7.449 | 7.726 | 8.461 | 8.600 | 2.638 / 5.984 / 0.077 |
| OpenVINO 1 thread | 6.766 | 6.600 | 8.140 | 10.477 | 2.479 / 5.614 / 0.065 |
| OpenVINO 2 threads | 6.007 | 5.670 | 7.940 | 9.565 | 2.735 / 5.567 / 0.064 |

Every backend comfortably passes the hard P95 <100 ms gate and the preferred ≈20 ms target. There is no CPU regression.

## 19. Final model and ONNX hashes

- Checkpoint: `67c4cf50fb4aa17f823cd8a87de9f4b38039ce7c252428790b8081c4b6ac9d1c`.
- ONNX: `24497c78e6a5f4cf6b3199a5192581a486fc157ed83329a2c16be63d1af89006`.
- Both hashes still matched after final evaluation.

## 20. Final holdout annotation distribution

- 168 uniformly sampled frames spanning 0.0–96.95 s, all human-reviewed before model inference.
- Ground-truth SHA-256: `984f309d50c488f5afac864b3ddeba9d3705af72e23e50adf57bfee219fa213b`.
- Valid 23, invalid 137, ambiguous 8. The evaluator conservatively treats invalid plus ambiguous as 145 no-lane frames.
- The first ≈17.5 s contains the 23 supported marked-lane positives and 8 ambiguous transition frames. The remaining footage is a prolonged large signalized intersection/crosswalk stop with cross traffic and pedestrians, producing 137 genuine negatives. Balance was not fabricated.
- Four proposal terminal points slightly outside the image were clipped to x=1279 during pre-inference integrity validation; no points were added or extrapolated.

## 21. Final holdout metrics

This was the frozen model’s **single** holdout inference pass. No parameters, thresholds, labels, or postprocessing were changed afterward.

| Metric | Result | Final target | Gate |
|---|---:|---:|---|
| Valid-lane detection | **0.00% (0/23)** | ≥80% | **FAIL** |
| False-lane rate | 0.00% (0/145) | ≤5% | PASS |
| Missed valid-lane rate | **100.00% (23/23)** | — | **FAIL** |
| Correct abstention | 100.00% (145/145) | ≥95% | PASS |
| Left-boundary detection | 0.00% | — | **FAIL** |
| Right-boundary detection | 0.00% | — | **FAIL** |
| Mean/P95 boundary error | N/A | ≤5% / ≈≤12% | **Not demonstrated** |

No geometry error can be computed because the model emitted no matched visible boundary on any valid frame. This is not a geometry pass.

Scene-type result (tags are non-exclusive): all 23 clear-lane/occluded positives were missed; all 137 intersection/crosswalk negatives and all 8 ambiguous transition frames were correctly rejected. See [`scene_breakdown.json`](final_holdout/frozen_eval/scene_breakdown.json).

## 22. Visual examples of final failures

- [`final_missed_valid_examples.jpg`](final_holdout/frozen_eval/final_missed_valid_examples.jpg) shows representative misses across the positive segment. Green/cyan ground-truth boundaries are visible, but the frozen model predicts invalid.
- [`final_correct_abstention_examples.jpg`](final_holdout/frozen_eval/final_correct_abstention_examples.jpg) shows representative safe abstentions from the negative segment.
- The original evaluator sheets are in [`frozen_eval`](final_holdout/frozen_eval/). There are no false-lane or predicted-boundary-error examples because the model never emitted a valid lane on this holdout.

## 23. PASS / REJECT

**REJECT. Do not integrate this model.**

The model passes CPU, false-lane and abstention gates, but it fails the central usefulness requirement on both validation and final holdout. Safe “always abstain” behavior is not success. Validation already missed the ≥80% recall target by 38.76 percentage points; the untouched holdout then fell to zero recall. Geometry acceptance was not demonstrated on holdout.

No production, steering, path-planning, collision, distance, road-model, risk-demo or temporal-tracking integration was performed.

## 24. Exact next technical step

Create a **new v2 data split by complete drive**, using one consistent production-intended camera and mounting position:

1. Record additional independent drives with the holdout-like urban viewpoint, prioritizing long, confidently marked ego lanes, faded paint, partial occlusion, intersections and safely visible right boundaries.
2. Assign whole drives—not nearby frames—to TRAIN, VALIDATION and a brand-new untouched FINAL HOLDOUT before labeling. The now-consumed holdout may be used only for diagnosis/training in v2, never again for final acceptance.
3. Human-review enough same-camera positive geometry to remove the current stitched-camera and right-boundary gaps; keep hard intersection/crosswalk/curb negatives.
4. Retrain the same tiny corrected architecture from Candidate A and ImageNet as the two bounded starting strategies. Do not lower the present threshold: validation shows that doing so causes catastrophic false lanes.
5. Require the validation gates before freezing, then evaluate exactly once on the new untouched holdout.

The immediate engineering action is therefore **data acquisition and drive-level partitioning**, not architecture search or threshold tuning.
