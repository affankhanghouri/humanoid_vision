# Path-relation V2 final report

## 1. Why V1 failed

V1 sampled one bottom-centre patch. Vehicles and pedestrians occluded that road evidence, while the reused ~1 Hz mask remained in its source-frame coordinates. Its point-based corridor decision then amplified mask holes and alignment error. Result: 39.13% exact accuracy, 0/16 IN_CORRIDOR recall, and severe display-sequence flicker.

## 2. Exact probe geometry

For box `(x1,y1,x2,y2)`, width `w`, height `h`: contact centre `(cx,y2-0.015h)`, radii `(0.10w,0.025h)`; below centre `(cx,y2+0.18h)`, radii `(0.18w,0.05h)`; left centre `(x1-0.16w,y2-0.02h)`, radii `(0.075w,0.06h)`; right is symmetric at `x2+0.16w`. A probe is usable only when its complete rectangle is in-frame; coordinates are still safely clamped.

## 3. Road-support formula

The score is the available-probe renormalized weighted mean: `0.25 contact + 0.35 below + 0.20 left + 0.20 right`. Direct support requires contact >= 0.52 and score >= 0.48. Surrounding inference requires either (below >= 0.60 and a flank >= 0.50), both flanks >= 0.62, or one surrounding probe >= 0.65 with score >= 0.12. Inferred score is capped at 0.85. NOT_SUPPORTED requires at least three probes, score <= 0.30, and every usable probe <= 0.48. Everything ambiguous is UNKNOWN.

## 4. Occluded-road inference

`RoadSupportKind` explicitly records `ROAD_VISIBLE_AND_SUPPORTED`, `ROAD_INFERRED_FROM_SURROUNDINGS`, `ROAD_NOT_SUPPORTED`, or `UNKNOWN`. Inferred evidence is separately preserved and discounted in confidence; it is not silently treated as visible road.

## 5. Motion-compensation method

V2 reuses `MotionSnapshot.steps`. It composes the existing global partial-affine chain and inverse-transforms current probe coordinates into the original road-mask frame. It does not warp a full mask, calculate flow, or create a worker. Local object residuals are intentionally excluded because road alignment needs camera motion.

## 6. Affine-chain freshness limits

Original road time is never refreshed. Nominal road age limit is 1.05 s; every affine step must be contiguous and non-null. Existing flow enforces <=0.25 s step gaps and retains only the existing render-age history (0.75 s). At most 0.20 s of the latest affine is extrapolated, matching existing motion behavior. A missing/broken/old chain yields UNKNOWN.

## 7. Corridor-cleaning method

At native mask resolution V2 applies one elliptical morphological close, odd kernel `clamp(round(0.008*content_width),3,7)`. It then flood-fills only the connected component touching the lower-middle seed (x 39–61%, y 70–97%). Remote components remain excluded. The corridor is the conservative central 76% of each selected component row; no cross-component or large-hole fill is performed.

## 8. Footprint/corridor overlap

A lower-footprint patch is centred at bottom-centre minus `0.02h`, with radii `(0.38w,0.04h)`. Its integral-mask overlap replaces V1's single-point membership.

## 9. Hysteresis thresholds

Corridor entry requires overlap >=0.42. An already-inside track remains inside until overlap <0.24. Road support uses the asymmetric thresholds in section 3.

## 10. Temporal state-machine rules

Per track V2 retains stable state, pending state/count, state-entry time, ten recent samples, recent overlap/support, distance, and corridor side. Obvious initial OFF/OUTSIDE/IN states can settle immediately; later changes and ENTERING/CROSSING/LEAVING require two distinct tracking-source observations. ENTERING also needs reliable motion and decreasing corridor distance. CROSSING needs meaningful lateral speed and an outside-side-to-interior/opposite-side transition. State expires after 2 s absence. Stale road returns UNKNOWN without erasing the internal stable state.

## 11. Acceptance dataset distribution

The unchanged sealed human-reviewed set has 69 frames: OUTSIDE 31, IN 16, OFF 6, ENTERING 6, LEAVING 4, UNKNOWN 4, CROSSING 2. It has two reviewed temporal events. Frame coverage is useful for IN/OUTSIDE and marginal for OFF/ENTERING; two CROSSING frames and two events are insufficient for defensible percentage claims. No V2-generated labels were added. Additional examples require an actual human review, so none were fabricated.

## 12. V1 metrics

Exact accuracy 27/69 = 39.13%. IN recall 0/16. ENTERING event TP/FP/FN = 1/0/1. CROSSING event TP/FP/FN = 1/0/1. Full replay: 737 all-state switches, 81 non-UNKNOWN switches, 85 one-observation non-UNKNOWN runs. Offline geometry mean/P95/P99 = 0.831/2.002/6.352 ms.

## 13. V2 metrics

Exact accuracy 21/69 = 30.43%. IN recall 0/16. ENTERING event TP/FP/FN = 0/19/2. CROSSING event TP/FP/FN = 0/0/2. These are regressions, not a pass.

## 14. V1 versus V2 state switches

All-state switches fell 737 -> 138 (81.3% reduction), and one-observation non-UNKNOWN runs fell 85 -> 6 (92.9%). Non-UNKNOWN switches increased 81 -> 100, however, so improved superficial stability did not produce correct semantic states.

## 15. ENTERING raw counts

V1: TP 1, FP 0, FN 1. V2: TP 0, FP 19, FN 2. With only two reviewed events, percentages would be misleading.

## 16. CROSSING raw counts

V1: TP 1, FP 0, FN 1. V2: TP 0, FP 0, FN 2. Again, only two reviewed events exist.

## 17. IN_CORRIDOR recall

V1: 0/16 (0%). V2: 0/16 (0%). The footprint method emitted IN on replay but not on the reviewed IN frames; eight were called OFF, six OUTSIDE, and two UNKNOWN.

## 18. UNKNOWN and stale-road behavior

V2 produced UNKNOWN on 164/1523 unique tracking observations (10.77%) in full replay. The 1 Hz schedule creates no observations older than the 1.05 s nominal limit, so replay has no naturally stale samples. Deterministic tests verify stale road and broken affine chains return UNKNOWN, preserve original road time, and leave transform time unset (4/4 focused tests passed). V1's measured >0.55 s stale abstention remains 375/375 (100%).

## 19. Geometry compute time

Offline replay: mean 2.201 ms, median 1.825 ms, P95 5.911 ms, P99 9.643 ms. Live pipeline instrumentation: mean 3.920 ms, median 3.167 ms, P95 10.055 ms, P99 14.324 ms. V2 misses both mean <2 ms and P95 <5 ms targets in the live pipeline.

## 20. Full-pipeline A/B

A (road, no path): display 24.058 FPS, detector 4.001 Hz, pose 1.350 Hz, road 1.000 Hz, stale tracking 0%, worker utilization 58.720%. B (+V2): display 24.055 FPS, detector 4.001 Hz, pose 1.300 Hz, road 1.000 Hz, stale tracking 0%, utilization 58.701%. There is no new heavy worker and display/detector/road cadence is effectively unchanged; pose sampled one fewer execution in this 20 s run.

## 21. Diagnostic visual examples

`output/replay_v2/diagnostics/frame_*.jpg` shows relation overlays. `masks_*.jpg` shows labeled RAW ROAD MASK, CLEANED ROAD MASK, and FORWARD CORRIDOR panels. The sealed result and confusion matrix are in `acceptance_v2/results.json`.

## 22. Limitations

The central image-space corridor is not a reliable ego-path surrogate on curved or laterally offset roads. Surrounding road probes cannot recover road when segmentation is absent across all four regions. Broad flank inference trades occlusion recovery for false on-road support. Image-space box velocity and sparse-flow camera compensation do not resolve depth/parallax. Event sample size is too small.

## 23. ACCEPT / REJECT

**REJECT.** V2 fails exact accuracy, IN recall, ENTERING/CROSSING recall and false-event targets, and misses geometry latency targets. Its state-run stability improvement is real but insufficient.

## 24. Was `--path-demo` justified?

No. It was not added. V2 remains benchmark-only and is not connected to the renderer or Priority Map.

## 25. Exact next technical step

Do not tune more thresholds on these 69 frames. First obtain a substantially larger, independently human-reviewed sequence set with balanced IN/OFF/OUTSIDE states and at least 20 distinct ENTERING and 20 CROSSING events. Then replace the fixed central-row corridor with a temporally tracked drivable-component centreline/width model (still geometry-only), validate it on that sealed set, and optimize probe queries only if quality passes.
