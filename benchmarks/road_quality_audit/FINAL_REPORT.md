# Final bounded road-reliability report

## 1. Definitely working

Object detection, stable tracking IDs, pose, latest-frame publication, the single serial heavy-compute worker, deadline scheduling, image-space motion, road-only TwinLiteNet+ inference, the Priority Map demo, and approximately 4 Hz detector / 1 Hz road / 24 FPS display operation remain the reliable stack.

## 2. Rejected for now

All lane-model work remains stopped. Path-relation V1 and V2 remain rejected. No additional probe rules, event states, lane training, neural models, workers, or production features were added.

## 3. Road-mask reviewed quality

A sealed manual review used 80 source frames at one-second intervals. Each source frame was reviewed beside its raw mask before any corridor output was shown. Counts: GOOD 21/80 (26.25%), USABLE 55/80 (68.75%), BAD 4/80 (5.00%), UNKNOWN 0/80. GOOD+USABLE is 76/80 (95.00%); a safe forward corridor was judged possible in 76/80 (95.00%). The raw-mask gate therefore passed. Dataset SHA-256: `643ce6d73bed109cfd3a8168dce09f093f6cf8e8a83d0b3adc7a528f92d65733`.

## 4. Road geometry attempted

Yes—one fixed representation only. It selects the raw drivable component connected to the lower-middle image, rejects disconnected components, samples 16 rows from 40–92% image height, rejects rows narrower than 15% of image width, records road left/right/center/width/confidence, and uses the central 35% of each road extent as the conservative corridor. These are road extents, not lane boundaries. No cleanup sweep was performed. A fail-closed sparse-point affine projection reuses existing motion history, but no temporal rescue was applied after static geometry failed.

## 5. Corridor metrics

Human-possible frames: 76. Available when possible: 66/76 (86.84%). Total available: 68/80. False corridors: 2/80 (2.50%). Corridors visibly outside real drivable road: 0/68. Only 28/66 (42.42%) available, human-possible corridors had a credible stable shape. Material centre-position error (>10% of image width at any reviewed sample) occurred in 38/66 (57.58%); material width error (>20% of judged safe width) also occurred in 38/66. Across 67 adjacent available observations there were 42 centre jumps >5% image width and 34 width jumps >15%. The geometry therefore fails the stability/no-large-jump gate despite passing availability and false-corridor limits. Failures concentrate around vehicle occlusion, congested parked-car scenes, crosswalks, turns/intersections, and irregular mask holes.

## 6. Simple object/corridor metrics

Not attempted. The decision tree requires stopping when corridor geometry fails. No INSIDE/OUTSIDE/UNKNOWN object relation was built.

## 7. CPU cost

At cached native-scale 640×360 masks, derivation cost was 4.712 ms mean, 6.596 ms P95, and 8.197 ms P99. It runs only on a new road observation, not per display frame. There is no new inference or worker.

## 8. Pipeline regression

No production code path imports or executes this experimental geometry, and no production default changed. Therefore no geometry A/B run was justified after the failed gate. The previously verified shared-worker result remains authoritative: approximately 4 Hz detector, 1 Hz road, 24 FPS display, and 0% stale tracking. The experimental module cannot regress runtime while disconnected.

## 9. Tests

The complete repository suite passes: 108/108. Three focused tests cover lower-middle component selection, disconnected-component abstention, conservative containment, affine projection, and broken-chain UNKNOWN behavior.

## 10. Decision

**REJECT road-corridor reasoning.** The lowest reliable level is the raw road segmentation as a coarse visualization/observation. It is not geometrically stable enough to support a forward corridor, so object/corridor and event reasoning must remain disabled.

## 11. One recommended next step

Replace or improve the road representation specifically for temporally stable road extents, then repeat this same sealed 80-frame corridor audit before implementing any road-relative object logic.
