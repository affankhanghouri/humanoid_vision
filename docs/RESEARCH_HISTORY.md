# Research history

The production runtime contains detection, tracking, pose, motion handling, and
an optional drivable-area observation. The experiments below were retained as
decision evidence, not supported features.

## TwinLiteNet+ lane branch

**Why tested:** reuse the two-head road model for lane pixels.

**Result:** the road branch was useful, but semantic lane pixels did not provide
reliable ego-lane geometry.

**Why rejected:** fragmented or misleading markings and road-scene ambiguity
could not support trustworthy lane boundaries.

**Lesson:** drivable area and ego-lane geometry are different products. The live
optional model contains only the road output.

[Road study](../benchmarks/road_study/RESULTS.md)

## LSTR

**Why tested:** small structured lane model with separate parametric curves.

**Result:** CPU latency was feasible enough to inspect, but predictions crossed
parked vehicles and invented boundaries on unmarked roads and intersections.

**Why rejected:** visual quality failed; neural confidence did not imply correct
geometry.

**Lesson:** structured output and temporal smoothing cannot repair missing or
wrong scene understanding.

[Road study](../benchmarks/road_study/RESULTS.md)

## LaneATT

**Why tested:** CULane-trained structured proposals offered a closer urban domain
match than highway-only alternatives.

**Result:** the best CPU configuration measured about 264 ms mean and 301 ms P95;
outputs were sparse and sometimes followed curbs or parked-vehicle boundaries.

**Why rejected:** both compute cost and visual trustworthiness missed the gate.

**Lesson:** benchmark latency and target-camera visual acceptance before adding a
candidate to the shared worker.

[Perception study](../benchmarks/PERCEPTION_STUDY_RESULTS.md)

## CondLaneNet

**Why tested:** instance-aware conditional lane prediction could potentially
handle complex scenes better than semantic masks.

**Result:** OpenVINO two-thread inference measured about 237 ms mean and 259 ms
P95, while evaluated boundary errors remained large.

**Why rejected:** latency exceeded the CPU budget and quality was insufficient.

**Lesson:** a more expressive head is not useful if it consumes detector
freshness without meeting the geometry gate.

[CondLaneNet results](../benchmarks/condlane_study/RESULTS.md)

## UFLD

**Why tested:** a fast row-classification lane architecture was a plausible edge
candidate.

**Result:** the tested UFLD v1 checkpoint measured 264.66 ms mean and 326.21 ms
P95 with OpenVINO FP32. A separate UFLDv2 candidate was screened out because a
large dense classification layer made the checkpoint unsuitable for this CPU
budget.

**Why rejected:** the measured variant was too slow and did not justify
integration; no production-quality claim was established.

**Lesson:** architecture labels such as “ultra fast” do not replace measurement
on the actual hardware and export.

[UFLD results](../benchmarks/ufld_v1_study/RESULTS.md)

## Custom ego-lane model

**Why tested:** domain-specific training might outperform public checkpoints on
the target camera.

**Result:** development validation appeared promising, but the sealed 100-frame
holdout produced 0/36 valid lanes, 36/36 misses, and 100% abstention.

**Why rejected:** it was safe against false lanes but entirely unusable on the
independent domain.

**Lesson:** a frozen, one-time holdout can expose domain overfitting that
development metrics hide.

[Custom model result](../benchmarks/custom_ego_lane/RESULTS.md)

## Path relation V1

**Why tested:** classify whether tracked objects were on, entering, crossing, or
leaving a forward image-space corridor.

**Result:** exact accuracy was 39.13%, `IN_CORRIDOR` recall was 0/16, and temporal
switching was excessive.

**Why rejected:** a stale 1 Hz road mask, occlusion, and a single contact probe
made the states unreliable.

**Lesson:** low compute cost does not make an unstable semantic state useful.

[V1 report](../benchmarks/path_relation/FINAL_REPORT.md)

## Path relation V2

**Why tested:** improve V1 with multiple support probes, affine mask projection,
overlap tests, and stronger hysteresis.

**Result:** exact accuracy fell to 30.43%, `IN_CORRIDOR` recall remained 0/16,
entering/crossing events regressed, and live geometry missed latency targets.

**Why rejected:** fewer superficial flickers did not produce correct states.

**Lesson:** a fixed central image corridor is not a defensible ego-path surrogate
on curved, offset, or occluded urban roads.

[V2 report](../benchmarks/path_relation/FINAL_REPORT_V2.md)

## Road-corridor audit

**Why tested:** determine whether connected road-mask geometry could provide a
stable forward corridor without lane inference.

**Result:** sparse road extents can support visualization, but audit evidence did
not establish a reliable vehicle path.

**Why rejected:** road boundaries, holes, crossings, and camera motion make a
central corridor unstable or semantically wrong.

**Lesson:** keep road visualization conservative and do not turn coarse
segmentation into planning geometry.

[Road quality audit](../benchmarks/road_quality_audit/FINAL_REPORT.md)

## Current decision

No rejected lane or path checkpoint is loaded by the normal application. Further
lane/path/corridor experiments are outside the feature-frozen scope. The reports
remain to show measured engineering decisions and prevent failed approaches from
being rediscovered without new evidence.
