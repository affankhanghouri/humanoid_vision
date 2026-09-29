# Road perception study — 27 September 2026

The useful result is a faster **full-resolution drivable-area candidate**. The tested structured-lane model is **rejected for this urban video**. Nothing from this study has been integrated into the live pipeline, and reliable ego-lane perception has not been achieved.

## Decisions

- Keep TwinLiteNet+ Nano at 640×384. Extract its drivable-area output into a separate ONNX graph, removing the unused lane branch. Decode a binary mask at model resolution and remove padding without resizing to the source frame.
- Prefer testing OpenVINO FP32 with two threads in the existing serial worker next. Standalone total latency was **47.9–53.5 ms mean**, **59.5–77.5 ms P95** across two runs. Every sampled road mask matched the original exactly. Keep the one-thread option for a future shared-worker comparison.
- Reject the 320×192 road variant: faster, but it loses substantial visible road area. Mean mask agreement IoU was **0.8146**, minimum **0.3688**. This is agreement with the baseline, not ground-truth accuracy.
- Reject the tested TuSimple LSTR lane checkpoint at both resolutions. Full resolution is fast enough to investigate, but predictions cross parked vehicles and invent boundaries on unmarked roads/intersections. High neural scores do not establish correctness. Temporal smoothing cannot fix this information problem.
- Do not extend the existing semantic-mask curve-fitting experiment or activate a road/lane task in production yet.

## What was inspected and preserved

The current `LatestFrameBuffer`, scheduler, serial compute worker, perception store/types, tracker, configuration and ONNX setup were inspected first. The scheduler replaces waiting work by task type and rejects stale input. Tracking and pose retain independent source metadata; rendering must not refresh it. Its current compute-budget admission is specifically for pose, so merely adding road/lane task names would be insufficient.

All **37 Python files under `src/` remained byte-for-byte unchanged**. Existing user edits were preserved. All downloaded checkpoints, ONNX exports and generated visual results remain Git-ignored. Only experiment code, tests, source hashes and this measured summary are intended for version control.

## Model selection and export

[UFLDv2's official repository](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2) provides pretrained checkpoints and an ONNX exporter. Its [TuSimple ResNet-18 configuration](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/blob/master/configs/tusimple_res18.py) uses 800×320 input and structured row/column classification. It was not benchmarked or rejected on measured performance. LSTR was selected for the bounded first experiment because its architecture and checkpoint are much smaller and directly available.

[Official LSTR](https://github.com/liuruijin17/LSTR) predicts separate parametric lane curves from seven queries. The published baseline uses 360×640 input and reports about 766k parameters and 574M MACs. This local instance has 765,754 trainable parameters. Each ONNX export is 3,203,298 bytes. Its two encoder/two decoder transformer stages execute on the CPU without custom GPU operators.

The converted-model archive stalled; the reproducible route now downloads only approximately 3.3 MB of official source/checkpoint files. `fetch_lstr.py` pins revision `c8fe59a1d8e1c456c982c9e954ee660ae9116b49` and verifies every SHA-256. It includes the upstream BSD-3-Clause license locally. `export_lstr.py` copies source into a temporary directory, removes training-only imports/classes, excludes profiling counters from the checkpoint and strictly loads all inference parameters with `weights_only=True`. Neither the original source nor checkpoint is modified.

Preprocessing follows the original [test code](https://github.com/liuruijin17/LSTR/blob/main/test/tusimple.py) and [dataset constants](https://github.com/liuruijin17/LSTR/blob/main/db/tusimple.py): resized BGR, mean `[0.40789654, 0.44719302, 0.47026115]`, standard deviation `[0.28863828, 0.27408164, 0.27809835]`, and a zero padding-mask input. A third-party ImageNet normalization example was deliberately not used. ONNX predictions were checked against PyTorch on three real frames at each resolution; maximum absolute errors were **1.07e-5** and **7.45e-6**.

## Measurement method and results

Actual machine: Intel i5-6300U, two physical cores/four logical threads, CPU only. Python 3.12.3, OpenCV 5.0.0, ONNX Runtime 1.30.0; OpenVINO version and all model hashes are in [results_summary.json](results_summary.json).

Each configuration uses five warmups and **70 unique frames**: the union of 60 evenly distributed timing samples and 12 visual checkpoints over 5–95% of `test_data/test_video.mp4`. Runs are serial. OpenCV uses one thread; ORT uses sequential execution, one inter-op thread, controlled intra-op threads and disabled spinning. OpenVINO uses one stream and explicit FP32. Video decoding, reference-mask loading, visualization, image saving and session initialization are excluded from stage timings. Setup time is separately recorded. Lane geometry uses a cached, same-frame road observation; this is not a simulation of asynchronous production freshness.

These are exploratory percentiles from 70 samples on a working laptop, not long-duration service guarantees. CPU frequency/load were not locked. Repeats show real tail variation. The current baseline already uses the new native-mask postprocessing, so its 111–113 ms totals must not be presented as directly comparable to the historical 271 ms benchmark with different preprocessing/postprocessing and operating conditions.

All numbers below are milliseconds. `t1/t2` mean ORT with one/two threads; `ov1/ov2` mean OpenVINO. `640` road is 640×384; `640` lane is 640×360; `320` halves each spatial dimension. `baseline` retains both Nano heads; all other road variants keep only the drivable head. Graph optimizations are enabled except `noopt`.

| Run | Pre mean | Inference mean | Post mean | Total mean | Total median | Total P95 | Total P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 7.4 | 104.4 | 1.0 | 112.8 | 106.6 | 149.4 | 178.1 |
| road640_t1 | 8.0 | 76.1 | 1.0 | 85.1 | 79.5 | 135.9 | 177.1 |
| road640_t2 | 8.6 | 67.3 | 1.0 | 76.9 | 72.2 | 109.5 | 149.3 |
| road640_ov1 | 7.3 | 52.7 | 0.9 | 60.9 | 58.7 | 88.5 | 95.5 |
| road640_ov2 | 5.8 | 41.1 | 1.0 | 47.9 | 46.8 | 59.5 | 65.8 |
| road320_t1 | 1.4 | 22.4 | 0.4 | 24.2 | 22.6 | 33.6 | 37.6 |
| road320_t2 | 1.4 | 18.7 | 0.3 | 20.4 | 19.9 | 25.7 | 29.2 |
| road640_noopt | 7.8 | 74.1 | 0.9 | 82.8 | 80.9 | 98.2 | 123.8 |
| lane640_t1 | 13.1 | 29.8 | 1.3 | 44.2 | 39.7 | 57.9 | 98.0 |
| lane640_t2 | 12.0 | 21.4 | 1.4 | 34.9 | 33.2 | 47.9 | 55.5 |
| lane320_t1 | 3.2 | 8.6 | 1.0 | 12.8 | 11.4 | 16.6 | 32.1 |
| lane320_t2 | 2.7 | 7.0 | 1.1 | 10.8 | 10.6 | 14.8 | 18.7 |
| lane640_ov1 | 12.3 | 30.2 | 1.2 | 43.7 | 39.6 | 67.6 | 100.8 |
| lane640_ov2 | 13.5 | 28.2 | 1.6 | 43.3 | 37.4 | 80.5 | 110.3 |
| baseline_repeat | 7.7 | 102.5 | 0.9 | 111.0 | 102.6 | 163.6 | 222.2 |
| road640_ov2_repeat | 7.6 | 44.9 | 1.1 | 53.5 | 49.1 | 77.5 | 138.5 |

Full mean/median/P95/P99 for **every stage**, warmup/sample counts, setup time, versions and model hashes are retained in [results_summary.json](results_summary.json). Local `validation/road_study/<run>/results.json` additionally holds every timing sample and decoded observation.

Removing the unused road head preserved masks exactly across ORT and OpenVINO. Direct channel comparison has the same background-on-tie behavior as two-class argmax. Native-mask postprocessing averaged about 1 ms, including finite-output checking. The graph-optimization toggle did not demonstrate an additional speed improvement in this noisy sample; no such claim is made.

## Visual review: 12 checkpoints

The generated local montages compare the baseline road model, the retained
full-resolution road-only model, the rejected reduced road model, and both
rejected LSTR resolutions. They remain under ignored `validation/road_study/`;
the measured findings are retained below so a fresh clone does not contain
broken links to generated files.

The standalone geometry check selects nearest surrounding lane instances at normalized y=0.9. It requires sufficient observed extent, noncrossing curves, plausible image-space width, smoothness and interior road-mask support. It never substitutes road edges for missing lane instances and never extrapolates beyond an observed curve's extent. An acceptance means those checks passed, **not that an ego lane was correctly detected**.

| Frame | Scene / finding | LSTR geometry check |
|---:|---|---|
| 100 | Crosswalk/intersection; inferred-looking converging lines without sufficient lane evidence | Pass, unsupported lane claim |
| 263 | Turning, crossing marks, close parked car; boundaries cut across scene | Reject: road support |
| 427 | Mostly unmarked road and parked cars; broad road-like boundaries, ego lane unverified | Pass, unverified |
| 591 | Crosswalk and sparse paint; usable surrounding pair missing | Reject: missing boundary |
| 755 | Parked vehicles and dashed paint; predictions extend across vehicles | Reject: road support |
| 919 | Dashed paint/cycle marking; inconsistent extent and surrounding pair | Reject: missing boundary |
| 1082 | Vehicle occludes the forward road; spurious confident curves | Reject: road support |
| 1246 | Continued vehicle occlusion, confident curves through vehicles | Reject: road support |
| 1410 | Wider, mostly unmarked road; lane extent inadequate | Reject: missing boundary |
| 1574 | Wide intersection/gentle road change; no supported surrounding pair | Reject: missing boundary |
| 1738 | Wide road with sparse markings; unsupported curved partition of road | Pass, unsupported lane claim |
| 1902 | Stop-controlled intersection; curves do not establish a valid ego lane | Pass, unsupported lane claim |

Across all 70 samples, the full model passes the geometry gate **21/70**, including **4/12** visual checkpoints. These are availability counts, not accuracy scores. Acceptance is identical across tested backends/thread counts. The reduced lane model passes **0/70**, including **0/12** visual checkpoints. Confidence frequently approaches 1.0 even for poor boundaries.

Road masks broadly follow the visible surface, but the baseline itself can spill onto crossing approaches/curb margins (notably 263 and 1902) and contain holes around paint/occlusion. Exact backend agreement preserves those limitations. The reduced road model adds obvious missing patches at 755, 919 and 1082. No ground-truth labels, lane accuracy, lateral error or quantified production temporal stability are claimed. The clip covers intersections, parked cars, occlusion, sparse/faded markings and turns/gentle curvature; sharp curves, night and adverse weather remain untested.

## Serial compute budget

Use `sum(task_seconds × requested_Hz)` as an initial admission constraint, then enforce freshness deadlines using measured rolling costs. Mean utilization alone does not prevent individual long tasks from making detection stale.

A **hypothetical** combination at detection 3 Hz, road 1 Hz, lane 2 Hz and pose 0.5 Hz would consume:

| Task | Assumed/measured cost | Rate | Seconds per second |
|---|---:|---:|---:|
| Detector, brief's standalone assumption | 196 ms | 3 Hz | 0.5880 |
| Road, repeated OpenVINO total mean | 53.52 ms | 1 Hz | 0.0535 |
| Lane, rejected LSTR ORT total mean (illustration only) | 34.86 ms | 2 Hz | 0.0697 |
| Pose, illustrative cost, not remeasured here | 139 ms | 0.5 Hz | 0.0695 |
| Total | | | **0.7807 (~78%)** |

This is **not an approved deployment configuration**, because the lane model failed quality. Replacing the detector assumption with the earlier live measurement of **258.54 ms** in the local `CONSISTENCY_REPORT.md` raises this to approximately **0.9684 (~97%)**, before unaccounted worker overhead. Running the detector at 3.5 Hz increases pressure further. The road model alone at 1 Hz is the sensible first future addition to measure, with lane disabled until quality passes and pose admitted only when deadlines permit. Display/capture near 24 FPS remains a target, not an achieved result of this standalone work.

## Next architecture, after a lane candidate passes quality

Retain one serial heavy worker and extend the existing scheduler. Keep only the newest pending frame for each of `detection`, `road`, `lane`, `pose`. Drop stale inputs. Detection freshness takes priority; generalize pose's current budget guard to every optional task and use rolling costs plus margin. Do not let task fairness force a long optional inference past detection's deadline. Measure detector Hz, source-to-display age, stale-track percentage, display FPS and CPU tails in a shared-worker replay before enabling anything by default.

Store independent immutable `RoadSegObservation` and `LaneObservation`, each with source frame ID/time, produced time and inference cost. Derived `RoadGeometryObservation` must retain **both parent references**, not pretend asynchronous masks and lanes came from one frame. Keep drivable area, connected road corridor, observed ego lane and ego corridor as different outputs. Only compute normalized image-space offset for a valid surrounding pair; no meters without calibration.

Temporal lane tracking remains mandatory for a future usable lane system, but has **not** been implemented or validated here after this candidate failed. The required behavior is:

1. Associate boundaries by sampled curve geometry and side, not fixed neural query ID. Smooth accepted observations; reject implausible jumps and lane swaps.
2. Track each boundary's last measurement source frame/time separately from its propagated state time, confidence and `inferred` flag. An observation of the other boundary must not refresh its age.
3. Propagate briefly using measured camera motion when available. Otherwise hold with rapid confidence decay. A 0.5-second initial retention limit is a proposal to validate, not permission to keep unobserved boundaries indefinitely.
4. Mark retained boundaries inferred immediately, and expire them or abstain sooner at cuts, intersections, contradictory geometry or uncertain motion. A display update must never refresh the source/last-observed timestamp.
5. Test contiguous clips at planned neural rates, including temporary one-side occlusion, reacquisition, lane change, camera turn and scene cut. Measure boundary jitter, false retention duration and valid-observation age against annotated evidence.

Next lane work should prioritize a checkpoint trained/evaluated on representative urban scenes and a small labelled set from the target cameras. UFLDv2 with urban training is a candidate for a subsequent bounded quality/CPU experiment, not an assumed solution. Do not add further networks to disguise the rejected model's failure. No depth, collision warning, planning, traffic-light classification or control is part of this study.

## Reproduce

From the repository root with the existing environment:

```bash
source myenv/bin/activate
python benchmarks/road_study/fetch_lstr.py
python benchmarks/road_study/export_lstr.py --source models/lstr_source
python benchmarks/road_study/prepare_road.py --source /path/to/TwinLiteNetPlus
python benchmarks/road_study/run_matrix.py
python -m unittest discover -s tests -v
```

The road exporter needs the original Nano source and existing local Nano checkpoint for the reduced-resolution experiment; `prepare_road.py` without `--source` only extracts the full-resolution road-only graph. The current local source used was `/tmp/tlnp`. The pinned LSTR downloader was executed successfully and checked against its manifest. No new runtime dependencies were added.

For a single useful road comparison:

```bash
python benchmarks/road_study/benchmark.py \
  --kind road --model models/road_nano_640.onnx \
  --backend openvino --threads 2 --frames 60 \
  --out validation/road_study/my_road_run
```

For lane benchmarking, supply `--road-masks` pointing to a baseline run with matching frame sampling; without it the script can report lane candidates but cannot validate road support. Timing samples and images stay under ignored `validation/`; models stay under ignored `models/`.

Validation completed: **65 unittest tests passed**, including eight new checks for decoder shape/numerics, curve poles, confidence, observed extent, missing road support/boundaries, crossing rejection, nearest-instance selection and binary-mask tie handling. Both LSTR exports passed ONNX checking and numerical comparison; all produced road graphs passed ONNX checking. These checks establish software behavior, not perception accuracy.
