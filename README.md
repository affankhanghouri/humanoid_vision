# Humanoid vision

## Setup

Validated with Python 3.12 on Linux. For a fresh checkout, create an environment
and install the pinned dependencies from the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` pins the application and benchmark dependencies to versions
in the working environment. It is not a complete transitive lockfile, and a clean
installation on other platforms has not been verified. Torch wheel selection can
vary by platform; the configured application uses CPU inference.

Copy the local assets described in [models/README.md](models/README.md) and
[test_data/README.md](test_data/README.md). Model weights, videos, virtual
environments and generated validation output are deliberately excluded from Git.
The GUI requires a desktop session and GUI-enabled OpenCV.

## Run

From the project root, use your existing environment:

```bash
source myenv/bin/activate
python3 src/main.py
```

For a fresh `.venv`, activate it instead of `myenv` and use the same run command.

The video opens fullscreen without the top title banner. `q` or Esc closes the window. Playback ends automatically at video EOF.
Detection (512-pixel ONNX) and pose (320-pixel ONNX) load through Ultralytics and
finish their configured warmups before capture starts.

## Project layout

```text
humanoid_vision/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── camera.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── latest_frame.py
│   │   ├── perception_state.py
│   │   ├── perception_store.py
│   │   └── metrics.py
│   ├── detectors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ultralytics_detector.py
│   │   └── factory.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   └── smoothing.py
│   ├── scheduling/
│   │   ├── __init__.py
│   │   ├── task.py
│   │   └── scheduler.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── capture_worker.py
│   │   ├── compute_worker.py
│   │   └── vision_pipeline.py
│   └── rendering/
│       ├── __init__.py
│       └── renderer.py
├── models/
│   ├── README.md
│   ├── yolo26n.onnx
│   ├── yolo26n.pt
│   ├── yolo26n_openvino_model/
│   ├── yolo26n-pose.pt
│   └── yolo26n-pose_openvino_model/
├── benchmarks/
│   ├── benchmark_backends.py
│   └── benchmark_pipeline.py
├── test_data/test_video.mp4
├── tests/
├── validation/
├── requirements.txt
├── README.md
└── REFACTOR_REPORT.md
```

The existing environment, optional camera helper, pose weights and validation
artifacts are retained. Default model/video paths are resolved from the project
location, so launching from another working directory also works. Internal
imports resolve through `src/main.py`; no package installation is required.

Run the backend comparison with `python benchmarks/benchmark_backends.py`.

## Architecture

`src/main.py` creates `VisionConfig` and runs `VisionPipeline`. Configuration includes
detector, Kalman, association, prediction and smoothing settings.

Capture publishes a timestamped `FramePacket` into a single-slot
`LatestFrameBuffer`. Every publish replaces the previous reference. Consumers
remember their last frame ID and independently read the newest packet. There is
no queue. Published image arrays must never be mutated; the renderer copies the
image before drawing. A consumer retaining a packet keeps that frame alive only
until its current work finishes.

The scheduler worker reads `LatestFrameBuffer` and submits immutable `ComputeTask`
items. `ComputeScheduler` keeps only the newest waiting task for each task type;
there is no FIFO queue. Higher priority runs first, with newer frame IDs breaking
ties. While frame 100 is running, waiting frames 101–104 replace each other so
the compute worker next processes 104. Detection starts no more than four times
per second; pose is offered for fresh tracked people at most twice per second. A fairness
guard allows another ready task type to run between detections. The scheduler also
learns detection/pose costs: after one pose probe, it defers pose when that work
plus the next detection would exceed the current tracking freshness deadline.
Rates are upper bounds, not guarantees; under CPU load, tracking takes precedence.

The single compute worker owns YOLO detection, pose, and the tracker. Backend-specific objects are converted to
`Detection` inside the detector adapter. The worker updates tracking with the
source packet's monotonic timestamp and publishes an immutable `PerceptionState`
containing `PerceptionEntity` snapshots into `PerceptionStore`. `source_timestamp`
is the frame capture time; `produced_timestamp` is recorded after detection and
tracking finish, using the same monotonic clock. The renderer reads the latest
frame and perception state independently. It accepts entities only when the source
frame ID is no newer than the display frame ID and the age
(`packet.timestamp - state.source_timestamp`) is between zero and
`max_render_age`, inclusive. Rejected or missing states clear display smoothing.
For accepted states, it predicts from the source timestamp to the display frame
timestamp, limits prediction age, applies display-only smoothing, and draws labels
and metrics. Processing latency is `produced_timestamp - source_timestamp`.

Every future result must identify its source frame, capture timestamp, and
processing completion timestamp. Consumers must check whether that result is still
useful for their current frame before using it. Pose observations have their own
metadata and never refresh the tracking timestamp. Skeletons are attached by
person ID and projected onto the unclipped display box, so an image-edge crop
does not stretch the skeleton.
Detector timing statistics remain in worker logs and the display, alongside
processing latency, state age, and freshness. GUI operations run on the main thread.

Worker errors print their traceback, request shutdown and propagate to the main
thread. Shutdown sets the stop event and wakes the scheduler before joining
capture, scheduler, and compute workers; pending work is discarded on shutdown.
Shutdown joins workers before destroying windows. An in-flight inference
is allowed to finish; a backend that hangs indefinitely cannot be forcibly
cancelled by a Python thread.

Future expensive modules can submit their own task types through the scheduler
and execute through the same compute lane, publishing typed result snapshots.
YOLO detection, tracking, and 320-pixel pose are implemented now. The old `detector_worker.py`
is retained temporarily and is not used by the runtime.

## Validation

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_pipeline.py --seconds 20 --exit-key q --output /tmp/vision-q.json
python benchmarks/benchmark_pipeline.py --seconds 20 --exit-key esc --output /tmp/vision-esc.json
python benchmarks/benchmark_pipeline.py --seconds 0 --output /tmp/vision-full.json
```

The benchmark uses both real models, capture, the configured renderer and GUI.
Use `--headless` for a repeatable comparison without GUI overhead, and
`--start-frame 144` to replay the section in the box-consistency recording.
It reports pose throughput and the fraction of displayed packets rejected for
stale tracking, in addition to detection and rendering measurements. It measures display
cadence, detector/pose latency, tracker/render cost and tracking freshness.
Warm-up is excluded. GUI/display work is included in frame cadence, while the
renderer measurement covers copying and drawing only. Throughput counts publications during the measured display interval, excluding
work that finishes during shutdown. Per-worker logs report their running rates.

Tracking tests cover confidence recovery, confirmation, association gates,
expiry, invalid inputs and replayed timestamps. Motion tests use known camera
translations/reversals, dropped frames, blank images, source changes and shared
capture snapshots. The legacy smoothing fixture still requires exact equality;
tracking behavior has intentionally changed from that fixture.

See `REFACTOR_REPORT.md` for measured results and the change inventory.

## Box consistency

The detector still uses the same 512-pixel model, CPU thread settings and scheduling
limits. Its adapter now returns candidates down to 0.25 confidence. Only detections
at or above `confidence` (0.50) can start or confirm a track; weaker candidates can
maintain an already confirmed ID with stricter overlap matching. Two consecutive
observations confirm a new track. Association rejects implausible size changes and
distant matches before Hungarian assignment. Invalid geometry/confidence is
ignored, and duplicate or out-of-order tracking timestamps do not advance state.
Overlapping car/truck/bus subtype duplicates are suppressed; subtype fluctuations
can maintain the same vehicle ID and its original label. This stabilizes identity,
but does not resolve an initially incorrect vehicle subtype classification.

Tracks expire after at most two missed updates or one second without an observation.
A single missed update can publish a prediction, explicitly identified by `misses`
and `last_observed_timestamp` on `PerceptionEntity`. Both renderers stop drawing it
450 ms after its last observation, even if the overall tracking state is newer.
Predicted-only people do not trigger pose inference. Set `max_snapshot_misses=0` to
publish only tracks observed in the latest detection pass.

`BoxMotionHistory` carries delayed boxes through measured camera/object image
motion. It shares a sparse field across all entities, uses forward/backward flow
checks and robust camera fitting, and adds local displacement where features are
available. Capture publishes the frame first, then estimates motion on a 192-pixel
image at roughly 6 Hz. The next packet carries an immutable motion snapshot;
rendering does not rerun optical flow. Projection bridges at most 200 ms beyond the
latest flow sample and falls back to bounded Kalman prediction if history is missing
or unreliable. History is limited to the 0.75-second tracking freshness window.
`motion_enabled=False` disables this path. Standalone renderers can also estimate
motion when packets have no capture snapshot.

Display smoothing remains separate from perception observations. Motion projection
never refreshes detector/pose timestamps or claims a new detection. Velocities and
motion arrows describe image pixels, not robot/world motion. Pose retains its own
freshness checks. Stale tracking clears boxes and display history.

To compare alignment on actual source frames with identical detections and a fixed
333 ms delay:

```bash
python benchmarks/benchmark_box_consistency.py --populate \
  --cache /tmp/boxes-cache.json --output /tmp/boxes-after
```

Reuse the same cache without `--populate` for subsequent runs. For a saved pre-change
`src` tree, set `VISION_BENCH_SRC=/path/to/baseline_src`. Candidate caches include
weak detections, but references use the 0.50 threshold for both implementations.
The source video is required to replay optical flow. This compares against
same-frame YOLO detections, **not annotated ground truth or mAP**. It reports missing
references and extra displayed boxes as well as overlap and center error.

Production acceptance still needs labeled recordings from the intended camera and
environment, with explicit false-positive, missed-object, ID-switch and latency
limits. Objects the detector never recognizes cannot be recovered by tracking;
blur, occlusion, abrupt cuts and textureless objects can defeat image motion.
The included street-video measurements establish a regression baseline, not
reliability across other conditions. See [CONSISTENCY_REPORT.md](CONSISTENCY_REPORT.md)
for measured results and their limits.

## First Git commit

The project has not been initialized or committed by this preparation step.
After reviewing the files:

```bash
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Initial modular vision pipeline"
```

`.gitignore` excludes local assets, environments, caches, secrets and generated
output. `.gitattributes` normalizes source line endings and `.editorconfig`
provides editor formatting defaults. Regression fixtures remain tracked.
No license has been selected; choose one before offering reuse rights publicly.
