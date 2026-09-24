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

`q` or Esc closes the window. Playback ends automatically at video EOF.
OpenVINO loads through Ultralytics and warms up twice before capture starts.

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
│   │   ├── result_store.py
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
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── capture_worker.py
│   │   ├── detector_worker.py
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
the original detector, Kalman, association, prediction and smoothing settings.

Capture publishes a timestamped `FramePacket` into a single-slot
`LatestFrameBuffer`. Every publish replaces the previous reference. Consumers
remember their last frame ID and independently read the newest packet. There is
no queue. Published image arrays must never be mutated; the renderer copies the
image before drawing. A consumer retaining a packet keeps that frame alive only
until its current work finishes.

The detector worker owns the tracker. Backend-specific objects are converted to
`Detection` inside the detector adapter. The worker updates tracking with the
source packet's monotonic timestamp and publishes an immutable `TrackingResult`.
The renderer predicts from that timestamp to the display frame timestamp, limits
prediction age, applies display-only smoothing, and draws labels and metrics.
GUI operations run on the main thread.

Worker errors print their traceback, request shutdown and propagate to the main
thread. Shutdown joins workers before destroying windows. An in-flight inference
is allowed to finish; a backend that hangs indefinitely cannot be forcibly
cancelled by a Python thread.

Future expensive modules can each consume the same latest-frame buffer with
their own worker/rate and publish their own typed result snapshot. No future
pose, scene, action or memory modules are implemented here.

## Validation

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_pipeline.py --seconds 20 --exit-key q --output /tmp/vision-q.json
python benchmarks/benchmark_pipeline.py --seconds 20 --exit-key esc --output /tmp/vision-esc.json
python benchmarks/benchmark_pipeline.py --seconds 0 --output /tmp/vision-full.json
```

The benchmark uses the real model, capture, renderer and GUI. It measures display
cadence, detector latency, tracker/render cost and Linux process resident memory.
Warm-up is excluded. GUI/display work is included in frame cadence, while the
renderer measurement covers copying and drawing only. Completed inference count
divided by display duration is an approximate detector rate because shutdown can
finish one in-flight inference. Per-worker logs report its running rate.

`tests/golden_tracking.json` records outputs from the original implementation for
80 updates including crossings, missing detections, deletion and rebirth, plus
all adaptive smoothing branches. Tests require exact output equality. The
original standalone backend benchmark and optional camera helper are retained.

See `REFACTOR_REPORT.md` for measured results and the change inventory.

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
