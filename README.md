# Humanoid Vision

A feature-frozen, production-style CPU road-scene perception pipeline for object
detection, multi-object tracking, human pose, image-space motion compensation,
and freshness-aware visualization.

The project asks a practical edge-AI question: how much useful multi-task
perception can run on an **Intel i5-6300U (2 physical / 4 logical cores), CPU
only**? Its central design choice is to prefer a recent result over processing
every captured frame. It is an engineering/research release, not an autonomous
driving system.

## What works

- YOLO object detection through ONNX Runtime
- confidence-aware Kalman multi-object tracking
- human pose associated to tracked people
- sparse optical-flow camera/image-motion compensation
- immutable perception snapshots with source and completion timestamps
- latest-frame capture and newest-pending task replacement
- deadline-aware scheduling on one serial heavy-compute worker
- optional TwinLiteNet+ drivable-area segmentation
- opt-in Perception Priority Map demo
- deterministic tests and replay benchmarks

Lane, path-entry, collision-prediction, TTC, planning, and control are not
production features.

## Architecture

```text
Video / camera
      |
      v
LatestFrameBuffer (single replaceable frame)
      |
      v
ComputeScheduler (newest pending task by type)
      |
      v
Single serial compute worker
  detector -> tracker
  pose (when useful)
  road (demo only)
      |
      v
PerceptionStore (immutable timestamped snapshots)
      |
      v
Freshness checks -> motion projection -> renderer
```

There is no FIFO frame queue. If frames 101-104 arrive while inference runs on
frame 100, pending work is replaced until frame 104 is the next useful input.
This bounds backlog and protects display freshness on a slow CPU. See
[Architecture](docs/ARCHITECTURE.md).

## Hardware and software

Development and measured replay used:

- Intel i5-6300U, CPU only, 2 cores / 4 threads
- Linux
- Python 3.12
- GUI-enabled OpenCV

Other platforms have not been validated. A desktop session is required for the
normal GUI.

## Setup

```bash
git clone <repository-url>
cd humanoid_vision
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Copy the local assets described in [models/README.md](models/README.md) and
[test_data/README.md](test_data/README.md). Model weights, videos, training data,
virtual environments, recordings, and generated validation output are not
committed.

## Run

```bash
python src/main.py
```

Useful options:

```bash
python src/main.py --help
python src/main.py --video /path/to/video.mp4
python src/main.py --verbose
```

Press `q` or Esc to stop.

## Demo

The priority map is an opt-in visualization based on real tracked class,
confidence, image-space size/position, and motion. It is not calibrated risk,
depth, intent, TTC, or human attention.

```bash
python src/main.py --risk-demo
python src/main.py --risk-demo --no-road-overlay
python src/main.py --risk-demo --record outputs/demo.mp4
```

The first command requires the optional road model. Recording writes rendered
frames only; it does not capture desktop or window chrome.

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests benchmarks
python src/main.py --help
python -m pip check
```

The unit suite uses synthetic data and committed fixtures; it does not require
the local video or model files unless a test explicitly states otherwise.

## Benchmarks

Run a 20-second headless replay with the production detector and pose models:

```bash
python benchmarks/benchmark_pipeline.py --headless --seconds 20 --output validation/pipeline.json
```

Include the optional road task:

```bash
python benchmarks/benchmark_pipeline.py --headless --seconds 20 --road --output validation/pipeline-road.json
```

Generated JSON stays under ignored `validation/`. Research-specific commands
and evidence are indexed in [benchmarks/README.md](benchmarks/README.md).

## Measured performance

Full-video A/B replay on the i5-6300U:

| Configuration | Detector | Pose | Road | Display | Stale tracking |
|---|---:|---:|---:|---:|---:|
| Core pipeline mean | 3.985 Hz | workload-dependent | off | 24.013 FPS | 0.00% |
| Optional road mean | 3.978 Hz | workload-dependent | 0.998 Hz | 24.014 FPS | 0.00% |

The separate 20-second priority-demo replay measured 4.000 Hz detection,
1.350 Hz pose, 1.000 Hz road, 22.648 FPS display, and 0.00% stale tracking.
Road inference was 43.39 ms mean and 52.51 ms P95 in that run. These are replay
results on one laptop, not service-level guarantees. See
[Performance](docs/PERFORMANCE.md) and the
[measured study](benchmarks/PERCEPTION_STUDY_RESULTS.md).

## Intentionally rejected research

Lane and forward-path approaches were tested and rejected when they failed
quality, latency, or stability gates. LSTR, LaneATT, CondLaneNet, UFLD, the
custom ego-lane model, and path-relation V1/V2 are not loaded by the application
and are not presented as supported features. The retained reports document the
decisions; see [Research history](docs/RESEARCH_HISTORY.md).

## Limitations

- image-space motion is not world velocity
- no calibrated distance, physical speed, TTC, or collision prediction
- no reliable ego-lane or stable forward-corridor estimate
- coarse road segmentation is not path geometry
- occlusion, blur, abrupt cuts, camera motion, and domain shift can degrade output
- no steering, planning, or autonomous control
- no production autonomous-driving safety claim

See [Limitations](docs/LIMITATIONS.md).

## Repository layout

```text
src/          production runtime
tests/        runtime regression tests plus clearly named retained research tests
benchmarks/   replay tools, measured reports, and rejected research evidence
docs/         architecture, performance, limitations, and research history
models/       local model placement instructions
test_data/    local replay input instructions
validation/   ignored generated benchmark output
```

## Status and licensing

This repository is feature-frozen. Changes should focus on correctness,
maintainability, documentation, and reproducibility rather than new perception
features.

No open-source license has been selected. Until the owner adds one, normal
copyright restrictions apply; see [license guidance](docs/LICENSE_GUIDANCE.md).
