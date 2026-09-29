# Local model assets

Model weights and exported inference artifacts are intentionally excluded from
Git. Copy compatible assets into this directory; setup does not download models
and this repository does not provide unofficial download links.

## Required for the normal application

| File | Purpose | Runtime | Input | Required |
|---|---|---|---:|---:|
| `models/yolo26n.onnx` | object detection | Ultralytics + ONNX Runtime, CPU | 512 px `imgsz` | yes |
| `models/yolo26n-pose-320.onnx` | human pose | Ultralytics + ONNX Runtime, CPU | 320 px `imgsz` | yes |

The exported models must be compatible with the installed Ultralytics version.
The authoritative paths, confidence thresholds, warm-up counts, and image sizes
are in `src/config.py`.

## Optional demo asset

| File | Purpose | Runtime | Expected tensor shapes |
|---|---|---|---|
| `models/road_nano_640.onnx` | TwinLiteNet+ road-only drivable mask | OpenVINO FP32, CPU | input `[1,3,384,640]`, output `[1,2,384,640]` |

The road model is loaded only by `--risk-demo` unless `--no-road-overlay` is
given. It is not a lane model and its mask is not path geometry.

## Research-only local assets

Retained benchmark scripts may reference additional PyTorch checkpoints, ONNX
exports, OpenVINO directories, or source trees. Those assets are not required to
run or test the production pipeline and remain ignored. Consult the relevant
report under `benchmarks/` for the exact experiment.

Do not commit weights or exports. Different exports can change predictions,
timing, and backend behavior even when filenames match.
