# Local model assets

Model weights and inference exports are excluded from Git, including files placed
outside this directory. Obtain or copy the assets separately when setting up a
fresh clone; the setup instructions do not download them.

The default application requires these local files:

```text
models/
├── yolo26n.onnx             # Object detector, 512-pixel input
└── yolo26n-pose-320.onnx    # Pose estimator, 320-pixel input
```

Both run on CPU through ONNX Runtime. The authoritative paths, input sizes and
thresholds are in `src/config.py`.

Optional backend/size benchmarks also use local PyTorch `.pt` weights, other pose
ONNX sizes, and OpenVINO export directories containing `.xml`, `.bin` and metadata
files. These assets must also remain out of Git. Keep the same exports when
reproducing measurements: different exports can change detections and performance.
