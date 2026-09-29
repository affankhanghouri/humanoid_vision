# UFLD v1 CULane CPU study

This directory contains a bounded timing study of UFLD v1 with a ResNet-18
backbone trained on CULane. It does not modify or integrate production code.

## Provenance

- Official implementation: <https://github.com/cfzd/Ultra-Fast-Lane-Detection>,
  revision `353df107756b8c03c22c27201e33fc63d84ecfe6`.
- Official checkpoint ID: `culane_18.pth`, Google Drive file
  <https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view>.
- Tested artifact: `culane_18.onnx` mirrored by `ailia-ai/ailia-models` from
  the official implementation, mirror file revision
  `44e193bbc824e92330bdb733c76a849a9156d8d0`, downloaded from
  <https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/culane_18.onnx>.
- Tested artifact SHA-256:
  `da6e2e26380c430043f52a84c838ed5fce3bfbd0c5315ea7bc95e4c328956200`.
- License: MIT in both the official repository and the traceable mirror.
- ONNX: opset 11, input `float32[1,3,288,800]`, output
  `float32[1,201,18,4]`, 44,517,392 parameters, 178,139,214 bytes.

Official preprocessing resizes to 800x288, converts BGR to RGB, scales to
0..1, and applies ImageNet normalization with mean `(0.485, 0.456, 0.406)`
and standard deviation `(0.229, 0.224, 0.225)`. The decoder applies softmax
over the first 200 row-grid bins, uses their expectation, treats class 200 as
background, reverses the 18 CULane row anchors, and retains lane slots with at
least three points.

## Reproduce

```bash
myenv/bin/python benchmarks/ufld_v1_study/verify_onnx.py --frames 3
myenv/bin/python benchmarks/ufld_v1_study/benchmark.py --frames 80 --warmup 10
```

The benchmark uses the fixed acceptance images only as inputs. It neither
reads labels into model logic nor changes `dataset.json`.

See [RESULTS.md](RESULTS.md) for the early-stop decision.
