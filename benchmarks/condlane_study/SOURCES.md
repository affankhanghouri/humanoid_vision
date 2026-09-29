# CondLaneNet source record

## Measured candidate

- Model: CondLaneNet Small, ResNet-18, CULane checkpoint.
- Official repository: <https://github.com/aliyun/conditional-lane-detection>
- Source revision inspected: `c57b242deea809800d1bb3ec547f12cc357ce95b`.
- License: Apache License 2.0 in the official repository.
- Official checkpoint: <https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/CondLaneNet/models/culane/culane_small.pth>
- Checkpoint SHA-256: `8453cca101d98e83300659ec8f7e06cb4b9b1ea9111d335ad3b66514c8962ab0`.
- Checkpoint size: 143,405,479 bytes. It contains training optimizer state; the
  actual state dictionary has 12,056,745 tensors/parameters occupying about
  46 MiB in FP32.

The official configuration uses a 1640x590 CULane image, removes the top 270
rows, resizes the remainder to 800x320, keeps BGR channel order, and normalizes
with mean `[75.3, 76.6, 77.6]` and standard deviation
`[50.5, 53.8, 54.3]`. This study applies the same crop as a fraction of image
height because the acceptance images are 1152x720.

The official project reports 10.2 GFLOPs, 220 GPU FPS, and 78.14 CULane F1 for
this checkpoint. Those GPU results are source metadata, not measurements on
the target CPU.

The official project targets an obsolete mmdetection/mmcv stack. `model.py`
therefore provides an inference-only PyTorch reconstruction of the exact used
layers and checkpoint names. Every executed checkpoint tensor loads. The
released checkpoint also contains an old `bbox_head.reg_branch`; the official
inference source explicitly aliases regression to `mask_branch`, so those old
tensors are intentionally unused. The fixed neural graph exports heatmap
logits, dynamic parameters, and mask features; documented official dynamic
lane decoding runs as CPU postprocessing.

## Preflight rejection

Polar R-CNN ResNet-18/CULane was screened first because it reports 80.81 F1,
uses 20 structured anchors, and avoids NMS. Its official repository at
<https://github.com/ShqWW/PolarRCNN> had no license file or stated source/model
license at revision `7cc0c094ebd187992ab475d93baf4a4469ffdddc`. It was rejected
before checkpoint download or benchmarking because the licensing status is not
acceptable for a business recommendation.

