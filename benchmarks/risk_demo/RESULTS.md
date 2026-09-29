# Perception Priority Map demo

## Scope

This is a demo-only engineering visualization. It is not human attention and
not a calibrated collision-risk, depth, or intent model. It uses only current
tracked class/confidence, displayed box size and position, tracker velocity,
short motion projection, and a fresh drivable-area observation when available.
The rejected custom lane model is not loaded and no lane lines are rendered.

## Scoring

Each visible track receives a bounded relative priority score:

- 28% class relevance: person > bicycle/motorcycle > vehicle > other;
- 23% image-space closeness from box area and vertical image position;
- 25% proximity to the current drivable-corridor center, falling back to image center;
- 15% measured image-space motion down-frame or laterally toward the corridor;
- 5% detector confidence, plus a small base value.

Coasting tracks are discounted. Soft Gaussian blobs are drawn at the current
and short projected track positions. The colors are relative priority only:
green is lower, yellow is medium, and red is higher.

## Measured replay

Matched 20-second headless replays used `test_data/test_video.mp4`. The risk run
used `models/road_nano_640.onnx`, OpenVINO FP32, two CPU threads, one stream, and
a one-second road request interval.

| Metric | Existing demo | Priority + road demo |
|---|---:|---:|
| Detector rate | 4.002 Hz | 4.000 Hz |
| Pose rate | 1.351 Hz | 1.350 Hz |
| Road rate | 0 Hz | 1.000 Hz |
| Display rate | 24.060 FPS | 22.648 FPS |
| Stale tracking | 0.00% | 0.00% |

Road inference mean/median/P95/P99 was 43.39/42.82/52.51/55.06 ms. Road source
age at display averaged 562.29 ms with 1001.03 ms P95, as expected for a 1 Hz
result. The renderer rejects the observation after 1.35 seconds and never
refreshes its source timestamp.

The machine-readable measurements and a captured frame are stored beside this
file. Rendering was benchmarked with road opacity 0.10 and heatmap opacity 0.30;
the final visual defaults use a lighter 0.06 road tint and 0.36 heatmap opacity.
This changes compositing strength, not inference or scheduling work.
