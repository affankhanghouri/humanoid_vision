# Performance

## Measurement context

Measurements were collected on an Intel i5-6300U (2 physical / 4 logical cores),
CPU only, using the local replay video. They are engineering baselines from one
laptop, not service-level guarantees. CPU frequency and external load were not
locked in every study.

The pipeline targets recent results. It does not maximize the number of frames
processed by every model.

## Shared-worker full-video A/B

The bracketing A/B/B/A study replayed the complete decodable video. Road-enabled
runs used the road-only TwinLiteNet+ graph at 640x384, OpenVINO FP32, two threads,
one stream, and a one-second request interval.

| Metric | Core pipeline mean | Optional road mean |
|---|---:|---:|
| Detector throughput | 3.985 Hz | 3.978 Hz |
| Road throughput | 0 Hz | 0.998 Hz |
| Display throughput | 24.013 FPS | 24.014 FPS |
| Stale tracking | 0.00% | 0.00% |
| Shared-worker utilization | 56.26% | 57.70% |
| Detector source age, mean | 281.84 ms | 272.52 ms |
| Detector source age, mean P95 | 416.61 ms | 398.64 ms |

The small source-age difference is treated as run-to-run variation. The supported
conclusion is that the optional 1 Hz road task caused no measurable freshness
regression in this replay.

Road-stage averages across the two enabled runs were:

| Stage | Mean | P95 | P99 |
|---|---:|---:|---:|
| Preprocess | 3.94 ms | 6.94 ms | 9.65 ms |
| Inference | 42.00 ms | 73.62 ms | 107.44 ms |
| Postprocess | 0.66 ms | 1.24 ms | 2.47 ms |
| Full worker task | 46.73 ms | 79.74 ms | 115.09 ms |
| Capture to result | 68.23 ms | 106.59 ms | 142.75 ms |

No overlapping heavy-compute intervals were observed. Each task type retained
only its newest pending frame; 1,916 waiting road frames were replaced rather
than queued in each enabled full-video run.

Source: [perception study](../benchmarks/PERCEPTION_STUDY_RESULTS.md).

## Priority demo replay

A separate matched 20-second headless comparison measured:

| Metric | Existing demo | Priority + road demo |
|---|---:|---:|
| Detector | 4.002 Hz | 4.000 Hz |
| Pose | 1.351 Hz | 1.350 Hz |
| Road | 0 Hz | 1.000 Hz |
| Display | 24.060 FPS | 22.648 FPS |
| Stale tracking | 0.00% | 0.00% |

Road inference was 43.39 ms mean, 42.82 ms median, 52.51 ms P95, and 55.06 ms
P99. The display cost includes the optional visualization.

Source: [priority demo results](../benchmarks/risk_demo/RESULTS.md).

## Standalone road screening

The full-resolution road-only OpenVINO two-thread candidate measured 47.9-53.5 ms
mean and 59.5-77.5 ms P95 across two 70-frame runs. The reduced-resolution road
model was faster but rejected because mask agreement with the full model fell to
0.8146 mean IoU and 0.3688 minimum. Agreement is not ground-truth accuracy.

Source: [road study](../benchmarks/road_study/RESULTS.md).

## Interpretation

Display FPS and model Hz measure different loops. The renderer may show a recent
tracking state on many source frames while the detector runs near 4 Hz. A stale
tracking rate of 0% means accepted rendered states stayed within the configured
freshness window in these validated replays; it does not establish detection
accuracy or safety.

The scheduler's useful behavior under load is replacement: work for obsolete
frames disappears before execution. This is why throughput alone is not the
primary optimization target.
