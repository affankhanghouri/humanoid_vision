# Benchmarks and retained research

This directory is separate from the production runtime in `src/`.

## Runtime benchmarks

- `benchmark_pipeline.py`: end-to-end detector, pose, display, and freshness replay
- `benchmark_backends.py`: detector backend comparison
- `benchmark_pose.py` and `benchmark_pose_sizes.py`: pose timing
- `benchmark_box_consistency.py`: delayed-box alignment regression
- `benchmark_shared_road.py`: optional road task in the shared worker

Primary measured summary: [PERCEPTION_STUDY_RESULTS.md](PERCEPTION_STUDY_RESULTS.md).

## Optional road and demo

- [road study](road_study/RESULTS.md)
- [priority demo](risk_demo/RESULTS.md)
- [road quality audit](road_quality_audit/FINAL_REPORT.md)

## Rejected research retained as evidence

- [LaneATT / road integration](PERCEPTION_STUDY_RESULTS.md)
- [LSTR / road study](road_study/RESULTS.md)
- [CondLaneNet](condlane_study/RESULTS.md)
- [UFLD](ufld_v1_study/RESULTS.md)
- [custom ego lane](custom_ego_lane/RESULTS.md)
- [path relation V1](path_relation/FINAL_REPORT.md)
- [path relation V2](path_relation/FINAL_REPORT_V2.md)

These reports document measured rejection. Their models are not loaded by the
application. Training and experiment scripts remain research-only and should not
be treated as supported CLI surfaces.

Generated timing JSON, recordings, extracted frames, review sheets, caches, and
model exports are local artifacts unless a small file is deliberately retained
as evidence.
