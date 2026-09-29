"""Measure detection, pose, freshness and drawing in the real pipeline.

Use --headless to omit the GUI. Warmup is excluded from throughput.
"""
import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, os.environ.get('VISION_BENCH_SRC', str(Path(__file__).resolve().parents[1] / 'src')))
from config import VisionConfig
from pipeline.vision_pipeline import VisionPipeline
from rendering.demo_renderer import DemoRenderer
from rendering.renderer import Renderer
from core.perception_store import PerceptionStore
from scheduling.scheduler import ComputeScheduler
from road.segmentation import RoadSegmenter
from tracking.tracker import MultiObjectTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--road-model', help='Explicit opt-in: full-resolution road-only ONNX')
    parser.add_argument('--risk-demo', action='store_true')
    parser.add_argument('--save-frame')
    parser.add_argument('--exit-key', choices=('q', 'esc'), default='q')
    parser.add_argument('--output', default='pipeline_measurement.json')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--onnx-threads', type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = VisionConfig()
    if os.environ.get('VISION_BENCH_SRC'):
        config = replace(config, video_path=str(root/'test_data/test_video.mp4'),
                         model_path=str(root/'models/yolo26n.onnx'),
                         pose_model_path=str(root/'models/yolo26n-pose-320.onnx'))
    if args.onnx_threads is not None:
        config = replace(config, onnx_intra_op_threads=args.onnx_threads)
    if args.risk_demo:
        config = replace(config, render_mode='demo_risk', risk_heatmap_enabled=True,
                         road_demo_enabled=bool(args.road_model))
    road_model = RoadSegmenter(args.road_model, threads=2) if args.road_model else None
    original_capture = cv2.VideoCapture
    def capture(source):
        cap = original_capture(source)
        if args.start_frame and not isinstance(source, int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        return cap
    shown, rendering, tracking, ages = [], [], [], []
    saved_frame = False
    motion_times = []
    detection, poses, frame_ids, result_ids = [], [], [], []
    fresh = stale = missing = 0
    road_samples, road_ages, wall_ages, road_wall_ages = [], [], [], []
    busy, dispatches, cpu_samples = [], [], []
    scheduler_ref = []
    road_missing = 0
    def system_ticks():
        values = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        return sum(values), values[3]+values[4]
    original_road = PerceptionStore.publish_road
    original_mark = ComputeScheduler.mark_executed
    original_next = ComputeScheduler.get_next
    def publish_road(self, observation):
        road_samples.append((
            time.perf_counter(), observation.inference_ms,
            observation.processing_latency_ms, observation.preprocessing_ms,
            observation.postprocessing_ms,
        ))
        return original_road(self, observation)
    def mark_executed(self, kind, processing_seconds=None, source_timestamp=None):
        end = time.perf_counter()
        if processing_seconds is not None:
            busy.append((end-processing_seconds, end, kind, processing_seconds*1000))
        scheduler_ref[:] = [self]
        return original_mark(self, kind, processing_seconds, source_timestamp)
    def get_next(self, stop_event):
        task = original_next(self, stop_event)
        if task is not None:
            dispatches.append(dict(kind=task.task_type, frame_id=task.source_frame_id,
                                   source=task.source_timestamp, start=time.perf_counter()))
        return task
    original_show, original_wait = cv2.imshow, cv2.waitKey
    original_tracking = PerceptionStore.publish_tracking
    original_pose = PerceptionStore.publish_pose
    renderer_type = DemoRenderer if config.render_mode in {'demo','demo_risk'} else Renderer
    original_render = renderer_type.render
    original_update = MultiObjectTracker.update

    def publish_tracking(self, state):
        detection.append((time.perf_counter(), state.inference_ms, state.processing_latency_ms))
        return original_tracking(self, state)

    def publish_pose(self, observations, meta, inference_ms, *metrics):
        poses.append((time.perf_counter(), inference_ms, meta.processing_latency_ms))
        return original_pose(self, observations, meta, inference_ms, *metrics)

    def update(self, *pos, **kw):
        start = time.perf_counter()
        result = original_update(self, *pos, **kw)
        tracking.append((time.perf_counter() - start) * 1000)
        return result

    def render(self, packet, state, fps):
        nonlocal fresh, stale, missing, road_missing
        frame_ids.append(packet.frame_id)
        if state is None:
            missing += 1
        else:
            result_ids.append(state.source_frame_id)
            ages.append(state.age_at(packet.timestamp) * 1000)
            if state.is_valid_for(packet.frame_id, packet.timestamp, config.max_render_age):
                fresh += 1
            else:
                stale += 1
        if state is not None:
            wall_ages.append((time.perf_counter()-state.source_timestamp)*1000)
        if state is not None and state.road is not None:
            road_ages.append(state.road.age_at(packet.timestamp)*1000)
            road_wall_ages.append((time.perf_counter()-state.road.source_timestamp)*1000)
        else:
            road_missing += 1
        start = time.perf_counter()
        result = original_render(self, packet, state, fps)
        rendering.append((time.perf_counter()-start) * 1000)
        return result

    def show(name, frame):
        nonlocal saved_frame
        shown.append(time.perf_counter())
        if args.save_frame and not saved_frame and len(shown) > 1 and time.perf_counter()-shown[0] >= max(2., args.seconds*.7):
            Path(args.save_frame).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.save_frame, frame)
            saved_frame = True
        cpu_samples.append((time.process_time(), *system_ticks()))
        if not args.headless:
            original_show(name, frame)

    def wait(delay):
        key = -1 if args.headless else original_wait(delay)
        if args.seconds > 0 and shown and time.perf_counter()-shown[0] >= args.seconds:
            return ord('q') if args.exit_key == 'q' else 27
        return key

    with ExitStack() as stack:
        if args.headless:
            stack.enter_context(patch.object(cv2, 'namedWindow'))
            stack.enter_context(patch.object(cv2, 'setWindowProperty'))
        if getattr(config, 'motion_enabled', False):
            from tracking.motion import BoxMotionHistory
            original_estimate = BoxMotionHistory._estimate
            def estimate(self, *args):
                start = time.perf_counter()
                result = original_estimate(self, *args)
                motion_times.append((time.perf_counter()-start)*1000)
                return result
            stack.enter_context(patch.object(BoxMotionHistory, '_estimate', estimate))
        for target, name, replacement in (
            (PerceptionStore, 'publish_road', publish_road),
            (ComputeScheduler, 'mark_executed', mark_executed),
            (ComputeScheduler, 'get_next', get_next),
            (PerceptionStore, 'publish_tracking', publish_tracking),
            (PerceptionStore, 'publish_pose', publish_pose),
            (MultiObjectTracker, 'update', update),
            (renderer_type, 'render', render),
            (cv2, 'imshow', show), (cv2, 'waitKey', wait),
            (cv2, 'VideoCapture', capture),
        ):
            stack.enter_context(patch.object(target, name, replacement))
        VisionPipeline(config, road_estimator=road_model).run()

    def stats(values):
        return dict(count=len(values), average_ms=float(np.mean(values)),
                    median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                    p99_ms=float(np.percentile(values, 99))) if values else {}

    duration = shown[-1]-shown[0] if len(shown) > 1 else 0
    def rate(samples):
        return sum(shown[0] <= t <= shown[-1] for t, *_ in samples)/duration if duration else 0
    result = dict(duration_seconds=duration, headless=args.headless,
                  onnx_threads=getattr(config, "onnx_intra_op_threads", None),
                  displayed_frames=len(shown), display_fps=(len(shown)-1)/duration if duration else 0,
                  display_intervals=stats(list(np.diff(shown)*1000)),
                  renderer=stats(rendering), tracker=stats(tracking), motion=stats(motion_times),
                  detector_hz=rate(detection), pose_hz=rate(poses),
                  detection=stats([s[1] for s in detection]), pose=stats([s[1] for s in poses]),
                  capture_to_detection=stats([s[2] for s in detection]),
                  state_age=stats(ages), fresh_frames=fresh, stale_frames=stale, missing_frames=missing,
                  stale_percent=100*stale/(fresh+stale) if fresh+stale else 0,
                  skipped_display_frames=sum(max(0,b-a-1) for a,b in zip(frame_ids,frame_ids[1:])),
                  unique_results_seen=len(set(result_ids)))
    active = [item for item in busy if shown[0] <= item[1] <= shown[-1]]
    occupied = sum(max(0, min(end, shown[-1])-max(start, shown[0])) for start,end,*_ in busy)
    process_cpu = (cpu_samples[-1][0]-cpu_samples[0][0])/duration*100
    ticks = cpu_samples[-1][1]-cpu_samples[0][1]
    idle = cpu_samples[-1][2]-cpu_samples[0][2]
    result.update(
        road_enabled=road_model is not None, road_request_interval_seconds=1.0,
        road_hz=rate(road_samples), road_inference=stats([s[1] for s in road_samples]),
        road_preprocessing=stats([s[3] for s in road_samples]),
        road_postprocessing=stats([s[4] for s in road_samples]),
        road_source_age=stats(road_ages), road_wall_source_age=stats(road_wall_ages),
        detector_wall_source_age=stats(wall_ages), road_missing_frames=road_missing,
        road_capture_to_result=stats([s[2] for s in road_samples]),
        task_total={kind: stats([v[3] for v in active if v[2]==kind]) for kind in ('detection','road','pose')},
        scheduler=scheduler_ref[0].metrics().__dict__ if scheduler_ref else {},
        worker_utilization_percent=occupied/duration*100,
        process_cpu_one_core_percent=process_cpu,
        process_cpu_machine_percent=process_cpu/(os.cpu_count() or 1),
        system_cpu_percent=100*(1-idle/ticks) if ticks else None,
        dispatches=dispatches, task_intervals=busy,
        measurement_start=shown[0], measurement_end=shown[-1],
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('MEASURE', json.dumps({k:v for k,v in result.items() if k not in ('dispatches','task_intervals')}))


if __name__ == '__main__':
    main()
