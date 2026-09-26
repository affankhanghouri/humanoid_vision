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
from tracking.tracker import MultiObjectTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
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
    original_capture = cv2.VideoCapture
    def capture(source):
        cap = original_capture(source)
        if args.start_frame and not isinstance(source, int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        return cap
    shown, rendering, tracking, ages = [], [], [], []
    motion_times = []
    detection, poses, frame_ids, result_ids = [], [], [], []
    fresh = stale = missing = 0
    original_show, original_wait = cv2.imshow, cv2.waitKey
    original_tracking = PerceptionStore.publish_tracking
    original_pose = PerceptionStore.publish_pose
    renderer_type = DemoRenderer if config.render_mode == 'demo' else Renderer
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
        nonlocal fresh, stale, missing
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
        start = time.perf_counter()
        result = original_render(self, packet, state, fps)
        rendering.append((time.perf_counter()-start) * 1000)
        return result

    def show(name, frame):
        shown.append(time.perf_counter())
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
            (PerceptionStore, 'publish_tracking', publish_tracking),
            (PerceptionStore, 'publish_pose', publish_pose),
            (MultiObjectTracker, 'update', update),
            (renderer_type, 'render', render),
            (cv2, 'imshow', show), (cv2, 'waitKey', wait),
            (cv2, 'VideoCapture', capture),
        ):
            stack.enter_context(patch.object(target, name, replacement))
        VisionPipeline(config).run()

    def stats(values):
        return dict(count=len(values), average_ms=float(np.mean(values)),
                    p95_ms=float(np.percentile(values, 95))) if values else {}

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
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print('MEASURE', json.dumps(result))


if __name__ == '__main__':
    main()
