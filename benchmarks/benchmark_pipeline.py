"""Measure the actual GUI pipeline, including inference, tracking and rendering.

Run with --seconds 0 to reach EOF, or --exit-key esc to exercise Esc.
Measurements start after warm-up. No inference settings are overridden.
"""
import sys
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
# Allow direct execution without installing the src tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import VisionConfig
from pipeline.vision_pipeline import VisionPipeline
from detectors.factory import create_detector
from tracking.tracker import MultiObjectTracker
from rendering.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--exit-key', choices=('q', 'esc'), default='q')
    parser.add_argument('--output', default='pipeline_measurement.json')
    args = parser.parse_args()
    shown, inference, tracking, rendering, rss = [], [], [], [], []
    frame_ids, result_ids, track_counts = [], [], []
    original_show, original_wait = cv2.imshow, cv2.waitKey

    def timed(method, samples):
        def call(*pos, **kw):
            start = time.perf_counter()
            result = method(*pos, **kw)
            samples.append((time.perf_counter() - start) * 1000)
            return result
        return call

    def measured_detector(config):
        detector = create_detector(config)
        detector.detect = timed(detector.detect, inference)
        return detector

    render = timed(Renderer.render, rendering)
    def record_render(self, packet, result, fps):
        frame_ids.append(packet.frame_id)
        if result is not None:
            result_ids.append(result.frame_id)
            track_counts.append(len(result.tracks))
        return render(self, packet, result, fps)

    def show(name, frame):
        now = time.perf_counter()
        shown.append(now)
        if not rss or now - rss[-1][0] >= 1:
            # Linux resident memory, sampled after model warm-up.
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    rss.append((now, int(line.split()[1]) / 1024))
                    break
        original_show(name, frame)

    def wait(delay):
        key = original_wait(delay)
        if args.seconds > 0 and shown and time.perf_counter() - shown[0] >= args.seconds:
            return ord('q') if args.exit_key == 'q' else 27
        return key

    with patch('pipeline.vision_pipeline.create_detector', measured_detector), patch.object(MultiObjectTracker, 'update', timed(MultiObjectTracker.update, tracking)), patch.object(Renderer, 'render', record_render), patch('cv2.imshow', show), patch('cv2.waitKey', wait):
        VisionPipeline(VisionConfig()).run()
    def stats(values):
        return dict(count=len(values), average_ms=float(np.mean(values)),
                    median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95))) if len(values) else {}
    duration = shown[-1] - shown[0] if len(shown) > 1 else 0
    result = dict(duration_seconds=duration, displayed_frames=len(shown),
                  display_fps=(len(shown)-1)/duration if duration else 0,
                  display_intervals=stats(np.diff(shown)*1000),
                  inference=stats(inference), tracker=stats(tracking), renderer=stats(rendering),
                  detector_hz=len(inference)/duration if duration else 0,
                  first_frame=frame_ids[0] if frame_ids else None,
                  last_frame=frame_ids[-1] if frame_ids else None,
                  skipped_display_frames=sum(max(0,b-a-1) for a,b in zip(frame_ids,frame_ids[1:])),
                  unique_results_seen=len(set(result_ids)), max_visible_tracks=max(track_counts, default=0),
                  rss_mb=[dict(seconds=t-shown[0], mb=mb) for t,mb in rss])
    Path(args.output).write_text(json.dumps(result, indent=2))
    print('MEASURE', json.dumps(result))


if __name__ == '__main__':
    main()
