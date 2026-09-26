"""Replay identical delayed detections to compare display-box alignment.

Reference boxes are same-frame YOLO outputs, not human-annotated ground truth.
Use --populate once, then reuse the cache with each source tree via VISION_BENCH_SRC.
"""
import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.environ.get('VISION_BENCH_SRC', str(ROOT/'src')))
from config import VisionConfig
from core.types import Detection, FramePacket
from core.perception_state import PerceptionState
from rendering.demo_renderer import DemoRenderer
from tracking.tracker import MultiObjectTracker, calculate_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--populate', action='store_true')
    parser.add_argument('--video', default=str(ROOT/'test_data/test_video.mp4'))
    parser.add_argument('--start-frame', type=int, default=144)
    parser.add_argument('--end-frame', type=int, default=672)
    parser.add_argument('--reference-stride', type=int, default=4)
    parser.add_argument('--detection-stride', type=int, default=8)
    parser.add_argument('--delay-frames', type=int, default=8)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    config = VisionConfig()
    if args.populate:
        from detectors.factory import create_detector
        detector = create_detector(config)
        detector.warmup()
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        records = []
        for index in range(args.start_frame, args.end_frame+1):
            ok, frame = cap.read()
            if not ok:
                break
            if (index-args.start_frame) % args.reference_stride:
                continue
            records.append(dict(frame_id=index, detections=[asdict(d) for d in detector.detect(frame)]))
        cap.release()
        Path(args.cache).write_text(json.dumps(dict(video=args.video, fps=fps, width=width, height=height,
                                                    image_size=config.image_size, confidence=min(config.confidence, config.track_low_confidence),
                                                    records=records)))
    cache = json.loads(Path(args.cache).read_text())
    records = {r['frame_id']: r for r in cache['records']}
    first, last = min(records), max(records)
    required = set(range(first, last+1, args.detection_stride))
    if required-set(records):
        raise ValueError('Cache lacks frames required by detection stride')
    fps = cache['fps']
    cap = cv2.VideoCapture(cache['video'])
    if not cap.isOpened():
        raise ValueError('Source video is required to evaluate image-motion compensation')
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tracker, renderer = MultiObjectTracker(config), DemoRenderer(config)
    capture_motion = type(renderer.motion)(config) if hasattr(renderer, "motion") else None
    pending, boxes = {}, {}
    state = None
    reference_count = display_count = hits = matched = 0
    summed_iou = 0.0
    centers, matched_ious = [], []
    for index in range(first, last+1):
        timestamp = index/fps
        ok, blank = cap.read()
        if not ok:
            raise ValueError(f'Source video ended before frame {index}')
        if index in required:
            tracker.update([Detection(**d) for d in records[index]['detections']
                            if d['confidence'] >= getattr(config, 'track_low_confidence', config.confidence)], timestamp)
            pending[index+args.delay_frames] = PerceptionState(
                index, timestamp, timestamp+args.delay_frames/fps, tracker.snapshots(), 0, fps/args.detection_stride, 0, 0)
        if index in pending:
            state = pending.pop(index)
        packet = FramePacket(index, timestamp, blank)
        views = []
        if capture_motion is not None:
            packet.motion = capture_motion.snapshot()
            capture_motion.update(FramePacket(index, timestamp, blank))
            renderer.motion.update(packet)
        if state is not None and state.is_valid_for(index, timestamp, config.max_render_age):
            views = renderer.build_views(blank, packet, state, state.age_at(timestamp))
        renderer.box_smoother.keep_only([v.entity.entity_id for v in views])
        current = [dict(id=v.entity.entity_id, class_name=v.entity.class_name, bbox=list(v.box)) for v in views]
        boxes[index] = current
        if index not in records or index < first+2*args.detection_stride:
            continue
        reference = [d for d in records[index]['detections'] if d['class_name'] in ('car', 'person')
                     and d['confidence'] >= config.confidence]
        predicted = [d for d in current if d['class_name'] in ('car', 'person')]
        reference_count += len(reference)
        display_count += len(predicted)
        if not reference or not predicted:
            continue
        matrix = np.array([[calculate_iou(a['bbox'], b['bbox']) if a['class_name']==b['class_name'] else 0
                            for b in predicted] for a in reference])
        rows, cols = linear_sum_assignment(-matrix)
        for row, col in zip(rows, cols):
            iou = float(matrix[row, col])
            summed_iou += iou
            hits += int(iou >= .5)
            if iou < .1:
                continue
            matched += 1
            matched_ious.append(iou)
            a,b = reference[row]['bbox'], predicted[col]['bbox']
            centers.append(float(np.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)))
    cap.release()
    result = dict(reference='same-frame YOLO car/person detections, not labeled ground truth',
                  start_frame=first, end_frame=last, detection_hz=fps/args.detection_stride,
                  delay_ms=args.delay_frames/fps*1000, reference_boxes=reference_count, display_boxes=display_count,
                  mean_iou_per_reference=summed_iou/reference_count if reference_count else 0,
                  matched_mean_iou=float(np.mean(matched_ious)) if matched_ious else 0,
                  matched_center_error_px=float(np.mean(centers)) if centers else 0,
                  aligned_at_iou50=hits, reference_alignment_percent=100*hits/reference_count if reference_count else 0,
                  display_alignment_percent=100*hits/display_count if display_count else 0,
                  unmatched_reference=reference_count-matched, unmatched_display=display_count-matched)
    Path(args.output+'.json').write_text(json.dumps(result, indent=2)+'\n')
    Path(args.output+'_boxes.json').write_text(json.dumps(boxes))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
