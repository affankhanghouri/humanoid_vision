"""Process only fresh latest frames; source timestamps drive the tracker."""
import time
from threading import Event
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.result_store import ResultStore
from core.metrics import RollingLatency
from core.types import TrackingResult
from detectors.base import Detector
from tracking.tracker import MultiObjectTracker


def detector_worker(config: VisionConfig, detector: Detector,
                    tracker: MultiObjectTracker, frame_buffer: LatestFrameBuffer,
                    result_store: ResultStore, stop_event: Event) -> None:
    previous_frame_id = -1
    count = 0
    started = time.perf_counter()
    latency = RollingLatency(config.metrics_window)
    while not stop_event.is_set():
        packet = frame_buffer.get_latest()
        if packet is None or packet.frame_id == previous_frame_id:
            if frame_buffer.is_finished():
                break
            stop_event.wait(config.empty_poll_seconds if packet is None
                            else config.repeat_poll_seconds)
            continue
        start = time.perf_counter()
        detections = detector.detect(packet.frame)
        inference_ms = (time.perf_counter() - start) * 1000
        latency.add(inference_ms)
        tracker.update(detections, packet.timestamp)
        count += 1
        hz = count / (time.perf_counter() - started)
        tracks = tracker.snapshots()
        result = TrackingResult(packet.frame_id, packet.timestamp, tracks,
                                inference_ms, hz, latency.average, latency.p95)
        result_store.publish(result)
        print(f"DET frame={packet.frame_id} inference={inference_ms:.1f}ms "
              f"avg={result.average_ms:.1f}ms p95={result.p95_ms:.1f}ms "
              f"hz={hz:.2f} detections={len(detections)} tracks={len(tracks)}")
        previous_frame_id = packet.frame_id
