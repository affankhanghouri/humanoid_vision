"""Run detection on the newest useful frame and publish perception state."""

import time
from threading import Event

from config import VisionConfig

from core.latest_frame import LatestFrameBuffer
from core.metrics import RollingLatency
from core.perception_state import PerceptionState
from core.perception_store import PerceptionStore

from detectors.base import Detector
from tracking.tracker import MultiObjectTracker


def detector_worker(
    config: VisionConfig,
    detector: Detector,
    tracker: MultiObjectTracker,
    frame_buffer: LatestFrameBuffer,
    perception_store: PerceptionStore,
    stop_event: Event,
) -> None:

    previous_frame_id = -1

    count = 0
    started = time.perf_counter()

    latency = RollingLatency(
        config.metrics_window
    )

    while not stop_event.is_set():

        packet = frame_buffer.get_latest()

        # --------------------------------------------------
        # No usable new frame
        # --------------------------------------------------

        if (
            packet is None
            or packet.frame_id == previous_frame_id
        ):

            if frame_buffer.is_finished():
                break

            wait_time = (
                config.empty_poll_seconds
                if packet is None
                else config.repeat_poll_seconds
            )

            stop_event.wait(wait_time)

            continue

        # --------------------------------------------------
        # Detection
        # --------------------------------------------------

        inference_start = time.perf_counter()

        detections = detector.detect(
            packet.frame
        )

        inference_ms = (
            time.perf_counter()
            - inference_start
        ) * 1000.0

        latency.add(inference_ms)

        # --------------------------------------------------
        # Tracking
        # --------------------------------------------------

        tracker.update(
            detections,
            packet.timestamp,
        )

        # Tracker snapshots are already immutable PerceptionEntity objects.
        entities = tracker.snapshots()

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        count += 1

        produced_timestamp = time.perf_counter()

        elapsed = (
            produced_timestamp - started
        )

        detector_hz = (
            count / elapsed
            if elapsed > 0
            else 0.0
        )

        # --------------------------------------------------
        # Publish immutable perception state
        # --------------------------------------------------

        state = PerceptionState(
            source_frame_id=packet.frame_id,
            source_timestamp=packet.timestamp,
            produced_timestamp=produced_timestamp,

            entities=entities,

            inference_ms=inference_ms,
            detector_hz=detector_hz,
            average_ms=latency.average,
            p95_ms=latency.p95,
        )

        perception_store.publish(state)

        # --------------------------------------------------
        # Debug output
        # --------------------------------------------------

        print(
            f"DET "
            f"frame={packet.frame_id} "
            f"inference={inference_ms:.1f}ms "
            f"pipeline={state.processing_latency_ms:.1f}ms "
            f"avg={state.average_ms:.1f}ms "
            f"p95={state.p95_ms:.1f}ms "
            f"hz={detector_hz:.2f} "
            f"detections={len(detections)} "
            f"entities={len(entities)}"
        )

        previous_frame_id = packet.frame_id
