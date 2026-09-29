"""Run detection, pose and opt-in road through one serial compute worker."""
import logging
import time
from dataclasses import replace
from threading import Event
from config import VisionConfig
from core.metrics import RollingLatency
from core.perception_state import ObservationMeta, PerceptionState
from core.perception_store import PerceptionStore
from detectors.base import Detector
from pose.base import PoseEstimator
from pose.matching import has_fresh_person, match_pose_to_entities
from scheduling.scheduler import ComputeScheduler
from tracking.tracker import MultiObjectTracker


LOGGER = logging.getLogger(__name__)


def compute_worker(
    config: VisionConfig,
    scheduler: ComputeScheduler,
    detector: Detector,
    pose_estimator: PoseEstimator,
    tracker: MultiObjectTracker,
    perception_store: PerceptionStore,
    stop_event: Event,
    road_estimator=None,
) -> None:
    detection_count = pose_count = 0
    started = time.perf_counter()
    pose_started = None
    detection_latency = RollingLatency(config.metrics_window)
    pose_latency = RollingLatency(config.metrics_window)

    while not stop_event.is_set():
        task = scheduler.get_next(stop_event)
        if task is None:
            break
        packet = task.packet

        if task.task_type == 'detection':
            inference_start = time.perf_counter()
            detections = detector.detect(packet.frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            detection_latency.add(inference_ms)
            tracker.update(detections, task.source_timestamp)
            entities = tracker.snapshots()
            finished = time.perf_counter()
            detection_count += 1
            hz = detection_count / (finished-started) if finished > started else 0.0
            state = PerceptionState(
                source_frame_id=task.source_frame_id,
                source_timestamp=task.source_timestamp,
                produced_timestamp=finished, entities=entities,
                inference_ms=inference_ms, detector_hz=hz,
                average_ms=detection_latency.average, p95_ms=detection_latency.p95,
            )
            perception_store.publish_tracking(state)
            scheduler.mark_executed(task.task_type, finished-inference_start, task.source_timestamp)
            metrics = scheduler.metrics()
            LOGGER.debug(
                'DET frame=%d inference=%.1fms pipeline=%.1fms avg=%.1fms '
                'p95=%.1fms hz=%.2f detections=%d entities=%d sched_sub=%d '
                'sched_exec=%d replaced=%d stale=%d',
                task.source_frame_id, inference_ms, state.processing_latency_ms,
                state.average_ms, state.p95_ms, hz, len(detections), len(entities),
                metrics.submitted, metrics.executed, metrics.replaced,
                metrics.dropped_stale,
            )
            continue

        if task.task_type == 'road':
            if road_estimator is None:
                scheduler.mark_skipped('road')
                continue
            start = time.perf_counter()
            observation = road_estimator.observe(packet.frame, task.source_frame_id, task.source_timestamp)
            perception_store.publish_road(observation)
            scheduler.mark_executed('road', time.perf_counter()-start, task.source_timestamp)
            continue

        if task.task_type == 'pose':
            state = perception_store.get_latest()
            # A pending pose task can outlive the person that triggered it.
            if not has_fresh_person(state, task.source_frame_id, task.source_timestamp,
                                    config.max_render_age):
                scheduler.mark_skipped("pose")
                continue
            inference_start = time.perf_counter()
            if pose_started is None:
                pose_started = inference_start
            people = pose_estimator.estimate(packet.frame)
            inference_ms = (time.perf_counter() - inference_start) * 1000
            pose_latency.add(inference_ms)
            source_meta = ObservationMeta(task.source_frame_id, task.source_timestamp, task.source_timestamp)
            matched = match_pose_to_entities(people, state, source_meta, config)
            finished = time.perf_counter()
            meta = replace(source_meta, produced_timestamp=finished)
            matched = {entity_id: replace(pose, meta=meta) for entity_id, pose in matched.items()}
            pose_count += 1
            hz = pose_count / (finished-pose_started) if finished > pose_started else 0.0
            perception_store.publish_pose(matched, meta, inference_ms, hz,
                                          pose_latency.average, pose_latency.p95)
            scheduler.mark_executed(task.task_type, finished-inference_start, task.source_timestamp)
            LOGGER.debug(
                'POSE frame=%d inference=%.1fms pipeline=%.1fms hz=%.2f '
                'people=%d matched=%d',
                task.source_frame_id, inference_ms, meta.processing_latency_ms,
                hz, len(people), len(matched),
            )
            continue

        LOGGER.warning('Unknown compute task: %s', task.task_type)
