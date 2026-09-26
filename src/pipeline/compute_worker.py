"""Run detection and pose serially through one heavy-compute lane."""
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


def compute_worker(
    config: VisionConfig,
    scheduler: ComputeScheduler,
    detector: Detector,
    pose_estimator: PoseEstimator,
    tracker: MultiObjectTracker,
    perception_store: PerceptionStore,
    stop_event: Event,
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
            print(f'DET frame={task.source_frame_id} inference={inference_ms:.1f}ms '
                  f'pipeline={state.processing_latency_ms:.1f}ms '
                  f'avg={state.average_ms:.1f}ms p95={state.p95_ms:.1f}ms '
                  f'hz={hz:.2f} detections={len(detections)} entities={len(entities)} '
                  f'sched_sub={metrics.submitted} sched_exec={metrics.executed} '
                  f'replaced={metrics.replaced} stale={metrics.dropped_stale}')
            continue

        if task.task_type == 'pose':
            state = perception_store.get_latest()
            # A pending pose task can outlive the person that triggered it.
            if not has_fresh_person(state, task.source_frame_id, task.source_timestamp,
                                    config.max_render_age):
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
            print(f'POSE frame={task.source_frame_id} inference={inference_ms:.1f}ms '
                  f'pipeline={meta.processing_latency_ms:.1f}ms hz={hz:.2f} '
                  f'people={len(people)} matched={len(matched)}')
            continue

        print(f'Warning: unknown compute task {task.task_type}')