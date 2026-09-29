"""Freshness-aware, multi-rate scheduler for expensive perception tasks."""
import time
from dataclasses import dataclass, field
from threading import Condition, Event

from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.perception_store import PerceptionStore
from pose.matching import has_fresh_person
from scheduling.task import ComputeTask, TaskPolicy


@dataclass(frozen=True)
class SchedulerMetrics:
    submitted: int
    executed: int
    replaced: int
    dropped_stale: int
    by_type: dict = field(default_factory=dict)


class ComputeScheduler:
    def __init__(self, policies: tuple[TaskPolicy, ...], max_tracking_age: float | None = None):
        self._condition = Condition()
        self._policies = {policy.task_type: policy for policy in policies}
        # Only the newest waiting task of each type is kept; there is no queue.
        self._pending: dict[str, ComputeTask] = {}
        self._last_run_time: dict[str, float] = {}
        self._last_task_type: str | None = None
        self._submitted = 0
        self._executed = 0
        self._replaced = 0
        self._dropped_stale = 0
        self._max_tracking_age = max_tracking_age
        self._duration: dict[str, float] = {}
        self._last_detection_source: float | None = None
        self._counts = {name: dict(submitted=0, executed=0, replaced=0, dropped_stale=0,
                                  budget_deferred=0, skipped=0) for name in self._policies}
        self._last_deferred_frame = {}

    def has_task(self, task_type: str) -> bool:
        return task_type in self._policies

    def seed_cost(self, task_type: str, seconds: float) -> None:
        """Warmup cost, not an executed video observation."""
        with self._condition:
            self._duration[task_type] = max(0.0, seconds)

    def mark_skipped(self, task_type: str) -> None:
        with self._condition:
            self._counts[task_type]['skipped'] += 1

    def submit(self, task: ComputeTask) -> None:
        """Reject stale input; replace waiting work only with a newer frame."""
        with self._condition:
            if task.task_type not in self._policies:
                raise ValueError(f"No policy for task type: {task.task_type}")
            self._submitted += 1
            self._counts[task.task_type]["submitted"] += 1
            policy = self._policies[task.task_type]
            if time.perf_counter() - task.source_timestamp > policy.max_input_age:
                self._dropped_stale += 1
                self._counts[task.task_type]["dropped_stale"] += 1
                return
            existing = self._pending.get(task.task_type)
            if existing is not None:
                if task.source_frame_id <= existing.source_frame_id:
                    return
                self._replaced += 1
                self._counts[task.task_type]["replaced"] += 1
            self._pending[task.task_type] = task
            self._condition.notify_all()

    def get_next(self, stop_event: Event) -> ComputeTask | None:
        """Dispatch the highest-priority fresh task whose start interval elapsed."""
        with self._condition:
            while not stop_event.is_set():
                now = time.perf_counter()
                stale_types = [
                    task_type for task_type, task in self._pending.items()
                    if now - task.source_timestamp > self._policies[task_type].max_input_age
                ]
                for task_type in stale_types:
                    del self._pending[task_type]
                    self._dropped_stale += 1
                    self._counts[task_type]["dropped_stale"] += 1

                ready = []
                earliest_future_time = None
                for task_type, task in self._pending.items():
                    policy = self._policies[task_type]
                    last_run = self._last_run_time.get(task_type)
                    if last_run is None or now >= last_run + policy.min_interval:
                        # Leave time for the next detection before current boxes
                        # expire. Under overload, optional work yields to fresh tracking.
                        if task_type != "detection" and not self._optional_fits(task_type, now):
                            if self._last_deferred_frame.get(task_type) != task.source_frame_id:
                                self._counts[task_type]['budget_deferred'] += 1
                                self._last_deferred_frame[task_type] = task.source_frame_id
                            continue
                        ready.append(task)
                    else:
                        next_allowed = last_run + policy.min_interval
                        if earliest_future_time is None or next_allowed < earliest_future_time:
                            earliest_future_time = next_allowed

                if ready:
                    task = max(ready, key=lambda item: (
                        self._policies[item.task_type].priority,
                        item.source_frame_id,
                    ))
                    # A slow detector may already be due again at completion.
                    # Give another ready type a turn instead of starving it.
                    if task.task_type == self._last_task_type:
                        alternatives = [item for item in ready if item.task_type != task.task_type]
                        if alternatives:
                            task = max(alternatives, key=lambda item: (
                                self._policies[item.task_type].priority, item.source_frame_id))
                    self._last_task_type = task.task_type
                    del self._pending[task.task_type]
                    self._last_run_time[task.task_type] = now
                    return task

                timeout = 0.05
                if earliest_future_time is not None:
                    timeout = max(0.001, min(timeout, earliest_future_time - now))
                self._condition.wait(timeout=timeout)
            return None

    def _optional_fits(self, task_type: str, now: float) -> bool:
        if self._max_tracking_age is None:
            return True
        # Road must never probe blindly ahead of the first detection. Its warmup
        # seeds cost; pose retains its existing first-probe behavior.
        if self._last_detection_source is None:
            return task_type != 'road'
        detection_seconds = self._duration.get('detection')
        optional_seconds = self._duration.get(task_type)
        if detection_seconds is None or optional_seconds is None:
            return task_type != 'road'
        detection_due = self._last_run_time.get('detection', now) + self._policies['detection'].min_interval
        next_detection_finish = max(now + optional_seconds, detection_due) + detection_seconds + 0.02
        return next_detection_finish <= self._last_detection_source + self._max_tracking_age

    def mark_executed(self, task_type: str, processing_seconds: float | None = None,
                      source_timestamp: float | None = None) -> None:
        """Count successful work and learn costs for freshness-aware admission."""
        with self._condition:
            self._executed += 1
            self._counts[task_type]["executed"] += 1
            if processing_seconds is not None:
                previous = self._duration.get(task_type, processing_seconds)
                self._duration[task_type] = .5 * previous + .5 * max(0.0, processing_seconds)
            if task_type == "detection" and source_timestamp is not None:
                self._last_detection_source = source_timestamp

    def metrics(self) -> SchedulerMetrics:
        with self._condition:
            return SchedulerMetrics(
                submitted=self._submitted,
                executed=self._executed,
                replaced=self._replaced,
                dropped_stale=self._dropped_stale,
                by_type={name: dict(counts) for name, counts in self._counts.items()},
            )

    def wake(self) -> None:
        """Wake a waiting compute worker during shutdown."""
        with self._condition:
            self._condition.notify_all()


def scheduler_worker(
    config: VisionConfig,
    frame_buffer: LatestFrameBuffer,
    perception_store: PerceptionStore,
    scheduler: ComputeScheduler,
    stop_event: Event,
) -> None:
    """Offer newest inputs; pose needs people, road needs explicit opt-in."""
    previous_frame_id = -1
    while not stop_event.is_set():
        packet = frame_buffer.get_latest()
        if packet is None or packet.frame_id <= previous_frame_id:
            if frame_buffer.is_finished():
                break
            stop_event.wait(config.empty_poll_seconds if packet is None
                            else config.repeat_poll_seconds)
            continue
        scheduler.submit(ComputeTask(
            task_type="detection",
            source_frame_id=packet.frame_id,
            source_timestamp=packet.timestamp,
            packet=packet,
        ))
        if scheduler.has_task('road'):
            scheduler.submit(ComputeTask('road', packet.frame_id, packet.timestamp, packet))
        state = perception_store.get_latest()
        if has_fresh_person(state, packet.frame_id, packet.timestamp, config.max_render_age):
            scheduler.submit(ComputeTask(
                task_type="pose", source_frame_id=packet.frame_id,
                source_timestamp=packet.timestamp, packet=packet,
            ))
        previous_frame_id = packet.frame_id
