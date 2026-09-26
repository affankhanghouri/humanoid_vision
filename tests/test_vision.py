"""Behavioral regression and lifecycle checks; no model dependency."""
import sys
import time
import json
import threading
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import VisionConfig
from core.types import Detection, FramePacket
from core.perception_state import PerceptionEntity, PerceptionState
from core.latest_frame import LatestFrameBuffer
from core.perception_store import PerceptionStore
from core.metrics import RollingLatency
from detectors.factory import create_detector
from pipeline.compute_worker import compute_worker
from scheduling.scheduler import ComputeScheduler, scheduler_worker
from scheduling.task import ComputeTask, TaskPolicy
from pipeline.vision_pipeline import VisionPipeline
from rendering.renderer import Renderer
from tracking.tracker import MultiObjectTracker, Track
from tracking.smoothing import DisplayBoxSmoother


class VisionTests(unittest.TestCase):
    def setUp(self):
        for name in ('namedWindow', 'setWindowProperty'):
            window_patch = patch(f'pipeline.vision_pipeline.cv2.{name}')
            window_patch.start()
            self.addCleanup(window_patch.stop)
        self.pose = Mock()
        self.pose.estimate.return_value = ()
        patcher = patch('pipeline.vision_pipeline.UltralyticsPoseEstimator', return_value=self.pose)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_perception_store_publishes_immutable_snapshots(self):
        store = PerceptionStore()
        self.assertIsNone(store.get_latest())
        entity = PerceptionEntity(1, 'person', .8, 100, 100, 0, 0, 40, 40)
        first = PerceptionState(1, 10.0, 10.2, (entity,), 0, 0, 0, 0)
        store.publish(first)
        self.assertIs(store.get_latest(), first)
        with self.assertRaises(FrozenInstanceError):
            first.source_timestamp = 11.0
        with self.assertRaises(FrozenInstanceError):
            entity.center_x = 200
        second = PerceptionState(2, 11.0, 11.2, (), 0, 0, 0, 0)
        store.publish(second)
        self.assertEqual(store.get_latest(), second)
        self.assertEqual(first.entities, (entity,))

    def test_produced_timestamp_is_recorded_after_tracking(self):
        frames, store = LatestFrameBuffer(), PerceptionStore()
        frames.publish(FramePacket(7, 10.0, np.zeros((10, 10, 3), np.uint8)))
        frames.mark_finished()
        clock = [10.1]
        config = replace(VisionConfig(), min_hits=1)
        tracker = MultiObjectTracker(config)
        snapshot = tracker.snapshots

        class Detector:
            def detect(self, frame):
                clock[0] = 10.2
                return [Detection((0, 0, 10, 10), 0, 'person', .8)]

        def snapshots():
            entities = snapshot()
            clock[0] = 10.3
            return entities

        scheduler, stop = ComputeScheduler((TaskPolicy('detection', 100, 0, float('inf')),)), threading.Event()
        packet = frames.get_latest()
        scheduler.submit(ComputeTask('detection', packet.frame_id, packet.timestamp, packet))
        publish = store.publish_tracking

        def publish_and_stop(state):
            publish(state)
            stop.set()

        with patch('pipeline.compute_worker.time.perf_counter', side_effect=lambda: clock[0]), patch.object(tracker, 'snapshots', side_effect=snapshots), patch.object(store, 'publish_tracking', side_effect=publish_and_stop):
            compute_worker(config, scheduler, Detector(), self.pose, tracker, store, stop)
        state = store.get_latest()
        self.assertEqual(state.source_frame_id, 7)
        self.assertEqual(state.source_timestamp, 10.0)
        self.assertEqual(state.produced_timestamp, 10.3)
        self.assertIsInstance(state.entities, tuple)
        self.assertEqual(len(state.entities), 1)
        self.assertIsInstance(state.entities[0], PerceptionEntity)
        self.assertEqual(state.entities[0].entity_id, tracker.tracks[0].id)
        self.assertEqual(tracker.tracks[0].last_state_time, 10.0)

    def test_original_smoother_regression(self):
        # Tracking association/lifetimes intentionally changed; smoothing remains
        # compatible with the recorded legacy tuning.
        data = json.loads(Path(__file__).with_name('golden_tracking.json').read_text())
        config = replace(VisionConfig(), smoothing_alphas=(.12, .30, .55, .80),
                         display_size_alpha=.18)
        smoother = DisplayBoxSmoother(config)
        for box, expected in zip(data['boxes'], data['smoothed']):
            np.testing.assert_array_equal(smoother.smooth(1, box), expected)
        smoother.keep_only([])
        self.assertEqual(smoother.boxes, {})

    def test_busy_compute_skips_to_latest_scheduled_frame(self):
        scheduler, store, stop = ComputeScheduler((TaskPolicy('detection', 100, 0, float('inf')),)), PerceptionStore(), threading.Event()
        started, release, done = threading.Event(), threading.Event(), threading.Event()
        seen, published, errors = [], [], []
        image = np.zeros((10, 10, 3), np.uint8)

        class SlowDetector:
            def detect(self, frame):
                seen.append(frame)
                if len(seen) == 1:
                    started.set()
                    if not release.wait(2):
                        raise RuntimeError('test release timed out')
                return []

        def submit(frame_id):
            packet = FramePacket(frame_id, frame_id * .1, image)
            scheduler.submit(ComputeTask('detection', packet.frame_id, packet.timestamp, packet))

        publish = store.publish_tracking
        def record(state):
            publish(state)
            published.append(state.source_frame_id)
            if len(published) == 2:
                done.set()
                stop.set()

        def run():
            try:
                compute_worker(VisionConfig(), scheduler, SlowDetector(), self.pose, MultiObjectTracker(), store, stop)
            except Exception as exc:
                errors.append(exc)
                done.set()

        submit(100)
        with patch.object(store, 'publish_tracking', side_effect=record):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(started.wait(2))
                for frame_id in range(101, 105):
                    submit(frame_id)
                release.set()
                self.assertTrue(done.wait(3))
            finally:
                release.set()
                stop.set()
                scheduler.wake()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(published, [100, 104])
        self.assertEqual(store.get_latest().source_timestamp, 10.4)
        self.assertTrue(all(frame is image for frame in seen))
        self.assertFalse(image.any())

    def test_renderer_preserves_shared_frame_and_limits_age(self):
        image = np.zeros((250, 250, 3), np.uint8)
        track = PerceptionEntity(1, 'person', .8, 100, 100, 100, 0, 40, 40)
        result = PerceptionState(1, 0, .4, (track,), 0, 0, 0, 0)
        renderer = Renderer(VisionConfig())
        rendered = renderer.render(FramePacket(2, .5, image), result, 24)
        self.assertTrue(rendered.any())
        self.assertFalse(image.any())
        np.testing.assert_array_equal(renderer.smoother.boxes[1], (130, 80, 170, 120))
        with patch('rendering.renderer.draw_entity') as draw:
            renderer.render(FramePacket(3, 1, image), result, 24)
            draw.assert_not_called()

    def test_renderer_freshness_boundaries(self):
        image = np.zeros((250, 250, 3), np.uint8)
        entity = PerceptionEntity(1, 'person', .8, 100, 100, 0, 0, 40, 40)
        packet = FramePacket(10, 10.0, image)
        for frame_id, source_time, accepted in (
            (10, 10.0, True),   # Same frame, zero age.
            (9, 9.25, True),    # Exactly max_render_age.
            (9, 9.249, False),  # Stale result.
            (11, 9.5, False),   # Future frame ID despite positive age.
            (9, 10.1, False),   # Negative age despite older frame ID.
        ):
            with self.subTest(frame_id=frame_id, source_time=source_time):
                renderer = Renderer(VisionConfig())
                renderer.smoother.smooth(1, (80, 80, 120, 120))
                state = PerceptionState(frame_id, source_time, source_time + .2, (entity,), 0, 0, 0, 0)
                with patch('rendering.renderer.draw_entity') as draw, patch('cv2.putText') as text:
                    renderer.render(packet, state, 24)
                if accepted:
                    draw.assert_called_once()
                    self.assertEqual(draw.call_args.args[2], packet.timestamp - source_time)
                else:
                    draw.assert_not_called()
                    self.assertEqual(renderer.smoother.boxes, {})
                labels = [call.args[1] for call in text.call_args_list]
                self.assertIn('Processing latency: 200 ms', labels)
        renderer = Renderer(VisionConfig())
        renderer.smoother.smooth(1, (80, 80, 120, 120))
        with patch('rendering.renderer.draw_entity') as draw:
            renderer.render(packet, None, 24)
            draw.assert_not_called()
        self.assertEqual(renderer.smoother.boxes, {})

    def test_metrics_bounded_and_empty(self):
        metric = RollingLatency(3)
        self.assertEqual((metric.count, metric.average, metric.p95), (0, 0, 0))
        for value in (100, 1, 2, 3):
            metric.add(value)
        self.assertEqual((metric.count, metric.last, metric.average, metric.median), (3, 3, 2, 2))
        self.assertAlmostEqual(metric.p95, 2.9)

    def test_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, 'Unknown detector backend'):
            create_detector(replace(VisionConfig(), detector_backend='invalid'))

    def test_warmup_precedes_capture_and_keys_stop_workers(self):
        for key in (ord('q'), 27):
            warmed = threading.Event()
            def capture(config, frames, stop):
                self.assertTrue(warmed.is_set())
                frames.publish(FramePacket(1, time.perf_counter(), np.zeros((20, 20, 3), np.uint8)))
                stop.wait(2)
                frames.mark_finished()
            with patch('pipeline.vision_pipeline.create_detector') as factory, patch('pipeline.vision_pipeline.capture_worker', capture), patch('cv2.imshow'), patch('cv2.waitKey', return_value=key), patch('cv2.destroyAllWindows') as destroy:
                factory.return_value.warmup.side_effect = warmed.set
                factory.return_value.detect.return_value = []
                VisionPipeline(VisionConfig()).run()
                destroy.assert_called_once()
            self.assertFalse(any(t.name in ('capture', 'scheduler', 'compute') for t in threading.enumerate()))

    def test_missing_video_fails_clearly(self):
        from pipeline.capture_worker import capture_worker
        frames = LatestFrameBuffer()
        with self.assertRaisesRegex(RuntimeError, 'Could not open video'):
            capture_worker(replace(VisionConfig(), video_path='/tmp/nonexistent-vision-video.mp4'), frames, threading.Event())
        self.assertTrue(frames.is_finished())

    def test_detector_failure_propagates(self):
        def capture(config, frames, stop):
            frames.publish(FramePacket(1, time.perf_counter(), np.zeros((20, 20, 3), np.uint8)))
            stop.wait(2)
            frames.mark_finished()
        with patch('pipeline.vision_pipeline.create_detector') as factory, patch('pipeline.vision_pipeline.capture_worker', capture), patch('cv2.imshow'), patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            factory.return_value.detect.side_effect = RuntimeError('inference failure')
            with self.assertRaisesRegex(RuntimeError, 'Vision worker failed'):
                VisionPipeline(VisionConfig()).run()

    def test_slow_inference_does_not_block_display(self):
        import time
        shown = []
        detecting, release, displayed = threading.Event(), threading.Event(), threading.Event()
        def capture(config, frames, stop):
            frames.publish(FramePacket(0, time.perf_counter(), np.zeros((20, 20, 3), np.uint8)))
            self.assertTrue(detecting.wait(2))
            try:
                for i in range(1, 9):
                    displayed.clear()
                    frames.publish(FramePacket(i, time.perf_counter(), np.zeros((20, 20, 3), np.uint8)))
                    self.assertTrue(displayed.wait(2), 'display blocked behind inference')
            finally:
                release.set()
                frames.mark_finished()
        def detect(frame):
            detecting.set()
            if not release.wait(5):
                raise RuntimeError('display did not progress during inference')
            return []
        def show(*args):
            if detecting.is_set() and not release.is_set():
                shown.append(True)
            displayed.set()
        with patch('pipeline.vision_pipeline.create_detector') as factory, patch('pipeline.vision_pipeline.capture_worker', capture), patch('cv2.imshow', side_effect=show), patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows'):
            factory.return_value.detect.side_effect = detect
            VisionPipeline(VisionConfig()).run()
        self.assertGreaterEqual(len(shown), 8)

    def test_worker_failure_propagates_and_cleans_up(self):
        def fail(*args):
            raise RuntimeError('capture failure')
        with patch('pipeline.vision_pipeline.create_detector'), patch('pipeline.vision_pipeline.capture_worker', fail), patch('cv2.waitKey', return_value=-1), patch('cv2.destroyAllWindows') as destroy:
            with self.assertRaisesRegex(RuntimeError, 'Vision worker failed'):
                VisionPipeline(VisionConfig()).run()
            destroy.assert_called_once()


if __name__ == '__main__':
    unittest.main()
