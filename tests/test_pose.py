"""Pose identity, freshness, publication, drawing, and compute scheduling."""
import sys
import threading
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.types import FramePacket
from core.perception_state import ObservationMeta, PerceptionEntity, PerceptionState, PoseKeypoint, PoseObservation
from core.pose_types import PosePerson, PosePoint
from core.perception_store import PerceptionStore
from pipeline.compute_worker import compute_worker
from pose.matching import match_pose_to_entities
from rendering.renderer import Renderer, draw_pose
from tracking.smoothing import DisplayPoseSmoother
from scheduling.scheduler import ComputeScheduler, scheduler_worker
from scheduling.task import ComputeTask, TaskPolicy
from tracking.tracker import MultiObjectTracker


def entity(entity_id=7, cx=50, velocity=0):
    return PerceptionEntity(entity_id, 'person', .9, cx, 30, velocity, 0, 20, 40)


def state(frame_id=1, timestamp=10, entities=None):
    return PerceptionState(frame_id, timestamp, timestamp+.2,
                           (entity(),) if entities is None else entities, 200, 3, 200, 200)


def pose(frame_id=2, timestamp=10.3):
    return PoseObservation(ObservationMeta(frame_id, timestamp, timestamp+.16),
                           (PoseKeypoint(.5, .5, .9),), .9)


class PoseTests(unittest.TestCase):
    def test_tracking_updates_preserve_pose_without_refreshing_its_clock(self):
        store = PerceptionStore()
        original = state()
        store.publish_tracking(original)
        skeleton = pose()
        store.publish_pose({7: skeleton}, skeleton.meta, 160, 1.5, 160, 170)
        with_pose = store.get_latest()
        self.assertEqual(with_pose.source_timestamp, 10)
        self.assertIsNone(original.entities[0].pose)
        store.publish_tracking(state(3, 10.6, (entity(cx=80),)))
        latest = store.get_latest()
        self.assertEqual(latest.entities[0].center_x, 80)
        self.assertIs(latest.entities[0].pose, skeleton)
        self.assertEqual(latest.pose_meta.source_timestamp, 10.3)
        self.assertEqual(latest.pose_inference_ms, 160)
        with self.assertRaises(FrozenInstanceError):
            skeleton.keypoints[0].x = .2

    def test_out_of_order_updates_cannot_overwrite_newer_results(self):
        store = PerceptionStore()
        store.publish_tracking(state(5, 11))
        skeleton = pose(6, 11.2)
        store.publish_pose({7: skeleton}, skeleton.meta, 160, 1, 160, 160)
        latest = store.get_latest()
        store.publish_tracking(state(1, 10))
        store.publish_pose({}, pose(2, 10.1).meta, 100, 1, 100, 100)
        self.assertIs(store.get_latest(), latest)
        store.publish_pose({}, pose(7, 11.5).meta, 100, 1, 100, 100)
        self.assertIsNone(store.get_latest().entities[0].pose)
        self.assertEqual(store.get_latest().source_frame_id, 5)

    def test_removed_ids_do_not_transfer_pose_to_new_people(self):
        store = PerceptionStore()
        store.publish_tracking(state())
        skeleton = pose()
        store.publish_pose({7: skeleton}, skeleton.meta, 160, 1, 160, 160)
        store.publish_tracking(state(3, 10.5, (entity(8),)))
        self.assertIsNone(store.get_latest().entities[0].pose)

    def test_matching_predicts_to_pose_source_and_normalizes_keypoints(self):
        tracking = state(1, 10, (entity(7, 50, 100),))
        person = PosePerson((70, 10, 90, 50), .9, (PosePoint(80, 30, .8),))
        meta = ObservationMeta(2, 10.3, 10.46)
        matched = match_pose_to_entities((person,), tracking, meta, VisionConfig())
        self.assertEqual(set(matched), {7})
        self.assertEqual(matched[7].keypoints[0], PoseKeypoint(.5, .5, .8))
        self.assertEqual(match_pose_to_entities((person,), tracking, replace(meta, source_timestamp=11), VisionConfig()), {})

    def test_matching_is_one_to_one_and_ignores_nonpeople(self):
        car = replace(entity(9), class_name='car')
        tracking = state(1, 10, (entity(7), entity(8, 100), car))
        people = (PosePerson((90, 10, 110, 50), .9, ()), PosePerson((40, 10, 60, 50), .8, ()))
        matched = match_pose_to_entities(people, tracking, ObservationMeta(2, 10.1, 10.2), VisionConfig())
        self.assertEqual(set(matched), {7, 8})
        self.assertEqual(matched[7].confidence, .8)
        self.assertEqual(matched[8].confidence, .9)
        duplicate = match_pose_to_entities((people[1], people[1]), state(), ObservationMeta(2, 10.1, 10.2), VisionConfig())
        self.assertEqual(len(duplicate), 1)

    def test_pose_has_independent_freshness_and_never_refreshes_tracking(self):
        image = np.zeros((250, 250, 3), np.uint8)
        packet = FramePacket(10, 11, image)
        for skeleton, tracking_time, allowed in (
            (pose(8, 10.6), 10.8, True),
            (pose(3, 10.0), 10.8, False),
            (pose(11, 10.8), 10.8, False),
            (pose(8, 11.1), 10.8, False),
            (pose(8, 10.6), 10.0, False),
        ):
            with self.subTest(skeleton=skeleton, tracking_time=tracking_time):
                renderer = Renderer(VisionConfig())
                snapshot = state(5, tracking_time, (replace(entity(), pose=skeleton),))
                with patch('rendering.renderer.draw_pose') as draw:
                    renderer.render(packet, snapshot, 24)
                self.assertEqual(draw.call_count, int(allowed))
        self.assertFalse(image.any())

    def test_skeleton_projection_and_weak_joint_filter(self):
        image = np.zeros((200, 200, 3), np.uint8)
        joints = [PoseKeypoint(.5, .5, 0) for _ in range(17)]
        joints[5] = PoseKeypoint(.25, .2, .9)
        joints[6] = PoseKeypoint(.75, .2, .9)
        joints[7] = PoseKeypoint(.1, .5, .1)
        skeleton = replace(pose(), keypoints=tuple(joints))
        with patch('cv2.line') as line, patch('cv2.circle') as circle:
            draw_pose(image, replace(entity(), pose=skeleton), (20, 30, 120, 130), 10.3,
                      DisplayPoseSmoother(VisionConfig()), VisionConfig())
        line.assert_called_once()
        self.assertEqual(line.call_args.args[1:3], ((45, 50), (95, 50)))
        self.assertEqual(circle.call_count, 2)

    def test_scheduler_offers_pose_only_for_fresh_people(self):
        for snapshot, expected in ((None, 1), (state(1, 9), 1), (state(1, 10, ()), 1), (state(1, 10), 2)):
            with self.subTest(snapshot=snapshot):
                frames, store, scheduler = LatestFrameBuffer(), PerceptionStore(), Mock()
                if snapshot is not None:
                    store.publish_tracking(snapshot)
                frames.publish(FramePacket(2, 10.3, np.zeros((5, 5, 3), np.uint8)))
                frames.mark_finished()
                scheduler_worker(VisionConfig(), frames, store, scheduler, threading.Event())
                self.assertEqual(scheduler.submit.call_count, expected)

    def test_slow_detection_does_not_starve_ready_pose(self):
        scheduler = ComputeScheduler((TaskPolicy('detection', 100, .25, .15), TaskPolicy('pose', 70, .5, .15)))
        stop = threading.Event()
        image = np.zeros((5, 5, 3), np.uint8)
        def submit(frame_id, timestamp):
            for kind in ('detection', 'pose'):
                scheduler.submit(ComputeTask(kind, frame_id, timestamp, FramePacket(frame_id, timestamp, image)))
        with patch('scheduling.scheduler.time.perf_counter', return_value=10):
            submit(1, 10)
            self.assertEqual(scheduler.get_next(stop).task_type, 'detection')
        with patch('scheduling.scheduler.time.perf_counter', return_value=10.35):
            submit(2, 10.35)
            self.assertEqual(scheduler.get_next(stop).task_type, 'pose')
        with patch('scheduling.scheduler.time.perf_counter', return_value=10.51):
            submit(3, 10.51)
            self.assertEqual(scheduler.get_next(stop).task_type, 'detection')
            # Pose remains rate-limited until 10.85, even with fresh pending input.
            self.assertEqual(scheduler._last_run_time['pose'], 10.35)

    def test_compute_publishes_pose_and_rechecks_person_need(self):
        image = np.zeros((80, 120, 3), np.uint8)
        task = ComputeTask('pose', 2, 10.1, FramePacket(2, 10.1, image))
        for with_person in (False, True):
            store, scheduler, detector, estimator = PerceptionStore(), Mock(), Mock(), Mock()
            store.publish_tracking(state(entities=None if with_person else ()))
            estimator.estimate.return_value = (PosePerson((40, 10, 60, 50), .9, (PosePoint(50, 30, .8),)),)
            scheduler.get_next.side_effect = [task, None]
            with patch('pipeline.compute_worker.time.perf_counter', side_effect=[10.0, 10.1, 10.2, 10.21] if with_person else [10.0]):
                compute_worker(VisionConfig(), scheduler, detector, estimator, MultiObjectTracker(), store, threading.Event())
            self.assertEqual(estimator.estimate.call_count, int(with_person))
            detector.detect.assert_not_called()
            if with_person:
                latest = store.get_latest()
                self.assertEqual(latest.entities[0].pose.meta.produced_timestamp, 10.21)
                self.assertEqual(latest.source_timestamp, 10)
                scheduler.mark_executed.assert_called_once()
                call = scheduler.mark_executed.call_args.args
                self.assertEqual(call[0], 'pose')
                self.assertAlmostEqual(call[1], .11)
                self.assertEqual(call[2], 10.1)


if __name__ == '__main__':
    unittest.main()
