"""Delayed box projection against known image motion, with no neural model."""
import sys
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.types import FramePacket
from core.perception_state import PerceptionEntity, PerceptionState
from rendering.demo_renderer import DemoRenderer
from rendering.renderer import Renderer
from tracking.motion import BoxMotionHistory


class MotionTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        self.image = np.random.default_rng(42).integers(0, 255, (180, 320, 3), dtype=np.uint8)
        self.box = (90, 60, 140, 130)

    def packet(self, index, dx):
        frame = cv2.warpAffine(self.image, np.float32(((1, 0, dx), (0, 1, 0))), (320, 180))
        return FramePacket(index, index/24, frame)

    def test_delayed_boxes_follow_camera_reversal_instead_of_old_velocity(self):
        motion = BoxMotionHistory(replace(VisionConfig(), motion_min_interval=0))
        shifts = (0, 3, 6, 9, 12, 9, 6, 3, 0)
        for index, dx in enumerate(shifts):
            motion.update(self.packet(index, dx))
            if index >= 4:
                expected = np.array(self.box) + (dx, 0, dx, 0)
                np.testing.assert_allclose(motion.project(1, self.box, 0), expected, atol=1.5)
        # A new delayed detector result starts at its own source frame.
        observed = np.array(self.box) + (12, 0, 12, 0)
        np.testing.assert_allclose(motion.project(1, observed, 4/24), self.box, atol=1.5)

    def test_duplicate_packets_do_not_advance_motion(self):
        motion = BoxMotionHistory(replace(VisionConfig(), motion_min_interval=0))
        motion.update(self.packet(0, 0))
        motion.update(self.packet(1, 4))
        first = motion.project(1, self.box, 0)
        motion.update(self.packet(1, 40))
        self.assertEqual(motion.project(1, self.box, 0), first)
        self.assertEqual(len(motion.steps), 1)

    def test_gap_textureless_frame_and_resolution_change_reject_incomplete_history(self):
        motion = BoxMotionHistory(replace(VisionConfig(), motion_min_interval=0))
        motion.update(self.packet(0, 0))
        motion.update(self.packet(1, 4))
        motion.update(FramePacket(2, .1, np.zeros_like(self.image)))
        self.assertIsNone(motion.project(1, self.box, 0))
        motion.update(self.packet(24, 10))
        self.assertIsNone(motion.project(1, self.box, 0))
        motion.update(FramePacket(25, 25/24, self.image[:90]))
        self.assertEqual(len(motion.steps), 0)
        self.assertIsNone(motion.project(1, self.box, 1))

    def test_history_and_prediction_are_bounded(self):
        motion = BoxMotionHistory(replace(VisionConfig(), motion_min_interval=0))
        for index in range(60):
            motion.update(self.packet(index, 0))
        self.assertLessEqual(len(motion.steps), 20)
        self.assertIsNone(motion.project(1, self.box, 0))
        self.assertIsNone(motion.project(1, self.box, 100))

    def test_capture_snapshot_skips_render_flow_and_keeps_source_pixels_unchanged(self):
        capture = BoxMotionHistory(VisionConfig())
        display = BoxMotionHistory(VisionConfig())
        for index in range(9):
            packet = self.packet(index, index*3)
            original = packet.frame.copy()
            packet.motion = capture.snapshot()
            capture.update(FramePacket(packet.frame_id, packet.timestamp, packet.frame))
            with patch.object(display, '_estimate', side_effect=AssertionError('flow on display')):
                display.update(packet)
                if index >= 5:
                    expected = np.array(self.box) + (index*3, 0, index*3, 0)
                    np.testing.assert_allclose(display.project(1, self.box, 0), expected, atol=2)
            np.testing.assert_array_equal(packet.frame, original)
        self.assertLess(len(capture.steps), 8)
        with self.assertRaises(ValueError):
            packet.motion.steps[-1].before[0, 0] = 99

    def test_both_renderers_use_flow_and_preserve_the_input(self):
        entity = PerceptionEntity(1, 'person', .9, 115, 95, 0, 0, 50, 70)
        state = PerceptionState(0, 0, .2, (entity,), 200, 3, 200, 200)
        for renderer_type in (Renderer, DemoRenderer):
            renderer = renderer_type(replace(VisionConfig(), motion_min_interval=0))
            for index in range(7):
                packet = self.packet(index, index*3)
                original = packet.frame.copy()
                renderer.render(packet, state if index >= 4 else None, 24)
                np.testing.assert_array_equal(packet.frame, original)
            smoother = renderer.smoother if isinstance(renderer, Renderer) else renderer.box_smoother
            np.testing.assert_allclose(smoother.boxes[1], np.array(self.box)+(18, 0, 18, 0), atol=3)


if __name__ == '__main__':
    unittest.main()
