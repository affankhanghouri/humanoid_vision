"""Regressions for the active demo's box timing, freshness and drawing cost."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.types import FramePacket
from core.perception_state import PerceptionEntity, PerceptionState, ObservationMeta, PoseObservation, PoseKeypoint
from rendering.demo_renderer import DemoRenderer
from rendering.overlays import dark_panel
from tracking.smoothing import DisplayBoxSmoother


class DemoConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((720, 1152, 3), np.uint8)
        self.entity = PerceptionEntity(1, 'person', .9, 10, 300, 100, 0, 100, 200)
        self.state = PerceptionState(10, 1, 1.3, (self.entity,), 300, 3, 300, 300)

    def test_smoothing_uses_elapsed_time_and_ignores_duplicate_frames(self):
        config = replace(VisionConfig(), smoothing_alphas=(.5,)*4, display_size_alpha=.5)
        slow, fast = DisplayBoxSmoother(config), DisplayBoxSmoother(config)
        for smoother in (slow, fast):
            smoother.smooth(1, (0, 0, 20, 20), timestamp=0)
        target = (20, 20, 60, 60)
        for i in range(1, 5):
            fast.smooth(1, target, timestamp=i/24)
        box = slow.smooth(1, target, timestamp=4/24)
        np.testing.assert_allclose(box, fast.boxes[1])
        np.testing.assert_array_equal(slow.smooth(1, (100, 100, 200, 200), timestamp=4/24), box)
        slow.keep_only(())
        self.assertEqual(slow.timestamps, {})

    def test_demo_predicts_to_current_frame_and_projects_unclipped_pose(self):
        renderer = DemoRenderer(VisionConfig())
        packet = FramePacket(20, 1.6, self.image)
        views = renderer.build_views(self.image, packet, self.state, .6)
        self.assertEqual(views[0].center[0], 70)
        # A person cropped at the image edge must not have its skeleton stretched.
        skeleton = PoseObservation(ObservationMeta(10, 1, 1.2), (PoseKeypoint(.5, .5, .9),), .9)
        state = replace(self.state, entities=(replace(self.entity, velocity_x=0, pose=skeleton),))
        renderer = DemoRenderer(VisionConfig())
        views = renderer.build_views(self.image, packet, state, .6)
        self.assertEqual(views[0].box[0], 0)
        self.assertEqual(views[0].projection_box[0], -40)
        with patch('rendering.demo_renderer.draw_pose', return_value=True) as draw:
            renderer.draw_entities(self.image.copy(), packet, views)
        self.assertEqual(draw.call_args.args[2], (-40, 200, 60, 400))

    def test_future_pose_frame_is_rejected_even_when_timestamp_is_old(self):
        renderer = DemoRenderer(VisionConfig())
        pose = PoseObservation(ObservationMeta(99, 1, 1.2), (), .9)
        state = replace(self.state, entities=(replace(self.entity, pose=pose),))
        packet = FramePacket(20, 1.3, self.image)
        views = renderer.build_views(self.image, packet, state, .3)
        self.assertFalse(views[0].pose_ready)
        with patch('rendering.demo_renderer.draw_pose') as draw:
            renderer.draw_entities(self.image.copy(), packet, views)
        draw.assert_not_called()

    def test_missing_or_invalid_state_clears_all_display_history_and_metrics(self):
        for state in (None, self.state, replace(self.state, source_frame_id=99)):
            with self.subTest(state=state):
                renderer = DemoRenderer(VisionConfig())
                renderer.box_smoother.smooth(1, (0, 0, 10, 10))
                renderer.history.update(1, (5, 5), 1)
                packet = FramePacket(20, 2, self.image)
                with patch.object(renderer, 'draw_metrics') as metrics, patch.object(renderer, 'draw_status_bar') as status:
                    renderer.render(packet, state, 24)
                self.assertEqual(renderer.box_smoother.boxes, {})
                self.assertEqual(renderer.history.trail(1), ())
                self.assertIsNone(metrics.call_args.args[1])
                self.assertIsNone(status.call_args.args[1])
                self.assertFalse(self.image.any())

    def test_fully_offscreen_boxes_are_not_pinned_to_an_edge(self):
        renderer = DemoRenderer(VisionConfig())
        state = replace(self.state, entities=(replace(self.entity, center_x=-300),))
        self.assertEqual(renderer.build_views(self.image, FramePacket(20, 1.3, self.image), state, .3), [])

    def test_panel_blends_only_its_visible_region(self):
        frame = np.full((720, 1152, 3), 100, np.uint8)
        expected = frame.copy()
        overlay = frame.copy()
        overlay[20:70, 10:90] = (7, 13, 20)
        cv2.addWeighted(overlay, .8, expected, .2, 0, dst=expected)
        original = cv2.addWeighted
        with patch('rendering.overlays.cv2.addWeighted', wraps=original) as blend:
            dark_panel(frame, 10, 20, 80, 50, alpha=.8)
        np.testing.assert_array_equal(frame, expected)
        self.assertEqual(blend.call_args.args[0].shape, (50, 80, 3))
        # Clipping panels must never index from the opposite edge of the image.
        dark_panel(frame, -10, -10, 20, 20)
        dark_panel(frame, 2000, 2000, 20, 20)


if __name__ == '__main__':
    unittest.main()
