"""Regressions for delayed overlays during camera/object motion."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.perception_state import PerceptionEntity, PerceptionState
from core.types import Detection, FramePacket
from rendering.renderer import Renderer
from tracking.tracker import MultiObjectTracker


class BoxAlignmentTests(unittest.TestCase):
    def test_old_but_valid_result_keeps_predicting_motion(self):
        renderer = Renderer(VisionConfig())
        entity = PerceptionEntity(1, 'person', .9, 100, 100, 100, 0, 40, 40)
        state = PerceptionState(1, 0, .4, (entity,), 400, 2.5, 400, 400)
        image = np.zeros((250, 250, 3), np.uint8)
        renderer.render(FramePacket(15, .6, image), state, 24)
        np.testing.assert_array_equal(renderer.smoother.boxes[1], (140, 80, 180, 120))
        self.assertFalse(image.any())

    def test_missed_detection_marks_prediction_and_preserves_reassociation(self):
        tracker = MultiObjectTracker()
        detection = Detection((100, 100, 140, 200), 0, 'person', .9)
        tracker.update([detection], 0)
        tracker.update([detection], .3)
        entity_id = tracker.snapshots()[0].entity_id
        tracker.update([], .6)
        predicted = tracker.snapshots()[0]
        self.assertEqual(predicted.misses, 1)
        self.assertEqual(predicted.last_observed_timestamp, .3)
        self.assertTrue(predicted.visible_at(.7, tracker.config.max_coast_age))
        self.assertFalse(predicted.visible_at(.8, tracker.config.max_coast_age))
        self.assertEqual(len(tracker.tracks), 1)
        tracker.update([detection], .9)
        self.assertEqual(tracker.snapshots()[0].entity_id, entity_id)

    def test_approaching_object_box_follows_new_measurement(self):
        tracker = MultiObjectTracker()
        tracker.update([Detection((100, 100, 140, 200), 0, 'person', .9)], 0)
        tracker.update([Detection((90, 90, 170, 230), 0, 'person', .9)], .3)
        entity = tracker.snapshots()[0]
        # A doubled-width detection should not keep a box close to its old size.
        self.assertGreater(entity.width, 70)
        self.assertGreater(entity.height, 125)
        self.assertAlmostEqual(entity.center_x, 130, delta=2)


if __name__ == '__main__':
    unittest.main()
