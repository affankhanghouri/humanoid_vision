"""Deterministic geometry, freshness and temporal path-relation tests."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/path_relation'))

from core.perception_state import PerceptionEntity, PerceptionState, RoadSegObservation
from rejected_path_relation.engine import PathRelationEngine
from rejected_path_relation.geometry import (
    contact_point, construct_forward_corridor, image_to_mask, patch_fraction,
)
from rejected_path_relation.types import PathRelationState


def road(mask=None, frame=1, timestamp=0.0):
    if mask is None:
        mask = np.zeros((60, 100), np.uint8)
        mask[12:60, 10:90] = 1
    return RoadSegObservation(frame, timestamp, timestamp + .01, 1., mask,
                              (0, 0, 100, 60), (100, 60))


def entity(track_id=1, x=50., y=45., vx=0., vy=0., timestamp=0.,
           width=8., height=10., kind="car", misses=0):
    return PerceptionEntity(track_id, kind, .9, x, y, vx, vy, width, height,
                            last_observed_timestamp=timestamp, misses=misses)


def state(frame, timestamp, item):
    return PerceptionState(frame, timestamp, timestamp + .01, (item,), 1., 4., 1., 1.)


class GeometryTests(unittest.TestCase):
    def test_coordinate_transform_image_to_native_content(self):
        self.assertEqual(image_to_mask((0, 0), (200, 100), (10, 5, 100, 50), (60, 120)), (10, 5))
        self.assertEqual(image_to_mask((199, 99), (200, 100), (10, 5, 100, 50), (60, 120)), (109, 54))

    def test_bottom_center_contact_and_invalid_boxes(self):
        self.assertEqual(contact_point((-5, 2, 15, 70), (100, 60)), (5., 59.))
        self.assertIsNone(contact_point((4, 4, 3, 8), (100, 60)))
        self.assertIsNone(contact_point((0, 0, float("nan"), 4), (100, 60)))

    def test_foot_patch_fraction_and_edge_clipping(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[:, :5] = 1
        self.assertEqual(patch_fraction(mask, (0, 9), (3, 2)), 1.)
        fraction = patch_fraction(mask, (5, 5), (2, 2))
        self.assertGreater(fraction, 0.)
        self.assertLess(fraction, 1.)

    def test_corridor_selects_lower_middle_component(self):
        mask = np.zeros((60, 100), np.uint8)
        mask[10:60, 20:80] = 1
        mask[5:20, 0:8] = 1
        corridor = construct_forward_corridor(mask, (0, 0, 100, 60))
        self.assertIsNotNone(corridor)
        self.assertFalse(corridor.mask[10, 3])
        self.assertTrue(corridor.mask[50, 50])
        self.assertFalse(corridor.mask[50, 22])

    def test_empty_or_disconnected_from_lower_middle_abstains(self):
        self.assertIsNone(construct_forward_corridor(np.zeros((60, 100), np.uint8),
                                                     (0, 0, 100, 60)))
        mask = np.zeros((60, 100), np.uint8)
        mask[2:20, 2:20] = 1
        self.assertIsNone(construct_forward_corridor(mask, (0, 0, 100, 60)))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = PathRelationEngine(transition_persistence=2)
        self.road = road()

    def update(self, frame, timestamp, item, road_value=None, display_timestamp=None):
        return self.engine.update(
            state(frame, timestamp, item), self.road if road_value is None else road_value,
            frame_id=frame, timestamp=timestamp if display_timestamp is None else display_timestamp,
            frame_size=(100, 60))[0]

    def test_stale_road_and_tracking_abstain_without_timestamp_refresh(self):
        old = road(frame=1, timestamp=2.)
        result = self.engine.update(state(5, 4., entity(timestamp=4.)), old,
                                    frame_id=5, timestamp=4., frame_size=(100, 60))[0]
        self.assertEqual(result.state, PathRelationState.UNKNOWN)
        self.assertFalse(result.road_reliable)
        self.assertEqual(result.road_source_timestamp, 2.)
        current = road(frame=5, timestamp=4.)
        result = self.engine.update(state(5, 2., entity(timestamp=2.)), current,
                                    frame_id=5, timestamp=4., frame_size=(100, 60))[0]
        self.assertEqual(result.state, PathRelationState.UNKNOWN)

    def test_stationary_object_is_in_corridor_after_motion_history(self):
        result = None
        for frame, timestamp in enumerate((0., .1, .2, .3), 1):
            result = self.update(frame, timestamp, entity(timestamp=timestamp))
        self.assertTrue(result.motion_reliable)
        self.assertEqual(result.predicted_contact_point, result.contact_point)
        self.assertEqual(result.state, PathRelationState.IN_CORRIDOR)

    def test_object_moving_toward_corridor_requires_persistence(self):
        states = []
        for frame, (timestamp, x) in enumerate(((0., 10.), (.1, 14.), (.2, 18.),
                                                (.3, 22.), (.4, 26.)), 1):
            result = self.update(frame, timestamp, entity(x=x, vx=55., timestamp=timestamp))
            states.append(result.state)
        self.assertNotEqual(states[2], PathRelationState.ENTERING_CORRIDOR)
        self.assertEqual(states[3], PathRelationState.ENTERING_CORRIDOR)

    def test_crossing_corridor(self):
        result = None
        for frame, (timestamp, x) in enumerate(((0., 20.), (.1, 26.), (.2, 34.),
                                                (.3, 43.), (.4, 48.)), 1):
            result = self.update(frame, timestamp, entity(x=x, vx=100., timestamp=timestamp))
        self.assertEqual(result.state, PathRelationState.CROSSING_CORRIDOR)

    def test_leaving_corridor(self):
        result = None
        for frame, (timestamp, x) in enumerate(((0., 50.), (.1, 50.), (.2, 52.),
                                                (.3, 54.), (.4, 56.)), 1):
            result = self.update(frame, timestamp, entity(x=x, vx=70., timestamp=timestamp))
        self.assertEqual(result.state, PathRelationState.LEAVING_CORRIDOR)

    def test_repeated_source_frame_does_not_advance_hysteresis(self):
        for frame, timestamp, x in ((1, 0., 10.), (2, .1, 14.), (3, .2, 18.)):
            result = self.update(frame, timestamp, entity(x=x, vx=55., timestamp=timestamp))
        first_count = result.evidence_count
        repeated = self.engine.update(state(3, .2, entity(x=18., vx=55., timestamp=.2)), self.road,
                                      frame_id=4, timestamp=.22, frame_size=(100, 60))[0]
        self.assertEqual(repeated.evidence_count, first_count)
        self.assertNotEqual(repeated.state, PathRelationState.ENTERING_CORRIDOR)

    def test_track_expiry(self):
        self.update(1, 0., entity(timestamp=0.))
        empty = PerceptionState(2, 3., 3., (), 1., 4., 1., 1.)
        self.engine.update(empty, self.road, frame_id=2, timestamp=3., frame_size=(100, 60))
        self.assertNotIn(1, self.engine.tracked_ids)

    def test_unsupported_class_and_edge_box_are_safe(self):
        unsupported = self.update(1, 0., entity(kind="dog", timestamp=0.))
        self.assertEqual(unsupported.state, PathRelationState.UNKNOWN)
        edge = self.update(2, .1, entity(x=99., y=58., timestamp=.1, width=20., height=20.))
        self.assertIsNotNone(edge.contact_point)
        self.assertTrue(0 <= edge.contact_point[0] < 100)
        self.assertTrue(0 <= edge.contact_point[1] < 60)


if __name__ == "__main__":
    unittest.main()
