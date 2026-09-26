"""Behavioral checks for confidence recovery, identity gates and expiry."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.types import Detection
from tracking.tracker import MultiObjectTracker


def detection(x=100, confidence=.9, width=40, name='person'):
    return Detection((x, 100, x+width, 200), 0, name, confidence)


class TrackingConsistencyTests(unittest.TestCase):
    def confirmed(self):
        tracker = MultiObjectTracker()
        tracker.update([detection()], 0)
        tracker.update([detection()], .25)
        return tracker, tracker.snapshots()[0].entity_id

    def test_weak_observations_recover_existing_id_but_never_create_or_confirm(self):
        tracker, entity_id = self.confirmed()
        for t in (.5, .75, 1.0):
            tracker.update([detection(102, .25), detection(500, .3)], t)
            self.assertEqual(len(tracker.tracks), 1)
            self.assertEqual(tracker.snapshots()[0].entity_id, entity_id)
            self.assertEqual(tracker.snapshots()[0].misses, 0)
        empty = MultiObjectTracker()
        empty.update([detection(confidence=.3)], 0)
        self.assertEqual(empty.tracks, [])
        empty.update([detection()], .25)
        empty.update([detection(confidence=.3)], .5)
        self.assertEqual(empty.snapshots(), ())

    def test_tentative_track_needs_consecutive_observations(self):
        tracker = MultiObjectTracker()
        tracker.update([detection()], 0)
        old_id = tracker.tracks[0].id
        tracker.update([], .25)
        tracker.update([detection()], .5)
        self.assertEqual(tracker.snapshots(), ())
        self.assertNotEqual(tracker.tracks[0].id, old_id)

    def test_distant_and_wrong_size_objects_cannot_steal_id(self):
        for other in (detection(240), detection(100, width=500), detection(name='car')):
            with self.subTest(other=other):
                tracker, entity_id = self.confirmed()
                tracker.update([other], .5)
                self.assertEqual(tracker.tracks[0].id, entity_id)
                self.assertEqual(tracker.tracks[0].misses, 1)
                self.assertNotEqual(tracker.tracks[1].id, entity_id)

    def test_long_gap_expires_id_before_reassociation(self):
        tracker, entity_id = self.confirmed()
        tracker.update([detection()], 2)
        self.assertNotEqual(tracker.tracks[0].id, entity_id)
        self.assertEqual(tracker.snapshots(), ())

    def test_missing_tracks_are_bounded_and_do_not_refresh_observation_time(self):
        tracker, entity_id = self.confirmed()
        tracker.update([], .5)
        entity = tracker.snapshots()[0]
        self.assertEqual(entity.last_observed_timestamp, .25)
        tracker.update([], .75)
        self.assertEqual(tracker.snapshots(), ())
        tracker.update([], 1)
        self.assertEqual(tracker.tracks, [])

    def test_bad_detections_and_replayed_timestamps_do_not_corrupt_tracks(self):
        tracker, entity_id = self.confirmed()
        for t in (.25, .1):
            tracker.update([detection(500)], t)
        self.assertEqual(tracker.snapshots()[0].entity_id, entity_id)
        self.assertEqual(tracker.tracks[0].hits, 2)
        invalid = [replace(detection(), bbox=box) for box in
                   ((float('nan'), 0, 10, 10), (10, 10, 0, 0), (0, 0, 0, 10))]
        invalid += [detection(confidence=float('nan')), detection(confidence=1.1)]
        tracker.update(invalid, .5)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.tracks[0].misses, 1)
        with self.assertRaises(ValueError):
            tracker.update([], float('nan'))

    def test_vehicle_subtype_flicker_does_not_duplicate_or_replace_the_id(self):
        tracker = MultiObjectTracker()
        for timestamp, name in ((0, 'car'), (.25, 'car'), (.5, 'truck')):
            tracker.update([detection(name=name)], timestamp)
        entity_id = tracker.snapshots()[0].entity_id
        tracker.update([detection(name='truck'), detection(name='car', confidence=.7)], .75)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.snapshots()[0].entity_id, entity_id)
        self.assertEqual(tracker.snapshots()[0].class_name, 'car')

    def test_high_confidence_assignment_takes_priority_over_weak_match(self):
        tracker, entity_id = self.confirmed()
        tracker.update([detection(100, .2), detection(104, .8)], .5)
        self.assertEqual(len(tracker.tracks), 1)
        self.assertEqual(tracker.snapshots()[0].confidence, .8)


if __name__ == '__main__':
    unittest.main()
