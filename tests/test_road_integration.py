"""Road freshness, independent publication and optional admission invariants."""
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from core.perception_state import PerceptionState, RoadSegObservation, ObservationMeta
from core.perception_store import PerceptionStore
from core.types import FramePacket
from scheduling.scheduler import ComputeScheduler
from scheduling.task import TaskPolicy, ComputeTask


class RoadIntegrationTests(unittest.TestCase):
    def road(self, frame=3, source=10., produced=10.1):
        return RoadSegObservation(frame, source, produced, 50., np.ones((4,8),np.uint8), (0,0,8,4),(80,40))

    def scheduler(self):
        return ComputeScheduler((TaskPolicy('detection',100,.25,.15),
                                 TaskPolicy('road',80,1.,.15)),max_tracking_age=.75)

    def submit(self, scheduler, kind, frame, timestamp):
        packet=FramePacket(frame,timestamp,np.zeros((2,2,3),np.uint8))
        scheduler.submit(ComputeTask(kind,frame,timestamp,packet))

    def test_independent_updates_preserve_source_and_produced_times(self):
        store=PerceptionStore();road=self.road();store.publish_road(road)
        tracking=PerceptionState(5,10.2,10.3,(),200,3,200,250)
        store.publish_tracking(tracking)
        store.publish_pose({},ObservationMeta(6,10.4,10.5),100,1,100,120)
        state=store.get_latest()
        self.assertIs(state.road,road)
        self.assertEqual(state.tracking_meta,tracking.tracking_meta)
        for display_time in [10.6,11.,12.]:
            self.assertEqual(state.road.age_at(display_time), display_time-10.)
        store.publish_road(self.road(2,9.,20.))
        self.assertIs(store.get_road(),road)
        store.publish_tracking(PerceptionState(7,10.6,10.8,(),200,3,200,250))
        self.assertIs(store.get_latest().road,road)

    def test_mask_cannot_be_mutated_or_made_writable(self):
        road=self.road()
        with self.assertRaises(ValueError):road.drivable_mask[0,0]=0
        with self.assertRaises(ValueError):road.drivable_mask.setflags(write=True)

    def test_stage_timings_do_not_change_observation_timestamps(self):
        road=RoadSegObservation(3, 10., 10.1, 40., np.ones((4,8),np.uint8),
                                (0,0,8,4), (80,40), 7., 2.)
        self.assertAlmostEqual(road.processing_latency_ms, 100.)
        self.assertEqual((road.preprocessing_ms, road.inference_ms,
                          road.postprocessing_ms), (7., 40., 2.))

    def test_newest_road_only_and_stale_submit_drop(self):
        scheduler=self.scheduler();scheduler.seed_cost('road',.05)
        scheduler.mark_executed('detection',.2,10.)
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.1):
            self.submit(scheduler,'road',1,10.)
            self.submit(scheduler,'road',2,10.05)
            self.submit(scheduler,'road',0,9.)
            task=scheduler.get_next(threading.Event())
        self.assertEqual(task.source_frame_id,2)
        counts=scheduler.metrics().by_type['road']
        self.assertEqual(counts['replaced'],1)
        self.assertEqual(counts['dropped_stale'],1)

    def test_road_cannot_probe_before_detection_or_cost_is_known(self):
        scheduler=self.scheduler()
        self.assertFalse(scheduler._optional_fits('road',10.))
        scheduler.mark_executed('detection',.2,10.)
        self.assertFalse(scheduler._optional_fits('road',10.2))
        scheduler.seed_cost('road',.05)
        self.assertTrue(scheduler._optional_fits('road',10.2))
        self.assertEqual(scheduler.metrics().by_type['road']['executed'],0)

    def test_fairness_cannot_override_detection_freshness_deadline(self):
        scheduler=self.scheduler();scheduler.seed_cost('road',.08)
        stop=threading.Event()
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.):
            self.submit(scheduler,'detection',1,10.)
            self.assertEqual(scheduler.get_next(stop).task_type,'detection')
        scheduler.mark_executed('detection',.38,10.)
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.38):
            self.submit(scheduler,'detection',2,10.38)
            self.submit(scheduler,'road',2,10.38)
            self.assertEqual(scheduler.get_next(stop).task_type,'detection')
        self.assertEqual(scheduler.metrics().by_type['road']['budget_deferred'],1)

    def test_guard_accounts_for_next_detection_start_interval(self):
        scheduler=ComputeScheduler((TaskPolicy('detection',100,.6,.15),TaskPolicy('road',80,1.,.15)),max_tracking_age=.75)
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.):
            self.submit(scheduler,'detection',1,10.)
            scheduler.get_next(threading.Event())
        scheduler.mark_executed('detection',.2,10.)
        scheduler.seed_cost('road',.05)
        # A 50-ms optional task fits immediately, but detection cannot start until
        # 10.6 and would complete after the current observation's deadline.
        self.assertFalse(scheduler._optional_fits('road',10.2))

    def test_pending_road_is_dropped_if_it_ages_before_dispatch(self):
        scheduler=self.scheduler()
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.):
            self.submit(scheduler,'road',1,10.)
        with patch('scheduling.scheduler.time.perf_counter',return_value=10.2):
            self.submit(scheduler,'detection',2,10.2)
            self.assertEqual(scheduler.get_next(threading.Event()).task_type,'detection')
        self.assertEqual(scheduler.metrics().by_type['road']['dropped_stale'],1)
