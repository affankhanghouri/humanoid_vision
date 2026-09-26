"""Scheduler replacement, selection, forwarding, and shutdown behavior."""
import sys
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from core.perception_store import PerceptionStore
from core.types import FramePacket
from scheduling.scheduler import ComputeScheduler, scheduler_worker
from scheduling.task import ComputeTask, TaskPolicy


class SchedulingTests(unittest.TestCase):
    def task(self, frame_id, task_type='detection'):
        packet = FramePacket(frame_id, frame_id * .1, np.zeros((2, 2, 3), np.uint8))
        return ComputeTask(task_type, frame_id, packet.timestamp, packet)

    def test_latest_replaces_pending_but_older_or_equal_cannot(self):
        scheduler, stop = ComputeScheduler((TaskPolicy('detection', 0, 0, float('inf')), TaskPolicy('test_high', 1, 0, float('inf')), TaskPolicy('test_equal', 0, 0, float('inf')))), threading.Event()
        newest = self.task(104)
        for task in (self.task(101), newest, self.task(103), self.task(104)):
            scheduler.submit(task)
        self.assertIs(scheduler.get_next(stop), newest)
        with self.assertRaises(FrozenInstanceError):
            newest.source_frame_id = 1

    def test_priority_then_frame_id_across_types(self):
        scheduler, stop = ComputeScheduler((TaskPolicy('detection', 0, 0, float('inf')), TaskPolicy('test_high', 1, 0, float('inf')), TaskPolicy('test_equal', 0, 0, float('inf')))), threading.Event()
        tasks = (self.task(104), self.task(102, 'test_high'), self.task(103, 'test_equal'))
        for task in tasks:
            scheduler.submit(task)
        self.assertEqual([scheduler.get_next(stop) for _ in tasks], [tasks[1], tasks[0], tasks[2]])

    def test_stop_wakes_empty_worker_and_prevents_pending_execution(self):
        scheduler, stop, entered = ComputeScheduler((TaskPolicy('detection', 0, 0, float('inf')), TaskPolicy('test_high', 1, 0, float('inf')), TaskPolicy('test_equal', 0, 0, float('inf')))), threading.Event(), threading.Event()
        results = []
        def wait():
            entered.set()
            results.append(scheduler.get_next(stop))
        thread = threading.Thread(target=wait)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
        finally:
            stop.set()
            scheduler.wake()
            thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, [None])
        scheduler.submit(self.task(1))
        self.assertIsNone(scheduler.get_next(stop))

    def test_scheduler_submits_latest_once_with_original_packet(self):
        frames, scheduler, stop = LatestFrameBuffer(), ComputeScheduler((TaskPolicy('detection', 0, 0, float('inf')), TaskPolicy('test_high', 1, 0, float('inf')), TaskPolicy('test_equal', 0, 0, float('inf')))), threading.Event()
        packet = self.task(104).packet
        frames.publish(packet)
        frames.mark_finished()
        with patch.object(scheduler, 'submit', wraps=scheduler.submit) as submit:
            scheduler_worker(VisionConfig(), frames, PerceptionStore(), scheduler, stop)
        submit.assert_called_once()
        task = scheduler.get_next(stop)
        self.assertEqual(task.task_type, 'detection')
        self.assertEqual((task.source_frame_id, task.source_timestamp), (104, 10.4))
        self.assertIs(task.packet, packet)

    def test_scheduler_exits_for_empty_finished_source(self):
        frames = LatestFrameBuffer()
        frames.mark_finished()
        scheduler = ComputeScheduler((TaskPolicy('detection', 0, 0, float('inf')), TaskPolicy('test_high', 1, 0, float('inf')), TaskPolicy('test_equal', 0, 0, float('inf'))))
        with patch.object(scheduler, 'submit') as submit:
            scheduler_worker(VisionConfig(), frames, PerceptionStore(), scheduler, threading.Event())
        submit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
