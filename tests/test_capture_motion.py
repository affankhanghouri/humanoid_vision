"""Capture must publish before spending its spare frame budget on optical flow."""
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from config import VisionConfig
from core.latest_frame import LatestFrameBuffer
from pipeline.capture_worker import capture_worker


class CaptureMotionTests(unittest.TestCase):
    def test_frame_is_available_before_flow_and_eof_releases_capture(self):
        frames = LatestFrameBuffer()
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.return_value = 1000
        cap.read.side_effect = [(True, image), (True, image), (False, None)]
        def estimate(packet):
            self.assertEqual(frames.get_latest().frame_id, packet.frame_id)
            self.assertIs(frames.get_latest().frame, image)
        with patch('pipeline.capture_worker.cv2.VideoCapture', return_value=cap), \
             patch('pipeline.capture_worker.BoxMotionHistory') as motion:
            motion.return_value.snapshot.return_value = None
            motion.return_value.update.side_effect = estimate
            capture_worker(VisionConfig(), frames, threading.Event())
            self.assertEqual(motion.return_value.update.call_count, 2)
        self.assertTrue(frames.is_finished())
        cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
