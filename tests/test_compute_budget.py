"""A slow pose pass must not knowingly starve tracking freshness."""
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core.types import FramePacket
from core.onnx_runtime import configure_cpu_onnx
from scheduling.scheduler import ComputeScheduler
from scheduling.task import ComputeTask, TaskPolicy


class ComputeBudgetTests(unittest.TestCase):
    def test_pose_yields_when_estimated_cost_would_expire_tracking(self):
        scheduler = ComputeScheduler((TaskPolicy('detection', 100, .25, .15),
                                      TaskPolicy('pose', 70, .5, .15)), max_tracking_age=.75)
        image = np.zeros((2, 2, 3), np.uint8)
        stop = threading.Event()
        def submit(frame_id, timestamp):
            for kind in ('detection', 'pose'):
                scheduler.submit(ComputeTask(kind, frame_id, timestamp, FramePacket(frame_id, timestamp, image)))
        with patch('scheduling.scheduler.time.perf_counter', return_value=10):
            submit(1, 10)
            self.assertEqual(scheduler.get_next(stop).task_type, 'detection')
        scheduler.mark_executed('detection', .4, 10)
        scheduler.mark_executed('pose', .3, 9.7)
        with patch('scheduling.scheduler.time.perf_counter', return_value=10.4):
            submit(2, 10.4)
            # Fairness would normally pick pose; its budget now forces detection.
            self.assertEqual(scheduler.get_next(stop).task_type, 'detection')
        # When measured costs recover, pose becomes eligible again.
        for _ in range(5):
            scheduler.mark_executed('detection', .20, 10.4)
            scheduler.mark_executed('pose', .12, 10.1)
        with patch('scheduling.scheduler.time.perf_counter', return_value=10.65):
            submit(3, 10.65)
            self.assertEqual(scheduler.get_next(stop).task_type, 'pose')

    def test_cpu_onnx_session_configured_once_and_other_backends_untouched(self):
        import onnxruntime as ort
        model, predictor = Mock(), Mock()
        predictor.device.type = 'cpu'
        predictor.model.backend.session.get_providers.return_value = ['CPUExecutionProvider']
        configure_cpu_onnx(model, 'weights.onnx', 2)
        callback = model.add_callback.call_args.args[1]
        with patch('onnxruntime.InferenceSession') as session, patch('torch.set_num_threads') as threads:
            callback(predictor)
            callback(predictor)
        session.assert_called_once()
        self.assertEqual(threads.call_count, 2)
        threads.assert_called_with(1)
        options = session.call_args.kwargs['sess_options']
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(options.get_session_config_entry('session.intra_op.allow_spinning'), '0')
        untouched = Mock()
        configure_cpu_onnx(untouched, 'weights.pt', 2)
        untouched.add_callback.assert_not_called()
