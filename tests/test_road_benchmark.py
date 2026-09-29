"""Input and mask semantics for the standalone TwinLiteNet benchmark."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from benchmark_road_twinlitenet import decode, preprocess


class RoadBenchmarkTests(unittest.TestCase):
    def test_input_is_rgb_unit_range_nchw_and_preserves_source(self):
        image = np.full((2, 3, 3), (0, 127, 255), dtype=np.uint8)
        original = image.copy()
        tensor = preprocess(image, 2, 3)
        self.assertEqual(tensor.shape, (1, 3, 2, 3))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0], (1, 127/255, 0), atol=1e-7)
        np.testing.assert_array_equal(image, original)

    def test_separate_heads_argmax_and_nearest_resize(self):
        road = np.array([[[[2, -3]], [[1, -1]]]], dtype=np.float32)
        lane = road[:, ::-1].copy()
        road_mask, lane_mask = decode([road, lane], (4, 2), 1, 2)
        np.testing.assert_array_equal(road_mask, ((0, 0, 1, 1), (0, 0, 1, 1)))
        np.testing.assert_array_equal(lane_mask, 1-road_mask)
        self.assertEqual(road_mask.dtype, np.uint8)

    def test_invalid_outputs_fail_instead_of_producing_misleading_masks(self):
        valid = np.zeros((1, 2, 2, 3), np.float32)
        for bad in (np.zeros((1, 1, 2, 3)), np.full_like(valid, np.nan)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                decode([valid, bad], (3, 2), 2, 3)


if __name__ == '__main__':
    unittest.main()
