"""Small CLI and startup-configuration smoke tests."""
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import VisionConfig
from main import parse_args, validate_config


class CliTests(unittest.TestCase):
    def test_supported_flags_parse(self):
        args = parse_args([
            "--risk-demo", "--no-road-overlay", "--record", "outputs/demo.mp4",
            "--video", "clip.mp4", "--verbose",
        ])
        self.assertTrue(args.risk_demo)
        self.assertTrue(args.no_road_overlay)
        self.assertTrue(args.verbose)
        self.assertEqual(args.record, "outputs/demo.mp4")
        self.assertEqual(args.video, "clip.mp4")

    def test_missing_model_reports_exact_path(self):
        missing = Path("/tmp/humanoid-vision-missing-detector.onnx")
        config = replace(VisionConfig(), model_path=str(missing))
        with self.assertRaisesRegex(FileNotFoundError, str(missing)):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
