from argparse import Namespace
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = PROJECT_ROOT / "models" / "twinlitenetplus_nano.pth"
OUTPUT = PROJECT_ROOT / "models" / "twinlitenetplus_nano_640.onnx"

# Model code we downloaded earlier
sys.path.insert(0, "/tmp/tlnp")

from model.model import TwinLiteNetPlus


def main():
    print("Loading TwinLiteNet+ Nano...")

    args = Namespace(config="nano")

    model = TwinLiteNetPlus(args)

    checkpoint = torch.load(
        WEIGHTS,
        map_location="cpu",
    )

    # Handle either plain state_dict or wrapped checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # Handle DataParallel checkpoints
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.eval()

    dummy = torch.zeros(
        1,
        3,
        384,
        640,
        dtype=torch.float32,
    )

    print("Exporting ONNX...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(OUTPUT),

            input_names=["images"],
            output_names=[
                "drivable_area",
                "lane_line",
            ],

            opset_version=17,

            do_constant_folding=True,

            dynamo=False,
        )

    print()
    print("Export complete:")
    print(OUTPUT)

    print()
    print("Testing ONNX file...")

    import onnx

    onnx_model = onnx.load(str(OUTPUT))
    onnx.checker.check_model(onnx_model)

    print("ONNX check passed.")


if __name__ == "__main__":
    main()