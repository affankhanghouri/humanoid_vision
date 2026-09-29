#!/usr/bin/env python3
"""Check UFLD ONNX validity and compare it with a PyTorch execution of its graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import numpy_helper

from benchmark import DEFAULT_DATASET, DEFAULT_MODEL, preprocess


def attributes(node) -> dict:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def torch_graph(model: onnx.ModelProto, input_tensor: np.ndarray) -> np.ndarray:
    """Run the small, static UFLD export with PyTorch operators."""
    values = {model.graph.input[0].name: torch.from_numpy(input_tensor)}
    values.update({item.name: torch.from_numpy(numpy_helper.to_array(item).copy()) for item in model.graph.initializer})
    for node in model.graph.node:
        args = [values[name] for name in node.input if name]
        attr = attributes(node)
        if node.op_type == "Conv":
            pads = attr.get("pads", [0, 0, 0, 0])
            if pads[:2] != pads[2:]:
                raise ValueError(f"asymmetric padding is unsupported: {pads}")
            result = F.conv2d(
                args[0], args[1], args[2] if len(args) > 2 else None,
                stride=tuple(attr.get("strides", [1, 1])), padding=tuple(pads[:2]),
                dilation=tuple(attr.get("dilations", [1, 1])), groups=attr.get("group", 1),
            )
        elif node.op_type == "Relu":
            result = F.relu(args[0])
        elif node.op_type == "MaxPool":
            pads = attr.get("pads", [0, 0, 0, 0])
            result = F.max_pool2d(
                args[0], tuple(attr["kernel_shape"]), tuple(attr.get("strides", attr["kernel_shape"])),
                tuple(pads[:2]), tuple(attr.get("dilations", [1, 1])), attr.get("ceil_mode", 0) != 0,
            )
        elif node.op_type == "Add":
            result = args[0] + args[1]
        elif node.op_type == "Constant":
            result = torch.from_numpy(numpy_helper.to_array(attr["value"]).copy())
        elif node.op_type == "Reshape":
            shape = tuple(int(value) for value in args[1].tolist())
            result = args[0].reshape(shape)
        elif node.op_type == "Gemm":
            left = args[0].transpose(-1, -2) if attr.get("transA", 0) else args[0]
            right = args[1].transpose(-1, -2) if attr.get("transB", 0) else args[1]
            result = attr.get("alpha", 1.0) * (left @ right)
            if len(args) > 2:
                result = result + attr.get("beta", 1.0) * args[2]
        else:
            raise ValueError(f"unsupported operation: {node.op_type}")
        values[node.output[0]] = result
    return values[model.graph.output[0].name].detach().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("verification.json"))
    args = parser.parse_args()
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dataset = json.loads(args.dataset.read_text())["annotations"]
    comparisons = []
    for record in dataset[: args.frames]:
        image = cv2.imread(str(args.dataset.parent / record["image"]))
        tensor = preprocess(image)
        expected = torch_graph(model, tensor)
        actual = session.run(None, {input_name: tensor})[0]
        delta = np.abs(expected - actual)
        comparisons.append(
            {
                "frame_id": record["frame_id"],
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
                "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(expected, actual, rtol=1e-4, atol=1e-5)),
            }
        )
    payload = {
        "onnx_checker": "passed",
        "pytorch_method": "PyTorch operator recreation loaded from ONNX initializers",
        "comparisons": comparisons,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not all(item["allclose_rtol_1e-4_atol_1e-5"] for item in comparisons):
        raise SystemExit("PyTorch and ONNX outputs differ")


if __name__ == "__main__":
    main()
