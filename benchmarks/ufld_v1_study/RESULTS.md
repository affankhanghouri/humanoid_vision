# UFLD v1 ResNet-18 / CULane result

## Decision

**Rejected for this CPU target.** The fastest tested two-thread path was
OpenVINO FP32 with 264.66 ms mean and 326.21 ms P95 total latency. Its P95 is
3.26 times the 100 ms early-stop ceiling. Per the study protocol, the work
stopped before the 20-frame visual screen, 100-frame prediction generation,
and acceptance-set evaluation.

No production code or defaults were changed. The fixed acceptance annotations
were not modified or used for training.

## Model and checks

- Model: UFLD v1, ResNet-18, CULane, four structured lane slots.
- Input/output: `1x3x288x800` RGB-normalized input; `1x201x18x4` logits.
- Parameters: 44,517,392; artifact size: 178,139,214 bytes (169.9 MiB).
- ONNX Checker: passed.
- PyTorch/ONNX comparison: passed on acceptance frame IDs 0, 20, and 40.
  Maximum absolute logit errors were `1.359e-5`, `1.335e-5`, and `8.106e-6`.
  PyTorch execution used an operator-for-operator recreation loaded from the
  ONNX initializers because the official checkpoint host was severely
  throttled; this validates the exported graph numerically but is not a fresh
  export from the original `.pth` file.

## CPU timing

Hardware: Intel Core i5-6300U (2 cores / 4 logical CPUs). Each result uses 10
warmup runs followed by 80 measured acceptance images. Times are milliseconds.

| Backend | Stage | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|
| ORT FP32 1 thread | preprocess | 11.72 | 11.46 | 13.92 | 21.92 |
| | inference | 387.92 | 381.40 | 448.15 | 537.96 |
| | postprocess | 0.63 | 0.57 | 1.01 | 1.26 |
| | **total** | **400.26** | **394.73** | **459.81** | **553.05** |
| ORT FP32 2 threads | preprocess | 13.49 | 12.31 | 20.35 | 26.74 |
| | inference | 285.79 | 279.59 | 423.99 | 561.85 |
| | postprocess | 0.89 | 0.76 | 1.21 | 4.52 |
| | **total** | **300.17** | **293.63** | **444.84** | **587.01** |
| OpenVINO FP32 1 thread | preprocess | 11.27 | 10.71 | 15.60 | 23.38 |
| | inference | 431.65 | 394.33 | 633.23 | 755.95 |
| | postprocess | 0.71 | 0.59 | 1.08 | 2.50 |
| | **total** | **443.63** | **405.21** | **651.33** | **768.11** |
| OpenVINO FP32 2 threads | preprocess | 11.53 | 11.30 | 14.35 | 15.86 |
| | inference | 252.46 | 243.20 | 316.01 | 404.42 |
| | postprocess | 0.67 | 0.66 | 0.96 | 1.51 |
| | **total** | **264.66** | **254.46** | **326.21** | **421.22** |

## Quality gate status

The CPU gate failed before visual quality testing. Therefore:

- 20-frame representative visual screen: not run by design.
- Full 100-frame acceptance metrics: not run by design.
- Visual failure cases: none generated; timing alone caused rejection.
- Production integration: not attempted.

The next lane direction is a separate, versioned urban training dataset built
from training-only footage that represents intersections, parked vehicles,
curbs, crosswalks, faded or absent markings, turns, and occlusion. The existing
100-frame acceptance set must remain frozen and evaluation-only.
