# Custom ego-lane experiment

This research pipeline is isolated from `src/`. It uses MobileNetV3-Small at
320x192 and predicts left/right x coordinates and visibility at 24 row anchors,
plus a global valid/invalid/ambiguous decision.

## Current decision

The experiment is complete and the v3 checkpoint is **rejected**. It passed the
development validation and CPU gates but abstained on all 100 frozen acceptance
frames, missing all 36 valid lanes. Do not integrate it and do not tune against
that consumed final set. See `RESULTS.md`.

## Dataset workflow

Build independent source-video splits, then propose and review labels:

```bash
python benchmarks/custom_ego_lane/build_dataset.py \
  --source /data/train_a.mp4:train_a:train \
  --source /data/validation_a.mp4:validation_a:validation
python benchmarks/custom_ego_lane/propose_labels.py DATASET.json --clip CLIP_ID
python benchmarks/custom_ego_lane/temporal_fill.py DATASET.json
python benchmarks/custom_ego_lane/review_sheets.py DATASET.json --output REVIEW_DIR
python benchmarks/custom_ego_lane/annotate.py --dataset DATASET.json
python benchmarks/custom_ego_lane/validate_dataset.py DATASET.json
python benchmarks/custom_ego_lane/dataset_quality.py DATASET.json
```

Machine proposals remain pending. Only completed `HUMAN_REVIEWED` records enter
training. Rejected proposals keep their original provenance. The append scripts
add an independent validation clip or selected deduplicated timestamps without
rebuilding the existing train split.

## Training, export, and timing

```bash
python benchmarks/custom_ego_lane/train.py DATASET.json \
  --freeze-backbone --patience 8 --output candidate.pt
python benchmarks/custom_ego_lane/export_onnx.py \
  --checkpoint candidate.pt --manifest DATASET.json --output candidate.onnx
python benchmarks/custom_ego_lane/benchmark.py candidate.onnx \
  --dataset DATASET.json --split validation --frames 80
```

The benchmark requires an explicit development dataset, preventing accidental
use of the frozen acceptance set for timing or tuning. Model files (`*.pt`,
`*.onnx`) and local training media are ignored by Git.

The final acceptance workflow is documented for reproducibility, but must not
be rerun for this v3 experiment:

```bash
python benchmarks/custom_ego_lane/predict_onnx.py candidate.onnx \
  benchmarks/lane_acceptance/dataset.json final_predictions.json \
  --valid-threshold VALID_THRESHOLD --visibility-threshold 0.5
python benchmarks/lane_acceptance/evaluate.py final_predictions.json
```
