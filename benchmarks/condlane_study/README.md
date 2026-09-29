# CondLaneNet acceptance study

This standalone study does not import or modify production code. Model files
are stored under `models/` and are ignored by Git.

From the repository root, export and verify the official checkpoint:

```bash
python benchmarks/condlane_study/export_onnx.py
```

Run ORT and OpenVINO with one and two CPU threads, create all 100 common-format
predictions, invoke the acceptance metrics, and save failure visuals:

```bash
python benchmarks/condlane_study/benchmark.py
python benchmarks/lane_acceptance/evaluate.py \
  benchmarks/condlane_study/output/predictions.json \
  --output benchmarks/condlane_study/output/evaluator_report
```

The adapter uses the official heatmap threshold of 0.5. It declares a valid
ego lane only when two decoded lane instances plausibly bracket the image
center. Partial or geometrically questionable predictions are marked
ambiguous, which the fixed evaluator treats as abstention. No acceptance labels
are used to tune thresholds or geometry.

See `RESULTS.md` for the decision, `SOURCES.md` for provenance, and
`output/visuals/failure_contact_sheet.jpg` for green ground truth versus red
prediction overlays.

