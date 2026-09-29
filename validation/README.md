# Local validation output

This directory is for generated benchmark JSON, timing samples, screenshots, and
recordings. Its contents are intentionally ignored by Git; durable conclusions
belong in small Markdown reports under `benchmarks/`.

With the environment active:

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_pipeline.py \
  --headless --seconds 20 --output validation/pipeline.json
python benchmarks/benchmark_pipeline.py \
  --headless --seconds 20 --road --output validation/pipeline-road.json
```

The replay benchmarks require the local models and video documented in
`models/README.md` and `test_data/README.md`. `tests/golden_tracking.json` is
committed regression data, not generated validation output.
