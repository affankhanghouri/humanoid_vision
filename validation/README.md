# Local validation output

Generated logs, benchmark JSON and screenshots stay local and are ignored by Git.
Historical measured results are summarized in `REFACTOR_REPORT.md`.
From the repository root, with the environment activated:

```bash
python -m unittest discover -s tests -v
python benchmarks/benchmark_pipeline.py --seconds 20 --output validation/pipeline.json
```

The GUI benchmark requires the local model and video described in
`models/README.md` and `test_data/README.md`. Its JSON output is intentionally
excluded from commits; `tests/golden_tracking.json` is tracked regression data.
