#!/usr/bin/env python3
"""Evaluate common-format lane predictions against reviewed ground truth."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import atomic_write_json, evaluate_predictions, load_json, report_markdown


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "benchmarks/lane_acceptance/dataset.json"
DEFAULT_REPORTS = ROOT / "benchmarks/lane_acceptance/reports"


def safe_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._")
    return value or "unnamed_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_json(args.dataset)
    predictions = load_json(args.predictions)
    result = evaluate_predictions(dataset, predictions)
    name = safe_name(result.get("model", {}).get("name", "unnamed_model"))
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / f"{name}.json"
    markdown_path = args.output / f"{name}.md"
    atomic_write_json(json_path, result)
    markdown_path.write_text(report_markdown(result), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
