#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import engine as core
from src.utils.io import dump_json, dump_jsonl, load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate or repackage generated prediction JSONL records.")
    parser.add_argument("--predictions_jsonl", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="local",
        choices=["local", "none", "llm_judge"],
        help="local uses task parsers/code tests; llm_judge emits judge-ready records without scoring.",
    )
    parser.add_argument(
        "--judge_jsonl",
        type=str,
        default="",
        help="Optional path for LLM-judge-ready JSONL. Useful with --metric_mode llm_judge.",
    )
    return parser


def build_judge_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in records:
        meta = dict(row.get("meta", {}) or {})
        rows.append(
            {
                "split": row.get("split"),
                "task": row.get("task"),
                "example_index": row.get("example_index"),
                "prompt": row.get("prompt"),
                "target": row.get("target"),
                "prediction": row.get("prediction"),
                "meta": meta,
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.predictions_jsonl).expanduser().resolve()
    records = load_jsonl(path)
    metrics = core.evaluate_prediction_records(records, metric_mode=args.metric_mode)
    result = {
        "prediction_outputs": {"path": str(path), "n_records": int(len(records))},
        "metric_mode": str(args.metric_mode),
        "metrics": metrics,
    }
    if str(args.judge_jsonl).strip():
        judge_path = Path(args.judge_jsonl).expanduser()
        dump_jsonl(judge_path, build_judge_records(records))
        result["judge_inputs"] = {"path": str(judge_path), "n_records": int(len(records))}
    if str(args.output_json).strip():
        dump_json(Path(args.output_json).expanduser(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
