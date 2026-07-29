#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import ExperimentConfig, config_from_dict, parse_eval_max_new_tokens_by_task
from data import prepare_data
from state import PrototypeFederatedState
from lora import add_expert_adapters_if_needed, build_model_bundle
import engine as core
from train import _evaluate_split
from src.utils.io import dump_json, dump_jsonl, ensure_dir
from src.utils.logging import SwanLabRun, flatten_metrics


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved FedWeave checkpoint on val/test without retraining.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        default=[],
        help="Checkpoint path. Repeat this flag to evaluate multiple checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        type=str,
        action="append",
        default=[],
        help="Glob pattern for checkpoints, e.g. 'outputs/fedweave/**/checkpoints/best_val_loss.pt'.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        action="append",
        default=[],
        help="Root directory to search recursively for checkpoints/<name>.pt files.",
    )
    parser.add_argument(
        "--checkpoint_names",
        type=str,
        default="best_val_loss.pt,final.pt",
        help="Comma-separated checkpoint filenames used with --checkpoint_root.",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "both"])
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--batch_output_json", type=str, default="")
    parser.add_argument("--predictions_jsonl", type=str, default="")
    parser.add_argument("--include_predictions", dest="include_predictions", action="store_true")
    parser.add_argument("--no_include_predictions", dest="include_predictions", action="store_false")
    parser.add_argument(
        "--metric_mode",
        type=str,
        default="local",
        choices=["local", "none", "llm_judge"],
        help="How to score generated predictions: local parsers/code tests, no scoring, or LLM-judge-ready export.",
    )
    parser.add_argument("--eval_max_new_tokens", type=int, default=None)
    parser.add_argument("--eval_max_new_tokens_by_task", type=str, default="")
    parser.add_argument(
        "--eval_routing",
        type=str,
        default="",
        choices=["", "checkpoint", "soft", "topk"],
        help="FedWeave eval routing override: checkpoint config, full soft routing, or sparse top-k.",
    )
    parser.add_argument(
        "--eval_top_k",
        type=int,
        default=None,
        help="Eval-only router top-k. Use with --eval_routing topk; does not change the checkpoint config.",
    )
    parser.add_argument(
        "--eval_m_tau",
        type=float,
        default=None,
        help="Eval-only cumulative-probability threshold for top-k routing. Defaults to 1.0 in topk mode.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="Override evaluation batch size; 0 keeps checkpoint/default auto sizing.",
    )
    parser.add_argument("--use_swanlab", dest="use_swanlab", action="store_true")
    parser.add_argument("--no_use_swanlab", dest="use_swanlab", action="store_false")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--swanlab_project", type=str, default="FedWeave")
    parser.add_argument("--swanlab_name", type=str, default="")
    parser.add_argument(
        "--swanlab_log_every_n_examples",
        type=int,
        default=10,
        help="Log live eval progress to SwanLab every N examples; task-end and final metrics are always logged.",
    )
    parser.add_argument("--hf_offline", dest="hf_offline", action="store_true")
    parser.add_argument("--no_hf_offline", dest="hf_offline", action="store_false")
    parser.add_argument("--continue_on_error", dest="continue_on_error", action="store_true")
    parser.add_argument("--no_continue_on_error", dest="continue_on_error", action="store_false")
    parser.set_defaults(include_predictions=True)
    parser.set_defaults(use_swanlab=False)
    parser.set_defaults(continue_on_error=False)
    parser.set_defaults(hf_offline=None)
    return parser


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint at {path} is not a dict payload.")
    return payload


def _validate_checkpoint_method(payload: Dict[str, Any]) -> None:
    method = str(payload.get("method", "")).strip().lower()
    if method != "fedweave":
        raise ValueError("The selected checkpoint is not a FedWeave checkpoint.")


def _resolve_config(payload: Dict[str, Any], hf_offline_override: Optional[bool]) -> ExperimentConfig:
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain a serialized config dict.")
    config = config_from_dict(raw_config)
    if hf_offline_override is not None:
        config.data.hf_offline = bool(hf_offline_override)
    config.logging.use_swanlab = False
    config.logging.show_progress = True
    return config


def _pop_prediction_records(metrics: Dict[str, Any], split_name: str) -> list[Dict[str, Any]]:
    records = metrics.pop("prediction_records", [])
    if not isinstance(records, list):
        return []
    rows = []
    for row in records:
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        payload["split"] = str(split_name)
        rows.append(payload)
    return rows


def _checkpoint_arg(args: argparse.Namespace) -> str:
    checkpoints = getattr(args, "checkpoint", [])
    if isinstance(checkpoints, str):
        return checkpoints
    if isinstance(checkpoints, list) and len(checkpoints) == 1:
        return str(checkpoints[0])
    raise ValueError("run_checkpoint_eval expects exactly one checkpoint.")


def _swanlab_scalar_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool):
            out[key] = value
            continue
        if isinstance(value, int):
            out[key] = int(value)
            continue
        if isinstance(value, float):
            if value == value and value not in (float("inf"), float("-inf")):
                out[key] = float(value)
            continue
        if isinstance(value, str):
            out[key] = value
    return out


def _apply_eval_routing_overrides(config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    routing = str(getattr(args, "eval_routing", "") or "checkpoint").strip().lower()
    eval_top_k = getattr(args, "eval_top_k", None)
    eval_m_tau = getattr(args, "eval_m_tau", None)

    if routing == "soft" and eval_top_k is not None:
        raise ValueError("--eval_routing soft cannot be combined with --eval_top_k.")
    if routing == "topk" and (eval_top_k is None or int(eval_top_k) <= 0):
        raise ValueError("--eval_routing topk requires --eval_top_k > 0.")
    if eval_top_k is not None and int(eval_top_k) <= 0:
        raise ValueError("--eval_top_k must be > 0.")
    if eval_m_tau is not None and not (0.0 <= float(eval_m_tau) <= 1.0):
        raise ValueError("--eval_m_tau must be in [0, 1].")

    if routing == "soft":
        config.router.m_select = 0
    elif routing == "topk":
        config.router.m_select = int(eval_top_k)
        config.router.m_tau = 1.0 if eval_m_tau is None else float(eval_m_tau)
    elif eval_top_k is not None:
        config.router.m_select = int(eval_top_k)

    if routing != "topk" and eval_m_tau is not None:
        config.router.m_tau = float(eval_m_tau)

    return {
        "eval_routing": routing,
        "eval_m_select": int(config.router.m_select),
        "eval_m_tau": float(config.router.m_tau),
    }


def run_checkpoint_eval(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_path = Path(_checkpoint_arg(args)).expanduser().resolve()
    payload = _load_checkpoint(ckpt_path)
    _validate_checkpoint_method(payload)
    method = "fedweave"
    config = _resolve_config(payload, args.hf_offline)
    if args.eval_max_new_tokens is not None:
        config.train.eval_max_new_tokens = int(args.eval_max_new_tokens)
    if str(args.eval_max_new_tokens_by_task).strip():
        config.train.eval_max_new_tokens_by_task = parse_eval_max_new_tokens_by_task(args.eval_max_new_tokens_by_task)
    if int(args.eval_batch_size) < 0:
        raise ValueError("--eval_batch_size must be >= 0; use 0 for automatic sizing.")
    if int(args.eval_batch_size) > 0:
        config.train.eval_batch_size = int(args.eval_batch_size)
    eval_routing_config = _apply_eval_routing_overrides(config, args)
    data = prepare_data(config)
    model_bundle = build_model_bundle(config)
    swanlab_name = str(args.swanlab_name).strip() or f"eval_{method}_{ckpt_path.parent.parent.name}_{ckpt_path.stem}"
    swanlab_run = SwanLabRun(
        enabled=bool(args.use_swanlab),
        project=str(args.swanlab_project),
        name=swanlab_name,
        mode=str(args.swanlab_mode),
        config={
            "checkpoint": str(ckpt_path),
            "method": method,
            "split": str(args.split),
            "metric_mode": str(args.metric_mode),
            "include_predictions": bool(args.include_predictions),
            "eval_batch_size": int(getattr(config.train, "eval_batch_size", 0) or 0),
            "eval_max_new_tokens": config.train.eval_max_new_tokens,
            "eval_max_new_tokens_by_task": dict(config.train.eval_max_new_tokens_by_task),
            "eval_routing": eval_routing_config,
            "checkpoint_config": payload.get("config", {}),
        },
    )

    def _make_eval_progress_callback(split_name: str):
        if not bool(args.use_swanlab):
            return None
        log_every = max(1, int(args.swanlab_log_every_n_examples))

        def _callback(event_payload: Dict[str, Any]) -> None:
            n_examples = int(event_payload.get("n_examples", 0) or 0)
            event = str(event_payload.get("event", ""))
            if event == "example" and n_examples not in (0, 1) and n_examples % log_every != 0:
                return
            flat = flatten_metrics(event_payload, prefix=f"eval/live/{split_name}")
            flat["eval/live/checkpoint"] = str(ckpt_path)
            flat["eval/live/method"] = method
            swanlab_run.log(_swanlab_scalar_payload(flat), step=max(0, n_examples))

        return _callback

    split_to_examples = {}
    if args.split in ("val", "both"):
        split_to_examples["val"] = data.global_val_by_task
    if args.split in ("test", "both"):
        split_to_examples["test"] = data.global_test_by_task

    results: Dict[str, Any] = {
        "checkpoint": str(ckpt_path),
        "method": method,
        "split": args.split,
        "metric_mode": args.metric_mode,
        "eval_routing": eval_routing_config,
        "config": payload.get("config", {}),
        "metrics": {},
        "prediction_outputs": {},
    }
    prediction_records: list[Dict[str, Any]] = []
    include_predictions = bool(args.include_predictions)
    compute_inline_metrics = bool(str(args.metric_mode) == "local")

    try:
        add_expert_adapters_if_needed(model_bundle.model, int(payload["k_experts"]))
        state = PrototypeFederatedState(
            k_experts=int(payload["k_experts"]),
            server_expert_sd=payload["server_expert_sd"],
            router_state_dicts=payload.get("router_state_dicts"),
            current_round=int(payload.get("round", 0)),
            round_history=[],
        )
        for split_name, split_by_task in split_to_examples.items():
            results["metrics"][split_name] = _evaluate_split(
                model_bundle=model_bundle,
                state=state,
                split_by_task=split_by_task,
                config=config,
                include_predictions=include_predictions,
                compute_task_metrics=compute_inline_metrics,
                progress_desc=f"fedweave {split_name}",
                disable_progress=not bool(config.logging.show_progress),
                eval_progress_callback=_make_eval_progress_callback(split_name),
            )
            prediction_records.extend(_pop_prediction_records(results["metrics"][split_name], split_name))

        if include_predictions:
            predictions_path = str(args.predictions_jsonl).strip()
            if not predictions_path:
                predictions_path = str(ensure_dir(ckpt_path.parent / "eval") / f"{method}_{args.split}_predictions.jsonl")
            dump_jsonl(Path(predictions_path), prediction_records)
            results["prediction_outputs"] = {
                "path": predictions_path,
                "n_records": int(len(prediction_records)),
            }
            if str(args.metric_mode) == "local" and compute_inline_metrics:
                for split_metrics in results["metrics"].values():
                    if isinstance(split_metrics, dict):
                        split_metrics["metric_mode"] = "local"
            else:
                results["metrics"] = core.evaluate_prediction_records(
                    prediction_records,
                    metric_mode=str(args.metric_mode),
                )
        else:
            if str(args.predictions_jsonl).strip():
                raise ValueError("--predictions_jsonl requires --include_predictions.")
            results["prediction_outputs"] = {"path": "", "n_records": 0}

        output_path = str(args.output_json).strip()
        if output_path:
            dump_json(Path(output_path), results)
        else:
            default_path = ensure_dir(ckpt_path.parent / "eval") / f"{method}_{args.split}_eval.json"
            dump_json(default_path, results)
            results["output_json"] = str(default_path)
        if bool(args.use_swanlab):
            final_payload = flatten_metrics(results.get("metrics", {}), prefix="eval/final")
            final_payload["eval/final/prediction_records"] = int(len(prediction_records))
            swanlab_run.log(_swanlab_scalar_payload(final_payload))
        return results
    finally:
        swanlab_run.finish()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _resolve_checkpoints(args: argparse.Namespace) -> List[Path]:
    candidates: List[Path] = []
    for path in getattr(args, "checkpoint", []) or []:
        candidates.append(Path(path).expanduser())
    for pattern in getattr(args, "checkpoint_glob", []) or []:
        matches = glob.glob(str(Path(pattern).expanduser()), recursive=True)
        candidates.extend(Path(match) for match in matches)
    checkpoint_names = set(_split_csv(getattr(args, "checkpoint_names", "")))
    for root in getattr(args, "checkpoint_root", []) or []:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            raise FileNotFoundError(f"Checkpoint root does not exist: {root_path}")
        for ckpt in root_path.rglob("*.pt"):
            if ckpt.parent.name == "checkpoints" and (not checkpoint_names or ckpt.name in checkpoint_names):
                candidates.append(ckpt)

    resolved: List[Path] = []
    seen = set()
    for candidate in candidates:
        ckpt = candidate.resolve()
        if ckpt in seen:
            continue
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
        if not ckpt.is_file():
            raise ValueError(f"Checkpoint path is not a file: {ckpt}")
        seen.add(ckpt)
        resolved.append(ckpt)
    resolved.sort(key=lambda path: str(path))
    return resolved


def _compact_eval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint": result.get("checkpoint"),
        "method": result.get("method"),
        "split": result.get("split"),
        "metric_mode": result.get("metric_mode"),
        "output_json": result.get("output_json"),
        "prediction_outputs": result.get("prediction_outputs"),
        "metrics": result.get("metrics", {}),
    }


def run_many_checkpoint_eval(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoints = _resolve_checkpoints(args)
    if not checkpoints:
        raise ValueError("No checkpoints selected. Pass --checkpoint, --checkpoint_glob, or --checkpoint_root.")
    if len(checkpoints) == 1:
        single_args = copy.copy(args)
        single_args.checkpoint = [str(checkpoints[0])]
        return run_checkpoint_eval(single_args)
    if str(args.predictions_jsonl).strip():
        raise ValueError("--predictions_jsonl is only supported for a single checkpoint; omit it for batch eval.")

    batch: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "n_checkpoints": int(len(checkpoints)),
        "checkpoints": [str(path) for path in checkpoints],
        "results": [],
        "errors": [],
    }
    for ckpt in checkpoints:
        run_args = copy.copy(args)
        run_args.checkpoint = [str(ckpt)]
        run_args.output_json = ""
        run_args.batch_output_json = ""
        run_args.predictions_jsonl = ""
        try:
            result = run_checkpoint_eval(run_args)
            batch["results"].append(_compact_eval_result(result))
        except Exception as exc:
            error_payload = {"checkpoint": str(ckpt), "error": str(exc)}
            batch["errors"].append(error_payload)
            if not bool(args.continue_on_error):
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output_path = str(args.batch_output_json or args.output_json).strip()
    if not output_path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(ensure_dir(PROJECT_ROOT / "results" / "eval_batches") / f"eval_batch_{stamp}.json")
    dump_json(Path(output_path), batch)
    batch["output_json"] = output_path
    return batch


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_eval_parser()
    args = parser.parse_args(argv)
    results = run_many_checkpoint_eval(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
