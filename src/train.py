#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from torch.utils.data import DataLoader

from config import (
    ExperimentConfig,
    build_parser,
    build_config_from_namespace,
    config_to_dict,
)
from data import PreparedData, prepare_data
from state import (
    LocalClusterAssignment,
    PrototypeFederatedState,
    aggregate_expert_and_router_updates,
    aggregate_weighted_state_dicts,
    initialize_prototype_federated_state,
)
from lora import (
    Example,
    ModelBundle,
    adapter_name_for_cluster,
    add_expert_adapters_if_needed,
    build_local_scheduler,
    build_model_bundle,
    build_optimizer,
    build_router_dict,
    build_split_lr_optimizer,
    clone_state_dict_cpu,
    collate_fn_builder,
    compute_batch_expert_weights,
    compute_fixed_weight_log_probs,
    compute_mixture_logits,
    compute_nll_loss,
    dump_router_states,
    extract_lora_a_vector,
    extract_lora_ab_vector,
    extract_lora_b_vector,
    get_adapter_state,
    greedy_generate,
    greedy_generate_batch,
    greedy_generate_mixture,
    load_all_expert_states,
    load_router_states,
    make_dataloader_generator,
    move_batch_to_device,
    set_trainable_adapter,
    set_trainable_router,
    _per_example_ce_loss,
    _per_example_nll_loss,
)
import engine as core
from src.utils.io import dump_json, ensure_dir
from src.utils.paths import OUTPUTS_DIR, RESULTS_DIR
from src.utils.progress import format_metric, make_progress, progress_write
from src.utils.logging import SwanLabRun, flatten_metrics


def _default_output_dir() -> Path:
    return OUTPUTS_DIR / "base" / "fedweave"


def _expert_key(expert_id: int) -> str:
    return f"expert_{int(expert_id)}"


def _named_expert_counts(raw: Dict[int, float | int], k_experts: int, *, as_float: bool = False) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for expert_id in range(int(k_experts)):
        value = raw.get(int(expert_id), 0.0 if as_float else 0)
        out[_expert_key(expert_id)] = float(value) if as_float else int(value)
    return out


def _build_usage_summary(
    *,
    k_experts: int,
    examples_per_expert: Dict[int, int],
    uploads_per_expert: Dict[int, int],
    clients_per_expert: Dict[int, int],
) -> Dict[str, Any]:
    touched_experts = sum(
        1
        for expert_id in range(int(k_experts))
        if int(examples_per_expert.get(expert_id, 0)) > 0 or int(uploads_per_expert.get(expert_id, 0)) > 0
    )
    total_examples = int(sum(int(v) for v in examples_per_expert.values()))
    total_uploads = int(sum(int(v) for v in uploads_per_expert.values()))
    return {
        "touched_experts": int(touched_experts),
        "touched_expert_fraction": float(touched_experts / max(1, int(k_experts))),
        "total_examples": int(total_examples),
        "total_uploads": int(total_uploads),
        "avg_examples_per_touched_expert": float(total_examples / max(1, touched_experts)),
        "examples_per_expert": _named_expert_counts(examples_per_expert, int(k_experts), as_float=False),
        "uploads_per_expert": _named_expert_counts(uploads_per_expert, int(k_experts), as_float=False),
        "clients_per_expert": _named_expert_counts(clients_per_expert, int(k_experts), as_float=False),
    }


@dataclass
class _EvalStats:
    task_metrics: Dict[str, float] = field(default_factory=dict)
    task_losses: Dict[str, float] = field(default_factory=dict)
    task_official_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    prediction_records: List[Dict[str, Any]] = field(default_factory=list)
    total_examples: int = 0
    weighted_loss_sum: float = 0.0
    running_examples: int = 0
    running_loss_sum: float = 0.0


def _capture_eval_generation(
    *,
    generate_predictions: bool,
    pred: Optional[str],
    ex: Example,
    preds: List[str],
    targets: List[str],
    metas: List[Dict[str, Any]],
) -> None:
    if generate_predictions:
        preds.append(str(pred))
        targets.append(str(ex.target))
        metas.append(dict(getattr(ex, "meta", {}) or {}))


def _record_eval_loss(
    *,
    stats: _EvalStats,
    loss: torch.Tensor,
    losses: List[float],
    task: str,
    example_index: int,
    max_new_tokens: int,
    eval_progress_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    if torch.isfinite(loss):
        loss_value = float(loss.item())
        losses.append(loss_value)
        stats.running_examples += 1
        stats.running_loss_sum += loss_value
        if eval_progress_callback is not None:
            eval_progress_callback(
                {
                    "event": "example",
                    "task": str(task),
                    "example_index": int(example_index),
                    "n_examples": int(stats.running_examples),
                    "avg_loss": float(stats.running_loss_sum / max(1, stats.running_examples)),
                    "task_loss": float(np.mean(losses)) if losses else 0.0,
                    "loss": loss_value,
                    "max_new_tokens": int(max_new_tokens),
                }
            )


def _append_eval_prediction_record(
    *,
    stats: _EvalStats,
    include_predictions: bool,
    task: str,
    example_index: int,
    ex: Example,
    pred: Optional[str],
    max_new_tokens: int,
    loss: torch.Tensor,
    weights_key: Optional[str] = None,
    weights_payload: Optional[List[float]] = None,
) -> None:
    if not include_predictions:
        return
    raw_meta = dict(getattr(ex, "meta", {}) or {})
    record: Dict[str, Any] = {
        "task": str(task),
        "example_index": int(example_index),
    }
    if weights_key is not None:
        record[weights_key] = weights_payload
    record.update(
        {
            "prompt": str(ex.prompt),
            "target": str(ex.target),
            "meta": raw_meta,
            "prediction": str(pred),
            "max_new_tokens": int(max_new_tokens),
            "loss": float(loss.item()) if torch.isfinite(loss) else None,
        }
    )
    stats.prediction_records.append(record)


def _finish_eval_task(
    *,
    stats: _EvalStats,
    task: str,
    n_examples: int,
    losses: List[float],
    preds: List[str],
    targets: List[str],
    metas: List[Dict[str, Any]],
    compute_task_metrics: bool,
    eval_progress_callback: Optional[Callable[[Dict[str, Any]], None]],
) -> None:
    if compute_task_metrics:
        official_metrics = core.compute_task_official_metrics(task, preds, targets, metas=metas)
        stats.task_metrics[task] = float(official_metrics["main_score"])
        stats.task_official_metrics[task] = official_metrics
    stats.task_losses[task] = float(np.mean(losses)) if losses else 0.0
    stats.total_examples += int(n_examples)
    stats.weighted_loss_sum += int(n_examples) * stats.task_losses[task]
    if eval_progress_callback is not None:
        eval_progress_callback(
            {
                "event": "task_end",
                "task": str(task),
                "n_examples": int(stats.running_examples),
                "avg_loss": float(stats.running_loss_sum / max(1, stats.running_examples)),
                "avg_macro": float(np.mean(list(stats.task_metrics.values()))) if stats.task_metrics else None,
                "task_losses": dict(stats.task_losses),
                "task_metrics": dict(stats.task_metrics),
            }
        )


def _build_eval_metrics(
    *,
    stats: _EvalStats,
    compute_task_metrics: bool,
    include_predictions: bool,
) -> Dict[str, Any]:
    metrics = {
        "n_examples": int(stats.total_examples),
        "avg_macro": float(np.mean(list(stats.task_metrics.values()))) if stats.task_metrics else None,
        "avg_loss": float(stats.weighted_loss_sum / max(1, stats.total_examples)),
        "task_metrics_computed": bool(compute_task_metrics),
        "task_metrics": stats.task_metrics,
        "task_official_metrics": stats.task_official_metrics,
        "task_losses": stats.task_losses,
    }
    if include_predictions:
        metrics["prediction_records"] = stats.prediction_records
    return metrics


def _allocate_bucket_local_steps(
    bucket_sizes: Sequence[int],
    total_steps: int,
    *,
    min_steps: int,
) -> List[int]:
    sizes = [max(0, int(size)) for size in bucket_sizes]
    num_buckets = len(sizes)
    if num_buckets == 0:
        return []

    total_steps = max(0, int(total_steps))
    min_steps = max(0, int(min_steps))
    if total_steps == 0:
        return [0 for _ in sizes]

    required_min_budget = int(num_buckets * min_steps)
    if min_steps > 0 and total_steps < required_min_budget:
        # If the total budget is too small to satisfy every bucket, prioritize
        # larger buckets first and leave the remaining buckets at zero.
        allocations = [0 for _ in sizes]
        ranked = sorted(range(num_buckets), key=lambda idx: (-sizes[idx], idx))
        for idx in ranked[:total_steps]:
            allocations[idx] = 1
        return allocations

    allocations = [min_steps for _ in sizes]
    remaining_steps = int(total_steps - required_min_budget)
    if remaining_steps <= 0:
        return allocations

    total_size = int(sum(sizes))
    if total_size <= 0:
        shares = [float(remaining_steps) / float(num_buckets) for _ in sizes]
    else:
        shares = [float(remaining_steps) * float(size) / float(total_size) for size in sizes]

    extra_steps = [int(share) for share in shares]
    leftover = int(remaining_steps - sum(extra_steps))
    remainders = [float(share - int(share)) for share in shares]
    ranked = sorted(range(num_buckets), key=lambda idx: (-remainders[idx], -sizes[idx], idx))
    for idx in ranked[:leftover]:
        extra_steps[idx] += 1

    return [int(allocations[idx] + extra_steps[idx]) for idx in range(num_buckets)]


def _build_interleaved_bucket_schedule(step_budgets: Sequence[int]) -> List[int]:
    budgets = [max(0, int(value)) for value in step_budgets]
    total_steps = int(sum(budgets))
    if total_steps <= 0:
        return []

    remaining = list(budgets)
    scores = [0 for _ in budgets]
    schedule: List[int] = []
    for _ in range(total_steps):
        for idx, budget in enumerate(budgets):
            if remaining[idx] > 0:
                scores[idx] += int(budget)
        candidates = [idx for idx, value in enumerate(remaining) if value > 0]
        if not candidates:
            break
        selected = max(candidates, key=lambda idx: (scores[idx], remaining[idx], -idx))
        schedule.append(int(selected))
        scores[selected] -= total_steps
        remaining[selected] -= 1
    return schedule


def _save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length.")
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))


def _should_save_periodic_checkpoint(config: ExperimentConfig, round_idx: int) -> bool:
    current_round = int(round_idx) + 1
    save_every = int(getattr(config.train, "save_every_n_rounds", 0))
    if save_every <= 0:
        return False
    return (current_round % save_every) == 0


def _periodic_checkpoint_path(checkpoint_root: Path, round_idx: int) -> Path:
    return checkpoint_root / f"best_val_loss_at_round_{int(round_idx):04d}.pt"


def _record_periodic_checkpoint(
    periodic_checkpoints: List[Dict[str, Any]],
    *,
    round_idx: int,
    path: Path,
    best_val_loss: Dict[str, Any],
) -> None:
    periodic_checkpoints.append(
        {
            "round": int(round_idx),
            "path": str(path),
            "source": "best_val_loss",
            "best_val_loss_metric": best_val_loss.get("metric"),
            "best_val_loss_round": best_val_loss.get("round"),
            "best_val_loss_path": best_val_loss.get("path"),
        }
    )


def _save_periodic_best_val_loss_checkpoint(
    *,
    config: ExperimentConfig,
    round_idx: int,
    checkpoint_root: Path,
    best_val_loss: Dict[str, Any],
    periodic_checkpoints: List[Dict[str, Any]],
) -> bool:
    if not _should_save_periodic_checkpoint(config, round_idx):
        return False
    best_path_value = best_val_loss.get("path")
    if best_path_value is None:
        return False
    best_path = Path(str(best_path_value))
    if not best_path.exists():
        return False
    current_round = int(round_idx) + 1
    periodic_checkpoint_path = _periodic_checkpoint_path(checkpoint_root, current_round)
    ensure_dir(periodic_checkpoint_path.parent)
    shutil.copy2(best_path, periodic_checkpoint_path)
    _record_periodic_checkpoint(
        periodic_checkpoints,
        round_idx=current_round,
        path=periodic_checkpoint_path,
        best_val_loss=best_val_loss,
    )
    return True


def _eval_max_new_tokens_for_task(config: ExperimentConfig, task: str) -> int:
    per_task = getattr(config.train, "eval_max_new_tokens_by_task", {}) or {}
    value = per_task.get(str(task).lower())
    if value is None:
        value = config.train.eval_max_new_tokens
    return max(1, int(value))


def _should_compute_val_task_metrics(config: ExperimentConfig, round_idx: int) -> bool:
    if not bool(getattr(config.train, "val_compute_task_metrics", True)):
        return False
    current_round = int(round_idx) + 1
    if current_round >= int(config.train.global_rounds):
        return True
    every = max(1, int(getattr(config.train, "val_task_metrics_every_n_rounds", 1)))
    return (current_round % every) == 0


def _record_val_metrics(round_summary: Dict[str, Any], val_metrics: Dict[str, Any]) -> None:
    round_summary["val_loss"] = float(val_metrics["avg_loss"])
    round_summary["val_task_losses"] = {
        str(task): float(value) for task, value in dict(val_metrics.get("task_losses", {})).items()
    }
    task_metrics_computed = bool(val_metrics.get("task_metrics_computed", True))
    round_summary["val_task_metrics_computed"] = task_metrics_computed
    if task_metrics_computed:
        round_summary["val_macro"] = float(val_metrics["avg_macro"])
        round_summary["val_task_metrics"] = val_metrics["task_metrics"]


def _build_round_swanlab_payload(round_summary: Dict[str, Any], prefix: str = "train") -> Dict[str, Any]:
    """Keep per-round SwanLab charts focused; full details remain in summary JSON."""
    scalar_keys = (
        "train_loss",
        "train_task_loss",
        "train_route_ce_loss",
        "train_route_ce_weighted_loss",
        "router_entropy",
        "router_top1",
        "val_loss",
        "val_macro",
        "val_task_metrics_computed",
    )
    payload: Dict[str, Any] = {}
    for key in scalar_keys:
        if key in round_summary:
            payload[f"{prefix}/{key}"] = round_summary[key]
    if isinstance(round_summary.get("val_task_losses"), dict):
        for task, value in round_summary["val_task_losses"].items():
            payload[f"{prefix}/val_task_losses/{task}"] = value
    if isinstance(round_summary.get("val_task_metrics"), dict):
        for task, value in round_summary["val_task_metrics"].items():
            payload[f"{prefix}/val_task_metrics/{task}"] = value
    return payload


def _maybe_update_best_checkpoint(
    *,
    tracker: Dict[str, Any],
    metric_value: Optional[float],
    mode: str,
    round_idx: int,
    path: Path,
    payload: Dict[str, Any],
) -> bool:
    if metric_value is None:
        return False
    current = tracker.get("metric")
    improved = current is None
    if current is not None:
        if mode == "max":
            improved = float(metric_value) > float(current)
        elif mode == "min":
            improved = float(metric_value) < float(current)
        else:
            raise ValueError(f"Unsupported checkpoint comparison mode: {mode}")
    if not improved:
        return False
    _save_checkpoint(path, payload)
    tracker["metric"] = float(metric_value)
    tracker["round"] = int(round_idx)
    tracker["path"] = str(path)
    return True


def _build_fedweave_checkpoint_payload(
    *,
    config: ExperimentConfig,
    round_idx: int,
    state: PrototypeFederatedState,
    prototypes: PrototypeArtifacts,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    bucket_min_steps: int,
    discovery_warmup_steps: int,
    discovery_warmup_batch_size: int,
    discovery_warmup_mode: str = "steps",
    discovery_warmup_epochs: int = 1,
    interleave_client_buckets: bool = True,
    router_aggregation_scope: str = "client",
    oracle_task_routing: bool = False,
    val_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "method": "fedweave",
        "round": int(round_idx),
        "config": config_to_dict(config),
        "k_experts": int(state.k_experts),
        "server_expert_sd": {int(k): clone_state_dict_cpu(v) for k, v in state.server_expert_sd.items()},
        "router_state_dicts": None if state.router_state_dicts is None else {
            int(k): clone_state_dict_cpu(v) for k, v in state.router_state_dicts.items()
        },
        "group_priors": [float(v) for v in prototypes.group_priors],
        "prototype_summary": prototypes.summary,
        "train_args": {
            "embedding_batch_size": int(embedding_batch_size),
            "normalize_embeddings": bool(normalize_embeddings),
            "bucket_min_steps": int(bucket_min_steps),
            "discovery_warmup_steps": int(discovery_warmup_steps),
            "discovery_warmup_batch_size": int(discovery_warmup_batch_size),
            "discovery_warmup_mode": str(discovery_warmup_mode),
            "discovery_warmup_epochs": int(discovery_warmup_epochs),
            "local_cluster_algorithm": str(prototypes.summary["local"].get("cluster_algorithm", "kmeans")),
            "prototype_signature_type": str(
                prototypes.summary["global"].get("prototype_signature_type", "lora_b")
            ),
            "interleave_client_buckets": bool(interleave_client_buckets),
            "router_aggregation_scope": str(router_aggregation_scope),
            "oracle_task_routing": bool(oracle_task_routing),
        },
        "val_metrics": val_metrics,
    }


def _default_results_dir() -> Path:
    return RESULTS_DIR / "base" / "fedweave"


def build_train_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Prototype-aligned MoE-FedLoRA training entry."
    parser.set_defaults(show_progress=True)
    parser.add_argument("--train_out_dir", type=str, default=str(_default_output_dir()))
    parser.add_argument("--train_results_dir", type=str, default=str(_default_results_dir()))
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--bucket_min_steps", type=int, default=1)
    parser.add_argument("--discovery_warmup_steps", type=int, default=5)
    parser.add_argument("--discovery_warmup_batch_size", type=int, default=4)
    parser.add_argument(
        "--discovery_warmup_mode",
        type=str,
        default="steps",
        choices=["steps", "epochs"],
    )
    parser.add_argument("--discovery_warmup_epochs", type=int, default=1)
    parser.add_argument(
        "--local_cluster_algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "spectral"],
        help="Per-client local clustering algorithm used during prototype discovery.",
    )
    parser.add_argument(
        "--prototype_signature_type",
        type=str,
        default="lora_b",
        choices=["lora_a", "lora_b", "lora_ab"],
        help="LoRA signature used for cross-client prototype alignment after local warmup.",
    )
    parser.add_argument("--normalize_embeddings", dest="normalize_embeddings", action="store_true")
    parser.add_argument("--no_normalize_embeddings", dest="normalize_embeddings", action="store_false")
    parser.add_argument(
        "--interleave_client_buckets",
        dest="interleave_client_buckets",
        action="store_true",
        help="In FedWeave, train one continuous client-local router across interleaved local buckets.",
    )
    parser.add_argument("--no_interleave_client_buckets", dest="interleave_client_buckets", action="store_false")
    parser.add_argument(
        "--router_aggregation_scope",
        type=str,
        default="client",
        choices=["client", "bucket"],
        help="FedWeave router aggregation scope: aggregate one router upload per client or per bucket.",
    )
    parser.add_argument(
        "--oracle_task_routing",
        dest="oracle_task_routing",
        action="store_true",
        help="Use task labels as an oracle FedWeave assignment: one expert per task, while keeping the normal training loop.",
    )
    parser.add_argument("--no_oracle_task_routing", dest="oracle_task_routing", action="store_false")
    parser.set_defaults(normalize_embeddings=True, interleave_client_buckets=True, oracle_task_routing=False)
    return parser


def _build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if int(args.bucket_min_steps) < 0:
        raise ValueError("--bucket_min_steps must be >= 0.")
    if int(args.discovery_warmup_steps) < 0:
        raise ValueError("--discovery_warmup_steps must be >= 0.")
    if int(args.discovery_warmup_batch_size) <= 0:
        raise ValueError("--discovery_warmup_batch_size must be > 0.")
    if int(args.discovery_warmup_epochs) <= 0:
        raise ValueError("--discovery_warmup_epochs must be > 0.")
    return build_config_from_namespace(
        args,
        out_dir=args.train_out_dir,
        results_dir=args.train_results_dir,
    )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return (matrix / norms).astype(np.float32)


def _auto_k_candidates(n_samples: int, *, scope: str) -> List[int]:
    if int(n_samples) < 3:
        return []
    if scope == "local":
        upper = min(8, int(n_samples) - 1)
    elif scope == "global":
        upper = min(8, int(n_samples) - 1)
    else:
        raise ValueError(f"Unsupported scope: {scope}")
    if upper < 2:
        return []
    return list(range(2, int(upper) + 1))


def _fit_adaptive_kmeans(
    features: np.ndarray,
    *,
    scope: str,
    seed: int,
) -> Dict[str, Any]:
    n_samples = int(features.shape[0])
    valid_k = _auto_k_candidates(n_samples, scope=scope)
    rows: List[Dict[str, Any]] = []
    if n_samples == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")
    if not valid_k:
        labels = np.zeros(n_samples, dtype=np.int64)
        centroid = np.mean(features, axis=0, keepdims=True)
        return {
            "selected_k": 1,
            "selected_silhouette": None,
            "labels": labels,
            "centroids": centroid.astype(np.float32),
            "rows": rows,
            "candidate_k_values": valid_k,
        }

    best_payload: Optional[Dict[str, Any]] = None
    for k in valid_k:
        model = KMeans(n_clusters=int(k), random_state=int(seed), n_init=10)
        model.fit(features)
        labels = np.asarray(model.labels_, dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {str(int(label)): int(count) for label, count in zip(unique, counts)}
        if np.unique(labels).size < 2:
            score = None
            error = "degenerate_clusters"
        else:
            try:
                score = float(silhouette_score(features, labels, metric="euclidean"))
                error = None
            except Exception as exc:
                score = None
                error = str(exc)
        row = {
            "k": int(k),
            "silhouette": score,
            "cluster_sizes": cluster_sizes,
            "error": error,
        }
        rows.append(row)
        if score is None:
            continue
        if best_payload is None or float(score) > float(best_payload["selected_silhouette"]):
            best_payload = {
                "selected_k": int(k),
                "selected_silhouette": float(score),
                "labels": labels,
                "centroids": np.asarray(model.cluster_centers_, dtype=np.float32),
                "rows": rows,
                "candidate_k_values": valid_k,
            }

    if best_payload is not None:
        best_payload["rows"] = rows
        return best_payload

    fallback_k = int(valid_k[0])
    fallback = KMeans(n_clusters=fallback_k, random_state=int(seed), n_init=10)
    fallback.fit(features)
    return {
        "selected_k": fallback_k,
        "selected_silhouette": None,
        "labels": np.asarray(fallback.labels_, dtype=np.int64),
        "centroids": np.asarray(fallback.cluster_centers_, dtype=np.float32),
        "rows": rows,
        "candidate_k_values": valid_k,
    }


def _pairwise_euclidean_distance_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {matrix.shape}.")
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float64)


def _cluster_centroids_from_labels(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centroids: List[np.ndarray] = []
    for cluster_id in range(int(n_clusters)):
        cluster_features = features[labels == cluster_id]
        if cluster_features.size == 0:
            centroids.append(np.zeros((features.shape[1],), dtype=np.float32))
        else:
            centroids.append(np.mean(cluster_features, axis=0).astype(np.float32))
    return np.stack(centroids, axis=0).astype(np.float32)


def _fit_adaptive_agglomerative(
    features: np.ndarray,
    *,
    scope: str,
    seed: int,
) -> Dict[str, Any]:
    del seed  # Agglomerative clustering is deterministic for fixed features.
    n_samples = int(features.shape[0])
    valid_k = _auto_k_candidates(n_samples, scope=scope)
    rows: List[Dict[str, Any]] = []
    if n_samples == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")
    if not valid_k:
        labels = np.zeros(n_samples, dtype=np.int64)
        centroid = np.mean(features, axis=0, keepdims=True)
        return {
            "selected_k": 1,
            "selected_silhouette": None,
            "labels": labels,
            "centroids": centroid.astype(np.float32),
            "rows": rows,
            "candidate_k_values": valid_k,
        }

    dist_matrix = _pairwise_euclidean_distance_matrix(features)
    best_payload: Optional[Dict[str, Any]] = None
    for k in valid_k:
        labels = core.fit_predict_precomputed_agglomerative(dist_matrix, int(k))
        labels = np.asarray(labels, dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {str(int(label)): int(count) for label, count in zip(unique, counts)}
        if np.unique(labels).size < 2:
            score = None
            error = "degenerate_clusters"
        else:
            try:
                score = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
                error = None
            except Exception as exc:
                score = None
                error = str(exc)
        row = {
            "k": int(k),
            "silhouette": score,
            "cluster_sizes": cluster_sizes,
            "error": error,
        }
        rows.append(row)
        if score is None:
            continue
        if best_payload is None or float(score) > float(best_payload["selected_silhouette"]):
            centroids = _cluster_centroids_from_labels(features, labels, int(k))
            best_payload = {
                "selected_k": int(k),
                "selected_silhouette": float(score),
                "labels": labels,
                "centroids": centroids,
                "rows": rows,
                "candidate_k_values": valid_k,
            }

    if best_payload is not None:
        best_payload["rows"] = rows
        return best_payload

    fallback_k = int(valid_k[0])
    labels = core.fit_predict_precomputed_agglomerative(dist_matrix, int(fallback_k))
    labels = np.asarray(labels, dtype=np.int64)
    centroids = _cluster_centroids_from_labels(features, labels, fallback_k)
    return {
        "selected_k": fallback_k,
        "selected_silhouette": None,
        "labels": labels,
        "centroids": centroids,
        "rows": rows,
        "candidate_k_values": valid_k,
    }


def _fit_adaptive_spectral(
    features: np.ndarray,
    *,
    scope: str,
    seed: int,
) -> Dict[str, Any]:
    n_samples = int(features.shape[0])
    valid_k = _auto_k_candidates(n_samples, scope=scope)
    rows: List[Dict[str, Any]] = []
    if n_samples == 0:
        raise ValueError("Cannot cluster an empty feature matrix.")
    if not valid_k:
        labels = np.zeros(n_samples, dtype=np.int64)
        centroid = np.mean(features, axis=0, keepdims=True)
        return {
            "selected_k": 1,
            "selected_silhouette": None,
            "labels": labels,
            "centroids": centroid.astype(np.float32),
            "rows": rows,
            "candidate_k_values": valid_k,
        }

    best_payload: Optional[Dict[str, Any]] = None
    for k in valid_k:
        n_neighbors = min(max(int(k) + 1, 5), n_samples - 1)
        if n_neighbors < 1:
            continue
        model = SpectralClustering(
            n_clusters=int(k),
            affinity="nearest_neighbors",
            n_neighbors=int(n_neighbors),
            assign_labels="kmeans",
            random_state=int(seed),
        )
        labels = np.asarray(model.fit_predict(features), dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {str(int(label)): int(count) for label, count in zip(unique, counts)}
        if np.unique(labels).size < 2:
            score = None
            error = "degenerate_clusters"
        else:
            try:
                score = float(silhouette_score(features, labels, metric="euclidean"))
                error = None
            except Exception as exc:
                score = None
                error = str(exc)
        row = {
            "k": int(k),
            "silhouette": score,
            "cluster_sizes": cluster_sizes,
            "n_neighbors": int(n_neighbors),
            "error": error,
        }
        rows.append(row)
        if score is None:
            continue
        if best_payload is None or float(score) > float(best_payload["selected_silhouette"]):
            best_payload = {
                "selected_k": int(k),
                "selected_silhouette": float(score),
                "labels": labels,
                "centroids": _cluster_centroids_from_labels(features, labels, int(k)),
                "rows": rows,
                "candidate_k_values": valid_k,
            }

    if best_payload is not None:
        best_payload["rows"] = rows
        return best_payload

    fallback_k = int(valid_k[0])
    fallback_neighbors = min(max(fallback_k + 1, 5), n_samples - 1)
    fallback = SpectralClustering(
        n_clusters=int(fallback_k),
        affinity="nearest_neighbors",
        n_neighbors=int(max(1, fallback_neighbors)),
        assign_labels="kmeans",
        random_state=int(seed),
    )
    labels = np.asarray(fallback.fit_predict(features), dtype=np.int64)
    return {
        "selected_k": fallback_k,
        "selected_silhouette": None,
        "labels": labels,
        "centroids": _cluster_centroids_from_labels(features, labels, fallback_k),
        "rows": rows,
        "candidate_k_values": valid_k,
    }


def _fit_adaptive_local_clustering(
    features: np.ndarray,
    *,
    algorithm: str,
    scope: str,
    seed: int,
) -> Dict[str, Any]:
    algorithm = str(algorithm).strip().lower()
    if algorithm == "kmeans":
        return _fit_adaptive_kmeans(features, scope=scope, seed=seed)
    if algorithm == "agglomerative":
        return _fit_adaptive_agglomerative(features, scope=scope, seed=seed)
    if algorithm == "spectral":
        return _fit_adaptive_spectral(features, scope=scope, seed=seed)
    raise ValueError(f"Unsupported local clustering algorithm: {algorithm}")


def _prototype_signature_state_keys(adapter_state: Dict[str, torch.Tensor], signature_type: str) -> List[str]:
    signature_type = str(signature_type).strip().lower()
    if signature_type == "lora_a":
        target_tokens = ("lora_A",)
    elif signature_type == "lora_b":
        target_tokens = ("lora_B",)
    elif signature_type == "lora_ab":
        target_tokens = ("lora_A", "lora_B")
    else:
        raise ValueError(f"Unsupported prototype signature type: {signature_type}")
    return sorted(key for key in adapter_state.keys() if any(token in key for token in target_tokens))


def _compute_layerwise_signature_distance_matrix(
    adapter_states: Sequence[Dict[str, torch.Tensor]],
    *,
    signature_type: str,
) -> np.ndarray:
    if not adapter_states:
        return np.zeros((0, 0), dtype=np.float64)

    keys = _prototype_signature_state_keys(adapter_states[0], signature_type)
    if not keys:
        raise ValueError(
            f"{signature_type} layer-wise clustering requires at least one matching LoRA parameter."
        )
    for idx, state in enumerate(adapter_states[1:], start=1):
        state_keys = _prototype_signature_state_keys(state, signature_type)
        if state_keys != keys:
            raise ValueError(
                f"{signature_type} parameter keys differ between clients 0 and {idx}; cannot compute layer-wise distance."
            )

    n_clients = len(adapter_states)
    dist = np.zeros((n_clients, n_clients), dtype=np.float64)
    eps = 1e-12
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            layer_distances: List[float] = []
            for key in keys:
                left = adapter_states[i][key].detach().cpu().float().flatten()
                right = adapter_states[j][key].detach().cpu().float().flatten()
                left_norm = float(torch.linalg.vector_norm(left).item())
                right_norm = float(torch.linalg.vector_norm(right).item())
                if left_norm <= eps and right_norm <= eps:
                    distance = 0.0
                elif left_norm <= eps or right_norm <= eps:
                    distance = 1.0
                else:
                    cosine = float(torch.dot(left, right).item() / (left_norm * right_norm))
                    distance = 1.0 - float(np.clip(cosine, -1.0, 1.0))
                layer_distances.append(distance)
            mean_distance = float(np.mean(layer_distances))
            dist[i, j] = mean_distance
            dist[j, i] = mean_distance
    return dist


def _encode_prompt_texts(tokenizer: Any, prompts: Sequence[str], max_length: int) -> Dict[str, torch.Tensor]:
    prompt_texts = [core.render_chat_text_cached(tokenizer, prompt, add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(
        prompt_texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


@torch.inference_mode()
def _compute_prompt_embeddings(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    batch_size: int,
    max_length: int,
    normalize_embeddings: bool,
    disable_progress: bool,
    desc: str,
) -> np.ndarray:
    device = next(model.parameters()).device
    outputs: List[np.ndarray] = []
    total = len(examples)
    with make_progress(total=total, desc=desc, disable=disable_progress) as bar:
        for start in range(0, total, max(1, int(batch_size))):
            chunk = examples[start : start + max(1, int(batch_size))]
            prompts = [ex.prompt for ex in chunk]
            encoded = _encode_prompt_texts(tokenizer, prompts, max_length=max_length)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            emb = core.compute_query_embedding(model, input_ids, attention_mask)
            outputs.append(emb.detach().cpu().float().numpy())
            bar.update(len(chunk))
    matrix = np.concatenate(outputs, axis=0).astype(np.float32)
    if normalize_embeddings:
        matrix = _l2_normalize(matrix)
    return matrix


def _adapter_gradient_param_info(model: Any, adapter_name: str = "default") -> List[Tuple[str, int]]:
    info: List[Tuple[str, int]] = []
    for name, param in model.named_parameters():
        if adapter_name in name and "lora_B" in name:
            info.append((name, int(param.numel())))
    if not info:
        raise RuntimeError(f"No LoRA-B parameters found for adapter '{adapter_name}'.")
    return info


def _clear_model_gradients(model: Any) -> None:
    for param in model.parameters():
        param.grad = None


def _collect_adapter_gradient_vector(
    model: Any,
    param_info: Sequence[Tuple[str, int]],
) -> np.ndarray:
    params_by_name = dict(model.named_parameters())
    slices: List[torch.Tensor] = []
    for name, numel in param_info:
        param = params_by_name.get(str(name))
        grad = None if param is None else param.grad
        if grad is None:
            slices.append(torch.zeros(int(numel), dtype=torch.float32))
        else:
            slices.append(grad.detach().cpu().float().reshape(-1))
    if not slices:
        return np.zeros(0, dtype=np.float32)
    return torch.cat(slices, dim=0).numpy().astype(np.float32)


def _pca_reduce_features(features: np.ndarray, *, output_dim: int) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {matrix.shape}.")
    n_samples, n_features = int(matrix.shape[0]), int(matrix.shape[1])
    out_dim = int(output_dim)
    if out_dim <= 0:
        raise ValueError("PCA output dimension must be positive.")
    if n_samples <= 0:
        return np.zeros((0, out_dim), dtype=np.float32)
    if n_samples < 2 or n_features <= 0:
        return np.zeros((n_samples, out_dim), dtype=np.float32)

    n_components = min(out_dim, n_samples, n_features)
    reduced = PCA(n_components=n_components).fit_transform(matrix).astype(np.float32)
    if n_components >= out_dim:
        return reduced.astype(np.float32)
    padded = np.zeros((n_samples, out_dim), dtype=np.float32)
    padded[:, :n_components] = reduced
    return padded


def _compute_gradient_pca_features(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    *,
    device: torch.device,
    max_length: int,
    pca_dim: int,
    disable_progress: bool,
    desc: str,
) -> np.ndarray:
    param_info = _adapter_gradient_param_info(model, "default")
    collate_fn = collate_fn_builder(tokenizer, max_length)
    vectors: List[np.ndarray] = []

    set_trainable_adapter(model, "default")
    model.eval()
    with make_progress(total=len(examples), desc=desc, disable=disable_progress) as bar:
        for example in examples:
            _clear_model_gradients(model)
            batch = move_batch_to_device(collate_fn([example]), device)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
            loss = core.compute_ce_loss(out.logits, batch.labels)
            loss.backward()
            vectors.append(_collect_adapter_gradient_vector(model, param_info))
            _clear_model_gradients(model)
            bar.update(1)
    model.eval()

    if not vectors:
        return np.zeros((0, int(pca_dim)), dtype=np.float32)
    full = np.stack(vectors, axis=0).astype(np.float32)
    return _pca_reduce_features(full, output_dim=int(pca_dim))


def _compute_local_discovery_features(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    feature_type: str,
    embedding_batch_size: int,
    max_length: int,
    normalize_embeddings: bool,
    gradient_pca_dim: int,
    device: torch.device,
    disable_progress: bool,
    desc_prefix: str,
) -> np.ndarray:
    feature_type = str(feature_type).strip().lower()
    if feature_type == "embedding":
        return _compute_prompt_embeddings(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            batch_size=embedding_batch_size,
            max_length=max_length,
            normalize_embeddings=normalize_embeddings,
            disable_progress=disable_progress,
            desc=f"embed[{desc_prefix}]",
        )
    if feature_type == "gradient_pca":
        return _compute_gradient_pca_features(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            max_length=max_length,
            pca_dim=gradient_pca_dim,
            disable_progress=disable_progress,
            desc=f"grad-pca[{desc_prefix}]",
        )
    raise ValueError(f"Unsupported local feature type: {feature_type}")


def _sample_level_purity(labels_true: Sequence[str], labels_pred: Sequence[int]) -> float:
    grouped: Dict[int, Counter[str]] = defaultdict(Counter)
    total = 0
    for task, cluster_id in zip(labels_true, labels_pred):
        grouped[int(cluster_id)][str(task)] += 1
        total += 1
    if total <= 0:
        return 0.0
    correct = sum(max(counter.values()) for counter in grouped.values() if counter)
    return float(correct / total)


def _majority_task(task_counts: Dict[str, int]) -> str:
    if not task_counts:
        return ""
    return max(sorted(task_counts.items()), key=lambda item: item[1])[0]


def _compute_local_clustering_metrics(
    sample_tasks: List[str],
    sample_labels: List[int],
) -> Dict[str, float]:
    """Compute purity, NMI, ARI for a single client's local clustering vs task labels."""
    if not sample_tasks or len(set(sample_labels)) <= 1:
        return {"purity": 0.0, "nmi": 0.0, "ari": 0.0}
    task_to_id = {}
    sample_task_ids = []
    for task in sample_tasks:
        if task not in task_to_id:
            task_to_id[task] = len(task_to_id)
        sample_task_ids.append(task_to_id[task])
    purity = _sample_level_purity(sample_tasks, sample_labels)
    nmi = float(normalized_mutual_info_score(sample_task_ids, sample_labels))
    ari = float(adjusted_rand_score(sample_task_ids, sample_labels))
    return {"purity": purity, "nmi": nmi, "ari": ari}


@dataclass
class PrototypeArtifacts:
    selected_global_k: int
    group_priors: List[float]
    initial_expert_states: Dict[int, Dict[str, torch.Tensor]]
    local_assignments_by_client: Dict[int, List[LocalClusterAssignment]]
    summary: Dict[str, Any]


def discover_prototypes(
    *,
    config: ExperimentConfig,
    data: PreparedData,
    model_bundle: ModelBundle,
    embedding_batch_size: int,
    normalize_embeddings: bool,
    discovery_warmup_steps: int,
    discovery_warmup_batch_size: int,
    local_feature_type: str = "embedding",
    local_cluster_algorithm: str = "kmeans",
    prototype_signature_type: str = "lora_b",
    gradient_pca_dim: int = 64,
    discovery_warmup_mode: str = "steps",
    discovery_warmup_epochs: int = 1,
) -> PrototypeArtifacts:
    local_feature_type = str(local_feature_type).strip().lower()
    local_cluster_algorithm = str(local_cluster_algorithm).strip().lower()
    prototype_signature_type = str(prototype_signature_type).strip().lower()
    if local_feature_type not in {"embedding", "gradient_pca"}:
        raise ValueError(f"Unsupported local_feature_type: {local_feature_type}")
    if local_cluster_algorithm not in {"kmeans", "agglomerative", "spectral"}:
        raise ValueError(f"Unsupported local_cluster_algorithm: {local_cluster_algorithm}")
    if prototype_signature_type not in {"lora_a", "lora_b", "lora_ab"}:
        raise ValueError(f"Unsupported prototype_signature_type: {prototype_signature_type}")
    if int(gradient_pca_dim) <= 0:
        raise ValueError("gradient_pca_dim must be positive.")
    discovery_warmup_mode = str(discovery_warmup_mode).strip().lower()
    if discovery_warmup_mode not in {"steps", "epochs"}:
        raise ValueError(f"Unsupported discovery_warmup_mode: {discovery_warmup_mode}")
    if int(discovery_warmup_epochs) <= 0:
        raise ValueError("discovery_warmup_epochs must be positive.")

    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    model.eval()
    model.set_adapter("default")
    init_state = get_adapter_state(model, "default")
    device = next(model.parameters()).device
    disable_progress = not bool(config.logging.show_progress)

    local_selection_rows: List[Dict[str, Any]] = []
    local_cluster_rows: List[Dict[str, Any]] = []
    global_input_client_ids: List[int] = []
    global_input_local_ids: List[int] = []
    global_input_task_counts: List[Dict[str, int]] = []
    global_input_adapter_states: List[Dict[str, torch.Tensor]] = []
    warmup_rows: List[Dict[str, Any]] = []
    assignments_by_client: Dict[int, List[LocalClusterAssignment]] = defaultdict(list)
    local_cluster_sample_indices: Dict[Tuple[int, int], List[int]] = {}
    sample_tasks: List[str] = []
    sample_to_meta: List[Tuple[int, int]] = []

    with make_progress(total=len(data.clients), desc="prototype-local", disable=disable_progress) as client_bar:
        for client in data.clients:
            core.load_adapter_state(model, init_state)
            model.set_adapter("default")
            model.eval()
            features = _compute_local_discovery_features(
                model=model,
                tokenizer=tokenizer,
                examples=client.train_examples,
                feature_type=local_feature_type,
                embedding_batch_size=embedding_batch_size,
                max_length=config.train.max_length,
                normalize_embeddings=normalize_embeddings,
                gradient_pca_dim=int(gradient_pca_dim),
                device=device,
                disable_progress=disable_progress,
                desc_prefix=f"c{client.client_id}",
            )
            core.load_adapter_state(model, init_state)
            model.set_adapter("default")
            model.eval()
            clustering = _fit_adaptive_local_clustering(
                features,
                algorithm=local_cluster_algorithm,
                scope="local",
                seed=config.seed + client.client_id,
            )
            labels = np.asarray(clustering["labels"], dtype=np.int64)

            local_selection_rows.append(
                {
                    "client_id": int(client.client_id),
                    "selected_k": int(clustering["selected_k"]),
                    "selected_silhouette": clustering["selected_silhouette"],
                    "candidate_k_values": [int(k) for k in clustering["candidate_k_values"]],
                    "n_train": int(client.n_train),
                    "dominant_task": str(client.dominant_task),
                    "feature_type": local_feature_type,
                    "cluster_algorithm": local_cluster_algorithm,
                }
            )

            cluster_to_task_counter: Dict[int, Counter[str]] = defaultdict(Counter)
            cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
            for sample_idx, (ex, cluster_id) in enumerate(zip(client.train_examples, labels.tolist())):
                cluster_to_task_counter[int(cluster_id)][str(ex.task)] += 1
                cluster_to_indices[int(cluster_id)].append(int(sample_idx))
                sample_tasks.append(str(ex.task))
                sample_to_meta.append((int(client.client_id), int(cluster_id)))

            for local_cluster_id in range(int(clustering["selected_k"])):
                task_counts = {
                    task: int(count)
                    for task, count in sorted(cluster_to_task_counter[int(local_cluster_id)].items())
                }
                cluster_examples = [client.train_examples[idx] for idx in cluster_to_indices[int(local_cluster_id)]]
                core.load_adapter_state(model, init_state)
                model.set_adapter("default")
                warmup_stats = _run_discovery_warmup_steps(
                    model=model,
                    tokenizer=tokenizer,
                    examples=cluster_examples,
                    config=config,
                    device=device,
                    shuffle_seed=int(config.seed + 500_000 + client.client_id * 1_000 + local_cluster_id),
                    local_steps_override=int(discovery_warmup_steps),
                    batch_size_override=int(discovery_warmup_batch_size),
                    warmup_mode=discovery_warmup_mode,
                    warmup_epochs=int(discovery_warmup_epochs),
                )
                if prototype_signature_type == "lora_a":
                    signature_vector = extract_lora_a_vector(model, "default")
                elif prototype_signature_type == "lora_ab":
                    signature_vector = extract_lora_ab_vector(model, "default")
                else:
                    signature_vector = extract_lora_b_vector(model, "default")
                signature_np = signature_vector.detach().cpu().float().numpy().astype(np.float32)
                global_input_adapter_states.append(clone_state_dict_cpu(get_adapter_state(model, "default")))
                global_input_client_ids.append(int(client.client_id))
                global_input_local_ids.append(int(local_cluster_id))
                global_input_task_counts.append(task_counts)
                local_cluster_sample_indices[(int(client.client_id), int(local_cluster_id))] = list(
                    cluster_to_indices[int(local_cluster_id)]
                )
                warmup_rows.append(
                    {
                        "client_id": int(client.client_id),
                        "local_cluster_id": int(local_cluster_id),
                        "warmup_steps": int(discovery_warmup_steps),
                        "warmup_batch_size": int(discovery_warmup_batch_size),
                        "warmup_loss": float(warmup_stats["loss"]),
                        "signature_type": prototype_signature_type,
                        "signature_dim": int(signature_np.shape[0]),
                        "signature_norm": float(np.linalg.norm(signature_np)),
                    }
                )
                local_cluster_rows.append(
                    {
                        "client_id": int(client.client_id),
                        "local_cluster_id": int(local_cluster_id),
                        "cluster_size": int(len(cluster_to_indices[int(local_cluster_id)])),
                        "majority_task": _majority_task(task_counts),
                        "task_counts": task_counts,
                        "warmup_loss": float(warmup_stats["loss"]),
                    }
                )
            # Compute per-client local clustering quality vs ground-truth task labels.
            client_sample_tasks = [str(ex.task) for ex in client.train_examples]
            client_sample_labels = [int(c) for c in labels.tolist()]
            local_metrics = _compute_local_clustering_metrics(client_sample_tasks, client_sample_labels)
            local_selection_rows[-1]["local_purity"] = local_metrics["purity"]
            local_selection_rows[-1]["local_nmi"] = local_metrics["nmi"]
            local_selection_rows[-1]["local_ari"] = local_metrics["ari"]
            client_bar.update(1)

    dist_matrix = _compute_layerwise_signature_distance_matrix(
        global_input_adapter_states,
        signature_type=prototype_signature_type,
    )
    candidate_k_values = _auto_k_candidates(len(global_input_adapter_states), scope="global")
    global_search_rows: List[Dict[str, Any]] = []
    best_payload: Optional[Dict[str, Any]] = None
    for k in candidate_k_values:
        labels = core.fit_predict_precomputed_agglomerative(dist_matrix, int(k))
        labels = np.asarray(labels, dtype=np.int64)
        try:
            score = float(silhouette_score(dist_matrix, labels, metric="precomputed"))
            error = None
        except Exception as exc:
            score = None
            error = str(exc)
        global_search_rows.append(
            {
                "k": int(k),
                "silhouette": score,
                "error": error,
                "cluster_sizes": {
                    str(int(cluster_id)): int(count)
                    for cluster_id, count in zip(*np.unique(labels, return_counts=True))
                },
            }
        )
        if score is None:
            continue
        if best_payload is None or float(score) > float(best_payload["selected_silhouette"]):
            best_payload = {
                "selected_k": int(k),
                "selected_silhouette": float(score),
                "labels": labels,
            }

    if best_payload is None:
        fallback_k = 1 if len(global_input_adapter_states) <= 1 else min(2, len(global_input_adapter_states))
        labels = core.fit_predict_precomputed_agglomerative(dist_matrix, int(fallback_k)) if fallback_k > 1 else np.zeros(
            len(global_input_adapter_states), dtype=np.int64
        )
        best_payload = {
            "selected_k": int(fallback_k),
            "selected_silhouette": None,
            "labels": np.asarray(labels, dtype=np.int64),
        }

    global_labels = np.asarray(best_payload["labels"], dtype=np.int64)

    local_to_global = {
        (int(client_id), int(local_cluster_id)): int(global_cluster_id)
        for client_id, local_cluster_id, global_cluster_id in zip(
            global_input_client_ids,
            global_input_local_ids,
            global_labels.tolist(),
        )
    }
    sample_global_labels = [
        int(local_to_global[(int(client_id), int(local_cluster_id))])
        for client_id, local_cluster_id in sample_to_meta
    ]

    for row in local_cluster_rows:
        client_id = int(row["client_id"])
        local_cluster_id = int(row["local_cluster_id"])
        assigned_expert = int(local_to_global[(client_id, local_cluster_id)])
        sample_indices = list(local_cluster_sample_indices[(client_id, local_cluster_id)])
        assignments_by_client[client_id].append(
            LocalClusterAssignment(
                client_id=client_id,
                local_cluster_id=local_cluster_id,
                sample_indices=sample_indices,
                assigned_expert=assigned_expert,
            )
        )
        row["assigned_expert"] = assigned_expert

    task_to_id = {task: idx for idx, task in enumerate(config.data.tasks)}
    sample_task_ids = [int(task_to_id[task]) for task in sample_tasks]
    selected_global_k = int(best_payload["selected_k"])
    initial_expert_states = _average_adapter_states_by_cluster(
        client_adapter_states=global_input_adapter_states,
        cluster_ids=global_labels.tolist(),
        k_experts=selected_global_k,
    )
    group_counts = np.bincount(sample_global_labels, minlength=selected_global_k).astype(np.float64)
    denom = float(group_counts.sum())
    group_priors = (
        (group_counts / denom).astype(np.float64).tolist()
        if denom > 0
        else [1.0 / max(1, selected_global_k) for _ in range(max(1, selected_global_k))]
    )
    metrics = {
        "purity": _sample_level_purity(sample_tasks, sample_global_labels),
        "nmi": float(normalized_mutual_info_score(sample_task_ids, sample_global_labels)),
        "ari": float(adjusted_rand_score(sample_task_ids, sample_global_labels)),
    }

    client_purities = [float(row["local_purity"]) for row in local_selection_rows if "local_purity" in row]
    client_nmis = [float(row["local_nmi"]) for row in local_selection_rows if "local_nmi" in row]
    client_aris = [float(row["local_ari"]) for row in local_selection_rows if "local_ari" in row]

    def _agg_stats(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.array(vals, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    local_metrics_agg = {
        "purity": _agg_stats(client_purities),
        "nmi": _agg_stats(client_nmis),
        "ari": _agg_stats(client_aris),
    }

    summary = {
        "local": {
            "selection_rows": local_selection_rows,
            "cluster_rows": local_cluster_rows,
            "warmup_rows": warmup_rows,
            "feature_type": local_feature_type,
            "cluster_algorithm": local_cluster_algorithm,
            "gradient_pca_dim": int(gradient_pca_dim),
            "prototype_signature_type": prototype_signature_type,
            "metrics_agg": local_metrics_agg,
        },
        "global": {
            "selected_k": int(best_payload["selected_k"]),
            "selected_silhouette": best_payload["selected_silhouette"],
            "candidate_k_values": [int(k) for k in candidate_k_values],
            "search_rows": global_search_rows,
            "alignment_space": f"mean_layerwise_{prototype_signature_type}_cosine_distance",
            "distance_metric": f"mean_layerwise_{prototype_signature_type}_cosine_distance",
            "prototype_signature_type": prototype_signature_type,
            "routing_priors": [float(v) for v in group_priors],
            "inference_routing": "population_prior_mixture",
            "server_receives_embedding_centroids": False,
        },
        "metrics": metrics,
    }
    return PrototypeArtifacts(
        selected_global_k=selected_global_k,
        group_priors=[float(v) for v in group_priors],
        initial_expert_states={int(k): clone_state_dict_cpu(v) for k, v in initial_expert_states.items()},
        local_assignments_by_client=dict(assignments_by_client),
        summary=summary,
    )



def discover_oracle_task_prototypes(
    *,
    config: ExperimentConfig,
    data: PreparedData,
    model_bundle: ModelBundle,
    discovery_warmup_steps: int,
    discovery_warmup_batch_size: int,
    discovery_warmup_mode: str = "steps",
    discovery_warmup_epochs: int = 1,
) -> PrototypeArtifacts:
    """Build a FedWeave assignment with perfect knowledge of each sample's task label."""
    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    model.eval()
    model.set_adapter("default")
    init_state = get_adapter_state(model, "default")
    device = next(model.parameters()).device
    disable_progress = not bool(config.logging.show_progress)

    task_to_expert = {str(task): idx for idx, task in enumerate(config.data.tasks)}
    k_experts = int(len(task_to_expert))
    if k_experts <= 0:
        raise ValueError("Oracle task routing requires at least one task.")

    local_selection_rows: List[Dict[str, Any]] = []
    local_cluster_rows: List[Dict[str, Any]] = []
    warmup_rows: List[Dict[str, Any]] = []
    assignments_by_client: Dict[int, List[LocalClusterAssignment]] = defaultdict(list)
    task_adapter_states: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
    group_counts = np.zeros(k_experts, dtype=np.float64)
    sample_tasks: List[str] = []
    sample_global_labels: List[int] = []

    with make_progress(total=len(data.clients), desc="oracle-task", disable=disable_progress) as client_bar:
        for client in data.clients:
            task_to_indices: Dict[str, List[int]] = {str(task): [] for task in config.data.tasks}
            for sample_idx, ex in enumerate(client.train_examples):
                task = str(ex.task)
                if task not in task_to_expert:
                    raise ValueError(f"Sample task '{task}' is not present in config.data.tasks.")
                task_to_indices[task].append(int(sample_idx))
                expert_id = int(task_to_expert[task])
                group_counts[expert_id] += 1.0
                sample_tasks.append(task)
                sample_global_labels.append(expert_id)

            present_tasks = [str(task) for task in config.data.tasks if task_to_indices[str(task)]]
            local_selection_rows.append(
                {
                    "client_id": int(client.client_id),
                    "selected_k": int(len(present_tasks)),
                    "selected_silhouette": None,
                    "candidate_k_values": [],
                    "n_train": int(client.n_train),
                    "dominant_task": str(client.dominant_task),
                    "assignment_source": "task_label_oracle",
                    "local_purity": 1.0,
                    "local_nmi": 1.0,
                    "local_ari": 1.0,
                }
            )

            for task in present_tasks:
                expert_id = int(task_to_expert[task])
                sample_indices = list(task_to_indices[task])
                cluster_examples = [client.train_examples[idx] for idx in sample_indices]
                core.load_adapter_state(model, init_state)
                model.set_adapter("default")
                warmup_stats = _run_discovery_warmup_steps(
                    model=model,
                    tokenizer=tokenizer,
                    examples=cluster_examples,
                    config=config,
                    device=device,
                    shuffle_seed=int(config.seed + 700_000 + client.client_id * 1_000 + expert_id),
                    local_steps_override=int(discovery_warmup_steps),
                    batch_size_override=int(discovery_warmup_batch_size),
                    warmup_mode=discovery_warmup_mode,
                    warmup_epochs=int(discovery_warmup_epochs),
                )
                task_adapter_states[expert_id].append(clone_state_dict_cpu(get_adapter_state(model, "default")))
                assignments_by_client[int(client.client_id)].append(
                    LocalClusterAssignment(
                        client_id=int(client.client_id),
                        local_cluster_id=expert_id,
                        sample_indices=sample_indices,
                        assigned_expert=expert_id,
                    )
                )
                warmup_rows.append(
                    {
                        "client_id": int(client.client_id),
                        "local_cluster_id": int(expert_id),
                        "task": task,
                        "warmup_steps": int(discovery_warmup_steps),
                        "warmup_batch_size": int(discovery_warmup_batch_size),
                        "warmup_loss": float(warmup_stats["loss"]),
                    }
                )
                local_cluster_rows.append(
                    {
                        "client_id": int(client.client_id),
                        "local_cluster_id": int(expert_id),
                        "cluster_size": int(len(sample_indices)),
                        "majority_task": task,
                        "task_counts": {task: int(len(sample_indices))},
                        "assigned_expert": int(expert_id),
                        "assignment_source": "task_label_oracle",
                        "warmup_loss": float(warmup_stats["loss"]),
                    }
                )
            client_bar.update(1)

    initial_expert_states: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id in range(k_experts):
        member_states = task_adapter_states.get(int(expert_id), [])
        if not member_states:
            task = list(task_to_expert.keys())[int(expert_id)]
            raise ValueError(f"Cannot initialize oracle expert {expert_id} for task '{task}': no training examples.")
        adapter_name = adapter_name_for_cluster(expert_id)
        averaged_default: Dict[str, torch.Tensor] = {}
        for key in member_states[0].keys():
            total = torch.zeros_like(member_states[0][key], dtype=torch.float32)
            for state in member_states:
                total.add_(state[key].detach().cpu().float())
            averaged_default[key] = total / float(len(member_states))
        initial_expert_states[int(expert_id)] = core.remap_adapter_state(
            averaged_default,
            source_adapter_name="default",
            target_adapter_name=adapter_name,
        )

    denom = float(group_counts.sum())
    group_priors = (
        (group_counts / denom).astype(np.float64).tolist()
        if denom > 0
        else [1.0 / max(1, k_experts) for _ in range(k_experts)]
    )
    sample_task_ids = [int(task_to_expert[task]) for task in sample_tasks]
    metrics = {
        "purity": _sample_level_purity(sample_tasks, sample_global_labels),
        "nmi": float(normalized_mutual_info_score(sample_task_ids, sample_global_labels)) if sample_tasks else 0.0,
        "ari": float(adjusted_rand_score(sample_task_ids, sample_global_labels)) if sample_tasks else 0.0,
    }
    summary = {
        "local": {
            "selection_rows": local_selection_rows,
            "cluster_rows": local_cluster_rows,
            "warmup_rows": warmup_rows,
            "metrics_agg": {
                "purity": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                "nmi": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
                "ari": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0},
            },
        },
        "global": {
            "selected_k": int(k_experts),
            "selected_silhouette": None,
            "candidate_k_values": [],
            "search_rows": [],
            "alignment_space": "task_label_oracle",
            "routing_priors": [float(v) for v in group_priors],
            "inference_routing": "learned_absolute_router",
            "server_receives_embedding_centroids": False,
            "task_to_expert": {task: int(expert_id) for task, expert_id in task_to_expert.items()},
        },
        "metrics": metrics,
        "oracle_task_routing": True,
    }
    return PrototypeArtifacts(
        selected_global_k=k_experts,
        group_priors=[float(v) for v in group_priors],
        initial_expert_states={int(k): clone_state_dict_cpu(v) for k, v in initial_expert_states.items()},
        local_assignments_by_client=dict(assignments_by_client),
        summary=summary,
    )


def _build_infinite_train_iterator(
    tokenizer: Any,
    examples: Sequence[Example],
    batch_size: int,
    max_length: int,
    shuffle_seed: int,
):
    collate_fn = collate_fn_builder(tokenizer, max_length)
    generator = make_dataloader_generator(shuffle_seed)
    use_cuda = torch.cuda.is_available()
    dataloader = DataLoader(
        list(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=True if use_cuda else False,
        prefetch_factor=2 if use_cuda else None,
    )
    while True:
        for batch in dataloader:
            yield batch


def _clip_gradients_if_needed(trainable_params: Iterable[torch.nn.Parameter], grad_clip: float) -> None:
    params = [param for param in trainable_params if param.grad is not None]
    if not params or grad_clip <= 0:
        return
    torch.nn.utils.clip_grad_norm_(params, max_norm=float(grad_clip))


def _build_epoch_dataloader(
    tokenizer: Any,
    examples: Sequence[Example],
    batch_size: int,
    max_length: int,
    shuffle_seed: int,
) -> DataLoader:
    collate_fn = collate_fn_builder(tokenizer, max_length)
    generator = make_dataloader_generator(shuffle_seed)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        list(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0,
        persistent_workers=use_cuda,
        prefetch_factor=2 if use_cuda else None,
    )


def _run_discovery_warmup_steps(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    config: ExperimentConfig,
    device: torch.device,
    shuffle_seed: int,
    local_steps_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    adapter_name: str = "default",
    warmup_mode: str = "steps",
    warmup_epochs: int = 1,
) -> Dict[str, float]:
    """Warm up one local adapter before extracting its prototype signature."""
    batch_size = int(config.train.batch_size) if batch_size_override is None else int(batch_size_override)
    if not examples:
        return {"loss": 0.0}

    set_trainable_adapter(model, adapter_name)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters found for adapter '{adapter_name}'.")

    optimizer = build_optimizer(trainable_params, config.train.lr, config.train.wd)
    model.train()
    grad_accum = max(1, int(config.train.grad_accum))

    if warmup_mode == "epochs":
        n_batches = max(1, int(math.ceil(len(examples) / batch_size)))
        steps_per_epoch = max(1, int(math.ceil(n_batches / grad_accum)))
        scheduler = build_local_scheduler(
            optimizer=optimizer,
            num_optimizer_steps=int(warmup_epochs) * steps_per_epoch,
            schedule=config.train.local_lr_schedule,
        )
        losses: List[float] = []
        for epoch in range(int(warmup_epochs)):
            loader = _build_epoch_dataloader(
                tokenizer=tokenizer,
                examples=examples,
                batch_size=batch_size,
                max_length=config.train.max_length,
                shuffle_seed=int(shuffle_seed) + epoch,
            )
            micro_buffer: List[Any] = []
            for batch in loader:
                micro_buffer.append(move_batch_to_device(batch, device))
                if len(micro_buffer) < grad_accum:
                    continue
                optimizer.zero_grad(set_to_none=True)
                step_losses: List[float] = []
                for micro_batch in micro_buffer:
                    out = model(
                        input_ids=micro_batch.input_ids,
                        attention_mask=micro_batch.attention_mask,
                        use_cache=False,
                    )
                    raw_loss = core.compute_ce_loss(out.logits, micro_batch.labels)
                    (raw_loss / float(len(micro_buffer))).backward()
                    step_losses.append(float(raw_loss.detach().item()))
                _clip_gradients_if_needed(trainable_params, config.train.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(np.mean(step_losses)))
                micro_buffer = []
            if micro_buffer:
                optimizer.zero_grad(set_to_none=True)
                step_losses = []
                for micro_batch in micro_buffer:
                    out = model(
                        input_ids=micro_batch.input_ids,
                        attention_mask=micro_batch.attention_mask,
                        use_cache=False,
                    )
                    raw_loss = core.compute_ce_loss(out.logits, micro_batch.labels)
                    (raw_loss / float(len(micro_buffer))).backward()
                    step_losses.append(float(raw_loss.detach().item()))
                _clip_gradients_if_needed(trainable_params, config.train.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                losses.append(float(np.mean(step_losses)))
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    local_steps = int(config.train.local_steps) if local_steps_override is None else int(local_steps_override)
    if local_steps <= 0:
        return {"loss": 0.0}
    scheduler = build_local_scheduler(
        optimizer=optimizer,
        num_optimizer_steps=local_steps,
        schedule=config.train.local_lr_schedule,
    )
    iterator = _build_infinite_train_iterator(
        tokenizer=tokenizer,
        examples=list(examples),
        batch_size=batch_size,
        max_length=config.train.max_length,
        shuffle_seed=shuffle_seed,
    )
    losses: List[float] = []
    for _ in range(local_steps):
        optimizer.zero_grad(set_to_none=True)
        step_losses: List[float] = []
        for _ in range(grad_accum):
            batch = move_batch_to_device(next(iterator), device)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
            raw_loss = core.compute_ce_loss(out.logits, batch.labels)
            (raw_loss / float(grad_accum)).backward()
            step_losses.append(float(raw_loss.detach().item()))
        _clip_gradients_if_needed(trainable_params, config.train.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(np.mean(step_losses)))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


def _state_dict_difference(updated_state: Dict[str, torch.Tensor], base_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: updated_state[key].detach().cpu().float() - base_state[key].detach().cpu().float()
        for key in base_state.keys()
    }


def _router_out_dim_from_state(
    router_state_dicts: Optional[Dict[int, Dict[str, torch.Tensor]]],
    router_id: int,
    fallback: int,
) -> int:
    if router_state_dicts is None:
        return int(fallback)
    state = router_state_dicts.get(int(router_id))
    if not isinstance(state, dict):
        return int(fallback)
    for key, value in state.items():
        if str(key).endswith("mlp.3.weight") and getattr(value, "ndim", 0) >= 2:
            return int(value.shape[0])
    return int(fallback)


def _run_local_cluster_steps(
    *,
    model: Any,
    tokenizer: Any,
    routers: Dict[int, Any],
    examples: Sequence[Example],
    assigned_expert: int,
    k_experts: int,
    config: ExperimentConfig,
    device: torch.device,
    expert_id_to_adapter: Dict[int, str],
    shuffle_seed: int,
    local_steps_override: Optional[int] = None,
    absolute_routing: bool = False,
    disable_lora_for_query: bool = False,
    train_lora_adapters: bool = True,
) -> Dict[str, float]:
    local_steps = int(config.train.local_steps) if local_steps_override is None else int(local_steps_override)
    if len(examples) == 0 or local_steps <= 0:
        return {
            "loss": 0.0,
            "task_loss": 0.0,
            "route_ce_loss": 0.0,
            "route_ce_weighted_loss": 0.0,
            "mix_entropy": 0.0,
            "mix_top1": 0.0,
            "router_keff": 0.0,
            "mix_m_used": 0.0,
        }

    if int(k_experts) == 1:
        stats = _run_discovery_warmup_steps(
            model=model, tokenizer=tokenizer, examples=examples,
            config=config, device=device, shuffle_seed=shuffle_seed,
            local_steps_override=local_steps_override,
            adapter_name=adapter_name_for_cluster(int(assigned_expert)),
        )
        mean_loss = float(stats["loss"])
        return {
            "loss": mean_loss, "task_loss": mean_loss,
            "route_ce_loss": 0.0, "route_ce_weighted_loss": 0.0,
            "mix_entropy": 0.0, "mix_top1": 1.0,
            "router_keff": 1.0, "mix_m_used": 1.0,
        }

    single_router = len(routers) == 1
    router = routers[0] if single_router else routers[int(assigned_expert)]
    assigned_adapter = adapter_name_for_cluster(int(assigned_expert))
    if train_lora_adapters:
        set_trainable_adapter(model, assigned_adapter)
    else:
        core.freeze_all_params(model)
        model.set_adapter(assigned_adapter)
    set_trainable_router(routers, 0 if single_router else int(assigned_expert))
    model_trainable_params = [param for param in model.parameters() if param.requires_grad]
    router_trainable_params = [param for param in router.parameters() if param.requires_grad]
    trainable_params = model_trainable_params + router_trainable_params
    optimizer = build_split_lr_optimizer(
        model_params=model_trainable_params,
        router_params=router_trainable_params,
        model_lr=config.train.lr,
        router_lr=config.train.router_lr,
        wd=config.train.wd,
    )
    scheduler = build_local_scheduler(
        optimizer=optimizer,
        num_optimizer_steps=local_steps,
        schedule=config.train.local_lr_schedule,
    )
    iterator = _build_infinite_train_iterator(
        tokenizer=tokenizer,
        examples=list(examples),
        batch_size=config.train.batch_size,
        max_length=config.train.max_length,
        shuffle_seed=shuffle_seed,
    )

    model.train()
    router.train()
    losses: List[float] = []
    task_losses: List[float] = []
    route_ce_losses: List[float] = []
    route_ce_weighted_losses: List[float] = []
    entropies: List[float] = []
    top1s: List[float] = []
    keffs: List[float] = []
    m_useds: List[float] = []

    for _ in range(max(0, local_steps)):
        optimizer.zero_grad(set_to_none=True)
        step_losses: List[float] = []
        step_task_losses: List[float] = []
        step_route_ce_losses: List[float] = []
        step_route_ce_weighted_losses: List[float] = []
        step_entropies: List[float] = []
        step_top1s: List[float] = []
        step_keffs: List[float] = []
        step_m_useds: List[float] = []
        for _micro in range(max(1, int(config.train.grad_accum))):
            batch = move_batch_to_device(next(iterator), device)
            log_probs_mix, stats, route_ce_loss = compute_mixture_logits(
                model=model,
                router=router,
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                query_input_ids=batch.query_input_ids,
                query_attention_mask=batch.query_attention_mask,
                k_experts=k_experts,
                assigned_expert=int(assigned_expert),
                m_select=config.router.m_select,
                m_tau=config.router.m_tau,
                expert_id_to_adapter=expert_id_to_adapter,
                mix_in_logprob=True,
                absolute_routing=absolute_routing,
                disable_lora_for_query=disable_lora_for_query,
            )
            task_loss = compute_nll_loss(log_probs_mix, batch.labels)
            weighted_route_ce_loss = config.router.route_ce_weight * route_ce_loss
            total_loss = task_loss + weighted_route_ce_loss
            (total_loss / max(1, int(config.train.grad_accum))).backward()
            step_losses.append(float(total_loss.detach().item()))
            step_task_losses.append(float(task_loss.detach().item()))
            step_route_ce_losses.append(float(route_ce_loss.detach().item()))
            step_route_ce_weighted_losses.append(float(weighted_route_ce_loss.detach().item()))
            step_entropies.append(float(stats["mix_entropy"]))
            step_top1s.append(float(stats["mix_top1"]))
            step_keffs.append(float(stats["router_keff"]))
            step_m_useds.append(float(stats["mix_m_used"]))
        _clip_gradients_if_needed(trainable_params, config.train.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(np.mean(step_losses)) if step_losses else 0.0)
        task_losses.append(float(np.mean(step_task_losses)) if step_task_losses else 0.0)
        route_ce_losses.append(float(np.mean(step_route_ce_losses)) if step_route_ce_losses else 0.0)
        route_ce_weighted_losses.append(
            float(np.mean(step_route_ce_weighted_losses)) if step_route_ce_weighted_losses else 0.0
        )
        entropies.append(float(np.mean(step_entropies)) if step_entropies else 0.0)
        top1s.append(float(np.mean(step_top1s)) if step_top1s else 0.0)
        keffs.append(float(np.mean(step_keffs)) if step_keffs else 0.0)
        m_useds.append(float(np.mean(step_m_useds)) if step_m_useds else 0.0)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
        "route_ce_loss": float(np.mean(route_ce_losses)) if route_ce_losses else 0.0,
        "route_ce_weighted_loss": float(np.mean(route_ce_weighted_losses)) if route_ce_weighted_losses else 0.0,
        "mix_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mix_top1": float(np.mean(top1s)) if top1s else 0.0,
        "router_keff": float(np.mean(keffs)) if keffs else 0.0,
        "mix_m_used": float(np.mean(m_useds)) if m_useds else 0.0,
    }


def _set_trainable_all_lora_adapters(model: Any) -> None:
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def _run_client_interleaved_bucket_steps(
    *,
    model: Any,
    tokenizer: Any,
    routers: Dict[int, Any],
    client_examples: Sequence[Example],
    assignments: Sequence[LocalClusterAssignment],
    bucket_step_budgets: Sequence[int],
    base_expert_states: Dict[int, Dict[str, torch.Tensor]],
    base_router_state: Optional[Dict[str, torch.Tensor]],
    k_experts: int,
    config: ExperimentConfig,
    device: torch.device,
    expert_id_to_adapter: Dict[int, str],
    shuffle_seed: int,
    absolute_routing: bool = False,
    disable_lora_for_query: bool = False,
) -> Dict[str, Any]:
    active_buckets: List[Dict[str, Any]] = []
    expert_sample_counts: Dict[int, int] = defaultdict(int)
    for bucket_idx, (assignment, step_budget) in enumerate(zip(assignments, bucket_step_budgets)):
        step_budget = int(step_budget)
        if step_budget <= 0:
            continue
        examples = [client_examples[idx] for idx in assignment.sample_indices]
        if not examples:
            continue
        assigned_expert = int(assignment.assigned_expert)
        expert_sample_counts[assigned_expert] += int(len(examples))
        active_buckets.append(
            {
                "bucket_idx": int(bucket_idx),
                "local_cluster_id": int(assignment.local_cluster_id),
                "assigned_expert": assigned_expert,
                "examples": examples,
                "step_budget": step_budget,
                "sample_count": int(len(examples)),
            }
        )

    schedule = _build_interleaved_bucket_schedule([int(bucket["step_budget"]) for bucket in active_buckets])
    if not active_buckets or not schedule:
        return {
            "loss": 0.0,
            "task_loss": 0.0,
            "route_ce_loss": 0.0,
            "route_ce_weighted_loss": 0.0,
            "mix_entropy": 0.0,
            "mix_top1": 0.0,
            "router_keff": 0.0,
            "mix_m_used": 0.0,
            "expert_sample_counts": dict(expert_sample_counts),
            "total_examples": int(sum(expert_sample_counts.values())),
            "num_bucket_uploads": 0,
            "num_router_uploads": 0,
        }

    load_all_expert_states(model, base_expert_states, k_experts)
    if routers and base_router_state is not None:
        routers[0].load_state_dict({key: value.to(device) for key, value in base_router_state.items()})

    _set_trainable_all_lora_adapters(model)
    router_trainable_params: List[torch.nn.Parameter] = []
    if routers and int(k_experts) > 1:
        set_trainable_router(routers, 0)
        router_trainable_params = [param for param in routers[0].parameters() if param.requires_grad]
    else:
        set_trainable_router(routers, None)
    model_trainable_params = [param for param in model.parameters() if param.requires_grad]
    trainable_params = model_trainable_params + router_trainable_params
    if int(k_experts) == 1 or not router_trainable_params:
        optimizer = build_optimizer(model_trainable_params, config.train.lr, config.train.wd)
    else:
        optimizer = build_split_lr_optimizer(
            model_params=model_trainable_params,
            router_params=router_trainable_params,
            model_lr=config.train.lr,
            router_lr=config.train.router_lr,
            wd=config.train.wd,
        )
    scheduler = build_local_scheduler(
        optimizer=optimizer,
        num_optimizer_steps=len(schedule),
        schedule=config.train.local_lr_schedule,
    )

    iterators = [
        _build_infinite_train_iterator(
            tokenizer=tokenizer,
            examples=bucket["examples"],
            batch_size=config.train.batch_size,
            max_length=config.train.max_length,
            shuffle_seed=int(shuffle_seed + 97 * bucket["bucket_idx"] + bucket["local_cluster_id"]),
        )
        for bucket in active_buckets
    ]

    model.train()
    for router in routers.values():
        router.train()

    losses: List[float] = []
    task_losses: List[float] = []
    route_ce_losses: List[float] = []
    route_ce_weighted_losses: List[float] = []
    entropies: List[float] = []
    top1s: List[float] = []
    keffs: List[float] = []
    m_useds: List[float] = []

    for bucket_idx in schedule:
        bucket = active_buckets[int(bucket_idx)]
        assigned_expert = int(bucket["assigned_expert"])
        assigned_adapter = adapter_name_for_cluster(assigned_expert)
        optimizer.zero_grad(set_to_none=True)
        step_losses: List[float] = []
        step_task_losses: List[float] = []
        step_route_ce_losses: List[float] = []
        step_route_ce_weighted_losses: List[float] = []
        step_entropies: List[float] = []
        step_top1s: List[float] = []
        step_keffs: List[float] = []
        step_m_useds: List[float] = []
        for _micro in range(max(1, int(config.train.grad_accum))):
            batch = move_batch_to_device(next(iterators[int(bucket_idx)]), device)
            if int(k_experts) == 1:
                model.set_adapter(assigned_adapter)
                out = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )
                task_loss = core.compute_ce_loss(out.logits, batch.labels)
                route_ce_loss = task_loss.detach() * 0.0
                total_loss = task_loss
                stats = {
                    "mix_entropy": 0.0,
                    "mix_top1": 1.0,
                    "router_keff": 1.0,
                    "mix_m_used": 1.0,
                }
            else:
                log_probs_mix, stats, route_ce_loss = compute_mixture_logits(
                    model=model,
                    router=routers[0],
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    query_input_ids=batch.query_input_ids,
                    query_attention_mask=batch.query_attention_mask,
                    k_experts=k_experts,
                    assigned_expert=assigned_expert,
                    m_select=config.router.m_select,
                    m_tau=config.router.m_tau,
                    expert_id_to_adapter=expert_id_to_adapter,
                    mix_in_logprob=True,
                    absolute_routing=absolute_routing,
                    disable_lora_for_query=disable_lora_for_query,
                )
                task_loss = compute_nll_loss(log_probs_mix, batch.labels)
                total_loss = task_loss + config.router.route_ce_weight * route_ce_loss
            weighted_route_ce_loss = config.router.route_ce_weight * route_ce_loss
            (total_loss / max(1, int(config.train.grad_accum))).backward()
            step_losses.append(float(total_loss.detach().item()))
            step_task_losses.append(float(task_loss.detach().item()))
            step_route_ce_losses.append(float(route_ce_loss.detach().item()))
            step_route_ce_weighted_losses.append(float(weighted_route_ce_loss.detach().item()))
            step_entropies.append(float(stats["mix_entropy"]))
            step_top1s.append(float(stats["mix_top1"]))
            step_keffs.append(float(stats["router_keff"]))
            step_m_useds.append(float(stats["mix_m_used"]))

        _clip_gradients_if_needed(trainable_params, config.train.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(np.mean(step_losses)) if step_losses else 0.0)
        task_losses.append(float(np.mean(step_task_losses)) if step_task_losses else 0.0)
        route_ce_losses.append(float(np.mean(step_route_ce_losses)) if step_route_ce_losses else 0.0)
        route_ce_weighted_losses.append(
            float(np.mean(step_route_ce_weighted_losses)) if step_route_ce_weighted_losses else 0.0
        )
        entropies.append(float(np.mean(step_entropies)) if step_entropies else 0.0)
        top1s.append(float(np.mean(step_top1s)) if step_top1s else 0.0)
        keffs.append(float(np.mean(step_keffs)) if step_keffs else 0.0)
        m_useds.append(float(np.mean(step_m_useds)) if step_m_useds else 0.0)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "task_loss": float(np.mean(task_losses)) if task_losses else 0.0,
        "route_ce_loss": float(np.mean(route_ce_losses)) if route_ce_losses else 0.0,
        "route_ce_weighted_loss": float(np.mean(route_ce_weighted_losses)) if route_ce_weighted_losses else 0.0,
        "mix_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "mix_top1": float(np.mean(top1s)) if top1s else 0.0,
        "router_keff": float(np.mean(keffs)) if keffs else 0.0,
        "mix_m_used": float(np.mean(m_useds)) if m_useds else 0.0,
        "expert_sample_counts": {int(k): int(v) for k, v in expert_sample_counts.items()},
        "total_examples": int(sum(expert_sample_counts.values())),
        "num_bucket_uploads": int(len(active_buckets)),
        "num_router_uploads": int(1 if routers and int(k_experts) > 1 else 0),
    }


@torch.no_grad()
def _evaluate_split_common(
    *,
    model: Any,
    tokenizer: Any,
    split_by_task: Dict[str, List[Example]],
    config: ExperimentConfig,
    include_predictions: bool,
    compute_task_metrics: bool,
    progress_desc: str,
    disable_progress: bool,
    eval_progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    score_example: Callable[
        [Example, str, int, bool],
        Tuple[Optional[str], torch.Tensor, Optional[str], Optional[List[float]]],
    ],
    score_batch: Optional[
        Callable[
            [Sequence[Example], str, int, bool],
            List[Tuple[Optional[str], torch.Tensor, Optional[str], Optional[List[float]]]],
        ]
    ] = None,
) -> Dict[str, Any]:
    generate_predictions = bool(compute_task_metrics or include_predictions)
    configured_eval_batch_size = int(getattr(config.train, "eval_batch_size", 0) or 0)
    if configured_eval_batch_size > 0:
        eval_batch_size = configured_eval_batch_size
    else:
        eval_batch_size = max(1, int(getattr(config.train, "batch_size")))
    use_batching = score_batch is not None and eval_batch_size > 1
    stats = _EvalStats()
    eval_bar = make_progress(
        total=sum(len(examples) for examples in split_by_task.values()),
        desc=progress_desc,
        disable=disable_progress,
        leave=False,
    )
    try:
        for task, examples in split_by_task.items():
            eval_bar.set_postfix({"task": task}, refresh=False)
            max_new_tokens = _eval_max_new_tokens_for_task(config, task)
            preds: List[str] = []
            targets: List[str] = []
            metas: List[Dict[str, Any]] = []
            losses: List[float] = []
            if use_batching:
                for batch_start in range(0, len(examples), eval_batch_size):
                    batch_examples = examples[batch_start : batch_start + eval_batch_size]
                    batch_results = score_batch(batch_examples, task, max_new_tokens, generate_predictions)
                    for local_idx, (pred, loss, weights_key, weights_payload) in enumerate(batch_results):
                        example_index = batch_start + local_idx
                        ex = batch_examples[local_idx]
                        _capture_eval_generation(
                            generate_predictions=generate_predictions,
                            pred=pred,
                            ex=ex,
                            preds=preds,
                            targets=targets,
                            metas=metas,
                        )
                        _record_eval_loss(
                            stats=stats, loss=loss, losses=losses, task=task,
                            example_index=example_index, max_new_tokens=max_new_tokens,
                            eval_progress_callback=eval_progress_callback,
                        )
                        _append_eval_prediction_record(
                            stats=stats, include_predictions=include_predictions,
                            task=task, example_index=example_index, ex=ex,
                            pred=pred, max_new_tokens=max_new_tokens, loss=loss,
                            weights_key=weights_key, weights_payload=weights_payload,
                        )
                        eval_bar.update(1)
            else:
                for example_index, ex in enumerate(examples):
                    pred, loss, weights_key, weights_payload = score_example(
                        ex, task, max_new_tokens, generate_predictions
                    )
                    _capture_eval_generation(
                        generate_predictions=generate_predictions,
                        pred=pred,
                        ex=ex,
                        preds=preds,
                        targets=targets,
                        metas=metas,
                    )
                    _record_eval_loss(
                        stats=stats,
                        loss=loss,
                        losses=losses,
                        task=task,
                        example_index=example_index,
                        max_new_tokens=max_new_tokens,
                        eval_progress_callback=eval_progress_callback,
                    )
                    _append_eval_prediction_record(
                        stats=stats,
                        include_predictions=include_predictions,
                        task=task,
                        example_index=example_index,
                        ex=ex,
                        pred=pred,
                        max_new_tokens=max_new_tokens,
                        loss=loss,
                        weights_key=weights_key,
                        weights_payload=weights_payload,
                    )
                    eval_bar.update(1)
            _finish_eval_task(
                stats=stats,
                task=task,
                n_examples=len(examples),
                losses=losses,
                preds=preds,
                targets=targets,
                metas=metas,
                compute_task_metrics=compute_task_metrics,
                eval_progress_callback=eval_progress_callback,
            )
    finally:
        eval_bar.close()
    return _build_eval_metrics(
        stats=stats,
        compute_task_metrics=compute_task_metrics,
        include_predictions=include_predictions,
    )


@torch.no_grad()
def _evaluate_split(
    *,
    model_bundle: ModelBundle,
    state: PrototypeFederatedState,
    split_by_task: Dict[str, List[Example]],
    config: ExperimentConfig,
    include_predictions: bool = False,
    compute_task_metrics: bool = True,
    progress_desc: str = "eval",
    disable_progress: bool = False,
    eval_progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    _eval_routers: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    k_experts = int(state.k_experts)
    expert_id_to_adapter = {eid: adapter_name_for_cluster(eid) for eid in range(k_experts)}
    load_all_expert_states(model, state.server_expert_sd, k_experts)
    model.eval()
    routers = None
    absolute_routing = False
    if state.router_state_dicts is not None:
        router_out_dim = _router_out_dim_from_state(state.router_state_dicts, 0, k_experts)
        absolute_routing = int(router_out_dim) == int(k_experts)
        if _eval_routers is not None:
            routers = _eval_routers
            load_router_states(routers, state.router_state_dicts, next(model.parameters()).device)
        else:
            routers = build_router_dict(
                config, k_experts, model_bundle.d_model, next(model.parameters()).device,
                num_routers=1, out_dim=router_out_dim,
            )
            load_router_states(routers, state.router_state_dicts, next(model.parameters()).device)
        for router in routers.values():
            router.eval()

    device = next(model.parameters()).device
    collate_fn = collate_fn_builder(tokenizer, config.train.max_length)

    def _score(ex: Example, _task: str, max_new_tokens: int, generate_predictions: bool):
        pred: Optional[str] = None
        weights_payload = None
        if routers is None or k_experts == 1:
            model.set_adapter(expert_id_to_adapter[0])
            if generate_predictions:
                pred = greedy_generate(model, tokenizer, ex.prompt, max_new_tokens=max_new_tokens)
            batch = collate_fn([ex])
            batch = move_batch_to_device(batch, device)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
            loss = core.compute_ce_loss(out.logits, batch.labels)
        else:
            weights_k = core.compute_example_expert_weights(
                model=model, tokenizer=tokenizer, router=routers[0], ex=ex,
                k_experts=k_experts, assigned_expert=0,
                expert_id_to_adapter=expert_id_to_adapter,
                max_length=config.train.max_length,
                m_select_eval=config.router.m_select,
                m_tau_eval=config.router.m_tau,
                disable_lora_for_query=True,
                absolute_routing=absolute_routing,
            )
            weights_payload = [float(v) for v in weights_k.detach().cpu().view(-1).tolist()]
            if generate_predictions:
                pred = greedy_generate_mixture(
                    model=model, tokenizer=tokenizer, user_prompt=ex.prompt,
                    weights_k=weights_k, expert_id_to_adapter=expert_id_to_adapter,
                    max_new_tokens=max_new_tokens,
                )
            batch = collate_fn([ex])
            batch = move_batch_to_device(batch, device)
            log_probs = compute_fixed_weight_log_probs(
                model=model, input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                weights_k=weights_k, expert_id_to_adapter=expert_id_to_adapter,
            )
            loss = compute_nll_loss(log_probs, batch.labels)
        return pred, loss, "weights", weights_payload

    def _score_batch(examples: Sequence[Example], _task: str, max_new_tokens: int, generate_predictions: bool):
        preds: List[Optional[str]] = [None] * len(examples)
        weights_payloads: List[Optional[List[float]]] = [None] * len(examples)

        if routers is None or k_experts == 1:
            if generate_predictions:
                prompts = [ex.prompt for ex in examples]
                preds = greedy_generate_batch(model, tokenizer, prompts, max_new_tokens=max_new_tokens)
            batch = collate_fn(list(examples))
            batch = move_batch_to_device(batch, device)
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
            per_ex_losses = _per_example_ce_loss(out.logits, batch.labels)
            return [(preds[i], per_ex_losses[i], None, None) for i in range(len(examples))]

        # Batch compute expert weights via single query-embedding forward pass
        weights_batch = compute_batch_expert_weights(
            model=model, tokenizer=tokenizer, router=routers[0],
            examples=list(examples), k_experts=k_experts, assigned_expert=0,
            expert_id_to_adapter=expert_id_to_adapter,
            max_length=config.train.max_length,
            m_select_eval=config.router.m_select,
            m_tau_eval=config.router.m_tau,
            disable_lora_for_query=True,
            absolute_routing=absolute_routing,
        )

        for i, ex in enumerate(examples):
            weights_payloads[i] = [float(v) for v in weights_batch[i].detach().cpu().view(-1).tolist()]
            if generate_predictions:
                preds[i] = greedy_generate_mixture(
                    model=model, tokenizer=tokenizer, user_prompt=ex.prompt,
                    weights_k=weights_batch[i:i+1], expert_id_to_adapter=expert_id_to_adapter,
                    max_new_tokens=max_new_tokens,
                )

        # Batch loss computation
        batch = collate_fn(list(examples))
        batch = move_batch_to_device(batch, device)
        log_probs = compute_fixed_weight_log_probs(
            model=model, input_ids=batch.input_ids, attention_mask=batch.attention_mask,
            weights_k=weights_batch, expert_id_to_adapter=expert_id_to_adapter,
        )
        per_ex_losses = _per_example_nll_loss(log_probs, batch.labels)

        return [(preds[i], per_ex_losses[i], "weights", weights_payloads[i]) for i in range(len(examples))]

    return _evaluate_split_common(
        model=model, tokenizer=tokenizer, split_by_task=split_by_task,
        config=config, include_predictions=include_predictions,
        compute_task_metrics=compute_task_metrics, progress_desc=progress_desc,
        disable_progress=disable_progress, eval_progress_callback=eval_progress_callback,
        score_example=_score, score_batch=_score_batch,
    )


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    config = _build_config_from_args(args)

    disable_progress = not bool(config.logging.show_progress)
    out_root = ensure_dir(config.out_dir / f"seed_{config.seed}")
    result_root = ensure_dir(config.results_dir / f"seed_{config.seed}")
    checkpoint_root = ensure_dir(out_root / "checkpoints")
    swanlab_name = str(config.logging.swanlab_name).strip() or f"train_seed{config.seed}"
    swanlab_run = SwanLabRun(
        enabled=bool(config.logging.use_swanlab),
        project=str(config.logging.swanlab_project),
        name=swanlab_name,
        mode=str(config.logging.swanlab_mode),
        config=config_to_dict(config),
    )
    dump_json(out_root / "config.json", config_to_dict(config))

    progress_write("[train] preparing data", disable=disable_progress)
    data = prepare_data(config)
    dump_json(
        out_root / "data_summary.json",
        {
            "token_stats": data.token_stats,
            "split_index_path": str(data.split_index_path),
            "client_partition_path": str(data.client_partition_path),
            "num_clients": int(len(data.clients)),
            "tasks": list(config.data.tasks),
        },
    )

    interleave_client_buckets = bool(getattr(args, "interleave_client_buckets", True))
    router_aggregation_scope = str(getattr(args, "router_aggregation_scope", "client")).strip().lower()
    oracle_task_routing = bool(getattr(args, "oracle_task_routing", False))
    discovery_label = "oracle task routing" if oracle_task_routing else "prototype discovery"
    progress_write(f"[train] building model for {discovery_label}", disable=disable_progress)
    discovery_bundle = build_model_bundle(config)
    if oracle_task_routing:
        prototypes = discover_oracle_task_prototypes(
            config=config,
            data=data,
            model_bundle=discovery_bundle,
            discovery_warmup_steps=int(args.discovery_warmup_steps),
            discovery_warmup_batch_size=int(args.discovery_warmup_batch_size),
            discovery_warmup_mode=str(getattr(args, "discovery_warmup_mode", "steps")),
            discovery_warmup_epochs=int(getattr(args, "discovery_warmup_epochs", 1)),
        )
    else:
        prototypes = discover_prototypes(
            config=config,
            data=data,
            model_bundle=discovery_bundle,
            embedding_batch_size=int(args.embedding_batch_size),
            normalize_embeddings=bool(args.normalize_embeddings),
            discovery_warmup_steps=int(args.discovery_warmup_steps),
            discovery_warmup_batch_size=int(args.discovery_warmup_batch_size),
            local_cluster_algorithm=str(getattr(args, "local_cluster_algorithm", "kmeans")),
            prototype_signature_type=str(getattr(args, "prototype_signature_type", "lora_b")),
            discovery_warmup_mode=str(getattr(args, "discovery_warmup_mode", "steps")),
            discovery_warmup_epochs=int(getattr(args, "discovery_warmup_epochs", 1)),
        )
    dump_json(out_root / "prototype_discovery.json", prototypes.summary)
    dump_json(out_root / "routing_priors.json", {"group_priors": [float(v) for v in prototypes.group_priors]})
    swanlab_run.log(
        {
            "prototype/purity": float(prototypes.summary["metrics"]["purity"]),
            "prototype/nmi": float(prototypes.summary["metrics"]["nmi"]),
            "prototype/ari": float(prototypes.summary["metrics"]["ari"]),
            "prototype/global_selected_k": int(prototypes.selected_global_k),
        }
    )
    del discovery_bundle
    core.release_cuda_memory()

    state = initialize_prototype_federated_state(config, prototypes.selected_global_k)
    state.server_expert_sd = {
        int(k): clone_state_dict_cpu(v) for k, v in prototypes.initial_expert_states.items()
    }
    progress_write(
        f"[train] discovered global_k={prototypes.selected_global_k} purity={prototypes.summary['metrics']['purity']:.4f}",
        disable=disable_progress,
    )

    progress_write("[train] building training model", disable=disable_progress)
    model_bundle = build_model_bundle(config)
    model = model_bundle.model
    tokenizer = model_bundle.tokenizer
    device = next(model.parameters()).device
    k_experts = int(state.k_experts)
    expert_id_to_adapter = {eid: adapter_name_for_cluster(eid) for eid in range(k_experts)}
    add_expert_adapters_if_needed(model, k_experts)
    routers = None
    eval_routers = None
    if k_experts > 1:
        routers = build_router_dict(config, k_experts, model_bundle.d_model, device, num_routers=1, out_dim=k_experts)
        state.router_state_dicts = dump_router_states(routers)
        eval_routers = build_router_dict(config, k_experts, model_bundle.d_model, device, num_routers=1, out_dim=k_experts)

    final_val_metrics: Dict[str, Any] = {}
    best_val_macro = {"metric": None, "round": None, "path": None}
    best_val_loss = {"metric": None, "round": None, "path": None}
    periodic_checkpoints: List[Dict[str, Any]] = []

    with make_progress(total=int(config.train.global_rounds), desc="train", disable=disable_progress) as round_bar:
        for round_idx in range(int(config.train.global_rounds)):
            base_expert_states = {eid: clone_state_dict_cpu(state.server_expert_sd[eid]) for eid in range(k_experts)}
            base_router_states = None if state.router_state_dicts is None else {
                0: clone_state_dict_cpu(state.router_state_dicts[0])
            }
            expert_deltas_by_group: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
            router_deltas: List[Dict[str, torch.Tensor]] = []
            router_delta_weights: List[float] = []
            upload_weights: Dict[int, List[float]] = defaultdict(list)
            round_losses: List[float] = []
            round_loss_weights: List[float] = []
            round_task_losses: List[float] = []
            round_route_ce_losses: List[float] = []
            round_route_ce_weighted_losses: List[float] = []
            round_entropies: List[float] = []
            round_top1s: List[float] = []
            examples_per_expert: Dict[int, int] = defaultdict(int)
            uploads_per_expert: Dict[int, int] = defaultdict(int)
            clients_per_expert_sets: Dict[int, set[int]] = defaultdict(set)

            with make_progress(total=len(data.clients), desc=f"round={round_idx + 1} clients", disable=disable_progress, leave=False) as client_bar:
                for client in data.clients:
                    assignments = prototypes.local_assignments_by_client.get(int(client.client_id), [])
                    bucket_sizes = [len(assignment.sample_indices) for assignment in assignments]
                    bucket_step_budgets = _allocate_bucket_local_steps(
                        bucket_sizes,
                        int(config.train.local_steps),
                        min_steps=int(args.bucket_min_steps),
                    )
                    client_router_bucket_deltas: List[Dict[str, torch.Tensor]] = []
                    client_router_bucket_weights: List[float] = []

                    if interleave_client_buckets:
                        local_stats = _run_client_interleaved_bucket_steps(
                            model=model,
                            tokenizer=tokenizer,
                            routers=routers if routers is not None else {},
                            client_examples=client.train_examples,
                            assignments=assignments,
                            bucket_step_budgets=bucket_step_budgets,
                            base_expert_states=base_expert_states,
                            base_router_state=None if base_router_states is None else base_router_states[0],
                            k_experts=k_experts,
                            config=config,
                            device=device,
                            expert_id_to_adapter=expert_id_to_adapter,
                            shuffle_seed=int(config.seed + round_idx * 10_000 + client.client_id * 100),
                            absolute_routing=True,
                            disable_lora_for_query=True,
                        )
                        round_losses.append(float(local_stats["loss"]))
                        round_task_losses.append(float(local_stats["task_loss"]))
                        round_route_ce_losses.append(float(local_stats["route_ce_loss"]))
                        round_route_ce_weighted_losses.append(float(local_stats["route_ce_weighted_loss"]))
                        round_loss_weights.append(float(max(1, int(local_stats["total_examples"]))))
                        round_entropies.append(float(local_stats["mix_entropy"]))
                        round_top1s.append(float(local_stats["mix_top1"]))
                        expert_sample_counts = {
                            int(k): int(v) for k, v in dict(local_stats["expert_sample_counts"]).items()
                        }
                        for assigned_expert, expert_sample_count in expert_sample_counts.items():
                            examples_per_expert[assigned_expert] += int(expert_sample_count)
                            uploads_per_expert[assigned_expert] += 1
                            clients_per_expert_sets[assigned_expert].add(int(client.client_id))
                            assigned_adapter = adapter_name_for_cluster(assigned_expert)
                            expert_deltas_by_group[assigned_expert].append(
                                _state_dict_difference(
                                    get_adapter_state(model, assigned_adapter),
                                    base_expert_states[assigned_expert],
                                )
                            )
                            upload_weights[assigned_expert].append(float(max(1, int(expert_sample_count))))
                        if routers is not None and base_router_states is not None:
                            if router_aggregation_scope == "client":
                                router_deltas.append(
                                    _state_dict_difference(
                                        dump_router_states({0: routers[0]})[0],
                                        base_router_states[0],
                                    )
                                )
                                router_delta_weights.append(float(max(1, int(local_stats["total_examples"]))))
                            else:
                                trained_expert_states = {
                                    expert_id: clone_state_dict_cpu(
                                        get_adapter_state(model, adapter_name_for_cluster(expert_id))
                                    )
                                    for expert_id in range(k_experts)
                                }
                                for assignment, bucket_local_steps in zip(assignments, bucket_step_budgets):
                                    if int(bucket_local_steps) <= 0:
                                        continue
                                    bucket_examples = [client.train_examples[idx] for idx in assignment.sample_indices]
                                    if not bucket_examples:
                                        continue
                                    load_all_expert_states(model, trained_expert_states, k_experts)
                                    routers[0].load_state_dict(
                                        {key: value.to(device) for key, value in base_router_states[0].items()}
                                    )
                                    _run_local_cluster_steps(
                                        model=model,
                                        tokenizer=tokenizer,
                                        routers=routers,
                                        examples=bucket_examples,
                                        assigned_expert=int(assignment.assigned_expert),
                                        k_experts=k_experts,
                                        config=config,
                                        device=device,
                                        expert_id_to_adapter=expert_id_to_adapter,
                                        shuffle_seed=int(
                                            config.seed
                                            + round_idx * 10_000
                                            + client.client_id * 100
                                            + assignment.local_cluster_id
                                        ),
                                        local_steps_override=int(bucket_local_steps),
                                        absolute_routing=True,
                                        disable_lora_for_query=True,
                                        train_lora_adapters=False,
                                    )
                                    router_deltas.append(
                                        _state_dict_difference(
                                            dump_router_states({0: routers[0]})[0],
                                            base_router_states[0],
                                        )
                                    )
                                    router_delta_weights.append(float(max(1, len(bucket_examples))))
                    else:
                        for assignment, bucket_local_steps in zip(assignments, bucket_step_budgets):
                            if int(bucket_local_steps) <= 0:
                                continue
                            bucket_examples = [client.train_examples[idx] for idx in assignment.sample_indices]
                            bucket_sample_count = int(len(bucket_examples))
                            assigned_expert = int(assignment.assigned_expert)
                            examples_per_expert[assigned_expert] += bucket_sample_count
                            uploads_per_expert[assigned_expert] += 1
                            clients_per_expert_sets[assigned_expert].add(int(client.client_id))
                            load_all_expert_states(model, base_expert_states, k_experts)
                            if routers is not None and base_router_states is not None:
                                routers[0].load_state_dict(
                                    {key: value.to(device) for key, value in base_router_states[0].items()}
                                )
                            local_stats = _run_local_cluster_steps(
                                model=model,
                                tokenizer=tokenizer,
                                routers=routers if routers is not None else {},
                                examples=bucket_examples,
                                assigned_expert=assigned_expert,
                                k_experts=k_experts,
                                config=config,
                                device=device,
                                expert_id_to_adapter=expert_id_to_adapter,
                                shuffle_seed=int(config.seed + round_idx * 10_000 + client.client_id * 100 + assignment.local_cluster_id),
                                local_steps_override=int(bucket_local_steps),
                                absolute_routing=True,
                                disable_lora_for_query=True,
                            )
                            round_losses.append(float(local_stats["loss"]))
                            round_task_losses.append(float(local_stats["task_loss"]))
                            round_route_ce_losses.append(float(local_stats["route_ce_loss"]))
                            round_route_ce_weighted_losses.append(float(local_stats["route_ce_weighted_loss"]))
                            round_loss_weights.append(float(bucket_sample_count))
                            round_entropies.append(float(local_stats["mix_entropy"]))
                            round_top1s.append(float(local_stats["mix_top1"]))
                            assigned_adapter = adapter_name_for_cluster(assigned_expert)
                            expert_deltas_by_group[assigned_expert].append(
                                _state_dict_difference(get_adapter_state(model, assigned_adapter), base_expert_states[assigned_expert])
                            )
                            if routers is not None and base_router_states is not None:
                                bucket_router_delta = _state_dict_difference(
                                    dump_router_states({0: routers[0]})[0],
                                    base_router_states[0],
                                )
                                bucket_router_weight = float(max(1, bucket_sample_count))
                                if router_aggregation_scope == "bucket":
                                    router_deltas.append(bucket_router_delta)
                                    router_delta_weights.append(bucket_router_weight)
                                else:
                                    client_router_bucket_deltas.append(bucket_router_delta)
                                    client_router_bucket_weights.append(bucket_router_weight)
                            upload_weights[assigned_expert].append(float(max(1, bucket_sample_count)))
                        if (
                            routers is not None
                            and base_router_states is not None
                            and router_aggregation_scope == "client"
                            and client_router_bucket_deltas
                        ):
                            client_router_state = aggregate_weighted_state_dicts(
                                base_router_states[0],
                                client_router_bucket_deltas,
                                client_router_bucket_weights,
                            )
                            router_deltas.append(
                                _state_dict_difference(
                                    client_router_state,
                                    base_router_states[0],
                                )
                            )
                            router_delta_weights.append(float(sum(client_router_bucket_weights)))
                    client_bar.update(1)

            for expert_id in range(k_experts):
                if not expert_deltas_by_group[expert_id]:
                    continue
                new_expert_state, _ = aggregate_expert_and_router_updates(
                    base_expert_state=base_expert_states[expert_id],
                    expert_deltas=expert_deltas_by_group[expert_id],
                    client_weights=upload_weights[expert_id],
                )
                state.server_expert_sd[expert_id] = new_expert_state

            if routers is not None and base_router_states is not None and router_deltas:
                new_router_state = aggregate_weighted_state_dicts(
                    base_router_states[0],
                    router_deltas,
                    router_delta_weights,
                )
                state.router_state_dicts[0] = new_router_state
                routers[0].load_state_dict({key: value.to(device) for key, value in new_router_state.items()})

            usage_summary = _build_usage_summary(
                k_experts=k_experts,
                examples_per_expert=examples_per_expert,
                uploads_per_expert=uploads_per_expert,
                clients_per_expert={eid: len(client_ids) for eid, client_ids in clients_per_expert_sets.items()},
            )
            round_summary = {
                "round": int(round_idx + 1),
                "train_loss": _weighted_mean(round_losses, round_loss_weights),
                "train_task_loss": _weighted_mean(round_task_losses, round_loss_weights),
                "train_route_ce_loss": _weighted_mean(round_route_ce_losses, round_loss_weights),
                "train_route_ce_weighted_loss": _weighted_mean(round_route_ce_weighted_losses, round_loss_weights),
                "train_num_examples": int(sum(round_loss_weights)),
                "router_entropy": float(np.mean(round_entropies)) if round_entropies else 0.0,
                "router_top1": float(np.mean(round_top1s)) if round_top1s else 0.0,
                "usage": usage_summary,
            }

            if bool(config.train.eval_every_round) and (
                ((round_idx + 1) % max(1, int(config.train.eval_every_n_rounds)) == 0)
                or ((round_idx + 1) == int(config.train.global_rounds))
            ):
                compute_val_task_metrics = _should_compute_val_task_metrics(config, round_idx)
                final_val_metrics = _evaluate_split(
                    model_bundle=model_bundle,
                    state=state,
                    split_by_task=data.global_val_by_task,
                    config=config,
                    compute_task_metrics=compute_val_task_metrics,
                    progress_desc=f"fedweave val r={round_idx + 1}",
                    disable_progress=disable_progress,
                    _eval_routers=eval_routers,
                )
                _record_val_metrics(round_summary, final_val_metrics)
                checkpoint_payload = _build_fedweave_checkpoint_payload(
                    config=config,
                    round_idx=int(round_idx + 1),
                    state=state,
                    prototypes=prototypes,
                    embedding_batch_size=int(args.embedding_batch_size),
                    normalize_embeddings=bool(args.normalize_embeddings),
                    bucket_min_steps=int(args.bucket_min_steps),
                    discovery_warmup_steps=int(args.discovery_warmup_steps),
                    discovery_warmup_batch_size=int(args.discovery_warmup_batch_size),
                    discovery_warmup_mode=str(getattr(args, "discovery_warmup_mode", "steps")),
                    discovery_warmup_epochs=int(getattr(args, "discovery_warmup_epochs", 1)),
                    interleave_client_buckets=interleave_client_buckets,
                    router_aggregation_scope=router_aggregation_scope,
                    oracle_task_routing=oracle_task_routing,
                    val_metrics=final_val_metrics,
                )
                if bool(final_val_metrics.get("task_metrics_computed", True)):
                    _maybe_update_best_checkpoint(
                        tracker=best_val_macro,
                        metric_value=float(final_val_metrics["avg_macro"]),
                        mode="max",
                        round_idx=int(round_idx + 1),
                        path=checkpoint_root / "best_val_macro.pt",
                        payload=checkpoint_payload,
                    )
                _maybe_update_best_checkpoint(
                    tracker=best_val_loss,
                    metric_value=float(final_val_metrics["avg_loss"]),
                    mode="min",
                    round_idx=int(round_idx + 1),
                    path=checkpoint_root / "best_val_loss.pt",
                    payload=checkpoint_payload,
                )

            _save_periodic_best_val_loss_checkpoint(
                config=config,
                round_idx=round_idx,
                checkpoint_root=checkpoint_root,
                best_val_loss=best_val_loss,
                periodic_checkpoints=periodic_checkpoints,
            )

            state.round_history.append(round_summary)
            state.current_round = int(round_idx + 1)
            swanlab_payload = _build_round_swanlab_payload(round_summary, prefix="train")
            swanlab_run.log(swanlab_payload, step=int(round_idx + 1))
            round_bar.update(1)
            round_bar.set_postfix(
                {
                    "loss": format_metric(round_summary["train_loss"]),
                    "val": format_metric(round_summary.get("val_macro", round_summary.get("val_loss"))),
                }
            )
            progress_write(
                f"[train][round={round_idx + 1}] "
                f"loss={format_metric(round_summary['train_loss'])} "
                f"task={format_metric(round_summary['train_task_loss'])} "
                f"route_ce={format_metric(round_summary['train_route_ce_weighted_loss'])} "
                f"val={format_metric(round_summary.get('val_macro', round_summary.get('val_loss')))}",
                disable=disable_progress,
            )

    final_checkpoint_path = checkpoint_root / "final.pt"
    _save_checkpoint(
        final_checkpoint_path,
        _build_fedweave_checkpoint_payload(
            config=config,
            round_idx=int(state.current_round),
            state=state,
            prototypes=prototypes,
            embedding_batch_size=int(args.embedding_batch_size),
            normalize_embeddings=bool(args.normalize_embeddings),
            bucket_min_steps=int(args.bucket_min_steps),
            discovery_warmup_steps=int(args.discovery_warmup_steps),
            discovery_warmup_batch_size=int(args.discovery_warmup_batch_size),
            discovery_warmup_mode=str(getattr(args, "discovery_warmup_mode", "steps")),
            discovery_warmup_epochs=int(getattr(args, "discovery_warmup_epochs", 1)),
            interleave_client_buckets=interleave_client_buckets,
            router_aggregation_scope=router_aggregation_scope,
            oracle_task_routing=oracle_task_routing,
            val_metrics=final_val_metrics,
        ),
    )

    summary = {
        "config": config_to_dict(config),
        "method": "fedweave",
        "train_args": {
            "embedding_batch_size": int(args.embedding_batch_size),
            "normalize_embeddings": bool(args.normalize_embeddings),
            "bucket_min_steps": int(args.bucket_min_steps),
            "discovery_warmup_steps": int(args.discovery_warmup_steps),
            "discovery_warmup_batch_size": int(args.discovery_warmup_batch_size),
            "local_cluster_algorithm": str(getattr(args, "local_cluster_algorithm", "kmeans")),
            "prototype_signature_type": str(getattr(args, "prototype_signature_type", "lora_b")),
            "interleave_client_buckets": bool(interleave_client_buckets),
            "router_aggregation_scope": str(router_aggregation_scope),
            "oracle_task_routing": bool(oracle_task_routing),
        },
        "prototype_discovery": prototypes.summary,
        "final_val": final_val_metrics,
        "round_history": list(state.round_history),
        "checkpoints": {
            "best_val_macro": best_val_macro,
            "best_val_loss": best_val_loss,
            "periodic": periodic_checkpoints,
            "final": str(final_checkpoint_path),
        },
    }
    dump_json(out_root / "training_summary.json", summary)
    dump_json(result_root / "training_summary.json", summary)
    swanlab_run.finish()
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_train_parser()
    args = parser.parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
