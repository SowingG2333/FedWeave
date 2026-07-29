#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    ExperimentConfig,
    build_parser,
    build_config_from_namespace,
    config_to_dict,
)
from data import prepare_data
from lora import build_model_bundle
import engine as core
from train import PrototypeArtifacts, discover_prototypes
from src.utils.io import dump_json, ensure_dir
from src.utils.paths import OUTPUTS_DIR, RESULTS_DIR
from src.utils.progress import progress_write
from src.utils.logging import SwanLabRun, flatten_metrics


def _default_output_dir() -> Path:
    return OUTPUTS_DIR / "base" / "mve"


def _default_results_dir() -> Path:
    return RESULTS_DIR / "base" / "mve"


def _default_data_cache_dir() -> Path:
    return OUTPUTS_DIR / "data_cache"


def build_mve_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Minimal viability experiment for adaptive local/global prototype discovery."
    parser.set_defaults(show_progress=True, data_cache_dir=str(_default_data_cache_dir()))
    parser.add_argument("--mve_out_dir", type=str, default=str(_default_output_dir()))
    parser.add_argument("--mve_results_dir", type=str, default=str(_default_results_dir()))
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--discovery_warmup_steps", type=int, default=5)
    parser.add_argument("--discovery_warmup_batch_size", type=int, default=4)
    parser.add_argument(
        "--discovery_warmup_mode",
        type=str,
        default="steps",
        choices=["steps", "epochs"],
        help="Warmup iteration mode: 'steps' runs a fixed number of optimizer steps; 'epochs' iterates over all cluster data.",
    )
    parser.add_argument("--discovery_warmup_epochs", type=int, default=1)
    parser.add_argument("--normalize_embeddings", dest="normalize_embeddings", action="store_true")
    parser.add_argument("--no_normalize_embeddings", dest="normalize_embeddings", action="store_false")
    parser.set_defaults(normalize_embeddings=True)
    parser.add_argument(
        "--local_feature_type",
        type=str,
        default="embedding",
        choices=["embedding", "gradient_pca"],
        help="Feature representation for per-client local clustering.",
    )
    parser.add_argument(
        "--local_cluster_algorithm",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "spectral"],
        help="Per-client local clustering algorithm.",
    )
    parser.add_argument(
        "--prototype_signature_type",
        type=str,
        default="lora_b",
        choices=["lora_a", "lora_b", "lora_ab"],
        help="LoRA signature used for cross-client prototype alignment after local warmup.",
    )
    parser.add_argument("--gradient_pca_dim", type=int, default=64)
    return parser


def _build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if int(args.discovery_warmup_steps) < 0:
        raise ValueError("--discovery_warmup_steps must be >= 0.")
    if int(args.discovery_warmup_batch_size) <= 0:
        raise ValueError("--discovery_warmup_batch_size must be > 0.")
    if int(args.discovery_warmup_epochs) <= 0:
        raise ValueError("--discovery_warmup_epochs must be > 0.")
    if int(args.gradient_pca_dim) <= 0:
        raise ValueError("--gradient_pca_dim must be > 0.")
    return build_config_from_namespace(
        args,
        out_dir=args.mve_out_dir,
        results_dir=args.mve_results_dir,
    )


def run_mve(args: argparse.Namespace) -> Dict[str, Any]:
    config = _build_config_from_args(args)
    disable_progress = not bool(config.logging.show_progress)

    out_root = ensure_dir(config.out_dir / f"seed_{config.seed}")
    result_root = ensure_dir(config.results_dir / f"seed_{config.seed}")
    swanlab_name = str(config.logging.swanlab_name).strip() or f"mve_seed{config.seed}"
    swanlab_run = SwanLabRun(
        enabled=bool(config.logging.use_swanlab),
        project=str(config.logging.swanlab_project),
        name=swanlab_name,
        mode=str(config.logging.swanlab_mode),
        config=config_to_dict(config),
    )
    dump_json(out_root / "config.json", config_to_dict(config))
    dump_json(
        out_root / "mve_args.json",
        {
            "embedding_batch_size": int(args.embedding_batch_size),
            "discovery_warmup_steps": int(args.discovery_warmup_steps),
            "discovery_warmup_batch_size": int(args.discovery_warmup_batch_size),
            "discovery_warmup_mode": str(args.discovery_warmup_mode),
            "discovery_warmup_epochs": int(args.discovery_warmup_epochs),
            "normalize_embeddings": bool(args.normalize_embeddings),
            "local_feature_type": str(args.local_feature_type),
            "local_cluster_algorithm": str(args.local_cluster_algorithm),
            "prototype_signature_type": str(args.prototype_signature_type),
            "gradient_pca_dim": int(args.gradient_pca_dim),
        },
    )

    progress_write("[mve] preparing data", disable=disable_progress)
    data = prepare_data(config)
    dump_json(
        out_root / "data_summary.json",
        {
            "token_stats": data.token_stats,
            "split_index_path": str(data.split_index_path),
            "client_partition_path": str(data.client_partition_path),
        },
    )

    progress_write("[mve] building model for discovery", disable=disable_progress)
    model_bundle = build_model_bundle(config)

    progress_write("[mve] running prototype discovery", disable=disable_progress)
    prototypes: PrototypeArtifacts = discover_prototypes(
        config=config,
        data=data,
        model_bundle=model_bundle,
        embedding_batch_size=int(args.embedding_batch_size),
        normalize_embeddings=bool(args.normalize_embeddings),
        discovery_warmup_steps=int(args.discovery_warmup_steps),
        discovery_warmup_batch_size=int(args.discovery_warmup_batch_size),
        discovery_warmup_mode=str(args.discovery_warmup_mode),
        discovery_warmup_epochs=int(args.discovery_warmup_epochs),
        local_feature_type=str(args.local_feature_type),
        local_cluster_algorithm=str(args.local_cluster_algorithm),
        prototype_signature_type=str(args.prototype_signature_type),
        gradient_pca_dim=int(args.gradient_pca_dim),
    )
    local_selection_rows = list(prototypes.summary["local"]["selection_rows"])
    local_cluster_rows = list(prototypes.summary["local"]["cluster_rows"])
    global_section = dict(prototypes.summary["global"])
    metrics = dict(prototypes.summary["metrics"])
    local_cluster_rows_with_global = local_cluster_rows
    prototype_catalog = list(global_section.get("prototype_catalog", []))
    prototype_rows = list(global_section.get("prototype_rows", []))
    global_selection_rows = list(global_section.get("search_rows", []))
    dump_json(out_root / "routing_priors.json", {"group_priors": [float(v) for v in prototypes.group_priors]})
    del model_bundle
    core.release_cuda_memory()

    local_metrics_agg = dict(prototypes.summary["local"].get("metrics_agg", {}))
    mve_args_export = {
        "embedding_batch_size": int(args.embedding_batch_size),
        "discovery_warmup_steps": int(args.discovery_warmup_steps),
        "discovery_warmup_batch_size": int(args.discovery_warmup_batch_size),
        "discovery_warmup_mode": str(args.discovery_warmup_mode),
        "discovery_warmup_epochs": int(args.discovery_warmup_epochs),
        "normalize_embeddings": bool(args.normalize_embeddings),
        "local_feature_type": str(args.local_feature_type),
        "local_cluster_algorithm": str(args.local_cluster_algorithm),
        "prototype_signature_type": str(args.prototype_signature_type),
        "gradient_pca_dim": int(args.gradient_pca_dim),
    }
    summary = {
        "config": config_to_dict(config),
        "mve_args": mve_args_export,
        "data_summary": {
            "token_stats": data.token_stats,
            "split_index_path": str(data.split_index_path),
            "client_partition_path": str(data.client_partition_path),
            "num_clients": int(len(data.clients)),
            "tasks": list(config.data.tasks),
        },
        "local": {
            "selection_rows": local_selection_rows,
            "k_distribution": dict(sorted(Counter(int(row["selected_k"]) for row in local_selection_rows).items())),
            "warmup_rows": list(prototypes.summary["local"].get("warmup_rows", [])),
            "feature_type": str(prototypes.summary["local"].get("feature_type", args.local_feature_type)),
            "cluster_algorithm": str(
                prototypes.summary["local"].get("cluster_algorithm", args.local_cluster_algorithm)
            ),
            "gradient_pca_dim": int(prototypes.summary["local"].get("gradient_pca_dim", args.gradient_pca_dim)),
            "prototype_signature_type": str(
                prototypes.summary["local"].get("prototype_signature_type", args.prototype_signature_type)
            ),
            "metrics_agg": local_metrics_agg,
        },
        "global": {
            "selected_k": int(global_section["selected_k"]),
            "selected_silhouette": global_section["selected_silhouette"],
            "candidate_k_values": [int(k) for k in global_section.get("candidate_k_values", [])],
            "search_rows": global_selection_rows,
            "alignment_space": str(global_section.get("alignment_space", "mean_layerwise_lora_b_cosine_distance")),
            "distance_metric": str(global_section.get("distance_metric", "mean_layerwise_lora_b_cosine_distance")),
            "prototype_signature_type": str(
                global_section.get("prototype_signature_type", args.prototype_signature_type)
            ),
            "inference_routing": str(global_section.get("inference_routing", "population_prior_mixture")),
            "routing_priors": [float(v) for v in global_section.get("routing_priors", prototypes.group_priors)],
            "server_receives_embedding_centroids": bool(global_section.get("server_receives_embedding_centroids", False)),
            "prototype_rows": prototype_rows,
            "prototype_catalog": prototype_catalog,
        },
        "metrics": metrics,
    }

    dump_json(out_root / "local_selection_rows.json", local_selection_rows)
    dump_json(out_root / "local_cluster_rows.json", local_cluster_rows_with_global)
    dump_json(out_root / "global_selection_rows.json", global_selection_rows)
    dump_json(out_root / "prototype_rows.json", prototype_rows)
    dump_json(out_root / "prototype_catalog.json", prototype_catalog)
    dump_json(result_root / "mve_summary.json", summary)
    swanlab_payload = {
        "mve/purity": float(metrics["purity"]),
        "mve/nmi": float(metrics["nmi"]),
        "mve/ari": float(metrics["ari"]),
        "mve/global_selected_k": int(global_section["selected_k"]),
        "mve/global_selected_silhouette": (
            None if global_section["selected_silhouette"] is None else float(global_section["selected_silhouette"])
        ),
    }
    if local_metrics_agg:
        swanlab_payload["mve/local_purity_mean"] = local_metrics_agg["purity"]["mean"]
        swanlab_payload["mve/local_nmi_mean"] = local_metrics_agg["nmi"]["mean"]
        swanlab_payload["mve/local_ari_mean"] = local_metrics_agg["ari"]["mean"]
    swanlab_payload.update(flatten_metrics({"mve_args": summary["mve_args"]}))
    swanlab_run.log(swanlab_payload)
    swanlab_run.finish()

    local_msg = ""
    if local_metrics_agg:
        lp = local_metrics_agg["purity"]
        ln = local_metrics_agg["nmi"]
        la = local_metrics_agg["ari"]
        local_msg = (
            f" local: purity_mean={lp['mean']:.4f}±{lp['std']:.4f} "
            f"nmi_mean={ln['mean']:.4f}±{ln['std']:.4f} "
            f"ari_mean={la['mean']:.4f}±{la['std']:.4f}"
        )
    progress_write(
        f"[mve] purity={metrics['purity']:.4f} nmi={metrics['nmi']:.4f} ari={metrics['ari']:.4f} global_k={int(global_section['selected_k'])}{local_msg}",
        disable=disable_progress,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_mve_parser()
    args = parser.parse_args(argv)
    summary = run_mve(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
