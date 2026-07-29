from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from config import ExperimentConfig
from lora import (
    ModelBundle,
    adapter_name_for_cluster,
    add_expert_adapters_if_needed,
    build_model_bundle,
    build_router_dict,
    clone_state_dict_cpu,
    dump_router_states,
    get_adapter_state,
    remap_adapter_state,
)


@dataclass
class LocalClusterAssignment:
    client_id: int
    local_cluster_id: int
    sample_indices: List[int]
    assigned_expert: int


@dataclass
class PrototypeFederatedState:
    k_experts: int
    server_expert_sd: Dict[int, Dict[str, torch.Tensor]]
    router_state_dicts: Optional[Dict[int, Dict[str, torch.Tensor]]]
    current_round: int = 0
    round_history: List[Dict[str, Any]] = field(default_factory=list)


def initialize_prototype_federated_state(
    config: ExperimentConfig,
    k_experts: int,
) -> PrototypeFederatedState:
    if int(k_experts) <= 0:
        raise ValueError("k_experts must be positive.")

    model_bundle: ModelBundle = build_model_bundle(config)
    model = model_bundle.model
    add_expert_adapters_if_needed(model, int(k_experts))
    base_state = get_adapter_state(model, "default")

    server_expert_sd: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id in range(int(k_experts)):
        adapter_name = adapter_name_for_cluster(expert_id)
        server_expert_sd[expert_id] = remap_adapter_state(
            base_state,
            source_adapter_name="default",
            target_adapter_name=adapter_name,
        )

    if int(k_experts) == 1:
        router_state_dicts = None
    else:
        device = next(model.parameters()).device
        routers = build_router_dict(config, int(k_experts), model_bundle.d_model, device)
        router_state_dicts = dump_router_states(routers)

    del model_bundle
    return PrototypeFederatedState(
        k_experts=int(k_experts),
        server_expert_sd=server_expert_sd,
        router_state_dicts=router_state_dicts,
        current_round=0,
        round_history=[],
    )


def aggregate_weighted_state_dicts(
    base_state: Dict[str, torch.Tensor],
    deltas: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
) -> Dict[str, torch.Tensor]:
    if len(deltas) != len(weights):
        raise ValueError("deltas and weights must have the same length.")
    if not deltas:
        return clone_state_dict_cpu(base_state)

    total_weight = float(sum(float(weight) for weight in weights))
    if total_weight <= 0:
        normalized = [1.0 / float(len(weights)) for _ in weights]
    else:
        normalized = [float(weight) / total_weight for weight in weights]

    aggregated = {key: value.detach().cpu().float().clone() for key, value in base_state.items()}
    for key in aggregated:
        aggregated[key].zero_()

    for idx, delta in enumerate(deltas):
        alpha = float(normalized[idx])
        for key, value in delta.items():
            aggregated[key].add_(value.detach().cpu().float(), alpha=alpha)

    return {
        key: base_state[key].detach().cpu().float() + aggregated[key]
        for key in base_state.keys()
    }


def aggregate_expert_and_router_updates(
    *,
    base_expert_state: Dict[str, torch.Tensor],
    expert_deltas: Sequence[Dict[str, torch.Tensor]],
    client_weights: Sequence[float],
    base_router_state: Optional[Dict[str, torch.Tensor]] = None,
    router_deltas: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
    updated_expert = aggregate_weighted_state_dicts(base_expert_state, expert_deltas, client_weights)
    if base_router_state is None:
        return updated_expert, None
    if router_deltas is None:
        raise ValueError("router_deltas must be provided when base_router_state is not None.")
    updated_router = aggregate_weighted_state_dicts(base_router_state, router_deltas, client_weights)
    return updated_expert, updated_router
