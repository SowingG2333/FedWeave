from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from config import ExperimentConfig
import engine as core


Example = core.Example
RouterNet = core.RouterNet

adapter_name_for_cluster = core.adapter_name_for_cluster
add_expert_adapters_if_needed = core.add_expert_adapters_if_needed
build_optimizer = core.build_optimizer
build_split_lr_optimizer = core.build_split_lr_optimizer
build_local_scheduler = core.build_local_scheduler
collate_fn_builder = core.collate_fn_builder
compute_batch_expert_weights = core.compute_batch_expert_weights
compute_ce_loss = core.compute_ce_loss
compute_example_expert_weights = core.compute_example_expert_weights
compute_mixture_logits = core.compute_mixture_logits
compute_nll_loss = core.compute_nll_loss
_per_example_ce_loss = core._per_example_ce_loss
_per_example_nll_loss = core._per_example_nll_loss
extract_lora_a_vector = core.extract_lora_a_vector
extract_lora_ab_vector = core.extract_lora_ab_vector
extract_lora_b_vector = core.extract_lora_b_vector
get_adapter_state = core.get_adapter_state
greedy_generate = core.greedy_generate
greedy_generate_batch = core.greedy_generate_batch
greedy_generate_mixture = core.greedy_generate_mixture
load_adapter_state = core.load_adapter_state
load_all_expert_states = core.load_all_expert_states
make_dataloader_generator = core.make_dataloader_generator
move_batch_to_device = core.move_batch_to_device
remap_adapter_state = core.remap_adapter_state
set_trainable_adapter = core.set_trainable_adapter
set_trainable_router = core.set_trainable_router


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    d_model: int


def build_model_bundle(config: ExperimentConfig) -> ModelBundle:
    core.set_all_seeds(config.seed)
    core.configure_hf_environment(
        hf_offline=config.data.hf_offline,
        hf_download_timeout=config.data.hf_download_timeout,
        hf_etag_timeout=config.data.hf_etag_timeout,
    )
    core.set_system_prompt(config.model.system_prompt)
    core.set_prompt_format(config.model.prompt_format)
    model, tokenizer, d_model = core.build_model_and_tokenizer(
        model_name=config.model.model_name,
        dtype=core.torch_dtype_from_str(config.model.dtype),
        use_4bit=config.model.use_4bit,
        lora_r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        gradient_checkpointing=config.model.gradient_checkpointing,
    )
    return ModelBundle(model=model, tokenizer=tokenizer, d_model=int(d_model))


def build_router_dict(
    config: ExperimentConfig,
    k: int,
    d_model: int,
    device: torch.device,
    *,
    num_routers: Optional[int] = None,
    out_dim: Optional[int] = None,
) -> Dict[int, RouterNet]:
    routers: Dict[int, RouterNet] = {}
    n = int(k) if num_routers is None else int(num_routers)
    for gid in range(n):
        router = RouterNet(
            d_model=int(d_model),
            k_experts=int(k),
            hidden=int(config.router.hidden),
            dropout=float(config.router.dropout),
            out_dim=out_dim,
        ).to(device)
        router.train()
        router.float()
        routers[int(gid)] = router
    return routers


def clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


@torch.no_grad()
def load_router_states(
    routers: Dict[int, RouterNet],
    router_state_dicts: Optional[Dict[int, Dict[str, torch.Tensor]]],
    device: torch.device,
) -> None:
    if router_state_dicts is None:
        return
    for gid, router in routers.items():
        state_dict = router_state_dicts.get(int(gid))
        if state_dict is None:
            continue
        router.load_state_dict({key: value.to(device) for key, value in state_dict.items()})


@torch.no_grad()
def dump_router_states(routers: Dict[int, RouterNet]) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        int(gid): {key: value.detach().cpu().clone() for key, value in router.state_dict().items()}
        for gid, router in routers.items()
    }


@torch.no_grad()
def compute_fixed_weight_log_probs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    weights_k: torch.Tensor,
    expert_id_to_adapter: Dict[int, str],
) -> torch.Tensor:
    if weights_k.ndim != 2:
        raise ValueError("weights_k must be a [B, K] tensor.")
    active_experts = [eid for eid in range(weights_k.size(1)) if bool((weights_k[:, eid] > 1e-6).any().item())]
    if not active_experts:
        active_experts = [int(torch.argmax(weights_k[0]).item())]
    with core.layerwise_lora_routing_context(
        weights=weights_k,
        default_expert=int(active_experts[-1]),
        active_experts=active_experts,
    ):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    return F.log_softmax(out.logits.float(), dim=-1)


@torch.no_grad()
def compute_population_example_weights(
    model: Any,
    tokenizer: Any,
    routers: Optional[Dict[int, RouterNet]],
    group_priors: Sequence[float],
    ex: Example,
    k_experts: int,
    expert_id_to_adapter: Dict[int, str],
    max_length: int,
    m_select_eval: int,
    m_tau_eval: float,
) -> torch.Tensor:
    if routers is None or k_experts == 1:
        device = getattr(model, "device", None) or next(model.parameters()).device
        weights = torch.zeros((1, max(1, k_experts)), device=device, dtype=torch.float32)
        weights[:, 0] = 1.0
        return weights
    if len(group_priors) != k_experts:
        raise ValueError("group_priors length must match k_experts.")

    device = getattr(model, "device", None) or next(model.parameters()).device
    agg = torch.zeros((1, k_experts), device=device, dtype=torch.float32)
    for gid in range(k_experts):
        prior = float(group_priors[gid])
        if prior <= 0:
            continue
        weights_gid = core.compute_example_expert_weights(
            model=model,
            tokenizer=tokenizer,
            router=routers[gid],
            ex=ex,
            k_experts=k_experts,
            assigned_expert=gid,
            expert_id_to_adapter=expert_id_to_adapter,
            max_length=max_length,
            m_select_eval=m_select_eval,
            m_tau_eval=m_tau_eval,
        )
        agg.add_(weights_gid, alpha=prior)
    if float(agg.sum().item()) <= 0:
        agg[:, 0] = 1.0
    agg = agg / agg.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return agg
