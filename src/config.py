from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.paths import OUTPUTS_DIR, RESULTS_DIR


@dataclass
class DataConfig:
    tasks: List[str] = field(
        default_factory=lambda: [
            "text_editing",
            "math_reasoning",
            "sentiment_analysis",
            "commonsense_reasoning",
        ]
    )
    train_samples_per_task: int = 2000
    val_samples_per_task: int = 100
    test_samples_per_task: int = 400
    sample_seed: int = 42
    client_partition_seed: int = 42
    num_clients: int = 20
    dirichlet_alpha: float = 0.3
    arc_variant: str = "ARC-Challenge"
    hf_offline: bool = False
    hf_download_timeout: int = 180
    hf_etag_timeout: int = 30
    hf_max_retries: int = 5
    reuse_splits: bool = True
    data_cache_dir: str = ""


@dataclass
class ModelConfig:
    model_name: str = "meta-llama/Llama-3.2-3B"
    dtype: str = "bf16"
    use_4bit: bool = False
    gradient_checkpointing: bool = False
    system_prompt: str = ""
    prompt_format: str = "auto"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class RouterConfig:
    hidden: int = 512
    dropout: float = 0.0
    # 0 means adaptive top-M with M equal to the number of discovered experts.
    m_select: int = 0
    m_tau: float = 0.8
    route_ce_weight: float = 0.0


@dataclass
class TrainConfig:
    global_rounds: int = 20
    local_steps: int = 10
    batch_size: int = 4
    grad_accum: int = 1
    lr: float = 1e-4
    router_lr: float = 1e-4
    local_lr_schedule: str = "cosine"
    wd: float = 0.0
    grad_clip: float = 1.0
    max_length: int = 512
    save_every_n_rounds: int = 0
    eval_every_round: bool = True
    eval_every_n_rounds: int = 1
    val_compute_task_metrics: bool = True
    val_task_metrics_every_n_rounds: int = 1
    # 0 keeps the historical auto policy: eval batch size = train batch size * 2.
    eval_batch_size: int = 0
    eval_max_new_tokens: int = 32
    eval_max_new_tokens_by_task: Dict[str, int] = field(
        default_factory=lambda: {
            "text_editing": 64,
            "struct_to_text": 64,
            "summarization": 64,
            "math_reasoning": 128,
            "intent_detection": 8,
            "sentiment_analysis": 4,
            "commonsense_reasoning": 4,
        }
    )


@dataclass
class LoggingConfig:
    show_progress: bool = True
    use_swanlab: bool = False
    swanlab_mode: str = "cloud"
    swanlab_project: str = "FedWeave"
    swanlab_name: str = ""


@dataclass
class ExperimentConfig:
    out_dir: Path
    results_dir: Path
    seed: int
    data: DataConfig
    model: ModelConfig
    lora: LoRAConfig
    router: RouterConfig
    train: TrainConfig
    logging: LoggingConfig


def dataclass_to_dict(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [dataclass_to_dict(v) for v in value]
    if isinstance(value, dict):
        return {str(k): dataclass_to_dict(v) for k, v in value.items()}
    return value


def _default_output_dir() -> Path:
    return OUTPUTS_DIR / "runs"


def _default_results_dir() -> Path:
    return RESULTS_DIR / "runs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(_default_output_dir()))
    parser.add_argument("--results_dir", type=str, default=str(_default_results_dir()))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--tasks",
        type=str,
        default="text_editing,math_reasoning,sentiment_analysis,commonsense_reasoning",
    )
    parser.add_argument("--train_samples_per_task", type=int, default=2000)
    parser.add_argument("--val_samples_per_task", type=int, default=100)
    parser.add_argument("--test_samples_per_task", type=int, default=400)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--client_partition_seed", type=int, default=42)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--arc_variant", type=str, default="ARC-Challenge")
    parser.add_argument("--hf_offline", dest="hf_offline", action="store_true")
    parser.add_argument("--no_hf_offline", dest="hf_offline", action="store_false")
    parser.add_argument("--hf_download_timeout", type=int, default=180)
    parser.add_argument("--hf_etag_timeout", type=int, default=30)
    parser.add_argument("--hf_max_retries", type=int, default=5)
    parser.add_argument("--reuse_splits", dest="reuse_splits", action="store_true")
    parser.add_argument("--no_reuse_splits", dest="reuse_splits", action="store_false")
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="",
        help="Directory for reusable sampled splits and client partitions; defaults to out_dir/data.",
    )
    parser.set_defaults(hf_offline=False, reuse_splits=True)

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--use_4bit", dest="use_4bit", action="store_true")
    parser.add_argument("--no_use_4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
    )
    parser.add_argument("--prompt_format", type=str, default="auto", choices=["auto", "chat", "plain"])
    parser.set_defaults(use_4bit=False, gradient_checkpointing=False)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")

    parser.add_argument("--router_hidden", type=int, default=512)
    parser.add_argument("--router_dropout", type=float, default=0.0)
    parser.add_argument("--m_select", type=int, default=0)
    parser.add_argument("--m_tau", type=float, default=0.8)
    parser.add_argument(
        "--route_ce_weight",
        type=float,
        default=0.0,
        help="Weight for hard routing CE toward the assigned expert; only applies to absolute routing.",
    )

    parser.add_argument("--global_rounds", type=int, default=20)
    parser.add_argument("--local_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--router_lr", type=float, default=1e-4)
    parser.add_argument("--local_lr_schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--save_every_n_rounds",
        type=int,
        default=0,
        help="Archive the current best-val-loss checkpoint every N rounds; 0 disables periodic best checkpoints.",
    )
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--eval_max_new_tokens_by_task",
        type=str,
        default="text_editing:64,struct_to_text:64,summarization:64,math_reasoning:192,intent_detection:8,sentiment_analysis:4,commonsense_reasoning:4",
        help="Comma-separated task:max_new_tokens overrides for evaluation generation.",
    )
    parser.add_argument("--eval_every_round", dest="eval_every_round", action="store_true")
    parser.add_argument("--no_eval_every_round", dest="eval_every_round", action="store_false")
    parser.add_argument("--eval_every_n_rounds", type=int, default=1)
    parser.add_argument("--val_compute_task_metrics", dest="val_compute_task_metrics", action="store_true")
    parser.add_argument("--no_val_compute_task_metrics", dest="val_compute_task_metrics", action="store_false")
    parser.add_argument(
        "--val_task_metrics_every_n_rounds",
        type=int,
        default=1,
        help="Compute validation generation-based task metrics every N training rounds; val loss still follows eval_every_n_rounds.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=0,
        help="Batch size for val/test evaluation; 0 uses the legacy default of batch_size * 2.",
    )
    parser.set_defaults(eval_every_round=True, val_compute_task_metrics=True)

    parser.add_argument("--show_progress", dest="show_progress", action="store_true")
    parser.add_argument("--no_show_progress", dest="show_progress", action="store_false")
    parser.add_argument("--use_swanlab", dest="use_swanlab", action="store_true")
    parser.add_argument("--no_use_swanlab", dest="use_swanlab", action="store_false")
    parser.add_argument("--swanlab_mode", type=str, default="cloud")
    parser.add_argument("--swanlab_project", type=str, default="FedWeave")
    parser.add_argument("--swanlab_name", type=str, default="")
    parser.set_defaults(show_progress=True, use_swanlab=False)
    return parser


def _canonical_task_name(task_name: str) -> str:
    name = str(task_name).strip().lower()
    alias_map = {
        "coedit": "text_editing",
        "co-edit": "text_editing",
        "text_editing": "text_editing",
        "text-editing": "text_editing",
        "e2e": "struct_to_text",
        "e2e_nlg": "struct_to_text",
        "e2e-nlg": "struct_to_text",
        "struct_to_text": "struct_to_text",
        "struct-to-text": "struct_to_text",
        "xsum": "summarization",
        "xsum_summarization": "summarization",
        "xsum-summarization": "summarization",
        "summarization": "summarization",
        "summarisation": "summarization",
        "summary": "summarization",
        "gsm8k": "math_reasoning",
        "grade_school_math": "math_reasoning",
        "grade-school-math": "math_reasoning",
        "math": "math_reasoning",
        "math_reasoning": "math_reasoning",
        "math-reasoning": "math_reasoning",
        "banking77": "intent_detection",
        "banking_77": "intent_detection",
        "banking-77": "intent_detection",
        "intent": "intent_detection",
        "intent_detection": "intent_detection",
        "intent-detection": "intent_detection",
        "tweeteval_sentiment": "sentiment_analysis",
        "tweet_eval_sentiment": "sentiment_analysis",
        "tweet_eval": "sentiment_analysis",
        "tweet-eval": "sentiment_analysis",
        "tweeteval": "sentiment_analysis",
        "tweeteval-sentiment": "sentiment_analysis",
        "sentiment": "sentiment_analysis",
        "sentiment_analysis": "sentiment_analysis",
        "sentiment-analysis": "sentiment_analysis",
        "arc": "commonsense_reasoning",
        "arc_challenge": "commonsense_reasoning",
        "arc-challenge": "commonsense_reasoning",
        "commonsense_reasoning": "commonsense_reasoning",
        "commonsense-reasoning": "commonsense_reasoning",
    }
    return alias_map.get(name, name)


def parse_eval_max_new_tokens_by_task(value: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in str(value or "").split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError("--eval_max_new_tokens_by_task entries must use task:max_new_tokens format.")
        task_name, token_value = item.split(":", 1)
        task_name = _canonical_task_name(task_name)
        if not task_name:
            raise ValueError("--eval_max_new_tokens_by_task contains an empty task name.")
        try:
            token_budget = int(token_value)
        except ValueError as exc:
            raise ValueError(f"Invalid max_new_tokens for task '{task_name}': {token_value}") from exc
        if token_budget <= 0:
            raise ValueError(f"max_new_tokens for task '{task_name}' must be > 0.")
        out[task_name] = int(token_budget)
    return out


def _parse_task_list(value: str) -> List[str]:
    return [_canonical_task_name(task) for task in str(value).split(",") if task.strip()]


def _parse_target_modules(value: str) -> List[str]:
    return [name.strip() for name in str(value).split(",") if name.strip()]


def validate_common_config_args(args: argparse.Namespace) -> None:
    tasks = _parse_task_list(args.tasks)
    target_modules = _parse_target_modules(args.target_modules)
    if not tasks:
        raise ValueError("At least one task must be provided.")
    if not target_modules:
        raise ValueError("At least one LoRA target module must be provided.")
    if args.hf_download_timeout <= 0:
        raise ValueError("--hf_download_timeout must be > 0.")
    if args.hf_etag_timeout <= 0:
        raise ValueError("--hf_etag_timeout must be > 0.")
    if args.hf_max_retries < 0:
        raise ValueError("--hf_max_retries must be >= 0.")
    if args.route_ce_weight < 0:
        raise ValueError("--route_ce_weight must be >= 0.")
    if args.m_select < 0:
        raise ValueError("--m_select must be >= 0; use 0 for adaptive top-M with M = #experts.")
    if args.eval_every_n_rounds < 1:
        raise ValueError("--eval_every_n_rounds must be >= 1.")
    if args.val_task_metrics_every_n_rounds < 1:
        raise ValueError("--val_task_metrics_every_n_rounds must be >= 1.")
    if args.eval_batch_size < 0:
        raise ValueError("--eval_batch_size must be >= 0; use 0 for automatic sizing.")
    if args.save_every_n_rounds < 0:
        raise ValueError("--save_every_n_rounds must be >= 0.")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0.")
    if args.router_lr <= 0:
        raise ValueError("--router_lr must be > 0.")
    if args.max_length <= 0:
        raise ValueError("--max_length must be > 0.")
    if args.max_length > 512:
        raise ValueError("--max_length must be <= 512 for the supported dataset setup.")


def build_config_from_namespace(
    args: argparse.Namespace,
    *,
    out_dir: str | Path,
    results_dir: str | Path,
) -> ExperimentConfig:
    validate_common_config_args(args)
    tasks = _parse_task_list(args.tasks)
    target_modules = _parse_target_modules(args.target_modules)
    eval_max_new_tokens_by_task = parse_eval_max_new_tokens_by_task(args.eval_max_new_tokens_by_task)

    data = DataConfig(
        tasks=tasks,
        train_samples_per_task=args.train_samples_per_task,
        val_samples_per_task=args.val_samples_per_task,
        test_samples_per_task=args.test_samples_per_task,
        sample_seed=args.sample_seed,
        client_partition_seed=args.client_partition_seed,
        num_clients=args.num_clients,
        dirichlet_alpha=args.dirichlet_alpha,
        arc_variant=args.arc_variant,
        hf_offline=args.hf_offline,
        hf_download_timeout=args.hf_download_timeout,
        hf_etag_timeout=args.hf_etag_timeout,
        hf_max_retries=args.hf_max_retries,
        reuse_splits=args.reuse_splits,
        data_cache_dir=args.data_cache_dir,
    )
    model = ModelConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
        system_prompt=args.system_prompt,
        prompt_format=args.prompt_format,
    )
    lora = LoRAConfig(
        rank=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    router = RouterConfig(
        hidden=args.router_hidden,
        dropout=args.router_dropout,
        m_select=args.m_select,
        m_tau=args.m_tau,
        route_ce_weight=args.route_ce_weight,
    )
    train = TrainConfig(
        global_rounds=args.global_rounds,
        local_steps=args.local_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        router_lr=args.router_lr,
        local_lr_schedule=args.local_lr_schedule,
        wd=args.wd,
        grad_clip=args.grad_clip,
        max_length=args.max_length,
        save_every_n_rounds=args.save_every_n_rounds,
        eval_every_round=args.eval_every_round,
        eval_every_n_rounds=args.eval_every_n_rounds,
        val_compute_task_metrics=args.val_compute_task_metrics,
        val_task_metrics_every_n_rounds=args.val_task_metrics_every_n_rounds,
        eval_batch_size=args.eval_batch_size,
        eval_max_new_tokens=args.eval_max_new_tokens,
        eval_max_new_tokens_by_task=eval_max_new_tokens_by_task,
    )
    logging = LoggingConfig(
        show_progress=args.show_progress,
        use_swanlab=args.use_swanlab,
        swanlab_mode=args.swanlab_mode,
        swanlab_project=args.swanlab_project,
        swanlab_name=args.swanlab_name,
    )
    return ExperimentConfig(
        out_dir=Path(out_dir),
        results_dir=Path(results_dir),
        seed=args.seed,
        data=data,
        model=model,
        lora=lora,
        router=router,
        train=train,
        logging=logging,
    )


def parse_config(argv: Optional[List[str]] = None) -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return build_config_from_namespace(
        args,
        out_dir=args.out_dir,
        results_dir=args.results_dir,
    )


def _dc_defaults(dc_type: type) -> Dict[str, Any]:
    """Return {field_name: default_value} for a dataclass without constructing an instance."""
    defaults: Dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def _populate_dc(dc_type: type, payload: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Populate a dataclass from a dict using field introspection for type-safe conversion."""
    defaults = _dc_defaults(dc_type)
    overrides = overrides or {}
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(dc_type):
        if f.name in overrides:
            kwargs[f.name] = overrides[f.name]
            continue
        raw = payload.get(f.name)
        default = defaults.get(f.name)
        if raw is None:
            kwargs[f.name] = default
        elif isinstance(default, Path):
            kwargs[f.name] = Path(raw)
        elif isinstance(default, bool):
            kwargs[f.name] = bool(raw)
        elif isinstance(default, int):
            kwargs[f.name] = int(raw)
        elif isinstance(default, float):
            kwargs[f.name] = float(raw)
        elif isinstance(default, str):
            kwargs[f.name] = str(raw)
        elif isinstance(default, list):
            elem_type = type(default[0]) if default else str
            kwargs[f.name] = [elem_type(v) for v in raw]
        elif isinstance(default, dict):
            if default:
                key_type = type(next(iter(default.keys())))
                val_type = type(next(iter(default.values())))
                kwargs[f.name] = {key_type(k): val_type(v) for k, v in raw.items()}
            else:
                kwargs[f.name] = dict(raw)
        else:
            kwargs[f.name] = raw
    return dc_type(**kwargs)


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return dataclass_to_dict(config)


def config_from_dict(payload: Dict[str, Any]) -> ExperimentConfig:
    data_payload = dict(payload.get("data", {}))
    train_payload = dict(payload.get("train", {}))

    data_overrides: Dict[str, Any] = {}
    if "tasks" in data_payload:
        data_overrides["tasks"] = [_canonical_task_name(t) for t in data_payload["tasks"]]

    train_overrides: Dict[str, Any] = {}
    if "eval_max_new_tokens_by_task" in train_payload:
        train_overrides["eval_max_new_tokens_by_task"] = {
            _canonical_task_name(k): int(v)
            for k, v in train_payload["eval_max_new_tokens_by_task"].items()
        }

    return ExperimentConfig(
        out_dir=Path(payload.get("out_dir", _default_output_dir())),
        results_dir=Path(payload.get("results_dir", _default_results_dir())),
        seed=int(payload.get("seed", 42)),
        data=_populate_dc(DataConfig, data_payload, data_overrides),
        model=_populate_dc(ModelConfig, dict(payload.get("model", {}))),
        lora=_populate_dc(LoRAConfig, dict(payload.get("lora", {}))),
        router=_populate_dc(RouterConfig, dict(payload.get("router", {}))),
        train=_populate_dc(TrainConfig, train_payload, train_overrides),
        logging=_populate_dc(LoggingConfig, dict(payload.get("logging", {}))),
    )
