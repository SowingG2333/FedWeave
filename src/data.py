from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import DownloadConfig, load_dataset
from transformers import AutoTokenizer

from config import ExperimentConfig
import engine as core
from src.utils.io import dump_json, ensure_dir, load_json


DATASET_GROUPS = {
    "text_editing": {
        "paper_name": "CoEdIT",
        "hf_id": "grammarly/coedit",
        "config": None,
        "splits": ("train", "validation"),
    },
    "struct_to_text": {
        "paper_name": "E2E NLG",
        "hf_id": "GEM/e2e_nlg",
        "config": None,
        "splits": ("train", "validation", "test"),
    },
    "intent_detection": {
        "paper_name": "BANKING77",
        "hf_id": "PolyAI/banking77",
        "config": None,
        "splits": ("train", "test"),
    },
    "summarization": {
        "paper_name": "XSum",
        "hf_id": "EdinburghNLP/xsum",
        "config": None,
        "splits": ("train", "validation", "test"),
    },
    "math_reasoning": {
        "paper_name": "GSM8K",
        "hf_id": "openai/gsm8k",
        "config": "main",
        "splits": ("train", "test"),
    },
    "sentiment_analysis": {
        "paper_name": "TweetEval-Sentiment",
        "hf_id": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "splits": ("train", "validation", "test"),
    },
    "commonsense_reasoning": {
        "paper_name": "ARC-Challenge",
        "hf_id": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "splits": ("train", "validation", "test"),
    },
}


@dataclass
class ClientSpec:
    client_id: int
    train_examples: List[core.Example]
    task_mix: Dict[str, float]
    dominant_task: str
    n_train: int


@dataclass
class PreparedData:
    train_by_task: Dict[str, List[core.Example]]
    global_val_by_task: Dict[str, List[core.Example]]
    global_test_by_task: Dict[str, List[core.Example]]
    clients: List[ClientSpec]
    token_stats: Dict[str, Any]
    split_index_path: Path
    client_partition_path: Path

    @property
    def global_val_examples(self) -> List[core.Example]:
        merged: List[core.Example] = []
        for task_examples in self.global_val_by_task.values():
            merged.extend(task_examples)
        return merged

    @property
    def global_test_examples(self) -> List[core.Example]:
        merged: List[core.Example] = []
        for task_examples in self.global_test_by_task.values():
            merged.extend(task_examples)
        return merged


def _sampling_dir(config: ExperimentConfig) -> Path:
    data_cache_dir = str(getattr(config.data, "data_cache_dir", "") or "").strip()
    if data_cache_dir:
        return ensure_dir(Path(data_cache_dir).expanduser())
    return ensure_dir(config.out_dir / "data")


def split_index_path(config: ExperimentConfig) -> Path:
    task_tag = "-".join(config.data.tasks)
    arc_tag = str(config.data.arc_variant).replace("/", "-")
    return _sampling_dir(config) / f"sampled_splits_{task_tag}_{arc_tag}_seed{config.data.sample_seed}.json"


def client_partition_path(config: ExperimentConfig) -> Path:
    task_tag = "-".join(config.data.tasks)
    alpha_tag = str(config.data.dirichlet_alpha).replace(".", "p")
    return _sampling_dir(config) / f"client_partition_{task_tag}_alpha{alpha_tag}_seed{config.data.client_partition_seed}.json"


def _task_source_spec(task: str, arc_variant: str) -> Tuple[str, str | None, Sequence[str]]:
    task = task.lower()
    if task == "commonsense_reasoning":
        spec = dict(DATASET_GROUPS[task])
        spec["config"] = arc_variant
    else:
        spec = DATASET_GROUPS.get(task)
    if spec is None:
        raise ValueError(f"Unsupported task: {task}")
    return str(spec["hf_id"]), spec["config"], tuple(spec["splits"])


def _load_source_splits(task: str, arc_variant: str, hf_max_retries: int) -> Dict[str, Any]:
    dataset_name, config_name, splits = _task_source_spec(task, arc_variant)
    loaded: Dict[str, Any] = {}
    download_config = DownloadConfig()
    if hasattr(download_config, "max_retries"):
        download_config.max_retries = max(0, int(hf_max_retries))
    for split_name in splits:
        try:
            if config_name is None:
                loaded[split_name] = load_dataset(
                    dataset_name,
                    trust_remote_code=True,
                    split=split_name,
                    download_config=download_config,
                )
            else:
                loaded[split_name] = load_dataset(
                    dataset_name,
                    config_name,
                    trust_remote_code=True,
                    split=split_name,
                    download_config=download_config,
                )
        except Exception as exc:
            source_name = dataset_name if config_name is None else f"{dataset_name}/{config_name}"
            raise RuntimeError(
                f"Failed to download/load Hugging Face dataset '{source_name}' split '{split_name}' for task '{task}'. "
                "This usually means the remote machine cannot reach Hugging Face reliably. "
                "If the dataset is already cached, rerun with `--hf_offline`. "
                "Otherwise configure a permitted dataset mirror, "
                "or increase `--hf_download_timeout` / `--hf_etag_timeout` / `--hf_max_retries`."
            ) from exc
    return loaded


def _build_serialized_example(
    task: str,
    source_split: str,
    source_index: int,
    item: Dict[str, Any],
    label_names: Sequence[str] | None = None,
) -> Dict[str, Any]:
    if label_names is not None:
        item = dict(item)
        item["_label_names"] = list(label_names)
    example = core.build_prompt_target(task, item)
    return {
        "task": task,
        "source_split": source_split,
        "source_index": int(source_index),
        "prompt": example.prompt,
        "target": example.target,
        "meta": example.meta,
    }


def _deserialize_example(record: Dict[str, Any]) -> core.Example:
    return core.Example(
        task=str(record["task"]),
        prompt=str(record["prompt"]),
        target=str(record["target"]),
        meta=dict(record.get("meta", {})),
    )


def _compute_length_summary(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "max": 0, "mean": 0.0, "p95": 0}
    ordered = sorted(int(v) for v in lengths)
    count = len(ordered)
    p95_index = max(0, min(count - 1, math.ceil(0.95 * count) - 1))
    return {
        "count": int(count),
        "max": int(ordered[-1]),
        "mean": float(sum(ordered) / count),
        "p95": int(ordered[p95_index]),
    }


def _compute_token_stats(
    tokenizer: AutoTokenizer,
    serialized_payload: Dict[str, Any],
) -> Dict[str, Any]:
    task_stats: Dict[str, Any] = {}
    all_prompt_lengths: List[int] = []
    all_full_lengths: List[int] = []
    for task, task_payload in serialized_payload["tasks"].items():
        split_stats: Dict[str, Any] = {}
        for split_name in ("train", "val", "test"):
            prompt_lengths: List[int] = []
            full_lengths: List[int] = []
            for record in task_payload[split_name]:
                ex = _deserialize_example(record)
                prompt_text = core.render_chat_text(tokenizer, ex.prompt, add_generation_prompt=True)
                full_text = core.render_chat_text(tokenizer, ex.prompt, assistant_text=ex.target)
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                full_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
                prompt_lengths.append(int(prompt_len))
                full_lengths.append(int(full_len))
                all_prompt_lengths.append(int(prompt_len))
                all_full_lengths.append(int(full_len))
            split_stats[split_name] = {
                "prompt_tokens": _compute_length_summary(prompt_lengths),
                "full_tokens": _compute_length_summary(full_lengths),
            }
        task_stats[task] = split_stats
    return {
        "tasks": task_stats,
        "global": {
            "prompt_tokens": _compute_length_summary(all_prompt_lengths),
            "full_tokens": _compute_length_summary(all_full_lengths),
        },
    }


def _serialize_sampled_splits(config: ExperimentConfig) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "meta": {
            "tasks": list(config.data.tasks),
            "sample_seed": int(config.data.sample_seed),
            "arc_variant": config.data.arc_variant,
            "dataset_groups": {
                task: {
                    "paper_name": DATASET_GROUPS[task]["paper_name"],
                    "hf_id": DATASET_GROUPS[task]["hf_id"],
                    "config": DATASET_GROUPS[task]["config"] if task != "commonsense_reasoning" else config.data.arc_variant,
                }
                for task in config.data.tasks
            },
            "train_samples_per_task": int(config.data.train_samples_per_task),
            "val_samples_per_task": int(config.data.val_samples_per_task),
            "test_samples_per_task": int(config.data.test_samples_per_task),
            "tokenizer_name": config.model.model_name,
            "system_prompt": config.model.system_prompt,
            "prompt_format": config.model.prompt_format,
            "prompt_template_version": int(core.PROMPT_TEMPLATE_VERSION),
        },
        "tasks": {},
    }
    for task_idx, task in enumerate(config.data.tasks):
        split_datasets = _load_source_splits(
            task,
            config.data.arc_variant,
            config.data.hf_max_retries,
        )
        refs: List[Tuple[str, int]] = []
        for split_name, split_dataset in split_datasets.items():
            refs.extend((split_name, int(idx)) for idx in range(len(split_dataset)))
        required_total = (
            int(config.data.train_samples_per_task)
            + int(config.data.val_samples_per_task)
            + int(config.data.test_samples_per_task)
        )
        if len(refs) < required_total:
            raise ValueError(
                f"Task '{task}' only has {len(refs)} labeled examples across available splits, "
                f"which is fewer than the requested {required_total}."
            )
        rng = np.random.default_rng(config.data.sample_seed + 1_000 * (task_idx + 1))
        all_indices = rng.permutation(len(refs)).tolist()

        train_count = int(config.data.train_samples_per_task)
        val_count = int(config.data.val_samples_per_task)
        test_count = int(config.data.test_samples_per_task)

        train_ids = all_indices[:train_count]
        remaining_ids = all_indices[train_count:]
        if len(remaining_ids) < val_count + test_count:
            raise ValueError(
                f"Task '{task}' only has {len(refs)} labeled examples across available splits, "
                f"which is fewer than the requested train/val/test budget after reserving the train split."
            )
        val_ids = remaining_ids[:val_count]
        test_ids = remaining_ids[val_count : val_count + test_count]

        buckets = {
            "train": [refs[int(i)] for i in train_ids],
            "val": [refs[int(i)] for i in val_ids],
            "test": [refs[int(i)] for i in test_ids],
        }
        serialized_task: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for split_name, chosen_refs in buckets.items():
            for source_split, source_index in chosen_refs:
                split_dataset = split_datasets[source_split]
                item = split_dataset[int(source_index)]
                label_names = None
                if task == "intent_detection":
                    label_feature = getattr(split_dataset, "features", {}).get("label")
                    label_names = getattr(label_feature, "names", None)
                serialized_task[split_name].append(
                    _build_serialized_example(task, source_split, int(source_index), item, label_names=label_names)
                )
        payload["tasks"][task] = serialized_task
    return payload


def _load_or_create_split_payload(config: ExperimentConfig) -> Tuple[Dict[str, Any], Path]:
    path = split_index_path(config)
    if config.data.reuse_splits and path.exists():
        payload = load_json(path)
        meta = payload.get("meta", {})
        same_sampling = (
            list(meta.get("tasks", [])) == list(config.data.tasks)
            and int(meta.get("sample_seed", -1)) == int(config.data.sample_seed)
            and int(meta.get("train_samples_per_task", -1)) == int(config.data.train_samples_per_task)
            and int(meta.get("val_samples_per_task", -1)) == int(config.data.val_samples_per_task)
            and int(meta.get("test_samples_per_task", -1)) == int(config.data.test_samples_per_task)
            and str(meta.get("arc_variant", "")) == str(config.data.arc_variant)
            and int(meta.get("prompt_template_version", -1)) == int(core.PROMPT_TEMPLATE_VERSION)
        )
        if same_sampling:
            prompting_compatible = (
                str(meta.get("tokenizer_name", "")) == str(config.model.model_name)
                and str(meta.get("system_prompt", "")) == str(config.model.system_prompt)
                and str(meta.get("prompt_format", "auto")) == str(config.model.prompt_format)
                and isinstance(payload.get("token_stats"), dict)
            )
            if prompting_compatible:
                return payload, path
            core.configure_hf_environment(
                hf_offline=config.data.hf_offline,
                hf_download_timeout=config.data.hf_download_timeout,
                hf_etag_timeout=config.data.hf_etag_timeout,
            )
            core.set_system_prompt(config.model.system_prompt)
            core.set_prompt_format(config.model.prompt_format)
            tokenizer = core.load_tokenizer_prefer_local(config.model.model_name, use_fast=True)
            payload["meta"]["tokenizer_name"] = config.model.model_name
            payload["meta"]["system_prompt"] = config.model.system_prompt
            payload["meta"]["prompt_format"] = config.model.prompt_format
            payload["meta"]["prompt_template_version"] = int(core.PROMPT_TEMPLATE_VERSION)
            payload["token_stats"] = _compute_token_stats(tokenizer, payload)
            dump_json(path, payload)
            return payload, path

    core.configure_hf_environment(
        hf_offline=config.data.hf_offline,
        hf_download_timeout=config.data.hf_download_timeout,
        hf_etag_timeout=config.data.hf_etag_timeout,
    )
    core.set_system_prompt(config.model.system_prompt)
    core.set_prompt_format(config.model.prompt_format)
    tokenizer = core.load_tokenizer_prefer_local(config.model.model_name, use_fast=True)

    payload = _serialize_sampled_splits(config)
    payload["token_stats"] = _compute_token_stats(tokenizer, payload)
    dump_json(path, payload)
    return payload, path


def _task_examples_from_payload(payload: Dict[str, Any], split_name: str) -> Dict[str, List[core.Example]]:
    out: Dict[str, List[core.Example]] = {}
    for task, task_payload in payload["tasks"].items():
        out[task] = [_deserialize_example(record) for record in task_payload[split_name]]
    return out


def _rebalance_empty_clients(client_examples: List[List[core.Example]], rng: np.random.Generator) -> None:
    while True:
        empty_ids = [idx for idx, items in enumerate(client_examples) if len(items) == 0]
        if not empty_ids:
            return
        donor_id = max(range(len(client_examples)), key=lambda idx: len(client_examples[idx]))
        if len(client_examples[donor_id]) <= 1:
            raise RuntimeError("Unable to rebalance empty clients because no donor client has spare examples.")
        rng.shuffle(client_examples[donor_id])
        example = client_examples[donor_id].pop()
        receiver_id = empty_ids[0]
        client_examples[receiver_id].append(example)


def _client_from_examples(client_id: int, examples: List[core.Example], tasks: Sequence[str]) -> ClientSpec:
    counts = {task: 0 for task in tasks}
    for ex in examples:
        counts[ex.task] += 1
    total = max(1, len(examples))
    task_mix = {task: float(counts[task] / total) for task in tasks}
    dominant_task = max(task_mix.items(), key=lambda item: item[1])[0]
    return ClientSpec(
        client_id=int(client_id),
        train_examples=examples,
        task_mix=task_mix,
        dominant_task=dominant_task,
        n_train=len(examples),
    )


def _client_partition_meta(
    config: ExperimentConfig,
    train_by_task: Dict[str, List[core.Example]],
) -> Dict[str, Any]:
    return {
        "version": 1,
        "tasks": list(config.data.tasks),
        "sample_seed": int(config.data.sample_seed),
        "client_partition_seed": int(config.data.client_partition_seed),
        "alpha": float(config.data.dirichlet_alpha),
        "num_clients": int(config.data.num_clients),
        "arc_variant": str(config.data.arc_variant),
        "train_counts_by_task": {
            str(task): int(len(train_by_task.get(task, []))) for task in config.data.tasks
        },
        "prompt_template_version": int(core.PROMPT_TEMPLATE_VERSION),
    }


def _client_partition_payload_matches(
    payload: Dict[str, Any],
    config: ExperimentConfig,
    train_by_task: Dict[str, List[core.Example]],
) -> bool:
    meta = payload.get("meta", {})
    expected = _client_partition_meta(config, train_by_task)
    if int(meta.get("version", -1)) != int(expected["version"]):
        return False
    if list(meta.get("tasks", [])) != list(expected["tasks"]):
        return False
    if int(meta.get("sample_seed", -1)) != int(expected["sample_seed"]):
        return False
    if int(meta.get("client_partition_seed", -1)) != int(expected["client_partition_seed"]):
        return False
    if int(meta.get("num_clients", -1)) != int(expected["num_clients"]):
        return False
    if str(meta.get("arc_variant", "")) != str(expected["arc_variant"]):
        return False
    if int(meta.get("prompt_template_version", -1)) != int(expected["prompt_template_version"]):
        return False
    if dict(meta.get("train_counts_by_task", {})) != dict(expected["train_counts_by_task"]):
        return False
    if not np.isclose(float(meta.get("alpha", -1.0)), float(expected["alpha"])):
        return False
    clients = payload.get("clients", [])
    return isinstance(clients, list) and len(clients) == int(config.data.num_clients)


def _load_clients_from_partition_payload(payload: Dict[str, Any], config: ExperimentConfig) -> List[ClientSpec]:
    clients: List[ClientSpec] = []
    for row in payload.get("clients", []):
        client_id = int(row["client_id"])
        examples = [_deserialize_example(record) for record in row.get("examples", [])]
        clients.append(_client_from_examples(client_id, examples, config.data.tasks))
    clients.sort(key=lambda client: int(client.client_id))
    return clients


def _build_client_partition_payload(
    config: ExperimentConfig,
    train_by_task: Dict[str, List[core.Example]],
    clients: Sequence[ClientSpec],
    task_vectors: np.ndarray,
) -> Dict[str, Any]:
    serialized_clients: List[Dict[str, Any]] = []
    for client in clients:
        serialized_clients.append(
            {
                "client_id": int(client.client_id),
                "n_train": int(client.n_train),
                "dominant_task": client.dominant_task,
                "task_mix": client.task_mix,
                "dirichlet_vector": {
                    task: float(task_vectors[client.client_id, task_idx])
                    for task_idx, task in enumerate(config.data.tasks)
                },
                "examples": [
                    {
                        "task": ex.task,
                        "prompt": ex.prompt,
                        "target": ex.target,
                    }
                    for ex in client.train_examples
                ],
            }
        )
    return {
        "meta": _client_partition_meta(config, train_by_task),
        "clients": serialized_clients,
    }


def _allocate_task_examples_to_clients(
    task_examples: Dict[str, List[core.Example]],
    tasks: Sequence[str],
    num_clients: int,
    alpha: float,
    seed: int,
) -> Tuple[List[List[core.Example]], np.ndarray]:
    rng = np.random.default_rng(seed)
    task_vectors = rng.dirichlet([float(alpha)] * len(tasks), size=int(num_clients))
    client_examples: List[List[core.Example]] = [[] for _ in range(num_clients)]
    for task_idx, task in enumerate(tasks):
        examples = list(task_examples.get(task, []))
        rng.shuffle(examples)
        probs = task_vectors[:, task_idx].astype(np.float64)
        probs = probs / probs.sum()
        alloc = rng.multinomial(len(examples), probs).tolist()
        offset = 0
        for client_id, count in enumerate(alloc):
            if count <= 0:
                continue
            client_examples[client_id].extend(examples[offset : offset + count])
            offset += count
    for client_id in range(num_clients):
        rng.shuffle(client_examples[client_id])
    _rebalance_empty_clients(client_examples, rng)
    return client_examples, task_vectors


def load_client_task_vectors(
    partition_path: Path,
    tasks: Sequence[str],
    num_clients: int,
) -> np.ndarray:
    payload = load_json(partition_path)
    clients = payload.get("clients", [])
    if not isinstance(clients, list) or len(clients) != int(num_clients):
        raise ValueError(
            f"Client partition payload at {partition_path} does not match num_clients={int(num_clients)}."
        )
    task_vectors = np.zeros((int(num_clients), len(tasks)), dtype=np.float64)
    for row in clients:
        client_id = int(row["client_id"])
        raw_vector = row.get("dirichlet_vector") or row.get("task_mix") or {}
        if not isinstance(raw_vector, dict):
            raise ValueError(f"Client {client_id} in {partition_path} is missing a valid task vector.")
        vector = np.asarray([float(raw_vector.get(str(task), 0.0)) for task in tasks], dtype=np.float64)
        if float(vector.sum()) <= 0.0:
            vector[:] = 1.0 / max(1, len(tasks))
        else:
            vector /= float(vector.sum())
        task_vectors[client_id] = vector
    return task_vectors


def partition_examples_by_client(
    *,
    split_by_task: Dict[str, List[core.Example]],
    tasks: Sequence[str],
    num_clients: int,
    partition_path: Path,
    seed: int,
) -> List[List[core.Example]]:
    task_vectors = load_client_task_vectors(partition_path, tasks, num_clients)
    rng = np.random.default_rng(int(seed))
    client_examples: List[List[core.Example]] = [[] for _ in range(int(num_clients))]
    for task_idx, task in enumerate(tasks):
        examples = list(split_by_task.get(str(task), []))
        if not examples:
            continue
        rng.shuffle(examples)
        probs = task_vectors[:, task_idx].astype(np.float64)
        if float(probs.sum()) <= 0.0:
            probs[:] = 1.0 / max(1, int(num_clients))
        else:
            probs /= float(probs.sum())
        alloc = rng.multinomial(len(examples), probs).tolist()
        offset = 0
        for client_id, count in enumerate(alloc):
            if count <= 0:
                continue
            client_examples[client_id].extend(examples[offset : offset + count])
            offset += count
    for client_id in range(int(num_clients)):
        rng.shuffle(client_examples[client_id])
    return client_examples


def _build_clients(
    config: ExperimentConfig,
    train_by_task: Dict[str, List[core.Example]],
) -> Tuple[List[ClientSpec], Path]:
    path = client_partition_path(config)
    if config.data.reuse_splits and path.exists():
        payload = load_json(path)
        if _client_partition_payload_matches(payload, config, train_by_task):
            return _load_clients_from_partition_payload(payload, config), path

    client_examples, task_vectors = _allocate_task_examples_to_clients(
        task_examples=train_by_task,
        tasks=config.data.tasks,
        num_clients=config.data.num_clients,
        alpha=config.data.dirichlet_alpha,
        seed=config.data.client_partition_seed,
    )

    clients: List[ClientSpec] = []
    for client_id, examples in enumerate(client_examples):
        clients.append(_client_from_examples(client_id, examples, config.data.tasks))
    dump_json(path, _build_client_partition_payload(config, train_by_task, clients, task_vectors))
    return clients, path


def prepare_data(config: ExperimentConfig) -> PreparedData:
    core.configure_hf_environment(
        hf_offline=config.data.hf_offline,
        hf_download_timeout=config.data.hf_download_timeout,
        hf_etag_timeout=config.data.hf_etag_timeout,
    )
    core.set_system_prompt(config.model.system_prompt)
    core.set_prompt_format(config.model.prompt_format)
    payload, split_path = _load_or_create_split_payload(config)
    train_by_task = _task_examples_from_payload(payload, "train")
    global_val_by_task = _task_examples_from_payload(payload, "val")
    global_test_by_task = _task_examples_from_payload(payload, "test")
    clients, partition_path = _build_clients(config, train_by_task)
    return PreparedData(
        train_by_task=train_by_task,
        global_val_by_task=global_val_by_task,
        global_test_by_task=global_test_by_task,
        clients=clients,
        token_stats=payload.get("token_stats", {}),
        split_index_path=split_path,
        client_partition_path=partition_path,
    )
