from __future__ import annotations

import dataclasses
import gc
import os
import random
import re
from contextlib import contextmanager
from types import MethodType

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed as hf_set_seed


SYSTEM_PROMPT = ""
PROMPT_FORMAT = "auto"
PROMPT_TEMPLATE_VERSION = 3
_CHAT_RENDER_CACHE: Dict[Tuple[str, str, str, str, Optional[str], bool], str] = {}
_TOKEN_ID_CACHE: Dict[Tuple[str, int, bool, str], List[int]] = {}
_LAYERWISE_LORA_STATE: Dict[str, Any] = {
    "weights": None,
    "disabled": False,
    "default_expert": 0,
    "active_experts": None,
}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


_COMPILE_ENABLED = hasattr(torch, "compile")


def _maybe_compile(fn):
    if not _COMPILE_ENABLED:
        return fn
    try:
        return torch.compile(fn)
    except Exception:
        return fn


def configure_torch_runtime() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True


def maybe_enable_hf_offline(hf_offline: bool) -> None:
    if not hf_offline:
        return
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def configure_hf_environment(
    hf_offline: bool,
    hf_download_timeout: int,
    hf_etag_timeout: int,
) -> None:
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(max(1, int(hf_download_timeout)))
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(max(1, int(hf_etag_timeout)))
    maybe_enable_hf_offline(hf_offline)


def _prefer_local_hf_files() -> bool:
    raw = os.environ.get("FEDWEAVE_PREFER_LOCAL_HF", "1").strip().lower()
    return raw not in ("0", "false", "no")


def _transformers_offline_enabled() -> bool:
    return os.environ.get("TRANSFORMERS_OFFLINE", "").strip() == "1"


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def is_flash_attn2_available() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_dtype_from_str(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def ensure_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def simple_normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_first_int(text: str) -> Optional[int]:
    match = re.search(r"(-?\d+)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def extract_first_choice_label(text: str) -> Optional[str]:
    stripped = str(text).strip()
    leading = re.match(r"^[\s\(\[（【]*([A-Ea-e])(?:[\s\)\]）】\.:：,，]|$)", stripped)
    if leading is not None:
        return leading.group(1).upper()
    explicit = re.search(
        r"(?:answer|option|choice|答案|选项|选择)\s*(?:is|为|是)?\s*[:：\-]?\s*[\(（]?\s*([A-Ea-e])",
        stripped,
        flags=re.I,
    )
    if explicit is not None:
        return explicit.group(1).upper()
    match = re.search(r"\b([A-Ea-e])\b", text)
    if match is not None:
        return match.group(1).upper()
    match = re.search(r"\b([1-5])\b", text)
    if match is not None:
        digit = match.group(1)
        return chr(ord("A") + int(digit) - 1)
    return None


def _tokenize_rouge_text(text: str, *, char_level: bool = False) -> List[str]:
    normalized = simple_normalize_text(text)
    if char_level:
        return [char for char in normalized if not char.isspace()]
    return normalized.split()


def _rouge_n_f1(prediction_tokens: Sequence[str], reference_tokens: Sequence[str], n: int) -> float:
    if n <= 0 or len(prediction_tokens) < n or len(reference_tokens) < n:
        return 0.0

    from collections import Counter

    pred_ngrams = Counter(tuple(prediction_tokens[idx : idx + n]) for idx in range(len(prediction_tokens) - n + 1))
    ref_ngrams = Counter(tuple(reference_tokens[idx : idx + n]) for idx in range(len(reference_tokens) - n + 1))
    overlap = 0
    for gram, count in pred_ngrams.items():
        overlap += min(count, ref_ngrams.get(gram, 0))
    precision = overlap / max(1, sum(pred_ngrams.values()))
    recall = overlap / max(1, sum(ref_ngrams.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    if not seq_a or not seq_b:
        return 0
    dp = [0] * (len(seq_b) + 1)
    for token_a in seq_a:
        prev = 0
        for j, token_b in enumerate(seq_b, start=1):
            current = dp[j]
            if token_a == token_b:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = current
    return int(dp[-1])


def _rouge_l_f1_from_tokens(prediction_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / max(1, len(prediction_tokens))
    recall = lcs / max(1, len(reference_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge1_f1(prediction: str, reference: str, *, char_level: bool = False) -> float:
    pred_tokens = _tokenize_rouge_text(prediction, char_level=char_level)
    ref_tokens = _tokenize_rouge_text(reference, char_level=char_level)
    if not pred_tokens or not ref_tokens:
        return 0.0
    return _rouge_n_f1(pred_tokens, ref_tokens, 1)


def rouge2_f1(prediction: str, reference: str, *, char_level: bool = False) -> float:
    pred_tokens = _tokenize_rouge_text(prediction, char_level=char_level)
    ref_tokens = _tokenize_rouge_text(reference, char_level=char_level)
    return _rouge_n_f1(pred_tokens, ref_tokens, 2)


def rouge_l_f1(prediction: str, reference: str, *, char_level: bool = False) -> float:
    pred_tokens = _tokenize_rouge_text(prediction, char_level=char_level)
    ref_tokens = _tokenize_rouge_text(reference, char_level=char_level)
    return _rouge_l_f1_from_tokens(pred_tokens, ref_tokens)


def rouge_lsum_f1(prediction: str, reference: str, *, char_level: bool = False) -> float:
    sentence_sep = r"[\n\.。！？!?]+"
    pred_sentences = [
        _tokenize_rouge_text(sent, char_level=char_level)
        for sent in re.split(sentence_sep, prediction)
        if simple_normalize_text(sent)
    ]
    ref_sentences = [
        _tokenize_rouge_text(sent, char_level=char_level)
        for sent in re.split(sentence_sep, reference)
        if simple_normalize_text(sent)
    ]
    pred_tokens = [token for sent in pred_sentences for token in sent]
    ref_tokens = [token for sent in ref_sentences for token in sent]
    return _rouge_l_f1_from_tokens(pred_tokens, ref_tokens)


def _normalize_generated_label(text: str) -> str:
    text = simple_normalize_text(text).lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    tokens = [tok for tok in text.split() if tok]
    if not tokens:
        return ""
    aliases = {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }
    first = tokens[0]
    if first in aliases:
        return aliases[first]
    return first


def _normalize_intent_label(text: str, label_names: Sequence[str]) -> str:
    labels = [str(label).strip() for label in label_names if str(label).strip()]
    normalized_to_label = {simple_normalize_text(label).lower().replace("-", "_").replace(" ", "_"): label for label in labels}
    text_norm = simple_normalize_text(text).lower().replace("-", "_")
    text_norm = re.sub(r"[^a-z0-9_ ]+", " ", text_norm)
    text_norm = re.sub(r"\s+", " ", text_norm).strip()
    first_line = simple_normalize_text(str(text).splitlines()[0] if str(text).splitlines() else text).lower().replace("-", "_").replace(" ", "_")
    first_line = re.sub(r"[^a-z0-9_]+", "_", first_line).strip("_")
    if first_line in normalized_to_label:
        return normalized_to_label[first_line]
    compact = text_norm.replace(" ", "_")
    for candidate, label in sorted(normalized_to_label.items(), key=lambda pair: len(pair[0]), reverse=True):
        if compact.startswith(candidate) or candidate in compact:
            return label
    tokens = [tok for tok in text_norm.split() if tok]
    if not tokens:
        return ""
    return tokens[0]


def extract_final_numeric_answer(text: str) -> Optional[int]:
    text = str(text)
    hash_match = re.search(r"####\s*(-?\d[\d,]*)", text)
    if hash_match is not None:
        candidate = hash_match.group(1).replace(",", "")
        try:
            return int(candidate)
        except Exception:
            pass
    answer_match = re.search(r"(?:answer is|final answer is|the answer is)\s*[:\-]?\s*(-?\d[\d,]*)", text, flags=re.I)
    if answer_match is not None:
        candidate = answer_match.group(1).replace(",", "")
        try:
            return int(candidate)
        except Exception:
            pass
    matches = re.findall(r"-?\d[\d,]*", text)
    if not matches:
        return None
    candidate = matches[-1].replace(",", "")
    try:
        return int(candidate)
    except Exception:
        return None



def compute_task_official_metrics(
    task: str,
    pred_texts: Sequence[str],
    target_texts: Sequence[str],
    metas: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    task = str(task).lower()
    if len(pred_texts) != len(target_texts):
        raise ValueError("pred_texts and target_texts must have the same length.")
    if not pred_texts:
        return {"main_score": 0.0}

    if task == "commonsense_reasoning":
        preds = [extract_first_choice_label(text) or "" for text in pred_texts]
        golds = [extract_first_choice_label(text) or "" for text in target_texts]
        accuracy = float(np.mean([float(pred == gold) for pred, gold in zip(preds, golds)]))
        return {"main_score": accuracy, "accuracy": accuracy}

    if task == "math_reasoning":
        preds = [extract_final_numeric_answer(text) for text in pred_texts]
        golds = [extract_final_numeric_answer(text) for text in target_texts]
        accuracy = float(np.mean([float(pred is not None and pred == gold) for pred, gold in zip(preds, golds)]))
        return {"main_score": accuracy, "accuracy": accuracy}

    if task in ("sentiment_analysis", "intent_detection"):
        if task == "intent_detection":
            label_names: List[str] = []
            if metas:
                for meta in metas:
                    for label in list((meta or {}).get("label_names", []) or []):
                        label = str(label).strip()
                        if label and label not in label_names:
                            label_names.append(label)
            if not label_names:
                label_names = sorted({str(text).strip() for text in target_texts if str(text).strip()})
            preds = [_normalize_intent_label(text, label_names) for text in pred_texts]
            golds = [_normalize_intent_label(text, label_names) for text in target_texts]
        else:
            preds = [_normalize_generated_label(text) for text in pred_texts]
            golds = [_normalize_generated_label(text) for text in target_texts]
        accuracy = float(np.mean([float(pred == gold) for pred, gold in zip(preds, golds)]))
        macro_labels = sorted(set(golds) | set(preds))
        macro_f1_scores: List[float] = []
        for label in macro_labels:
            tp = sum(1 for pred, gold in zip(preds, golds) if pred == label and gold == label)
            fp = sum(1 for pred, gold in zip(preds, golds) if pred == label and gold != label)
            fn = sum(1 for pred, gold in zip(preds, golds) if pred != label and gold == label)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            macro_f1_scores.append(float(f1))
        macro_f1 = float(np.mean(macro_f1_scores)) if macro_f1_scores else 0.0
        return {"main_score": accuracy, "accuracy": accuracy, "macro_f1": macro_f1}

    if task in ("text_editing", "struct_to_text", "summarization"):
        rouge1 = float(
            np.mean([rouge1_f1(pred, gold, char_level=False) for pred, gold in zip(pred_texts, target_texts)])
        )
        rouge2 = float(
            np.mean([rouge2_f1(pred, gold, char_level=False) for pred, gold in zip(pred_texts, target_texts)])
        )
        rougel = float(
            np.mean([rouge_l_f1(pred, gold, char_level=False) for pred, gold in zip(pred_texts, target_texts)])
        )
        rougelsum = float(
            np.mean([rouge_lsum_f1(pred, gold, char_level=False) for pred, gold in zip(pred_texts, target_texts)])
        )
        main_score = float((rouge1 + rouge2 + rougelsum) / 3.0)
        return {
            "main_score": main_score,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougel,
            "rougeLsum": rougelsum,
        }

    raise ValueError(task)


def set_system_prompt(prompt: str) -> None:
    global SYSTEM_PROMPT
    cleaned = (prompt or "").strip()
    SYSTEM_PROMPT = cleaned


def set_prompt_format(prompt_format: str) -> None:
    global PROMPT_FORMAT
    cleaned = str(prompt_format or "auto").strip().lower()
    if cleaned not in {"auto", "chat", "plain"}:
        raise ValueError(f"Unsupported prompt format: {prompt_format}")
    PROMPT_FORMAT = cleaned


def resolve_prompt_format(model_name: Optional[str], prompt_format: Optional[str] = None) -> str:
    candidate = str(prompt_format or PROMPT_FORMAT or "auto").strip().lower()
    if candidate in {"chat", "plain"}:
        return candidate
    model_key = str(model_name or "").strip().lower()
    chat_markers = ("instruct", "chat", "-it", "_it", "sft")
    if any(marker in model_key for marker in chat_markers):
        return "chat"
    return "plain"


def _render_plain_text(
    user_prompt: str,
    assistant_text: Optional[str] = None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    prompt = str(user_prompt).rstrip()
    prefix = "Response:"
    parts: List[str] = []
    if SYSTEM_PROMPT:
        parts.append(SYSTEM_PROMPT.strip())
    parts.append(prompt)
    base_text = "\n\n".join(parts)
    if prompt.endswith("Answer:"):
        if assistant_text is not None:
            return f"{base_text} {str(assistant_text).strip()}".strip()
        return base_text
    if assistant_text is not None:
        return f"{base_text}\n\n{prefix} {str(assistant_text).strip()}".strip()
    if add_generation_prompt:
        return f"{base_text}\n\n{prefix}"
    return base_text


def render_chat_text(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    assistant_text: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    resolved_format = resolve_prompt_format(getattr(tokenizer, "name_or_path", ""), PROMPT_FORMAT)
    if resolved_format == "plain":
        return _render_plain_text(
            user_prompt=user_prompt,
            assistant_text=assistant_text,
            add_generation_prompt=add_generation_prompt,
        )
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    if SYSTEM_PROMPT:
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt and assistant_text is None,
    )


def _tokenizer_cache_prefix(tokenizer: AutoTokenizer) -> str:
    return str(getattr(tokenizer, "name_or_path", "tokenizer"))


def render_chat_text_cached(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    assistant_text: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    resolved_format = resolve_prompt_format(getattr(tokenizer, "name_or_path", ""), PROMPT_FORMAT)
    cache_key = (
        _tokenizer_cache_prefix(tokenizer),
        resolved_format,
        SYSTEM_PROMPT,
        user_prompt,
        assistant_text,
        bool(add_generation_prompt),
    )
    cached = _CHAT_RENDER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rendered = render_chat_text(
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        assistant_text=assistant_text,
        add_generation_prompt=add_generation_prompt,
    )
    _CHAT_RENDER_CACHE[cache_key] = rendered
    return rendered


def encode_text_cached(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    max_length: Optional[int] = None,
    add_special_tokens: bool = True,
) -> List[int]:
    max_length_key = -1 if max_length is None else int(max_length)
    cache_key = (_tokenizer_cache_prefix(tokenizer), max_length_key, bool(add_special_tokens), text)
    cached = _TOKEN_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    encoded = tokenizer(
        text,
        truncation=max_length is not None,
        max_length=None if max_length is None else int(max_length),
        add_special_tokens=add_special_tokens,
    )
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    _TOKEN_ID_CACHE[cache_key] = input_ids
    return input_ids


class RouterNet(nn.Module):
    def __init__(self, d_model: int, k_experts: int, hidden: int = 512, dropout: float = 0.0, out_dim: Optional[int] = None):
        super().__init__()
        self.d_model = int(d_model)
        self.k = int(k_experts)
        self.out_dim = int(out_dim) if out_dim is not None else 2 * int(k_experts) - 1
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.out_dim),
        )
        self._compiled_mlp = None

    def _get_mlp(self):
        compiled_mlp = self.__dict__.get("_compiled_mlp")
        if compiled_mlp is None:
            compiled_mlp = _maybe_compile(self.mlp)
            # Keep torch.compile's OptimizedModule out of nn.Module registration.
            object.__setattr__(self, "_compiled_mlp", compiled_mlp)
        return compiled_mlp

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        canonical_state = OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if not key.startswith("_compiled_mlp.")
        )
        for key, value in state_dict.items():
            prefix = "_compiled_mlp._orig_mod."
            if key.startswith(prefix):
                canonical_state.setdefault(f"mlp.{key[len(prefix):]}", value)
        return super().load_state_dict(canonical_state, strict=strict, assign=assign)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._get_mlp()(x)


@dataclass
class Example:
    task: str
    prompt: str
    target: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor


def build_prompt_target(task: str, item: Dict[str, Any]) -> Example:
    task = task.lower()
    if task == "text_editing":
        return Example(
            task="text_editing",
            prompt=str(item["src"]).strip(),
            target=str(item["tgt"]).strip(),
            meta={"subtask": str(item.get("task", ""))},
        )
    if task == "struct_to_text":
        return Example(
            task="struct_to_text",
            prompt=(
                "Generate a fluent natural-language description from the structured record below.\n"
                "Use only the provided information.\n\n"
                f"Meaning representation: {str(item['meaning_representation']).strip()}"
            ),
            target=str(item["target"]).strip(),
        )
    if task == "intent_detection":
        label_names = [str(label).strip() for label in item.get("_label_names", []) if str(label).strip()]
        label_id = int(item["label"])
        if label_names:
            if label_id < 0 or label_id >= len(label_names):
                raise ValueError(f"Unsupported BANKING77 intent label: {label_id}")
            target = label_names[label_id]
        else:
            target = str(item.get("label_text", item.get("intent", item["label"]))).strip()
        labels_text = ", ".join(label_names) if label_names else "the correct BANKING77 intent label"
        return Example(
            task="intent_detection",
            prompt=(
                "Classify the banking customer query below.\n"
                "Reply with exactly one intent label.\n\n"
                f"Available intent labels: {labels_text}\n\n"
                f"Query: {str(item['text']).strip()}"
            ),
            target=target,
            meta={"label_names": label_names},
        )
    if task == "summarization":
        return Example(
            task="summarization",
            prompt=(
                "Summarize the document below in one concise sentence.\n\n"
                f"Document: {str(item['document']).strip()}"
            ),
            target=str(item["summary"]).strip(),
        )
    if task == "math_reasoning":
        return Example(
            task="math_reasoning",
            prompt=(
                "Solve the grade-school math problem below.\n"
                "Give the reasoning briefly, and end with a line in the format #### <answer>.\n\n"
                f"Problem: {str(item['question']).strip()}"
            ),
            target=str(item["answer"]).strip(),
        )
    if task == "sentiment_analysis":
        label_id = int(item["label"])
        label_map = {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
        if label_id not in label_map:
            raise ValueError(f"Unsupported TweetEval sentiment label: {label_id}")
        return Example(
            task="sentiment_analysis",
            prompt=(
                "Classify the sentiment of the tweet below.\n"
                "Reply with exactly one label: negative, neutral, or positive.\n\n"
                f"Tweet: {str(item['text']).strip()}"
            ),
            target=label_map[label_id],
        )
    if task == "commonsense_reasoning":
        choice_lines = [f"({label}) {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])]
        return Example(
            task="commonsense_reasoning",
            prompt=(
                "Answer the multiple-choice question below.\n"
                "Reply with only the correct option letter.\n\n"
                f"Question: {item['question']}\n\n"
                "Options:\n"
                + "\n".join(choice_lines)
            ),
            target=item.get("answerKey", ""),
        )
    raise ValueError(f"Unknown task: {task}")


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: int):
    if tokenizer.pad_token_id is None:
        ensure_tokenizer_padding(tokenizer)
    pad_token_id = int(tokenizer.pad_token_id)

    def _pad_token_lists(token_lists: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not token_lists:
            empty = torch.empty((0, 0), dtype=torch.long)
            return empty, empty
        max_seq_len = max(len(tokens) for tokens in token_lists)
        input_ids = torch.full((len(token_lists), max_seq_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(token_lists), max_seq_len), dtype=torch.long)
        for row_idx, tokens in enumerate(token_lists):
            if not tokens:
                continue
            seq = torch.tensor(tokens, dtype=torch.long)
            seq_len = int(seq.numel())
            input_ids[row_idx, :seq_len] = seq
            attention_mask[row_idx, :seq_len] = 1
        return input_ids, attention_mask

    def collate(examples: List[Example]) -> Batch:
        prompts = [example.prompt for example in examples]
        targets = [example.target for example in examples]
        prompt_texts = [render_chat_text_cached(tokenizer, prompt, add_generation_prompt=True) for prompt in prompts]
        full_texts = [
            render_chat_text_cached(tokenizer, prompt, assistant_text=target)
            for prompt, target in zip(prompts, targets)
        ]

        full_token_lists = [
            encode_text_cached(tokenizer, text, max_length=max_length, add_special_tokens=True)
            for text in full_texts
        ]
        prompt_token_lists = [
            encode_text_cached(tokenizer, text, max_length=max_length, add_special_tokens=True)
            for text in prompt_texts
        ]
        input_ids, attention_mask = _pad_token_lists(full_token_lists)
        query_input_ids, query_attention_mask = _pad_token_lists(prompt_token_lists)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        for row_idx in range(labels.size(0)):
            seq_len = int(attention_mask[row_idx].sum().item())
            prompt_len = min(int(query_attention_mask[row_idx].sum().item()), seq_len)
            labels[row_idx, :prompt_len] = -100

        return Batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
        )

    return collate


def build_lora_config(rank: int, alpha: int, dropout: float, target_modules: List[str]) -> Dict[str, Any]:
    return {
        "rank": int(rank),
        "alpha": int(alpha),
        "dropout": float(dropout),
        "target_modules": list(target_modules),
    }


def _expert_id_from_adapter_name(adapter_name: str) -> int:
    name = str(adapter_name).strip()
    if name in ("", "default"):
        return 0
    if name.startswith("expert_"):
        return int(name.split("_", 1)[1])
    raise ValueError(f"Unsupported adapter/expert name: {adapter_name}")


def _adapter_name_from_expert_id(expert_id: int) -> str:
    return "default" if int(expert_id) == 0 else f"expert_{int(expert_id)}"


def _matches_expert_param_name(name: str, collection_name: str, expert_id: int) -> bool:
    parts = str(name).split(".")
    return len(parts) >= 2 and parts[-2] == str(collection_name) and parts[-1] == str(int(expert_id))


def _remap_expert_key(name: str, source_expert_id: int, target_expert_id: int) -> str:
    parts = str(name).split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "lora_A_experts" and parts[idx + 1] == str(int(source_expert_id)):
            parts[idx + 1] = str(int(target_expert_id))
            return ".".join(parts)
        if parts[idx] == "lora_B_experts" and parts[idx + 1] == str(int(source_expert_id)):
            parts[idx + 1] = str(int(target_expert_id))
            return ".".join(parts)
    return name


def remap_adapter_state(
    adapter_state: Dict[str, torch.Tensor],
    *,
    source_adapter_name: str = "default",
    target_adapter_name: str,
) -> Dict[str, torch.Tensor]:
    source_expert_id = _expert_id_from_adapter_name(source_adapter_name)
    target_expert_id = _expert_id_from_adapter_name(target_adapter_name)
    return {
        _remap_expert_key(name, source_expert_id, target_expert_id): value.detach().cpu().clone()
        for name, value in adapter_state.items()
    }


@contextmanager
def layerwise_lora_routing_context(
    *,
    weights: Optional[torch.Tensor] = None,
    disable: Optional[bool] = None,
    default_expert: Optional[int] = None,
    active_experts: Optional[Sequence[int]] = None,
):
    old_state = dict(_LAYERWISE_LORA_STATE)
    if weights is not None:
        _LAYERWISE_LORA_STATE["weights"] = weights
    if disable is not None:
        _LAYERWISE_LORA_STATE["disabled"] = bool(disable)
    if default_expert is not None:
        _LAYERWISE_LORA_STATE["default_expert"] = int(default_expert)
    if active_experts is not None:
        _LAYERWISE_LORA_STATE["active_experts"] = [int(v) for v in active_experts]
    try:
        yield
    finally:
        _LAYERWISE_LORA_STATE.clear()
        _LAYERWISE_LORA_STATE.update(old_state)


class MultiExpertLoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        rank: int,
        alpha: int,
        dropout: float,
        num_experts: int = 1,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.rank = int(rank)
        self.lora_alpha = int(alpha)
        self.scaling = float(alpha) / max(1, int(rank))
        self.num_experts = 0
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_A_experts = nn.ParameterList()
        self.lora_B_experts = nn.ParameterList()
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.expand_num_experts(int(num_experts))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float,
        num_experts: int = 1,
    ) -> "MultiExpertLoRALinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            num_experts=num_experts,
        )
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and module.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        module.to(device=linear.weight.device, dtype=linear.weight.dtype)
        return module

    def _init_new_expert(self, expert_id: int, clone_from: Optional[int] = None) -> None:
        a = nn.Parameter(torch.empty((self.rank, self.in_features), device=self.weight.device, dtype=self.weight.dtype))
        b = nn.Parameter(torch.empty((self.out_features, self.rank), device=self.weight.device, dtype=self.weight.dtype))
        if clone_from is None:
            nn.init.kaiming_uniform_(a, a=np.sqrt(5.0))
            nn.init.zeros_(b)
        else:
            a.data.copy_(self.lora_A_experts[int(clone_from)].data)
            b.data.copy_(self.lora_B_experts[int(clone_from)].data)
        self.lora_A_experts.append(a)
        self.lora_B_experts.append(b)
        self.num_experts += 1

    def expand_num_experts(self, target_num_experts: int) -> None:
        target = max(1, int(target_num_experts))
        while self.num_experts < target:
            clone_from = 0 if self.num_experts > 0 else None
            self._init_new_expert(self.num_experts, clone_from=clone_from)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.rank <= 0 or bool(_LAYERWISE_LORA_STATE.get("disabled", False)):
            return result
        batch_size = int(x.size(0))
        weights = _LAYERWISE_LORA_STATE.get("weights")
        if weights is None:
            default_expert = int(_LAYERWISE_LORA_STATE.get("default_expert", 0))
            weights = torch.zeros((batch_size, int(self.num_experts)), device=x.device, dtype=result.dtype)
            weights[:, max(0, min(default_expert, int(self.num_experts) - 1))] = 1.0
        else:
            if int(weights.size(0)) != batch_size:
                raise ValueError(
                    f"Layer-wise LoRA routing batch size mismatch: weights batch={int(weights.size(0))}, input batch={batch_size}."
                )
            weights = weights.to(device=x.device, dtype=result.dtype)
        active_experts = _LAYERWISE_LORA_STATE.get("active_experts")
        if active_experts is None:
            active_experts = [
                expert_id
                for expert_id in range(int(self.num_experts))
                if bool((weights[:, expert_id].abs() > 1e-6).any().item())
            ]
        if not active_experts:
            active_experts = [int(torch.argmax(weights[0]).item())]
        dropped = self.lora_dropout(x)
        for expert_id in active_experts:
            expert_id = int(expert_id)
            lora_hidden = F.linear(dropped, self.lora_A_experts[expert_id], bias=None)
            lora_out = F.linear(lora_hidden, self.lora_B_experts[expert_id], bias=None) * self.scaling
            coeff_shape = [batch_size] + [1] * (lora_out.ndim - 1)
            coeff = weights[:, expert_id].view(*coeff_shape)
            result = result + lora_out * coeff
        return result


def _replace_target_linear_with_layerwise_lora(
    model: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str],
) -> None:
    key_list = [key for key, _ in model.named_modules()]
    matched = False
    for key in key_list:
        target_found = any(str(key).endswith(str(target_key)) for target_key in target_modules)
        if not target_found:
            continue
        module = model.get_submodule(key)
        if not isinstance(module, nn.Linear):
            continue
        parent_path, _, target_name = key.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(
            parent,
            target_name,
            MultiExpertLoRALinear.from_linear(
                module,
                rank=int(rank),
                alpha=int(alpha),
                dropout=float(dropout),
                num_experts=1,
            ),
        )
        matched = True
    if not matched:
        raise ValueError(f"Target modules {list(target_modules)} not found in the base model.")


def _attach_layerwise_lora_api(model: Any) -> None:
    def _set_adapter(self, adapter_name: str) -> None:
        _LAYERWISE_LORA_STATE["default_expert"] = _expert_id_from_adapter_name(adapter_name)

    @contextmanager
    def _disable_adapter(self):
        with layerwise_lora_routing_context(disable=True):
            yield

    model.set_adapter = MethodType(_set_adapter, model)
    model.disable_adapter = MethodType(_disable_adapter, model)


def list_lora_b_params(model: nn.Module, adapter_name: str) -> List[torch.Tensor]:
    params: List[torch.Tensor] = []
    expert_id = _expert_id_from_adapter_name(adapter_name)
    for name, param in model.named_parameters():
        if _matches_expert_param_name(name, "lora_B_experts", expert_id):
            params.append(param.detach())
    return params


def list_lora_a_params(model: nn.Module, adapter_name: str) -> List[torch.Tensor]:
    params: List[torch.Tensor] = []
    expert_id = _expert_id_from_adapter_name(adapter_name)
    for name, param in model.named_parameters():
        if _matches_expert_param_name(name, "lora_A_experts", expert_id):
            params.append(param.detach())
    return params


def extract_lora_b_vector(model: nn.Module, adapter_name: str) -> torch.Tensor:
    params = list_lora_b_params(model, adapter_name)
    if not params:
        raise RuntimeError(f"No LoRA-B params found for adapter '{adapter_name}'.")
    return torch.cat([param.float().flatten().cpu() for param in params], dim=0)


def extract_lora_a_vector(model: nn.Module, adapter_name: str) -> torch.Tensor:
    params = list_lora_a_params(model, adapter_name)
    if not params:
        raise RuntimeError(f"No LoRA-A params found for adapter '{adapter_name}'.")
    return torch.cat([param.float().flatten().cpu() for param in params], dim=0)


def extract_lora_ab_vector(model: nn.Module, adapter_name: str) -> torch.Tensor:
    a_vec = extract_lora_a_vector(model, adapter_name)
    b_vec = extract_lora_b_vector(model, adapter_name)
    return torch.cat([a_vec, b_vec], dim=0)


def get_adapter_state(model: nn.Module, adapter_name: str) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    expert_id = _expert_id_from_adapter_name(adapter_name)
    for name, param in model.named_parameters():
        if _matches_expert_param_name(name, "lora_A_experts", expert_id) or _matches_expert_param_name(name, "lora_B_experts", expert_id):
            state[name] = param.detach().cpu().clone()
    if not state:
        raise RuntimeError(f"Adapter state is empty for '{adapter_name}'.")
    return state


@torch.no_grad()
def load_adapter_state(model: nn.Module, adapter_state: Dict[str, torch.Tensor]) -> None:
    name_to_param = dict(model.named_parameters())
    for name, value in adapter_state.items():
        if name not in name_to_param:
            continue
        target = name_to_param[name]
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))


def freeze_all_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def set_trainable_adapter(model: nn.Module, adapter_name: str) -> None:
    expert_id = _expert_id_from_adapter_name(adapter_name)
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = (
                _matches_expert_param_name(name, "lora_A_experts", expert_id)
                or _matches_expert_param_name(name, "lora_B_experts", expert_id)
            )
        else:
            param.requires_grad = False


def add_expert_adapters_if_needed(model: Any, k: int) -> None:
    for module in model.modules():
        if isinstance(module, MultiExpertLoRALinear):
            module.expand_num_experts(int(k))


def set_trainable_router(routers: Dict[int, nn.Module], active_gid: Optional[int]) -> None:
    for gid, router in routers.items():
        flag = active_gid is not None and int(gid) == int(active_gid)
        for param in router.parameters():
            param.requires_grad = flag


def adapter_name_for_cluster(cluster_id: int) -> str:
    return _adapter_name_from_expert_id(int(cluster_id))


@torch.no_grad()
def load_all_expert_states(
    model: nn.Module,
    server_expert_sd: Dict[int, Dict[str, torch.Tensor]],
    k_experts: int,
) -> None:
    for expert_id in range(int(k_experts)):
        load_adapter_state(model, server_expert_sd[expert_id])


def fit_predict_precomputed_agglomerative(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        clusterer = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            affinity="precomputed",
            linkage="average",
        )
    return clusterer.fit_predict(dist_matrix)


def make_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _model_device(model: Any) -> torch.device:
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


@torch.inference_mode()
def compute_query_embedding(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    base_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = None
    for attr in ("model", "transformer"):
        candidate = getattr(base_lm, attr, None)
        if candidate is not None:
            backbone = candidate
            break

    if backbone is not None:
        try:
            out = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        except TypeError:
            backbone = None

    if backbone is None:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = out.hidden_states[-1]

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def select_top_m_indices_adaptive(
    logits_2k1: torch.Tensor,
    weights_2k1: torch.Tensor,
    k_experts: int,
    m_select: int,
    m_tau: float,
    min_m: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if int(m_select) <= 0:
        m_select = max(1, min(int(k_experts), 2 * int(k_experts) - 1))
        chosen_idx = torch.topk(logits_2k1, k=m_select, dim=-1).indices
        chosen_mask = torch.ones_like(chosen_idx, dtype=torch.bool)
        return chosen_idx, chosen_mask
    m_select = max(1, min(int(m_select), 2 * int(k_experts) - 1))
    min_m = max(1, min(int(min_m), m_select))
    m_tau = float(np.clip(m_tau, 0.0, 1.0))

    first = logits_2k1[:, :k_experts]
    rest = logits_2k1[:, k_experts:]
    best_first = torch.argmax(first, dim=-1, keepdim=True)
    k_rest = min(max(0, m_select - 1), rest.size(1))

    if k_rest == 0:
        return best_first, torch.ones_like(best_first, dtype=torch.bool)

    top_rest_local = torch.topk(rest, k=k_rest, dim=-1).indices
    top_rest = top_rest_local + int(k_experts)
    chosen_idx = torch.cat([best_first, top_rest], dim=-1)

    rest_probs = torch.gather(weights_2k1[:, k_experts:], 1, top_rest_local)
    rest_total = weights_2k1[:, k_experts:].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cumulative = torch.cumsum(rest_probs, dim=-1) / rest_total
    need_rest = (cumulative < m_tau).sum(dim=-1) + 1
    need_rest = torch.where(
        weights_2k1[:, k_experts:].sum(dim=-1) <= 1e-12,
        torch.zeros_like(need_rest),
        need_rest,
    )
    min_rest = max(0, min_m - 1)
    need_rest = need_rest.clamp(min=min_rest, max=k_rest)

    mask = torch.zeros((logits_2k1.size(0), 1 + k_rest), dtype=torch.bool, device=logits_2k1.device)
    mask[:, 0] = True
    cols = torch.arange(k_rest, device=logits_2k1.device).view(1, -1)
    mask[:, 1:] = cols < need_rest.unsqueeze(1)
    return chosen_idx, mask


def fold_2k1_weights_to_k(
    weights_2k1: torch.Tensor,
    chosen_idx: torch.Tensor,
    chosen_mask: Optional[torch.Tensor],
    k_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chosen_weights = torch.gather(weights_2k1, dim=1, index=chosen_idx)
    if chosen_mask is not None:
        chosen_weights = chosen_weights * chosen_mask.to(dtype=chosen_weights.dtype)

    is_first = (chosen_idx < int(k_experts)).to(chosen_weights.dtype)
    assigned = (chosen_weights * is_first).sum(dim=1, keepdim=True)
    other = chosen_weights * (1.0 - is_first)
    return assigned, other, chosen_idx


def sparsify_absolute_router_weights(
    logits_k: torch.Tensor,
    *,
    m_select: int,
    m_tau: float,
) -> torch.Tensor:
    weights_k = F.softmax(logits_k, dim=-1)
    k_experts = int(weights_k.size(1))
    if k_experts <= 1:
        return weights_k

    if int(m_select) <= 0:
        m_select = k_experts
    m_select = max(1, min(int(m_select), k_experts))
    if m_select >= k_experts:
        return weights_k

    top_weights, top_idx = torch.topk(weights_k, k=m_select, dim=-1)
    m_tau = float(np.clip(m_tau, 0.0, 1.0))
    cumulative = torch.cumsum(top_weights, dim=-1) / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    need = (cumulative < m_tau).sum(dim=-1) + 1
    need = need.clamp(min=1, max=m_select)
    cols = torch.arange(m_select, device=weights_k.device).view(1, -1)
    top_mask = (cols < need.unsqueeze(1)).to(dtype=weights_k.dtype)
    sparse = torch.zeros_like(weights_k)
    sparse.scatter_add_(1, top_idx, top_weights * top_mask)
    return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_mixture_logits(
    model: Any,
    router: RouterNet,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    query_input_ids: torch.Tensor,
    query_attention_mask: torch.Tensor,
    k_experts: int,
    assigned_expert: int,
    m_select: int,
    m_tau: float,
    expert_id_to_adapter: Dict[int, str],
    mix_in_logprob: bool = False,
    absolute_routing: bool = False,
    disable_lora_for_query: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    if disable_lora_for_query:
        with model.disable_adapter():
            qemb = compute_query_embedding(model, query_input_ids, query_attention_mask)
    else:
        model.set_adapter(expert_id_to_adapter[int(assigned_expert)])
        with torch.no_grad():
            qemb = compute_query_embedding(model, query_input_ids, query_attention_mask)

    router_logits = router(qemb.float())

    if absolute_routing:
        weights_k = sparsify_absolute_router_weights(
            router_logits,
            m_select=m_select,
            m_tau=m_tau,
        )
        entropy_weights = weights_k
    else:
        weights_2k1 = F.softmax(router_logits, dim=-1)
        entropy_weights = weights_2k1

        chosen_idx, chosen_mask = select_top_m_indices_adaptive(
            logits_2k1=router_logits,
            weights_2k1=weights_2k1,
            k_experts=k_experts,
            m_select=m_select,
            m_tau=m_tau,
            min_m=1,
        )
        assigned_weight, _, chosen_idx = fold_2k1_weights_to_k(
            weights_2k1=weights_2k1,
            chosen_idx=chosen_idx,
            chosen_mask=chosen_mask,
            k_experts=k_experts,
        )

        other_ids = [expert_id for expert_id in range(int(k_experts)) if expert_id != int(assigned_expert)]
        weights_k = torch.zeros((input_ids.size(0), int(k_experts)), device=input_ids.device, dtype=torch.float32)
        weights_k[:, int(assigned_expert)] += assigned_weight.squeeze(1)

        for col_idx in range(chosen_idx.size(1)):
            slot_ids = chosen_idx[:, col_idx]
            slot_mask = chosen_mask[:, col_idx].to(dtype=weights_2k1.dtype)
            slot_weights = torch.gather(weights_2k1, 1, slot_ids.unsqueeze(1)).squeeze(1) * slot_mask
            is_other = (slot_ids >= int(k_experts)) & (slot_mask > 0)
            if not is_other.any():
                continue
            selected_slots = (slot_ids[is_other] - int(k_experts)).long()
            mapped = torch.tensor(
                [other_ids[slot.item()] for slot in selected_slots],
                device=input_ids.device,
                dtype=torch.long,
            )
            weights_k[is_other, mapped] += slot_weights[is_other]

        weights_k = weights_k / weights_k.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    entropy = (-entropy_weights * entropy_weights.clamp_min(1e-12).log()).sum(dim=-1)
    keff = torch.exp(entropy)
    top1 = torch.max(weights_k, dim=-1).values.mean().item()
    nonzero_mask = (weights_k > 1e-6).any(dim=0)
    active_experts = [expert_id for expert_id in range(int(k_experts)) if bool(nonzero_mask[expert_id].item())]
    if int(assigned_expert) not in active_experts:
        active_experts.append(int(assigned_expert))
    with layerwise_lora_routing_context(
        weights=weights_k,
        default_expert=int(assigned_expert),
        active_experts=active_experts,
    ):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    logits = output.logits
    mix_out = F.log_softmax(logits, dim=-1) if mix_in_logprob else logits

    stats = {
        "mix_entropy": float(entropy.mean().item()),
        "mix_top1": float(top1),
        "router_keff": float(keff.mean().item()),
        "mix_m_used": float((weights_k > 1e-6).sum(dim=-1).float().mean().item()),
    }
    if absolute_routing:
        targets = torch.full(
            (router_logits.size(0),),
            int(assigned_expert),
            device=router_logits.device,
            dtype=torch.long,
        )
        route_ce_loss = F.cross_entropy(router_logits.float(), targets)
    else:
        route_ce_loss = router_logits.sum() * 0.0
    return mix_out, stats, route_ce_loss


def build_optimizer(params: Iterable[torch.nn.Parameter], lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(wd))


def build_split_lr_optimizer(
    model_params: Iterable[torch.nn.Parameter],
    router_params: Iterable[torch.nn.Parameter],
    model_lr: float,
    router_lr: float,
    wd: float,
) -> torch.optim.Optimizer:
    param_groups = []
    model_params = list(model_params)
    router_params = list(router_params)
    if model_params:
        param_groups.append(
            {
                "params": model_params,
                "lr": float(model_lr),
                "weight_decay": float(wd),
            }
        )
    if router_params:
        param_groups.append(
            {
                "params": router_params,
                "lr": float(router_lr),
                "weight_decay": float(wd),
            }
        )
    if not param_groups:
        raise ValueError("At least one parameter group is required to build an optimizer.")
    return torch.optim.AdamW(param_groups)


def build_local_scheduler(
    optimizer: torch.optim.Optimizer,
    num_optimizer_steps: int,
    schedule: str,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    schedule_name = str(schedule).strip().lower()
    if schedule_name == "constant" or num_optimizer_steps <= 0:
        return None
    if schedule_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(num_optimizer_steps)),
            eta_min=0.0,
        )
    raise ValueError(f"Unsupported local lr schedule: {schedule}")


def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return dataclasses.replace(
        batch,
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
        query_input_ids=batch.query_input_ids.to(device),
        query_attention_mask=batch.query_attention_mask.to(device),
    )


def compute_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not bool((shift_labels != -100).any().item()):
        return shift_logits.sum() * 0.0
    vocab_size = shift_logits.size(-1)
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def compute_nll_loss(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_log_probs = log_probs[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if not bool((shift_labels != -100).any().item()):
        return shift_log_probs.sum() * 0.0
    vocab_size = shift_log_probs.size(-1)
    return F.nll_loss(
        shift_log_probs.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def greedy_generate(
    model: Any,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    max_new_tokens: int = 16,
) -> str:
    model.eval()
    device = _model_device(model)
    prompt_text = render_chat_text_cached(tokenizer, user_prompt, add_generation_prompt=True)
    prompt_token_ids = encode_text_cached(tokenizer, prompt_text, add_special_tokens=True)
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][input_ids.size(1) :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def greedy_generate_batch(
    model: Any,
    tokenizer: AutoTokenizer,
    user_prompts: Sequence[str],
    max_new_tokens: int = 16,
) -> List[str]:
    model.eval()
    device = _model_device(model)
    prompt_texts = [render_chat_text_cached(tokenizer, p, add_generation_prompt=True) for p in user_prompts]
    enc = tokenizer(prompt_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated: List[str] = []
    input_len = input_ids.size(1)
    for i in range(output.size(0)):
        gen_ids = output[i][input_len:]
        generated.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return generated


def _per_example_nll_loss(log_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B = log_probs.size(0)
    losses: List[torch.Tensor] = []
    for i in range(B):
        shift_log_probs = log_probs[i, :-1, :].contiguous()
        shift_labels = labels[i, 1:].contiguous()
        if not bool((shift_labels != -100).any().item()):
            losses.append(torch.tensor(0.0, device=log_probs.device, dtype=log_probs.dtype))
            continue
        losses.append(F.nll_loss(
            shift_log_probs.view(-1, shift_log_probs.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        ))
    return torch.stack(losses)


def _per_example_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    B = logits.size(0)
    losses: List[torch.Tensor] = []
    for i in range(B):
        shift_logits = logits[i, :-1, :].contiguous()
        shift_labels = labels[i, 1:].contiguous()
        if not bool((shift_labels != -100).any().item()):
            losses.append(torch.tensor(0.0, device=logits.device, dtype=logits.dtype))
            continue
        vocab_size = shift_logits.size(-1)
        losses.append(F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        ))
    return torch.stack(losses)


@torch.no_grad()
def compute_batch_expert_weights(
    model: Any,
    tokenizer: AutoTokenizer,
    router: Optional[RouterNet],
    examples: Sequence[Any],
    k_experts: int,
    assigned_expert: int,
    expert_id_to_adapter: Dict[int, str],
    max_length: int,
    m_select_eval: int,
    m_tau_eval: float,
    *,
    disable_lora_for_query: bool = False,
    absolute_routing: bool = False,
) -> torch.Tensor:
    device = _model_device(model)
    if router is None or int(k_experts) == 1:
        weights = torch.zeros((len(examples), max(1, int(k_experts))), device=device, dtype=torch.float32)
        weights[:, 0] = 1.0
        return weights

    prompts = [ex.prompt for ex in examples]
    prompt_texts = [render_chat_text_cached(tokenizer, p, add_generation_prompt=True) for p in prompts]
    enc = tokenizer(prompt_texts, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if disable_lora_for_query:
        with model.disable_adapter():
            qemb = compute_query_embedding(model, input_ids, attention_mask)
    else:
        model.set_adapter(expert_id_to_adapter[int(assigned_expert)])
        qemb = compute_query_embedding(model, input_ids, attention_mask)

    router_logits = router(qemb.float())

    if absolute_routing:
        return sparsify_absolute_router_weights(
            router_logits,
            m_select=m_select_eval,
            m_tau=m_tau_eval,
        )

    probs_2k1 = F.softmax(router_logits, dim=-1)
    chosen_idx, chosen_mask = select_top_m_indices_adaptive(
        logits_2k1=router_logits,
        weights_2k1=probs_2k1,
        k_experts=k_experts,
        m_select=m_select_eval,
        m_tau=m_tau_eval,
        min_m=1,
    )

    B = len(examples)
    other_ids = [expert_id for expert_id in range(int(k_experts)) if expert_id != int(assigned_expert)]
    weights_k = torch.zeros((B, int(k_experts)), device=device, dtype=torch.float32)
    assigned_weight, _, _ = fold_2k1_weights_to_k(
        weights_2k1=probs_2k1,
        chosen_idx=chosen_idx,
        chosen_mask=chosen_mask,
        k_experts=k_experts,
    )
    weights_k[:, int(assigned_expert)] += assigned_weight.squeeze(1)

    for col_idx in range(chosen_idx.size(1)):
        slot_ids = chosen_idx[:, col_idx]
        slot_mask = chosen_mask[:, col_idx].to(dtype=probs_2k1.dtype)
        slot_weights = torch.gather(probs_2k1, 1, slot_ids.unsqueeze(1)).squeeze(1) * slot_mask
        is_other = (slot_ids >= int(k_experts)) & (slot_mask > 0)
        if not is_other.any():
            continue
        selected_slots = (slot_ids[is_other] - int(k_experts)).long()
        mapped = torch.tensor(
            [other_ids[int(slot.item())] for slot in selected_slots],
            device=device,
            dtype=torch.long,
        )
        weights_k[is_other, mapped] += slot_weights[is_other]

    return weights_k / weights_k.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def compute_example_expert_weights(
    model: Any,
    tokenizer: AutoTokenizer,
    router: Optional[RouterNet],
    ex: Example,
    k_experts: int,
    assigned_expert: int,
    expert_id_to_adapter: Dict[int, str],
    max_length: int,
    m_select_eval: int,
    m_tau_eval: float,
    *,
    disable_lora_for_query: bool = False,
    absolute_routing: bool = False,
) -> torch.Tensor:
    device = _model_device(model)
    if router is None or int(k_experts) == 1:
        weights = torch.zeros((1, max(1, int(k_experts))), device=device, dtype=torch.float32)
        weights[:, 0] = 1.0
        return weights

    prompt_text = render_chat_text_cached(tokenizer, ex.prompt, add_generation_prompt=True)
    prompt_token_ids = encode_text_cached(
        tokenizer,
        prompt_text,
        max_length=int(max_length),
        add_special_tokens=True,
    )
    input_ids = torch.tensor(prompt_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    if disable_lora_for_query:
        with model.disable_adapter():
            qemb = compute_query_embedding(model, input_ids, attention_mask)
    else:
        model.set_adapter(expert_id_to_adapter[int(assigned_expert)])
        qemb = compute_query_embedding(model, input_ids, attention_mask)
    router_logits = router(qemb.float())

    if absolute_routing:
        return sparsify_absolute_router_weights(
            router_logits,
            m_select=m_select_eval,
            m_tau=m_tau_eval,
        )

    probs_2k1 = F.softmax(router_logits, dim=-1)

    chosen_idx, chosen_mask = select_top_m_indices_adaptive(
        logits_2k1=router_logits,
        weights_2k1=probs_2k1,
        k_experts=k_experts,
        m_select=m_select_eval,
        m_tau=m_tau_eval,
        min_m=1,
    )
    assigned_weight, _, chosen_idx = fold_2k1_weights_to_k(
        weights_2k1=probs_2k1,
        chosen_idx=chosen_idx,
        chosen_mask=chosen_mask,
        k_experts=k_experts,
    )
    other_ids = [expert_id for expert_id in range(int(k_experts)) if expert_id != int(assigned_expert)]
    weights_k = torch.zeros((1, int(k_experts)), device=device, dtype=torch.float32)
    weights_k[:, int(assigned_expert)] += assigned_weight.squeeze(1)
    for col_idx in range(chosen_idx.size(1)):
        if not bool(chosen_mask[:, col_idx].item()):
            continue
        idx = chosen_idx[:, col_idx]
        if idx.item() >= int(k_experts):
            slot = int(idx.item() - int(k_experts))
            weights_k[:, other_ids[slot]] += probs_2k1[:, idx.item()]
    return weights_k / weights_k.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def greedy_generate_mixture(
    model: Any,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    weights_k: torch.Tensor,
    expert_id_to_adapter: Dict[int, str],
    max_new_tokens: int = 32,
) -> str:
    active_experts = [expert_id for expert_id in range(weights_k.size(1)) if float(weights_k[0, expert_id].item()) > 1e-6]
    if not active_experts:
        active_experts = [int(torch.argmax(weights_k, dim=-1).item())]
    default_expert = active_experts[-1]
    with layerwise_lora_routing_context(
        weights=weights_k,
        default_expert=int(default_expert),
        active_experts=active_experts,
    ):
        return greedy_generate(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
        )


@torch.no_grad()
def score_generated_example(
    task: str,
    pred_text: str,
    target_text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> float:
    metrics = compute_task_official_metrics(task, [pred_text], [target_text], metas=[dict(meta or {})])
    return float(metrics["main_score"])


def evaluate_prediction_records(
    records: Sequence[Dict[str, Any]],
    *,
    metric_mode: str = "local",
) -> Dict[str, Dict[str, Any]]:
    metric_mode = str(metric_mode or "local").strip().lower()
    if metric_mode not in {"local", "none", "llm_judge"}:
        raise ValueError(f"Unsupported metric_mode: {metric_mode}")

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        split = str(row.get("split", "eval"))
        task = str(row.get("task", "")).lower()
        if not task:
            continue
        grouped.setdefault(split, {}).setdefault(task, []).append(row)

    out: Dict[str, Dict[str, Any]] = {}
    for split, tasks in grouped.items():
        task_metrics: Dict[str, float] = {}
        task_official_metrics: Dict[str, Dict[str, float]] = {}
        task_losses: Dict[str, float] = {}
        total_examples = 0
        weighted_loss_sum = 0.0
        for task, rows in sorted(tasks.items()):
            losses = [
                float(row["loss"])
                for row in rows
                if row.get("loss") is not None and np.isfinite(float(row["loss"]))
            ]
            task_loss = float(np.mean(losses)) if losses else 0.0
            task_losses[task] = task_loss
            total_examples += len(rows)
            weighted_loss_sum += len(rows) * task_loss
            if metric_mode == "local":
                preds = [str(row.get("prediction", "")) for row in rows]
                targets = [str(row.get("target", "")) for row in rows]
                metas = [dict(row.get("meta", {}) or {}) for row in rows]
                official = compute_task_official_metrics(task, preds, targets, metas=metas)
                task_metrics[task] = float(official["main_score"])
                task_official_metrics[task] = official
        out[split] = {
            "n_examples": int(total_examples),
            "avg_macro": float(np.mean(list(task_metrics.values()))) if task_metrics else None,
            "avg_loss": float(weighted_loss_sum / max(1, total_examples)),
            "task_metrics_computed": bool(metric_mode == "local"),
            "metric_mode": metric_mode,
            "task_metrics": task_metrics,
            "task_official_metrics": task_official_metrics,
            "task_losses": task_losses,
        }
    return out


def _hidden_size_from_model(model: Any) -> int:
    for config_owner in (
        getattr(model, "config", None),
        getattr(getattr(model, "base_model", None), "config", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "config", None),
    ):
        if config_owner is None:
            continue
        hidden_size = getattr(config_owner, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise RuntimeError("Unable to determine hidden size for router construction.")


def load_tokenizer_prefer_local(
    model_name: str,
    *,
    use_fast: bool = True,
) -> AutoTokenizer:
    if _prefer_local_hf_files():
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast,
                local_files_only=True,
            )
            ensure_tokenizer_padding(tokenizer)
            return tokenizer
        except Exception:
            if _transformers_offline_enabled():
                raise
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    ensure_tokenizer_padding(tokenizer)
    return tokenizer


def build_model_and_tokenizer(
    model_name: str,
    dtype: torch.dtype,
    use_4bit: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    gradient_checkpointing: bool,
) -> Tuple[Any, AutoTokenizer, int]:
    configure_torch_runtime()
    device = get_device()
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model_kwargs: Dict[str, Any] = {
        "quantization_config": quant_config,
    }
    if not use_4bit:
        model_kwargs["dtype"] = dtype
    if device.type == "cuda" and use_4bit:
        model_kwargs["device_map"] = "auto"
    attn_impl = "flash_attention_2" if is_flash_attn2_available() else "sdpa"
    model_kwargs["attn_implementation"] = attn_impl

    prefer_local = _prefer_local_hf_files()
    offline_only = _transformers_offline_enabled()

    def _load_model(local_files_only: bool) -> Any:
        kwargs = dict(model_kwargs)
        kwargs["local_files_only"] = local_files_only
        try:
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
                try:
                    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                except TypeError:
                    pass
            kwargs.pop("attn_implementation", None)
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if prefer_local:
        try:
            model = _load_model(local_files_only=True)
        except Exception:
            if offline_only:
                raise
            model = _load_model(local_files_only=False)
    else:
        model = _load_model(local_files_only=False)

    if device.type == "cuda" and not use_4bit:
        model = model.to(device)

    tokenizer = load_tokenizer_prefer_local(model_name, use_fast=True)
    if getattr(model, "resize_token_embeddings", None) is not None:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = build_lora_config(lora_r, lora_alpha, lora_dropout, target_modules)
    _replace_target_linear_with_layerwise_lora(
        model,
        rank=int(lora_config["rank"]),
        alpha=int(lora_config["alpha"]),
        dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
    )
    _attach_layerwise_lora_api(model)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    freeze_all_params(model)
    set_trainable_adapter(model, "default")
    return model, tokenizer, _hidden_size_from_model(model)
