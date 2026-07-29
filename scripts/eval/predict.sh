#!/usr/bin/env bash
# =============================================================================
# Prediction / evaluation script for FedWeave checkpoints.
#
# Three ways to specify checkpoints:
#   1. Single checkpoint:      --checkpoint PATH
#   2. Glob pattern:           --checkpoint_glob 'outputs/**/checkpoints/*.pt'
#   3. Root + names:           --checkpoint_root OUTPUTS_DIR --checkpoint_names best_val_loss.pt,final.pt
#
# Usage:
#   bash scripts/eval/predict.sh --checkpoint outputs/fedweave/alpha_0p5/seed_42/checkpoints/final.pt
#   bash scripts/eval/predict.sh --checkpoint_root outputs/fedweave --gpu 0
#   bash scripts/eval/predict.sh --checkpoint_root outputs/fedweave SPLIT=val HF_OFFLINE=true
#   bash scripts/eval/predict.sh --checkpoint_glob 'outputs/fedweave/**/checkpoints/best_val_loss.pt'
#
# All parameters below can be overridden via environment variables.
# =============================================================================
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

# ── GPU ────────────────────────────────────────────────────────────────────

GPU="${GPU:-}"

# ── Checkpoint selection ───────────────────────────────────────────────────
# Specify at least one of these (via CLI or env).

CHECKPOINT="${CHECKPOINT:-}"            # Single .pt path
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-}"   # Glob pattern, e.g. 'outputs/fedweave/**/checkpoints/*.pt'
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"   # Root directory to search for checkpoints/<name>.pt
CHECKPOINT_NAMES="${CHECKPOINT_NAMES:-best_val_loss.pt}"

# ── Evaluation parameters ──────────────────────────────────────────────────

SPLIT="${SPLIT:-test}"                  # val | test | both
INCLUDE_PREDICTIONS="${INCLUDE_PREDICTIONS:-true}"
METRIC_MODE="${METRIC_MODE:-local}"     # local | none | llm_judge

# ── Generation limits (override checkpoint defaults) ───────────────────────

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"  # empty/0 = keep checkpoint default auto sizing
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-}"
EVAL_MAX_NEW_TOKENS_BY_TASK="${EVAL_MAX_NEW_TOKENS_BY_TASK:-text_editing:64,struct_to_text:64,summarization:64,math_reasoning:192,intent_detection:8,sentiment_analysis:4,commonsense_reasoning:4}"
EVAL_ROUTING="${EVAL_ROUTING:-}"          # checkpoint | soft | topk
EVAL_TOP_K="${EVAL_TOP_K:-}"              # e.g. 2 with EVAL_ROUTING=topk
EVAL_M_TAU="${EVAL_M_TAU:-}"              # topk defaults to 1.0 unless overridden

# ── Output files ───────────────────────────────────────────────────────────

OUTPUT_JSON="${OUTPUT_JSON:-}"           # Single-checkpoint result JSON
BATCH_OUTPUT_JSON="${BATCH_OUTPUT_JSON:-}"  # Batch result JSON
PREDICTIONS_JSONL="${PREDICTIONS_JSONL:-}"  # Predictions JSONL (single ckpt only)

# ── Runtime ────────────────────────────────────────────────────────────────

HF_OFFLINE="${HF_OFFLINE:-}"            # true/false override; empty = keep checkpoint config
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"

# ── Logging ────────────────────────────────────────────────────────────────

USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-FedWeave}"
SWANLAB_NAME="${SWANLAB_NAME:-}"
SWANLAB_LOG_EVERY_N_EXAMPLES="${SWANLAB_LOG_EVERY_N_EXAMPLES:-10}"

# ── Help ───────────────────────────────────────────────────────────────────

print_help() {
  cat <<'EOF'
Evaluate checkpoints without retraining — generate predictions and score them.

Checkpoint selection (choose one approach):

  --checkpoint PATH              Single checkpoint .pt file
  --checkpoint_glob GLOB         Glob pattern (recursive), e.g.
                                 'outputs/fedweave/**/checkpoints/best_val_loss.pt'
  --checkpoint_root DIR          Root dir to search for checkpoints/<name>.pt
                                 (combine with --checkpoint_names)

Common examples:

  # Single checkpoint
  bash scripts/eval/predict.sh --checkpoint outputs/fedweave/alpha_0p5/seed_42/checkpoints/final.pt

  # All checkpoints under the FedWeave output tree, test split only
  bash scripts/eval/predict.sh --checkpoint_root outputs/fedweave

  # Specific checkpoint names in a root
  CHECKPOINT_NAMES=best_val_loss.pt SPLIT=test bash scripts/eval/predict.sh --checkpoint_root outputs/fedweave

  # Batch eval with error tolerance
  CHECKPOINT_ROOT=outputs/fedweave CONTINUE_ON_ERROR=true bash scripts/eval/predict.sh

  # Override generation tokens per task
  EVAL_MAX_NEW_TOKENS_BY_TASK=commonsense_reasoning:4,math_reasoning:192 bash scripts/eval/predict.sh --checkpoint ...

  # Override eval batch size for generation-heavy metrics
  EVAL_BATCH_SIZE=4 bash scripts/eval/predict.sh --checkpoint ...

  # Eval-only routing overrides for FedWeave
  EVAL_ROUTING=soft bash scripts/eval/predict.sh --checkpoint ...
  EVAL_ROUTING=topk EVAL_TOP_K=2 bash scripts/eval/predict.sh --checkpoint ...

  # Offline mode + no SwanLab
  HF_OFFLINE=true USE_SWANLAB=false bash scripts/eval/predict.sh --checkpoint ...

  # LLM-judge mode (export judge-ready JSONL, no local scoring)
  METRIC_MODE=llm_judge bash scripts/eval/predict.sh --checkpoint ...

Environment variables match the uppercase names in the parameter section.
CLI --gpu ID is also supported.
EOF
}

# ── Parse CLI ──────────────────────────────────────────────────────────────

GPU_DEVICES="$GPU"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)  print_help; exit 0 ;;
    --gpu|--gpus)
      GPU_DEVICES="$2"; shift 2 ;;
    --gpu=*|--gpus=*)
      GPU_DEVICES="${1#*=}"; shift ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --checkpoint=*)
      CHECKPOINT="${1#*=}"; shift ;;
    --checkpoint_glob)
      CHECKPOINT_GLOB="$2"; shift 2 ;;
    --checkpoint_glob=*)
      CHECKPOINT_GLOB="${1#*=}"; shift ;;
    --checkpoint_root)
      CHECKPOINT_ROOT="$2"; shift 2 ;;
    --checkpoint_root=*)
      CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --checkpoint_names)
      CHECKPOINT_NAMES="$2"; shift 2 ;;
    --checkpoint_names=*)
      CHECKPOINT_NAMES="${1#*=}"; shift ;;
    *)  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$GPU_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

# ── Build args ─────────────────────────────────────────────────────────────

NO_CHECKPOINT_SPECIFIED=true

ARGS=()

# Checkpoint flags
if [[ -n "$CHECKPOINT" ]]; then
  split_csv "$CHECKPOINT"
  for ckpt in "${SPLIT_RESULT[@]}"; do
    [[ -n "$ckpt" ]] && ARGS+=(--checkpoint "$ckpt")
  done
  NO_CHECKPOINT_SPECIFIED=false
fi

if [[ -n "$CHECKPOINT_GLOB" ]]; then
  split_csv "$CHECKPOINT_GLOB"
  for glob_pat in "${SPLIT_RESULT[@]}"; do
    [[ -n "$glob_pat" ]] && ARGS+=(--checkpoint_glob "$glob_pat")
  done
  NO_CHECKPOINT_SPECIFIED=false
fi

if [[ -n "$CHECKPOINT_ROOT" ]]; then
  split_csv "$CHECKPOINT_ROOT"
  for root in "${SPLIT_RESULT[@]}"; do
    [[ -n "$root" ]] && ARGS+=(--checkpoint_root "$root")
  done
  NO_CHECKPOINT_SPECIFIED=false
fi

if $NO_CHECKPOINT_SPECIFIED; then
  echo "error: no checkpoint specified. Use --checkpoint, --checkpoint_glob, or --checkpoint_root." >&2
  print_help
  exit 2
fi

# Core eval args
ARGS+=(
  --checkpoint_names "$CHECKPOINT_NAMES"
  --split "$SPLIT"
  $(bool_flag "$INCLUDE_PREDICTIONS" --include_predictions --no_include_predictions)
  --metric_mode "$METRIC_MODE"
  $(bool_flag "$CONTINUE_ON_ERROR" --continue_on_error --no_continue_on_error)
)

# Optional output paths
[[ -n "$OUTPUT_JSON" ]] && ARGS+=(--output_json "$OUTPUT_JSON")
[[ -n "$BATCH_OUTPUT_JSON" ]] && ARGS+=(--batch_output_json "$BATCH_OUTPUT_JSON")
[[ -n "$PREDICTIONS_JSONL" ]] && ARGS+=(--predictions_jsonl "$PREDICTIONS_JSONL")

# Generation overrides
[[ -n "$EVAL_BATCH_SIZE" ]] && ARGS+=(--eval_batch_size "$EVAL_BATCH_SIZE")
[[ -n "$EVAL_MAX_NEW_TOKENS" ]] && ARGS+=(--eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS")
[[ -n "$EVAL_MAX_NEW_TOKENS_BY_TASK" ]] && ARGS+=(--eval_max_new_tokens_by_task "$EVAL_MAX_NEW_TOKENS_BY_TASK")
[[ -n "$EVAL_ROUTING" ]] && ARGS+=(--eval_routing "$EVAL_ROUTING")
[[ -n "$EVAL_TOP_K" ]] && ARGS+=(--eval_top_k "$EVAL_TOP_K")
[[ -n "$EVAL_M_TAU" ]] && ARGS+=(--eval_m_tau "$EVAL_M_TAU")

# SwanLab
ARGS+=(
  $(bool_flag "$USE_SWANLAB" --use_swanlab --no_use_swanlab)
  --swanlab_mode "$SWANLAB_MODE"
  --swanlab_project "$SWANLAB_PROJECT"
  --swanlab_name "$SWANLAB_NAME"
  --swanlab_log_every_n_examples "$SWANLAB_LOG_EVERY_N_EXAMPLES"
)

# HF offline override
if [[ -n "$HF_OFFLINE" ]]; then
  ARGS+=($(bool_flag "$HF_OFFLINE" --hf_offline --no_hf_offline))
fi

# ── Run ────────────────────────────────────────────────────────────────────

"$PYTHON_BIN" "$PROJECT_DIR/src/predict.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
