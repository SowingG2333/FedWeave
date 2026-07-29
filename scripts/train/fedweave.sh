#!/usr/bin/env bash
# =============================================================================
# FedWeave training script.
#
# Usage:
#   bash scripts/train/fedweave.sh                        # single run, defaults
#   bash scripts/train/fedweave.sh --gpu 0               # pick GPU
#   SEED=42,43 ALPHA=0.3,0.5 bash scripts/train/fedweave.sh  # sweep
#
# All parameters below can be overridden via environment variables.
# =============================================================================
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../common.sh"

# ── GPU ────────────────────────────────────────────────────────────────────

GPU="${GPU:-}"

# ── Output paths ───────────────────────────────────────────────────────────

OUT_DIR="${OUT_DIR:-$PROJECT_DIR/outputs/runs}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results/runs}"
TRAIN_OUT_ROOT="${TRAIN_OUT_ROOT:-$PROJECT_DIR/outputs/fedweave}"

# ── Seeds & alphas ─────────────────────────────────────────────────────────

SEED="${SEED:-42}"
ALPHA="${ALPHA:-0.3}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-}"   # empty = use ALPHA value

# ── Data ───────────────────────────────────────────────────────────────────

TASKS="${TASKS:-text_editing,math_reasoning,sentiment_analysis,commonsense_reasoning}"
TRAIN_SAMPLES_PER_TASK="${TRAIN_SAMPLES_PER_TASK:-2000}"
VAL_SAMPLES_PER_TASK="${VAL_SAMPLES_PER_TASK:-100}"
TEST_SAMPLES_PER_TASK="${TEST_SAMPLES_PER_TASK:-400}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
CLIENT_PARTITION_SEED="${CLIENT_PARTITION_SEED:-42}"
NUM_CLIENTS="${NUM_CLIENTS:-20}"
ARC_VARIANT="${ARC_VARIANT:-ARC-Challenge}"
HF_OFFLINE="${HF_OFFLINE:-false}"
HF_DOWNLOAD_TIMEOUT="${HF_DOWNLOAD_TIMEOUT:-180}"
HF_ETAG_TIMEOUT="${HF_ETAG_TIMEOUT:-30}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-5}"
REUSE_SPLITS="${REUSE_SPLITS:-true}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$PROJECT_DIR/outputs/data_cache}"

# ── Model & LoRA ───────────────────────────────────────────────────────────

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B}"
DTYPE="${DTYPE:-bf16}"
USE_4BIT="${USE_4BIT:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
PROMPT_FORMAT="${PROMPT_FORMAT:-auto}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"

# ── Router ─────────────────────────────────────────────────────────────────

ROUTER_HIDDEN="${ROUTER_HIDDEN:-512}"
ROUTER_DROPOUT="${ROUTER_DROPOUT:-0.0}"
M_SELECT="${M_SELECT:-0}"
M_TAU="${M_TAU:-0.8}"
ROUTE_CE_WEIGHT="${ROUTE_CE_WEIGHT:-0.0}"

# ── Training ───────────────────────────────────────────────────────────────

GLOBAL_ROUNDS="${GLOBAL_ROUNDS:-20}"
LOCAL_STEPS="${LOCAL_STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"
ROUTER_LR="${ROUTER_LR:-5e-5}"
LOCAL_LR_SCHEDULE="${LOCAL_LR_SCHEDULE:-constant}"
WD="${WD:-0.0}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"

# ── Eval during training ───────────────────────────────────────────────────

SAVE_EVERY_N_ROUNDS="${SAVE_EVERY_N_ROUNDS:-5}"
EVAL_EVERY_ROUND="${EVAL_EVERY_ROUND:-true}"
EVAL_EVERY_N_ROUNDS="${EVAL_EVERY_N_ROUNDS:-1}"
VAL_COMPUTE_TASK_METRICS="${VAL_COMPUTE_TASK_METRICS:-false}"
VAL_TASK_METRICS_EVERY_N_ROUNDS="${VAL_TASK_METRICS_EVERY_N_ROUNDS:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-64}"
EVAL_MAX_NEW_TOKENS_BY_TASK="${EVAL_MAX_NEW_TOKENS_BY_TASK:-text_editing:64,struct_to_text:64,summarization:64,math_reasoning:192,intent_detection:8,sentiment_analysis:4,commonsense_reasoning:4}"

# ── Discovery (FedWeave-specific) ──────────────────────────────────────────

EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"
BUCKET_MIN_STEPS="${BUCKET_MIN_STEPS:-1}"
DISCOVERY_WARMUP_STEPS="${DISCOVERY_WARMUP_STEPS:-10}"
DISCOVERY_WARMUP_BATCH_SIZE="${DISCOVERY_WARMUP_BATCH_SIZE:-8}"
NORMALIZE_EMBEDDINGS="${NORMALIZE_EMBEDDINGS:-true}"
INTERLEAVE_CLIENT_BUCKETS="${INTERLEAVE_CLIENT_BUCKETS:-true}"
ROUTER_AGGREGATION_SCOPE="${ROUTER_AGGREGATION_SCOPE:-client}"
ORACLE_TASK_ROUTING="${ORACLE_TASK_ROUTING:-false}"
LOCAL_CLUSTER_ALGORITHM="${LOCAL_CLUSTER_ALGORITHM:-kmeans}"
PROTOTYPE_SIGNATURE_TYPE="${PROTOTYPE_SIGNATURE_TYPE:-lora_b}"

# ── Logging ────────────────────────────────────────────────────────────────

SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
USE_SWANLAB="${USE_SWANLAB:-false}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-FedWeave}"
SWANLAB_NAME_PREFIX="${SWANLAB_NAME_PREFIX:-FedWeave}"

# ── Help ───────────────────────────────────────────────────────────────────

print_help() {
  cat <<'EOF'
FedWeave — Prototype-aligned MoE Federated LoRA.

Usage:
  bash scripts/train/fedweave.sh [--gpu ID] [--extra ...]

All parameters are at the top of the script; override via env vars:

  # Quick GPU switch
  bash scripts/train/fedweave.sh --gpu 0

  # Sweep seeds and alphas
  SEED=42,43,44 ALPHA=0.3,0.5 bash scripts/train/fedweave.sh

  # Change tasks
  TASKS=commonsense_reasoning,sentiment_analysis bash scripts/train/fedweave.sh

  # Quiet / no SwanLab
  SHOW_PROGRESS=false USE_SWANLAB=false bash scripts/train/fedweave.sh

  # Full custom run
  MODEL_NAME=google/gemma-2-2b GLOBAL_ROUNDS=30 LR=5e-5 bash scripts/train/fedweave.sh

  # Router bucket-aggregation ablation
  ROUTER_AGGREGATION_SCOPE=bucket bash scripts/train/fedweave.sh

  # Cluster ablation
  LOCAL_CLUSTER_ALGORITHM=spectral PROTOTYPE_SIGNATURE_TYPE=lora_ab bash scripts/train/fedweave.sh
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
    *)  EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$GPU_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"

# ── Run ────────────────────────────────────────────────────────────────────

split_csv "$SEED";   SEEDS=("${SPLIT_RESULT[@]}")
split_csv "$ALPHA";  ALPHAS=("${SPLIT_RESULT[@]}")

for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    a="${DIRICHLET_ALPHA:-$alpha}"
    slug="$(sanitize_value "$a")"
    run_dir="$TRAIN_OUT_ROOT/alpha_$slug"
    sw_name="${SWANLAB_NAME_PREFIX}-alpha${a}-seed${seed}"

    echo "=== FedWeave seed=$seed alpha=$a out=$run_dir ==="

    "$PYTHON_BIN" "$PROJECT_DIR/src/train.py" \
      --train_out_dir "$run_dir" \
      --train_results_dir "$run_dir" \
      --out_dir "$OUT_DIR" \
      --results_dir "$RESULTS_DIR" \
      --seed "$seed" \
      --tasks "$TASKS" \
      --train_samples_per_task "$TRAIN_SAMPLES_PER_TASK" \
      --val_samples_per_task "$VAL_SAMPLES_PER_TASK" \
      --test_samples_per_task "$TEST_SAMPLES_PER_TASK" \
      --sample_seed "$SAMPLE_SEED" \
      --client_partition_seed "$CLIENT_PARTITION_SEED" \
      --num_clients "$NUM_CLIENTS" \
      --dirichlet_alpha "$a" \
      --arc_variant "$ARC_VARIANT" \
      $(bool_flag "$HF_OFFLINE" --hf_offline --no_hf_offline) \
      --hf_download_timeout "$HF_DOWNLOAD_TIMEOUT" \
      --hf_etag_timeout "$HF_ETAG_TIMEOUT" \
      --hf_max_retries "$HF_MAX_RETRIES" \
      $(bool_flag "$REUSE_SPLITS" --reuse_splits --no_reuse_splits) \
      --data_cache_dir "$DATA_CACHE_DIR" \
      --model_name "$MODEL_NAME" \
      --dtype "$DTYPE" \
      $(bool_flag "$USE_4BIT" --use_4bit --no_use_4bit) \
      $(bool_flag "$GRADIENT_CHECKPOINTING" --gradient_checkpointing --no_gradient_checkpointing) \
      --system_prompt "$SYSTEM_PROMPT" \
      --prompt_format "$PROMPT_FORMAT" \
      --lora_r "$LORA_R" \
      --lora_alpha "$LORA_ALPHA" \
      --lora_dropout "$LORA_DROPOUT" \
      --target_modules "$TARGET_MODULES" \
      --router_hidden "$ROUTER_HIDDEN" \
      --router_dropout "$ROUTER_DROPOUT" \
      --m_select "$M_SELECT" \
      --m_tau "$M_TAU" \
      --route_ce_weight "$ROUTE_CE_WEIGHT" \
      --global_rounds "$GLOBAL_ROUNDS" \
      --local_steps "$LOCAL_STEPS" \
      --batch_size "$BATCH_SIZE" \
      --grad_accum "$GRAD_ACCUM" \
      --lr "$LR" \
      --router_lr "$ROUTER_LR" \
      --local_lr_schedule "$LOCAL_LR_SCHEDULE" \
      --wd "$WD" \
      --grad_clip "$GRAD_CLIP" \
      --max_length "$MAX_LENGTH" \
      --save_every_n_rounds "$SAVE_EVERY_N_ROUNDS" \
      $(bool_flag "$EVAL_EVERY_ROUND" --eval_every_round --no_eval_every_round) \
      --eval_every_n_rounds "$EVAL_EVERY_N_ROUNDS" \
      $(bool_flag "$VAL_COMPUTE_TASK_METRICS" --val_compute_task_metrics --no_val_compute_task_metrics) \
      --val_task_metrics_every_n_rounds "$VAL_TASK_METRICS_EVERY_N_ROUNDS" \
      --eval_max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
      --eval_max_new_tokens_by_task "$EVAL_MAX_NEW_TOKENS_BY_TASK" \
      --embedding_batch_size "$EMBEDDING_BATCH_SIZE" \
      --bucket_min_steps "$BUCKET_MIN_STEPS" \
      --discovery_warmup_steps "$DISCOVERY_WARMUP_STEPS" \
      --discovery_warmup_batch_size "$DISCOVERY_WARMUP_BATCH_SIZE" \
      --local_cluster_algorithm "$LOCAL_CLUSTER_ALGORITHM" \
      --prototype_signature_type "$PROTOTYPE_SIGNATURE_TYPE" \
      $(bool_flag "$NORMALIZE_EMBEDDINGS" --normalize_embeddings --no_normalize_embeddings) \
      $(bool_flag "$INTERLEAVE_CLIENT_BUCKETS" --interleave_client_buckets --no_interleave_client_buckets) \
      --router_aggregation_scope "$ROUTER_AGGREGATION_SCOPE" \
      $(bool_flag "$ORACLE_TASK_ROUTING" --oracle_task_routing --no_oracle_task_routing) \
      $(bool_flag "$SHOW_PROGRESS" --show_progress --no_show_progress) \
      $(bool_flag "$USE_SWANLAB" --use_swanlab --no_use_swanlab) \
      --swanlab_mode "$SWANLAB_MODE" \
      --swanlab_project "$SWANLAB_PROJECT" \
      --swanlab_name "${sw_name}-${LOCAL_CLUSTER_ALGORITHM}-${PROTOTYPE_SIGNATURE_TYPE}" \
      "${EXTRA_ARGS[@]}"
  done
done
