#!/usr/bin/env bash
# Shared helpers for all FedWeave scripts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

bool_flag() {
  local value="$1"
  local true_flag="$2"
  local false_flag="$3"
  if is_true "$value"; then
    printf '%s' "$true_flag"
  else
    printf '%s' "$false_flag"
  fi
}

split_csv() {
  local value="$1"
  value="${value// /}"
  IFS=',' read -r -a SPLIT_RESULT <<< "$value"
}

sanitize_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/}"
  value="${value//\//_}"
  value="${value//:/_}"
  value="${value// /_}"
  printf '%s' "$value"
}

# --- GPU handling -----------------------------------------------------------

parse_gpu_arg() {
  # Parse --gpu / --gpus from remaining positional args.
  # Sets GPU_DEVICES and removes those tokens from the arg list.
  GPU_DEVICES=""
  local tmp=()
  local i=0
  while [[ $i -lt $# ]]; do
    local arg="${!i}"
    case "$arg" in
      --gpu|--gpus)
        i=$((i + 1))
        if [[ $i -lt $# ]]; then
          GPU_DEVICES="${!i}"
        else
          echo "error: $arg requires a GPU id, e.g. --gpu 0 or --gpu 0,1" >&2
          exit 2
        fi
        ;;
      --gpu=*|--gpus=*)
        GPU_DEVICES="${arg#*=}"
        ;;
      *)
        tmp+=("$arg")
        ;;
    esac
    i=$((i + 1))
  done
  if [[ -n "$GPU_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  fi
  # Return remaining args by printing them — caller should capture with $(parse_gpu_arg "$@")
  printf '%s\n' "${tmp[@]}"
}

# --- Build common base args (data, model, lora, training, logging) ----------

build_base_args() {
  # This function echoes flags for every ExperimentConfig parameter.
  # Training launchers call this and then add FedWeave-specific flags.
  cat <<ARGS
--out_dir $OUT_DIR
--results_dir $RESULTS_DIR
--seed $SEED
--tasks $TASKS
--train_samples_per_task $TRAIN_SAMPLES_PER_TASK
--val_samples_per_task $VAL_SAMPLES_PER_TASK
--test_samples_per_task $TEST_SAMPLES_PER_TASK
--sample_seed $SAMPLE_SEED
--client_partition_seed $CLIENT_PARTITION_SEED
--num_clients $NUM_CLIENTS
--dirichlet_alpha $DIRICHLET_ALPHA
--arc_variant $ARC_VARIANT
$(bool_flag "$HF_OFFLINE" --hf_offline --no_hf_offline)
--hf_download_timeout $HF_DOWNLOAD_TIMEOUT
--hf_etag_timeout $HF_ETAG_TIMEOUT
--hf_max_retries $HF_MAX_RETRIES
$(bool_flag "$REUSE_SPLITS" --reuse_splits --no_reuse_splits)
--data_cache_dir $DATA_CACHE_DIR
--model_name $MODEL_NAME
--dtype $DTYPE
$(bool_flag "$USE_4BIT" --use_4bit --no_use_4bit)
$(bool_flag "$GRADIENT_CHECKPOINTING" --gradient_checkpointing --no_gradient_checkpointing)
--system_prompt $SYSTEM_PROMPT
--prompt_format $PROMPT_FORMAT
--lora_r $LORA_R
--lora_alpha $LORA_ALPHA
--lora_dropout $LORA_DROPOUT
--target_modules $TARGET_MODULES
--global_rounds $GLOBAL_ROUNDS
--local_steps $LOCAL_STEPS
--batch_size $BATCH_SIZE
--grad_accum $GRAD_ACCUM
--lr $LR
--router_lr $ROUTER_LR
--local_lr_schedule $LOCAL_LR_SCHEDULE
--wd $WD
--grad_clip $GRAD_CLIP
--max_length $MAX_LENGTH
--save_every_n_rounds $SAVE_EVERY_N_ROUNDS
$(bool_flag "$EVAL_EVERY_ROUND" --eval_every_round --no_eval_every_round)
--eval_every_n_rounds $EVAL_EVERY_N_ROUNDS
$(bool_flag "$VAL_COMPUTE_TASK_METRICS" --val_compute_task_metrics --no_val_compute_task_metrics)
--val_task_metrics_every_n_rounds $VAL_TASK_METRICS_EVERY_N_ROUNDS
--eval_max_new_tokens $EVAL_MAX_NEW_TOKENS
--eval_max_new_tokens_by_task $EVAL_MAX_NEW_TOKENS_BY_TASK
$(bool_flag "$SHOW_PROGRESS" --show_progress --no_show_progress)
$(bool_flag "$USE_SWANLAB" --use_swanlab --no_use_swanlab)
--swanlab_mode $SWANLAB_MODE
--swanlab_project $SWANLAB_PROJECT
ARGS
}
