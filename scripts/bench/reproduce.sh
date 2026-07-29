#!/usr/bin/env bash
# Reproduce FedWeave runs for one paper backbone. Run once per backbone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

BACKBONE="${BACKBONE:-llama}"
SEED="${SEED:-42,43,44}"
ALPHA="${ALPHA:-0.3}"

case "$BACKBONE" in
  llama)
    MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B}"
    ;;
  gemma)
    MODEL_NAME="${MODEL_NAME:-google/gemma-2-2b}"
    ;;
  *)
    echo "error: BACKBONE must be 'llama' or 'gemma'." >&2
    exit 2
    ;;
esac

export MODEL_NAME SEED ALPHA
exec bash "$PROJECT_DIR/scripts/train/fedweave.sh" "$@"
