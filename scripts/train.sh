#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29618}"
CONFIG="${CONFIG:-configs/train.yaml}"

if [[ "$CONFIG" == *gan* ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
fi

XFORMERS_DISABLED=1 accelerate launch \
  --multi_gpu \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  train.py \
  --config "$CONFIG" \
  "$@"
