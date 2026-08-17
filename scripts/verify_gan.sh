#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

XFORMERS_DISABLED=1 \
TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
python smoke_test.py --mode gan
