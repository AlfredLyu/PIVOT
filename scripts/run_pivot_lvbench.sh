#!/usr/bin/env bash
# PIVOT on LVBench-Ref2Stream (Table 1). Backbone: Gemini-3.1-Flash-Lite-Preview,
# 0.5 fps (1 fps frames, stride 2), 16 frames / chunk, 50K-token memory budget.
# Requires GEMINI_API_KEY. Extra args are forwarded (e.g. --subset_path ...).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONHASHSEED=42 python -u pivot/run_pivot.py \
  --model_name gemini-3.1-flash-lite-preview \
  --chunk_size 16 \
  --frame_stride 2 \
  --max_memory_tokens 50000 \
  --max_tool_rounds 5 \
  --num_workers 8 \
  "$@"
