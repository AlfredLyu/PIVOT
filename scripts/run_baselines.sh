#!/usr/bin/env bash
# Generic-MLLM visual-context wrappers (Table 1), all under the same 50K
# token-equivalent budget.
#   API models need GEMINI_API_KEY / OPENAI_API_KEY.
#   Open-source models (qwen2.5-vl-7b, llava-onevision-7b, internvl3-8b) run
#   locally through HuggingFace transformers on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--chunk_size 16 --frame_stride 2 --max_visual_tokens 50000 --max_output_tokens 8192"

# --- FIFO buffer: API models ---
for M in gemini-3.1-flash-lite-preview gpt-4o; do
  PYTHONHASHSEED=42 python -u baselines/mllm_fifo.py --model_name $M --num_workers 8 $COMMON "$@"
done

# --- FIFO buffer: open-source models (one at a time on a single GPU) ---
for M in qwen2.5-vl-7b llava-onevision-7b internvl3-8b; do
  PYTHONHASHSEED=42 python -u baselines/mllm_fifo.py --model_name $M --num_workers 1 \
    --hf_dtype bfloat16 --hf_device auto --hf_max_pixels 200704 $COMMON "$@"
done

# --- Reservoir buffer: API models ---
for M in gemini-3.1-flash-lite-preview gpt-4o; do
  PYTHONHASHSEED=42 python -u baselines/mllm_reservoir.py --model_name $M --num_workers 8 $COMMON "$@"
done
