#!/usr/bin/env bash
# Start vLLM (OpenAI-compatible) for Qwen/Qwen3-8B on 127.0.0.1:8000.
# The API key is read from $VLLM_API_KEY_FILE (default $HOME/.config/vllm/api_key) and passed via the VLLM_API_KEY environment
# variable (not on the command line, so it does not appear in `ps`). It is never printed.
#
# Env overrides: MODEL, PORT, DP (data-parallel replicas, default 2 = one per H200), TP (tensor
# parallel, default 1), MAX_MODEL_LEN, GPU_UTIL, VLLM_ENV, VLLM_API_KEY_FILE, LORA_MODULES
# (Phase 1: 'name=/path/to/adapter' -> --enable-lora; requests with model=name use the adapter).
set -euo pipefail
cd "$(dirname "$0")"

VLLM_ENV=${VLLM_ENV:-$HOME/vllm-env}
KEY_FILE=${VLLM_API_KEY_FILE:-$HOME/.config/vllm/api_key}
MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-8000}
DP=${DP:-2}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}
LORA_MODULES=${LORA_MODULES:-}
# Phase 4: EXTRA_ARGS = additional vllm serve flags (word-split on purpose), e.g. '--language-model-only' for
# Qwen3.8-27B (Qwen3_5ForConditionalGeneration served text-only). Empty by default: earlier phases are unchanged.
EXTRA_ARGS=${EXTRA_ARGS:-}
# shellcheck disable=SC2206
EXTRA=($EXTRA_ARGS)
LORA_ARGS=()
if [ -n "$LORA_MODULES" ]; then
  # several adapters: space-separated name=path pairs (word-split on purpose)
  # shellcheck disable=SC2206
  LORA_ARGS=(--enable-lora --lora-modules $LORA_MODULES --max-lora-rank "${MAX_LORA_RANK:-16}" --max-loras "${MAX_LORAS:-2}")
fi

if [ ! -r "$KEY_FILE" ]; then
  echo "API key file $KEY_FILE not found or unreadable" >&2
  exit 1
fi
VLLM_API_KEY="$(tr -d '[:space:]' < "$KEY_FILE")"
if [ -z "$VLLM_API_KEY" ]; then
  echo "API key file $KEY_FILE is empty" >&2
  exit 1
fi
export VLLM_API_KEY

mkdir -p logs
LOG=logs/vllm_$(date -u +%Y%m%dT%H%M%SZ).log
echo "Serving $MODEL on 127.0.0.1:$PORT (DP=$DP, TP=$TP); log: $LOG"

# vLLM 0.32 tries to JIT-compile FlashInfer's top-k/top-p sampling kernel during warm-up, which
# needs ninja + nvcc. Neither was installed on our GPU server, so the engine died with
# "No such file or directory: 'ninja'". Fall back to the native PyTorch sampler; with temperature 0
# (greedy) the sampler kernel does not affect the outputs.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

# --generation-config vllm: ignore the model's generation_config.json sampling defaults, so the
# client's explicit parameters (temperature 0, seed) are the only ones in effect.
exec "$VLLM_ENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name "$MODEL" \
  --data-parallel-size "$DP" --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config vllm \
  --seed 1234 \
  "${LORA_ARGS[@]}" "${EXTRA[@]}" \
  >>"$LOG" 2>&1
