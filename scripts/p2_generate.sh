#!/usr/bin/env bash
# Phase 2 generation driver (DEV + TRAIN only; the test split is never passed). Steps (env STEPS, default all):
#   M  Mistral-Small-24B zero-shot on the full TRAIN split (DP=2)            -> gate training blocks (new family)
#   Q  Qwen2.5-72B zero-shot on the full TRAIN split (TP=2)                  -> gate training blocks + ensemble member
#   L  Mistral-24B + LoRA ep1/ep2 (TP=2): held-out train subset + dev         -> epoch selection on held-out, dev reporting
#   K  Qwen3-32B  + LoRA ep1/ep2 (TP=2): held-out train subset + dev (thinking kwarg ON, served under another name)
# Every vLLM lifetime is appended to results/server_segments_p2.tsv (start/end UTC, GPUs, note).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
SEG=results/server_segments_p2.tsv; STEPS=${STEPS:-MQLK}; CFG=configs/phase2.yaml; HO=configs/p2_heldout_subset.json
now() { date -u +%FT%TZ; }
nseg() { awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {n=$1} END {print n+1}' "$SEG"; }
VISIBLE=${VISIBLE:-}   # e.g. VISIBLE=0 to pin the server to GPU 0 (CUDA_VISIBLE_DEVICES inside the tmux string)
serve() { # MODEL DP TP "LORA_MODULES" NOTE
  local n; n=$(nseg); SEG_START=$(now); SEG_N=$n; SEG_GPUS=$(( $2 * $3 )); SEG_NOTE="$5"
  local vis=""; [ -n "$VISIBLE" ] && vis="CUDA_VISIBLE_DEVICES=$VISIBLE "
  tmux new -d -s vllm_p2 "cd $PWD && ${vis}VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=$1 DP=$2 TP=$3 LORA_MODULES='$4' ./serve_model.sh"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do sleep 5; done
  echo "[serve] $1 DP=$2 TP=$3 up at $(now)"
}
stop() {
  tmux kill-session -t vllm_p2 || true
  until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done
  sleep 3; printf '%s\t%s\t%s\t%s\t%s\n' "$SEG_N" "$SEG_START" "$(now)" "$SEG_GPUS" "$SEG_NOTE" >> "$SEG"
  echo "[stop] segment $SEG_N closed at $(now)"
}
LORA24=$PWD/runs/lora/mistral24b-p2; LORA32=$PWD/runs/lora/qwen3-32b-p2
blocks() { # TAG: label the edit blocks of the newest run with this tag (CPU)
  local run; run=$(ls -td runs/*_zeroshot_*_"$1" | head -1)
  $PY -m ocrgate.blocks --config $CFG --run "$run" | head -c 600; echo
}
DP_M=${DP_M:-2}

if [[ "$STEPS" == *M* ]]; then
  serve mistralai/Mistral-Small-24B-Instruct-2501 $DP_M 1 "" "vLLM Mistral-Small-24B DP=$DP_M: zero-shot on full TRAIN (gate blocks, new family)"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split train --model mistralai/Mistral-Small-24B-Instruct-2501 --tag mistral24b-p2
  stop; blocks mistral24b-p2
fi
if [[ "$STEPS" == *Q* ]]; then
  serve Qwen/Qwen2.5-72B-Instruct 1 2 "" "vLLM Qwen2.5-72B TP=2: zero-shot on full TRAIN (gate blocks, ensemble member)"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split train --model Qwen/Qwen2.5-72B-Instruct --tag q72b-p2
  stop; blocks q72b-p2
fi
if [[ "$STEPS" == *L* ]]; then
  serve mistralai/Mistral-Small-24B-Instruct-2501 1 2 "lora24b-ep1=$LORA24/ep1 lora24b-ep2=$LORA24/ep2" "vLLM Mistral-24B + P2 LoRA ep1/ep2 TP=2: held-out train subset + dev"
  for ep in 1 2; do
    $PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model lora24b-ep$ep --tag lora24b-ep$ep-ho
    $PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model lora24b-ep$ep --tag lora24b-ep$ep
  done
  stop; for ep in 1 2; do blocks lora24b-ep$ep-ho; blocks lora24b-ep$ep; done
fi
if [[ "$STEPS" == *K* ]]; then
  serve Qwen/Qwen3-32B 1 2 "lora32b-ep1=$LORA32/ep1 lora32b-ep2=$LORA32/ep2" "vLLM Qwen3-32B + P2 LoRA ep1/ep2 TP=2: held-out train subset + dev"
  for ep in 1 2; do
    $PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model lora32b-ep$ep --tag lora32b-ep$ep-ho --send-thinking-kwarg on
    $PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model lora32b-ep$ep --tag lora32b-ep$ep --send-thinking-kwarg on
  done
  stop; for ep in 1 2; do blocks lora32b-ep$ep-ho; blocks lora32b-ep$ep; done
fi
echo "P2 GENERATE DONE ($STEPS) $(now)"
