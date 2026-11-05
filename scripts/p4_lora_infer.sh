#!/usr/bin/env bash
# P4-c inference: serve Qwen3.8-27B text-only + the P4 LoRA adapters ep1/ep2 (TP=1, GPU ${VISIBLE:-0}), run the held-out train
# subset + dev for each epoch, label blocks; epoch selection on held-out (scripts/p4_select_epoch.py). Marker P4_C_DONE.
# Env: DP (default 2 = both GPUs, data parallel), VISIBLE (pin GPUs), LORA, TAG.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p4_lib.sh
LORA=${LORA:-$PWD/runs/lora/qwen38-27b-p4}; TAG=${TAG:-qwen38-lora}
DP=${DP:-2}   # one TP=1 replica per H200 when both are free (same GPU-hours, half the wall time)
EXTRA_ARGS='--language-model-only' serve Qwen/Qwen3.8-27B $DP 1 "$TAG-ep1=$LORA/ep1 $TAG-ep2=$LORA/ep2" \
  "vLLM Qwen3.8-27B text-only DP=$DP TP=1 + P4 LoRA ep1/ep2: held-out train subset + dev"
for ep in 1 2; do
  $PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model $TAG-ep$ep --tag $TAG-ep$ep-ho --send-thinking-kwarg on 2>&1 | grep -v -i warn | tail -1
  $PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model $TAG-ep$ep --tag $TAG-ep$ep --send-thinking-kwarg on 2>&1 | grep -v -i warn | tail -1
done
stop
for ep in 1 2; do blocks "$(latest_run $TAG-ep$ep-ho)"; blocks "$(latest_run $TAG-ep$ep)"; done
$PY scripts/p4_select_epoch.py $TAG | tee logs/p4_epoch_selection.md
EP=$($PY -c "import json;print(json.load(open('results/p4_epoch_selection.json'))['$TAG']['chosen_epoch'])")
mark "P4_C_DONE" "Qwen3.8-27B LoRA (P2 recipe, attn+GDN+MLP targets): epoch $EP chosen on held-out; dev=$(latest_run $TAG-ep$EP) ho=$(latest_run $TAG-ep$EP-ho)"
echo "P4-c INFER DONE epoch $EP $(now)"
