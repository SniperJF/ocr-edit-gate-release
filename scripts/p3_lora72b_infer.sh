#!/usr/bin/env bash
# P3-1 inference: serve Qwen2.5-72B-Instruct + the P3 LoRA adapters ep1/ep2 (TP=2), run the held-out train subset + dev for
# each epoch, label blocks; epoch selection on held-out (scripts/p2_select_epoch.py lora72b); then P2 gate on the chosen dev run.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
LORA=${LORA:-$PWD/runs/lora/qwen25-72b-p3}; TAG=${TAG:-lora72b}
serve Qwen/Qwen2.5-72B-Instruct 1 2 "$TAG-ep1=$LORA/ep1 $TAG-ep2=$LORA/ep2" "vLLM Qwen2.5-72B + P3 LoRA ep1/ep2 TP=2: held-out train subset + dev"
for ep in 1 2; do
  $PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model $TAG-ep$ep --tag $TAG-ep$ep-ho
  $PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model $TAG-ep$ep --tag $TAG-ep$ep
done
stop
for ep in 1 2; do blocks "$(latest_run $TAG-ep$ep-ho)"; blocks "$(latest_run $TAG-ep$ep)"; done
$PY scripts/p2_select_epoch.py lora24b lora32b $TAG | tee logs/p3_epoch_selection.md
EP=$($PY -c "import json;print(json.load(open('results/p2_epoch_selection.json'))['$TAG']['chosen_epoch'])")
DEV=$(latest_run $TAG-ep$EP); HOR=$(latest_run $TAG-ep$EP-ho)
echo "chosen epoch $EP: dev $DEV heldout $HOR"
seg_open 1 "GPU0: P2 gate inference on the LoRA-72B dev + held-out runs (full gate global, text-only gate, none, oracle)"
for RUN in $DEV $HOR; do
  CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode none
  CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode gate --gate runs/gates/p2_pooled --rule global
  CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode gate --gate runs/gates/p2_pooled_nolp --rule global
  CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode oracle_label
done
seg_close
mark "P3_1_INFER_DONE" "epoch $EP dev=$DEV ho=$HOR"
echo "P3-1 INFER DONE $(now)"
