#!/usr/bin/env bash
# P3-3 ablation: retrain the best rewriter's LoRA with the synthetic pairs mixed into the fit examples (same recipe, 2 epochs,
# checkpoint per epoch; epoch chosen on held-out train). Env: MODEL (HF id), OUT (adapter dir), EXTRA (synthetic jsonl),
# DEVMAP (auto = shard over both GPUs [72B]; 0/1 = single GPU), NOTE (ledger note).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
: "${MODEL:?}"; : "${OUT:?}"; EXTRA=${EXTRA:-data_synth/synth.jsonl}; DEVMAP=${DEVMAP:-auto}
GPUS=2; VIS=""; if [ "$DEVMAP" != auto ]; then GPUS=1; VIS="CUDA_VISIBLE_DEVICES=$DEVMAP "; DEVARG=0; else DEVARG=auto; fi
NOTE=${NOTE:-"LoRA-tune $MODEL on P2 fit + synthetic pairs ($EXTRA), 2 epochs, adapters ep1/ep2 (device_map $DEVMAP)"}
seg_open $GPUS "$NOTE"
mark "P3_3_TRAIN_START" "$OUT"
set +e
env $VIS $TPY -m ocrgate.lora_train --config $CFG --model "$MODEL" --out "$OUT" --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch --device-map $DEVARG --extra-jsonl "$EXTRA" > "logs/p3_lora_$(basename $OUT).log" 2>&1
RC=$?; set -e
seg_close
mark "P3_3_TRAIN_END" "$OUT rc=$RC"
echo "SYNTH TRAIN DONE rc=$RC $(now)"
