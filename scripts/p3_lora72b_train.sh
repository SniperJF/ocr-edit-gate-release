#!/usr/bin/env bash
# P3-1: bf16 LoRA on Qwen2.5-72B-Instruct, same recipe as the P2 LoRA-32B (r16/a32, lr 1e-4, ga16, 2 epochs, fit role of
# configs/p2_split.json, impresso-nzz excluded, checkpoint per epoch), base model sharded over both H200s (device_map auto).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
OUT=${OUT:-runs/lora/qwen25-72b-p3}; EXTRA=${EXTRA:-}; NOTE=${NOTE:-"GPU0+1: bf16 LoRA-tune Qwen2.5-72B-Instruct (sharded over 2 GPUs) on the P2 fit part of train, 2 epochs, adapters ep1/ep2"}
EXTRA_ARGS=(); [ -n "$EXTRA" ] && EXTRA_ARGS=(--extra-jsonl "$EXTRA")
seg_open 2 "$NOTE"
mark "P3_1_TRAIN_START" "$OUT"
set +e
$TPY -m ocrgate.lora_train --config $CFG --model Qwen/Qwen2.5-72B-Instruct --out "$OUT" --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch --device-map auto "${EXTRA_ARGS[@]}" > "logs/p3_lora_$(basename $OUT).log" 2>&1
RC=$?; set -e
seg_close
mark "P3_1_TRAIN_END" "$OUT rc=$RC"
echo "TRAIN DONE rc=$RC $(now)"
