#!/usr/bin/env bash
# P4-c: bf16 LoRA on Qwen/Qwen3.8-27B (Qwen3_5ForConditionalGeneration, text-only), P2 recipe (r16/a32, lr 1e-4, ga16, 2 epochs,
# fit role of configs/p2_split.json, impresso-nzz excluded, checkpoint per epoch), one H200 (GPU 0). Targets: full-attention
# q/k/v/o, Gated DeltaNet in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj, MLP gate/up/down (results/p4_feasibility.md).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p4_lib.sh
OUT=${OUT:-runs/lora/qwen38-27b-p4}
TM=q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj
seg_open 1 "GPU0: bf16 LoRA-tune Qwen3.8-27B (text-only class) on the P2 fit part of train, 2 epochs, adapters ep1/ep2 (attn+GDN+MLP targets)"
mark "P4_C_TRAIN_START" "$OUT"
set +e
CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.lora_train --config $CFG --model Qwen/Qwen3.8-27B --model-class Qwen3_5ForConditionalGeneration \
  --target-modules $TM --out "$OUT" --split configs/p2_split.json --roles fit --epochs 2 --save-each-epoch --device-map 0 \
  > "logs/p4_lora_$(basename $OUT).log" 2>&1
RC=$?; set -e
seg_close
mark "P4_C_TRAIN_END" "$OUT rc=$RC"
echo "TRAIN DONE rc=$RC $(now)"
