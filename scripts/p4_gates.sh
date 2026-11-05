#!/usr/bin/env bash
# P4-d: score the Qwen3.8-27B zero-shot and LoRA (chosen epoch) runs, held-out + dev, with the P2 gate UNCHANGED
# (runs/gates/p2_pooled rule global = full gate; runs/gates/p2_pooled_nolp = text-only gate), plus no gate and the label oracle.
# GPU ${VISIBLE:-0}. Marker P4_D_DONE.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p4_lib.sh
TAG=${TAG:-qwen38-lora}; ONLY=${ONLY:-all}   # zs | lora | all (zs can be scored while the LoRA still trains)
RUNS=(); EP=-
if [ "$ONLY" != lora ]; then RUNS+=("$(latest_run qwen38-zs-ho)" "$(latest_run qwen38-zs)"); fi
if [ "$ONLY" != zs ]; then EP=$($PY -c "import json;print(json.load(open('results/p4_epoch_selection.json'))['$TAG']['chosen_epoch'])"); RUNS+=("$(latest_run $TAG-ep$EP-ho)" "$(latest_run $TAG-ep$EP)"); fi
export CUDA_VISIBLE_DEVICES=${VISIBLE:-0}
seg_open 1 "GPU${VISIBLE:-0}: P2 gate inference (full gate global, text-only gate, none, oracle) on the Qwen3.8 zero-shot + LoRA ep$EP held-out and dev runs"
for RUN in "${RUNS[@]}"; do
  $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode none 2>&1 | grep -E "weighted cMER" || true
  $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode gate --gate runs/gates/p2_pooled --rule global 2>&1 | grep -E "weighted cMER" || true
  $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode gate --gate runs/gates/p2_pooled_nolp --rule global 2>&1 | grep -E "weighted cMER" || true
  $TPY -m ocrgate.gate_apply --config $CFG --run "$RUN" --mode oracle_label 2>&1 | grep -E "weighted cMER" || true
done
seg_close
[ "$ONLY" = zs ] && mark "P4_D_ZS_DONE" "P2 gate (unchanged) on Qwen3.8 zero-shot (held-out + dev): none / full global / text-only global / label oracle; runs ${RUNS[*]}"
[ "$ONLY" != zs ] && mark "P4_D_DONE" "P2 gate (unchanged) on Qwen3.8 zs + LoRA ep$EP (held-out + dev): none / full global / text-only global / label oracle; runs ${RUNS[*]} (zs part: see P4_D_ZS_DONE if run separately)"
echo "P4-d DONE $(now)"
