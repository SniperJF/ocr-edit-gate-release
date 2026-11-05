#!/usr/bin/env bash
# P2-2: retrain the Qwen3-0.6B gate (full and text-only) on the pooled Phase 2 blocks with the fixed P2 split
# (held-out = role gate_heldout of configs/p2_split.json), one gate per GPU in parallel; then choose the decision rules on
# the held-out scores (ocrgate.rules). Block files: zero-shot 8B/32B/Mistral-24B/Qwen2.5-72B on the full TRAIN split +
# the chosen-epoch LoRA-24B / LoRA-32B runs on the held-out train subset (env LORA24_HO / LORA32_HO = run dirs).
set -euo pipefail
cd "$(dirname "$0")/.."
TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}; export TOKENIZERS_PARALLELISM=false
SEG=results/server_segments_p2.tsv; now() { date -u +%FT%TZ; }
nseg() { awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {n=$1} END {print n+1}' "$SEG"; }
: "${LORA24_HO:?run dir of the chosen LoRA-24B held-out run}"; : "${LORA32_HO:?run dir of the chosen LoRA-32B held-out run}"
MISTRAL_TRAIN=${MISTRAL_TRAIN:-$(ls -td runs/*_zeroshot_train_mistral24b-p2 | head -1)}
Q72_TRAIN=${Q72_TRAIN:-$(ls -td runs/*_zeroshot_train_q72b-p2 | head -1)}
BLOCKS="runs/20261030T010915Z_zeroshot_train_8b-lp/blocks.jsonl runs/20261030T012216Z_zeroshot_train_32b-lp/blocks.jsonl $MISTRAL_TRAIN/blocks.jsonl $Q72_TRAIN/blocks.jsonl $LORA24_HO/blocks.jsonl $LORA32_HO/blocks.jsonl"
echo "blocks: $BLOCKS"
N0=$(nseg); S0=$(now)
CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.gate_train --config configs/phase2.yaml --blocks $BLOCKS --encoder Qwen/Qwen3-0.6B --name p2_pooled --split configs/p2_split.json --results results/gates_p2.jsonl --epochs 3 --lr 3e-5 --max-len 256 > logs/p2_gate_pooled.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 $TPY -m ocrgate.gate_train --config configs/phase2.yaml --blocks $BLOCKS --encoder Qwen/Qwen3-0.6B --name p2_pooled_nolp --no-lp --split configs/p2_split.json --results results/gates_p2.jsonl --epochs 3 --lr 3e-5 --max-len 256 > logs/p2_gate_pooled_nolp.log 2>&1 &
P1=$!
wait $P0; E0=$(now); printf '%s\t%s\t%s\t1\t%s\n' "$N0" "$S0" "$E0" "GPU0: P2 full gate (Qwen3-0.6B) on pooled zero-shot 8B/32B/Mistral-24B/72B train blocks + LoRA-24B/32B held-out blocks (P2 split)" >> "$SEG"
wait $P1; E1=$(now); printf '%s\t%s\t%s\t1\t%s\n' "$((N0+1))" "$S0" "$E1" "GPU1: P2 text-only gate (same blocks, --no-lp)" >> "$SEG"
$TPY -m ocrgate.rules --gate runs/gates/p2_pooled | tee logs/p2_rules_pooled.log
$TPY -m ocrgate.rules --gate runs/gates/p2_pooled_nolp | tee logs/p2_rules_pooled_nolp.log
echo "P2 GATES DONE $(now)"
