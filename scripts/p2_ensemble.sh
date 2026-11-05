#!/usr/bin/env bash
# P2-3: edit-pool ensembles. Env: LORA24_HO LORA32_HO (held-out train runs), LORA24_DEV LORA32_DEV (dev runs),
# Q72_TRAIN (72B zero-shot on full train), Q72_DEV (72B dev source run), GATE (default runs/gates/p2_pooled),
# RULE (decision rule chosen in P2-2, default global).
# 1. selection on held-out TRAIN (gate_heldout units only): dump the pooled candidates of LoRA-24B + LoRA-32B + 72B,
#    choose agreement thresholds (T1 single-rewriter blocks, T2 blocks proposed by >= 2 rewriters).
# 2. dev: pools of 2 and 3 rewriters under RULE; the 3-pool also with the agreement thresholds.
set -euo pipefail
cd "$(dirname "$0")/.."
TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}; export TOKENIZERS_PARALLELISM=false
GATE=${GATE:-runs/gates/p2_pooled}; RULE=${RULE:-global}; Q72_DEV=${Q72_DEV:-runs/20261030T055235Z_zeroshot_dev_q72b}
: "${LORA24_HO:?}"; : "${LORA32_HO:?}"; : "${LORA24_DEV:?}"; : "${LORA32_DEV:?}"; : "${Q72_TRAIN:?}"
SEG=results/server_segments_p2.tsv; now() { date -u +%FT%TZ; }
nseg() { awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {n=$1} END {print n+1}' "$SEG"; }
N=$(nseg); S=$(now)
mkdir -p results/p2_ensemble
# 1. held-out selection
$TPY -m ocrgate.ensemble --runs $LORA24_HO $LORA32_HO $Q72_TRAIN --gate $GATE --rule $RULE --tag ho-sel-3pool --dump-union results/p2_ensemble/union_heldout_3pool.jsonl
$TPY -m ocrgate.ensemble --select results/p2_ensemble/union_heldout_3pool.jsonl --select-split configs/p2_split.json --select-roles gate_heldout | tee results/p2_ensemble/agree_thresholds_3pool.json
T1=$($TPY -c "import json;d=json.load(open('results/p2_ensemble/agree_thresholds_3pool.json'));print(d['t_single'])")
T2=$($TPY -c "import json;d=json.load(open('results/p2_ensemble/agree_thresholds_3pool.json'));print(d['t_agree'])")
echo "agreement thresholds: single $T1 agree $T2"
# 2. dev
$TPY -m ocrgate.ensemble --runs $LORA24_DEV $LORA32_DEV --gate $GATE --rule $RULE --tag dev-2pool-lora24-lora32
$TPY -m ocrgate.ensemble --runs $LORA24_DEV $Q72_DEV --gate $GATE --rule $RULE --tag dev-2pool-lora24-q72b
$TPY -m ocrgate.ensemble --runs $LORA24_DEV $LORA32_DEV $Q72_DEV --gate $GATE --rule $RULE --tag dev-3pool-lora24-lora32-q72b
$TPY -m ocrgate.ensemble --runs $LORA24_DEV $LORA32_DEV $Q72_DEV --gate $GATE --agree-thresholds $T1 $T2 --tag dev-3pool-agree
$TPY -m ocrgate.ensemble --runs $LORA24_DEV $LORA32_DEV --gate $GATE --agree-thresholds $T1 $T2 --tag dev-2pool-agree
printf '%s\t%s\t%s\t1\t%s\n' "$N" "$S" "$(now)" "GPU0: P2-3 ensemble gate scoring (held-out selection + dev pools)" >> "$SEG"
echo "P2 ENSEMBLE DONE $(now)"
