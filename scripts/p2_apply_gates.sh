#!/usr/bin/env bash
# P2-2 dev reporting: apply the Phase 2 gates + decision rules (chosen on held-out train) to every rewriter's DEV run.
# Env: LORA24_DEV / LORA32_DEV = chosen-epoch LoRA dev source runs (zeroshot runs with blocks.jsonl).
# Rewriters: zero-shot 8B, 32B, Mistral-24B, Qwen2.5-72B (Phase 1 dev source runs), LoRA-8B (P1-b), LoRA-24B, LoRA-32B (P2-1).
# For each: no gate (new runs only; the old ones exist), label oracle (new runs only), P1 frozen full gate qwen06_pooled
# (new runs only; old ones exist), P2 full gate x rules {global, editrate, sysconf, lang, lang_editrate}, P2 text-only gate x {global, lang}.
set -euo pipefail
cd "$(dirname "$0")/.."
TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}; export TOKENIZERS_PARALLELISM=false
SEG=results/server_segments_p2.tsv; now() { date -u +%FT%TZ; }
nseg() { awk -F'\t' 'NR>1 && $1 ~ /^[0-9]+$/ {n=$1} END {print n+1}' "$SEG"; }
: "${LORA24_DEV:?}"; : "${LORA32_DEV:?}"
OLD="runs/20261030T010752Z_zeroshot_dev_8b-lp runs/20261030T011802Z_zeroshot_dev_32b-lp runs/20261030T053954Z_zeroshot_dev_mistral24b runs/20261030T055235Z_zeroshot_dev_q72b runs/20261030T023902Z_zeroshot_dev_lora8b"
NEW="$LORA24_DEV $LORA32_DEV"
N=$(nseg); S=$(now)
for RUN in $NEW; do
  $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode none
  $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode oracle_label
  $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode gate --gate runs/gates/qwen06_pooled
  $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode gate --gate runs/gates/qwen06_pooled_nolp
done
for RUN in $OLD $NEW; do
  for RULE in global editrate sysconf lang lang_editrate; do
    $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode gate --gate runs/gates/p2_pooled --rule $RULE
  done
  for RULE in global lang; do
    $TPY -m ocrgate.gate_apply --config configs/phase2.yaml --run "$RUN" --mode gate --gate runs/gates/p2_pooled_nolp --rule $RULE
  done
done
printf '%s\t%s\t%s\t1\t%s\n' "$N" "$S" "$(now)" "GPU0: P2 gate inference on all dev rewriter runs (2 gates x rules; scores cached per run)" >> "$SEG"
echo "P2 APPLY DONE $(now)"
