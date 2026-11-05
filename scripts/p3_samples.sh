#!/usr/bin/env bash
# P3-2: sampled pool candidates. For one rewriter (served adapter or base model) draw N sampled rewrites (temperature T,
# seeds 1234+i) on the held-out train subset + dev; label blocks. Env: MODEL (HF id), LORA_MODULES ('name=path' or ''),
# SERVED (model name to request), TAG (run tag prefix), N (default 4), T (default 0.7), THINK (--send-thinking-kwarg value or '').
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
: "${MODEL:?}"; : "${SERVED:?}"; : "${TAG:?}"; N=${N:-4}; T=${T:-0.7}; THINK=${THINK:-}; LORA_MODULES=${LORA_MODULES:-}
TK=(); [ -n "$THINK" ] && TK=(--send-thinking-kwarg "$THINK")
serve "$MODEL" ${DP:-1} ${TP:-2} "$LORA_MODULES" "vLLM $SERVED TP=2: $N sampled candidates (T=$T, seeds 1235..) on held-out train subset + dev"
for i in $(seq 1 $N); do
  SEED=$((1234 + i))
  $PY -m ocrgate.run --config $CFG --system zeroshot --split train --subset $HO --model "$SERVED" --tag $TAG-s$i-ho --gen temperature=$T --gen seed=$SEED "${TK[@]}"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split dev --model "$SERVED" --tag $TAG-s$i --gen temperature=$T --gen seed=$SEED "${TK[@]}"
done
stop
for i in $(seq 1 $N); do blocks "$(latest_run $TAG-s$i-ho)"; blocks "$(latest_run $TAG-s$i)"; done
mark "P3_2_SAMPLES_DONE" "$TAG N=$N T=$T"
echo "P3-2 SAMPLES DONE ($TAG) $(now)"
