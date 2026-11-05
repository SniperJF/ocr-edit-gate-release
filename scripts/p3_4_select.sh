#!/usr/bin/env bash
# P3-4: final pool / agreement-threshold selection on gate_heldout by the official cMER of the actual ensemble (declared in
# results/p3_2_samples.md), then the dev run of the chosen configuration. Env: SYN_HO / SYN_DEV = chosen-epoch LoRA-72B+synthetic
# runs (optional). Candidate members: LoRA-24B, LoRA-32B, 72B zs, LoRA-72B, [LoRA-72B-syn]; levels 2 / 3; thresholds per pool
# from labelled gain on gate_heldout (ocrgate.p3_pool, conflicts ignored) as in P2; ranking by gate_heldout cMER (tie: pref).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
L24H=runs/20261030T094959Z_zeroshot_train_lora24b-ep2-ho; L32H=runs/20261030T101923Z_zeroshot_train_lora32b-ep2-ho; Q72T=runs/20261030T090807Z_zeroshot_train_q72b-p2; L72H=runs/20261031T014354Z_zeroshot_train_lora72b-ep2-ho
L24D=runs/20261030T095711Z_zeroshot_dev_lora24b-ep2; L32D=runs/20261030T103005Z_zeroshot_dev_lora32b-ep2; Q72D=runs/20261030T055235Z_zeroshot_dev_q72b; L72D=runs/20261031T020135Z_zeroshot_dev_lora72b-ep2
SYN_HO=${SYN_HO:-}; SYN_DEV=${SYN_DEV:-}; GATE=runs/gates/p2_pooled; OUT=results/p3_pool_search_final.json; GH=configs/p3_gate_heldout_subset.json
export CUDA_VISIBLE_DEVICES=${VISIBLE:-0}
seg_open 1 "GPU${VISIBLE:-0}: P3-4 final pool selection (gate scoring of the synthetic-mix runs, held-out + dev ensembles)"
M=(lora24b=$L24H lora32b=$L32H q72b=$Q72T lora72b=$L72H); [ -n "$SYN_HO" ] && M+=("lora72bsyn=$SYN_HO")
$TPY -m ocrgate.p3_pool --members "${M[@]}" --levels 2 3 --out $OUT 2>&1 | grep -v -i warn | tee logs/p3_pool_final.log
cfg() { $PY scripts/p3_ens_cfg.py $OUT "$@"; }
runs_of() { local which=$1; shift; for m in "$@"; do case "$m:$which" in
  lora24b:ho) echo $L24H;; lora24b:dev) echo $L24D;; lora32b:ho) echo $L32H;; lora32b:dev) echo $L32D;; q72b:ho) echo $Q72T;; q72b:dev) echo $Q72D;;
  lora72b:ho) echo $L72H;; lora72b:dev) echo $L72D;; lora72bsyn:ho) echo $SYN_HO;; lora72bsyn:dev) echo $SYN_DEV;; esac; done; }
: > results/p3_4_candidates.tsv
POOLS="lora24b,lora32b,q72b lora24b,lora32b,q72b,lora72b lora24b,lora32b,lora72b lora32b,q72b,lora72b lora24b,q72b,lora72b"
[ -n "$SYN_HO" ] && POOLS="$POOLS lora24b,lora32b,q72b,lora72bsyn lora24b,lora32b,q72b,lora72b,lora72bsyn lora24b,lora32b,lora72b,lora72bsyn lora32b,q72b,lora72b,lora72bsyn"
for mem in $POOLS; do for lv in 2 3; do
  thr=$(cfg $lv families "$mem"); tag="p34-${mem//,/+}-L$lv"
  mapfile -t HR < <(runs_of ho ${mem//,/ }); fam=(${mem//,/ })
  $TPY -m ocrgate.ensemble --config $CFG --runs "${HR[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag ho-$tag 2>&1 | grep -E "^\[ensemble"
  res=$($PY -m ocrgate.p3_rescore --run "$(latest_run ho-$tag)" --subset $GH --tag gh-$tag | sed -E 's/.*cMER=([0-9.]+) pref=([0-9.]+).*/\1\t\2/')
  printf '%s\t%s\t%s\t%s\t%s\n' "$mem" "$lv" "$thr" "$res" "$(latest_run ho-$tag)" >> results/p3_4_candidates.tsv
done; done
sort -t$'\t' -k4,4n -k5,5nr results/p3_4_candidates.tsv | head -5
IFS=$'\t' read -r BMEM BLV BTHR BC BP BRUN <<<"$(sort -t$'\t' -k4,4n -k5,5nr results/p3_4_candidates.tsv | head -1)"
echo "WINNER (gate_heldout cMER $BC pref $BP): members $BMEM levels $BLV thresholds $BTHR"
mapfile -t DR < <(runs_of dev ${BMEM//,/ }); fam=(${BMEM//,/ })
$TPY -m ocrgate.ensemble --config $CFG --runs "${DR[@]}" --gate $GATE --agree-thresholds $BTHR --families "${fam[@]}" --tag dev-p3-frozen 2>&1 | grep -E "^\[ensemble"
# also the dev run of every candidate pool (reporting only)
for mem in $POOLS; do for lv in 2 3; do thr=$(cfg $lv families "$mem"); mapfile -t DR < <(runs_of dev ${mem//,/ }); fam=(${mem//,/ })
  $TPY -m ocrgate.ensemble --config $CFG --runs "${DR[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag "dev-p34-${mem//,/+}-L$lv" 2>&1 | grep -E "^\[ensemble"; done; done
seg_close
printf '{"members": "%s", "levels": %s, "thresholds": [%s], "gate_heldout_cmer": %s, "gate_heldout_pref": %s, "heldout_run": "%s", "criterion": "lowest official weighted cMER of the actual ensemble on the 360 gate_heldout units (tie: higher pref); thresholds per level by labelled gain on the same units"}\n' "$BMEM" "$BLV" "${BTHR// /, }" "$BC" "$BP" "$BRUN" > results/p3_frozen_ensemble.json
mark "P3_4_SELECT_DONE" "winner members=$BMEM levels=$BLV thr=$BTHR gate_heldout=$BC/$BP"
echo "P3-4 SELECT DONE $(now)"
