#!/usr/bin/env bash
# P4-e: re-run the P3-4 pool search on gate_heldout with Qwen3.8-27B zero-shot (qwen38zs) and Qwen3.8-LoRA (qwen38lora) as
# candidate families. Declared before any result was seen (results/p4_feasibility.md, section "P4-e protocol"):
#   * thresholds per level from ocrgate.p3_pool on the gate_heldout units (labelled gain, conflicts ignored), as in P2/P3;
#   * every candidate pool below x {2, 3} agreement levels (families) is run as an actual ensemble on the held-out subset and
#     rescored on the 360 gate_heldout units with the official scorer (ocrgate.p3_rescore);
#   * ranking: lowest UNROUNDED weighted cMER of that rescore (tie: higher pref); the P3 frozen pool is re-run in the same batch
#     as the reference; "the P4 ensemble beats P3" iff best_P4_cmer < P3_reference_cmer (unrounded, same units, same batch);
#   * dev runs of every candidate for reporting only. Writes results/p4_candidates.tsv, results/p4_pool_search.json,
#     results/p4_frozen_ensemble.json (beats_p3 true/false). Marker P4_E_DONE.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p4_lib.sh
L24H=runs/20261030T094959Z_zeroshot_train_lora24b-ep2-ho; L32H=runs/20261030T101923Z_zeroshot_train_lora32b-ep2-ho; Q72T=runs/20261030T090807Z_zeroshot_train_q72b-p2
L72H=runs/20261031T014354Z_zeroshot_train_lora72b-ep2-ho; SYNH=runs/20261031T071801Z_zeroshot_train_lora72b-syn-ep1-ho
L24D=runs/20261030T095711Z_zeroshot_dev_lora24b-ep2; L32D=runs/20261030T103005Z_zeroshot_dev_lora32b-ep2; Q72D=runs/20261030T055235Z_zeroshot_dev_q72b
L72D=runs/20261031T020135Z_zeroshot_dev_lora72b-ep2; SYND=runs/20261031T073628Z_zeroshot_dev_lora72b-syn-ep1
TAG=${TAG:-qwen38-lora}; EP=$($PY -c "import json;print(json.load(open('results/p4_epoch_selection.json'))['$TAG']['chosen_epoch'])")
zs_run() { ls -td runs/*_zeroshot_"$1" | head -1; }   # the rewriter run itself (latest_run would also match the derived gate runs, which share the tag)
Q38ZH=$(zs_run train_qwen38-zs-ho); Q38ZD=$(zs_run dev_qwen38-zs); Q38LH=$(zs_run train_$TAG-ep$EP-ho); Q38LD=$(zs_run dev_$TAG-ep$EP)
echo "members: zs ho=$Q38ZH dev=$Q38ZD; lora ho=$Q38LH dev=$Q38LD"
GATE=runs/gates/p2_pooled; OUT=results/p4_pool_search.json; GH=configs/p3_gate_heldout_subset.json
export CUDA_VISIBLE_DEVICES=${VISIBLE:-0}
seg_open 1 "GPU${VISIBLE:-0}: P4-e pool selection on gate_heldout with Qwen3.8 zs + LoRA as candidate families (gate scores cached per run; held-out + dev ensembles)"
$TPY -m ocrgate.p3_pool --members lora24b=$L24H lora32b=$L32H q72b=$Q72T lora72b=$L72H lora72bsyn=$SYNH qwen38zs=$Q38ZH qwen38lora=$Q38LH --levels 2 3 --out $OUT 2>&1 | grep -v -i warn | tee logs/p4_pool_search.log | tail -20
cfg() { $PY scripts/p3_ens_cfg.py $OUT "$@"; }
runs_of() { local which=$1; shift; for m in "$@"; do case "$m:$which" in
  lora24b:ho) echo $L24H;; lora24b:dev) echo $L24D;; lora32b:ho) echo $L32H;; lora32b:dev) echo $L32D;; q72b:ho) echo $Q72T;; q72b:dev) echo $Q72D;;
  lora72b:ho) echo $L72H;; lora72b:dev) echo $L72D;; lora72bsyn:ho) echo $SYNH;; lora72bsyn:dev) echo $SYND;;
  qwen38zs:ho) echo $Q38ZH;; qwen38zs:dev) echo $Q38ZD;; qwen38lora:ho) echo $Q38LH;; qwen38lora:dev) echo $Q38LD;; esac; done; }
unrounded() { $PY -c "import json;s=json.load(open('$1/summary.json'));print(s['cmer_micro_weighted'], s['pref_score_cmer_macro_weighted'])"; }
: > results/p4_candidates.tsv
P3POOL="lora24b,lora32b,lora72b,lora72bsyn"
POOLS="$P3POOL $P3POOL,qwen38zs $P3POOL,qwen38lora $P3POOL,qwen38zs,qwen38lora lora24b,lora32b,lora72b,qwen38lora lora24b,lora32b,lora72bsyn,qwen38lora lora32b,lora72b,lora72bsyn,qwen38lora lora24b,lora72b,lora72bsyn,qwen38lora lora72b,lora72bsyn,qwen38lora lora24b,lora32b,qwen38lora lora24b,lora32b,qwen38zs,qwen38lora lora72b,lora72bsyn,qwen38zs,qwen38lora qwen38zs,qwen38lora"
for mem in $POOLS; do for lv in 2 3; do
  thr=$(cfg $lv families "$mem"); tag="p4e-${mem//,/+}-L$lv"
  mapfile -t HR < <(runs_of ho ${mem//,/ }); fam=(${mem//,/ })
  $TPY -m ocrgate.ensemble --config $CFG --runs "${HR[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag ho-$tag 2>&1 | grep -E "^\[ensemble"
  $PY -m ocrgate.p3_rescore --run "$(latest_run ho-$tag)" --subset $GH --tag gh-$tag | tail -1
  read -r C P < <(unrounded "$(latest_run gh-$tag)")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mem" "$lv" "$thr" "$C" "$P" "$(latest_run ho-$tag)" "$(latest_run gh-$tag)" >> results/p4_candidates.tsv
done; done
echo "=== gate_heldout ranking (unrounded cMER, pref) ==="; sort -t$'\t' -k4,4g -k5,5gr results/p4_candidates.tsv | cut -f1,2,4,5 | head -8
# reference = P3 frozen pool, 3 levels, re-run in this batch
IFS=$'\t' read -r _ _ RTHR RC RP RRUN RGH < <(awk -F'\t' -v p="$P3POOL" '$1==p && $2==3' results/p4_candidates.tsv)
# best P4 pool = lowest cMER among pools containing a Qwen3.8 family
IFS=$'\t' read -r BMEM BLV BTHR BC BP BRUN BGH < <(grep qwen38 results/p4_candidates.tsv | sort -t$'\t' -k4,4g -k5,5gr | head -1)
BEATS=$($PY -c "print('true' if $BC < $RC else 'false')")
echo "P3 reference (same batch): $RC / $RP (P3 report: 0.0076 / 0.8832, thresholds $RTHR)"
echo "BEST P4 pool: members $BMEM levels $BLV thresholds $BTHR gate_heldout $BC / $BP -> beats P3: $BEATS"
# dev runs (reporting only): every candidate; the best P4 pool additionally under the tag dev-p4-best
for mem in $POOLS; do for lv in 2 3; do thr=$(cfg $lv families "$mem"); mapfile -t DR < <(runs_of dev ${mem//,/ }); fam=(${mem//,/ })
  $TPY -m ocrgate.ensemble --config $CFG --runs "${DR[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag "dev-p4e-${mem//,/+}-L$lv" 2>&1 | grep -E "^\[ensemble"; done; done
mapfile -t DR < <(runs_of dev ${BMEM//,/ }); fam=(${BMEM//,/ })
$TPY -m ocrgate.ensemble --config $CFG --runs "${DR[@]}" --gate $GATE --agree-thresholds $BTHR --families "${fam[@]}" --tag dev-p4-best 2>&1 | grep -E "^\[ensemble"
seg_close
$PY - <<PYEOF
import json
json.dump({"members": "$BMEM".split(","), "families": "$BMEM".split(","), "levels": $BLV, "thresholds": [float(t) for t in "$BTHR".split()],
  "gate_heldout_cmer": $BC, "gate_heldout_pref": $BP, "heldout_run": "$BRUN", "gate_heldout_rescore_run": "$BGH",
  "p3_reference_same_batch": {"members": "$P3POOL".split(","), "levels": 3, "thresholds": [float(t) for t in "$RTHR".split()], "gate_heldout_cmer": $RC, "gate_heldout_pref": $RP, "heldout_run": "$RRUN", "p3_reported": [0.0076, 0.8832]},
  "beats_p3": "$BEATS" == "true", "criterion": "lowest unrounded official weighted cMER of the actual ensemble on the 360 gate_heldout units among the declared candidate pools containing a Qwen3.8 family (tie: higher pref); beats_p3 iff strictly lower than the P3 frozen pool re-run in the same batch; per-level thresholds by labelled gain on the same units (ocrgate.p3_pool)",
  "candidates": "results/p4_candidates.tsv", "search": "$OUT"}, open("results/p4_frozen_ensemble.json", "w"), indent=1)
PYEOF
mark "P4_E_DONE" "best P4 pool $BMEM L$BLV gate_heldout $BC/$BP vs P3 reference $RC/$RP (same batch; P3 report 0.0076) -> beats_p3=$BEATS; dev run $(latest_run dev-p4-best)"
echo "P4-e DONE $(now)"
