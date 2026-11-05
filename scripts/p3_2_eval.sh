#!/usr/bin/env bash
# P3-2 evaluation: pool search on gate_heldout with all sampled candidates (LoRA-72B s1-4, LoRA-32B s1-4), then dev ensembles for
# the configurations of interest (thresholds = the held-out-chosen ones). Gate scoring on GPU0 (concurrent with training).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
L24H=runs/20261030T094959Z_zeroshot_train_lora24b-ep2-ho; L32H=runs/20261030T101923Z_zeroshot_train_lora32b-ep2-ho; Q72T=runs/20261030T090807Z_zeroshot_train_q72b-p2; L72H=runs/20261031T014354Z_zeroshot_train_lora72b-ep2-ho
L24D=runs/20261030T095711Z_zeroshot_dev_lora24b-ep2; L32D=runs/20261030T103005Z_zeroshot_dev_lora32b-ep2; Q72D=runs/20261030T055235Z_zeroshot_dev_q72b; L72D=runs/20261031T020135Z_zeroshot_dev_lora72b-ep2
S72H=(); S72D=(); S32H=(); S32D=()
for i in 1 2 3 4; do S72H+=("$(latest_run lora72b-ep2-s$i-ho)"); S72D+=("$(latest_run lora72b-ep2-s$i)"); S32H+=("$(latest_run lora32b-ep2-s$i-ho)"); S32D+=("$(latest_run lora32b-ep2-s$i)"); done
OUT=results/p3_pool_search_samples.json; GATE=runs/gates/p2_pooled; export CUDA_VISIBLE_DEVICES=0
seg_open 1 "GPU0 (concurrent with the synthetic-mix training): P3-2 pool search with the LoRA-72B + LoRA-32B sampled candidates (gate scoring of 8 held-out + 8 dev sampled runs) and dev ensembles"
M=(lora24b=$L24H lora32b=$L32H q72b=$Q72T lora72b=$L72H)
for i in 1 2 3 4; do M+=("l72s$i=${S72H[$((i-1))]}:lora72b" "l32s$i=${S32H[$((i-1))]}:lora32b"); done
$TPY -m ocrgate.p3_pool --members "${M[@]}" --groups "s72=l72s1,l72s2,l72s3,l72s4" "s32=l32s1,l32s2,l32s3,l32s4" --levels 2 3 --out $OUT 2>&1 | grep -v -i warn | tee logs/p3_pool_samples.log
cfg() { $PY scripts/p3_ens_cfg.py $OUT "$@"; }
ens() { # TAG LEVELS BY "members" ; dev + held-out ensembles
  local tag=$1 lv=$2 by=$3 mem=$4; local thr; thr=$(cfg $lv $by "$mem")
  local dr=() hr=() fam=()
  for m in ${mem//,/ }; do case $m in
    lora24b) dr+=($L24D); hr+=($L24H); fam+=(lora24b);; lora32b) dr+=($L32D); hr+=($L32H); fam+=(lora32b);; q72b) dr+=($Q72D); hr+=($Q72T); fam+=(q72b);; lora72b) dr+=($L72D); hr+=($L72H); fam+=(lora72b);;
    l72s[1-4]) i=${m#l72s}; dr+=(${S72D[$((i-1))]}); hr+=(${S72H[$((i-1))]}); fam+=(lora72b);; l32s[1-4]) i=${m#l32s}; dr+=(${S32D[$((i-1))]}); hr+=(${S32H[$((i-1))]}); fam+=(lora32b);; esac; done
  if [ "$by" = votes ]; then fam=("${dr[@]}"); fi   # distinct family per run = raw vote counting
  echo "[ens] $tag levels=$lv by=$by thr=$thr members=$mem"
  $TPY -m ocrgate.ensemble --config $CFG --runs "${dr[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag dev-p3-$tag 2>&1 | grep -E "^\[ensemble"
  $TPY -m ocrgate.ensemble --config $CFG --runs "${hr[@]}" --gate $GATE --agree-thresholds $thr --families "${fam[@]}" --tag ho-p3-$tag 2>&1 | grep -E "^\[ensemble"
}
ens 4greedy-L3 3 families lora24b,lora32b,q72b,lora72b
ens 4greedy-L2 2 families lora24b,lora32b,q72b,lora72b
ens 3pool-L3 3 families lora24b,lora32b,q72b
ens s72-L3 3 families lora24b,lora32b,q72b,lora72b,l72s1,l72s2,l72s3,l72s4
ens s32-L3 3 families lora24b,lora32b,q72b,lora72b,l32s1,l32s2,l32s3,l32s4
ens s72s32-L3 3 families lora24b,lora32b,q72b,lora72b,l72s1,l72s2,l72s3,l72s4,l32s1,l32s2,l32s3,l32s4
ens s72s32-L3-votes 3 votes lora24b,lora32b,q72b,lora72b,l72s1,l72s2,l72s3,l72s4,l32s1,l32s2,l32s3,l32s4
BEST=$($PY -c "import json;d=json.load(open('$OUT'))['best'];print(','.join(d['members'])+' '+str(d['levels'])+' '+d['by'])")
read -r BM BL BB <<<"$BEST"; echo "held-out best: $BM levels $BL by $BB"
ens best "$BL" "$BB" "$BM" || true
seg_close
mark "P3_2_EVAL_DONE" "pool search + dev ensembles with sampled candidates; best held-out: $BEST"
echo "P3-2 EVAL DONE $(now)"
