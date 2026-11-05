#!/usr/bin/env bash
# P3-2 (part 1): pool search on gate_heldout with the LoRA-72B sampled candidates (gate scoring on GPU0, concurrent with the server).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/p3_lib.sh
S=(); for i in 1 2 3 4; do S+=("l72s$i=$(latest_run lora72b-ep2-s$i-ho):lora72b"); done
seg_open 1 "GPU0 (concurrent with seg 5): P2 gate scoring of the LoRA-72B held-out + sampled runs for the pool search"
CUDA_VISIBLE_DEVICES=0 $TPY -m ocrgate.p3_pool --members lora24b=runs/20261030T094959Z_zeroshot_train_lora24b-ep2-ho lora32b=runs/20261030T101923Z_zeroshot_train_lora32b-ep2-ho q72b=runs/20261030T090807Z_zeroshot_train_q72b-p2 lora72b=runs/20261031T014354Z_zeroshot_train_lora72b-ep2-ho "${S[@]}" --groups "s72=l72s1,l72s2,l72s3,l72s4" --levels 2 3 --out results/p3_pool_search_72b.json | tee logs/p3_pool_72b.log
seg_close
