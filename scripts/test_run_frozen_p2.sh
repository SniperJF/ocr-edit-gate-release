#!/usr/bin/env bash
# FROZEN Phase 2 configuration: the exact commands of the single, later TEST run. NOT executed in Phase 2.
# Refuses to run unless CONFIRM_TEST_RUN=1 is set. Dry-run on dev (reads no test file):
#   SPLIT=dev CONFIRM_TEST_RUN=1 scripts/test_run_frozen_p2.sh        (ONLY=A|B|C restricts the parts; for a part-C-only
#   dry-run on existing dev runs set L24_RUN / L32_RUN / Q72_RUN to those run dirs and SUF to a fresh tag suffix)
# Frozen artefacts (sha256 in frozen/FROZEN_CONFIG_P2.md): gate runs/gates/p2_pooled (full) and runs/gates/p2_pooled_nolp
# (text-only), rule `global` of their rules.json; LoRA adapters runs/lora/mistral24b-p2/ep2 and runs/lora/qwen3-32b-p2/ep2;
# ensemble agreement thresholds from frozen/agree_thresholds_3pool.json (chosen on held-out train).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${CONFIRM_TEST_RUN:-}" != "1" ]; then echo "refusing: set CONFIRM_TEST_RUN=1 only for the approved final test run" >&2; exit 2; fi
SPLIT=${SPLIT:-test}; ONLY=${ONLY:-ABC}; DEVICE=${DEVICE:-cuda}; CFG=configs/phase2.yaml
PY=.venv/bin/python; TPY=${VLLM_PYTHON:-${VLLM_ENV:-$HOME/vllm-env}/bin/python}
: "${VLLM_API_KEY:?set VLLM_API_KEY to the bearer token of your local vLLM server}"; export VLLM_API_KEY TOKENIZERS_PARALLELISM=false
FULL=runs/gates/p2_pooled; TEXT=runs/gates/p2_pooled_nolp
ALLOW=(); [ "$SPLIT" = test ] && ALLOW=(--allow-test)
SUF=${SUF:-p2frozen}
T1=$($TPY -c "import json;print(json.load(open('frozen/agree_thresholds_3pool.json'))['t_single'])")
T2=$($TPY -c "import json;print(json.load(open('frozen/agree_thresholds_3pool.json'))['t_agree'])")
serve() { # MODEL DP TP "LORA_MODULES"
  tmux new -d -s vllm_test "cd $PWD && VLLM_USE_FLASHINFER_SAMPLER=0 MODEL=$1 DP=$2 TP=$3 LORA_MODULES='$4' ./serve_model.sh"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)" = 401 ]; do sleep 5; done
}
stop() { tmux kill-session -t vllm_test; until ! pgrep -f "^${VLLM_ENV:-$HOME/vllm-env}/bin/python.*vllm serve" >/dev/null; do sleep 2; done; }
gate_all() { # RUN
  $PY -m ocrgate.blocks --config $CFG --run "$1"                                   # labels: diagnostics only
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode none --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode gate --gate $FULL --rule global --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode gate --gate $TEXT --rule global --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.gate_apply --config $CFG --run "$1" --mode oracle_label --device "$DEVICE" "${ALLOW[@]}"   # upper bound only
}
# A. the two Phase 2 LoRA rewriters (primary systems), served with their frozen epoch-2 adapters (TP=2)
if [[ "$ONLY" == *A* ]]; then
  serve mistralai/Mistral-Small-24B-Instruct-2501 1 2 "lora24b-ep2=$PWD/runs/lora/mistral24b-p2/ep2"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split "$SPLIT" --model lora24b-ep2 --tag lora24b-$SUF
  stop; gate_all "$(ls -td runs/*_zeroshot_${SPLIT}_lora24b-$SUF | head -1)"
  serve Qwen/Qwen3-32B 1 2 "lora32b-ep2=$PWD/runs/lora/qwen3-32b-p2/ep2"
  $PY -m ocrgate.run --config $CFG --system zeroshot --split "$SPLIT" --model lora32b-ep2 --tag lora32b-$SUF --send-thinking-kwarg on
  stop; gate_all "$(ls -td runs/*_zeroshot_${SPLIT}_lora32b-$SUF | head -1)"
fi
# B. Qwen2.5-72B zero-shot (third ensemble member; also reported alone with the P2 gate)
if [[ "$ONLY" == *B* ]]; then
  serve Qwen/Qwen2.5-72B-Instruct 1 2 ""
  $PY -m ocrgate.run --config $CFG --system zeroshot --split "$SPLIT" --model Qwen/Qwen2.5-72B-Instruct --tag q72b-$SUF
  stop; gate_all "$(ls -td runs/*_zeroshot_${SPLIT}_q72b-$SUF | head -1)"
fi
# C. edit-pool ensembles under the frozen full gate: 3-pool with the agreement thresholds (primary), 3-pool global, 2-pool LoRA
if [[ "$ONLY" == *C* ]]; then
  L24=${L24_RUN:-$(ls -td runs/*_zeroshot_${SPLIT}_lora24b-$SUF | head -1)}; L32=${L32_RUN:-$(ls -td runs/*_zeroshot_${SPLIT}_lora32b-$SUF | head -1)}; Q=${Q72_RUN:-$(ls -td runs/*_zeroshot_${SPLIT}_q72b-$SUF | head -1)}
  $TPY -m ocrgate.ensemble --config $CFG --runs $L24 $L32 $Q --gate $FULL --agree-thresholds $T1 $T2 --tag ${SPLIT}-3pool-agree-$SUF --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.ensemble --config $CFG --runs $L24 $L32 $Q --gate $FULL --rule global --tag ${SPLIT}-3pool-global-$SUF --device "$DEVICE" "${ALLOW[@]}"
  $TPY -m ocrgate.ensemble --config $CFG --runs $L24 $L32 --gate $FULL --agree-thresholds $T1 $T2 --tag ${SPLIT}-2pool-agree-$SUF --device "$DEVICE" "${ALLOW[@]}"
fi
echo "FROZEN P2 RUN DONE ($SPLIT, $ONLY) $(date -u +%FT%TZ)"
